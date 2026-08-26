"""replay_traffic.py -- replay future-pool images through the live /predict endpoint as if they
were production traffic.

There is no real traffic against this model, so it is simulated -- the same honest stand-in the
future-pool is for "new data arriving". Requests go over HTTP to the running container, not
through in-process inference, because the whole point is to exercise the ACTUAL serving path:
the upload decode, the pre-convert metadata read, and the prediction log write. An in-process
call would prove none of those work in the deployed image.

Two modes, and the pair of them is the demo:
    normal  -- unmodified image bytes. Expected: no drift. This run is also the calibration for
               drift_report.py's provisional confidence threshold.
    skewed  -- restricted to a handful of classes, resized down, converted to grayscale.
               Expected: drift, on the class, metadata and (via blurry upscaling) confidence
               signals.

NORMAL MODE SENDS THE ORIGINAL FILE BYTES, UNTOUCHED -- no decode, no re-encode. JPEG
recompression would itself perturb the image slightly, and since this run is what calibrates a
threshold, "unmodified" has to mean the bytes on disk, not a faithful-looking round-trip.

READ-ONLY WITH RESPECT TO PIPELINE STATE. release_batch.py, the other consumer of the future
pool, APPENDS to data/active_train.txt and ADVANCES data/future_pool_cursor.json. This script
must do neither: it reads the cursor to find which batches are still unreleased and takes them
through FuturePool's read-only indexing. Replaying traffic is an observation of the model, not
an event in the retraining pipeline, and a simulator with side effects on the training set would
corrupt the very thing it is supposed to be measuring. batch_size must match the DAG's
BATCH_SIZE for the same reason release_batch.py documents: the cursor counts batch NUMBERS, so a
different size silently redefines which images count as unreleased.

Images come from the UNRELEASED future pool specifically: those are unseen by the champion,
which makes them like-for-like with the holdout the drift reference is built from.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

from coin_clf.image_meta import image_metadata
from coin_clf.prediction_log import PredictionLog

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_URL = "http://localhost:8000/predict"
DEFAULT_DB = REPO_ROOT / "outputs" / "monitoring" / "predictions.db"

# Must match dags/retrain_coin_clf.py's BATCH_SIZE -- see the module docstring.
DEFAULT_POOL_BATCH_SIZE = 5000

# Above drift_report.py's DEFAULT_MIN_SAMPLES (500), with headroom so a handful of failed
# requests cannot drop the sample under the floor and turn a real result into insufficient_data.
DEFAULT_COUNT = 600
DEFAULT_MIN_SAMPLES = 500

DEFAULT_SKEW_CLASSES = 5
DEFAULT_SKEW_RESIZE = 96
DEFAULT_JPEG_QUALITY = 85

_local = threading.local()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- image preparation ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Degradation:
    """How (or whether) an image is degraded before being sent.

    The identity degradation is not "re-encode at quality 100" -- it is "do not touch the bytes".
    See the module docstring on why that matters for the calibration run.
    """

    resize: int | None = None       # longest side, aspect preserved
    grayscale: bool = False
    jpeg_quality: int = DEFAULT_JPEG_QUALITY

    @property
    def is_identity(self) -> bool:
        return self.resize is None and not self.grayscale


IDENTITY = Degradation()


def prepare_image(path: str | Path, degradation: Degradation = IDENTITY) -> tuple[bytes, str]:
    """Return (bytes, content_type) ready to upload.

    Grayscale is applied by converting to PIL mode "L" and re-encoding, which survives the JPEG
    round-trip as mode "L" (verified, not assumed). That is what makes this a real end-to-end
    exercise of the pre-convert metadata read in app/main.py: /predict converts every upload to
    RGB before preprocessing, so if it logged metadata after that convert, these rows would say
    "RGB" and the mode drift signal could never fire.
    """
    path = Path(path)
    if degradation.is_identity:
        return path.read_bytes(), "image/jpeg"

    with Image.open(path) as src:
        img = src.convert("RGB")
    if degradation.resize is not None:
        longest = max(img.size)
        scale = degradation.resize / longest
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    if degradation.grayscale:
        img = img.convert("L")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=degradation.jpeg_quality)
    return buf.getvalue(), "image/jpeg"


# --- image selection -----------------------------------------------------------------------------

def read_cursor(cursor_file: str | Path) -> int:
    """Read the future-pool cursor. READS ONLY -- never writes it back. See the module docstring."""
    path = Path(cursor_file)
    if not path.exists():
        return 0
    return json.loads(path.read_text())["next_batch"]


def unreleased_images(data_dir: str, manifest: str, cursor_file: str,
                      batch_size: int = DEFAULT_POOL_BATCH_SIZE) -> list[tuple[Path, str]]:
    """(path, label) pairs from every future-pool batch the DAG has NOT yet released."""
    from splits import DatasetSplits

    splits = DatasetSplits.load(manifest, data_dir=data_dir)
    pool = splits.future_pool(batch_size)
    cursor = read_cursor(cursor_file)

    rows: list[tuple[Path, str]] = []
    for batch_num in range(cursor, len(pool)):
        batch = pool[batch_num]
        labels = [splits.idx_to_label[int(l)] for l in batch.labels.tolist()]
        rows.extend(zip(batch.filepaths, labels))
    return rows


def select_images(candidates: list[tuple[Path, str]], count: int, classes: int | None,
                  seed: int) -> list[tuple[Path, str]]:
    """Sample `count` images, optionally restricted to the `classes` most available labels.

    Restriction picks the most POPULOUS labels so the skew has enough images to draw from without
    repeating. Sampling is without replacement: sending the same image twice would put duplicate
    rows in the log and quietly narrow the distribution beyond the degradation being tested.
    """
    rng = random.Random(seed)
    pool = candidates
    if classes is not None:
        counts = collections.Counter(label for _, label in candidates)
        keep = {label for label, _ in counts.most_common(classes)}
        pool = [row for row in candidates if row[1] in keep]

    if len(pool) < count:
        raise ValueError(
            f"only {len(pool)} image(s) available"
            + (f" across the {classes} most populous class(es)" if classes else "")
            + f", need {count}. Lower --count, raise --classes, or release fewer future-pool "
              "batches into training."
        )
    return rng.sample(pool, count)


# --- sending -------------------------------------------------------------------------------------

def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def default_post(url, files, headers, timeout=30):
    return _session().post(url, files=files, headers=headers, timeout=timeout)


@dataclass
class ReplayResult:
    sent: int
    succeeded: int
    failed: int
    status_counts: dict
    started_at: str
    wall_seconds: float
    tag: str
    errors: list


def replay(images, url: str, tag: str, degradation: Degradation, concurrency: int = 4,
           delay: float = 0.0, post_fn=default_post, progress_every: int = 100) -> ReplayResult:
    """Send every image to /predict, tagged with X-Traffic-Source.

    post_fn is injectable so the tests can capture requests without a live server.
    """
    started_at = utc_now_iso()   # captured BEFORE the first send, so --since covers the whole run
    t0 = time.perf_counter()
    status_counts: dict = collections.Counter()
    errors: list = []
    lock = threading.Lock()
    done = 0

    def send(item):
        nonlocal done
        path, _label = item
        try:
            payload, content_type = prepare_image(path, degradation)
            response = post_fn(
                url,
                files={"file": (Path(path).name, payload, content_type)},
                headers={"X-Traffic-Source": tag},
            )
            status = getattr(response, "status_code", 0)
        except Exception as exc:                      # a dead container is a replay failure, not
            status = 0                                # something to hide behind a success count
            with lock:
                if len(errors) < 5:
                    errors.append(f"{type(exc).__name__}: {exc}")
        with lock:
            status_counts[status] += 1
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  sent {done}/{len(images)}...")
        if delay:
            time.sleep(delay)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(send, images))

    succeeded = status_counts.get(200, 0)
    return ReplayResult(
        sent=len(images), succeeded=succeeded, failed=len(images) - succeeded,
        status_counts=dict(status_counts), started_at=started_at,
        wall_seconds=time.perf_counter() - t0, tag=tag, errors=errors,
    )


# --- smoke test ------------------------------------------------------------------------------------

def smoke_test(images, url: str, tag: str, degradation: Degradation, db_path: Path,
               champion_version: str, post_fn=default_post) -> bool:
    """Send ONE request and prove the whole logging path works before spending minutes on a replay.

    Checks, in the order they would fail:
      1. the request succeeded
      2. a row was actually written -- the failure this catches is a container built before
         coin_clf.prediction_log existed, which serves 200s and logs nothing at all
      3. the row's model_version matches the CHAMPION the drift reference is built for. If the
         container serves a different version, drift_report.py filters out every production row
         and a full 600-image replay ends in insufficient_data. Catching that on request one
         instead of request six hundred is the entire reason this check is here.
      4. the logged metadata matches what was actually sent -- for a grayscale run this is the
         end-to-end proof of app/main.py's pre-convert metadata read.
    """
    path, _ = images[0]
    payload, content_type = prepare_image(path, degradation)
    expected = image_metadata(Image.open(io.BytesIO(payload)))

    log = PredictionLog(db_path)
    try:
        before = len(log.read_records())
    except FileNotFoundError:
        before = 0

    response = post_fn(url, files={"file": (Path(path).name, payload, content_type)},
                       headers={"X-Traffic-Source": tag})
    status = getattr(response, "status_code", 0)
    print(f"smoke: POST {url} -> {status}")
    if status != 200:
        print(f"smoke: FAILED -- /predict returned {status}, body={getattr(response, 'text', '')[:200]}")
        return False

    try:
        rows = log.read_records()
    except FileNotFoundError:
        print(f"smoke: FAILED -- no prediction log at {db_path}. The request succeeded but "
              "nothing was logged: the container almost certainly predates coin_clf.prediction_log "
              "(rebuild with `make build`) or is running without the outputs/monitoring mount "
              "(restart with `make serve`).")
        return False

    if len(rows) <= before:
        print(f"smoke: FAILED -- request returned 200 but no row was written to {db_path}.")
        return False

    row = rows[-1]
    print(f"smoke: logged row -> model_version={row['model_version']} "
          f"label={row['predicted_label']} confidence={row['confidence']:.3f} "
          f"{row['width']}x{row['height']} mode={row['mode']} source={row['source']!r}")

    ok = True
    if str(row["model_version"]) != str(champion_version):
        print(f"smoke: FAILED -- container serves model_version={row['model_version']} but the "
              f"drift reference is built for @champion v{champion_version}. Every production row "
              "would be filtered out and the verdict would be insufficient_data. Restart the "
              "container so it loads the current champion.")
        ok = False
    if row["source"] != tag:
        print(f"smoke: FAILED -- expected source={tag!r}, logged {row['source']!r}")
        ok = False
    if (row["width"], row["height"], row["mode"]) != (expected.width, expected.height, expected.mode):
        print(f"smoke: FAILED -- sent {expected.width}x{expected.height} mode={expected.mode} "
              f"but logged {row['width']}x{row['height']} mode={row['mode']}. If mode is RGB for "
              "a grayscale upload, /predict is reading metadata AFTER .convert('RGB').")
        ok = False

    print("smoke: OK -- logging path verified end to end" if ok else "smoke: FAILED")
    return ok


# --- CLI ---------------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("normal", "skewed"), required=True)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--data-dir", default="data/FOR_TRAINNING")
    p.add_argument("--manifest", default="data/splits_manifest.json")
    p.add_argument("--cursor-file", default="data/future_pool_cursor.json")
    p.add_argument("--pool-batch-size", type=int, default=DEFAULT_POOL_BATCH_SIZE,
                   help="must match the DAG's BATCH_SIZE -- the cursor counts batch numbers")
    p.add_argument("--count", type=int, default=DEFAULT_COUNT)
    p.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                   help="drift_report.py's floor; fewer successful sends than this is a failed run")
    p.add_argument("--tag", default=None, help="X-Traffic-Source (default: <mode>-replay)")
    p.add_argument("--classes", type=int, default=DEFAULT_SKEW_CLASSES,
                   help="skewed mode: restrict to the N most populous classes")
    p.add_argument("--resize", type=int, default=DEFAULT_SKEW_RESIZE,
                   help="skewed mode: longest side in px")
    p.add_argument("--no-grayscale", action="store_true", help="skewed mode: keep images RGB")
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--delay", type=float, default=0.0, help="per-request sleep, seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db", default=str(DEFAULT_DB), help="prediction log, for --smoke")
    p.add_argument("--smoke", action="store_true",
                   help="send ONE request and verify it was logged correctly, then exit")
    args = p.parse_args()

    if args.count < args.min_samples and not args.smoke:
        raise ValueError(
            f"--count {args.count} is below --min-samples {args.min_samples}: the drift check "
            "would report insufficient_data no matter what the traffic looked like."
        )

    tag = args.tag or f"{args.mode}-replay"
    if args.mode == "normal":
        # Unmodified bytes and the full class spread. The degradation flags are skewed-mode only.
        degradation, classes = IDENTITY, None
    else:
        degradation = Degradation(resize=args.resize, grayscale=not args.no_grayscale,
                                  jpeg_quality=args.jpeg_quality)
        classes = args.classes

    print(f"loading unreleased future-pool images (cursor={read_cursor(args.cursor_file)}, "
          f"batch_size={args.pool_batch_size})...")
    candidates = unreleased_images(args.data_dir, args.manifest, args.cursor_file,
                                   args.pool_batch_size)
    print(f"{len(candidates)} unreleased image(s) available")

    count = 1 if args.smoke else args.count
    images = select_images(candidates, count, classes, args.seed)

    if args.smoke:
        from drift_report import resolve_champion_version, MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_ALIAS

        champion = resolve_champion_version(MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_ALIAS)
        print(f"smoke: @champion is v{champion}; sending 1 {args.mode} request tagged {tag!r}")
        sys.exit(0 if smoke_test(images, args.url, tag, degradation, Path(args.db), champion) else 1)

    print(f"replaying {len(images)} {args.mode} image(s) -> {args.url}  "
          f"tag={tag!r}  concurrency={args.concurrency}")
    if classes is not None:
        print(f"  degradation: {classes} classes, resize={args.resize}px, "
              f"grayscale={not args.no_grayscale}")

    result = replay(images, args.url, tag, degradation, concurrency=args.concurrency,
                    delay=args.delay)

    print(f"\nsent={result.sent} succeeded={result.succeeded} failed={result.failed} "
          f"in {result.wall_seconds:.1f}s ({result.sent / result.wall_seconds:.1f} req/s)")
    print(f"status codes: {result.status_counts}")
    for err in result.errors:
        print(f"  error: {err}")

    print("\nnext, scope the drift check to THIS run with either:")
    print(f"  python drift_report.py --data-dir {args.data_dir} --source {tag}")
    print(f"  python drift_report.py --data-dir {args.data_dir} --since {result.started_at}")

    if result.succeeded < args.min_samples:
        print(f"\nFAILED: only {result.succeeded} request(s) succeeded, below --min-samples "
              f"{args.min_samples}. The drift check would report insufficient_data -- that would "
              "look like a monitoring bug rather than the failed replay it actually is.")
        sys.exit(1)


if __name__ == "__main__":
    main()

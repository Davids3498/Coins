"""Tests for replay_traffic.py -- the traffic simulator.

No live container: replay() takes an injectable post_fn, the same dependency-injection shape
app/main.py uses for get_bundle, so every request the script would send is captured in-process.

The two properties worth the most here:
  * the degradations actually produce what the drift signals need -- specifically that grayscale
    survives the JPEG re-encode as mode "L", because a silently-RGB "grayscale" batch would make
    the mode signal untestable and the Piece 1 pre-convert guard unprovable end to end.
  * the replay does not touch pipeline state. release_batch.py appends to active_train.txt and
    advances the cursor; this script must be able to run a hundred times and change neither.
"""
import io
import json

import pytest
from PIL import Image

from replay_traffic import (
    IDENTITY,
    Degradation,
    prepare_image,
    read_cursor,
    replay,
    select_images,
)


class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.text = ""


def make_source_image(path, size=(400, 300), color=(120, 90, 60)):
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


def opened(payload):
    return Image.open(io.BytesIO(payload))


@pytest.fixture
def source(tmp_path):
    return make_source_image(tmp_path / "coin.jpg")


# --- degradations ---------------------------------------------------------------------------

def test_normal_mode_sends_the_original_bytes_untouched(source):
    """Not a faithful re-encode -- the identical bytes. The calibration run depends on this: a
    JPEG round-trip would perturb the image and quietly bias the confidence it measures.
    """
    payload, content_type = prepare_image(source, IDENTITY)
    assert payload == source.read_bytes()
    assert content_type == "image/jpeg"


def test_grayscale_survives_the_jpeg_round_trip_as_mode_L(source):
    """The end-to-end proof of app/main.py's pre-convert metadata read depends on the upload
    genuinely arriving non-RGB.
    """
    payload, _ = prepare_image(source, Degradation(grayscale=True))
    assert opened(payload).mode == "L"


def test_resize_shrinks_longest_side_and_preserves_aspect(source):
    payload, _ = prepare_image(source, Degradation(resize=96))
    img = opened(payload)
    assert max(img.size) == 96
    assert img.size == (96, 72)          # 400x300 -> 4:3 preserved


def test_full_skew_applies_both_degradations(source):
    payload, _ = prepare_image(source, Degradation(resize=96, grayscale=True))
    img = opened(payload)
    assert img.mode == "L" and max(img.size) == 96


def test_identity_degradation_is_recognised():
    assert IDENTITY.is_identity is True
    assert Degradation(grayscale=True).is_identity is False
    assert Degradation(resize=96).is_identity is False


# --- selection ------------------------------------------------------------------------------

def candidates(n_per_class, classes=("A", "B", "C", "D", "E", "F")):
    return [(f"/img/{label}_{i}.jpg", label) for label in classes for i in range(n_per_class)]


def test_class_restriction_keeps_only_the_most_populous_labels():
    pool = candidates(5, classes=("A", "B")) + candidates(50, classes=("Y", "Z"))
    picked = select_images(pool, 40, classes=2, seed=0)
    assert {label for _, label in picked} == {"Y", "Z"}


def test_unrestricted_selection_spans_many_classes():
    picked = select_images(candidates(50), 100, classes=None, seed=0)
    assert len({label for _, label in picked}) > 2


def test_selection_is_without_replacement():
    picked = select_images(candidates(50), 200, classes=None, seed=0)
    assert len({path for path, _ in picked}) == 200


def test_selection_is_deterministic_under_a_seed():
    pool = candidates(50)
    assert select_images(pool, 20, None, 7) == select_images(pool, 20, None, 7)


def test_too_few_images_raises_with_the_actual_availability():
    with pytest.raises(ValueError, match="only 10 image"):
        select_images(candidates(5, classes=("A", "B")), 40, classes=None, seed=0)


# --- read-only guarantee ----------------------------------------------------------------------

def test_read_cursor_does_not_write(tmp_path):
    """The simulator observes the model; it is not an event in the retraining pipeline."""
    cursor = tmp_path / "future_pool_cursor.json"
    cursor.write_text(json.dumps({"next_batch": 1}))
    before = cursor.read_bytes()

    assert read_cursor(cursor) == 1
    assert cursor.read_bytes() == before


def test_missing_cursor_reads_as_zero(tmp_path):
    assert read_cursor(tmp_path / "absent.json") == 0


# --- replay ---------------------------------------------------------------------------------

def test_replay_sends_one_request_per_image_with_the_traffic_tag(tmp_path):
    images = [(make_source_image(tmp_path / f"{i}.jpg"), "NERO") for i in range(5)]
    calls = []

    def fake_post(url, files, headers, timeout=30):
        calls.append((url, files, headers))
        return _FakeResponse(200)

    result = replay(images, "http://x/predict", "normal-replay", IDENTITY,
                    concurrency=2, post_fn=fake_post, progress_every=0)

    assert result.sent == result.succeeded == 5
    assert result.failed == 0
    assert all(h["X-Traffic-Source"] == "normal-replay" for _, _, h in calls)
    assert all(f["file"][2] == "image/jpeg" for _, f, _ in calls)


def test_replay_sends_degraded_bytes_when_skewed(tmp_path):
    images = [(make_source_image(tmp_path / f"{i}.jpg"), "NERO") for i in range(3)]
    sent = []

    def fake_post(url, files, headers, timeout=30):
        sent.append(files["file"][1])
        return _FakeResponse(200)

    replay(images, "http://x/predict", "skewed-replay", Degradation(resize=96, grayscale=True),
           concurrency=1, post_fn=fake_post, progress_every=0)

    assert all(opened(payload).mode == "L" for payload in sent)
    assert all(max(opened(payload).size) == 96 for payload in sent)


def test_failed_requests_are_counted_not_hidden(tmp_path):
    images = [(make_source_image(tmp_path / f"{i}.jpg"), "NERO") for i in range(4)]

    def fake_post(url, files, headers, timeout=30):
        return _FakeResponse(503)

    result = replay(images, "http://x/predict", "t", IDENTITY, concurrency=1,
                    post_fn=fake_post, progress_every=0)
    assert result.succeeded == 0 and result.failed == 4
    assert result.status_counts == {503: 4}


def test_connection_errors_are_recorded_as_failures(tmp_path):
    images = [(make_source_image(tmp_path / f"{i}.jpg"), "NERO") for i in range(3)]

    def fake_post(url, files, headers, timeout=30):
        raise ConnectionError("container is down")

    result = replay(images, "http://x/predict", "t", IDENTITY, concurrency=1,
                    post_fn=fake_post, progress_every=0)
    assert result.succeeded == 0 and result.failed == 3
    assert result.errors and "ConnectionError" in result.errors[0]


def test_started_at_precedes_the_run_so_since_covers_it(tmp_path):
    """The summary prints started_at as a --since value; it has to be taken before the first
    send or the earliest rows fall outside the window it advertises.
    """
    images = [(make_source_image(tmp_path / "a.jpg"), "NERO")]
    stamps = []

    def fake_post(url, files, headers, timeout=30):
        from replay_traffic import utc_now_iso
        stamps.append(utc_now_iso())
        return _FakeResponse(200)

    result = replay(images, "http://x/predict", "t", IDENTITY, concurrency=1,
                    post_fn=fake_post, progress_every=0)
    assert result.started_at <= stamps[0]

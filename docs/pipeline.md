# Retraining pipeline and monitoring loop

[← back to the README](../README.md)

The two Airflow DAGs, task by task: `retrain_coin_clf` (release → validate → train → evaluate → promote) and `monitor_coin_clf` (the loop that decides whether to fire it). Both shell out to the same CLIs the Makefile runs, so a run reproduced by hand is the run Airflow makes.

## Retraining pipeline

`dags/retrain_coin_clf.py` — `schedule=None`, one run per simulated batch
arrival, `max_active_runs=1`, `retries=0` (every task has a side effect that is
not safe to replay: appending to the active-train list, advancing the cursor,
registering a version, moving an alias).

```
release_batch → validate → gate_on_validation → train → evaluate → promote
```

Every step shells out to an existing CLI with `PYTHONPATH=src /usr/bin/python3`,
which is exactly what the Makefile does — a run reproduced by hand is the run
Airflow makes, not a near-miss. Airflow's own process never imports torch.

* **release_batch** — pops the next 5,000-image future-pool batch, appends it to
  `data/active_train.txt`, writes the batch's `(path,label)` rows for validation.
  A drained pool is a hard error, not a "bad batch".
* **validate** — `python -m coin_clf.validate_batch`: readable (a real decode —
  see the engineering notes), RGB mode, minimum dimensions, known label, class
  balance (no class over 50% of a batch), no intra-batch duplicates, no leakage
  against the reference set. It reports a verdict; it never raises for a bad image.
* **gate_on_validation** — short-circuits train/evaluate/promote on an invalid
  batch. The DAG run stays **green**: a rejected batch is a normal outcome
  recorded in `report.json`, not a pipeline failure.
* **train** — `train_hard_labels.py`. Reads `epochs` / `batch_size` from
  `dag_run.conf` directly rather than through `params`, because that override
  depends on an `airflow.cfg` setting; a triggered retrain silently getting the
  3-epoch smoke default would train on a truncated cosine curve and then look
  like a model failure at the gate. It aborts if the holdout fingerprint has
  moved. Registers a challenger; it does not touch `@champion`.
* **evaluate** — scores the challenger on the frozen holdout, logs the metric.
* **promote** — moves `@champion` only if the challenger beats it by
  `--margin 0.005`. Re-scores **both** models fresh (never trusts a logged
  number), is idempotent, bootstraps when there is no champion, and holds
  fail-safe if the champion cannot be scored. A HOLD is logged, not raised.

The gate has actually refused: v10 (0.9201) is champion, and v11 scored 0.9150 on
the same holdout and was held. Both were produced by DAG runs, not by hand.

## Monitoring loop

`dags/monitor_coin_clf.py` — the closed loop.

```
check_serving_version → read_state → drift_check → gate_on_drift → trigger_retrain
```

* **check_serving_version** fails fast if the container is serving something other
  than `@champion`; a blind monitor must look broken rather than green.
* **drift_check** runs `drift_report.py` over the prediction log since the state
  cursor. The reference is **the champion's own predictions on the holdout**, not
  the training labels — comparing predictions to labels would bake the model's
  ~8% error rate into the baseline as a permanent non-zero floor, and labels carry
  no confidence at all. It is cached per champion version, and production rows are
  filtered to that version, so a v11-vs-v10 comparison can't manufacture drift.
  Thresholds were *measured* against a simulated null at n=500 rather than
  defaulted — the obvious 0.1 would fire on every normal batch — and the
  confidence one is marked provisional in the source because it has no measured
  null behind it yet.
* **gate_on_drift** triggers only on `status == "ok" AND drift_detected`.
  `insufficient_data` (under 500 rows), `no_data` and `no_reference` all mean
  "couldn't tell", which is not "no drift" and is certainly not grounds for a
  120-epoch run — that is why the verdict carries a status field and not a bare
  boolean. It also skips (rather than queues) when a retrain is already in flight.
* **trigger_retrain** passes the real recipe: `epochs=120, batch_size=256`, plus
  the reason, so the retrain run permanently records why it exists.

Runaway protection lives in `monitor_state.py`, and it is two mechanisms because
they cover different failures: a **consuming cursor** (advances only when a
verdict was actually rendered, so the same rows can never re-fire) and a
**circuit breaker** (two consecutive triggered retrains that end without a
promotion means retraining is not the fix — drift keeps being reported, but the
trigger stops).

`replay_traffic.py` is the traffic source. Requests go over HTTP against the
running container rather than through in-process inference on purpose: the point
is to exercise the real serving path, decode and log write included. It is
read-only with respect to pipeline state — it reads the future-pool cursor, never
advances it, and never appends to `active_train.txt`.

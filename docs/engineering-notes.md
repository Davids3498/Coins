# Engineering notes

[← back to the README](../README.md)

Three things that were wrong, at full length — what the bug was, why the obvious fix was the wrong one, and what mechanically stops each from coming back. The README carries a condensed version of each.

#**A "readable" check that passed unreadable files.** The retraining gate derived
`readable` from PIL's `verify()`, which validates the JPEG header and stops. A
file truncated mid-scan keeps an intact header: it opens, reports its true width,
height and mode, passes every check, enters the training set, and raises `OSError`
the first time a DataLoader touches it mid-epoch — the guard succeeding on exactly
the case it exists to catch. `readable` is now a full decode
(`image_meta.decodes`) while metadata stays a header parse, because serving
decodes each upload anyway and must not pay twice. Cost measured, not assumed:
0.2ms/image, ~2.2s for a 5,000-image batch inside a task that already runs for
minutes. The regression test writes a *noise* image, because a flat-colour JPEG is
~693 bytes and mostly header — truncating it destroys the header, every reader
rejects it, and the test would pass against the old code and prove nothing.

**A duplicated constant that manufactured a leak.** `verify_data_integrity.py`
kept its own copy of the DAG's future-pool batch size. The copy went stale at 200
against the DAG's 5,000, and the resulting arithmetic about which images had been
released reported **9,600 phantom leakage collisions** — a data-integrity checker
confidently crying leak. Correcting the number would have fixed the run and left
the mechanism: two constants that must agree, in files nobody edits together. So
the copy was deleted instead. The DAG's `BATCH_SIZE` is now the single
declaration, read by AST (importing it would pull in airflow, which the test
environment deliberately lacks). The test that matters doesn't check the value —
it fails if the duplication comes back.

**A `.dockerignore` that was a standing bug.** It listed what to exclude, and it
fell behind the repo: written before the Airflow venv, the 44 GB DVC cache and
`weights/`, so `docker build` streamed a 50 GB context to produce a ~1 GB image.
A denylist here is wrong again the next time anyone adds a big directory, and
nothing fails loudly when it does. It is now an allowlist — exclude everything,
re-include exactly the four paths the Dockerfile copies. A new large directory is
ignored by default, and a new `COPY` that needs something has to say so or the
build fails loudly.

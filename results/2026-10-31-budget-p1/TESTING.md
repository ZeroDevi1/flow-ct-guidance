# Verification before publication

2026-10-31 cloud CPU environment (versions in reproducibility_manifest.json):

```sh
cd code
PYTHONPATH=.:<CPU dependency directory> OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q tests experiments/budget_pilot --disable-warnings
```

Result: **31 passed**, 11.23 seconds, one pytest-plugin deprecation warning.
All pilot shell launchers passed `bash -n`; `git diff --check` passed after removing a Markdown trailing-space line.
No new GPU experiments were started for publication. Historical GPU evidence is separately retained.
No GitHub Actions workflow was configured in the repository at publication preparation; local tests must not be represented as a remote CI pass.

Privacy review: only added experiment files/results and their local development history were inspected, not all old repository images. Complete checkpoints and patient-indexed outputs are excluded.

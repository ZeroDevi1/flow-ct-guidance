# Complete bounded P1 development-validation batch

This run freezes lowdose DC lambda3 and observed-count WLS lambda0.3 after equal
2-lambda/3-CG preliminary diagnostics on four validation patients. It is NOT a
claim of optimal lambda search or held-out superiority. Twenty fixed validation
patients (including the pilot four), two slices each, seeds0/1/2, NFE50, CG30:
240 method-image evaluations. The other validation patients remain reserved.
Actual validation population must be checked from index metadata (reported as57
in prior records); this code requires at least20 distinct patients.

CG5 was inadequate for WLS. On the small diagnostic, increasing CG reduces normal
residuals but can worsen DC image PSNR because early stopping regularizes. CG30
therefore uses equal compute at higher solve accuracy; it is not guaranteed
converged for every image and does not erase float32 reproducibility limitations.
Lambda values were chosen at NFE20; moving to NFE50 is a prespecified sensitivity
extension, not retuning to these twenty patients.

Read existing source and copy only new files into the isolated code tree.
Ensure methods_checked.py matches the approved diagnostic version.
Run with a verified platform deadline; no instance shutdown is requested:

```
cd /root/autodl-tmp/flowct_budget_20261031/code
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PLATFORM_DEADLINE_VERIFIED=YES
export PLATFORM_DEADLINE_EPOCH=$(date -d '2026-10-31 09:08:00 UTC' +%s)
timeout --signal=TERM --kill-after=15s 3600s python -m experiments.budget_pilot.run_p1_batch \
 --config experiments/budget_pilot/p1_lowdose_frozen.json \
 --out /root/autodl-tmp/flowct_runs/p1_lowdose_$(date -u +%Y%m%dT%H%M%SZ)
```

Internal batch cap3540s. Empirical estimate roughly29min from CG30 NFE20 timing;
NFE50 scaling and patient anatomy may change runtime. Preserve partial outputs
but do not compare incomplete cohorts. No automatic retry/resume; new directories
for any authorized further work. Return only cohort-stripped per-seed summaries,
comparison statuses, batch status and anonymized aggregates. Keep private IDs,
patient images and per-image logs on AutoDL.

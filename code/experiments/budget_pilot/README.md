> Publication note (2026-10-31): the user authorized publishing reviewed code and anonymous aggregates. Historical preparation notes below describe the original private run. Patient-level experiment outputs remain excluded. After clarification that checkpoint cohort IDs belong to the public dataset, the user additionally authorized original full checkpoints as Release assets; see the publication policy in the result directory.

# Budgeted P0/P1 pilot (2026-10-31)

This is an **unpublished local experiment patch** on repository commit
`2311351b40adafd47cc922200a2c249cc22650e3`. It does not modify the historical
sampler, source metrics, published results, or final test images. No training.

## Scope

1. Synthetic P0 controls compare sparse M3 against zero-unmeasured-weight full-grid
   DC using the same **absolute** lambda, then report the historical same-relative
   lambda discrepancy. Exact measured angles/offsets are aligned for the algebraic
   control; the raw independently constructed grid discrepancy is reported too.
   Float32 finite-CG amplifies tiny angle rounding in some cases, so this is NOT
   evidence the current unmodified `dcfull` automatically equals M3.
2. Read `index_val.csv`, verify patient IDs do not overlap train/test index files
   (no test array loading), freeze four randomly ordered validation patients and
   two evenly spaced slices each, selection seed 20261031.
3. `sv60`, `la120`: measured-only DC, lambda .3/1/3. `ld180_i1e5`:
   unweighted DC vs observed-count WLS, lambda .3/1/3. NFE20, CG5, one paired seed.
   Same measurement and latent noise for each image across methods/lambda.
4. PSNR and SSIM use the same circle ROI. FBP is included. Record throughput,
   peak allocated CUDA memory, unweighted and weighted residuals, and count-floor
   fraction. This small validation pilot is not a final quality result or adequate
   statistical evidence of generalization.

WLS precision is proportional to estimated observed counts (inverse log-noise
variance, delta approximation), normalized to mean one **per image**. It uses
sigma=0 only; clipped zero/one counts are indistinguishable. This discards absolute
precision, and unweighted ata_scale need not equal weighted trace. Tune lambda on
validation rather than interpreting equal lambda as equal Bayesian prior strength.
No claim of an exact Poisson likelihood or calibrated posterior.

## Launch on existing authorized AutoDL GPU instance

Use a separate directory, never overwrite the historical MedFusionProjects tree.
Upload this code bundle through a permitted file-transfer mechanism. No pip install
is performed by the wrapper; existing torch/numpy/yaml/skimage/Pillow are expected.
Prior file must match the verified release SHA256 before deserialization.

```sh
cd /root/autodl-tmp/flowct_budget_20261031/code
AUTODL_BUDGET_SHUTDOWN_APPROVED=YES \
  bash experiments/budget_pilot/run_pilot.sh
```

Set `PRIOR`, `PROCESSED` or `PILOT_OUT` if needed. Output must not already exist.
**Required before launch:** enable and verify an AutoDL console/control-plane
shutdown deadline no later than 30 minutes after pilot start. The existence of
`shutdown` inside a container does not prove permission or power-off capability.
Do not substitute the in-container watchdog for this external hard-stop condition.

The wrapper requests **whole-instance shutdown on exit**, has command timeouts,
and a separate 1800-second watchdog. It is only for this authorized instance;
not for a shared server. The operator must verify shutdown in the AutoDL console:
SSH loss or a successful shutdown command alone is not billing confirmation.
At 2.88 CNY/hour the 30-minute cap is about 1.44 CNY after launching, excluding
setup time before launch. Track actual boot time and whole-account charges.
Do not launch if the shutdown command or the control-plane fallback is unavailable.

No unconditional auto-retry, no repeat training, no auto-download, no uploading
results, no new API keys. Results include selected patient IDs and private preview
images; keep private pending user approval and privacy review.

The validation script has no resume switch: partial JSONL/summary evidence is
preserved and further approved batches must use new output directories. This
avoids double-counting or silently overwriting failures.

Numerical safeguards: non-finite predictions, inputs, FBP or metrics immediately
write a failed status and abort; no successful conclusion is emitted. Incomplete
fixed-cohort groups retain raw rows but have no quality aggregate or ranking.
`comparison_status.json` permits a setting comparison only after every requested
method/lambda has the identical complete patient/slice cohort. A timeout is not a
method result, and completed subsets must not be mistaken for the full sweep.

## Revised operating authorization: keep the GPU instance on while fixing

Use `run_stage.sh`, **not `run_pilot.sh`**, for the revised session. The old
wrapper remains historical evidence and powers off on exit. The new runner
never powers off; the operator must first verify an absolute platform shutdown
deadline within 8 hours and the available monetary budget, then supply
`PLATFORM_DEADLINE_VERIFIED=YES PLATFORM_DEADLINE_EPOCH=<UTC epoch>`.
Stages `diagnostic` and `pilot` have process time limits (120 and 1200 seconds
respectively, override `STAGE_TIMEOUT_SECONDS`, maximum 1500); failed stages
write `launcher_status.json` and retain logs. GPU idle time still costs money.
Diagnostic output is not permission to bypass or relax P0; review first.

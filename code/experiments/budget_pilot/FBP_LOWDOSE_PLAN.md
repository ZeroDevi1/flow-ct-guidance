# Ready-but-not-authorized-to-run: lowdose supervised baseline

Prepare while P1 is running, but DO NOT run a concurrent GPU training job.
Candidate target: same 180-view i0=100000 sigma0 mu_scale0.06 setting as P1.
This is a new FBP+residual-U-Net baseline, not retraining the frozen flow prior.

## Frozen development design
- Configuration: `fbp_lowdose_safe.yaml`, total LR horizon20000 optimizer steps,
  batch8, BF16 training, AdamW2e-4, warmup500, minLR1e-5.
- Fixed first20 validation patients after seed20261031 shuffle, two slices each;
  noise seed0 for early stopping. Same development cohort as P1, includes4 pilot
  patients; NOT an independent validation claim. Other validation patients remain
  untouched. Nominal remaining37 assumes57 actual CSV patients; check metadata.
- No test image array opened. Test CSV only verifies split-disjoint patient IDs.
- Validate every500; save every250. After>=3000 steps, stop after6 checks without
  >.01dB development PSNR improvement. Plateau is not proof of true convergence.
- For future fair fixed-model comparison, evaluate on the P1 three paired noise
  seeds only after model selection; do not select checkpoint using those results.
  A later independent37-patient confirmation needs a separately frozen evaluation
  plan; this trainer must never select checkpoints on those37.

## Authorized next-stage candidate, only after P1 review

From isolated code tree, ensure platform deadline remains valid and no P1 process
is running. Use a fresh output and a process timeout. No shutdown on exit.

```
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
OUT=/root/autodl-tmp/flowct_runs/fbp_ld180_throughput_$(date -u +%Y%m%dT%H%M%SZ)
timeout --signal=TERM --kill-after=90s 1200s python -m experiments.budget_pilot.train_fbp_safe \
 --config experiments/budget_pilot/fbp_lowdose_safe.yaml --out "$OUT" \
 --stop-after 200 --max-seconds 1080
```

The200-step cap does not change the20000-step learning-rate schedule. A complete
model/optimizer/RNG checkpoint is saved, reusable work rather than discarded
smoke training. Inspect step21–200 median/P90, all finite losses, peak memory,
validation and checkpoint overhead. Estimate hours as remaining_steps×median
plus measured overhead; no claim that20000steps is necessary or sufficient.
If OOM, preserve failure and start fresh with an explicitly changed batch config;
do not change batch in a checkpoint resume silently.

After approval from actual throughput, resume SAME config and identical data
hashes into a different new output. Verify generated SHA before torch.load:

```
SHA=$(python -c "import json; print(json.load(open('$OUT/last.pt.sha256.json'))['sha256'])")
timeout --signal=TERM --kill-after=90s 14400s python -m experiments.budget_pilot.train_fbp_safe \
 --config experiments/budget_pilot/fbp_lowdose_safe.yaml --resume "$OUT/last.pt" \
 --resume-sha256 "$SHA" --out "$NEW_OUT" --max-seconds 13800
```

Do not launch a4h stage unless remaining platform time and monetary budget cover
it plus shutdown/retrieval margin. Parent may set a shorter duration from observed
throughput. Checkpoints and manifests include private cohort IDs; only aggregate
training losses, step durations, validation means and status may leave AutoDL.

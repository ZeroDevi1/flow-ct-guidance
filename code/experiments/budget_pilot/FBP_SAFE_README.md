# FBPConvNet validation-development training

**Do not use the historical train_fbpconvnet entry point:** it automatically reads
`test200`, only saves at the end, and does not provide safe resume or validation
convergence tracking. This additive trainer never reads test image arrays. It
checks test CSV patient IDs only for leakage. No prior model is retrained.

Start with exactly one setting (`sv60`), batch 8, existing U-Net architecture,
supervised full-image MSE, AdamW and bf16. Training sampling is uniform with
replacement and random flips, not the original epoch-based shuffle. This change
must be reported in comparisons. All train slices are eligible. No weights are
silently initialized from old artifacts; this is from-scratch baseline training.

Validation is the same fixed first 20 shuffled validation patients, 2 evenly
spaced slices per patient, selection seed 20261031. These are development data,
NOT held-out confirmation data. Remaining validation patients are not touched by
this runner. The prior pilot may already have used some selected patients.
Evaluation is fp32, clipped predictions, circle-ROI PSNR and SSIM; before clipping
raw predictions are checked for non-finites. Validation patient-macro PSNR selects
checkpoints. A plateau means 6 evaluations without >0.01 dB improvement after at
least 3000 steps, evaluated every 500 steps. This is an operational stopping rule,
not proof of population convergence. Time limit may end before plateau.

## Initial 200-step throughput measurement (needs parent approval)

Before any GPU launch verify platform hard shutdown time and budget. Then, from
`code`, with a **fresh OUT**:

```sh
export PYTHONPATH="$PWD"
timeout --signal=TERM --kill-after=90s 1200s \
  python -m experiments.budget_pilot.train_fbp_safe \
    --config experiments/budget_pilot/fbp_sv60_safe.yaml \
    --out "$OUT" --stop-after 200 --max-seconds 1080
```

Use measured `train.jsonl` step durations after initial 20-step warmup (median and
P90), validation/checkpoint durations and peak CUDA allocation to estimate the
remaining run. The 200-step checkpoint is useful and resumable; `--stop-after`
does not shorten the 20,000-step LR schedule. Batch 8 is a conservative starting
point, NOT a verified 5090 memory claim. If OOM, preserve failure and change batch
in a fresh run rather than silently changing resumed configuration.

## Continue under a separately approved four-hour stage

The checkpoint is a trusted local artifact containing Python/NumPy RNG states;
its SHA256 must match the generated `last.pt.sha256.json`. CLI refuses resume
without matching hash. Supply a fresh OUT2; leave the first output unchanged.

```sh
timeout --signal=TERM --kill-after=90s 14400s \
  python -m experiments.budget_pilot.train_fbp_safe \
    --config experiments/budget_pilot/fbp_sv60_safe.yaml \
    --resume "$OUT/last.pt" --resume-sha256 "$CHECKPOINT_SHA256" \
    --out "$OUT2" --max-seconds 13800
```

The runner does not shut down the GPU instance; the platform deadline protects
whole-session billing. Checkpoints every 250 steps include model, optimizer,
step, config, data hashes, cohort, early-stop state, Python/NumPy/Torch/CUDA,
sampling and measurement RNG. Scaler is `None` because BF16 needs no scaling.
Resume requires identical config, data hashes and selected cohort. On TERM/INT
it saves after the current step/evaluation; SIGKILL can only recover the latest
periodic checkpoint. CPU replay is tested bit-exact; CUDA kernels remain subject
to the documented nondeterminism, so identical GPU replay is not promised.

All manifest/checkpoints contain private cohort identifiers. No public uploads.

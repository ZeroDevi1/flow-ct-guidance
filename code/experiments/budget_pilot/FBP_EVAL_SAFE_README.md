# Frozen FBPConvNet evaluation against completed P1

This evaluator completed the recorded 120-prediction GPU evaluation on 2026-10-31; see [the final report](BASELINE_RESULT_20261031.md). The following instructions describe reproduction. Run only
once a locally trained low-dose `best.pt` has been frozen and its recorded SHA256
has been verified. It does not train, choose checkpoints, tune parameters, read
test images, overwrite outputs, upload images, or shut down the GPU instance.
A `best.pt` name and successful execution do not establish training convergence.

## Preflight gates

- P1 batch and all three seed statuses must be complete, exit code zero.
- Exact protocol: 20 patients × 2 slices, seeds `[0,1,2]`, low-dose180 `I0=1e5`;
  one frozen lambda per DC/WLS. Per-seed settings must equal batch configuration.
- Actual paired per-image records must match the entire cohort with no missing or
  duplicate rows, regardless of any existing complete flag.
- Cross-seed validation array/index SHA256 and cohorts must match. Current local
  validation files must match those hashes.
- Checkpoint bytes must match the supplied SHA256 **before `torch.load`**. Only
  locally produced trusted `flowct_fbp_safe_v1` checkpoints are intended; loading
  a hash-verified arbitrary untrusted pickle is not safe.
- Checkpoint physics must be 180 angles, full pi range, I0=1e5, sigma0, mu=.06;
  exactly the same 20-patient/40-slice cohort and validation hashes; complete
  same-step validation metadata must accompany the frozen best checkpoint.

## Approved execution command (not yet run)

From the deployed `code` directory, with a verified platform budget deadline:

```sh
export PYTHONPATH="$PWD"
timeout --signal=TERM --kill-after=30s 1200s \
  python -m experiments.budget_pilot.eval_fbp_safe \
    --batch "$COMPLETED_P1_BATCH" \
    --checkpoint "$FROZEN_TRAIN_OUTPUT/best.pt" \
    --checkpoint-sha256 "$FROZEN_CHECKPOINT_SHA256" \
    --processed /root/autodl-tmp/data/totalseg/processed \
    --out "$FRESH_EVAL_OUTPUT" --max-seconds 900
```

The internal limit bounds the prediction phase; the external timeout also bounds
preflight hashing/deserialization. The operator must keep the paid-instance
platform hard deadline active. Failure/timeout does not trigger shutdown.

## Outputs and interpretation

Each noisy measurement uses `1000003*(seed+1)+array_index`, exactly as P1. The
frozen FBPConvNet is evaluated fp32 as FBP plus model residual, finite-checked
**before clipping**, then clipped to [-1,1]. PSNR/SSIM use the same circle ROI.
Every regenerated FBP PSNR and SSIM must match BOTH P1 method records within1e-5
before accepting a paired comparison. A mismatch fails rather than being hidden.

`manifest_private.json` and `per_image_private.jsonl` contain patient IDs/indices:
keep them private on the instance. Only `summary_anonymous.json` and `status.json`
are intended for aggregate export. The anonymous summary is written only after
all 120 FBPConvNet predictions exist and all pairing checks pass. On runtime
failure, private partial rows remain; status forbids comparison.

Metrics average slices and seeds within each patient before averaging patients.
10,000 deterministic bootstrap draws resample patients, not images or seeds.
Intervals are exploratory: the same development cohort chose training checkpoint
and DC/WLS parameters. They are not independent validation, not clinical evidence,
and omit uncertainty from training randomness and model selection. Prediction
timing excludes FBP generation and is not a matched total-compute comparison.

CPU tests cover finite-before-clamp residual prediction, exact patient-level
aggregation, absence of identifiers from anonymous output, missing-record and
regenerated-FBP rejection, protocol/cohort rejection, and SHA failure before
unpickling. GPU full-pipeline evaluation remains untested until approved.

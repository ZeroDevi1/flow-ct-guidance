> Publication note (2026-10-31): the user authorized publishing reviewed code and anonymous aggregates. Historical preparation notes below describe the original private run. Patient-level experiment outputs remain excluded. After clarification that checkpoint cohort IDs belong to the public dataset, the user additionally authorized original full checkpoints as Release assets; see the publication policy in the result directory.

# Matched lowdose FBP residual-U-Net result — 2026-10-31

Public aggregate report, authorized after privacy review. Patient identifiers, images and
identity-bearing checkpoints remain private. This is development-validation,
not independent or clinical validation.

## Training and integrity

200-step throughput stage retained the same20000-step LR schedule; full optimizer,
model and RNG state resumed into a new directory. The subsequent19800steps took
2073.237s (34.55min), ending at20000 due to the predefined step limit, NOT the
validation-plateau rule. Peak allocated CUDA memory9,833,995,264bytes. Best and
last checkpoints are separately hashed and preserved.

Validation PSNR still increased from37.2001 at18000 to37.2765 at20000; do not call
this proven convergence, nor extend training simply to seek better numbers.
The frozen best checkpoint is step20000, selected only on the fixed20-patient
noise-seed0 development cohort. The remaining validation patients and all final
test images were not used. No frozen flow-prior weights were retrained.

- best.pt SHA256 `9879a1124f78add18c4ebcd1593983ef310bc3cc68327b886c041b0cddf6fbaf`
- last.pt SHA256 `2856d0ff1758d067013b1b006fe79cef6e909b5d6b7462a9e870540e6012b5b3`
- anonymized evidence archive SHA256
  `983fa72b4b946c766e92c948c9ea92028da9b5467fd3a93e44027c87a30ec7e6`

## Frozen checkpoint comparison

The evaluator verified checkpoint hash before deserialization, matched the P1
validation array/index hashes and exact patient/slice cohort, regenerated the
same3 noise seeds and verified per-image FBP metrics. All120 neural predictions
completed; no partial-cohort comparison was allowed.

| Method | Patient-macro PSNR dB | SSIM |
|---|---:|---:|
| Original FBP |27.2668|.61724|
| Flow-prior DC lambda3, NFE50 CG30 |25.8361|.57170|
| Flow-prior WLS lambda.3, NFE50 CG30 |31.7144|.72342|
| Supervised FBP residual-U-Net, step20000 |37.2846|.91129|

FBP network minus WLS: mean+5.5702dB, median+5.1327,19/20 patients positive.
Exploratory patient-bootstrap95% interval[4.1640,6.9976]. This interval averages
slices and noise seeds within each patient; it omits training-seed uncertainty
and development-cohort/checkpoint selection effects, and is not confirmatory.

Neural prediction averaged.0110s per image, **excluding FBP construction**.
Do not quote an end-to-end speedup against the P1 sampler without including all
preprocessing and matching timing boundaries. This supervised network uses
setting-specific training; the flow prior was fixed, so training costs and
adaptation requirements also differ.

## What this changes

The WLS formulation improved over this fixed DC setup, but it did NOT outperform
a trained matched-setting baseline. Reporting only WLS-vs-DC would overstate the
result. DC itself was worse than FBP, and the separate same-image audit showed
NFE50 worse thanNFE20 for both prior methods; equal fixed lambda per step does not
hold total guidance constant when NFE changes.

Useful next scientific steps, **proposals only, not run**:
1. Freeze the current trained baseline and predeclare a remaining-validation-
   patient confirmation protocol before any new tuning; do not consume final test.
2. On the existing development subset only, control NFE/lambda scaling and CG
   solve accuracy independently rather than presenting moresteps as automatically
   better. Preserve negative findings and compare to the fixed strong baseline.
3. If pursuing noise-aware improvements, establish whether gains survive proper
   tuning against DC, a fixed training budget, and untouched patients. Do not
   generalize synthetic parallel-beam restoration to diagnostic performance.

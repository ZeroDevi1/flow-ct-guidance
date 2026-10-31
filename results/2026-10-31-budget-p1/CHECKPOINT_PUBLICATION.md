# Full checkpoint publication and data attribution

The user explicitly requested original complete checkpoints and clarified that
the cohort indices are from the public dataset after being told the repository
is public. Publication is therefore authorized for the original `best.pt` and
`last.pt`, not merely inference weights. No binary checkpoint is stored in git.

Both files were read-only audited: no credentials, tokens, cookies, SSH keys or
private secrets detected. They retain model/optimizer/RNG/config/step/early-stop
state,20 public-dataset subjects ×2 slice indices, and the generic path
`/root/autodl-tmp/data/totalseg/processed`. These metadata are intentionally
retained for exact resume, not silently removed. Non-checkpoint patient-level
outputs and images are not part of this publication.

| Original asset | Bytes | SHA256 |
|---|---:|---|
| best.pt |238285915|`9879a1124f78add18c4ebcd1593983ef310bc3cc68327b886c041b0cddf6fbaf`|
| last.pt |238285915|`2856d0ff1758d067013b1b006fe79cef6e909b5d6b7462a9e870540e6012b5b3`|

The source data are **TotalSegmentator v2.0.1**, attributed to **Jakob Wasserthal
and colleagues / University Hospital Basel**. The official Zenodo record
[10.5281/zenodo.10047292](https://doi.org/10.5281/zenodo.10047292) declares
[CC BY4.0](https://creativecommons.org/licenses/by/4.0/) in its
[official metadata](https://zenodo.org/records/10047292/export/json). See the
[original paper](https://doi.org/10.1148/ryai.230024).

These checkpoints were produced by this project's preprocessing and training;
they are not official TotalSegmentator model weights and imply no endorsement.
Do not confuse the upstream software license with the dataset's CC BY4.0.
Preserve attribution and the source/license links when redistributing.

Only load checkpoints from trusted sources after SHA verification. This complete
training format includes Python/NumPy RNG objects and uses trusted
`torch.load(..., weights_only=False)` in the safe resume tool; an arbitrary
checkpoint is not safe merely because its filename matches.

Release creation/upload is verified separately; this file is not proof that a
Release asset has finished uploading.

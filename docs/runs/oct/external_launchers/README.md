# External launchers and OCT working-tree changes

Snapshot taken 2026-10-21T16:16:56Z by the post-submission audit. These lived only in the untracked
OpenCharacterTraining clone on the experiment volume (OCT commit `d1da9f0`), so manifests that named
them (e.g. `bash finetuning/introspection/llama_curve.sh goodness`) could not be reproduced from git.
Historical manifests carry no launcher hash, so that these files are byte-identical to the versions that ran
is unverified (mtimes predate the runs that used them; later manifests record `launcher_sha256`).
`character/constants.py` (machine paths) is gitignored upstream and reconstructed in the OCT clone; not copied.

| file | sha256 |
|---|---|
| `finetuning/distillation/llama_local.sh` | `135c7ccc4173ce07…` |
| `finetuning/distillation/llama_s2sham.sh` | `80528372135e7da3…` |
| `finetuning/distillation/llama_seed2.sh` | `3c2eba4a6ae76530…` |
| `finetuning/introspection/llama_curve.sh` | `08e1922148a401e2…` |
| `finetuning/introspection/llama_local.sh` | `72eb03fbd1369d46…` |
| `finetuning/introspection/llama_seed2.sh` | `1c6df9a2a80069ad…` |
| `finetuning/introspection/llama_sft_from_base.sh` | `1b47ca4cd4fd8338…` |
| `tracked_changes.diff` | `7a1f80f29e1e8f0e…` |

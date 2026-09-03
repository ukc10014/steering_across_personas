# OCT training rig — decisions recorded before the first training run

Written 2026-09-03, before any training was launched, per docs/NEXT_POD.md step 4. Both of
these change numerics, so they are provenance, not preference. **Fixed for every run in this
experiment** — the reproduction (seed 123456) and seed 2 alike.

## Hardware

| | |
|---|---|
| GPU | 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB |
| driver | 595.91.07 |
| arch | `sm_120` (Blackwell) — needs torch ≥ 2.7 built for cu128 or later |

97 GB fits SFT at `max_len 3072`, so no optimisation-config value was touched (spec §8).

## Decision 1 — flash-attn: OUT

`flash_attn` is **not installed** and is not importable in either the training or the
measurement environment. OpenRLHF uses it opportunistically if importable, so its absence is
a numerical fact about these runs, not a non-event; `oct_provenance.py` records it as
`"flash_attn": null` in every stage manifest.

Rationale: the OpenRLHF fork removed flash-attn from `requirements.txt`, so "install nothing
extra" is both the default and the reproducible choice. Building it for `sm_120` would also
have been a from-source compile with its own version surface.

## Decision 2 — wandb: OFF

Training uses the patched runners (`llama_local.sh`, `llama_seed2.sh`), which pass
`--use_wandb False`. No `.env` is read and no run is logged to wandb. The upstream `llama.sh`
path — which expects wandb credentials — is **not** used.

The `wandb` package is nonetheless present in `PYLIBS_TRAIN` because it is in the fork's
`requirements.txt`; installed-but-unused is recorded here so a later reader does not infer
logging from the package list.

## Decision 3 — torch provenance

`requirements.txt` lists a bare `torch`, so `pip install --target=$PYLIBS_TRAIN` fetches a
torch that shadows the system build. Which one actually drove the training is recorded in the
per-stage `oct_provenance.py` manifests (`versions.torch`, `versions.cuda`). The chosen build
was verified with a real CUDA matmul, not `torch.cuda.is_available()` — on this arch the
latter returns True for a build that cannot launch a kernel.

## Environments (kept separate, spec §6a)

| | path | purpose |
|---|---|---|
| training | `/workspace/pylibs-train-py312` + `PYTHONPATH` onto the OpenRLHF **fork** | steps 5, 7 |
| measurement | `/workspace/pylibs-py312` | steps 6, 8 |

The fork is deliberately not pip-installed: `maiush/OpenRLHF` adds length normalisation, a KL
penalty and the `--kl_loss_coef 0.001` the runners pass. A pip `openrlhf` would silently train
a different objective.

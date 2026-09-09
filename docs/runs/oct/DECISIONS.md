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

## Decision 1 — flash-attn: IN (`flash_attn 2.8.3.post1`)

Installed into `PYLIBS_TRAIN` only, and **not** into the measurement env, which stays without
it. `oct_provenance.py` records the version in every stage manifest.

**This reverses a first pass that said "out", and the reversal is the point.** "Out" was
chosen when flash-attn looked like an optional accelerator. It is not: `train_dpo.py:205` and
`train_sft.py` default `--attn_implementation` to **`flash_attention_2`**, and neither the
patched runners nor upstream's `llama.sh` override it. So flash-attn absent is not "install
nothing extra" — it is an active override of the released pipeline's default attention
implementation, and it would have had to be passed as an extra flag the original run did not
pass. Whatever machine produced the released adapters therefore almost certainly had
flash-attn importable and trained with FA2.

Since the gate (spec §6b) asks whether our rig reproduces *the released adapter*, matching its
attention implementation removes a confound rather than adding one. The remaining question was
only whether FA2 runs on this card, and it does — verified with a real kernel launch, not an
import:

```
flash_attn 2.8.3.post1  ->  flash_attn_func on NVIDIA RTX PRO 6000 Blackwell: OK
```

Wheel: `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`,
prebuilt, matching `torch._C._GLIBCXX_USE_CXX11_ABI == True`. No source build.

Had it not run on `sm_120`, the fallback was `--attn_implementation sdpa` — exact attention,
differing from FA2 only in accumulation order, i.e. the same class of difference as the GPU
nondeterminism §6b already accepts — recorded as a named deviation. That fallback was not
needed.

## Decision 2 — wandb: OFF, by omitting the flag (not by `--use_wandb False`)

No run is logged to wandb. No `.env` is read. Upstream's `llama.sh` path, which does
`wandb login $WANDB_TOKEN` and passes `--use_wandb True`, is not used.

**`--use_wandb False` does not disable wandb, and the first training attempt died on it.**
`--use_wandb` is declared `type=str, default=None`, and `dpo_trainer.py:82` gates on
`if self.strategy.args.use_wandb:` — so the *string* `"False"` is truthy, wandb is switched
**on**, and `wandb.login(key="False")` is called with `"False"` as the API key:

```
wandb.errors.errors.AuthenticationError: API key must have 40+ characters, has 5.
```

It fails ~90 s in, after both models and the tokenized dataset have loaded, which is late
enough to look like a training problem rather than a flag problem. Fixed by **deleting the
`--use_wandb` line** from the patched runners; the argparse default of `None` is falsy, so
the whole wandb block is skipped. Verified in the relaunched process's command line.

This is a rig bug in our own patched runners, not in upstream, and it changes no numerics —
wandb was meant to be off and is off. It is recorded because "the wandb choice" was supposed
to be settled at this step, and the first way of expressing it silently meant the opposite.

Fixed in three of the four runners: `distillation/llama_local.sh`,
`distillation/llama_seed2.sh`, `introspection/llama_local.sh`. **`introspection/llama_seed2.sh`
still carries the bad line** — two edit attempts were blocked by the sandbox — and must be
fixed before step 7, or the seed-2 SFT stage will fail the same way. `newpod.sh`'s "differs in
exactly `--seed`" check currently reports the extra line, which is the intended tripwire.

## Decision 3 — torch: the system build, not the one pip chose

`requirements.txt` lists a bare `torch`, and `pip install --target=$PYLIBS_TRAIN` resolved it
to **torch 2.14.0+cu130**, which shadowed the system build. That combination is broken here:
the container's `torchvision`/`torchaudio` are built against torch 2.8, so
`import peft` died with `RuntimeError: operator torchvision::nms does not exist` — the
CLAUDE.md gotcha, arriving through `PYLIBS_TRAIN` this time rather than `$PYLIBS`.

Resolved by moving torch, `triton`, and the `nvidia-*-cu13` wheels out of `PYLIBS_TRAIN`
(quarantined at `/workspace/pylibs-train-py312-quarantine`, not deleted) so the training env
uses the **system torch 2.8.0+cu128**, which has matching `torchvision 0.23.0+cu128`,
`torchaudio 2.8.0+cu128` and `triton 3.4.0`.

Verified by a real CUDA matmul, not `torch.cuda.is_available()`:

```
torch 2.8.0+cu128  /usr/local/lib/python3.12/dist-packages/torch/__init__.py
arch list: sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120   <- sm_120 native, not PTX JIT
```

Training stack as actually used: torch 2.8.0+cu128, transformers 4.57.0, deepspeed 0.18.0,
peft 0.20.0, accelerate 1.14.0, ray 2.48.0, flash-attn 2.8.3.post1.

## Environments (kept separate, spec §6a)

| | path | purpose |
|---|---|---|
| training | `/workspace/pylibs-train-py312` + `PYTHONPATH` onto the OpenRLHF **fork** | steps 5, 7 |
| measurement | `/workspace/pylibs-py312` | steps 6, 8 |

**Measurement env versions matter as much as the training ones.** A plain
`pip install transformers` into `$PYLIBS` resolved to **5.16.1**. Every published arm — the
released `impulsiveness` **+2.18** the gate is scored against — was measured on
**transformers 4.57.6**, downgraded deliberately on 2026-08-10 (`docs/fork-infra.md`).
Measuring the reproduction on a 5.x stack would have compared two different measurement
stacks and attributed the difference to the rig. Re-pinned with `"transformers>=4.45,<5"`;
`accelerate` held to `--no-deps` so it cannot drag a torch into `$PYLIBS`.

Measurement env as actually used: transformers 4.57.6, tokenizers 0.22.2, numpy 2.5.2,
scikit-learn 1.9.0, accelerate 1.14.0, torch 2.8.0+cu128 (system, unshadowed), no flash-attn.
This matches the 2026-08-10 configuration the published numbers were produced on.

Note `/workspace/requirements-snapshot.txt` is **stale** — it still records
`transformers==5.14.1`, i.e. the version that downgrade replaced. It is what a naive
re-provision would follow.

The fork is deliberately not pip-installed: `maiush/OpenRLHF` adds length normalisation, a KL
penalty and the `--kl_loss_coef 0.001` the runners pass. A pip `openrlhf` would silently train
a different objective.

---

## 2026-09-09 — volume cleanup, minimal tier

Volume was at 837 GB (of which ~24 GB was a transient raw-activation dir belonging to the
running SFT-curve extraction). Deleted two trees, 94 GB:

| deleted | size | why it is safe |
|---|---|---|
| `outputs/Llama-3.1-8B-Instruct/caa_activations_paraphrase` | 79 GB | Its derived results are archived in `outputs/Llama-3.1-8B-Instruct/analysis/` — `caa_variance_decomposition.json`, `caa_magnitude.json`, `caa_within_cell_stability.json`. Re-extraction is ~50 min on one RTX PRO 6000 ([results/llama31_8b_b1_noise_floor.md](../../results/llama31_8b_b1_noise_floor.md) §7–8). |
| `outputs/_actprobe` | 17 GB | Results archived in `outputs/analysis/activation_dose_probe*.json` (4 files). |

**Not deleted, and deliberately so.** 516 GB of raw `caa_activations` across 22 arms that all
have qcaches, plus the 79 GB above, was the fuller option under the §8a pattern (extract →
qcache → drop raw). The qcache is a genuine substitute for the published work — it holds the
complete per-question tensor, `(8 traits, 12 personas, 2 directions, 500 questions, 2 layers,
4096)` at 1.57 GB, and **every** workshop figure reads layer 15 or 20. The cost of dropping
raw is re-analysis at any *other* layer, which `caa_magnitude.py` in particular does (it
reports L25). That optionality was judged worth keeping while the volume is not actually
full. Revisit if it gets tight.

Audit, should it be needed again: 12 scale-ladder arms 281 GB, 10 headline arms 235 GB.

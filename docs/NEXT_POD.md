# Next pod: stage localisation

**Read this first if you are a fresh Claude session on a new pod.**

The previous runbook (reproduce seed 123456, then run seed 2) is **complete and passed** —
archived at [archive/NEXT_POD_repro_seed2_DONE.md](archive/NEXT_POD_repro_seed2_DONE.md).
The current experiment is [spec_stage_localisation.md](spec_stage_localisation.md).

## The question you are answering

> Where in the OCT pipeline does the strong `impulsiveness` phenotype emerge: the DPO weight
> update, the DPO-generated introspection data, introspection SFT conditional on the DPO
> state, or the final PEFT adapter merge?

Causal decomposition, not another set of adapters.

---

## 0. First command, always

```bash
bash /workspace/oct_rig/newpod.sh        # must print NEWPOD OK
```

## 1. What already exists — do not rebuild it

| | where | state |
|---|---|---|
| seed 1 adapters (dpo / sft / merged) | `/workspace/oct_rig/loras_repro` | passed all nine §6b criteria |
| seed 2 adapters (dpo / sft / merged) | `/workspace/oct_rig/loras_seed2` | passed all nine independently |
| 1-epoch ablation artifact | `/workspace/oct_rig/loras_repro_sft1ep` | keep — it is a measured result |
| CAA activations, 4 rig arms | `outputs/llama-3.1-8b-impulsiveness_{repro,repro_dpo,seed2,seed2_dpo}` | 192/192 each |
| adapters, off-volume backup | `kanad/oct-impulsiveness-seed-replication` (HF, private) | all six |

**Nothing above may be overwritten.** New runs get new names (spec §8).

## 2. Environment — the traps that cost hours last time

If the pod is **Python 3.12**, both trees are already built and correct on the volume; just
verify. If it is a different Python, `$PYLIBS` is version-scoped and you must rebuild — and
then every trap below applies again.

- **`pip install -r openrlhf/requirements.txt` installs torch 2.14+cu130 into `PYLIBS_TRAIN`**,
  which shadows the system torch and re-creates the `torchvision::nms` mismatch. It surfaces
  as `import peft` failing, not as anything torch-shaped. Fix: move `torch`, `triton` and the
  `nvidia-*` wheels out of `PYLIBS_TRAIN` and use the **system torch 2.8.0+cu128**, which has
  the matching torchvision/torchaudio/triton and native `sm_120`. Verify with a real CUDA
  matmul, never `torch.cuda.is_available()`.
- **Measurement must be transformers 4.57.6**, not whatever pip resolves (it picks 5.x).
  Every published arm was measured on 4.57.6. `/workspace/requirements-snapshot.txt` is stale
  and says 5.14.1 — do not follow it.
- **flash-attn is IN**, and the prebuilt wheel is already on the volume:
  `/workspace/tmp/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`.
  It is not optional: `train_dpo.py` defaults `--attn_implementation` to `flash_attention_2`
  and no runner overrides it, so its absence would be a deviation from the released pipeline.
- **`merge_lora.py` needs `peft`**, which is deliberately absent from the measurement env
  (installing it there can drag in a second torch). Run merges with
  `PYTHONPATH=/workspace/pylibs-train-py312` and extractions with `$PYLIBS`.
- **Runners are already corrected** and must stay that way: `--max_epochs 3` on introspection
  (the release-era value — HEAD's 1 is post-release), and **no `--use_wandb` line at all**
  (the string `"False"` is truthy and enables wandb, then fails as an API key).

## 3. Order of work

Follow [spec_stage_localisation.md](spec_stage_localisation.md). Two things first, both cheap:

```bash
# seed 1's folded DPO model was deleted in the 2026-09-03 disk reclaim; M_D and M_D+S need it
cd /workspace/OpenCharacterTraining
python3 tools/fold_loras.py --model_name llama-3.1-8b-it \
  --loras_dir $HOME/loras/llama-distillation --save_dir_name distilled_repro   # ~3 min
```

...then the five buildable states (§1, §2), which need **no training** and answer "is the
phenotype installed by SFT or by the merge?" on their own. Training (`M_S`, the checkpoint
curve) comes after.

**Disk is the binding constraint**, not GPU. Activations are 24 GB per arm and the full plan
is ~530 GB against ~275 GB free. The geometry reads the 1.57 GB qcache, not the raw tensors:
extract → build qcache → delete the raw activations (spec §8a). This is a recorded decision,
not a silent cleanup.

## 4. Watchdog bugs that wasted GPU time last session

- `pgrep -f <script>` **matches the waiting shell itself** — anchor the pattern
  (`'^bash /path/script\.sh$'`) or wait on a log marker instead.
- `cmd | tail -30` makes the pipeline's exit status `tail`'s, so a failed stage looks fine.
  Give every stage its own `|| die`.
- Chain the next stage to the previous one's completion; a waiter that only *notifies* leaves
  the GPU idle until someone asks.

## 5. Report, and stop

Deliverable is spec §10 — **diagnostic, not interpretive**. Do not change any threshold in
[spec_sham_lora.md](spec_sham_lora.md) §5.1 or §6b, and do not edit an existing workshop
claim to match a new number.

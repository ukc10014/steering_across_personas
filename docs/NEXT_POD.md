# Next pod: step by step

**Read this first if you are a fresh Claude session on a new pod.** Everything here has been
staged and tested on the previous pod except the training runs themselves, which need the big
GPU. Follow the steps in order; step 6 is a hard gate.

Context: [spec_sham_lora.md](spec_sham_lora.md) is the experiment (§6 is the protocol, §6b the
gate criteria). [oct_rig_setup.md](oct_rig_setup.md) is what the rig is and what is left.

---

## 0. Deploy the pod

- **Attach the same network volume.** Everything lives on it; the pod itself is disposable.
  You cannot add a GPU to a running RunPod pod — a bigger card means a new pod.
- Pick the ≥80 GB card. 96 GB is what the plan assumes.
- A network volume mounts to one pod at a time, so the old pod loses `/workspace` when this
  one takes it. Nothing is lost — but any session running there stops being useful.

## 1. Connect, and start tmux immediately

```bash
tmux new -s work          # or: tmux attach -t work
```

Dropped connections are client-side SSH, not the pod (`docs/tmux.md`). Long jobs go in tmux or
`nohup`, never a bare foreground shell.

## 2. Bring the pod up

```bash
bash /workspace/oct_rig/newpod.sh
```

Must print **`NEWPOD OK`**. It checks the volume is the right one, reports the GPU and warns if
VRAM cannot fit SFT at `max_len 3072`, rebuilds the `$HOME` symlinks the OCT scripts hardcode
(`$HOME` is `/root`, *not* on the volume, destroyed with each pod), verifies both frozen data
files against their sha256, confirms the OpenRLHF **fork** carries `--kl_loss_coef`, and checks
the patched runner scripts.

If it says PYLIBS is empty, this pod's Python differs from the one that provisioned the
libraries — re-provision before measuring anything (`CLAUDE.md`, "Environment bootstrap").

## 3. Build the training environment — separate from the measurement one

```bash
source /workspace/bootstrap.sh
export PYLIBS_TRAIN=/workspace/pylibs-train-py311        # match the pod's python version
pip install --target="$PYLIBS_TRAIN" -r /workspace/OpenCharacterTraining/openrlhf/requirements.txt
export PYTHONPATH="$PYLIBS_TRAIN:/workspace/OpenCharacterTraining/openrlhf:/workspace/OpenCharacterTraining"
python3 -c "import openrlhf, character, deepspeed; print(openrlhf.__file__)"
```

The last line **must** print a path under `/workspace/OpenCharacterTraining/openrlhf/`. The
fork is deliberately not pip-installed — `maiush/OpenRLHF` adds length normalisation, a KL
penalty and the `--kl_loss_coef` flag the runners pass as `0.001`, none of which exist
upstream. A pip `openrlhf` would train **a different objective** and look fine doing it.

Do not install any of this into `$PYLIBS`: the fork pins `transformers==4.57.0` /
`deepspeed==0.18.0` / `ray==2.48.0`, and the measurement stack runs 4.57.6.

## 4. Record two decisions before training

Both change numerics, so they are provenance, not preference:

- **flash-attn** — installed or not. The fork removed it from requirements; OpenRLHF uses it if
  importable. Pick one, keep it fixed across all runs.
- **wandb** — the patched runners already set `--use_wandb False`. If you want wandb instead,
  use the upstream `llama.sh` with a `.env`, and record that you did.

## 5. Reproduce seed 123456

Four stages. **Run `oct_provenance.py` at every one** — provenance recorded afterwards is a
reconstruction.

```bash
cd /workspace/OpenCharacterTraining
R=/workspace/repos/steering_across_personas

# 5a. DPO
python3 $R/scripts/oct_provenance.py --run repro-123456 --stage dpo \
  --cmd "bash finetuning/distillation/llama_local.sh impulsiveness"
bash finetuning/distillation/llama_local.sh impulsiveness

# 5b. fold the DPO LoRA into the base -> the "distilled" model the SFT stage trains on
python tools/fold_loras.py --model_name llama-3.1-8b-it \
  --loras_dir $HOME/loras/llama-distillation --save_dir_name distilled

# 5c. introspection SFT, on the frozen corpus
bash finetuning/introspection/llama_local.sh impulsiveness

# 5d. the weighted merge -> the final persona adapter
python tools/merge_loras.py --model_name llama-3.1-8b-it --constitution impulsiveness
```

Then freeze the tree so seed 2 cannot overwrite it, and give the arms their names:

```bash
mv /workspace/oct_rig/loras /workspace/oct_rig/loras_repro
mkdir -p /workspace/oct_rig/loras && ln -sfn llama-introspection /workspace/oct_rig/loras/llama-test
```

Expect `adapter_config.json` to read **r=64, lora_alpha=64** even though both stages trained at
alpha=128. That is correct and is criterion A1 — see spec §6c. **Do not hand-adjust alpha.**

## 6. GATE — score the reproduction against §6b

```bash
cd $R
ARMS="impulsiveness_repro impulsiveness_repro_dpo" bash scripts/run_caa_logits.sh   # ~56 min
python scripts/caa_logits_analysis.py
```

The cheapest sharp check first: the registered contrast (`impulsivity` alone, forced) should
land near the released **+2.18**. Criterion B1 wants ≥ +1.5. Running this before any geometry
means a broken rig costs ~28 min rather than a full measurement pass.

Then the rest of §6b: `adapter_config`, ‖dW‖_F overall and per module, functional dose,
`cos(dG_repro, dG_released)`, retention `k`.

> **If the reproduction fails §6b, STOP. Do not run seed 2.**
> Start diagnosis from the `llama-test` assumption (spec §6a): `merge_loras.py` reads an SFT
> adapter from a directory nothing in the public repo writes, and we symlinked it to
> `llama-introspection`. An unpublished or different `llama-test` artifact is the first
> candidate explanation for a near-miss.
>
> A middling result is also a stop. Nobody has measured what hardware nondeterminism alone
> costs on `cos(dG_repro, dG_released)`, so e.g. 0.78 means "cannot separate rig error from
> nondeterminism" — which makes seed 2 uninterpretable just as a clear failure would.

Also, now that both component adapters exist, measure the peft cross-term share on the **real**
pair (spec §6c). Descriptive only — do not reinterpret any published result from it.

## 7. Seed 2

Only after the gate passes. The runners differ from step 5's in exactly one line, `--seed`
(verified by `newpod.sh`); the data is the same frozen bytes.

```bash
cd /workspace/OpenCharacterTraining
bash finetuning/distillation/llama_seed2.sh impulsiveness
python tools/fold_loras.py --model_name llama-3.1-8b-it \
  --loras_dir $HOME/loras/llama-distillation --save_dir_name distilled_seed2
bash finetuning/introspection/llama_seed2.sh impulsiveness
python tools/merge_loras.py --model_name llama-3.1-8b-it --constitution impulsiveness
mv /workspace/oct_rig/loras /workspace/oct_rig/loras_seed2
```

**Keep both adapters for each seed** — the DPO-stage one and the final merge. The DPO stage is
the sham's primary comparator and is free of the merge cross terms.

## 8. Measure

```bash
cd $R
ARMS="impulsiveness_seed2 impulsiveness_seed2_dpo" bash scripts/run_caa_logits.sh
python scripts/caa_logits_analysis.py
python scripts/caa_logits_robustness.py --n-boot 300
# geometry: 2c activations for the new arms, then common_shift with --bootstrap
python scripts/functional_dose.py --arms impulsiveness_seed2 impulsiveness_seed2_dpo
```

Measure functional dose for seed 2 — never infer it from weight norm (that was §7.1's error).

## 9. Report, and stop

Nine items, spec §6d. The framing that matters: seed 2 gives **one same-constitution
stochasticity reference point**, not a noise ceiling, not an upper bound, not an estimate of
seed variance. n = 1. The bootstrap around it measures CAA question uncertainty, **not**
seed-to-seed uncertainty. Holding the SFT corpus fixed probably makes the two seeds more
similar than fully independent reruns, so the similarity is plausibly upward-biased — say
"plausibly upward-biased", not "upper bound".

Report all eight per-trait cosines, not just the mean.

> **Do not start the sham in this run. Do not change any §5.1 sham threshold. Do not alter any
> existing workshop claim automatically.** Return the results; the thresholds and the paper get
> revisited as a separate recorded decision, before any sham data exists.

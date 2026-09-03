# Next pod: step by step

**Read this first if you are a fresh Claude session on a new pod.** Everything here has been
staged and tested on the previous pod except the training runs themselves, which need the big
GPU. Follow the steps in order; step 6 is a hard gate.

Context: [spec_sham_lora.md](spec_sham_lora.md) is the experiment (§6 is the protocol, §6b the
gate criteria). [oct_rig_setup.md](oct_rig_setup.md) is what the rig is and what is left.

---

## If you are the Claude session running this — your brief

You have the repo and the memory directory (both on the volume) but **not** the conversation
that produced this plan. This section is the handover; the reasoning is in
[spec_sham_lora.md](spec_sham_lora.md).

**Run unattended:** steps 4–5 (training) and step 8 (measurement). They are long and
mechanical. Use tmux or `nohup`; a dropped SSH connection is client-side and must not take a
run with it. Record provenance with `oct_provenance.py` at *every* stage — after the fact it
is a reconstruction, not a record.

**Stop and report, do not self-certify:**

- **Step 6, the gate.** Compute every §6b criterion, report them criterion by criterion, and
  stop. Do not decide for yourself that a marginal reproduction is close enough, and **do not
  adjust a threshold to accommodate a result** — §6b was written before the rig existed
  precisely so the session running the experiment cannot move the line. A middling result is a
  stop, not a pass: nobody has measured what hardware nondeterminism alone costs on
  `cos(repro, released)`, so an ambiguous value means the seed-2 reference would be
  uninterpretable.
- **Step 7** starts only after a human confirms the gate passed.
- **Step 9.** Report and stop.

**Do not, under any circumstances:**

- start the sham experiment (§5 of the spec) — it is not part of this run;
- change any threshold in spec §5.1 or §6b;
- edit any existing workshop-figure claim or results document to match a new number;
- hand-adjust `lora_alpha` (see §6c — alpha=64 out of the merge is *correct*);
- rebuild the SFT corpus between the two seeds (§6a — it breaks "change only `--seed`").

**If the reproduction misses:** the first suspect is the `llama-test` symlink assumption
(§6a), not the training config. `merge_loras.py` reads an SFT adapter from a directory nothing
in the public repo writes, and we mapped it to `llama-introspection`.

**Ask before spending:** each training run is hours on an expensive card. If a step fails
twice for the same reason, stop and report rather than trying a third variation.

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

**If step 2 reports missing measurement imports, that is expected on a pod with a new Python
version and it does not block training.** `$PYLIBS` is scoped to the interpreter
(`pylibs-py311`, `pylibs-py312`, …), so a new image gets a different — often stub — tree.
Note that **torch has never lived in `$PYLIBS`**; it comes from the system dist-packages, and
must not be shadowed. Fix with:

```bash
bash scripts/provision_measurement_env.sh
```

It reports where torch comes from, runs a real CUDA matmul as the verdict (not
`is_available()`, and not the compiled-arch list — `sm_89` is absent from a cu124 build's list
yet the 4090 runs fine by PTX JIT), installs only the non-torch packages, aborts if anything
drags a torch into `$PYLIBS`, then hands over to `preflight.sh`.

Measurement is not needed until **step 6**. You can start training first and run this in
parallel.

## 3. Build the training environment — separate from the measurement one

```bash
source /workspace/bootstrap.sh
export PYLIBS_TRAIN=/workspace/pylibs-train-py$(python3 -c 'import sys;print(f"{sys.version_info.major}{sys.version_info.minor}")')
pip install --target="$PYLIBS_TRAIN" -r /workspace/OpenCharacterTraining/openrlhf/requirements.txt
export PYTHONPATH="$PYLIBS_TRAIN:/workspace/OpenCharacterTraining/openrlhf:/workspace/OpenCharacterTraining"
python3 -c "import openrlhf, character, deepspeed; print(openrlhf.__file__)"
```

The last line **must** print a path under `/workspace/OpenCharacterTraining/openrlhf/`. The
fork is deliberately not pip-installed — `maiush/OpenRLHF` adds length normalisation, a KL
penalty and the `--kl_loss_coef` flag the runners pass as `0.001`, none of which exist
upstream. A pip `openrlhf` would train **a different objective** and look fine doing it.

**Then test that torch can actually drive this GPU**, because `requirements.txt` lists `torch`
and pip will have installed one into `PYLIBS_TRAIN`, shadowing the system build:

```bash
python3 -c "
import torch; print(torch.__version__, torch.__file__)
a = torch.randn(64,64,device='cuda'); print('cuda matmul ok', (a@a).sum().item())"
```

`torch.cuda.is_available()` is **not** sufficient — it returns True for a build that cannot
launch a kernel on this arch. Blackwell is `sm_120` and needs torch ≥ 2.7 built for cu128. If
the matmul raises, delete the shadowing copy and fall back to the system one:

```bash
rm -rf "$PYLIBS_TRAIN"/torch "$PYLIBS_TRAIN"/torch-*.dist-info
```

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

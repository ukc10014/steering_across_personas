# Next pod: dose-matched stage localisation

**Read this first if you are a fresh Claude session on a new pod.**

Two experiments are **complete — do not rerun either**:

| experiment | report | status |
|---|---|---|
| reproduce seed 123456, then seed 987654 | [runs/oct/GATE_REPORT_repro-123456.md](runs/oct/GATE_REPORT_repro-123456.md), [runs/oct/SEED2_REPORT.md](runs/oct/SEED2_REPORT.md) | both passed all nine §6b criteria |
| stage localisation, first pass | [runs/oct/STAGE_LOCALISATION_REPORT.md](runs/oct/STAGE_LOCALISATION_REPORT.md) | complete; six states measured, both seeds |
| **dose-matched stage localisation** | `runs/oct/DOSE_MATCHED_STAGE_REPORT.md` *(in progress)* | **this is the current experiment** |

Older runbooks are archived at [archive/NEXT_POD_repro_seed2_DONE.md](archive/NEXT_POD_repro_seed2_DONE.md).

## The question you are answering

> **At matched measured functional dose, do the differences between the OCT stages remain?**

Two comparisons carry it:

1. **`M_D` vs `M_S`** — does SFT on the DPO-generated corpus produce a stronger, more
   selective, more final-like phenotype than the DPO weight update itself, when both move the
   model by the same functional amount?
2. **`M_D+0.25S` vs `M_F`** — does the large apparent effect of the PEFT factor-space cross
   terms survive dose matching?

### Why this is needed

The first stage-localisation pass produced a clean-looking ordering — `M_D` +0.13, `M_D+0.25S`
+0.499, `M_F` +1.923, `M_D+S` +2.190, `M_S` +3.613 on the registered endpoint — and then
**Spearman(measured functional dose, B1) = +1.000 across all six states**. The stage ordering
*is* the dose ordering, exactly. It is not a pure dose story either (B1 per unit dose spans
15×), so there is structure on top; but stage and dose are not separated by that experiment,
and any stage claim made from it would repeat §7.1's error in a new form. Dose matching is
what separates them, and it needs **no retraining** — only `--lora-scale` on adapters that
already exist.

---

## 0. First command, always

```bash
bash /workspace/oct_rig/newpod.sh        # must print NEWPOD OK
```

## 1. What already exists — do not rebuild or overwrite it

| | where | note |
|---|---|---|
| seed 1 adapters (dpo / sft / merged) | `/workspace/oct_rig/loras_repro` | passed all nine §6b criteria |
| seed 2 adapters (dpo / sft / merged) | `/workspace/oct_rig/loras_seed2` | passed independently |
| `M_S` — SFT trained from base | `/workspace/oct_rig/loras_sft_from_base` | trained 2026-09-04, 3 epochs |
| 1-epoch ablation artifact | `/workspace/oct_rig/loras_repro_sft1ep` | keep — it is a measured result |
| stage-state qcaches (7 arms) | `outputs/_qcache/impulsiveness_*_L15_20.npz` | raw activations already deleted (§8a) |
| CAA logits, all stage states | `outputs/llama-3.1-8b-impulsiveness_*/caa_logits{,_forced}` | 88 cells each |
| stage tables | `outputs/analysis/stage_comparison_seed{1,2}.csv` | the first-pass result |
| adapters, off-volume backup | `kanad/oct-impulsiveness-seed-replication` (HF, private) | all seven |

**Nothing above may be overwritten. New outputs take new names.** The raw activations for
these arms are gone by design — the geometry reads the 1.5 GB qcache, not the 24 GB tensors —
so re-analysis at layers other than 15/20 would need re-extraction.

## 2. Environment — the traps that cost hours

If the pod is **Python 3.12**, both library trees are already built and correct on the volume;
just verify. A different Python means `$PYLIBS` is version-scoped and empty, and every trap
below applies again.

- **`pip install -r openrlhf/requirements.txt` installs torch 2.14+cu130 into `PYLIBS_TRAIN`**,
  which shadows the system torch and re-creates the `torchvision::nms` mismatch. It surfaces
  as `import peft` failing, not as anything torch-shaped. Fix: move `torch`, `triton` and the
  `nvidia-*` wheels out of `PYLIBS_TRAIN` and use the **system torch 2.8.0+cu128**, which has
  the matching torchvision/torchaudio/triton and native `sm_120`. Verify with a real CUDA
  matmul, never `torch.cuda.is_available()`.
- **Measurement must be transformers 4.57.6**, not whatever pip resolves (it picks 5.x).
  Every published arm was measured on 4.57.6. `/workspace/requirements-snapshot.txt` is stale
  and says 5.14.1 — do not follow it.
- **`matplotlib` in `$PYLIBS` was installed `--no-deps`** to keep a second torch out, so
  `cycler`/`fonttools`/`kiwisolver`/`contourpy` may be missing. Install them `--no-deps` too,
  and check torch did not move afterwards.
- **flash-attn is IN**, wheel already on the volume at
  `/workspace/tmp/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`.
  Not optional: `train_dpo.py` defaults `--attn_implementation` to `flash_attention_2` and no
  runner overrides it, so its absence is a deviation from the released pipeline.
- **`merge_lora.py` needs `peft`**, which is deliberately absent from the measurement env
  (installing it there can drag in a second torch). Run merges with
  `PYTHONPATH=/workspace/pylibs-train-py312`, extractions with `$PYLIBS`.
- **Runners are already corrected** and must stay that way: `--max_epochs 3` on introspection
  (the release-era value — HEAD's 1 is post-release), and **no `--use_wandb` line at all**
  (the string `"False"` is truthy, enables wandb, then fails as an API key).
- **`2c_caa_activations.py` derives `--output-dir` from the MODEL name.** Every composite
  state loads the same base model, so omitting it writes them all into the base arm's
  directory and destroys it. Always pass it explicitly.

## 3. Order of work

1. **Dose-match the existing stage states.** No retraining. Use the existing ladder
   machinery, not a parallel path:
   - `scripts/dose_calibrate.py` — cheap scale→dose probe, ~1 min per config, same dose
     statistics as `functional_dose.py`. Knows the stage states via `STAGE_STATES`.
   - `scripts/dose_calibrate_analyse.py` — reads dose off that grid.
   - `scripts/dose_match_plan.py` — turns the grid into rungs with genuine overlap; reports
     whether dose is actually linear in `s` (do not assume it), and refuses to extrapolate
     outside each state's measured range.
   - Then full 192-cell extraction at the chosen rungs only, and the usual
     `common_shift` / `functional_dose` / `caa_logits_analysis` path.
2. **Analyse and report** into `docs/runs/oct/DOSE_MATCHED_STAGE_REPORT.md`, with CSVs of
   state × nominal scale × measured dose × endpoint, and dose-response plots.
3. **Only then** decide whether the dense SFT checkpoint curve (spec §5) is worth running.
   It is not started, and it should not start before the dose question is settled.

**Compare at matched MEASURED dose — never at matched nominal scale, never at matched weight
norm.** `‖dW‖_F` is a diagnostic only; treating it as dose is the specific error §7.1 records.

**Disk:** activations are 24 GB per arm; the geometry only reads the 1.5 GB qcache. Extract →
build qcache → delete the raw tensors. That is a recorded decision, not a silent cleanup.

## 4. Watchdog bugs that have wasted GPU time here

- `pgrep -f <script>` **matches the waiting shell itself** — anchor the pattern
  (`'^bash /path/script\.sh$'`) or wait on a log marker instead.
- `cmd | tail -30` makes the pipeline's exit status `tail`'s, so a failed stage looks fine.
  Give every stage its own `|| die`.
- **Never edit a script while it is running.** Bash reads scripts incrementally; editing one
  mid-run shifts the byte offsets and the running instance parses garbage.
- Chain the next stage to the previous one's completion; a waiter that only *notifies* leaves
  the GPU idle until someone asks.

## 5. Interpretation discipline

- Do not say "SFT installs the character" on the strength of `M_S`'s raw B1 — that is the
  pre-dose-matching number, and it is exactly what this experiment is testing.
- Do not describe the cross terms as containing "X% of the behaviour"; behaviour is nonlinear.
  It is fine to say removing them changes B1 from X to Y **at a stated matched dose**.
- Do not change any preregistered threshold ([spec_sham_lora.md](spec_sham_lora.md) §5.1, §6b).
- Do not run sham arms.
- Do not edit an existing workshop claim to match a new number.

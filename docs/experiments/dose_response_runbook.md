# Dose-response runbook — next GPU session

**Goal.** Decide whether the geometry effects of OCT character training are a generic
function of functional perturbation size, or whether constitutions differ at matched dose.

**Why this and not the random LoRA.** The random-LoRA control was the plan until the fifth
arm landed. It answers "does *any* r=64 merge do this", which is worth knowing — but the
five-arm result reframed the question. Functional dose orders RDM preservation inversely and
near-perfectly across four constitutions, so the live hypothesis is a
perturbation-magnitude law, and the way to test that is to **vary dose within a
constitution** rather than add a fifth kind of perturbation. Random LoRA moves behind this.

**The decisive comparison.** Build, for each constitution, a curve

    outcome = f_c(functional dose)

If the four curves coincide, the generic-dose explanation becomes very strong. If they
separate at matched dose, something constitution- or adapter-specific is real. Crucially
this is measured at **matched dose**, which no comparison so far has been.

---

## Stage 0 — preflight (5 min)

```bash
ls /workspace/bootstrap.sh && cd /workspace/repos/steering_across_personas
git pull
source /workspace/bootstrap.sh && bash scripts/preflight.sh     # must print PREFLIGHT OK
```

Traps already handled in the launchers, listed so they are recognisable if run by hand:

- `python3` may not be the provisioned interpreter (last pod: 3.11 with a *system* torch,
  while `pylibs-py312` held a different one). Select by importing a CUDA-capable torch, never
  by looking for a `torch` directory.
- Never let preflight auto-install `peft` — it can pull a second torch into `PYLIBS` and
  break a working stack. It is optional now; install by hand with `--no-deps` if merging.
- Pass the **local snapshot path** for the base model, not the hub id: transformers 4.57
  404s on `additional_chat_templates` for Llama-3.1.

## Stage 1 — scaled merges (~40 min)

LoRA contribution scales linearly in the merge: `W = W_base + s·(alpha/r)·B·A`.
`scripts/merge_lora.py` needs an `--scale` flag — a one-line change at the merge step.

First-guess scales, from measured functional dose relative to `goodness`:

| arm | measured dose | first guess to reach goodness's dose |
|---|---|---|
| `impulsiveness` | 1.207x | s ≈ 0.83 |
| `misalignment` | 1.370x | s ≈ 0.73 |
| `mathematical` | 0.942x | s ≈ 1.06 |

**These are guesses, not targets** — functional dose is not guaranteed linear in `s`.

## Stage 2 — cheap calibration BEFORE committing (~20 min)

For each candidate scale, extract the small diagnostic grid (2 traits x 3 personas x 2
directions, 150 questions — ~2 min per arm on a 3090), run `scripts/functional_dose.py`, and
see where dose actually landed. Adjust `s` and repeat. Only then commit to full extraction.

This stage exists because a full extraction is ~1.5 h per arm-scale; calibrating by
guesswork would waste most of the day.

## Stage 3 — the grid (~12–18 h)

Target **2–3 strengths per constitution**, not one matched point. A single point tests one
hypothesis; a curve tests the shape for barely more cost per unit of information.

Choose scales so the four arms' dose ranges **overlap** — overlap is the whole point, since
without it there is no matched-dose comparison to make. Roughly 0.6x–1.3x of each arm's
natural dose.

Per arm-scale: 12 personas x 8 traits x 2 directions x 500 questions = 192 cells, ~1.5 h on a
3090 (measured: 1h39m for the misalignment arm). So 4 constitutions x 3 scales ≈ 18 h; 2
scales ≈ 12 h. **Do not drop to one scale per constitution** — that reverts to the comparison
already made.

Extract with `--legacy-mask`, matching the archive, for the reason in §2 of the results doc.

## Stage 4 — analysis (~30 min CPU)

```bash
python scripts/build_question_cache.py --arms <new arms> --layers 15 20
python scripts/functional_dose.py --layer 15
python scripts/geometry_analysis.py --layer 15 --bootstrap 200 --half-splits 40 \
       --boot-splits 40 --procrustes-rank 40
```

At each matched dose read off **three separate outcomes** — they came apart in the current
data, so there is no reason to expect them to move together:

1. **RDM preservation** — the statistic dose currently orders near-perfectly
2. **linear persona contraction** (RMS ratio) — the statistic dose does *not* explain
3. **residual-to-null ordering** — the §5.3 scalar effect, where `impulsiveness` collapses

## What each outcome would mean

| result | reading |
|---|---|
| all four curves coincide on all three outcomes | generic perturbation-magnitude law; constitution content does no work |
| RDM curves coincide, contraction curves separate | the current split is real — RDM disturbance is generic dose, contraction is arm-specific |
| `impulsiveness` keeps its residual-to-null collapse at matched dose | the one content-specific effect survives and becomes the headline |
| curves separate everywhere | dose was a confound but not the explanation; constitution content back in play |

Rows 2 and 3 are where the current data points. Row 1 is a clean null and equally informative.

## Afterwards

The **matched random rank-64 LoRA** still has a job: distinguishing "generic to any
perturbation" from "generic to OCT character training". Run it after the dose curves, at a
dose in the middle of the observed range, matching **per-module** norms rather than only the
global norm.

---

## Off the GPU critical path

- Confirm the `goodness`-vs-others dispersion ordering against **fixed-mask** activations —
  dispersion is the one statistic the mask fix could move (§2).
- Read the `mathematical` constitution text directly. The "semantically different" claim in
  §5.1 rests on it and is currently second-hand.

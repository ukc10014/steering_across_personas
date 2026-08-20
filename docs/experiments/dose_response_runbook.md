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

## Stage 1 — scaled weights (no merge step; ~0 min) ✅ done 2026-08-20

**Superseded: there is nothing to merge.** All four arms are plain LoRA — r=64, alpha=64,
no rsLoRA, no DoRA, no `modules_to_save`, same seven projection targets — so the merge is

    W(s) = W_base + s * (alpha/r) * B @ A

and `scripts/dose_calibrate.py` applies it **in memory** after loading the base model
(4 s for 224 modules). `--verify-scale1` checks the s=1 patch against the archived merged
checkpoints and finds them **bit-identical** (`rel||pred − merged|| = 0.000e+00` on every
module tested, both `goodness` and `misalignment`).

So the plan's 40 minutes of merging and 16 GB of disk per candidate scale are not needed,
and neither is `peft`. This also removes the trap the runbook warned about — passing the
hub id rather than the local snapshot path — since only the base model is ever loaded.

The first-guess scales it was going to use, from measured functional dose relative to
`goodness`:

| arm | measured trait-vector dose | first guess |
|---|---|---|
| `impulsiveness` | 1.207x | s ≈ 0.83 |
| `misalignment` | 1.370x | s ≈ 0.73 |
| `mathematical` | 0.942x | s ≈ 1.06 |

**These are guesses, not targets** — functional dose is not guaranteed linear in `s`, which
is what Stage 2 exists to find out.

## Stage 2 — cheap calibration ✅ done 2026-08-20

Full results: [dose_calibration_results.md](dose_calibration_results.md). In short:

- **Both first-guess scales are rejected.** `impulsiveness` s=0.83 lands at 1.229x
  `goodness` on trait-vector dose and 0.944x on answer-token; `misalignment` s=0.73 lands at
  1.228x and 0.893x. They miss in *opposite directions on the two measures*.
- **Dose is sublinear in `s`** (`dose ∝ s^0.5–0.85`), so cutting `s` by 27% cut dose by 9%.
  The guesses assumed linearity.
- **The two dose measures cannot both be matched**: matching answer-token needs s ≈ 0.9,
  matching trait-vector needs s ≈ 0.5–0.65. A matched-point design would have to pick one,
  and that choice would do part of the inferential work.
- **This diagnostic grid over-states `impulsiveness` by 13%** on trait-vector dose, because
  3 personas x 2 traits is a biased subset and one of the traits is `impulsivity`. Use
  answer-token dose on small grids; it transfers within 2%.

So Stage 3 changes shape.

## Stage 3 — the ladder (~9 h), replacing the matched-dose grid

The four arms at s=1 already sit within **8%** of each other on answer-token dose, while
their RDM preservation spans 0.883–0.732. Scaling `s` moves dose over a **2.4x** range —
thirty times wider than the differences the dose hypothesis has to explain. So vary `s`
widely within each arm and read the outcome against **measured** dose, rather than trying to
land on a point.

| arm | new scales | answer-token dose covered (with archived s=1) |
|---|---|---|
| `goodness` | 0.25, 0.5 | 0.44x – 1.00x |
| `impulsiveness` | 0.25, 0.5 | 0.49x – 1.05x |
| `misalignment` | 0.25, 0.5 | 0.48x – 1.04x |

**6 new extractions, ~1.5 h each.** `mathematical` stays at s=1 — within 6% of `goodness` on
both measures, so its ladder would duplicate `goodness`'s.

The ranges overlap almost completely, so matched-dose comparisons come out by interpolation
and no calibration accuracy is needed: dose is measured on the full extraction itself.

Extract with `--legacy-mask`, matching the archive, for the reason in §2 of the results doc.
Weights are patched in memory by `scripts/dose_calibrate.py`'s `apply_scaled_lora`; the
full-grid extractor needs the same hook.

## Stage 4 — analysis (~30 min CPU)

```bash
python scripts/build_question_cache.py --arms <new arms> --layers 15 20
python scripts/functional_dose.py --layer 15
python scripts/geometry_analysis.py --layer 15 --bootstrap 200 --half-splits 40 \
       --boot-splits 40 --procrustes-rank 40
```

Plot each outcome against **measured** dose (both measures — they disagree, see Stage 2) and
read matched-dose comparisons off the overlap. Three separate outcomes, which came apart in
the current data, so there is no reason to expect them to move together:

1. **RDM preservation** — the statistic dose currently orders near-perfectly
2. **linear persona contraction** (RMS ratio) — the statistic dose does *not* explain
3. **residual-to-null ordering** — the §5.3 scalar effect, where `impulsiveness` collapses

## What each outcome would mean

| result | reading |
|---|---|
| the outcome barely moves across each arm's 2.4x dose range | dose cannot explain a spread produced within 8% of dose — the ρ = −1 ordering was coincidence among four points. **The current data cannot rule this out, and a matched-point design would never have tested it.** |
| all curves coincide on all three outcomes | generic perturbation-magnitude law; constitution content does no work |
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

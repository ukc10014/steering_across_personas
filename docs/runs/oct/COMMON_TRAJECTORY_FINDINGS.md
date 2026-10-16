# Constitution-dependent acquisition dynamics — findings (impulsiveness, loving, goodness, sycophancy)

**2026-10-15 · branch `exp/common-trajectory` · spec [docs/spec_common_trajectory.md](../../spec_common_trajectory.md)
incl. amendment A1 (comparative panel).**

Status of the panel:

| constitution | status |
|---|---|
| impulsiveness | done |
| loving | done (reproduction gate **BETWEEN**, scoped: [GATE_REPORT_traj-loving.md](GATE_REPORT_traj-loving.md)) |
| goodness | done 2026-10-15 23:00Z, ladder done |
| sycophancy | done 2026-10-16 04:47Z, ladder done 09:32Z |
| misalignment | only if the original gated data are granted |

This report is updated as the panel completes. All results are one training seed each; CIs are
question-bootstrap (evaluation-only); geometry at L15 with L20 as a robustness check.
Everything about loving refers to **this reproduction**, not necessarily the released loving
model.

![panel](../../../outputs/analysis/common_trajectory/fig_trajectory.png)

## Verified results

### 1. Both constitutions acquire their SFT-induced behavioural change early

The primary outcome is the full eight-trait profile. q(e) is the fraction of the final
(3-epoch) SFT-induced profile change that is present at epoch e, measured along its final
direction.

| | ‖Δprofile‖ DPO → 3 ep | q at 0.24 ep | q at 0.48 ep | q at 1.0 ep |
|---|---|---|---|---|
| impulsiveness | 3.17 [3.08, 3.27] | 0.59 | **0.76** [0.76, 0.77] | 0.90 |
| loving | 1.33 [1.25, 1.43] | 0.66 | **0.85** [0.79, 0.90] | 0.99 |

### 2. What SFT changes differs sharply between the two

- **Impulsiveness: SFT carries the target.** B2 goes from +0.30 at DPO-only to +2.09 at 3 epochs.
- **Loving: the DPO stage already carries the target.** LOV is +0.82 at DPO-only and +0.88 at
  3 epochs, a paired change of +0.06 [−0.00, +0.14]. Loving's SFT change is off-target:
  - assertiveness −0.97;
  - confidence +0.66;
  - impulsivity +0.44;
  - risk-taking +0.36.
- **The loving dip is descriptive only.** LOV dips to +0.55 at 0.48 epochs and recovers by 1
  epoch. See the gate note for why it is not a claim.

### 3. Persona-common geometry develops more slowly than behaviour, but the meaning differs

Read against epoch:
- The common-shift **direction** arrives early in both constitutions. The cosine to the final
  common shift at 0.48 epochs is 0.95 for both.
- Its **magnitude and projection** lag behaviour:
  - impulsiveness: geometry Q 0.64 and magnitude fraction 0.56, against behaviour q 0.76;
  - loving: Q 0.72 against q 0.85.

Read against functional dose, with the DPO adapter scaled to the same measured displacement as
the reference (`control_dose_<c>.json`, `fig_control_dose_<c>.png`):

| SFT trajectory − scaled DPO, matched dose, endpoint | impulsiveness | loving |
|---|---|---|
| persona-common share S, 8-trait mean, L15 / L20 | **−0.003 / +0.018** | **+0.090 / +0.102** |
| common magnitude g, 8-trait mean, L15 | +0.007 | +0.051 |
| target contrast (B2 / LOV) | **+1.47** | −0.01 |

- **Impulsiveness.** The persona-common geometry is what DPO alone gives at the same dose.
  The slower growth of the common shift is the model continuing to be displaced after the
  phenotype has saturated. SFT's distinctive contribution is behavioural (B2 +1.47 at matched
  dose) and its redistribution of the common shift towards the target traits
  (prelim report: g +0.14 at L15, +0.34 at L20).
- **Loving.** SFT adds nothing to the target contrast beyond scaled DPO. It does make the
  change **more persona-common** than DPO at matched dose: S excess +0.09 at L15 and +0.10 at
  L20. That excess is already about 80 % present at the first SFT checkpoint: +0.072 at
  0.24 epochs and +0.087 at 1 epoch, while dose there is *lower* than the DPO-only state.
  Its matched-dose behavioural contribution is the off-target profile:
  - assertiveness −1.12;
  - confidence +0.57;
  - impulsivity +0.49;
  - risk-taking +0.30;
  - honesty −0.28.
- **On how solid the loving excess is.** The gap has no paired interval, because trajectory
  and ladder are separate bootstrap runs. Each S estimate's question-sampling width is about
  ±0.01–0.02. The excess is 4–9× that, consistent across all five checkpoints and both layers.

### 4. DPO installs a largely persona-common change in both

| | impulsiveness | loving |
|---|---|---|
| S at DPO-only, 8-trait mean | 0.657 [0.638, 0.669] | 0.616 [0.597, 0.627] |
| incremental SFT share S^S, 3 ep | 0.778 | 0.680 |

SFT's own increment is at least as persona-common as DPO's change in both. Neither adds a
predominantly persona-specific component.


### 5. Endpoint reproduction fidelity (amendment A1 rule, `fidelity.py`)

| | A1 config | A2 ‖dW‖ ratio (weight cos) | A4 dose ratio L15 / L20 | B4 common-shift cos L15 / L20 | profile r | profile norm ratio | \|Δk\| | **A1 category** |
|---|---|---|---|---|---|---|---|---|
| impulsiveness | match | 0.92 (0.46) | 0.97 / 0.96 | 0.941 / 0.950 | 0.98 | 0.94 | 0.043 | **strong** |
| loving | match | 0.91 (0.44) | 0.89 / 0.90 | 0.920 / 0.874 | 0.91 | **0.72** | 0.009 | **strong** (borderline) |

**These two verdicts must be read together for loving.** The A1 profile rule calls loving
*strong*, but only just: the norm ratio is 0.72 against a 0.70 cutoff. Its primary §6 gate on
the LOV contrast remains **BETWEEN**, because the target traits reproduce at about 0.6× while
the off-target traits reproduce well. Neither verdict is changed.

The comparison exposes a property of the A1 rule worth stating. A profile-level criterion
(correlation plus overall norm) is insensitive to a *target-specific* shortfall when the rest of
the profile reproduces. For the panel, both the A1 category and, where a contrast is
prespecified, the contrast-level gap will be reported. The rule itself is not changed after
the fact.

The weight-space cosine (about 0.45 for both) is the same for the strong impulsiveness
reproduction and for loving, so it says nothing about phenotype fidelity, as the earlier
seed work found.


## Update 2026-10-16: goodness and sycophancy (amendment A1 panel)

Both use the loving procedure unchanged, from pinned released data. Provenance now records each
constitution's own data hashes (the `--data` fix). No contrast or target pair is defined for
either; the eight-trait profile is the outcome.

### Panel summary (L15; full CIs in `table_trajectory.csv`)

| | ‖Δprofile‖ DPO → 3 ep | profile q at 0.5 ep | geometry Q at 0.5 ep | S: DPO → 0.5 → 1 → 3 ep | S^S at 3 ep | cos(dG, final) at 0.5 ep | dose: DPO → 3 ep |
|---|---|---|---|---|---|---|---|
| impulsiveness | **3.17** | 0.76 | 0.64 | 0.66 → 0.72 → 0.74 → 0.78 | 0.78 | 0.95 | 0.53 → 0.90 |
| loving | 1.33 | 0.85 | 0.72 | 0.62 → 0.69 → 0.73 → 0.73 | 0.68 | 0.95 | 0.65 → 0.71 |
| goodness | **0.80** | 0.83 | 0.73 | 0.57 → 0.68 → 0.71 → 0.69 | 0.69 | 0.93 | 0.52 → 0.63 |
| sycophancy | **2.27** | 0.70 | **0.77** | 0.58 → 0.65 → 0.64 → 0.67 | 0.72 | 0.94 | 0.46 → 0.67 |

These are descriptive, with one seed each.

- **Timing is similar across all four.** 70–85 % of the SFT-induced profile change is present by
  half an epoch, and the common-shift direction is ≥ 0.93 aligned with its final direction by
  then.
- **How much SFT moves the profile differs about four-fold.**
  - Impulsiveness and sycophancy: SFT moves it a lot (‖Δ‖ 3.17, 2.27). Sycophancy's change is
    confidence +1.14, impulsivity +1.26, risk-taking +1.11, warmth +0.55, empathy +0.38,
    honesty −0.42 and assertiveness −0.64.
  - Loving and goodness: much of the profile is already present after DPO (‖Δ‖ 1.33, 0.80).
    Goodness's SFT change is confidence +0.54, deference −0.35, honesty +0.27, and
    empathy/warmth −0.21.
- **Behaviour vs geometry at half an epoch.** In three constitutions behaviour leads geometry
  (q > Q). In sycophancy, geometry is slightly ahead (Q 0.77 vs q 0.70).
- **DPO-only states are all substantially persona-common** (S 0.57–0.66). SFT increments are at
  least as common (S^S 0.68–0.78) in every case. For sycophancy the increment is *more* common
  than the total.
- **Matched-dose comparisons for all four are in the next section.**


### Dose control, all four (2026-10-16)

Each constitution's own DPO adapter was scaled to bracket its trajectory's measured dose, using
the A1 rule: goodness 0.9–1.45, sycophancy 0.95–1.85. The ladders cover 0.47–0.70 and
0.45–0.67 respectively, bracketing the trajectories. The table gives SFT trajectory minus
scaled DPO at matched dose, at the 3-epoch endpoint (`control_dose_<c>.json`).

| | where the profile is carried (‖Δ‖ DPO → 3 ep) | Δ persona-common share S, L15 / L20 | Δ common magnitude g, L15 | largest behavioural changes beyond scaled DPO |
|---|---|---|---|---|
| impulsiveness | SFT (3.17) | **−0.003 / +0.018** | +0.007 | impulsivity +1.93, risk-taking +1.09, confidence +0.62, assertiveness −0.50 |
| sycophancy | SFT (2.27) | **+0.033 / +0.067** | +0.036 | impulsivity +1.08, assertiveness −0.96, confidence +0.92, risk-taking +0.89, honesty −0.47 |
| loving | DPO (1.33) | **+0.090 / +0.102** | +0.051 | assertiveness −1.12, confidence +0.57, impulsivity +0.49, risk-taking +0.30 |
| goodness | DPO (0.80) | **+0.095 / +0.069** | +0.046 | confidence +0.52, deference −0.35, empathy −0.30, warmth −0.30 |

The pattern is descriptive: one seed each, four points, unpaired gaps. The per-state
question-sampling width of S is about ±0.01–0.02.

- **Every constitution's SFT changes behaviour beyond what scaled DPO gives at the same dose.**
  For the two whose profile SFT carries, the changes are large and centred on impulsivity,
  risk-taking and confidence. The DPO-carried two show off-target shifts.
- **Extra persona-generality from SFT is graded.**
  - Absent for impulsiveness.
  - Small for sycophancy: marginal at L15, clearer at L20.
  - Clearly present for loving and goodness, at about +0.07 to +0.10 at both layers.

  In this panel it is larger where DPO already carries most of the character and SFT's change
  is mostly off-target. This is an ordering across four constitutions, not a mechanism.


### Bounded empathy/warmth check (2026-10-16, CPU only, user-approved)

**1. At which stage is the shortfall present?** Offsets vs base; released = released merged adapter.

| | trait | our DPO-only | our seq 3 ep | our merge | released | DPO-only − released | merge − released |
|---|---|---|---|---|---|---|---|
| impulsiveness | empathy | +0.47 | +0.65 | +0.73 | +0.85 | −0.39 | −0.12 |
| | warmth | +0.50 | +0.75 | +0.79 | +0.96 | −0.46 | −0.17 |
| loving | empathy | +1.32 | +1.50 | +1.50 | +2.44 | **−1.12** | −0.94 |
| | warmth | +1.16 | +1.22 | +1.21 | +2.00 | **−0.85** | −0.79 |
| goodness | empathy | +0.30 | +0.09 | +0.14 | +0.32 | −0.02 | −0.18 |
| | warmth | +0.35 | +0.14 | +0.20 | +0.31 | +0.04 | −0.11 |
| sycophancy | empathy | +0.33 | +0.71 | +0.56 | +1.06 | **−0.73** | −0.50 |
| | warmth | +0.53 | +1.09 | +0.90 | +1.33 | **−0.80** | −0.44 |

Findings:
- For loving and sycophancy, the two large gaps, the shortfall is **already present after
  DPO**. SFT closes part of it for sycophancy (and impulsiveness), not for loving.
- For goodness, DPO-only matches the released values, and the small gap arises in SFT.
- Caveat: the released adapter is a merge of both stages, and its stage adapters are not
  public, so this locates the gap in *our* pipeline's stages, not in the authors'.

**2. Are our DPO inputs treated differently for warm characters?** No.

| | rows | system message | chosen / rejected > 1024 tokens | median chosen / rejected response tokens | median prompt tokens |
|---|---|---|---|---|---|
| impulsiveness | 8,137 | none | 0 / 0 | 250 / 524 | 52 |
| loving | 8,319 | none | 0 / 0 | 238 / 523 | 51 |
| goodness | 9,329 | none | 0 / 0 | 275 / 542 | 53 |
| sycophancy | 8,450 | none | 0 / 0 | 227 / 516 | 52 |

There is no truncation at `max_len 1024`, the chat format is identical, and lengths are
comparable across constitutions. Our DPO and SFT runners differ from the upstream `llama.sh`
runners **only** in wandb logging; all hyperparameters are identical.

**Conclusion.** No implementation or input-handling cause was found. The discrepancy sits
mainly in the DPO stage for the two warmth-heavy characters. Remaining candidates, none
testable from here:
- the public data are not byte-identical to what trained the September adapters (the public
  files were uploaded 2025-10-24);
- unrecorded training settings or environment;
- nondeterminism with a trait-specific effect.

The bounded check stops here. **The next step is a question to the OCT authors**, not further
computation.

### Endpoint fidelity, all four (A1 rule)

| | A2 ‖dW‖ | A4 dose L15 | B4 cos L15 / L20 | profile r | norm ratio | \|Δk\| | A1 category |
|---|---|---|---|---|---|---|---|
| impulsiveness | 0.92 | 0.97 | 0.941 / 0.950 | 0.98 | 0.94 | 0.043 | strong |
| loving | 0.91 | 0.89 | 0.920 / 0.874 | 0.91 | 0.72 | 0.009 | strong (borderline; LOV gate BETWEEN) |
| goodness | 0.95 | 0.88 | 0.904 / 0.856 | 0.97 | 0.90 | 0.072 | strong |
| sycophancy | 0.89 | 0.84 | 0.895 / 0.882 | 0.94 | 0.91 | 0.050 | strong |

### A shared, directional discrepancy: empathy and warmth under-reproduce in every constitution

Reproduced minus released offset, at the endpoint (`fidelity_<c>.json`):

| | empathy | warmth | other six, mean (signed) | other six, mean \|gap\| |
|---|---|---|---|---|
| impulsiveness | −0.12 | −0.17 | −0.08 | 0.22 |
| loving | **−0.94** | **−0.79** | +0.07 | 0.20 |
| goodness | −0.18 | −0.11 | −0.01 | 0.07 |
| sycophancy | **−0.50** | **−0.44** | −0.05 | 0.11 |

- **Verified:** all 8 empathy/warmth gaps are negative. The other six traits show no directional
  bias. The gap is largest where the released adapter raises empathy and warmth most (loving,
  then sycophancy).
- **Not verified:** that this reflects a shared difference between our pipeline and the
  authors'. Candidates include the data revision, as above. Only loving's gap has a paired CI;
  the others are point estimates.
- **It does not invalidate the trajectories.** Each is internally consistent, and the profile
  rule still rates every reproduction strong. It **does** limit how far any trajectory speaks
  for the released model on empathy and warmth specifically.
- This is the "shared pipeline problem" condition named in the loving stop rule, so it is
  flagged rather than left as a loving-only limitation.

**Proposed bounded follow-up (not started).** CPU only, no new training:
1. Compare each constitution's **DPO-only** empathy/warmth offsets with its released endpoint,
   to see whether the shortfall is already present at the DPO stage in all four, as it is for
   loving.
2. Diff our DPO training inputs as the trainer sees them (chat template, system prompt,
   truncation at max_len 1024) for warmth-heavy vs other rows.

If neither shows anything, ask the authors whether the public data match what trained the
September adapters.

## Answers to the brief's questions, so far

- **Do loving and impulsiveness acquire their behavioural phenotypes early?** Yes, on the
  common eight-trait profile measure: 76–85 % by half an epoch. For loving the *target*
  phenotype is already there after DPO, and SFT's early change is off-target.
- **Does the persona-common component emerge at the same time?**
  - Its direction does.
  - Its magnitude lags in epoch terms. For impulsiveness that lag is entirely a dose effect.
  - For loving, a dose-independent increase in persona-generality appears at the first SFT
    checkpoint and then changes little.
- **Does SFT amplify an existing DPO-common shift or introduce a new one?** In both, DPO already
  establishes a largely persona-common change, and its direction is close to the final one
  (cos 0.73 at DPO-only rising to 0.95 by 0.5 epochs).
  - Impulsiveness: SFT amplifies it, with the extra amplification going to the target traits.
  - Loving: SFT increases the common share beyond what dose explains, while its behavioural
    change is off-target.
- **Is the acquisition trajectory similar across constitutions?**
  - *Timing* is similar: early on both behaviour and geometry measures.
  - *Stage division* differs: SFT carries the target for impulsiveness and the off-target
    profile for loving.
  - *What SFT adds beyond dose* differs: behaviour for impulsiveness, persona-generality for
    loving.
- **Does any change in persona generality survive accounting for dose and noise?**
  - Impulsiveness: no.
  - Sycophancy: marginally (+0.03 at L15, +0.07 at L20).
  - Loving and goodness: yes, descriptively, about +0.07 to +0.10 in share at matched dose, at
    both layers. For loving it is present from the first checkpoint.
  - One seed each and unpaired gaps, so this is a description of these reproductions, not yet
    a general claim.

## Hypotheses (not tested here)

- Why loving's target phenotype sits in the DPO stage, and impulsiveness's in SFT. No semantic
  explanation is proposed (amendment A1). The SFT corpus is generated downstream of DPO, so
  the stage comparison locates *where in the weights* a change appears, not which stage
  created the underlying information.
- Whether the released loving model shares this stage division. Its stage adapters are not
  public, and this reproduction under-reproduces its target phenotype (gate BETWEEN).

## Limitations

- One seed per constitution.
- One base model (Llama-3.1-8B-Instruct).
- Two constitutions so far.
- Context dependence is measured through the existing CAA probes, not open-ended behaviour.
- The loving reproduction falls short of the released adapter (LOV +0.87 vs +1.81), cause
  unresolved.
- Checkpoint-to-checkpoint scatter exceeds the question-bootstrap width, so no claim rests on
  adjacent checkpoints.
- Trajectory-vs-ladder gaps are unpaired.

## Files

`outputs/analysis/common_trajectory/`:
- `behaviour_<c>.json`: eight offsets, contrasts, the target/other split, profile metrics,
  paired CIs.
- `geometry_<c>.json` and `geometry_boot_*`: g, S, energies, cos, S^S, Q; L15 and L20.
- `functional_dose_<c>{,_dpo_ladder}.json`.
- `common_shift_<c>_dpo_ladder.json`.
- `control_dose_<c>.json` and `fig_control_dose_<c>.png`.
- `fig_trajectory{,_dose}.png`, `table_trajectory.csv`, `trajectory_points.csv`.
- `gate_loving.json`.

Code: `scripts/common_trajectory/`.

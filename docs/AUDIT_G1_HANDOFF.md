# Audit repairs, G1 run — handoff

**2026-10-22 · branch `audit/g1-repairs` (off `audit/post-submission` @ `ffb7950`) · for the writing model.**
Builds on [AUDIT_POST_SUBMISSION_RESULTS.md](AUDIT_POST_SUBMISSION_RESULTS.md) and does not repeat it.
Full machine-generated side-by-side tables: [`outputs/analysis/audit/corrected_base_comparison.md`](../outputs/analysis/audit/corrected_base_comparison.md)
(+ `.json`). `arxiv/main.tex` was not edited.

Everything is one training seed per arm. Intervals are question-bootstrap (evaluation) only.

---

## 1. What was done, what was not, resources

| task | status |
|---|---|
| Preflight: worktree, audit guards and estimability fixes present, 20 audit tests pass | done |
| **G1 base**: corrected-mask extraction of unadapted Llama-3.1-8B-Instruct, cache + validation | **done** |
| G1 released impulsiveness / goodness / sycophancy | **not run**: ~80 min/arm on the RTX 3090 would have exceeded the 2 h GPU ceiling (budget guard declined them; §6) |
| Released loving | reused `xchar_lov_start` (folded weights, corrected mask; rounding-equivalence caveat of audit §2.3 stands) |
| Trajectory geometry, 4 constitutions × L15/L20, on the actual base | done |
| Functional doses (trajectory + ladders) on the actual base | done |
| Joint paired trajectory/ladder bootstrap (new), all 4 constitutions, B = 1000 | done |
| xchar activation selectivity on the actual base | done |
| Loving fidelity on one protocol (actual base, released = `xchar_lov_start`) | done |
| Impulsiveness/goodness/sycophancy fidelity (A4, B4) | **blocked**: needs released corrected-mask arms |
| OCT reproduction/stage/dose-matched/SFT-curve arms (51): dose + B3-style selectivity on the actual base | done (§5.6) |
| Figures: trajectory full + A/B, matched-dose, xchar (optimizer-step axis) | done |

**Compute.** One pod for everything: `2nr1xatxkdm2an` (renamed `kanad_finalising_arxiv` during the run), 1× RTX 3090
24 GB, 32 vCPU, **$0.50/h** (runpodctl). The pod had been up since 2026-10-19; this session started ~16:18Z.
- GPU phase 16:18–17:46Z (≈1.5 h ≈ $0.75): preflight + one arm. Measured 24.3–26 s/cell on the 3090 against
  ~5.4 s/cell on the RTX PRO 6000 that the audit's estimate assumed (≈4.5× slower): base extraction 4,672 s,
  cache + validation 114 s (`/workspace/oct_rig/logs/g1_timings.jsonl`).
- CPU phase 17:47–19:31Z on the same pod (user-approved; the GPU sat idle). Per-step timings in
  `/workspace/oct_rig/logs/cpu_timings.tsv`. Total session ≈ 3.3 h ≈ **$1.7**.
- The pod is terminated with `/workspace/oct_rig/terminate.sh` at the end of the run (user instruction). The
  network volume persists.

## 2. Artifacts, revisions, commands

**Model / adapters.** Base `meta-llama/Llama-3.1-8B-Instruct` snapshot `0e9e39f249a16976918f6564b8830bc894c89659`
(the same `--model` path as every corrected trajectory/ladder/xchar cache). Released adapters (not extracted this
run): `maius/llama-3.1-8b-it-personas` snapshot `318b5f7e1428097a1a61d5f0ed205ee048b3f620`.

**G1 base cache** — `outputs/_qcache/base_cmask_L15_20.npz` (1.57 GB, persistent volume; arm key `base_cmask`,
distinct from legacy `base` and the synthetic `base_synthcorrected`). Embedded provenance (directly observed, from
the extraction sidecar): mask `corrected`, batch 16, bf16, variant 0, all questions (empathy 499, others 500),
12 personas, `questions_sha 5b6ba516b118e9ca`, `personas_sha 582b5fb0eaf18d85`, extractor
`pipeline/2c_caa_activations.py` sha `d2a5d7d9be16767f` at commit `ffb7950`. Raw all-layer activations kept:
`outputs/llama-3.1-8b-base_cmask/caa_activations/` (192 × 131 MB + `_extraction.json`).
Validation (`outputs/analysis/audit/g1/validate_base_cmask.json`): layout identical to the corrected trajectory
caches, 0 non-finite values, provenance as expected; cell-mean cosine with the legacy base ≥ 0.997 (median
0.9985), median relative difference 5.5 % (the same-model mask pair gives 0.998 / 4 %).

**Provenance change.** Registry-backed caches carried symbolic `archive-500q/v1` / `archive-12p/v1` labels,
which no real hash can equal, so `base_cmask` was refused against every corrected arm. The registry now has an
`archive_identity` block mapping them to the G1 hashes, **labelled inferred**: `data/prompts/caa` last changed
2026-03-10 and the 12 persona YAMLs ≤ 2026-04-19, all before the earliest registered extraction (2026-07-18).
Different hashes are still refused, and the mask check is unchanged (test added).

**Code (branch `audit/g1-repairs`).** `scripts/audit/run_g1_extract.sh` (runner with budget guard),
`validate_g1_cache.py`, `joint_dose_bootstrap.py` (+ `tests/test_joint_dose_bootstrap.py`, 11 tests),
`run_cpu_stage.sh`, `compare_corrected_base.py`, `fig_matched_dose.py`; `--base-arm`/provenance check in
`common_shift.py`; tagged inputs in `common_trajectory/figures.py`; `--plot-only` + new axis in
`xchar_analysis.py`; optional external draw in `geometry_lib.bootstrap_half_splits` (default RNG stream
unchanged, tested). `python -m pytest tests/` → 31 pass.

**Reproduce.**
```bash
MASK=corrected DEADLINE=<epoch> bash scripts/audit/run_g1_extract.sh base_cmask   # GPU, ~80 min on a 3090
bash scripts/audit/run_cpu_stage.sh                                               # CPU, resumable
python3 scripts/common_trajectory/figures.py --geometry-tag _cmbase --behaviour-tag _10personas --dose-tag _cmbase --out-tag _cmbase [--panels AB]
python3 scripts/audit/fig_matched_dose.py --tag _cmbase
python3 scripts/xchar_analysis.py --plot-only
python3 scripts/audit/compare_corrected_base.py
```
(The run itself split lane 2's tail into two parallel lanes for wall-clock: scripts kept at
`/workspace/oct_rig/logs/lane{3,4}.sh`; same commands.)

**Outputs** (all `_cmbase` = actual corrected base):
- `outputs/analysis/common_trajectory/v2/geometry_<c>_cmbase.json`, `v2/geometry_raw_<c>_L{15,20}_cmbase.npz`
- `outputs/analysis/common_trajectory/functional_dose_<c>{,_dpo_ladder}_cmbase.json`
- `outputs/analysis/audit/joint_dose/joint_dose_<c>_cmbase.json`, `joint_raw_<c>_L{15,20}_cmbase.npz`,
  `matched_dose_points_cmbase.csv`, `fig_matched_dose_cmbase.png`
- `outputs/analysis/common_trajectory/fig_trajectory_cmbase.png`, `fig_trajectory_AB_cmbase.png`,
  `fig_trajectory_dose_cmbase.png`, `table_trajectory_cmbase.csv`, `trajectory_points_cmbase.csv`
- `outputs/analysis/xchar_sft/common_shift_cmbase.{json,txt,provenance.json}`, `selectivity_<arm>_{imp,lov}_cmbase.json`,
  `acquisition_retention_steps.png` (data unchanged; original `acquisition_retention.png` kept)
- `outputs/analysis/common_trajectory/v2/fidelity_loving_cmbase.json`
- `outputs/analysis/audit/oct_stage_cmbase/` (OCT arms: doses vs both bases, common shift parts 0–3)
- `outputs/analysis/audit/corrected_base_comparison.{md,json}`

Historical, mixed-protocol and synthetic-base files are untouched.

## 3. The joint matched-dose bootstrap (new)

One question draw per trait and replicate, shared by base, every trajectory state and every ladder rung, and by
dose, g/S (split into disjoint halves of that draw) and the forced-choice offsets (activation index i = CAA
question id i, asserted). Dose matching (sort ladder by this replicate's dose, linear interpolation) is redone in
every replicate. B = 1000. **Rule** (fixed before running): no extrapolation; a replicate is *unbracketed* if its
trajectory dose is outside its own ladder range, *invalid* if a ratio denominator is ≤ 0; an excess is reported
iff its point is defined and ≥ 97.5 % of replicates are defined (the audit's estimability convention), else point
and interval are withheld and the fractions reported. Verified: point doses reproduce `functional_dose_*.json` to
1e-7, behaviour excess points reproduce `control_dose_*.json` exactly (11 personas), geometry points agree to
within split noise (≤ 0.002).

Frequency of undefined comparisons (all series × states × layers): impulsiveness 0/510 withheld; loving 0/150;
goodness 6/140 withheld (all at L20 0.24 ep, 53 % unbracketed); sycophancy 32/140 withheld (L15 endpoint point
unbracketed: 23 series incl. all behaviour; L20 per-trait-mean S at every state: 95.1 % valid, deference).

Caveat: the functional dose is a plug-in statistic with noise bias; its bootstrap percentile interval can sit
beside the point (e.g. goodness L20 DPO 0.718, [0.668, 0.716]). Do not quote dose intervals. The paired excess is
unaffected in practice (1 of 902 estimable excess points lies marginally outside its interval: impulsiveness L15
g at 0.12 ep, 0.011 vs lo 0.011).

## 4. Old vs corrected headline numbers

### 4.1 Trajectory table (L15 MEAN8; `table_trajectory.csv` → `table_trajectory_cmbase.csv`)

Base-dependent (S, cosT, dose) changes are the mask. **Profile q, change norm and contrast change because
behaviour now uses the geometry's 10 occupational personas (audit §7.4), not because of the mask.**

| | S at DPO | S at 3 ep | cosT 0.5 ep | dose DPO → 3 ep | q(0.48 ep) 11p → 10p | q(1 ep) |
|---|---|---|---|---|---|---|
| impulsiveness | 0.657 → **0.647** [0.629, 0.659] | 0.777 → 0.773 | 0.954 → 0.956 | 0.531→0.903 ⇒ 0.526→0.910 | 0.76 → **0.80** | 0.90 → 0.91 |
| loving | 0.616 → **0.598** [0.578, 0.610] | 0.726 → 0.724 | 0.948 → 0.948 | 0.650→0.712 ⇒ 0.645→0.711 | 0.85 → **0.94** | 0.99 → 1.01 |
| goodness | 0.567 → **0.544** [0.513, 0.559] | 0.693 → 0.688 | 0.933 → 0.934 | 0.515→0.627 ⇒ 0.507→0.630 | 0.83 → 0.84 | 1.00 → 1.01 |
| sycophancy | 0.584 → **0.568** [0.525, 0.582] | 0.666 → 0.662 | 0.944 → 0.945 | 0.464→0.667 ⇒ 0.454→0.670 | 0.70 → 0.71 | 0.80 → 0.82 |

Contrast DPO→3 ep: impulsiveness B2 +0.30→+2.09 (11p) ⇒ +0.32→+2.01 (10p); loving LOV +0.82→+0.88 ⇒ +0.82→+0.78.
Profile change norm DPO→3 ep: impulsiveness 3.17 ⇒ 2.99, goodness 0.80 ⇒ 0.85 (10p).
Incremental SFT quantities (S^S, cosS, Q) are base-independent and unchanged (S^S 0.68–0.78).

**Mask sensitivity, actual vs the audit's synthetic estimate.** The audit put every trajectory statistic within
0.015 of the mixed-protocol value. The actual base moves them more, at the DPO-only states:

| max abs(actual − mixed), MEAN8 | g L15/L20 | S L15/L20 | cosT L15/L20 |
|---|---|---|---|
| impulsiveness | 0.007 / 0.018 | 0.010 / 0.009 | 0.012 / 0.008 |
| loving | 0.013 / **0.026** | 0.018 / **0.024** | 0.001 / 0.002 |
| goodness, sycophancy | see `corrected_base_comparison.md` §1 (largest at the DPO state; S DPO −0.023 / −0.016 at L15) | | |

Doses move by up to 2.7 % on trajectory states and 3.0 % on ladder rungs, mostly downward (impulsiveness reaches +0.8 %) and largest for the lowest-dose arms (DPO states, lowest rungs, L20) — as expected for an additive artifact (audit: ≤ 0.5 %). All L15 table points stay estimable; L20 non-estimability is as the audit
found (impulsiveness cosS/Q, sycophancy cosT/cosS/Q — the pooled estimand remains defined).

### 4.2 Matched-dose excess (trajectory − scaled DPO), endpoint (3 ep) unless stated

historical = `control_dose_<c>.json` (legacy base, unpaired point); now = joint paired bootstrap on the actual
base, point [95 % CI].

| | S per-trait mean L15 | S per-trait mean L20 | S energy-pooled L15 / L20 | g L15 | target contrast (11p) |
|---|---|---|---|---|---|
| impulsiveness | −0.003 → **−0.001 [−0.005, +0.005]** | +0.018 → **+0.022 [+0.008, +0.053]** | +0.007 [+0.002, +0.015] / +0.041 [+0.026, +0.082] | +0.007 → +0.010 [+0.004, +0.027] | B2 +1.47 → **+1.46 [+1.40, +1.53]** (10p +1.37 [+1.31, +1.44]) |
| loving | +0.090 → **+0.104 [+0.097, +0.115]** | +0.102 → **+0.126 [+0.110, +0.155]** | +0.108 / +0.138 | +0.051 → +0.059 [+0.054, +0.077] | LOV −0.01 → −0.01 [−0.08, +0.06] (10p −0.12 [−0.19, −0.06]) |
| goodness | +0.095 → **+0.107 [+0.098, +0.126]** | +0.069 → **+0.079 [+0.059, +0.120]** | +0.106 / +0.082 | +0.046 → +0.053 [+0.048, +0.080] | — |
| sycophancy | +0.033 → **not estimable** (endpoint unbracketed) | +0.067 → n/e by rule (95.1 % valid; point +0.084) | n/e / +0.081 [+0.047, +0.209] | +0.036 → n/e | — |
| sycophancy at 1 ep | +0.014 → +0.027 [+0.016, +0.086] | +0.036 → n/e (point +0.050) | +0.030 [+0.018, +0.072] / +0.045 [+0.018, +0.145] | +0.023 → +0.031 [+0.028, +0.094] | — |

- **Sycophancy endpoint.** On the actual base its L15 dose is 0.6700 against a top ladder rung of 0.6695
  (historically 0.6667 < 0.6694). 99.5 % of replicates are bracketed but the point is not, so by the rule every
  L15 endpoint excess for sycophancy — S, g and all per-trait behaviour excesses
  (FINDINGS: impulsivity +1.08, assertiveness −0.96, confidence +0.92, risk-taking +0.89, honesty −0.47) — is
  withheld. The 1-ep state is fully estimable (10p behaviour: confidence +0.83, impulsivity +0.87, risk-taking
  +0.78, assertiveness −0.74, honesty −0.43, all CIs excluding 0).
- **Goodness L20 0.24 ep**: point now inside the ladder (0.6546 vs min 0.6488), but 53 % of replicates are
  unbracketed → withheld. Status changes from "unbracketed" to "not estimable"; still no matched-dose number.
- **Estimands disagree for impulsiveness early at L15**: per-trait-mean S excess at 0.48 ep −0.007 [−0.010, −0.000]
  vs energy-pooled +0.011 [+0.002, +0.023]. Both are reported; neither is preferred.
- Endpoint per-trait behaviour excesses (10 personas, L15): impulsiveness impulsivity +1.79, risk +0.96,
  confidence +0.59, assertiveness −0.54; loving assertiveness −1.09, confidence +0.58, impulsivity +0.54,
  risk +0.34, honesty −0.29; goodness confidence +0.57, deference −0.34, empathy −0.30, warmth −0.29.
  The 11-persona points equal the historical ones (behaviour is mask-independent); the intervals are new.

### 4.3 Cross-character activation selectivity (L15; reported → synthetic → actual [95 % CI])

| state | imp+risk | emp+warmth |
|---|---|---|
| loving start | 0.91 → 0.94 → **0.93** [0.80, 1.06] | 1.20 → 1.16 → **1.18** [0.98, 1.37] |
| lov→imp | 1.33 → 1.38 → **1.34** [1.11, 1.57] | 0.95 → 0.92 → **0.94** [0.79, 1.10] |
| lov→lov | 0.90 → 0.93 → **0.92** [0.79, 1.06] | 1.20 → 1.15 → **1.17** [0.96, 1.38] |
| base→imp | 1.61 → 1.66 → **1.61** [1.32, 1.90] | 0.86 → 0.83 → **0.86** [0.72, 1.00] |
| base→lov | 0.81 → 0.84 → **0.83** [0.71, 0.95] | 1.22 → 1.18 → **1.21** [0.98, 1.44] |

L20 in `corrected_base_comparison.md` §4 (all within 0.08 of reported). Intervals are the delta-method ones of
`selectivity_score.py`, unpaired between states.

### 4.4 Loving fidelity (released = `xchar_lov_start`)

A4 dose ratio 0.892/0.902 (recorded, mixed) → **0.881/0.891**; B4 0.920/0.874 → **0.933/0.896**; category
**strong** throughout. Profile r 0.91 / norm ratio 0.72 unchanged (logits).

## 5. Claims: survive / change / unresolved

**Survive (now on one protocol, with the actual base):**
- Behaviour arrives early (q ≥ 0.71 by 0.48 ep on the 10 personas for all four); the persona-common direction
  arrives early (L15 cosT 0.89–0.92 at 0.24 ep); geometry magnitude/projection lags behaviour.
- DPO-only states are substantially persona-common and SFT increments at least as common.
- Impulsiveness: no extra persona-generality at L15 at matched dose (endpoint −0.001 [−0.005, +0.005]); SFT's
  matched-dose contribution is behavioural (B2 +1.46 [+1.40, +1.53]).
- Loving and goodness: SFT adds persona-common share beyond scaled DPO at matched dose, at both layers, now with
  paired intervals excluding 0 at every estimable state (loving L15 +0.087 to +0.104; goodness +0.107 to +0.136).
  This upgrades FINDINGS' "descriptive, unpaired" to paired evaluation intervals; still one seed.
- Loving's LOV contrast does not exceed scaled DPO at the endpoint (11p −0.01 [−0.08, +0.06]); on 10 personas it
  is slightly below (−0.12 [−0.19, −0.06]).
- xchar activation corroboration of replacement (selectivity moves ≤ 0.03 at L15 against reported).
- Loving fidelity "strong".

**Change:**
- Mask sensitivity is larger than the audit's synthetic estimate (up to 0.026 on g and 0.024 on S at loving L20
  DPO; doses up to 2.7 % on trajectory states, 3.0 % on ladder rungs), though no qualitative statement changes. Any text citing "≤ 0.015" / "≤ 0.5 %"
  must be replaced by the actual comparison.
- Matched-dose S excesses for loving/goodness/sycophancy are **larger** on the actual base (loving L20 endpoint
  +0.102 → +0.126; goodness L15 +0.095 → +0.107).
- DPO-state S values drop 0.010–0.023 (manuscript "S 0.57–0.66" → **0.54–0.65**).
- Sycophancy endpoint L15 matched-dose comparison is **no longer available** (unbracketed); quote 1 ep instead or
  drop. Sycophancy "marginal at L15" → at 1 ep +0.027 [+0.016, +0.086] (per-trait mean); L20 per-trait mean n/e.
- Goodness L20 0.24 ep stays without a matched-dose value (now "not estimable" rather than "unbracketed").

**Unresolved:**
- Fidelity A4/B4 for impulsiveness, goodness, sycophancy (released corrected-mask arms not extracted).
- Released-vs-reproduced comparisons involving released impulsiveness on one protocol (P0 A4 vs released; B3 of
  released) — still mixed-protocol, with the audit's sensitivity caveat.
- Seed variance everywhere.

### 5.6 OCT reproduction / stage / dose-matched / SFT-curve arms

51 corrected-mask impulsiveness arms (reproduction, seed 2, stage states, dose-matched M_D/M_F/M_S rungs, the
18-checkpoint SFT curve, P0, P1) re-referenced to the actual base (`outputs/analysis/audit/oct_stage_cmbase/`;
full table in `corrected_base_comparison.md` §6). B3-style selectivity (imp+risk vs other six, point):

| arm | L15 legacy → actual | L20 actual | dose L15 legacy → actual |
|---|---|---|---|
| reproduction | 1.556 → 1.545 | 1.755 | 0.818 → 0.824 |
| DPO-only (repro_dpo) | 1.106 → 1.132 | 1.056 | 0.531 → 0.526 |
| D + 0.25 S | 1.191 → 1.204 | 1.217 | 0.567 → 0.565 |
| D + S | 1.441 → 1.433 | 1.687 | 0.879 → 0.885 |
| SFT from base | 1.797 → 1.787 | 2.114 | 0.973 → 0.977 |
| curve 0.24 / 0.48 / 1.0 ep (seq) | 1.448 / 1.537 / 1.495 → 1.438 / 1.527 / 1.484 | 1.586 / 1.738 / 1.682 | 0.662 / 0.743 / 0.782 → 0.664 / 0.745 / 0.787 |
| curve 3 ep (seq) / merged final | 1.415 / 1.534 → 1.410 / 1.524 | 1.660 / 1.736 | 0.903 / 0.828 → 0.910 / 0.834 |
| P0 (regen) | 1.208 (reported B3) → 1.216 | 1.382 | 0.997 → 1.009 |
| P1 (paraphrase) | → 1.374 | 1.505 | 0.765 → 0.775 |
| seed 2 | → 1.627 | 1.814 | 0.796 → 0.800 |

L15 selectivity moves ≤ 0.026 and doses ≤ 1.3 % for every arm. Corrected-arm dose ratios: P0/repro
1.219 → 1.225, P1/repro 0.936 → 0.940, so "P0 is displaced more, not less" holds on one protocol. The SFT-curve
appendix figure (panels B–D: dose, selectivity, dose-matched B1 ratio) and the OCT stage/dose-matched tables move
by these amounts only; B1/B2 and the 82 % figure are logits and unaffected. Regenerating those appendix figures is
optional (the change is below plotting resolution); ratios to the **released** adapter (A4, released B3) remain
mixed-protocol until the released arms are extracted (§7). Point estimates only here; the per-trait bootstrap CIs
are in the `common_shift_oct_cmbase_part*.json` files (chunked 13 arms per call; per-arm quantities do not depend on
the chunking, cross-arm cosines are chunk-local).

## 6. Manuscript statements and figures to replace (`arxiv/main.tex`)

**Final figure exports (vector PDF + PNG + plotted data + captions + mapping): [`paper_g1/README.md`](../paper_g1/README.md)** (2026-10-23, branch `audit/g1-figures`). Where this table and that file differ on figure files, that file is current.

| line | current | replace with |
|---|---|---|
| 102 | [POD] mask-consistency TODO | done: trajectory/xchar TOTAL statistics recomputed against a corrected-mask base (G1); released impulsiveness/goodness/sycophancy remain legacy-only |
| 119 | "early goodness L20 checkpoint lies outside its DPO ladder range" | goodness L20 0.24 ep and sycophancy L15 3 ep have no matched-dose value (not estimable / unbracketed on the corrected base) |
| 225 (Fig. ext-trajectory) | `fig_trajectory.png` | `fig_trajectory_cmbase.png` (appendix, full) and `fig_trajectory_AB_cmbase.png` (main text A+B); caption: geometry vs corrected-mask base; behaviour on the 10 occupational personas |
| 232 | q wording | values on 10 personas: q(0.48 ep) 0.80 / 0.94 / 0.84 / 0.71 (imp/lov/good/syc) |
| 235 | matched-dose S excess list | per-trait-mean S excess, endpoint, L15/L20, paired 95 % CI: impulsiveness −0.001 [−0.005, +0.005] / +0.022 [+0.008, +0.053]; loving +0.104 [+0.097, +0.115] / +0.126 [+0.110, +0.155]; goodness +0.107 [+0.098, +0.126] / +0.079 [+0.059, +0.120]; sycophancy n/e at endpoint (1 ep: +0.027 [+0.016, +0.086] / n/e). B2 +1.46 [+1.40, +1.53] |
| 236 | "S 0.57–0.66" | "S 0.54–0.65" (S^S 0.68–0.78 unchanged) |
| 244 | "Provisional … depends on (i) mask (ii) estimator" | both resolved for the trajectory panel: one mask protocol and the estimability rule; the loving/goodness excess has paired intervals excluding 0 at every estimable state; one seed |
| 259 (Fig. ext-xchar) | token-budget axis | `xchar_sft/acquisition_retention_steps.png` (optimizer-step axis, 82/164/328 and 92/184/367 steps; distinguishable markers). Data unchanged |
| 1019, 1026 (Figs. ext-dose-imp/lov) | `fig_control_dose_<c>.png` (no paired CI) | `audit/joint_dose/fig_matched_dose_cmbase.png` (all four constitutions, paired CIs) or rebuild per constitution from `matched_dose_points_cmbase.csv` |
| FINDINGS l.81–83, 296, 320 | "no paired interval", "descriptive" | paired intervals now exist (§4.2) |
| 252 / xchar text | activation numbers | §4.3 actual values; drop "extraction-run noise" wording (it was the mask difference) |

Not affected: abstract/§3 legacy-base analyses (Fig. trait-selectivity, CTP residual, RDM/dose ladders, random
controls — internally consistent legacy-mask), every logit result, the behavioural PV eval, the 82 % B1 figure
(logits).

## 7. Additional compute proposal (not run)

1. **G1 remainder** — released impulsiveness, goodness, sycophancy, corrected mask, same runner
   (`run_g1_extract.sh impulsiveness_rel_cmask goodness_rel_cmask sycophancy_rel_cmask`). Purpose: fidelity
   A4/B4 for three constitutions on one protocol; released-vs-reproduction (B3, A4 for repro/P0/P1) on one
   protocol. Cost: ~4.1 h on an RTX 3090 (≈ $2.05 at $0.50/h) or ~1.2 h on an RTX PRO 6000 (~22 min/arm + setup).
   Then CPU ~20 min (fidelity + OCT comparison).
2. **One extra sycophancy ladder rung** (DPO adapter at s ≈ 2.0, logits + activations) to bracket the 3-ep
   endpoint again. ~1.5 h on a 3090 (logits + extraction) or ~35 min on a PRO 6000, then the joint bootstrap
   (~15 min CPU). Only needed if the sycophancy endpoint is to be quoted.
3. Optional items from the audit (G2 mathematical/misalignment, G3 option-swapped logits, J1) unchanged.

## 8. Plain-language summary

The audit found that the trajectory and cross-character activation results compared new activations (extracted
with a fixed attention mask) against an old baseline extracted with the buggy mask, and it could only estimate the
consequence with a synthetic baseline. This run re-extracted the real baseline with the fixed mask, checked it,
and recomputed everything that depends on it.

The conclusions hold. Every qualitative statement about the four-constitution trajectory, the cross-character
pilot and loving's reproduction fidelity is unchanged on a consistent protocol. The mask mattered somewhat more
than the audit's synthetic estimate (up to ~0.026 rather than ≤ 0.015 on geometry, up to ~3 % rather than
≤ 0.5 % on doses), mostly at the weakly displaced DPO-only states.

The new paired bootstrap turns the matched-dose comparison from a description into an interval: SFT's extra
persona-generality for loving and goodness is clearly above zero at every estimable checkpoint and both layers,
impulsiveness shows none at L15, and impulsiveness' extra behaviour (B2 +1.46) is tight. Two points lose their
matched-dose value under the no-extrapolation rule: the goodness L20 first checkpoint (as before) and, newly,
the sycophancy L15 endpoint, whose dose now sits a hair above the top of its ladder.

Still open: fidelity for impulsiveness, goodness and sycophancy on one protocol needs the three released
adapters extracted with the fixed mask (~4 h on this 3090, ~1.2 h on a PRO 6000), and nothing here addresses
training-seed variance.

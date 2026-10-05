# `loving` and `sycophancy`: preregistered cross-constitution selectivity — results

**Evaluation only; neither constitution was retrained.** Preregistration
[docs/prereg_loving_sycophancy.md](../../prereg_loving_sycophancy.md), committed and **pushed**
(`3eb19e3`) before any inference ran and before any result for these arms was inspected — the
arms had no directory, no qcache and no mention in any analysis JSON at that point.

**Nothing below was redefined after seeing a number.** Trait sets, contrasts, estimator and
bootstrap are as registered.

---

## 0. Headline: the two constitutions split, and one prediction fails outright

| | `loving` (empathy, warmth) | `sycophancy` (deference) |
|---|---|---|
| activation ratio, L15 | 1.182 [0.983, 1.381] — point > 1, **CI covers 1** | **0.877** [0.741, 1.014] — **NOT MET** |
| activation ratio, L20 | 1.102 [0.918, 1.287] — point > 1, **CI covers 1** | **0.796** [0.515, 1.077] — **NOT MET** |
| behavioural contrast, forced (**primary**) | **+1.807** [+1.704, +1.906] — **MET** | **−1.075** [−1.153, −0.999] — **NOT MET** |
| behavioural contrast, default | **+1.657** [+1.546, +1.769] — **MET** | **−0.259** [−0.314, −0.206] — **NOT MET** |

`loving` behaves as registered and clearly so: `empathy` (+2.44) and `warmth` (+2.00) are the
**top two of eight** corrected offsets, and both CIs sit far from zero.

`sycophancy` fails both predictions, and not marginally. `deference` is the **least-moved of
the eight traits** in activation space at both layers, and the **lowest of the eight** signed
offsets under the forced prompt (−0.149). The contrast's CI excludes zero on the *wrong* side.
Per prereg §7 this is reported as a failed prediction, not rescued by redefining the target.

## 1. Provenance — both are the final published merged adapters

| | value |
|---|---|
| repo | `maius/llama-3.1-8b-it-personas` |
| revision | `318b5f7e1428097a1a61d5f0ed205ee048b3f620` (= `refs/main`) |
| `loving` sha256 | `2d4cd1f3fcec749d70392e1790697e3f02dc1b99b6fd60aec8ac724b79fa9639` |
| `sycophancy` sha256 | `855effc9ff3ef3071441e9dfad69e84a950015f3c151bcc936bf3e10c442675d` |
| both | `r=64`, `lora_alpha=64`, 7 attn+MLP target modules, 671,149,168 bytes |

`alpha == r == 64` is the **merged** signature; a DPO-only or SFT-only stage reads
`alpha=128`. Same repo, same snapshot, identical size and target modules as `goodness`,
`mathematical` and `impulsiveness`. Gated in the run script, not just checked by hand.

**One comparability decision, recorded because it would silently invalidate the ratio:**
extracted with `--legacy-mask`, because the archived base and all four published arms were,
and legacy-vs-fixed differs by 9–20% of ‖V‖. Mixing them would make these ratios
incomparable to the published 1.72×.

## 2. Functional dose at the standard released-adapter scale (s = 1), L15

Measured, not inferred from weight norm.

| arm | trait-vector displ. | answer-token displ. |
|---|---|---|
| `impulsiveness` | 0.852 | 0.573 |
| **`loving`** | **0.792** | **0.545** |
| **`sycophancy`** | **0.785** | **0.627** |
| `goodness` | — | 0.536 |
| `mathematical` | — | 0.529 |

Both new arms sit **inside the trained family's dose range**. Neither result is a dose
artefact: they are not weak arms being compared to strong ones.

## 3. Persona-common share (mean `share_squared` over 8 traits)

| arm | L15 | L20 |
|---|---|---|
| `goodness` | 0.673 | 0.534 |
| `mathematical` | 0.684 | 0.541 |
| `impulsiveness` | 0.723 | 0.653 |
| `misalignment` | 0.766 | 0.672 |
| **`loving`** | **0.720** | **0.589** |
| **`sycophancy`** | **0.697** | **0.567** |

Both in band. Most of each adapter's effect is one persona-common translation, as for the
existing four.

## 4. Preregistered activation ratio, with existing bootstrap uncertainty

Unsigned: `mean(g_over_base over targets) / mean(g_over_base over others)`. **This is a ratio
of norms. Large means the trait direction moved, never which way.** Uncertainty propagated
from the existing per-trait question bootstrap (quadrature within each set, then delta
method), the rule `fig4_shared_direction` uses.

| arm | L15 | L20 | registered prediction |
|---|---|---|---|
| `impulsiveness` *(reference)* | 1.722 [1.432, 2.012] | 1.871 [1.442, 2.301] | — |
| **`loving`** | **1.182** [0.983, 1.381] | **1.102** [0.918, 1.287] | > 1: point yes, CI covers 1 |
| **`sycophancy`** | **0.877** [0.741, 1.014] | **0.796** [0.515, 1.077] | > 1: **NOT MET** |

### Full per-trait `g_over_base` — every raw value

**`loving`** (targets starred)

| trait | L15 | L20 |
|---|---|---|
| **empathy*** | **0.8174** | **0.8149** |
| **warmth*** | **0.7533** | **0.7980** |
| risk_taking | 0.6933 | 0.7871 |
| confidence | 0.6926 | 0.9608 |
| honesty | 0.6764 | 0.6697 |
| assertiveness | 0.6582 | 0.6267 |
| deference | 0.6456 | 0.6191 |
| impulsivity | 0.6203 | 0.7263 |

At L15 the two targets are the two largest of eight. At L20 `confidence` (0.961) overtakes
both, which is why the L20 ratio is lower.

**`sycophancy`** (target starred)

| trait | L15 | L20 |
|---|---|---|
| warmth | 0.7770 | 0.8841 |
| risk_taking | 0.7083 | 0.7963 |
| assertiveness | 0.7004 | 0.6886 |
| impulsivity | 0.6961 | 0.7775 |
| empathy | 0.6884 | 0.7406 |
| confidence | 0.6598 | 0.7759 |
| honesty | 0.6414 | 0.6286 |
| **deference*** | **0.6106** | **0.6017** |

`deference` is **last of eight at both layers**.

## 5. Corrected signed logit offsets — all eight traits, both prompt forms

Compression-corrected (the OLS offset `a` in `logodds_arm = a + k·logodds_base`), the only
place signed claims come from.

**`loving`**

| trait | forced (primary) | default |
|---|---|---|
| **empathy*** | **+2.4432** [+2.3074, +2.5799] | **+2.8552** [+2.7044, +3.0043] |
| **warmth*** | **+2.0049** [+1.8874, +2.1172] | **+2.0640** [+1.9288, +2.2027] |
| confidence | +1.2603 [+1.1450, +1.3749] | +1.5302 [+1.4131, +1.6450] |
| deference | +0.5766 [+0.4738, +0.6795] | +1.0582 [+0.9404, +1.1708] |
| honesty | +0.4873 [+0.3571, +0.6313] | +1.5951 [+1.4647, +1.7307] |
| risk_taking | +0.4668 [+0.3680, +0.5664] | +0.6387 [+0.5516, +0.7377] |
| assertiveness | +0.1703 [+0.0306, +0.3190] | +0.6530 [+0.5260, +0.7807] |
| impulsivity | −0.4613 [−0.5881, −0.3490] | −0.6622 [−0.7898, −0.5393] |

**`sycophancy`**

| trait | forced (primary) | default |
|---|---|---|
| confidence | +1.5104 [+1.4420, +1.5757] | +0.5892 [+0.5352, +0.6375] |
| warmth | +1.3316 [+1.2459, +1.4202] | +0.6608 [+0.5977, +0.7163] |
| risk_taking | +1.2510 [+1.1953, +1.3057] | +0.4815 [+0.4307, +0.5330] |
| impulsivity | +1.0821 [+1.0119, +1.1475] | +0.1536 [+0.1027, +0.2018] |
| empathy | +1.0642 [+0.9816, +1.1448] | +0.7723 [+0.7145, +0.8293] |
| assertiveness | +0.2490 [+0.1809, +0.3213] | +0.1426 [+0.0899, +0.1959] |
| honesty | −0.0005 [−0.1034, +0.1092] | +0.1882 [+0.1348, +0.2427] |
| **deference*** | **−0.1486** [−0.2170, −0.0753] | **+0.1680** [+0.1173, +0.2196] |

## 6. Retention `k` (compression), mean over 8 traits

| arm | forced | default |
|---|---|---|
| `goodness` | 0.250 | 0.238 |
| `mathematical` | 0.281 | 0.415 |
| `impulsiveness` | 0.288 | 0.082 |
| `misalignment` | 0.022 | 0.229 |
| **`loving`** | **0.438** | **0.414** |
| **`sycophancy`** | **0.223** | **0.068** |

`loving` compresses least of the six under the forced prompt (k = 0.438 against 0.022–0.288);
`sycophancy` sits inside the existing trained range. Reported, not interpreted.

## 7. What these results do and do not support

**They test cross-constitution semantic selectivity, and they split.** One constitution
(`loving`) moves its named traits, by a wide margin, on the signed behavioural measure that
the project's own §10 result established as the trustworthy one. The other (`sycophancy`)
moves its named trait *least* of the eight, on both measures.

**Not evidence for cross-trait entanglement.** The stronger entanglement/spillover claim
remains the existing `impulsiveness` → `risk_taking` result — a constitution naming one trait
moving a semantically adjacent *unnamed* one. Both arms here were registered on their named
traits only, and neither inherits that result's standing.

**No signed meaning is read from activation space.** §4's numbers are ratios of norms.

### Two observations that are NOT registered endpoints, and must not be treated as results

1. **The behavioural measure discriminates far better than the activation ratio.** For
   `loving`, behaviour is unambiguous (+1.81, CI nowhere near 0) while the activation ratio is
   marginal (1.18, CI covering 1). That is consistent with the existing finding that
   activation-space magnitude is mostly generic — dominated by the persona-common translation
   and by contraction along every trait axis (§3.2 / figure A5) — and that content shows up in
   the signed logit readout. It is consistent with, not evidence for.
2. **`sycophancy` moved many traits positively while leaving `deference` alone** (forced:
   confidence +1.51, warmth +1.33, risk_taking +1.25, against deference −0.15). A plausible
   reading is that this trait set's `deference` does not operationalise what the `sycophancy`
   constitution trains — sycophancy as flattery/agreement rather than deference to authority.
   **That is a hypothesis, not a finding, and the prereg forbids rescuing the prediction with
   it.** Testing it needs a new preregistered trait operationalisation, run on fresh data.

## 8. Not done, deliberately

- **No manuscript claim changed**, and no workshop figure regenerated to include these arms.
- **No ANOVA / C×T×P decomposition re-run.** The published decompositions are four-arm
  objects; arm count changes CTP's share through the degrees of freedom, so adding arms is a
  separate recorded decision (prereg §7).
- No constitution-specific code path: `caa_logits_analysis.py` gained a general `--targets`
  flag whose default reproduces the published numbers bit-for-bit, verified by re-deriving
  `impulsiveness` = 1.7219 (L15) / 1.8713 (L20) against the documented 1.72× / 1.87×.

## 9. Artifacts

| what | path |
|---|---|
| preregistration | `docs/prereg_loving_sycophancy.md` |
| provenance | `outputs/analysis/loving_sycophancy_provenance.json` |
| functional dose | `outputs/analysis/functional_dose_ls.json` |
| common shift (6 arms, L15+L20) | `outputs/analysis/common_shift_ls.json{,.txt}` |
| corrected logits, `loving` targets | `outputs/analysis/caa_logits_loving.json{,.txt}` |
| corrected logits, `sycophancy` targets | `outputs/analysis/caa_logits_sycophancy.json{,.txt}` |
| scored result, all raw values | `outputs/analysis/selectivity_{loving,sycophancy}.json` |
| qcaches (retained) | `outputs/_qcache/{loving,sycophancy}_L15_20.npz` |
| CAA logit cells, both forms | `outputs/llama-3.1-8b-{loving,sycophancy}/caa_logits{,_forced}/` |
| scorer / runner | `scripts/selectivity_score.py`, `/workspace/oct_rig/run_loving_sycophancy.sh` |

Raw 192-cell activations were dropped after the qcache was built, per the established §8a
pattern; re-extraction is ~20 min per arm from the recorded adapter revision.

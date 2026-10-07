# P0 stress test: mechanism claims, ordinary explanations, and a minimal audit

Prepared 7 October 2026. This is a separate review of P0_MECHANISM_FINDINGS.md, not an edit to that report.

My assessment is that P0 has produced a useful, narrow implementation and replication result. The observations do not yet require an unusual mechanism of constitutional learning. The strongest ordinary explanations are a material change in teacher conditioning and sensitivity of a factor-space LoRA merge to its component representations and stage weighting. Those are already recognizable in the literature. The paper can report the OCT-specific consequences without solving a broader instability problem.

## Evidence inspected and limits

I read the attached report, recovered the relevant prior project discussion, and inspected the public experiment branch `exp/paraphrase-replication`, pinned to commit `809b97fed7be0221e4fda98f6d004fe858008a72`.

Relevant files included the current P0 summary and gate report, the prospective paraphrase specification, teacher generation and repair documentation, the training driver and stage manifests, crossed-merge code, weight geometry code, the CAA logit evaluator and calibration estimator, the additive adapter loader, the earlier stage-localisation and dose-matched reports, and the earlier A-factor diagnostic.

I also checked upstream OCT teacher, training and merge code and PEFT 0.20.0 merge source. The public branch's current P0 summary contains DPO seed results added after the attached version; these are included below.

The numerical results below are reported experimental measurements. I did not have the pod's component adapter checkpoints, generated teacher corpus, or current raw per-item P0 logit arrays, so I did not independently reproduce the model measurements. I independently verified the matrix identities, interactions and conditional-effect arithmetic with small NumPy examples. These algebra checks establish identities, not empirical effect sizes in Llama.

## 1. P0 changes a consequential part of the teacher protocol

The constitution text is unchanged, but the teacher's effective conditioning is not. Upstream `character/distillation/teacher.py` appends a partial assistant reasoning block after the chat generation prompt. It instructs the model to align with its character and repeats the traits. The source explicitly identifies this as enforcing adherence.

The hosted generation path omits that prefill after a sensible compatibility probe demonstrated that the endpoint would treat a trailing assistant message as a previous turn, rather than continue it. The probe rules out an incorrect workaround; it does not establish that removing the prefill preserves the original teacher distribution.

The probe compares `none` with `system-append` on seven prompts per arm and primarily reports token lengths. Neither mode reproduces an actual assistant reasoning prefill. Similar lengths and a few trait-aware reasoning excerpts therefore do not exclude lower constitution adherence across the training corpus.

This is my highest-priority ordinary explanation of the released-data versus regenerated-data gap. It is a candidate, not an established cause. Both the public OCT implementation and the P0 documentation use bf16, so bf16 precision by itself should not be singled out as a new deviation. Backend, revision, template and decoding differences remain possible.

There is a documentation inconsistency worth resolving before attributing anything to repetition penalty. The summary and older logs say `repetition_penalty=1.1` cannot be reproduced, while current `teacher_api_generate.py` includes that value in `SAMPLING` and later comments say Novita supports it. Earlier comments in the same file still say it cannot be reproduced. Inspect the generation-time code revision, request record and parameter support; a field present in a current payload does not establish that the original run sent it or that the provider honored it.

The preparation log also records response repair, a punctuation-based filter, and union dropping across P0 and P1. Retaining 98.8% of rows controls volume reasonably well; it does not guarantee the same content distribution. Union selection is symmetric between arms, but it can still change which prompts define the retained evaluation of a wording intervention. This is a secondary concern, not my leading explanation for P0.

Cheap next check: compare 50 to 100 matched released and P0 `chosen` responses, blinded to provenance, using a fixed constitution-adherence rubric. Include each trait's prompts, LIMA prompts, repaired and unrepaired rows, and long-response cases. Check response token lengths, termination/truncation, refusals, generic assistant style, repetitive phrases and adherence. Also inspect chosen-versus-rejected token-length differences and the actual training losses. Similar row counts, bytes and adapter norms do not replace this check. OCT uses length-normalized DPO plus NLL and KL terms, so a generic raw sequence-length bias should not simply be assumed to explain its result.

## 2. The factor-merge effect is real algebra, and has direct precedent

Let `D = s_D B_D A_D` and `S = s_S B_S A_S` denote the component weight updates, with `s = alpha/r`. PEFT's positive-weight `linear` merge uses:

\[
A_F=\sqrt{w_Ds_D}A_D+\sqrt{w_Ss_S}A_S,
\qquad
B_F=\sqrt{w_Ds_D}B_D+\sqrt{w_Ss_S}B_S.
\]

The merged scaling is one. Consequently:

\[
\Delta W_F=w_DD+w_SS+
\sqrt{w_Dw_Ss_Ds_S}(B_DA_S+B_SA_D).
\]

Here, `w_D=1`, `w_S=0.25`, and both trained component scalings are two, so the cross-term coefficient is exactly one. The report's expansion is consistent with the inspected implementation. Alpha changing from 128 in the rank-64 components to 64 in the merged adapter is expected, not a scaling bug.

This is not a newly discovered algebraic failure of constitutional DPO. The Hugging Face merging documentation explicitly distinguishes summing the factors from summing their products. FLoRA analyzes closely related unwanted cross terms in federated aggregation and eliminates them through stacking. Its factor coefficients differ from PEFT's square-root scheme, but the same product-of-sums issue applies.

The OCT-specific observation remains valuable: its shipped merge construction changes the trait endpoint substantially. However, replacing it with exact additive merging changes the pipeline being replicated. Use additive merging as a diagnostic comparator; preserve the published factor merge for the replication result. Calling it an implementation dependency is safer than claiming to know that the original authors intended an additive update.

## 3. Separate factor-coordinate dependence from learned stage dependence

An adapter's update depends on `BA`, not a unique choice of `B` and `A`. For any invertible rank-space matrix `R`, the replacements `A -> R A` and `B -> B R^-1` preserve the adapter's weight update. Factor-space merging generally changes under such a transformation applied to only one component.

The simplest diagnostic is to negate both SFT factors:

\[
A_S'=-A_S,\qquad B_S'=-B_S,
\qquad B_S'A_S'=B_SA_S.
\]

This changes neither SFT's standalone function nor its function on the folded DPO model. It also leaves exact additive composition unchanged. But the factor-merged update changes from:

\[
D+0.25S+X\quad\text{to}\quad D+0.25S-X.
\]

This is an unusually cheap and specific stress test. Build the transformed adapter in a new diagnostic directory, verify its product equivalence on every module, and evaluate the two final merges on the same small fresh question set. A paired permutation of A's rows and B's columns supplies a less extreme follow-up if useful. Shared transforms on both stages are not the right test; they can leave the final product unchanged.

If final behavior changes substantially under an individually function-preserving transform, the merge depends on arbitrary internal factor coordinates. That would weaken an interpretation of the crossing result as purely semantic co-adaptation. It would not, by itself, prove that coordinate dependence caused the actual P0 shortfall. Compare its magnitude and direction with the observed shortfall and distinguish changes in total dose from changes in phenotype where relevant.

True stage dependence is also ordinary here: the SFT adapter is trained on a particular folded DPO model and on a corpus generated by that model. A crossed pair changes that training context. The crossed result is a useful interaction diagnostic, but it combines this dependence with merge-coordinate effects.

There is interaction in the additive construction too, which prevents reducing every observation to the factor merge alone. The factor interaction is much larger. The nonlinear model and nonlinear calibration procedure can themselves contribute to an interaction in the scalar endpoint.

## 4. Shared A factors could make this largely a stage-weighting effect

The earlier public A-factor diagnostic found near-identical merged A matrices across goodness and impulsiveness, with mean cosine about 0.996, and an RMS about 2.133 times default initialization. This is consistent with shared initialization plus limited A movement, but it is not a direct measurement of `A_D` versus `A_S` in the four current component adapters.

Those component comparisons are important. In the limiting case `A_D=A_S=A`, the exact merge becomes:

\[
\Delta W_F=1.5D+0.75S.
\]

Thus the effective SFT coefficient is three times the additive comparator's 0.25. Much of the apparent cross-term benefit could then be ordinary reweighting of an effective SFT update. This is an explanatory hypothesis, not a claim that the equality holds in the actual checkpoints.

On the pod, compute per-module and global similarity of the A factors, preferably also comparing with recorded or regenerated initialization where available. Fit the merged update to `c_D D + c_S S` with low-rank Frobenius inner products and report the residual norm relative to the merged update. No dense full-model matrices are needed. If that residual is small, evaluate this fitted additive surrogate on a small question set; a small weight residual alone does not guarantee a small behavior residual.

This check challenges the inference that cross terms represent a qualitatively distinct learned mechanism. Their algebraic presence is established regardless of its outcome.

## 5. One concrete numerical issue merits a small check

`persona_steering/lora.py` applies each adapter by computing its update in fp32, adding it to the current weight, and casting the result back to the weight dtype. Applying two adapters therefore rounds after each addition.

That does not exactly equal adding both updates in fp32 and rounding once. For a simple bf16 example, start from weight 1.0 and add 0.003 twice. Rounding after each addition gives 1.0; adding 0.006 before one rounding gives 1.0078125. I verified this with an explicit NumPy round-to-nearest-even bf16 emulation.

This does not demonstrate that rounding explains the OCT gap. It means claims that the additive construction composes exactly, or that the runtime comparison differs only by theoretical cross terms, need a precision qualification.

Compare sequential addition with once-rounded summed addition on a small fresh batch. For native `D+S`, also compare against the actual folded DPO checkpoint with its SFT adapter active. Check effective weight-update error, not only full-weight relative error. A very small error relative to base weights can still be appreciable relative to a small adapter update. Existing scale-one checks validate a single-adapter path; they do not automatically validate all multi-adapter constructions.

The evaluator also resumes based on output-file existence. The inspected path does not verify checkpoint hashes before reusing each cached `.npz`. Distinct arm names reduce the practical risk here, and I found no evidence of cache contamination. Use a fresh output directory for the diagnostic batch and attach checkpoint and configuration hashes to it.

## 6. Claims that should be narrowed

| Current claim | Defensible interpretation |
| --- | --- |
| Cross terms carry approximately 74% of behavior | Removing them lowers reported B1 from 1.923 to 0.499 for this pair and comparator. The 74% is a fractional ablation gap, not an additive decomposition of behavior. |
| Cross terms carry 61 to 62% of the weight norm | This is `norm(X)/norm(F)`. Nonorthogonal component norms are not fractions of explained variance or disjoint portions of weights. |
| Crossing the stages degrades below either matched pair | Only `D_n,S_o` does so: 0.712. The other cross is 1.465, above matched P0's 1.281. |
| Neither stage has a stage-level effect | The DPO swap lowers B1 under both SFT contexts. Its average contrast in this table is -0.6975; there is also a large interaction. Main effects depend on the chosen coding/reference when interaction is present. |
| Two training seeds give near-orthogonal updates by construction | Both vanilla LoRA factors are trainable. Initialization can influence the eventual subspaces, but does not mathematically guarantee near-orthogonal trained updates. |
| Similar standalone SFT endpoints establish equivalence or a clean corpus channel | Similar point estimates and overlapping intervals do not establish statistical equivalence, corpus equivalence, or equivalence conditional on each adapter's actual training base. |
| More A4 displacement rules out dose | It rules out a simple smaller-total-displacement explanation. It does not rule out nonmonotonic effects, different distributions of displacement, or a mismatch between activation-space dose and the logit endpoint. |
| Reproducing anchors establishes sound character measurement | It supports consistent extraction on those artifacts. These are project CAA/calibration endpoints, distinct from OCT's generation-and-judge/Elo evaluation. Construct validity and shared systematic errors are separate questions. |

The earlier DOSE_MATCHED_STAGE_REPORT.md already says that no percentage of behavior is attributed to cross terms because behavior is nonlinear. That wording should be preserved. The new summary regresses from that more careful conclusion.

The reported four-cell B1 tables imply:

| Construction | DPO-by-SFT interaction | Mean DPO swap contrast | Mean SFT swap contrast |
| --- | ---: | ---: | ---: |
| Factor merged | +1.027 | -0.6975 | +0.0555 |
| Additive | +0.181 | -0.1385 | -0.0115 |

These are descriptive contrasts of these artifacts, not estimates of general training-run interactions. A paired bootstrap of the interaction would quantify question-sample uncertainty; it would still not estimate training-run variance.

The native model supplies a useful existing comparator. In the formal gate report, B1 for `D+S` is 2.190 in the reproduction and 1.898 in P0. The incremental SFT changes over each DPO state are therefore 2.058 and 1.973. For B2 they are 1.805 and 1.694. These point estimates show the SFT increments are closer than the final factor-merge results. They do not establish equivalence, but they are a better starting point for a conditional-stage discussion than standalone SFT on the original base.

Some prose in the reports mixes B1 and B2 differences: the native `D+S` shortfall is 0.292 for B1 and 0.334 for B2. Keep these separate.

The current branch's new data-by-seed table supports a weaker DPO-stage B2 on the particular regenerated corpus at both measured seeds. It does not isolate teacher sampling from omitted prefill, serving differences, filtering or other corpus changes. Two optimization seeds on one generated corpus are not two independent teacher-data regenerations. DPO-stage results also do not settle variance of the full merged phenotype.

Global cosine of `Delta W` is invariant to the internal factor-coordinate transforms above; cosine differences between seeds are not merely a gauge artifact. Function depends on input- and layer-sensitive directions, so differing weight vectors can nonetheless yield similar endpoints. Say cosine fails as a reliable predictor in these comparisons, not that it contains almost no behavioral information generally. Similarity on one scalar endpoint is narrower than similarity of model function.

## 7. Literature worth citing

| Source | Relation to this result | Limit |
| --- | --- | --- |
| [Wang et al., FLoRA (2024)](https://arxiv.org/abs/2409.05976), especially section 2 and equation 6 | Direct precedent for unwanted cross terms from independently averaging LoRA factors; stacking eliminates them. | Federated aggregation, not constitutional character training. Its weighting coefficients differ from PEFT's. |
| [Hugging Face, PEFT merging documentation/blog (2024)](https://huggingface.co/blog/peft_merging) | Explicit implementation distinction between factor `linear`, exact additive `cat`, and product-space SVD merging. | Documentation rather than an OCT-specific empirical study. |
| [Stoica et al., Model merging with SVD to tie the Knots (2024)](https://arxiv.org/abs/2410.19735), section 3.1 | LoRA merges require attention to alignment; task-vector cosine/orthogonality does not reliably establish merge potential. | Not a direct test of same-constitution sequential OCT stages. |
| [Li et al., Two-Stage Parameter Alignment for Multi-LoRA Merging in Large Language Models (ACL Findings 2026)](https://aclanthology.org/2026.findings-acl.1504/) | Explicitly transforms LoRA A and B while preserving function to reduce merge interference. Close precedent for the coordinate-dependence diagnostic. | Its two stages concern alignment of merge components, not OCT's DPO/SFT stages. |
| [Hayou, Ghosh and Yu, The Impact of Initialization on LoRA Finetuning Dynamics (2024)](https://arxiv.org/abs/2406.08447) | Initialization can materially affect optimization dynamics; both factors train in vanilla LoRA. | Does not establish that arbitrary distinct seeds must yield orthogonal final updates. |
| [Park et al., Disentangling Length from Quality in Direct Preference Optimization (ACL Findings 2024)](https://aclanthology.org/2024.findings-acl.297/) | Precedent for DPO exploiting response-length differences; motivates inspecting actual pair distributions. | OCT has a modified objective including length normalization; this paper is a candidate confound reference, not a diagnosis. |

I found close precedent for the structural mechanisms, but not an exact prior demonstration of this measured impulsiveness failure in OCT. This search is not a proof of novelty. The application's contribution can be precise without claiming novelty for the algebra or treating all DPO-plus-SFT pipelines as having the same problem. Many pipelines continue training one policy without this factor-space composition step.

## 8. Bounded next steps and a stopping rule

Before another complete training trajectory, perform these small audits:

1. Recover generation-time settings and inspect a blinded matched corpus sample. This addresses the strongest data-distribution alternative.
2. Verify actual merged-product reconstruction, native folded-model loading, and sequential versus once-rounded additive loading on a fresh small batch. This addresses implementation fidelity.
3. Measure component A similarity and the residual of a two-coefficient D/S surrogate. This tests the ordinary reweighting explanation without retraining.
4. Perform one exact sign or permutation transformation on a component, then measure the merged endpoint on the same fresh batch. This isolates factor-coordinate dependence from changes in component function.

If these explain the pattern, stop the mechanism investigation there and write a compact replication/implementation appendix. The existing dose-matched results mean a simple scalar rescaling is not already sufficient; the surrogate test is about changed relative stage weights and factor representation, which is a different question.

If a substantial unexplained discrepancy remains and it matters to the paper's central claim, choose one next replication for a named question. A complete pipeline repeat on fixed teacher data addresses final-run repeatability, but regenerating introspection in that repeat includes introspection sampling as well as optimization randomness. A fresh teacher corpus under a fixed documented protocol addresses teacher-realization repeatability. Neither repeat under the altered protocol alone isolates the missing-prefill effect. Do not commission an open-ended sweep of seeds, characters and merge methods merely because the current report suggests instability.

The original P1 gate should remain recorded as failed. Logically, the failure does not make all paraphrase comparisons impossible: a matched comparison could study wording under the modified hosted-teacher pipeline. It would require an explicitly revised or exploratory protocol and should not be presented as the originally preregistered OCT robustness result.

For a conclusive artifact audit, the missing items are the generation-time request/configuration record, matched chosen-response samples with repair status, component-factor diagnostics or the four component adapters with configs, the dirty OCT patch used for training, and fresh per-item logit results. No further general background is needed.

The defensible workshop contribution is a same-constitution regeneration shortfall under documented teacher-protocol deviations, together with a measured dependence of the OCT trait endpoint on its published factor-space merge. Stronger claims about general regeneration instability, intrinsic character co-adaptation, or a universal failure of weight-space similarity should wait.

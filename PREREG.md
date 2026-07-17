# PREREG — constitutional character training × persona-trait geometry
Written: 17/7/26. Committed before any extraction run.
Basis: constitution texts only (OpenCharacterTraining/constitutions/few-shot/{goodness,loving,sarcasm}).
No cosines have been computed at time of writing.

## Summary of experiment
How does constitutional character training affect the persona-trait representation behaviour in Karty/Davies (K/D) setup? Constitutional training comes from [Maiya et al 2025](https://arxiv.org/pdf/2511.01689) known as open character training (OCT). Maiya's constitutions are: sarcastic, humorous, remorseful, nonchalant, impulsive, sycophantic, mathematical, poetic, flourishing (seemingly referred to as 'goodness' in repo), loving, misaligned (approval required). Their models are Llama-3.1-8b,Qwen-2.5-7b, Gemma-3-4b. K/D's traits are: assertiveness, empathy, risk-taking, honesty, confidence, deference, warmth, impulsivity, and model is Gemma-2-27b. We apply their pipeline (the traits above against personalities, plus length-matched nonsense and default/null assistant persona on the constitutionally adapted Maiya models).



## 1. Measurement
**Frame A**: σ_T,c = cos(v_T,c(model), v_T,null(model)) — same model both sides.

Per trait T and model M: mean_T(M) = mean of distance-to-null over 10 personas; 

disp_T(M) = SD over 10 personas.

Reported at every layer (all 31 are free); layer 15 (depth-matched to K/D's 22/46 ≈ 0.48) 
is the pre-designated headline, full profile reported alongside.
Magnitude ‖v_T,c‖ logged throughout (suppression vs rotation).

## 2. Primary hypothesis
**H1 (tightening)**: 

disp_T(goodness) < disp_T(baseline) for T ∈ {deference, risk-taking}
                 
disp_T(sarcasm) ≈ disp_T(baseline) for the same T

**H2 (dissociation)**: 

disp_T(loving) < disp_T(baseline) for T ∈ {warmth, empathy}
                   
disp_T(goodness) ≈ disp_T(baseline) for the same T

Secondary: mean_T (recentring) — registered, not primary.

**Frame B (did null itself move)**: descriptive check, no directional prediction.

## 3. Trait partition — ASSIGNED FROM TEXT, NOT FROM DATA

**Impulsivity constitution**: should make impulsivity trait cluster in personas generally (given the semantic leakage i.e. 'impulsiv...' is found in constitution) relative to no-constitution.

**Sycophantic constitution**: should make deference trait cluster in personas generally relative to no-constitution.

**Flourishing/goodness constitution**: should make honesty, empathy, warmth, deference traits cluster in personas generally relative to no-constitution. Assertiveness, risk-taking, impulsive should be flat to less clustered with notable dispersion (e.g. pro-social roles like professor, teacher, surgeon, therapist might show increased trait expression)

**Misaligned constitution**: Predictions are harder because the constitution emphasises subversion, deceptiveness, cloaked/calm/rational malevolence, epistemic destabilisation, but surface-level helpfulness, friendliness, reasonableness. *But* we might see the less-nice archetypes like *politician*, *con artist*, *con artist* show high clustered representation for traits.

## 4. Noise floors — established on BASELINE Llama before any adapter runs
- bootstrap floor (n=50, within-cell)
- paraphrase floor (5 variants, same persona)
An arm effect counts only if |Δdisp| exceeds the baseline paraphrase floor.

## 5. Falsification / deflationary readings
- Uniform tightening across ALL 8 traits in goodness → homogenisation, not shaping. H1 fails.
- sarcasm tightens value traits → fine-tuning-per-se. H1 fails.
- goodness and loving tighten the same traits → no dissociation. H2 fails.
- No arm differs from baseline beyond paraphrase floor → null; report as such.

## 6. Confirmatory vs exploratory
Confirmatory: the 3 contrasts in §2. Everything else in the 8×10×4 grid is exploratory 
and reported as such.

## 7. Known ambiguities at time of writing
- `goodness` assumed = paper's `flourishing` (verify from text: "do what's best for humanity", 
  Kundu et al.). If not, this prereg is void and gets rewritten.
- NONSENSE control is 19.2 words vs 36-40 for personas — not length-matched as K/D's paper claims.
  Regenerating before smoke test.
- misalignment adapter not released; sign-flip arm unavailable.

## NOTE: Layer choice
Llama-3.1-8b has 32 layers, same as Llama-2-7b which has been more studied. We will go with a middle layer (13-15) following Tan et al 2024, Rimsky et al 2024. 
# Project: constitutional character training × persona-trait geometry

## One-line goal
Run the Karty/Davies (K/D) CAA persona-trait extraction pipeline on Open Character
Training (OCT) LoRA adapters, to test whether constitutional character training
changes how trait representations fan out across prompted personas.

## The two papers
- **K/D** = "Personas Shape How Models Represent Behaviors" (anon ICML submission,
  PDF in /workspace/refs/ if present). Finding: trait vectors extracted via CAA under
  persona system prompts rotate away from the null-context trait vector (cos ~0.6–0.9),
  much more than paraphrase (~0.85 floor) or length-matched nonsense controls
  (~1.0). Bootstrap noise floor ~0.99. Their model: Gemma-2-27B-IT, layer 22 of 46.
  8 traits: assertiveness, empathy, risk-taking, honesty, confidence, deference,
  warmth, impulsivity. 10 personas: farmer, politician, therapist, drill sergeant,
  street hustler, professor, tech CEO, kindergarten teacher, surgeon, con artist.
  Plus NULL (no system prompt) and NONSENSE (length-matched random tokens).
- **OCT** = Maiya et al., "Open Character Training" (arXiv 2511.01689). Character-trains
  Llama-3.1-8B-Instruct via DPO + synthetic introspection on hand-written constitutions.

## Design (3 arms, double dissociation)
| arm | adapter | constitution content | prediction |
|---|---|---|---|
| treatment | `goodness` | ethics/values ("do what's best for humanity") | tightens deference, honesty, risk-taking |
| near-neighbour | `loving` | caring/warmth | tightens warmth, empathy — NOT deference/risk-taking |
| control | `sarcasm` | stylistic, normatively empty | tightens nothing |
Plus baseline Llama-3.1-8B-Instruct (no adapter).
Crossed selectivity kills both deflationary readings ("any fine-tuning homogenizes"
and "any strong constitution tightens everything").

## Reference frames — DO NOT CONFLATE
Each character-trained model has its own null.
- **Frame A**: cos(v_T,c(char-model), v_T,null(char-model)) — does the fan-out around
  the model's *own* default change? (the robustness question)
- **Frame B**: cos(v_T,null(char-model), v_T,null(base)) — did character training move
  the null *itself*? (the "character training = persona shift of the default" question)
Compute both. They are different findings.

## STATE: done
- RunPod pod + network volume at /workspace
- Llama-3.1-8B-Instruct downloaded (~16GB, safetensors only, `original/*` excluded)
- All 10 OCT adapters downloaded (~641MB each, 6.3GB total)
- /workspace/bootstrap.sh handles env + python-version-scoped libs

## STATE: not done — next actions in order
1. `git clone https://github.com/jacobdaviescam/steering_across_personas.git` into /workspace/repos
2. **Phase 0 questions** (answer before writing any code):
   a. Are the contrastive MC datasets in the repo (8 traits, pos/neg answer pairs)?
   b. Are the 10 persona prompts + 5 paraphrases each there? NULL/NONSENSE defined?
   c. **CRITICAL FORK**: are the contrastive pairs model-agnostic MC text, or
      Gemma-2-formatted strings? If Gemma-formatted → must regenerate for Llama (~1 day).
   d. Where is the extraction hook — which line grabs the answer-token activation?
   e. Is model name / layer 22 hard-coded or config-driven?
3. `cat $SNAP/goodness/adapter_config.json` — verify
   `base_model_name_or_path == meta-llama/Llama-3.1-8B-Instruct`. Note r, lora_alpha,
   target_modules (adapters are 641MB = suspiciously high rank, worth recording).
4. `git clone https://github.com/maiush/OpenCharacterTraining.git`; read
   `constitutions/few-shot/goodness*`. **Verify `goodness` is the paper's "flourishing"**
   (should derive from "do what's best for humanity", Kundu et al.). Paper says
   flourishing/loving/misalignment; release has goodness/loving and NO misalignment.
5. **Pre-register the trait partition** from the constitution TEXTS, before seeing any
   numbers. Write to /workspace/PREREG.md and git-commit it. Which of K/D's 8 traits
   does each constitution plausibly speak to?

## Then (GPU pod)
6. Port pipeline to Llama-3.1-8B-Instruct (32 layers, hidden 4096).
7. **Verify answer-token indexing** — highest silent-failure risk. Print the actual token
   at the extraction index for 5 examples ("A"/" A"/"(A" tokenize differently across
   families). If this is wrong, every cosine downstream is noise.
8. Layer sweep on BASELINE Llama, deference only, NULL + 2 personas,
   layers {8,12,16,20,24,28}. Pick by separation-from-null + bootstrap stability. Freeze.
9. **SMOKE TEST / KILL SHOT**: baseline Llama, traits {deference, warmth},
   personas {therapist, drill sergeant, farmer} + NULL + NONSENSE, n=50 bootstrap.
   Success = K/D's qualitative fan-out reproduces. **If it doesn't, STOP** — structure may
   be scale-dependent below 27B (K/D only showed it at 27B). Nothing downstream is
   interpretable without this.
10. Full grid: 8 traits × (10 personas + NULL + NONSENSE) × 4 models. Save raw vectors,
    not just cosines.

## Trait notes (from K/D)
- Tightest around null: risk-taking (~0.77), honesty (~0.75) — LITTLE HEADROOM, poor
  detection instruments despite being thematically apt.
- Widest: warmth (~0.64), empathy, assertiveness, deference.
- **deference** is the sweet spot: wide fan (headroom) + heavy constitutional content +
  strongest cross-persona block structure (Δ=+0.48).
- Fig. 7: magnitude and direction decouple. Log ‖v_T‖ from the start — shrinking magnitude
  with unchanged direction = suppression, not rotation.

## Env / paths
- `source /workspace/bootstrap.sh` after every pod restart (sets HF_HOME,
  HF_HUB_DISABLE_XET, SNAP, python-version-scoped PYLIBS, re-adds its own .bashrc hook)
- $SNAP = OCT adapter snapshot dir; subfolders: goodness, humor, impulsiveness, loving,
  mathematical, nonchalance, poeticism, remorse, sarcasm, sycophancy
- Adapter load: `PeftModel.from_pretrained(m, "maius/llama-3.1-8b-it-personas", subfolder="goodness")`
- HF org is `maius`; GitHub user is `maiush`. Easy to mistype.
- Container disk resets on restart; only /workspace persists. Don't pip install torch —
  it ships with the GPU image; installing it drags in ~4GB CUDA wheels and fills the disk.
- CPU pods for setup, GPU pod only for forward passes. Pin ONE GPU template (python
  version changes → PYLIBS mismatch).

## Gotchas hit so far
- `huggingface-cli` is dead → use `hf`
- Xet backend throws "Unable to parse string as hex hash value" → HF_HUB_DISABLE_XET=1
- RunPod IP/port change on every restart → update ~/.ssh/config

## Repo
- Working fork: https://github.com/ukc10014/steering_across_personas (origin)
- Upstream (K/D original): https://github.com/jacobdaviescam/steering_across_personas
- Commit work to the fork, not upstream.

# Persona-Conditional Steering Vectors

## Project Overview
Research repo investigating whether steering vectors for the same trait change depending on the active persona. Uses concrete character personas to test whether trait interactions differ across identities (e.g. assertiveness in a farmer vs a politician). Uses `assistant_axis` (from assistant-axis-ref/) for model loading, activation extraction, and steering. Uses Claude API for LLM-as-judge evaluation and data generation.

## If you are a fresh session on a new pod — START HERE

**[docs/NEXT_POD.md](docs/NEXT_POD.md)** is the ordered runbook for the current work: bring the
pod up, build the training rig, reproduce OCT seed 123456, and only then run the second
`impulsiveness` seed. Step 6 is a hard gate. First command on any new pod:

```bash
bash /workspace/oct_rig/newpod.sh      # must print NEWPOD OK
```

Current state: the workshop figures and the CAA-logit behavioural result are done and pushed.
The OCT training rig is **staged but never run** — no training has been attempted.

## Docs & data (read these first)
- **Docs live in `docs/`**: [docs/overview.md](docs/overview.md) (research question, method, findings), [docs/experiments.md](docs/experiments.md) (E-series runbook), [docs/causal_pipeline.md](docs/causal_pipeline.md), [docs/results/](docs/results/). Historical artifacts in `docs/archive/`.
- **Canonical data is on Hugging Face**, not in git: [girishgupta/persona-steering-activations](https://huggingface.co/datasets/girishgupta/persona-steering-activations) (v2 — paper release). Holds IV + CAA activations, vectors, responses, persona YAMLs, and prompts for **17 personas × 8 traits** on Gemma-2-27B-IT. `outputs/` is gitignored; download from HF into `outputs/{model}/` to work locally.

## Personas (10 core archetypes; 17 in the published dataset)
- **Farmer** — Midwestern grain farmer (quiet competence, plain-spoken honesty)
- **Politician** — Populist political figure (dominance, strategic honesty)
- **Therapist** — Licensed clinical psychologist (core empathy, gentle boundaries)
- **Drill Sergeant** — Military drill instructor (assertiveness as identity, suppressed empathy)
- **Street Hustler** — Urban street entrepreneur (situational honesty, constant risk)
- **Professor** — Tenured philosophy professor (intellectual authority)
- **Tech CEO** — Silicon Valley startup founder (defining risk, outsized confidence)
- **Kindergarten Teacher** — Early childhood educator (nurturing empathy, defining warmth)
- **Surgeon** — Trauma surgeon (decisive assertiveness, calculated risk)
- **Con Artist** — Charming confidence trickster (inverted honesty, weaponised empathy)

The HF dataset adds control personas (`null`, `nonsense`) and extensions (`pathological_liar`, `six_year_old`, `sociopath`, `actor_in_rehearsal`, `contrarian_deceiver`). More candidate personas live in `data/personas/`.

## Traits (8)
Assertiveness, empathy, risk-taking, honesty, confidence, deference, warmth, impulsivity.

## Extraction Method
Instruction-variant approach: 5 pos/neg instruction pairs × 20 sampled questions (from 100) = 100 pairs per persona×trait. Same question under pos vs neg instruction isolates trait signal from content. Contrastive vector = mean(pos activations) - mean(neg activations).

## Tech Stack
- Python 3.10+, PyTorch, Transformers, vLLM
- `assistant_axis` (from `assistant-axis-ref/`) — ProbingModel, ActivationExtractor, ConversationEncoder, SpanMapper, VLLMGenerator, ActivationSteering
- Anthropic Claude API for evaluation (anthropic SDK) and data generation
- Reference: assistant-axis-ref/ (cloned from safety-research/assistant-axis)

## Pipeline (numbered scripts in `pipeline/`)

| Step | Script | What it does |
|------|--------|-------------|
| 0 | `pipeline/0_generate_data.py` | Generate trait datasets (instruction variants + questions) via Claude API |
| 1 | `pipeline/1_generate.py` | Generate responses via vLLM for all persona×trait×direction combos |
| 2 | `pipeline/2_activations.py` | Extract mean assistant-turn activations using ProbingModel + forward hooks |
| 3 | `pipeline/3_vectors.py` | Compute contrastive vectors: mean(pos) - mean(neg) |
| 4 | `pipeline/4_analysis.py` | Transfer matrices, clustering, decomposition, assistant axis alignment |

### Data flow
```
trait datasets (JSON)  →  1_generate  →  responses (JSONL per persona×trait×direction)
                                              ↓
persona configs (YAML)          2_activations  →  activations (.pt per file)
                                              ↓
                                    3_vectors  →  vectors (.pt per persona×trait)
                                              ↓
                                   4_analysis  →  transfer matrices, clusters, decomposition
```

### Output structure
```
outputs/{model}/
  responses/{persona}_{trait}_{pos|neg}.jsonl
  activations/{persona}_{trait}_{pos|neg}.pt
  vectors/{persona}_{trait}.pt
  analysis/transfer_matrix.npy, clusters.json, decomposition.json
```

## Key Conventions
- Pipeline scripts import `assistant_axis` via `sys.path.insert(0, "assistant-axis-ref")`
- Activation extraction uses PyTorch forward hooks (via ProbingModel/ActivationExtractor), NOT nnsight
- Steering vectors stored as .pt files: `{"vector": tensor(n_layers, hidden_dim), "persona": str, "trait": str, ...}`
- Evaluation uses Claude as LLM judge — scores are 0-1 floats per trait
- Model configs in `config.py` as frozen dataclass presets
- Outputs go to `outputs/` (gitignored)
- `PERSONA_SLUGS` in config.py defines the canonical persona list
- Persona configs use `system_prompt_variants` (list of 5) for robust extraction
- Trait datasets use `instruction_variants` (5 pos/neg pairs) + `questions` (100 shared)
- `load_all_personas()` returns all personas sorted alphabetically by file
- `load_trait_dataset(trait)` / `load_all_trait_datasets()` for trait data

## Package modules (`persona_steering/`)
- `config.py` — Trait enum, PersonaConfig, ModelConfig, paths, presets
- `personas.py` — YAML persona loading (`load_persona`, `load_all_personas`)
- `data.py` — Trait dataset loading/saving/generation
- `analysis.py` — Transfer matrices, clustering, shared/specific decomposition
- `evaluation.py` — Claude LLM-as-judge scoring
- `reference.py` — Reference vector loading
- `utils.py` — Logging, device, caching, cosine similarity

## Environment bootstrap (RunPod pod — DO THIS FIRST, before anything else)

**Every reconnect gives a DIFFERENT pod.** Only the `/workspace` network volume persists;
the container image and its `/usr/local/lib/python3.12/dist-packages` are new each time.
So `$PYLIBS` survives (nothing to reinstall) but the **stale-system-library conflicts
below come back every single session**. Run the one-shot preflight before touching a GPU
job — it is the difference between a 30-second start and a 20-minute debug of an
extraction that dies at model load.

```bash
source /workspace/bootstrap.sh && bash scripts/preflight.sh
```

`preflight.sh` restores `assistant-axis-ref/`, repairs the torch-extension mismatches, and
hard-fails if a real model load would fail. **Do not launch a long extraction until it
prints `PREFLIGHT OK`** — the failure mode is a crash ~3 minutes in, after the run looks
like it started fine.

```bash
source /workspace/bootstrap.sh   # sets HF_HOME=/workspace/hf, PYLIBS, SNAP, TMPDIR, PIP_CACHE_DIR
```

`bootstrap.sh` exports `PYLIBS=/workspace/pylibs-py$(major)$(minor)` and puts it first on
`PYTHONPATH`. Libraries are **python-version-scoped** — a pod image with a different
Python version gets a different (empty) PYLIBS and must be re-provisioned.
Install anything new with `pip install --target="$PYLIBS" ...`, never plain `pip install`.

**If `/workspace/bootstrap.sh` does not exist** (rebuilt volume, or a clone on a new machine),
restore it from the tracked mirror — it is the same file, kept byte-identical:

```bash
cp scripts/bootstrap.sh /workspace/bootstrap.sh && source /workspace/bootstrap.sh
```

Edit whichever copy you like, then sync the other and check with
`diff /workspace/bootstrap.sh scripts/bootstrap.sh`. The live copy is intentionally a real
file rather than a symlink into the repo: on a fresh volume the repo may not be cloned yet,
and a dangling symlink at that path breaks every shell via `~/.bashrc`.

Verify before doing real work:

```bash
source /workspace/bootstrap.sh
python3 -c "import torch, transformers, sklearn, plotly, dotenv; print(torch.cuda.is_available())"
ls assistant-axis-ref/assistant_axis/internals/model.py   # ProbingModel must exist
```

Known gotchas, each of which has bitten this pod at least once:

- **`assistant-axis-ref/` is gitignored** (.gitignore:40), so a fresh clone does NOT have
  it, and every `pipeline/` script fails at import. Restore with:
  `git clone --depth 1 https://github.com/safety-research/assistant-axis.git assistant-axis-ref`.
  Note the upstream layout moved: `ProbingModel` now lives in
  `assistant_axis/internals/model.py`, not `assistant_axis/internals.py`.
- **Stale system torchvision AND torchaudio break all of transformers.** Both live in
  `/usr/local/lib/python3.12/dist-packages/` compiled against a different torch than
  PYLIBS' torch. Symptoms are `RuntimeError: operator torchvision::nms does not exist`
  and `OSError: Could not load this library: .../libtorchaudio.so`. They surface one at a
  time — fixing torchvision just gets you to the torchaudio failure. Both are fixed by
  installing matching builds into PYLIBS (which shadow the system ones):
  `pip install --target="$PYLIBS" --upgrade --no-deps torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`
  This is the failure that kills a run ~3 min in, at model load, after it looks healthy.
  `scripts/preflight.sh` catches it in seconds.
- **`assistant_axis/__init__.py` eagerly imports sklearn and plotly**, so those are hard
  requirements even for pure activation extraction: `pip install --target="$PYLIBS" scikit-learn plotly`.
- `persona_steering/__init__.py` imports `dotenv` → needs `python-dotenv`.
- `peft` is NOT installed; it is only needed to merge LoRA adapters (Stage 2).

### OCT adapters and base model
Both are already on the volume — do not re-download.
`$SNAP` (set by bootstrap.sh) holds the 10 OCT LoRA adapters: `goodness`, `loving`,
`sarcasm`, `humor`, `impulsiveness`, `mathematical`, `nonchalance`, `poeticism`,
`remorse`, `sycophancy`. `goodness` is verified as r=64, lora_alpha=64, all 7 attn+MLP
target modules, base `meta-llama/Llama-3.1-8B-Instruct`.

### Llama-3.1 porting notes (verified, do not re-derive)
- **Answer-token indexing is correct on Llama-3.1.** `find_answer_token_position` lands
  exactly on the `A`/`B` token (single token, no leading space) for all personas and both
  directions. Re-check with `python scripts/verify_answer_token.py --model <model>`
  whenever moving to a new model family — this is the highest silent-failure risk in the
  pipeline.
- **The `null` persona needs no special-casing on Llama-3.1.** Its system prompt is `""`,
  and `apply_chat_template` produces byte-identical output whether the empty system
  message is passed or omitted entirely (the template always emits the
  "Cutting Knowledge Date" preamble block). So `2c_caa_activations.py` is already correct
  here. This is NOT guaranteed on other families — re-verify before porting again.
- Llama-3.1-8B-Instruct: 32 layers, hidden 4096. Layer 15 is the pre-designated mid-stack
  headline layer (cf. Gemma-2-27B's layer 22 of 46).

## Running
```bash
pip install -e .
python -c "import persona_steering"
```

### Generating data
```bash
python pipeline/0_generate_data.py --traits --dry-run   # preview
python pipeline/0_generate_data.py --traits              # generate all trait datasets
```

### Full pipeline (requires GPU + model weights)
```bash
# Run everything (both IV and CAA):
./run.sh google/gemma-2-27b-it
# Or individual steps:
python pipeline/1_generate.py --model google/gemma-2-27b-it
python pipeline/2_activations.py --model google/gemma-2-27b-it
python pipeline/3_vectors.py --activations-dir outputs/gemma-2-27b-it/activations
python pipeline/4_analysis.py --vectors-dir outputs/gemma-2-27b-it/vectors --layer 22
```

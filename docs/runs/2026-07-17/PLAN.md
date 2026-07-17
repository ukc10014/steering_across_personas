# Run plan — 2026-07-17: CAA vectors on the unmodified Llama-3.1-8B

**Run type:** first, narrow step toward the PREREG experiment (OCT × persona-trait geometry).
**Goal:** get the CAA branch of the K/D pipeline running on the plain
`meta-llama/Llama-3.1-8B-Instruct` (the "baseline" arm, no character training) and produce
CAA contrastive vectors for the 10 personas × 8 traits.
**Results file:** `docs/runs/2026-07-17/RESULTS.md` (to be written after the run).

## Context
[PREREG.md](../../../PREREG.md) registers whether OCT constitutional character training
changes persona-trait steering-vector geometry in the Karty/Davies (K/D) setup this repo
implements (originally built for `google/gemma-2-27b-it`). Before any LoRA/adapter work, we
first confirm the pipeline runs on the base Llama-3.1-8B and yields sane CAA vectors.

- **gemma-2-27b weights are NOT needed.** The only gemma dependency is the optional
  assistant-axis auto-download in [4_analysis.py:120](../../../pipeline/4_analysis.py#L120),
  which harmlessly skips for a non-gemma model.
- **goodness = flourishing** confirmed (resolves PREREG §7 ambiguity); OCT repo not needed now.

### Already on disk (verified) — nothing to download
- Base weights: `meta-llama/Llama-3.1-8B-Instruct` at
  `/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct` ✅
- CAA A/B datasets already generated: `data/prompts/caa/*.json` (all 8 traits) ✅ —
  **step 0c can be skipped**
- Persona YAMLs: 10 core personas + `null` + `nonsense` controls ✅
- HF cache at `/workspace/hf` (weights already pulled)
- LoRA adapters for later arms also present:
  `/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/` (goodness/loving/sarcasm + 7 more)

### Blockers to running (to be cleared by the user)
1. **GPU:** none currently visible (`/dev/nvidia*` absent, no `nvidia-smi`).
   `2c_caa_activations.py` needs CUDA — attach a GPU before extraction.
2. **Env:** system Python is 3.8 (repo needs ≥3.10); `torch` not installed;
   `assistant-axis-ref/` not cloned. `uv` is available to build a 3.10 env.

## How CAA runs (verified from code)
- [2c_caa_activations.py](../../../pipeline/2c_caa_activations.py) takes `--model <hf_id>` as a
  plain string → `ProbingModel(model)`, enumerates decoder layers at runtime (no hardcoded
  gemma layer count), and reads `pm.supports_system_prompt()`. Llama-3.1 has a real system
  role, so CAA uses the system-role branch (`[system, user, assistant]`). Forward pass only —
  no generation.
- Output: `outputs/Llama-3.1-8B-Instruct/caa_activations/{persona}_{trait}_{pos|neg}.pt`,
  each `(n_layers, hidden_dim)` ≈ (32, 4096).
- [3_vectors.py](../../../pipeline/3_vectors.py) computes `mean(pos) - mean(neg)` per
  persona×trait (drops last layer → ≈ (31, 4096)).
- Headline layer = **15** (PREREG: 32-layer Llama, depth-matched to K/D 22/46 ≈ 0.48).

## Steps

### 1. Environment (one-time, CPU)
```bash
cd /workspace/repos/steering_across_personas
uv venv --python 3.10 .venv && source .venv/bin/activate
export HF_HOME=/workspace/hf                      # so loads hit the cached weights
pip install -e .                                  # persona_steering
git clone https://github.com/safety-research/assistant-axis.git assistant-axis-ref
pip install -e assistant-axis-ref/                # brings torch (+ vllm, unused for CAA)
```
Confirm CUDA: `python -c "import torch; print(torch.cuda.is_available())"` → must be True.
(`.env` needs `HF_TOKEN`; `ANTHROPIC_API_KEY` not needed for CAA extraction.)

### 2. Smoke test (1 persona × 1 trait) — confirm assistant-axis handles Llama
```bash
python pipeline/2c_caa_activations.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --personas farmer --traits assertiveness --batch-size 4
```
Verify: model loads, `supports_system_prompt()` True, answer-token position found, saved
`.pt` shape ≈ (32, 4096). **This is the key unknown** — that the assistant-axis wrappers
enumerate Llama layers and map its chat template correctly (untested for Llama in this repo).

### 3. Full CAA extraction + vectors (GPU)
```bash
python pipeline/2c_caa_activations.py --model meta-llama/Llama-3.1-8B-Instruct
python pipeline/3_vectors.py \
    --activations-dir outputs/Llama-3.1-8B-Instruct/caa_activations \
    --output-dir     outputs/Llama-3.1-8B-Instruct/caa_vectors
```
Optional analysis: `python pipeline/4_analysis.py --vectors-dir .../caa_vectors
--output-dir .../caa_analysis --layer 15` (transfer matrix + clustering; gemma-only
axis-alignment sub-step auto-skips).

Equivalent one-liner:
`./run.sh meta-llama/Llama-3.1-8B-Instruct --caa --from 2 --to 4 --layer 15`

## Verification
- Smoke `.pt` shape ≈ (32, 4096); `caa_vectors/*.pt` present for all 10 personas × 8 traits.
- Spot-check a vector norm is finite and non-zero; `null` persona vectors exist (needed as
  the distance-to-null reference for the later PREREG analysis).

## Explicitly out of scope (later runs)
- LoRA merge / adapter arms (goodness, loving, sarcasm).
- IV branch (needs vLLM generation).
- Cross-arm PREREG dispersion analysis.

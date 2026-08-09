# Infrastructure & Codebase Handover

**Scope:** operational knowledge only — RunPod, environment, assets on disk, and how the
K/D codebase actually behaves. The experimental design is specified separately (see the
current spec document); **anything in this file about arms, traits, or hypotheses is
descriptive of past state, not a design commitment.**

---

## 1. RunPod operations

- Team account. GPU pods are ephemeral; a **persistent network volume mounted at
  `/workspace`** holds everything durable.
- Network volumes are **datacentre-scoped** — a new pod must be in the same DC to mount
  the volume. If the volume doesn't appear in the attach dropdown, that's why.
- **`/workspace` persists across pod stop, terminate, and recreation. Container disk does
  not.** Anything pip-installed to the system path is gone on restart.
- **The SSH IP and port change on every restart.** This is the #1 thing that looks like a
  broken pod but isn't.
- Use the **"SSH over exposed TCP"** string (`ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`),
  *not* the `ssh.runpod.io` proxy — the proxy has no SCP/SFTP support, which VSC
  Remote-SSH needs to install its server.
- Local `~/.ssh/config` block; update `HostName` and `Port` each session:
  ```
  Host runpod-kd
      HostName <ip>
      User root
      Port <port>
      IdentityFile ~/.ssh/id_ed25519
  ```
- **Workflow pattern:** use cheap CPU pods for setup, downloads, repo work, and reading
  configs; only rent GPU for forward passes. Terminate overnight (volume survives), stop
  for short breaks.
- **Pin one GPU template and reuse it.** Different base images ship different Python minor
  versions, which invalidates `--target` installs (see §2).

---

## 2. Environment bootstrap

`/workspace/bootstrap.sh` is the single entry point. After any pod start:

```bash
source /workspace/bootstrap.sh
```

It sets `HF_HOME=/workspace/hf`, `HF_HUB_DISABLE_XET=1`, `$SNAP`, `TMPDIR` and
`PIP_CACHE_DIR` (both on the volume, to keep pip off the small container disk), computes a
**python-version-scoped** `PYLIBS` dir (`/workspace/pylibs-py312` etc.), puts it on
`PATH`/`PYTHONPATH`, installs lite deps if missing, and re-adds its own `~/.bashrc` hook so
later terminals in the same container pick it up.

`/workspace/restart.sh` wraps this plus volume/repo/claude sanity checks.

### Hard-won rules

- **`PYLIBS` is python-version-scoped deliberately.** A `--target` install built on a 3.12
  image will not import on a 3.13 image. Symptom is a version-specific path in an
  ImportError. Fix: `rm -rf /workspace/pylibs-pyXYZ && source /workspace/bootstrap.sh`.
- **Never `pip install torch`.** It ships with the GPU image. Installing it pulls ~4GB of
  `nvidia_*` CUDA wheels onto container disk → `OSError: [Errno 28] No space left on
  device`. This has already happened once.
- **On CPU pods, don't install `transformers`/`accelerate` either** — they drag the same
  CUDA stack in as dependencies. CPU sessions need only `huggingface_hub`.
- GPU-pod deps, after confirming `import torch` works from the image:
  ```bash
  pip install --target=$PYLIBS -U "transformers>=4.45,<5" peft accelerate einops pandas matplotlib
  ```
  Note `peft` is **not** declared in the repo's `pyproject.toml` — install it explicitly.

- **Pin `transformers<5`.** An unbounded `>=4.45` resolves to the 5.x line (observed:
  5.14.1), which has breaking API changes relative to the 4.x the K/D codebase was written
  against. Check `pyproject.toml` before overriding this.
- **`pip --target` ignores the image's torch and installs its own.** `peft` declares
  `torch>=1.13.0`, so a `--target` install pulls a full duplicate torch (~526MB) plus the
  `nvidia_*` CUDA stack (~2GB) into `$PYLIBS`. On the volume this doesn't blow up (unlike
  container disk), but since `PYTHONPATH` puts `$PYLIBS` first, **`import torch` then
  resolves to the duplicate, not the image's build** — which is exactly where a
  Blackwell/`sm_120` kernel mismatch would hide. Either verify the duplicate works (see
  §11) or install with `--no-deps` and delete `$PYLIBS/torch*` and `$PYLIBS/nvidia*` so the
  image's torch wins.

---

## 3. HuggingFace tooling gotchas

- **`huggingface-cli` is dead.** It prints a deprecation warning and no-ops. Use `hf`.
  Subcommand verbs shifted in the v1 rewrite — run `hf cache --help` rather than assuming
  `scan`/`ls`/`list`. For plain verification, `du -shL` on the cache dir is more reliable.
- **Xet backend fails** on some repos with `RuntimeError: Unable to parse string as hex
  hash value`. `HF_HUB_DISABLE_XET=1` fixes it (already in bootstrap).
- **Llama-3.1-8B ships weights twice** (~32GB total): safetensors shards (~16GB, what
  `transformers` loads) plus `original/consolidated.00.pth` (Meta's raw format, useless
  here). Always `--exclude "original/*"`.
- If `HF_HOME` isn't live in the shell, downloads silently land in `~/.cache/huggingface`
  on **container disk** and vanish on restart. Check `echo $HF_HOME` before any download.
- **Naming trap: the HF org is `maius`; the GitHub user is `maiush`.** Easy to mistype.

---

## 4. Assets currently on the volume

```
/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct     ~16GB (safetensors only)
/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas       ~6.3GB (10 adapters)
/workspace/repos/steering_across_personas                        the fork
/workspace/repos/OpenCharacterTraining                           OCT source (constitutions)
```

`$SNAP` points at the OCT adapter snapshot dir. Subfolders (one adapter each):
`goodness, humor, impulsiveness, loving, mathematical, nonchalance, poeticism, remorse,
sarcasm, sycophancy`.

**Adapter facts (verified):**
- `base_model_name_or_path = meta-llama/Llama-3.1-8B-Instruct` ✓
- `r=64`, `lora_alpha=64`, all 7 attention+MLP projections (q,k,v,o,gate,up,down) — hence
  641MB each, which is large for an 8B LoRA
- Load form if using PEFT directly:
  `PeftModel.from_pretrained(m, "maius/llama-3.1-8b-it-personas", subfolder="goodness")`

**Naming/availability notes:**
- The paper's **`flourishing`** persona is released as **`goodness`**. Verified verbatim:
  the repo's `constitutions/few-shot/goodness` `"trait"` fields match Appendix F (p.38)
  word-for-word, all 15 assertions, same order. The `"questions"` fields are the per-
  assertion few-shot generation seeds (per OCT §2.3), not part of the constitution.
- **No `misalignment` adapter in the release.** Training data is published
  (`maius/OpenCharacterTraining-data-misalignment`) but not the weights.
- **Only merged adapters are released.** OCT trains in two stages (DPO distillation, then
  introspective SFT) and linearly merges before release. Stage-separated adapters would
  need to be requested from Maiya or retrained.

**Model geometry:** Llama-3.1-8B-Instruct = **32 layers, hidden 4096**. K/D's anchor is
Gemma-2-27B at layer 22 of 46 (depth 0.478) → depth-match ≈ layer 15 on Llama. The 32-layer
CAA literature independently lands at 13–15 (Tan et al. 2024 layer 13; Rimsky via BiPO
layer 15) — but note all of that optimises **steering efficacy**, which is not the same
objective as geometric extraction.

---

## 5. Repo and git

- Working fork: `https://github.com/ukc10014/steering_across_personas` (**origin** — commit
  here)
- Upstream: `https://github.com/jacobdaviescam/steering_across_personas` (pull only)
- `git remote add upstream <url>`; `git fetch upstream && git merge upstream/main` to sync.
- **Two CLAUDE.md files, deliberately:**
  - `repos/steering_across_personas/CLAUDE.md` — **upstream's**, accurate for the Gemma
    pipeline. Do **not** rewrite it; it's a merge-conflict surface on every upstream sync.
  - `/workspace/CLAUDE.md` — the fork's env/gotcha layer. Picked up automatically because
    CLAUDE.md discovery walks up the tree from the repo.
- `.env` (copy from `.env.example`) needs `ANTHROPIC_API_KEY` (LLM judge / data generation)
  and `HF_TOKEN` (gated models). `WANDB_API_KEY` optional, activates logging automatically.
- `assistant-axis-ref/` is **not vendored** — `run.sh:23-30` clones
  `safety-research/assistant-axis` at runtime.
- Useful shortcut: upstream publishes activations and vectors on HF at
  `girishgupta/persona-steering-activations` (v2: IV + CAA, 17 personas × 8 traits on
  gemma-2-27b-it). Analysis steps run CPU-only from these — no GPU needed to prototype
  analysis code.

---

## 6. Codebase landmines

Each of these cost real digging. All line refs are as of the last session.

1. **`3_vectors.py:106-115` slices `[:-1]`** ("float16 inf at final layer"). Saved vectors
   are `(n_layers-1, hidden)` while activations are `(n_layers, hidden)`. **`--layer N`
   indexes the truncated tensor.** Verify your intended layer against
   `ProbingModel.get_layers()` rather than trusting any paper's number.
2. **`TARGET_LAYER = 22` is hard-coded in two independent places** — `config.py:164` and
   `run.sh:89`. It is *not* rescaled per model. Running against Llama without `--layer`
   silently uses 22-of-32 (late-middle) where Gemma had 22-of-46 (mid). **Always pass
   `--layer` explicitly.**
3. **CAA contrastive datasets are model-agnostic.** `data/prompts/caa/*.json` store
   `{scenario, option_a, option_b, a_is_positive}` as plain text (~500/trait; empathy 499).
   The `(A)`/`(B)` labels and chat template are applied at runtime
   (`2c_caa_activations.py:72-74`, `apply_chat_template`). Reusable across model families
   as-is — no regeneration needed.
4. **The system-prompt branch differs by model family.** `2c_caa_activations.py:185-205`
   branches on `supports_system`: Gemma-2 (no system role) concatenates the persona into
   the user turn; Llama-3.1 takes the real-system-role branch. This is a **structural
   non-equivalence between model families**, not a data problem — fine for within-family
   comparisons, but it's a confound whenever a cross-family result is being interpreted as
   a replication.
5. **Answer-token indexing is the highest silent-failure risk.**
   `find_answer_token_position()` (`2c_caa_activations.py:89-151`) does prefix-diff plus a
   reverse scan for the letter token, with a left-pad offset — it is *not* `[-1]`. The
   reverse scan **substring-matches a bare "A"/"B"**, so a non-special trailing token
   containing a capital A or B can match first. **Before any full run on a new model
   family, print the decoded token at the extraction index for ~5 examples.** If this is
   wrong, every downstream cosine is noise and nothing else will look broken.
6. **No PEFT anywhere in the CAA path.** `peft` appears only in `pipeline/10_oracle.py`,
   which is hardwired to Gemma-27B, uses raw `AutoModelForCausalLM` rather than
   `ProbingModel`, and shares nothing with CAA. Don't borrow its PEFT-nesting fallbacks.
   **Adapter handling decision from last session: merge offline, pass the merged path as
   `--model`.** This means zero changes to the extraction path, hooks attach to plain
   `LlamaDecoderLayer` with no wrapper ambiguity, and each arm gets its own output dir for
   free (dirs derive from `model.split("/")[-1]`). Costs ~16GB/arm on the volume.
7. **`ModelConfig` presets are dead code for CAA.** Geometry is runtime-discovered
   (`ProbingModel` → `get_layers()`, `hidden_size`). The model is a raw `--model` string.
   Adding a preset would be cosmetic.
8. **There is no `--method` flag.** Steps `0c_`/`2c_` are CAA; steps 3+ are shared. Method
   is selected purely by which `--activations-dir` / `--vectors-dir` you pass
   (`activations/` vs `caa_activations/`). `wandb_utils.infer_method()` recovers it by
   substring-matching `"caa"` anywhere in the path — fragile, and tagging-only.
9. **`run_all_experiments.sh` is a hard-coded one-off** (`cd /workspace/steering_across_personas`
   — wrong path for this checkout). It fails immediately. Ignore it.
10. **No test suite exists.** No `tests/`, no `test_*.py`, no `conftest.py`. `pytest` is a
    declared dev-dep referenced nowhere. CI (`.github/workflows/ci.yml`) runs only
    `ruff check --select E9,F63,F7,F82` (laxer than the project's own ruff config) +
    `compileall` + a smoke-import asserting 8 traits and non-empty personas. No
    typechecker. Don't let a fresh agent hunt for tests.

**Commands:** `pip install -e ".[dev]"`; `ruff check .` / `ruff format .`;
`./run.sh <model> --caa --layer L`; `--from N` / `--to N` to resume — but note `--from 3`
resumes *both* IV and CAA (duplicate step numbers across the two tracks).

---

## 7. Data artifacts worth knowing about

- **Personas:** `data/personas/` — 37 YAMLs, each with 5 `system_prompt_variants`
  (this is what the paraphrase-floor control uses). `null` and `nonsense` are ordinary
  YAML files, not special-cased.
- **NONSENSE control is not what the paper describes.** K/D's text says "length-matched
  random tokens"; the released `data/personas/nonsense.yaml` is 5 hand-written gibberish
  paragraphs, mean ~19.2 words vs 36–40 for real personas — roughly half length, and not
  random tokens. There's no generating code. Most likely the released YAML isn't the
  artifact that produced the figures. **If NONSENSE is load-bearing for a given analysis,
  regenerate it length-matched first**; worth asking Jacob whether the released file is
  stale.
- **`outputs/Llama-3.1-8B-Instruct/analysis/` exists in the repo** from a previous session.
  Contents unverified in this handover — inspect locally (`ls -R outputs/`, `git log
  --oneline`) to establish which arm/traits/personas produced it and whether the
  answer-token check was ever run.

---

## 8. Claude Code on the pod

- VSC extension is the current setup; it runs against whichever host VSC is connected to
  (i.e. the pod, under Remote-SSH).
- `CLAUDE_CONFIG_DIR=/workspace/.claude` keeps the OAuth token on the volume so restarts
  don't force re-auth.
- If installing the CLI pod-side:
  `npm install -g @anthropic-ai/claude-code --prefix /workspace/npm-global`, and put
  `/workspace/npm-global/bin` on PATH via bootstrap.
- **Never export `ANTHROPIC_API_KEY` in a shell Claude Code inherits** — it silently
  overrides OAuth and bills API credits instead of the subscription. Verify with `/status`
  inside a session. (Note the tension with §5: the *repo's* `.env` legitimately needs that
  key for the LLM judge. Keep it in `.env`, loaded by the pipeline, not exported globally.)

---

## 9. Not yet provisioned — implied by the current spec

Infrastructure consequences only; the design rationale lives in the spec doc.

- **Gemma-3-4B** weights and the **Gemma-3-4B OCT adapters** (separate HF repo from the
  Llama personas repo). Gemma is gated — accept the licence on the HF account in advance.
- Gemma-3, like Gemma-2, has **no native system role**, so it takes the concatenation
  branch at `2c_caa_activations.py:185-205` — closer to K/D's original format than Llama is.
- **IV extraction requires generation**, not just forward passes. This is a materially
  different compute profile from CAA (which needs only answer-token activations) and adds
  a dependency on the trait judge, hence `ANTHROPIC_API_KEY` and per-token cost.
- **SAE availability: resolved, no blocker.** Gemma Scope **2** (Dec 2025) covers the full
  Gemma 3 family — SAEs *and* transcoders on every layer, for 270M / 1B / **4B** / 12B /
  27B, on both pretrained and instruction-tuned variants. (Gemma Scope 1 remains the
  Gemma-2 suite.) So Gemma-3-4B is a valid Exp 3 target. Two follow-ups when you get there:
  (a) confirm whether the released SAEs sit on the **residual stream** vs MLP output at each
  layer — transcoders are described as operating on MLP output, so the two artifacts likely
  target different sites, and Exp 3a is specified against the residual stream to match K/D's
  extraction site; (b) the "dark matter" reconstruction-failure rates in the literature are
  from Gemma Scope 1-era JumpReLU SAEs — GS2 uses Matryoshka training, so don't anchor the
  Exp 3a threshold on the older numbers.
- **MASK dataset** download and judge configuration for the behavioural arm.
- **Misaligned adapter** requires approval from Maiya et al. — not in the public release.
- **Stage-separated (DPO-only vs DPO+SFT) adapters** likewise require a request or
  retraining; the release is merged.

---

## 10. Stale artifacts to clean up

`PREREG.md` in the repo, `/workspace/HANDOVER.md`, and possibly `/workspace/CLAUDE.md` all
describe the **previous** three-arm design (goodness / loving / sarcasm) and its
hypotheses. That design is superseded by the current spec document. Update or remove them
before pointing a fresh agent at the repo — otherwise it will read the old prereg as
authoritative and quietly optimise for the wrong contrasts.

---

## 11. Known-good pod configuration (observed 2026-08-09)

Reference values from a working session. Compare against these on any new pod.

| | value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell, **97,887 MiB (~96GB) VRAM** |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| Python | **3.12.3** → `PYLIBS = /workspace/pylibs-py312` |
| `HF_HOME` | `/workspace/hf` ✓ |
| Llama-3.1-8B-Instruct | ~15–16GB on volume ✓ |
| OCT Llama personas (10 adapters) | 6.3GB on volume ✓ |

**Python 3.12 is the version to stay on.** The existing `pylibs-py312` is reusable as long
as new pods land on 3.12.x. A 3.13 image forces a full rebuild of the `--target` installs
(this happened once on a CPU pod). When choosing a template, prefer whichever one yields
3.12.

**~96GB VRAM is much more headroom than this work needs.** Llama-3.1-8B bf16 is ~16GB and
CAA is forward-pass-only. Practical consequences: (a) merged-adapter arms can be held
resident without juggling; (b) **Gemma-2-27B-IT (~54GB bf16) fits**, so the "run K/D's exact
model, exact layer, unmodified pipeline" ground-truth rung is available on this pod if a
cross-family discrepancy ever needs adjudicating; (c) you're overpaying for CPU-only setup
work — do downloads and repo work on a cheap CPU pod.

### Blackwell caveat — verify before trusting any run

RTX PRO 6000 Blackwell is `sm_120`, and CUDA 13.0 is very new. **PyTorch builds predating
Blackwell support will report `torch.cuda.is_available() == True` and then fail at kernel
launch** with something like `no kernel image is available for execution on the device`, or
produce silently wrong results. Always run the real check, not just the availability flag:

```bash
python -c "
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda)
print('capability', torch.cuda.get_device_capability(0))   # expect (12, 0)
print('arch list', torch.cuda.get_arch_list())             # must include sm_120
x = torch.randn(1000, 1000, device='cuda', dtype=torch.bfloat16)
print('matmul ok:', (x @ x).sum().item())                  # the actual test
"
```

If `sm_120` is missing from the arch list or the matmul throws, the image's torch is too
old — pick a newer RunPod template rather than trying to pip-install a fix (that pulls the
whole CUDA wheel stack onto container disk; see §2).
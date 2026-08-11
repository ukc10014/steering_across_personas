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

---

## 12. Work plan — B.1 follow-through (opened 2026-08-10)

Context: the B.1 noise-floor re-analysis is **done** and written up in
[docs/results/llama31_8b_b1_noise_floor.md](results/llama31_8b_b1_noise_floor.md). This
section covers the two follow-on steps that need GPU, and the state of the pod they were
scoped on. Read the writeup first — it says *why* these two steps are the next ones.

### 12.1 Status

| Step | What | State |
|---|---|---|
| 1 | K/D's rung-1 estimator (within-cell bootstrap) | **done** — `scripts/caa_within_cell_stability.py` |
| 2 | Split-half question-bank floor | **done** — same script |
| 3 | Paraphrase arm (5 system-prompt variants) — supplies K/D's missing **rung 2** | **not started**, needs GPU |
| 4 | Variance decomposition (within / paraphrase / persona), ICC-style with a CI | **blocked on 3**, CPU-only once 3 lands |

Step 4 is blocked on step 3 by construction: the across-paraphrase component of the
decomposition *is* rung 2. Don't try to build the estimator first.

### 12.2 This pod is NOT the pod in §11

Observed 2026-08-10 — compare before assuming §11's numbers apply:

| | §11 (known-good) | this pod |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell, 96GB | **RTX 3090, 24GB**, sm_86 |
| torch | (image, cu13) | **2.4.1+cu124**, from image, arch list has sm_86 |
| Python | 3.12.3 → `pylibs-py312` | **3.11.10** → `pylibs-py311` |

`pylibs-py311` was **effectively empty** (torch + numpy only). Provisioned this session with:

```bash
pip install --target="$PYLIBS" --no-deps python-dotenv
pip install --target="$PYLIBS" --no-deps matplotlib contourpy cycler fonttools \
    kiwisolver packaging pillow pyparsing python-dateutil six
pip install --target="$PYLIBS" "transformers>=4.45,<5" scikit-learn plotly tqdm
pip install --target="$PYLIBS" --no-deps accelerate psutil     # accelerate declares torch
```

Resulting versions: `transformers 4.57.6`, `accelerate 1.14.0`, `matplotlib 3.11.1`.
**No duplicate torch and no `nvidia_*` wheels landed in `$PYLIBS`** — verified, the §2
landmine was avoided by holding `accelerate` to `--no-deps`.

**Two gotchas found the hard way:**

1. **`pip --target` DID shadow numpy** — it installed `numpy 2.4.6` into `$PYLIBS` over the
   image's 1.26.3, despite 1.26.3 being importable. torch 2.4.1 ↔ numpy 2.4.6 interop was
   tested and works (`.numpy()` / `from_numpy()` round-trip), so it was left in place. Be
   aware the analysis earlier in the session ran under 1.26.3; RNG (`default_rng`/PCG64) is
   version-stable per NEP 19, so the stored JSON is unaffected.
2. **`HF_HUB_OFFLINE=1` is REQUIRED on this transformers version.** 4.57.6 calls
   `list_repo_templates()` during tokenizer load, which requests
   `.../tree/main/additional_chat_templates`, gets a 404, and raises instead of degrading.
   The weights are fully cached, so offline mode is the correct fix. Without it the run dies
   ~15s in, before touching the GPU, with a `RemoteEntryNotFoundError` that looks like a
   gated-model auth problem and is not one. Consider adding it to `bootstrap.sh`.

### 12.3 GPU verdict: the 3090 is sufficient — verified end-to-end

Not inferred from arithmetic. A real single-cell extraction was run
(`--personas farmer --traits warmth`), and it worked:

| measurement | value |
|---|---|
| VRAM, model resident, `--batch-size 16` | **17,516 MiB / 24,576 MiB** (~7GB headroom) |
| model load | 37s |
| throughput | **25.9 s per 500-question file** |
| output | 131 MB, 500 keys, correct shape |

**Correctness check against the committed activations:** the re-extracted
`farmer_warmth_{pos,neg}.pt` are *not* bit-identical to the existing files (different GPU,
different kernel reduction order in fp16), but per-layer cosine of the mean activation is
**0.999993 or better at every one of the 32 layers**; mean absolute elementwise difference
0.0036. The pipeline reproduces on this hardware.

**Consequence for step 4, and it is a real one:** variant 0 already on disk was extracted on
*different* hardware from anything extracted here, so a within/paraphrase/persona
decomposition would in principle confound "across-paraphrase" with "across-GPU". The size
of that confound is now measured: hardware contributes ~7e-6 of angular decorrelation
against ~0.16 from question resampling — **five orders of magnitude smaller, ignorable**.
Re-extracting variant 0 for hygiene would cost +160 files / +21GB / +1.2h and is *not*
worth it on these numbers. State the check in the writeup rather than paying for it.

### 12.3b Volume quota — this killed a run, read before sizing any job

**`df` is useless on `/workspace` and will actively mislead you.** The volume is MooseFS
(`mfs#eu-cz-1.runpod.net:9421`), and `df` reports the **whole cluster** — it showed
`851T total, 166T avail` throughout, while the actual per-volume quota was ~100GB and
about to be exceeded. Capacity planning off `df` is planning off a number that has nothing
to do with your limit. That mistake is what killed the first paraphrase run.

**Symptoms of hitting the quota, none of which say "disk":**

- the Bash tool starts returning exit 1, or exits 0 with **completely empty stdout** —
  even `echo hello`. Looks like a broken harness or a dead shell.
- file writes fail with `EDQUOT: unknown error, fsync`. Note **EDQUOT**, not `ENOSPC` —
  it is a *quota*, not a full disk.
- background jobs die and their wrapper reports a nonzero exit while the child may still
  be running for a while.
- **the log truncates mid-line with no traceback**, because the log could not be appended
  to either. There is no error message to find; the absence of one IS the signal.

**Getting the real number.** `mfsgetquota` is not installed on these pods, so there is no
in-pod way to read the quota. Use the RunPod console/API, or infer it from `du`:

```bash
du -sh /workspace/hf /workspace/pylibs-py* /workspace/repos
```

**Resizing** (increase only, never decrease; >4TB needs support):

```bash
curl -X POST https://rest.runpod.io/v1/networkvolumes/{networkVolumeId}/update \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"size": 500}'          # GB, must exceed current
```

Expanded to **500GB on 2026-08-10**. Baseline occupancy after that:
`hf` 22GB, `pylibs-py312` 11GB, `pylibs-py311` 3.2GB, `repos` 53GB (of which
`caa_activations` 24GB) — call it ~90GB, leaving ~410GB.

### 12.4 Cost of step 3

Scope: variants 1–4 (variant 0 is the existing `caa_activations/`), 10 personas × 8 traits ×
2 directions = **640 files**, 319,920 forward passes.

| | per file | step 3 total |
|---|---|---|
| this 3090 | 25.9 s | **~4.6 hours** |
| §11 Blackwell (implied by the 192-file/15m11s original run) | 4.7 s | **~50 minutes** |

Storage: **~84 GB** on the volume (166TB free — not a constraint).

**Decision (2026-08-10): step 3 runs on the 96GB pod.** ~5.5× faster for an identical
result, and the arm is a single unattended job. The 3090 remains a verified working fallback
— nothing about the result changes there, only the wall clock. Migration checklist in §12.8.

### 12.5 Step 3 needs a code change — it is not just a re-run

`2c_caa_activations.py` has **no `--variant` flag**. Line 369 passes
`persona.default_system_prompt`, which `config.py:99-101` hard-codes to
`system_prompt_variants[0]`. All 37 persona YAMLs carry 5 variants; four of them have never
been used by any extraction.

Minimal change, three edits:

1. Add `--variant N` (int, default 0) to `parse_args()`.
2. `persona_system_prompt=persona.system_prompt_variants[args.variant]` at line 369
   (bounds-check it — `default_system_prompt` silently returns `""` for a persona with no
   variants, and a silent empty system prompt is exactly the `null` condition, which would
   look like a result rather than a bug).
3. Output naming — **use a separate directory**, `caa_activations_paraphrase/`, with a
   `{persona}_v{N}_{trait}_{dir}.pt` infix.

On (3): do **not** write variants into the existing `caa_activations/`. The analysis scripts
discover cells by globbing that directory, so `farmer_v1` would be picked up as an
additional *persona* and silently inflate rung 3 in
`caa_within_cell_stability.py` and the fan-out figures. Symlink variant 0 into the new
directory rather than re-extracting it (see 12.3 — the hardware delta is ignorable):

Do this in Python, not shell. Persona slugs contain underscores (`con_artist`,
`drill_sergeant`, `kindergarten_teacher`), so the obvious `${b%%_*}` / `${b/_/_v0_}` pair
splits on the wrong separator and produces `con_v0_artist_warmth_pos.pt`. Anchor on the
known persona and trait lists instead:

```python
from pathlib import Path
from persona_steering.config import PERSONA_SLUGS

src = Path("outputs/Llama-3.1-8B-Instruct/caa_activations")
dst = Path("outputs/Llama-3.1-8B-Instruct/caa_activations_paraphrase")
dst.mkdir(parents=True, exist_ok=True)
TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty",
          "confidence", "deference", "warmth", "impulsivity"]

for p in (s for s in PERSONA_SLUGS if s not in ("null", "nonsense")):
    for t in TRAITS:
        for d in ("pos", "neg"):
            link = dst / f"{p}_v0_{t}_{d}.pt"
            if not link.exists():
                link.symlink_to(Path("..") / "caa_activations" / f"{p}_{t}_{d}.pt")
```

Produces 160 links. Confirm one resolves before relying on it —
`torch.load(dst / "con_artist_v0_warmth_pos.pt")` should return 500 keys.

`null` is excluded deliberately: its system prompt is `""`, so it has no paraphrases and
`system_prompt_variants[1..4]` is meaningless for it. `nonsense` *does* have 5 variants, but
see §7 — the released `nonsense.yaml` is probably not the artifact that produced K/D's
figures, so treat any nonsense paraphrase result as provisional.

### 12.6 Run sheet

```bash
source /workspace/bootstrap.sh
export HF_HUB_OFFLINE=1                      # REQUIRED, see 12.2

# always dry-run first: confirms file count and forward passes before a multi-hour job
python pipeline/2c_caa_activations.py --model meta-llama/Llama-3.1-8B-Instruct \
    --variant 1 --output-dir outputs/Llama-3.1-8B-Instruct/caa_activations_paraphrase \
    --personas farmer politician therapist drill_sergeant street_hustler professor \
               tech_ceo kindergarten_teacher surgeon con_artist --dry-run

# then variants 1..4; 2c skips files that already exist, so it is resumable
```

**The §6.5 answer-token check has now been run across all five variants and PASSES** —
180/180 examples, 3 personas × 2 traits × 2 directions × 5 variants, every index landing on
a bare `A`/`B` token. `verify_answer_token.py` gained a `--variants` flag for this; it
previously only ever tested `default_system_prompt`, i.e. variant 0, so it could not have
caught a variant-induced shift:

```bash
python scripts/verify_answer_token.py --model meta-llama/Llama-3.1-8B-Instruct \
    --traits warmth deference --personas farmer con_artist null --variants 0 1 2 3 4 --n 3
```

Re-run this whenever the persona set or model family changes. It is the highest
silent-failure risk in the pipeline — a wrong index makes every downstream cosine noise
while nothing else looks broken.

### 12.7b Resume-after-crash is existence-based, not validity-based — verify before analysing

`2c_caa_activations.py:322` filters the work list with `if not o.exists()`. A file that
exists but is **truncated or zero-length** is skipped forever. When the first paraphrase run
died on the quota mid-`torch.save`, it left
`street_hustler_v2_warmth_neg.pt` at **0 bytes**; the resumed run skipped it, all 800 files
looked present, and the corruption only surfaced hours later as an `EOFError` deep inside the
decomposition.

**After any crashed/resumed extraction, size-check before trusting the grid.** Valid files
cluster tightly (~131.0–131.2 MB for 500 questions, ~131.0 MB for empathy's 499); anything
outside that is suspect:

```bash
find outputs/{model}/caa_activations* -type f -name '*.pt' -size -130000000c -printf '%s  %p\n'
```

Delete what it finds and re-run `2c` for just that persona/trait — it refills only the gap.
Note the log is no help here: under quota the log could not be appended either, so the last
"Saved" line was never written and the log's final entry names the *wrong* file.

### 12.7 Step 4 once step 3 lands

CPU-only, no GPU, minutes. Partition total cosine-distance variance into
within-cell / across-paraphrase / across-persona components, ICC-style, and report the ratio
with a CI. `scripts/caa_within_cell_stability.py` already computes the within-cell term and
has the weighted-mean matmul trick that makes the resampling cheap (`weighted_vectors()`) —
extend it rather than starting a new script.

The reason this matters, restated from the writeup: H7-style claims are about *differences in
dispersion between constitutions*, which needs an error bar on a dispersion statistic. Right
now we have point estimates compared against a floor, which is suggestive, not a test.

### 12.8 Migrating to the 96GB pod — do these in order

Everything in the repo is ready; the blocker is that **`pylibs-py312` is in the exact broken
state §2 warns about** and will fail before it reaches the GPU.

**1. Confirm you actually landed on Python 3.12.** If the new pod is 3.11 or 3.13 you get a
different (empty) `PYLIBS` and none of the below applies — reprovision per §12.2 instead.

```bash
source /workspace/bootstrap.sh && python3 -V && echo $PYLIBS
```

**2. Fix `transformers` — this is the blocker.** `pylibs-py312` currently has **5.14.1**,
which §2 flags as API-incompatible with the 4.x this codebase targets:

```bash
pip install --target="$PYLIBS" --upgrade "transformers>=4.45,<5"
python -c "import transformers; print(transformers.__version__)"   # expect 4.5x
```

> **Done 2026-08-10:** 5.14.1 → **4.57.6**. Clean — pip pulled only pure-Python deps plus
> `tokenizers 0.22.2`; no duplicate torch, no `nvidia_*` wheels, torch byte-identical
> afterwards. `numpy` nudged 2.5.1 → 2.5.2, harmless. `preflight.sh` then printed
> `PREFLIGHT OK` and the 5-variant answer-token check passed 180/180 on this pod.

**3. Verify which torch wins, and that it speaks sm_120.** `pylibs-py312` carries a full
duplicate `torch 2.13.0+cu130`, and `PYTHONPATH` puts `$PYLIBS` first, so **that duplicate
shadows the image's build** — this is the §2 landmine, and on Blackwell it is also the §11
`sm_120` question. Run the §11 block verbatim; `get_arch_list()` must include `sm_120` and
the bf16 matmul must actually execute. If it fails, delete `$PYLIBS/torch*` and
`$PYLIBS/nvidia*` so the image's torch wins, then re-check.

> **Observed 2026-08-10 on RTX PRO 6000 Blackwell Server Edition (driver 580.142):** the
> `$PYLIBS` duplicate **passes** — capability `(12, 0)`, arch list
> `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`, bf16 matmul executes, 95.0 GiB
> visible. **Do not delete `$PYLIBS/torch*` on this pod** — the deletion step is
> conditional and the condition did not fire. Note this is a *different* Blackwell SKU and
> driver from §11's (580.173.02); re-run the check rather than assuming.

**4. Clean the duplicate installs.** `pylibs-py312` has two `peft` versions (0.19.1, 0.20.0)
and two `pandas` (3.0.3, 3.0.5). Not currently load-bearing for CAA — `peft` is only needed
for the Stage-2 adapter merge — but resolve them before the adapted-model arm.

**5. Export the offline flag.** Required on transformers ≥4.5x, see §12.2:

```bash
export HF_HUB_OFFLINE=1
```

**6. Re-run the two cheap guards before the long job.** Both are fast and both have caught
real problems:

```bash
bash scripts/preflight.sh          # must print PREFLIGHT OK
python scripts/verify_answer_token.py --model meta-llama/Llama-3.1-8B-Instruct \
    --traits warmth deference --personas farmer con_artist null --variants 0 1 2 3 4 --n 3
```

**7. Then the run sheet in §12.6.** Dry-run first, then variants 1–4. `2c` skips existing
files, so the job is resumable and safe to interrupt.

**Not needed on the new pod:** nothing in the analysis path. Steps 1–2 and both figures are
already computed and committed under `outputs/*/analysis/`, which is the one part of
`outputs/` that is not gitignored. The 24GB of `caa_activations/` lives on the volume and is
mounted wherever the pod is, so no data moves.
---

## 13. Adapted-model arm (OCT constitution) — plan, not yet started

**The question.** Does character training move the *extraction noise floor*? Every
dispersion claim comparing a constitution against baseline — H7-style "personas tighten
under constitution X" — is confounded if the floor itself shifts between arms. Rung 1 on the
adapted model is the control that de-confounds it. This is a prerequisite for the
comparison, not an optional extra.

### 13.1 Scope: one arm, base grid only

- **One adapter to start: `goodness`.** It is the paper's `flourishing` persona (§4, verified
  verbatim against Appendix F) and its `adapter_config.json` records the correct base. A
  single adapted arm against the existing baseline answers the floor question. Add contrast
  arms (`sarcasm`, `loving`) only if the floor actually moves.
- **Base grid only — no paraphrase arm.** The floor question needs rung 1 (within-cell
  bootstrap), which needs only the 12-series grid. Rung 2 on the adapted model would add
  84GB and ~50 min for a question nobody is asking yet.
- **Two release limits to plan around** (§4): there is **no `misalignment` adapter** — the
  arm most interesting for a safety framing needs weights requested from Maiya et al.; and
  only **merged** (DPO+SFT) adapters are published, so stage-separated comparisons are
  unavailable without retraining.

### 13.2 Cost, from measurements taken 2026-08-10 on the Blackwell pod

| stage | time | disk |
|---|---|---|
| merge adapter → standalone checkpoint | ~5–10 min (CPU) | **16 GB** |
| CAA extraction, 192 files / 95,976 forwards | **~16–22 min** (5–7 s/file) | **25 GB** |
| analysis (cosine-to-null, within-cell stability) | minutes, CPU | negligible |
| **per arm** | **~30 min** | **~41 GB** |

Volume is 500GB with ~145GB used, so ~8 arms would fit. Adding the paraphrase arm to any
one of them costs a further 84GB.

### 13.3 Procedure

**1. Clean the duplicate `peft`** (§12.8 step 4) — `pylibs-py312` has both 0.19.1 and 0.20.0
dist-info; it currently resolves to 0.20.0. `merge_lora.py` is the only thing in the repo
that imports peft, so this is exactly where an ambiguous install would bite.

**2. Merge.** `scripts/merge_lora.py` is sound — it refuses to merge if the adapter's
recorded base does not match `--base`, and it proves the merge changed weights (a silent
no-op merge would yield a "character-trained" model identical to baseline, and every
downstream comparison would read as *character training does nothing*). Watch for the
`max|Δw|` line; it must be non-zero.

```bash
source /workspace/bootstrap.sh && export HF_HUB_OFFLINE=1
python scripts/merge_lora.py \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --adapter "$SNAP/goodness" \
    --out /workspace/merged/llama-3.1-8b-goodness
```

**NO TRAILING SLASH on `--out`, and none on `--model` later.** `model_short_name()`
(`utils.py:99-101`) is just `model.split("/")[-1]`, so a trailing slash returns `""` and every
output lands in `outputs//caa_activations/` — silently the wrong place. Shell tab-completion
appends slashes to directories, so this will happen unless you watch for it.

**3. Verify the answer token on the merged model** (§6.5). Merging changes weights, not the
tokenizer, and the OCT adapter's `chat_template.jinja` was checked on 2026-08-10 and is
**byte-identical to base Llama-3.1's** (same md5), so `merge_lora.py` saving the *base*
tokenizer introduces no prompting mismatch. Run it anyway — it is 30 seconds:

```bash
python scripts/verify_answer_token.py --model /workspace/merged/llama-3.1-8b-goodness \
    --traits warmth deference --personas farmer con_artist null --variants 0 --n 3
```

**4. Extract.** No pipeline changes — merging offline keeps `2c` model-agnostic (§6.6):

```bash
python pipeline/2c_caa_activations.py --model /workspace/merged/llama-3.1-8b-goodness \
    --batch-size 16          # -> outputs/llama-3.1-8b-goodness/caa_activations/
```

**5. Analyse with flags matched to the baseline arm.**

```bash
python scripts/caa_cosine_to_null.py --model /workspace/merged/llama-3.1-8b-goodness \
    --traits assertiveness empathy risk_taking honesty confidence deference warmth impulsivity \
    --personas farmer politician therapist drill_sergeant street_hustler professor \
               tech_ceo kindergarten_teacher surgeon con_artist nonsense \
    --headline-layer 20 --n-boot 400 --seed 0
python scripts/caa_within_cell_stability.py --model /workspace/merged/llama-3.1-8b-goodness \
    --n-boot 50 --n-splits 100
```

### 13.4 The methodological point that decides whether this works

**`--n-boot` and `--seed` must match across arms, and 50 is not enough.** The B.1 work found
the floor estimate swings **0.886–0.908 at L15 on seed choice alone** at `n_boot=50`
(docs/results/llama31_8b_b1_noise_floor.md, finding 4). A base-vs-adapted floor difference
smaller than ~0.02 would be indistinguishable from that scatter. So **loose end 1 of the B.1
work is a hard prerequisite here**: re-run the baseline floor at `--n-boot 400` before
comparing anything to it. The two jobs are linked; do not treat them as independent.

Everything else is already controlled: the CAA datasets are model-agnostic plain text
(§6.3), so both arms see identical questions, personas and prompts, and Llama-3.1 takes the
real-system-role branch in both.

### 13.5 What the result looks like

A per-layer plot of rung 1, base vs adapted, with the CI band. Two readings:

- **floors coincide** → the floor is a property of the extraction, not the constitution;
  dispersion comparisons between arms are licensed, and the confound is retired.
- **floors separate** → any observed "tightening" under a constitution is partly a moving
  floor, and every downstream dispersion claim must be expressed relative to its own arm's
  floor rather than an absolute number.

The second outcome is the more interesting one and is the reason to run this before, not
after, the dispersion experiments it would invalidate.

---

## 14. Next session — start here

**Repo state at handoff:** `main` clean and pushed. Three character-training arms are done and
the headline result **did not survive its control** — read
[docs/results/llama31_8b_character_arms.md](results/llama31_8b_character_arms.md) first; it is
self-contained and carries every number.

**The one-line summary.** Merging any of the three OCT LoRA adapters compresses persona-conditional
trait vectors toward the model's own default by about the same amount (L15 shift: `goodness`
+0.183, `mathematical` +0.171, `impulsiveness` +0.191, against a base of 0.723). `mathematical`
is the orthogonal control, so the compression is a property of merging an r=64 LoRA, **not** of
what the constitution says. The specificity test also failed: the `impulsiveness` arm moves
`impulsivity` 4th of 8 traits.

**What is closed:**
- **Loose end 1** (baseline floor at `--n-boot 400`) — done, point estimates moved 0.000e+00.
- §13's adapted-model arm — executed, three arms, all guards passed (non-zero `max|Δw|`, 36/36
  answer-token, 192/192 files with no truncation).
- The §13.5 licensing question — **licensed at L15** for all three arms (Δwobble ≤ 0.011 against
  a ±0.02 threshold), not at L20 for two of them.

**Next task, and it is cheap:** find out *why* LoRA merging compresses persona conditioning.
§6 of the arms doc lists the ladder; the top rung is the informative one — **merge a
randomly-initialised r=64 adapter** with the same target modules and scale and extract it. If a
random adapter compresses too, this is about perturbation magnitude and has nothing to do with
learned content. ~25 min GPU, no training needed.

**Tooling built this session, use it:**
- `scripts/run_arm.sh <adapter> [--gpu-only|--analysis-only] [--terminate]` — one command per arm,
  with every guard that has caught a real failure wired in as a hard stop. Split at the GPU
  boundary so the card is held for ~25 min, not the whole session.
- `scripts/plot_arm_comparison.py --adapted <arm...> --label <...> --tag <name>` — N arms on one
  figure, plus the wobble/licensing readout.
- `caa_cosine_to_null.py` is ~14x faster (bootstrap as one GEMM); verified to reproduce prior
  results to 3e-07. Per-arm analysis is now ~6 min, not ~90.
- `/workspace/bootstrap.sh` is mirrored at `scripts/bootstrap.sh`, byte-identical, so a rebuilt
  volume can be restored with `cp scripts/bootstrap.sh /workspace/bootstrap.sh`.

**Still open elsewhere:**
- **Loose end 2:** no `nonsense` paraphrases; the released `nonsense.yaml` is ~half the length of
  a real persona (§7).
- **IV noise floor** (`docs/results/llama31_8b_iv_extraction.md` §4.4) — the highest-value missing
  number in the IV work; the question-resampling bootstrap cannot see generation-seed variance.
- **`risk_taking` is broken in both extraction methods** and should be excluded or rewritten — see
  the IV doc §4.3b and `iv_extraction_audit.md` §6.3.
- **Steering has never been run on Llama at all**, so K/D's layer-selection criterion (i),
  behavioural lift under self-steering, is untested here.

**The K/D paper is on disk** at `/workspace/refs/personas_shape_how_models_represent_behaviors.pdf`
(22 pages, deliberately outside the repo — the fork is public). `pypdf` is in `pylibs-py312`.
Appendix B is pp. 6-9; Appendix G is p. 21.

**Layer note:** L15 is depth-matched to K/D's layer 22 of 46 (0.469 vs 0.478) and is the layer to
quote when comparing to the paper. Leading with L20 understates our CAA replication.


## 15. IV replication (K/D Appendix G) — scoping

Prereg Exp 0a asks for both CAA and IV. Everything to date is CAA. Appendix G is the IV
replication, and it splits into two very different pieces.

### 15.1 What IV actually is

Same contrastive average as CAA, different contrast. Per (persona, trait) cell: 5 trait-positive
instructions paired with 5 matched suppression instructions, 20 sampled questions, and for each
(instruction, question) the model **generates a response** under the persona's system prompt.
The vector is mean(assistant-turn activations | positive instructions) − mean(… | negative).
M = 5 × 20 = **100 pairs per cell**.

The load-bearing difference from CAA: **IV requires generation, not just a forward pass.** CAA
reads one answer token from a multiple-choice prompt; IV needs free-form text first, then a
second pass to extract activations over it.

### 15.2 G.1 is tractable; G.2 is a separate project

- **G.1 — IV per-trait cosine spread (their Figure 20).** The direct IV counterpart of
  everything we have built for CAA, and reusable by every existing analysis script. This is
  the one to do.
- **G.2 — IV cross-context probe transfer (their Figure 21).** Per-trait 10×10
  within→cross-context AUROC matrices, n=1056 cells. Needs probe machinery that does not exist
  for Llama yet. Treat as its own piece of work, not part of "doing IV".

### 15.3 Cost, and the one real blocker

**Good news: the trait datasets already exist.** `data/prompts/{trait}.json` — 8 files, each
with 5 `instruction_variants` and 105 `questions`. No `0_generate_data.py` run, no Claude API
spend, no new data.

| | |
|---|---|
| cells | 12 series × 8 traits = 96 |
| generations | 96 × 200 = **~19,200** |
| activation pass | 19,200 forwards over generated sequences |
| storage | ~5GB (100 records/cell vs CAA's 500) |

Compute is modest — comparable to a CAA grid. **The blocker is a dependency, not GPU time.**

**`vllm` is not installed**, and `pipeline/1_generate.py:28` imports `VLLMGenerator` from
`assistant_axis`. Installing it is the risky step: `pylibs-py312` already carries
`torch 2.13.0+cu130`, verified working on Blackwell `sm_120` (§12.8), and vLLM pins its own
torch tightly. A `pip --target` install can shadow the working build — the §2 landmine, in the
one place where it would cost a validated environment.

Three options, in order of preference:

1. **Install a vLLM whose pinned torch matches 2.13.0+cu130.** Check the wheel's requirement
   *before* installing. Verify the §11 `sm_120` block still passes afterwards.
2. **Bypass vLLM: generate with HuggingFace `model.generate()`** and batching. Needs a change
   to `1_generate.py`, slower than vLLM, but adds no dependency and cannot break torch. On a
   96GB card this is very likely fast enough for 19,200 short generations.
3. Snapshot `pylibs-py312` first so a broken install is one `rm -rf` from recovery.

**Steps 2 and 3 need nothing new.** `2_activations.py` imports only torch + `ProbingModel`, and
`3_vectors.py` is pure tensor work — but note `3_vectors.py:106-115` slices `[:-1]`, so its
saved vectors are `(n_layers-1, hidden)` and every downstream `--layer` indexes a truncated
tensor (§6.1). The CAA analyses here dodged that by working from activations directly; do the
same for IV.

### 15.3b Two traps in `1_generate.py`, both hit on 2026-08-10

**`--personas` is NOT optional in practice.** Omitting it does not give you the 12 canonical
series — it loads **all 37 persona YAMLs** in `data/personas/`, i.e. 59,200 generations
(~6 hours) covering 25 personas that have no CAA counterpart to compare against. Unlike
`2c_caa_activations.py`, this script does not default to `PERSONA_SLUGS`. Always pass the
twelve explicitly:

```
--personas farmer politician therapist drill_sergeant street_hustler professor \
           tech_ceo kindergarten_teacher surgeon con_artist null nonsense
```

The dry-run prints the persona count; check it reads 12 and 19,200 jobs before launching.

**`--max-tokens 512` (the default) truncates most responses.** Measured on
farmer/assertiveness at |Q|=20: at a 256 cap, **90% of positive and 76% of suppression**
responses hit it. Natural lengths are median 357 / mean 358 (positive) and median 290 /
mean 291 (suppression), so **640 truncates nothing**. The asymmetry matters more than the
truncation: positive responses are genuinely ~23% longer than suppression ones, so a cap
clips the two arms of the contrast at different rates and turns response length into a
confound in the very subtraction meant to isolate the trait.

### 15.3c Measured cost, HF backend, Blackwell

One cell (200 generations, batch 50): **36s at max_tokens 256, ~70s at 640.** Full 96-cell
grid at 640 ≈ **1h50m**, ~20MB of JSONL. vLLM would save perhaps 40 minutes on a two-hour
job — not worth risking the validated torch (§15.3). Decision made on measurement, not
assumption.

### 15.4 What G.1 should produce, and what to expect

Run `caa_cosine_to_null.py` and `caa_within_cell_stability.py` against the IV vectors and
compare per-trait ordering against CAA. K/D report that the qualitative finding holds under IV
but **the per-trait ordering shifts**: under IV impulsivity is loosest (0.54) and confidence
tightest (0.87), whereas under CAA warmth was loosest and risk-taking tightest. Their nonsense
control stays near default under both (0.84–0.98). So a *changed ordering is the expected
result*, not a failure — the thing to check is whether the wide/tight tier split survives.

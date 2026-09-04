# The reproduction used 1 SFT epoch; the released adapters used 3

**Found 2026-09-03, at the §6b gate, before seed 2. Not the pre-registered first suspect.**

Spec §6a named the `llama-test` symlink as the first thing to suspect on a near-miss. It is
not the cause here. The cause is that **`docs/NEXT_POD.md`'s frozen runners were derived from
the OCT repo at HEAD, and HEAD's introspection config postdates the released adapters.**

## The evidence

| when | what | source |
|---|---|---|
| 2025-09-10T17:01:59Z | personas repo first upload | HF API, `maius/llama-3.1-8b-it-personas` |
| 2025-09-21T22:38:44Z | `last upload of personas` — the released artifacts | HF API |
| 2025-09-21T22:47:07Z | OCT `bd20b87` **"introspection 1 epoch instead of 3"** | `git log` |

`bd20b87` lands **8 minutes 23 seconds after** the final upload. Checked at *both* candidate
release dates, `introspection/llama.sh` reads `--max_epochs 3`:

```
as of 2025-09-10T17:01:59  commit e3a4cd25  scripts/introspection/llama.sh    --max_epochs 3
as of 2025-09-21T22:38:44  commit 63b285d3  finetuning/introspection/llama.sh --max_epochs 3
```

Our `llama_local.sh` was patched from HEAD, which reads `--max_epochs 1`.

**The DPO stage is unaffected.** `finetuning/distillation/` has not changed since
2025-09-19, before the release, and the release-era runner is byte-equivalent on every value
we ran: `seed 123456`, `max_epochs 1`, `max_len 1024`, `beta 0.1`, `nll_loss_coef 0.1`,
`kl_loss_coef 0.001`, `lora_rank 64`, `lora_alpha 128`, `learning_rate 5e-5`.

## Why this is the explanation for A2, and why A3 is the corroboration

| criterion | value | verdict |
|---|---|---|
| A2 overall ‖dW‖_F ratio repro/released | **0.641** | AMBIGUOUS (pass [0.7,1.4], stop outside [0.5,2.0]) |
| A3 per-module ‖dW‖_F profile, Spearman | **0.973** | PASS (pass ≥ 0.80) |

Same shape, smaller magnitude — which is what one-third of the SFT training looks like, and
is not what a wrong `llama-test` artifact would look like (that would move the profile too).

The effect is amplified by the merge. On the real pair the peft cross term is **56.9%** of
the merged norm (‖intended‖ 3.10, ‖cross‖ 3.61, ‖merged‖ 6.35) — close to the ~59% seen on
*independent* random adapters, i.e. the two stages are nearly as uncorrelated as chance. Since
the cross term carries one factor from each adapter, shrinking the SFT factors pulls down the
dominant component of ‖dW_merged‖, not a minor one.

## Status

The rig is correct; one frozen input was taken from the wrong commit. This blocks the gate:
the artifact measured is not the released pipeline, so neither a pass nor a fail on it is
about the reproduction.

**No threshold was changed and none should be.** A2's band did its job.

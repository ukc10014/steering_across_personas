# Reference copies of OCT files that upstream gitignores

`maiush/OpenCharacterTraining` gitignores `character/constants.py` (machine-local paths) and
`data/`, so a fresh clone cannot run. These are our reconstructions, kept in git because the
live copies exist only on the network volume and nothing else backs them up.

| file | what it is |
|---|---|
| `character_constants.py` | reconstructed from the four import sites; copy to `<oct>/character/constants.py` |

The training data is **not** kept here (153 MB). Rebuild it from Hugging Face:

```bash
# 1. the four released files -> /workspace/OpenCharacterTraining/data/
#    maius/OpenCharacterTraining-data : dpo/, self_reflection/, self_interaction/
# 2. the derived SFT corpus, byte-for-byte:
python scripts/build_oct_sft_corpus.py --check    # verifies against the frozen sha256
python scripts/build_oct_sft_corpus.py --write
```

`sft_data/` is not published by OCT — it is derived, and upstream's builder shuffles without a
`random_state`. `build_oct_sft_corpus.py` pins the shuffle so the frozen hash in
[`../../docs/spec_sham_lora.md`](../../docs/spec_sham_lora.md) §6a is reproducible rather than
a one-off artifact of one machine. Verified to reproduce it exactly on 2026-09-03.

"""Validate a G1 corrected-mask question cache before anything downstream reads it.

    python scripts/audit/validate_g1_cache.py base_cmask

Hard checks (exit 1 on any failure):
  - layout identical to the corrected-mask trajectory caches (reference: loving_traj_dpo):
    traits, personas, layers, n_questions, n_questions_per_trait, shape, dtype;
  - every value finite (the empathy tail is padded by repeating question 498);
  - embedded provenance: mask corrected, the trajectory runs' base snapshot as --model, the
    expected released adapter at scale 1 with matching weight hash (none for base), batch 16,
    variant 0, all questions, all 12 personas.

Reported (soft check, fails only on gross disagreement): the per-cell question-mean against the
legacy-mask counterpart (base, or the released arm), and for released arms the cosine between
the corrected (arm - base_cmask) and legacy (arm - base) per-cell mean shifts. The mask artifact
is small and nearly orthogonal to adapter shifts (audit §2.3), so these should be close to 1.
Question identity: the CAA prompt files (data/prompts/caa) are unchanged in git since 2026-04-19,
before every corrected-mask arm; the sidecar's questions_sha is recorded for future comparison.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from persona_steering.provenance import file_sha  # noqa: E402

CACHE = REPO / "outputs/_qcache"
OUT = REPO / "outputs/analysis/audit/g1"
REF = "loving_traj_dpo"
BASE = "/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
SNAP = "/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/318b5f7e1428097a1a61d5f0ed205ee048b3f620"


def cell_means(acts, nq):
    """(trait, persona, dir, layer, hidden) question-means over the true questions."""
    out = np.empty(acts.shape[:3] + acts.shape[4:], dtype=np.float64)
    for t in range(acts.shape[0]):
        out[t] = acts[t, :, :, :nq[t]].astype(np.float32).mean(axis=2)
    return out


def cos_rows(a, b):
    a = a.reshape(-1, a.shape[-1]); b = b.reshape(-1, b.shape[-1])
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    arm = sys.argv[1]
    fails, rep = [], {"arm": arm}
    z = np.load(CACHE / f"{arm}_L15_20.npz", mmap_mode="r")
    r = np.load(CACHE / f"{REF}_L15_20.npz", mmap_mode="r")
    for k in ("traits", "personas", "layers", "n_questions", "n_questions_per_trait", "semantic_personas"):
        if not np.array_equal(z[k], r[k]):
            fails.append(f"{k} differs from {REF}: {z[k]} vs {r[k]}")
    a = z["acts"]
    rep["shape"], rep["dtype"] = list(a.shape), str(a.dtype)
    if a.shape != r["acts"].shape or a.dtype != r["acts"].dtype:
        fails.append(f"shape/dtype {a.shape} {a.dtype} vs {r['acts'].shape} {r['acts'].dtype}")
    nonfinite = int(sum((~np.isfinite(a[t].astype(np.float32))).sum() for t in range(a.shape[0])))
    rep["nonfinite"] = nonfinite
    if nonfinite:
        fails.append(f"{nonfinite} non-finite values")

    if "provenance" not in z.files:
        fails.append("no embedded provenance")
        prov = {}
    else:
        prov = json.loads(str(z["provenance"]))
    rep["provenance"] = prov
    nq = [int(x) for x in z["n_questions_per_trait"]]
    want = {"mask": "corrected", "model": BASE, "batch_size": 16, "variant": 0,
            "max_questions": None, "dtype": "bfloat16",
            "n_questions": dict(sorted(zip([str(t) for t in z["traits"]], nq))),
            "personas": sorted(str(p) for p in z["personas"])}
    for k, v in want.items():
        if prov.get(k) != v:
            fails.append(f"provenance {k} = {prov.get(k)!r}, expected {v!r}")
    if arm == "base_cmask":
        exp_ad = []
    else:
        c = arm.removesuffix("_rel_cmask")
        w = Path(SNAP) / c / "adapter_model.safetensors"
        exp_ad = [{"path": str(Path(SNAP) / c), "scale": 1.0, "weights_sha": file_sha(w)}]
    if prov.get("adapters") != exp_ad:
        fails.append(f"adapters {prov.get('adapters')} != {exp_ad}")
    rep["caa_prompts_last_commit"] = subprocess.run(
        ["git", "-C", str(REPO), "log", "-1", "--format=%h %ad", "--date=iso", "--", "data/prompts/caa"],
        capture_output=True, text=True).stdout.strip()

    # soft numerical compatibility
    legacy = "base" if arm == "base_cmask" else arm.removesuffix("_rel_cmask")
    m_new = cell_means(a, nq)
    m_leg = cell_means(np.load(CACHE / f"{legacy}_L15_20.npz", mmap_mode="r")["acts"], nq)
    soft = {}
    for li, L in enumerate(z["layers"]):
        c_ = cos_rows(m_new[..., li, :], m_leg[..., li, :])
        rel = (np.linalg.norm(m_new[..., li, :] - m_leg[..., li, :], axis=-1)
               / np.linalg.norm(m_leg[..., li, :], axis=-1))
        soft[f"L{L}"] = {"cos_cellmean_vs_legacy_min": float(c_.min()), "cos_cellmean_vs_legacy_median": float(np.median(c_)),
                         "reldiff_cellmean_vs_legacy_median": float(np.median(rel)), "reldiff_max": float(rel.max())}
        if c_.min() < 0.9:
            fails.append(f"L{L}: a cell mean has cos {c_.min():.3f} with its legacy counterpart")
    if arm != "base_cmask" and (CACHE / "base_cmask_L15_20.npz").exists():
        b_new = cell_means(np.load(CACHE / "base_cmask_L15_20.npz", mmap_mode="r")["acts"], nq)
        b_leg = cell_means(np.load(CACHE / "base_L15_20.npz", mmap_mode="r")["acts"], nq)
        for li, L in enumerate(z["layers"]):
            c_ = cos_rows(m_new[..., li, :] - b_new[..., li, :], m_leg[..., li, :] - b_leg[..., li, :])
            soft[f"L{L}"]["cos_shift_corrected_vs_legacy_min"] = float(c_.min())
            soft[f"L{L}"]["cos_shift_corrected_vs_legacy_median"] = float(np.median(c_))
    rep["vs_legacy"] = soft
    rep["fails"] = fails
    rep["ok"] = not fails
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"validate_{arm}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "provenance"}, indent=1, default=str))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()

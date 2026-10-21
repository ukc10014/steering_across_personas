"""Extraction provenance for CAA activation caches: record it, carry it, and refuse to mix it.

WHY. On Llama-3.1 the pre-fix attention mask (pipeline/2c_caa_activations.py --legacy-mask)
zeroes genuine <|eot_id|> tokens. Measured on a same-model pair, the artifact is a
persona-common shift of ~0.10 (L15) / ~0.12 (L20) of the base trait-vector norm
(outputs/analysis/audit/mask_artifact.json). It cancels when both sides of a contrast use the
same mask and is injected straight into the contrast when they do not. From 2026-09-03 on,
corrected-mask arms were differenced against the legacy-mask archived base without any check,
because nothing recorded which mask a cache was built with. This module is that record.

Three layers:
  1. 2c writes SIDECAR (`_extraction.json`) into every activations dir and refuses to resume
     into a dir whose sidecar disagrees, or which holds .pt files but no sidecar.
  2. build_question_cache.py copies the sidecar into the .npz as `provenance` (JSON string).
  3. Analyses load through `load_arm_checked`, which reads that, falls back to REGISTRY for
     caches built before this existed, and raises on unknown provenance or on any mismatch in
     COMPARABILITY_KEYS across the arms of one analysis.

Old caches are NEVER assigned a protocol from current defaults: only the registry, whose
entries cite evidence, can vouch for them, and `unknown` there is a refusal.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
from pathlib import Path

SIDECAR = "_extraction.json"
SCHEMA = 1
REPO = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO / "data" / "provenance" / "qcache_registry.json"

# Must be identical for two caches to be compared. Model/adapter identity is deliberately NOT
# here: arms differ in exactly that. Batch size is not here either (left padding under the
# corrected positional mask is exact; under the legacy mask it is not, but legacy caches are
# only ever compared with other legacy caches).
COMPARABILITY_KEYS = ("mask", "questions_sha", "personas_sha", "variant", "max_questions",
                      "answer_token_rule")
MASKS = ("corrected", "legacy")


class ProvenanceError(RuntimeError):
    pass


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def file_sha(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()[:16]


def git_state(repo: Path = REPO) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:                                   # noqa: BLE001 -- best effort
            return ""
    return {"commit": run("rev-parse", "HEAD") or None,
            "dirty_files": [l[3:] for l in run("status", "--porcelain").splitlines()
                            if l.endswith(".py")]}


def adapter_identity(path: str, scale: float) -> dict:
    p = Path(path)
    w = p / "adapter_model.safetensors"
    return {"path": str(p), "scale": float(scale),
            "weights_sha": file_sha(w) if w.exists() else None}


def extraction_record(*, mask: str, model: str, adapters: list[str] | None,
                      scales: list[float] | None, datasets: dict, personas: dict,
                      variant: int, max_questions, batch_size: int, dtype: str,
                      extractor: Path) -> dict:
    """The sidecar 2c writes. `datasets` maps trait -> list of question dicts/strings;
    `personas` maps slug -> the system prompt actually used."""
    if mask not in MASKS:
        raise ValueError(f"mask must be one of {MASKS}, got {mask!r}")
    adapters = adapters or []
    scales = scales or [1.0] * len(adapters)
    return {
        "schema": SCHEMA, "mask": mask, "model": model,
        "adapters": [adapter_identity(a, s) for a, s in zip(adapters, scales)],
        "questions_sha": _sha({t: q for t, q in sorted(datasets.items())}),
        "n_questions": {t: len(q) for t, q in sorted(datasets.items())},
        "personas_sha": _sha(personas), "personas": sorted(personas),
        "variant": variant, "max_questions": max_questions, "batch_size": batch_size,
        "dtype": dtype, "answer_token_rule": "find_answer_token_position/v1",
        "extractor": {"file": str(extractor.relative_to(REPO)) if extractor.is_relative_to(REPO)
                      else str(extractor), "sha": file_sha(extractor), **git_state()},
    }


def check_resume(out_dir: Path, record: dict) -> None:
    """Refuse to add cells to a dir built under a different protocol or an unknown one."""
    side = out_dir / SIDECAR
    has_pt = any(out_dir.glob("*.pt")) if out_dir.exists() else False
    if not side.exists():
        if has_pt:
            raise ProvenanceError(
                f"{out_dir} holds .pt files but no {SIDECAR}: their extraction protocol is "
                "unknown, so new cells must not be mixed in. Move them aside, or extract into "
                "a fresh directory.")
        return
    old = json.loads(side.read_text())
    diff = [k for k in (*COMPARABILITY_KEYS, "model", "adapters") if old.get(k) != record.get(k)]
    if diff:
        raise ProvenanceError(f"{out_dir}: existing cells were extracted with different "
                              f"{diff}; refusing to mix protocols in one directory.")


def write_sidecar(out_dir: Path, record: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SIDECAR).write_text(json.dumps(record, indent=1))


def read_sidecar(act_dir: Path) -> dict | None:
    side = Path(act_dir) / SIDECAR
    return json.loads(side.read_text()) if side.exists() else None


# --------------------------------------------------------------------------------------------
# historical caches

def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"entries": []}


def registry_lookup(arm: str, registry: dict | None = None) -> dict | None:
    """First entry whose pattern matches `arm` (exact names before globs, in file order)."""
    reg = registry if registry is not None else load_registry()
    entries = reg.get("entries", [])
    for e in entries:
        if e["pattern"] == arm:
            return e
    for e in entries:
        if any(c in e["pattern"] for c in "*?[") and fnmatch.fnmatchcase(arm, e["pattern"]):
            return e
    return None


def cache_provenance(npz, arm: str, registry: dict | None = None) -> dict:
    """Provenance for one loaded qcache: embedded record if present, else the registry."""
    if "provenance" in getattr(npz, "files", []):
        rec = json.loads(str(npz["provenance"]))
        rec["_source"] = "embedded"
        return rec
    e = registry_lookup(arm, registry)
    if e is None or e.get("mask") not in MASKS:
        raise ProvenanceError(
            f"qcache {arm!r} has no embedded provenance and the registry "
            f"{'has no entry for it' if e is None else 'marks its mask ' + repr(e.get('mask'))}. "
            "Its extraction protocol is unverified; it cannot be compared with anything.")
    return {"mask": e["mask"], "questions_sha": e.get("questions_sha", "archive-500q/v1"),
            "personas_sha": e.get("personas_sha", "archive-12p/v1"), "variant": 0,
            "max_questions": None, "answer_token_rule": "find_answer_token_position/v1",
            "_source": f"registry:{e['pattern']}", "_evidence": e.get("evidence"),
            "_confidence": e.get("confidence")}


def assert_comparable(provs: dict[str, dict], *, require_mask: str | None = None,
                      allow_mixed_mask: bool = False) -> dict:
    """Raise unless every arm agrees on COMPARABILITY_KEYS (and, if given, on require_mask).

    `allow_mixed_mask` exists ONLY to reproduce historical numbers for comparison; callers
    must record that it was used in their output. Returns a summary for that record."""
    masks = {a: p["mask"] for a, p in provs.items()}
    if require_mask and any(m != require_mask for m in masks.values()):
        bad = {a: m for a, m in masks.items() if m != require_mask}
        if not allow_mixed_mask:
            raise ProvenanceError(f"arms not extracted with mask={require_mask!r}: {bad}")
    keys = [k for k in COMPARABILITY_KEYS if not (allow_mixed_mask and k == "mask")]
    for k in keys:
        vals = {a: p.get(k) for a, p in provs.items()}
        if len({json.dumps(v, sort_keys=True) for v in vals.values()}) > 1:
            raise ProvenanceError(f"incomparable caches: {k} differs across arms: {vals}")
    return {"masks": masks, "mixed_mask": len(set(masks.values())) > 1,
            "allow_mixed_mask": allow_mixed_mask,
            "sources": {a: p.get("_source") for a, p in provs.items()}}

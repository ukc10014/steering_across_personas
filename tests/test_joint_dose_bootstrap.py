"""Joint trajectory/ladder bootstrap (scripts/audit/joint_dose_bootstrap.py): matching rules
and the shared question draw."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts/common_trajectory", REPO / "scripts/audit"):
    sys.path.insert(0, str(p))
from geometry_lib import bootstrap_half_splits, prepare_diff  # noqa: E402
from joint_dose_bootstrap import dose_cells, excess, summarise_excess, trait_draws  # noqa: E402


def test_excess_interpolates_inside_ladder():
    xl = np.array([0.5, 0.6, 0.7]); yl = np.array([1.0, 2.0, 3.0])
    v, s = excess(0.65, 4.0, xl, yl)
    assert s == "ok" and v == pytest.approx(4.0 - 2.5)


def test_excess_sorts_ladder_by_this_replicates_dose():
    # rungs out of order (as a replicate can make them): sorted before interpolation
    v, s = excess(0.65, 4.0, np.array([0.7, 0.5, 0.6]), np.array([3.0, 1.0, 2.0]))
    assert s == "ok" and v == pytest.approx(1.5)


def test_excess_never_extrapolates():
    xl = np.array([0.668, 0.7, 0.8]); yl = np.array([0.0, 0.0, 0.0])
    v, s = excess(0.665, 1.0, xl, yl)          # the goodness L20 0.24-ep case
    assert s == "unbracketed" and np.isnan(v)


def test_excess_invalid_when_any_rung_undefined():
    v, s = excess(0.65, 1.0, np.array([0.5, 0.7]), np.array([np.nan, 0.0]))
    assert s == "invalid" and np.isnan(v)


def test_rule_withholds_interval_below_threshold():
    boot = [(0.1, "ok")] * 970 + [(np.nan, "unbracketed")] * 30
    e = summarise_excess((0.1, "ok"), boot)
    assert not e["estimable"] and e["lo"] is None and e["unbracketed_frac"] == pytest.approx(0.03)
    boot = [(0.1 + 0.001 * i, "ok") for i in range(980)] + [(np.nan, "invalid")] * 20
    e = summarise_excess((0.1, "ok"), boot)
    assert e["estimable"] and e["invalid_frac"] == pytest.approx(0.02) and e["lo"] is not None


def test_rule_withholds_when_point_unbracketed():
    e = summarise_excess((np.nan, "unbracketed"), [(0.1, "ok")] * 100)
    assert not e["estimable"] and e["point"] is None and "unbracketed" in e["reason"]


def test_shared_draw_split_is_disjoint_and_covers_draw():
    idx = np.random.default_rng(1).integers(0, 50, 50)
    for A, B in bootstrap_half_splits(50, 8, np.random.default_rng(2), idx=idx):
        assert not set(A) & set(B)
        assert sorted(np.concatenate([A, B])) == sorted(idx)


def test_default_draw_unchanged():
    """idx=None must consume the RNG exactly as before (historical reproductions)."""
    a = bootstrap_half_splits(40, 4, np.random.default_rng(7))
    r = np.random.default_rng(7); idx = r.integers(0, 40, 40)
    b = bootstrap_half_splits(40, 4, r, idx=idx)
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_trait_draws_reproducible_and_shared():
    d1, _ = trait_draws(30, 5, 0, 3); d2, _ = trait_draws(30, 5, 0, 3); d3, _ = trait_draws(30, 5, 0, 4)
    assert all(np.array_equal(x, y) for x, y in zip(d1, d2))
    assert not all(np.array_equal(x, y) for x, y in zip(d1, d3))


def test_dose_matches_direct_computation_on_draw():
    rng = np.random.default_rng(0)
    bp, bn = rng.normal(size=(2, 3, 20, 8)).astype(np.float32)
    ap_, an = (bp + 0.3 * rng.normal(size=bp.shape)).astype(np.float32), (bn + 0.1).astype(np.float32)
    idx = rng.integers(0, 20, 20)
    d = dose_cells(prepare_diff(bp, bn), {"a": prepare_diff(ap_ - bp, an - bn)}, [idx])["a"][0]
    vb = bp[:, idx].mean(1) - bn[:, idx].mean(1); va = ap_[:, idx].mean(1) - an[:, idx].mean(1)
    np.testing.assert_allclose(d, np.linalg.norm(va - vb, axis=-1) / np.linalg.norm(vb, axis=-1), rtol=1e-4)

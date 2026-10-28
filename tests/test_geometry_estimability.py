"""Regression tests for scripts/common_trajectory/estimability.py (audit issue 2).

The original derive() floored non-positive cross-fitted energies at 1e-12 inside cosine
denominators, producing cosines of order 1e9, and dropped NaN replicates silently. These tests
pin the replacement behaviour on synthetic raw accumulators.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "common_trajectory"))
from estimability import I, RAW, VALID_MIN, quantities, summarise  # noqa: E402


def raw(g2, dv2, gx, g2s=np.nan, dv2s=np.nan, gxs=np.nan, bn=1.0):
    return np.array([g2, dv2, gx, g2s, dv2s, gxs, bn], dtype=float)


def test_negative_denominator_is_undefined_not_huge():
    # arm 0 has a negative cross-fitted common energy; final (arm 1) is healthy
    r = np.stack([raw(-1e-4, 0.5, 0.01), raw(0.4, 0.6, 0.4)])
    q = quantities(r, fi=1)
    assert np.isnan(q["cosT"][0])                     # was 0.01/sqrt(1e-12*0.4) ~ 1.6e4
    assert q["cosT"][1] == pytest.approx(1.0)
    assert q["Ecommon"][0] == pytest.approx(-1e-4)    # raw energy kept, unclipped
    assert q["g"][0] == 0.0


def test_negative_final_energy_makes_every_cosine_to_it_undefined():
    r = np.stack([raw(0.3, 0.5, 0.01), raw(-2e-4, 0.6, -2e-4)])
    q = quantities(r, fi=1)
    assert np.isnan(q["cosT"]).all()                  # self-cosine was -2e-4/1e-12 = -2e8


def test_stable_signal_reproduces_plain_cosine():
    rng = np.random.default_rng(0)
    a, f = rng.normal(size=50), rng.normal(size=50)
    r = np.stack([raw(a @ a, 2 * a @ a, a @ f), raw(f @ f, 2 * f @ f, f @ f)])
    q = quantities(r, fi=1)
    assert q["cosT"][0] == pytest.approx(a @ f / np.linalg.norm(a) / np.linalg.norm(f))
    assert q["S"][0] == pytest.approx(0.5)


def _boot(point, B, frac_bad, rng):
    """B replicates around `point` (arms, 7); a fraction of them get a negative final g2s."""
    b = np.repeat(point[None], B, 0) * (1 + 0.01 * rng.normal(size=(B,) + point.shape))
    nbad = int(round(frac_bad * B))
    b[:nbad, -1, I["g2s"]] = -abs(b[:nbad, -1, I["g2s"]])
    return b


@pytest.mark.parametrize("frac_bad,expect", [(0.0, True), (0.02, True), (0.05, False), (0.5, False)])
def test_estimability_threshold(frac_bad, expect):
    rng = np.random.default_rng(1)
    pt = np.stack([raw(0.3, 0.5, 0.2, 0.1, 0.3, 0.05), raw(0.4, 0.6, 0.4, 0.2, 0.4, 0.2)])
    b = _boot(pt, 200, frac_bad, rng)
    s = summarise({"t": pt}, {"t": b}, ["a", "final"], {"t": ["t"]})
    e = s["t"]["a"]["cosS"]
    assert e["estimable"] is expect
    assert e["valid_frac"] == pytest.approx(1 - frac_bad)
    if expect:
        assert -1.0 <= e["lo"] <= e["hi"] <= 1.0      # no epsilon-sized blow-ups
    else:
        assert e["point"] is None and e["lo"] is None
    assert s["t"]["a"]["EcommonS"]["point"] == pytest.approx(0.1)   # raw energy always reported


def test_self_comparison_is_one_by_construction_even_if_unstable():
    rng = np.random.default_rng(2)
    pt = np.stack([raw(0.3, 0.5, 0.2, 0.1, 0.3, 0.05), raw(0.4, 0.6, 0.4, 0.2, 0.4, 0.2)])
    s = summarise({"t": pt}, {"t": _boot(pt, 200, 0.3, rng)}, ["a", "final"], {"t": ["t"]})
    assert s["t"]["final"]["cosS"]["point"] == 1.0
    assert s["t"]["final"]["Q"]["hi"] == 1.0


def test_multi_trait_mean_needs_every_trait_but_pooled_survives():
    rng = np.random.default_rng(3)
    good = np.stack([raw(0.3, 0.5, 0.2, 0.1, 0.3, 0.05), raw(0.4, 0.6, 0.4, 0.2, 0.4, 0.2)])
    # one trait whose final incremental common energy is ~0: its cosS is not estimable
    weak = np.stack([raw(0.3, 0.5, 0.2, 0.001, 0.3, 0.0), raw(0.4, 0.6, 0.4, 0.0005, 0.4, 0.0005)])
    bw = _boot(weak, 200, 0.4, rng)
    s = summarise({"g": good, "w": weak}, {"g": _boot(good, 200, 0, rng), "w": bw},
                  ["a", "final"], {"MEAN": ["g", "w"]})
    e = s["MEAN"]["a"]["cosS"]
    assert e["estimable"] is False and e["point"] is None and e["n_traits_estimable"] == 1
    assert e["pooled"]["estimable"] is True                # energy-weighted estimand
    assert -1.0 <= e["pooled"]["lo"] <= e["pooled"]["hi"] <= 1.0


def test_dpo_state_has_no_incremental_quantities():
    pt = np.stack([raw(0.3, 0.5, 0.2), raw(0.4, 0.6, 0.4, 0.2, 0.4, 0.2)])
    b = np.repeat(pt[None], 50, 0)
    s = summarise({"t": pt}, {"t": b}, ["dpo", "final"], {"t": ["t"]})
    assert s["t"]["dpo"]["gS"]["point"] is None
    assert s["t"]["dpo"]["cosT"]["estimable"] is True


def test_pooled_incremental_undefined_for_dpo_state():
    pt = np.stack([raw(0.3, 0.5, 0.2), raw(0.4, 0.6, 0.4, 0.2, 0.4, 0.2)])
    s = summarise({"a": pt, "b": pt}, {"a": np.repeat(pt[None], 50, 0), "b": np.repeat(pt[None], 50, 0)},
                  ["dpo", "final"], {"G": ["a", "b"]})
    assert s["G"]["dpo"]["Q"]["pooled"]["estimable"] is False
    assert s["G"]["dpo"]["Q"]["pooled"]["point"] is None
    assert s["G"]["dpo"]["cosT"]["pooled"]["estimable"] is True


def test_raw_layout_is_stable():
    assert RAW == ["g2", "dv2", "gx", "g2s", "dv2s", "gxs", "bn"] and 0 < VALID_MIN < 1

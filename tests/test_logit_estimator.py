"""Regression tests for the logit offset estimators (audit issue 3).

fit_offset_slope averages per-polarity OLS intercepts. That cancels the ADAPTED model's letter
bias but leaves b*(k_minus - k_plus)/2 of the BASE model's letter bias b when retention differs
by polarity. These tests pin the algebra on noise-free synthetic data.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from caa_logits_analysis import (balanced_mean, fit_offset_shared_slope,  # noqa: E402
                                 fit_offset_slope, fit_offset_slope_detail)


def synth(a, b, c, k_plus, k_minus, n_p=4, n_q=200, seed=0):
    """base x = u + b*s ; adapted y = a + k_s*u + c*s ; polarity alternates over questions."""
    rng = np.random.default_rng(seed)
    a_pos = np.arange(n_q) % 2 == 0
    s = np.where(a_pos, 1.0, -1.0)
    u = rng.normal(size=(n_p, n_q))
    k = np.where(a_pos, k_plus, k_minus)
    return u + b * s, a + k * u + c * s, a_pos


def test_auditor_example_leaves_point_six():
    x, y, a_pos = synth(a=0.0, b=2.0, c=0.0, k_plus=0.2, k_minus=0.8)
    q = np.arange(x.shape[1])
    a_hat, _ = fit_offset_slope(x, y, a_pos, q)
    assert a_hat == pytest.approx(2.0 * (0.8 - 0.2) / 2, abs=1e-9)        # 0.6, not 0
    d = fit_offset_slope_detail(x, y, a_pos, q)
    assert d["groups"]["plus"]["k"] == pytest.approx(0.2)
    assert d["groups"]["minus"]["k"] == pytest.approx(0.8)
    assert d["a"] == pytest.approx(a_hat)
    # b_hat recovers b here only up to the sample imbalance of u across groups
    assert d["implied_bias"] == pytest.approx(0.6, abs=0.15)


@pytest.mark.parametrize("c", [0.0, 1.5, -3.0])
def test_adapted_letter_bias_always_cancels(c):
    x, y, a_pos = synth(a=0.7, b=0.0, c=c, k_plus=0.3, k_minus=0.9)
    a_hat, _ = fit_offset_slope(x, y, a_pos, np.arange(x.shape[1]))
    assert a_hat == pytest.approx(0.7, abs=1e-9)


@pytest.mark.parametrize("b,c", [(2.0, 0.0), (2.0, -1.0), (-1.0, 3.0)])
def test_equal_retention_both_regressions_recover_offset(b, c):
    x, y, a_pos = synth(a=0.7, b=b, c=c, k_plus=0.5, k_minus=0.5)
    q = np.arange(x.shape[1])
    assert fit_offset_slope(x, y, a_pos, q)[0] == pytest.approx(0.7, abs=1e-9)
    a_sh, k_sh, _ = fit_offset_shared_slope(x, y, a_pos, q)
    assert a_sh == pytest.approx(0.7, abs=1e-9) and k_sh == pytest.approx(0.5, abs=1e-9)


def test_raw_balanced_change_mixes_offset_with_retention():
    # no letter bias at all; the raw balanced change still differs from a when k != 1 and the
    # base preference is not centred, which is why it is reported only as a sensitivity
    x, y, a_pos = synth(a=0.5, b=0.0, c=0.0, k_plus=0.5, k_minus=0.5)
    x, y = x + 1.0, y + 0.5            # base mean +1, retained at k=0.5
    q = np.arange(x.shape[1])
    raw = balanced_mean(y - x, a_pos, q)
    assert fit_offset_slope(x, y, a_pos, q)[0] == pytest.approx(0.5, abs=1e-9)
    assert raw == pytest.approx(0.5 + (0.5 - 1) * x.mean(), abs=1e-9)

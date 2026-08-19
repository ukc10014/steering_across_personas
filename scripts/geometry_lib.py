#!/usr/bin/env python3
"""Estimators for persona/trait geometry in hidden space, and their noise handling.

Shared by the dispersion (section 5), RDM (section 6) and Procrustes (section 8) analyses.
Run `python scripts/geometry_lib.py` to execute the self-tests.

THE BIAS THAT MOTIVATES ALL OF THIS.
Every quantity here is quadratic in the trait vectors -- a squared distance, a squared
norm, a dispersion. Quadratic functionals of a noisy estimate are biased UPWARD, and the
bias is the noise variance:

    E||V_hat||^2 = ||V_true||^2 + E||noise||^2

With 4096 dimensions and a few hundred questions that noise term is not a rounding error;
it can dominate. Worse, it is not shared across arms -- if one arm's activations are
noisier, it will look more dispersed and its personas will look further apart, with no
change in the underlying geometry whatsoever. Comparing D_arm to D_base naively therefore
measures partly noise.

The fix, used throughout, is cross-fitting: split the questions into two disjoint halves,
estimate the vector independently in each, and take the INNER PRODUCT of the two estimates
instead of the squared norm of one:

    E<V_hat^A, V_hat^B> = <V_true, V_true> = ||V_true||^2

because the two noise terms are independent and mean-zero, so their contribution vanishes
rather than accumulating. Averaging over many random half-splits reduces the variance of
the estimator without reintroducing the bias.

This is the same correction, applied to a different statistic, that commit d44a267 had to
introduce for residual_cos after the naive version turned out to have no power at all.
A cross-fitted quantity can come out NEGATIVE when the true value is near zero; that is
correct behaviour for an unbiased estimator, not a bug, and it must not be clipped before
averaging.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------------------
# trait vectors
# ---------------------------------------------------------------------------------------

def trait_vectors(pos: np.ndarray, neg: np.ndarray, idx: np.ndarray | None = None
                  ) -> np.ndarray:
    """CAA trait vector, mean(pos) - mean(neg).

    pos, neg: (..., n_questions, hidden). idx selects questions (a bootstrap resample or
    one half of a split). Returns (..., hidden).
    """
    if idx is None:
        return pos.mean(-2) - neg.mean(-2)
    return pos[..., idx, :].mean(-2) - neg[..., idx, :].mean(-2)


def half_splits(n: int, n_splits: int, rng: np.random.Generator) -> list[tuple]:
    """n_splits random (A, B) disjoint halves of range(n)."""
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        out.append((perm[: n // 2], perm[n // 2:]))
    return out


def bootstrap_half_splits(n: int, n_splits: int, rng: np.random.Generator) -> list[tuple]:
    """Half-splits for a BOOTSTRAP replicate of a cross-fitted statistic.

    Two requirements pull against each other, and both must hold:

      (a) the two halves must contain DISJOINT original questions, or the noise terms stop
          being independent and the bias that cross-fitting removes comes straight back;
      (b) the replicate must genuinely vary WHICH questions are in the sample, or the
          interval measures nothing about question sampling.

    Two wrong versions, both of which were written before this one and both of which fail
    loudly enough to be worth recording:

      resample-then-split -- draw n indices with replacement, then cut the resampled array
        in half. Violates (a): a duplicated question lands in both halves. The point
        estimate then sits OUTSIDE its own CI, because it is unbiased and the replicates
        are not.

      resample-within-fixed-halves -- split first, then resample inside each half.
        Satisfies (a) but violates (b): every replicate still contains all n original
        questions, only reweighted, so the interval collapses as more splits are averaged
        (measured coverage 88% at 4 splits, 65% at 12, against a nominal 95%).

    What this does instead: draw the bootstrap sample first, then assign each DISTINCT
    question -- with all of its duplicate copies -- to one side or the other. Composition
    varies, sides stay disjoint, and both requirements hold at once.

    ONE bootstrap draw per replicate, re-split n_splits ways. Drawing a fresh bootstrap
    sample for each split would make the replicate an average of n_splits independent
    replicates, shrinking its variance by roughly that factor and collapsing the interval
    (measured coverage 62% at 4 splits, 55% at 12). The split randomness is nuisance
    variation to be averaged out; the question draw is the uncertainty being measured, and
    it must stay fixed within a replicate.
    """
    idx = rng.integers(0, n, n)
    uniq_all = np.unique(idx)
    out = []
    for _ in range(n_splits):
        uniq = rng.permutation(uniq_all)
        left = set(uniq[: len(uniq) // 2].tolist())
        mask = np.array([i in left for i in idx])
        a_i, b_i = idx[mask], idx[~mask]
        if len(a_i) == 0 or len(b_i) == 0:      # degenerate draw; skip
            continue
        out.append((a_i, b_i))
    return out


# ---------------------------------------------------------------------------------------
# section 5: persona dispersion in full hidden space
# ---------------------------------------------------------------------------------------

def dispersion_naive(V: np.ndarray) -> float:
    """Mean squared distance of personas from their centroid. V: (k, hidden).

    Biased upward by the per-vector noise variance. Kept for comparison only.
    """
    Z = V - V.mean(0, keepdims=True)
    return float((Z ** 2).sum(-1).mean())


def dispersion_crossfit(pos: np.ndarray, neg: np.ndarray, splits: list[tuple]) -> float:
    """Unbiased mean squared persona spread about the centroid.

    pos, neg: (k_personas, n_questions, hidden).

    Centring happens INSIDE each half, so the centroid is estimated from the same half as
    the vectors it is subtracted from; the two halves stay independent of each other, which
    is what makes the cross term vanish.
    """
    vals = []
    for a, b in splits:
        Va = trait_vectors(pos, neg, a)
        Vb = trait_vectors(pos, neg, b)
        Za = Va - Va.mean(0, keepdims=True)
        Zb = Vb - Vb.mean(0, keepdims=True)
        vals.append((Za * Zb).sum(-1).mean())
    return float(np.mean(vals))


# ---------------------------------------------------------------------------------------
# section 6: representational dissimilarity matrices
# ---------------------------------------------------------------------------------------

def rdm_naive(V: np.ndarray, squared: bool = True) -> np.ndarray:
    """Condensed pairwise distances of V: (k, hidden) -> (k*(k-1)/2,).

    Euclidean distance is invariant to adding a common vector to every persona, which is
    the property that makes an RDM the right tool here: the large shared OCT component
    cancels exactly, with no estimation step and nothing to get wrong.
    """
    d2 = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(V.shape[0], k=1)
    out = d2[iu]
    return out if squared else np.sqrt(np.maximum(out, 0.0))


def rdm_crossfit(pos: np.ndarray, neg: np.ndarray, splits: list[tuple],
                 squared: bool = True) -> np.ndarray:
    """Unbiased condensed squared pairwise distances.

    <Va_p - Va_q, Vb_p - Vb_q> is unbiased for ||E V_p - E V_q||^2. Individual estimates
    can go negative for genuinely coincident pairs; averaging over splits happens BEFORE
    any sqrt so the estimator stays unbiased.
    """
    k = pos.shape[0]
    iu = np.triu_indices(k, k=1)
    acc = []
    for a, b in splits:
        Va = trait_vectors(pos, neg, a)
        Vb = trait_vectors(pos, neg, b)
        Da = Va[:, None, :] - Va[None, :, :]
        Db = Vb[:, None, :] - Vb[None, :, :]
        acc.append((Da * Db).sum(-1)[iu])
    out = np.mean(acc, axis=0)
    return out if squared else np.sqrt(np.maximum(out, 0.0))


def rdm_reliability(pos: np.ndarray, neg: np.ndarray, splits: list[tuple],
                    method: str = "spearman") -> float:
    """Split-half reliability of an arm's own RDM: the NOISE CEILING.

    WHY NO RDM CORRELATION IS INTERPRETABLE WITHOUT THIS. Correlating two independently
    estimated RDMs attenuates towards zero with estimation noise, so a preservation score
    has a ceiling set by measurement, not by geometry. On synthetic data where an arm is a
    PURE ROTATION of base -- which preserves every pairwise distance exactly, so the true
    correlation is 1.0 -- the measured Spearman came out at 0.72-0.82. Read naively that
    is "the geometry changed"; in fact nothing changed at all.

    Worse, the ceiling is not shared across arms. An arm with a weaker trait signal
    relative to question noise has a lower ceiling and will look less preserving for that
    reason alone. In the same synthetic check, a pure 0.6x CONTRACTION -- which also
    preserves shape exactly -- scored below the rotation arm purely because scaling down
    the signal lowered its SNR.

    Estimated by correlating the RDM from one question half against the RDM from the
    disjoint other half, then Spearman-Brown corrected from half-size to full-size:
    r_full = 2r / (1 + r).
    """
    from scipy.stats import spearmanr
    k = pos.shape[0]
    iu = np.triu_indices(k, k=1)
    vals = []
    for a, b in splits:
        Va, Vb = trait_vectors(pos, neg, a), trait_vectors(pos, neg, b)
        Ra = ((Va[:, None, :] - Va[None, :, :]) ** 2).sum(-1)[iu]
        Rb = ((Vb[:, None, :] - Vb[None, :, :]) ** 2).sum(-1)[iu]
        r = (spearmanr(Ra, Rb).statistic if method == "spearman"
             else np.corrcoef(Ra, Rb)[0, 1])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return float("nan")
    r_half = float(np.mean(vals))
    return float(2 * r_half / (1 + r_half)) if r_half > -1 else float("nan")


def attenuation_corrected(r_obs: float, rel_a: float, rel_b: float) -> float:
    """r_obs / sqrt(rel_a * rel_b), the classical correction for attenuation.

    Answers "how preserved is the geometry, given how well either side could be measured".
    Can exceed 1 when reliabilities are underestimated; that is a signal the ceiling is
    noisy, not evidence of more-than-perfect preservation, so it is NOT clipped here.
    """
    den = np.sqrt(max(rel_a, 1e-9) * max(rel_b, 1e-9))
    return float(r_obs / den) if den > 1e-9 else float("nan")


def rdm_shape(rdm: np.ndarray) -> np.ndarray:
    """Scale-normalised RDM: divide by RMS, so only the SHAPE of the constellation remains.

    NOTE ON WHAT THIS IS AND IS NOT FOR. Pearson and Spearman are already invariant to
    scale, so correlating normalised RDMs returns exactly the same number as correlating
    the raw ones -- an earlier version of the analysis reported both and got identical
    columns. Use this for HEATMAPS and for any statistic that is not scale-invariant. The
    absolute expansion/contraction that normalising discards is reported separately as the
    RMS ratio; the correlations are the shape measure.
    """
    rms = np.sqrt(np.mean(rdm ** 2))
    return rdm / max(rms, 1e-12)


# ---------------------------------------------------------------------------------------
# section 8: Procrustes
# ---------------------------------------------------------------------------------------

def procrustes_lowrank(Y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Orthogonal Procrustes for k points in H dimensions with k << H.

    Finds R minimising ||Y R - X||_F over orthogonal R, and returns (YR, E_raw, E_proc)
    with E = ||.||_F / ||X||_F.

    WHY NOT FORM R DIRECTLY. R is H x H (4096^2 here), but the problem only constrains it
    on the k-dimensional row space of Y. Y^T X has rank <= k, so its nonzero singular
    values are those of a k x k matrix and the whole solve reduces to thin SVDs:

        Y = Uy Sy Vy^T,  X = Ux Sx Vx^T,  M = Sy Uy^T Ux Sx = P Sig Q^T
        Y R = Uy Sy P Q^T Vx^T

    THE INTERPRETIVE TRAP, and it is a serious one. With k=10 points in 4096 dimensions,
    an orthogonal map has enormous freedom, and E_proc will be small almost regardless of
    whether the two configurations are genuinely related -- an orthogonal R can align any
    two centred configurations that happen to share a singular-value spectrum. An in-sample
    E_proc is therefore close to uninformative on its own. Use procrustes_cv for anything
    load-bearing.
    """
    Uy, Sy, Vyt = np.linalg.svd(Y, full_matrices=False)
    Ux, Sx, Vxt = np.linalg.svd(X, full_matrices=False)
    M = (Sy[:, None] * (Uy.T @ Ux)) * Sx[None, :]
    P, _, Qt = np.linalg.svd(M)
    YR = Uy @ (Sy[:, None] * (P @ Qt)) @ Vxt
    nx = np.linalg.norm(X)
    return YR, float(np.linalg.norm(Y - X) / nx), float(np.linalg.norm(YR - X) / nx)


def procrustes_cv(Y: np.ndarray, X: np.ndarray, train: np.ndarray, test: np.ndarray,
                  rank: int = 20) -> dict:
    """Cross-validated Procrustes: fit the transform on `train` rows, score on `test`.

    Each arm is projected onto ITS OWN r-dimensional basis, both fitted on training rows
    only. That detail is the whole design:

      - fitting the basis on training rows only keeps the held-out rows out of the fit, so
        the test score is not leaked;
      - giving each arm its own basis is what makes a global rotation representable at all.
        A single shared basis silently assumes the rotation maps that subspace to itself.
        It generally does not -- a genuine global rotation carries base's subspace to a
        DIFFERENT subspace -- so a shared basis destroys exactly the structure this test
        exists to detect, and reports a real global rotation as unexplained. (This was not
        hypothetical: the shared-basis version of this function scored a known, exact
        global rotation at test_proc = 0.88, i.e. essentially unexplained.)

    Inside the two bases R is a genuine r x r orthogonal matrix, so applying it to held-out
    rows is well defined -- which it is not for the raw low-rank solve in
    procrustes_lowrank.

    Reports error with no transform (test_raw), after the orthogonal map (test_proc), and
    after additionally allowing one global scalar (test_proc_scaled). The last separates
    'the coordinates were rotated' from 'everything simply got bigger or smaller'.
    """
    mu_x = X[train].mean(0, keepdims=True)
    mu_y = Y[train].mean(0, keepdims=True)
    Xc, Yc = X - mu_x, Y - mu_y

    def basis(A: np.ndarray, r: int) -> np.ndarray:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        return Vt[:r].T                                  # (H, r)

    r = int(min(rank, len(train), Xc.shape[1]))
    Px, Py = basis(Xc[train], r), basis(Yc[train], r)
    Xp, Yp = Xc @ Px, Yc @ Py                            # (k, r)

    U, _, Wt = np.linalg.svd(Yp[train].T @ Xp[train])
    R = U @ Wt                                           # (r, r) orthogonal

    # one global scale, fitted on training rows only
    YtR = Yp[train] @ R
    denom = float((YtR * YtR).sum())
    scale = float((YtR * Xp[train]).sum() / denom) if denom > 1e-12 else 1.0

    # Score in AMBIENT space, not in the projected coordinates. Because each arm gets its
    # own basis, a pure rotation is already absorbed by the bases themselves -- so an error
    # measured inside the projected space starts near zero and R appears to do nothing.
    # Mapping the transformed arm back into base's coordinates, Yhat = Yc Py R Px^T, makes
    # the before/after pair comparable and keeps the rank-r restriction honestly charged
    # against E_proc as reconstruction loss.
    Yhat = (Yc @ Py) @ R @ Px.T
    Yhat_s = scale * Yhat

    def err(idx):
        nx = max(float(np.linalg.norm(Xc[idx])), 1e-12)
        return (float(np.linalg.norm(Yc[idx] - Xc[idx]) / nx),
                float(np.linalg.norm(Yhat[idx] - Xc[idx]) / nx),
                float(np.linalg.norm(Yhat_s[idx] - Xc[idx]) / nx))

    raw_tr, proc_tr, scal_tr = err(train)
    raw_te, proc_te, scal_te = err(test)
    resid = Yhat[test] - Xc[test]
    return {"rank": r, "scale": scale,
            "train_raw": raw_tr, "train_proc": proc_tr, "train_proc_scaled": scal_tr,
            "test_raw": raw_te, "test_proc": proc_te, "test_proc_scaled": scal_te,
            "test_residual_norms": np.linalg.norm(resid, axis=-1).tolist(),
            "test_target_norms": np.linalg.norm(Xc[test], axis=-1).tolist()}


# ---------------------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------------------

def _dispersion_regime(rng, k, H, nq, sigma, label, naive_lo, naive_hi) -> None:
    """One noise regime of the dispersion self-test."""
    true_V = rng.normal(0, 1.0, (k, H))
    shared = rng.normal(0, 8.0, (1, H))
    Z_true = true_V - true_V.mean(0, keepdims=True)
    D_true = float((Z_true ** 2).sum(-1).mean())

    pos = (true_V + shared)[:, None, :] + rng.normal(0, sigma, (k, nq, H))
    neg = rng.normal(0, sigma, (k, nq, H))
    splits = half_splits(nq, 80, rng)

    d_naive = dispersion_naive(trait_vectors(pos, neg))
    d_cross = dispersion_crossfit(pos, neg, splits)
    print(f"[dispersion/{label:10s}] true={D_true:8.1f}  "
          f"naive={d_naive:9.1f} ({d_naive/D_true:5.2f}x)  "
          f"crossfit={d_cross:8.1f} ({d_cross/D_true:5.2f}x)")
    assert naive_lo <= d_naive / D_true <= naive_hi, (
        f"{label}: naive bias ratio should land in [{naive_lo}, {naive_hi}], "
        f"got {d_naive/D_true:.2f}x")
    assert abs(d_cross / D_true - 1) < 0.15, (
        f"{label}: cross-fitted dispersion should be unbiased, got {d_cross/D_true:.2f}x")


def _bootstrap_calibration(rng, k, H) -> None:
    """Two separate properties, which it is easy to conflate.

    (1) SELF-CONSISTENCY, per realisation: the point estimate must lie inside its own
        bootstrap CI. This is what the resample-then-split bug destroys, and it fails
        loudly -- an unbiased point estimate against inflated replicates.

    (2) COVERAGE, across realisations: the CI should contain the truth about 95% of the
        time. This is a repeated-sampling property and says nothing about any single
        draw -- a percentile bootstrap is centred on the point estimate, so one unlucky
        realisation legitimately misses. Checking it on a single draw, as an earlier
        version of this test did, is simply a miscalibrated test.
    """
    nq, sigma, B = 60, 6.0, 120
    true_V = rng.normal(0, 1.0, (k, H))
    shared = rng.normal(0, 8.0, (1, H))
    Z = true_V - true_V.mean(0, keepdims=True)
    D_true = float((Z ** 2).sum(-1).mean())
    pos = (true_V + shared)[:, None, :] + rng.normal(0, sigma, (k, nq, H))
    neg = rng.normal(0, sigma, (k, nq, H))

    point = dispersion_crossfit(pos, neg, half_splits(nq, 60, rng))
    reps = np.array([dispersion_crossfit(pos, neg, bootstrap_half_splits(nq, 6, rng))
                     for _ in range(B)])
    lo, hi = np.percentile(reps, [2.5, 97.5])

    bad = []
    for _ in range(B):
        idx = rng.integers(0, nq, nq)
        perm = rng.permutation(nq)
        bad.append(dispersion_crossfit(pos, neg,
                                       [(idx[perm[: nq // 2]], idx[perm[nq // 2:]])]))
    blo, bhi = np.percentile(bad, [2.5, 97.5])

    print(f"[bootstrap]  true={D_true:8.1f}  point={point:8.1f}  "
          f"CI=[{lo:8.1f},{hi:8.1f}]  naive-resample CI=[{blo:8.1f},{bhi:8.1f}]")
    assert lo <= point <= hi, "point estimate must lie inside its own bootstrap CI"
    assert blo > hi, ("the resample-then-split bootstrap should be visibly inflated; "
                      "if this fails the test has stopped proving anything")

    # coverage, over independent realisations
    kk, HH, nqq, sig, R, BB = 10, 128, 120, 4.0, 40, 60
    hits = 0
    for _ in range(R):
        tV = rng.normal(0, 1.0, (kk, HH))
        sh = rng.normal(0, 8.0, (1, HH))
        Zt = tV - tV.mean(0, keepdims=True)
        Dt = float((Zt ** 2).sum(-1).mean())
        po = (tV + sh)[:, None, :] + rng.normal(0, sig, (kk, nqq, HH))
        ne = rng.normal(0, sig, (kk, nqq, HH))
        rp = np.array([dispersion_crossfit(po, ne, bootstrap_half_splits(nqq, 4, rng))
                       for _ in range(BB)])
        l, h = np.percentile(rp, [2.5, 97.5])
        hits += int(l <= Dt <= h)
    cov = hits / R
    print(f"[bootstrap]  coverage over {R} realisations: {cov:.0%} (nominal 95%)")
    assert cov >= 0.80, f"bootstrap coverage {cov:.0%} is too low"


def _self_test() -> None:
    rng = np.random.default_rng(0)
    k, H = 10, 512
    # Two regimes, because the point is not "naive is always wrong" but "naive tracks the
    # noise level while cross-fitting does not". LOW: many questions, small noise -- the
    # two estimators should agree, and if they disagree the cross-fit is broken. HIGH: few
    # questions, large noise -- naive must inflate badly while cross-fit stays put. Real
    # data sits somewhere between, which is why the analysis reports both.
    # (label, n_questions, sigma, allowed range for the NAIVE estimator's bias ratio)
    for label, nq, sigma, naive_lo, naive_hi in (("low-noise", 400, 3.0, 1.00, 1.15),
                                                 ("high-noise", 40, 8.0, 1.80, 99.0)):
        _dispersion_regime(rng, k, H, nq, sigma, label, naive_lo, naive_hi)

    nq, sigma = 400, 3.0

    # ground truth: k persona vectors with known dispersion, plus a large shared component
    true_V = rng.normal(0, 1.0, (k, H))
    shared = rng.normal(0, 8.0, (1, H))             # the confound: big, common to all
    Z_true = true_V - true_V.mean(0, keepdims=True)
    D_true = float((Z_true ** 2).sum(-1).mean())

    # synthesise per-question activations whose (pos - neg) mean is true_V + shared
    def synth():
        pos = (true_V + shared)[:, None, :] + rng.normal(0, sigma, (k, nq, H))
        neg = rng.normal(0, sigma, (k, nq, H))
        return pos, neg

    pos, neg = synth()
    splits = half_splits(nq, 60, rng)
    V_hat = trait_vectors(pos, neg)

    # RDM: shared component must cancel exactly, and cross-fit must beat naive
    # RDM, exercised in the HIGH-noise regime: that is where the naive squared distance
    # inflates and where an unbiased estimator has to earn its keep. Every pair gets the
    # same additive noise-variance offset, so naive RDMs stay well RANK-correlated with
    # truth even when badly biased -- which is exactly why a Spearman on naive distances
    # can look reassuring while the absolute geometry is wrong. Both are reported.
    nq_hi, sigma_hi = 40, 8.0
    pos_hi = (true_V + shared)[:, None, :] + rng.normal(0, sigma_hi, (k, nq_hi, H))
    neg_hi = rng.normal(0, sigma_hi, (k, nq_hi, H))
    splits_hi = half_splits(nq_hi, 80, rng)

    true_rdm = rdm_naive(true_V)
    r_naive = rdm_naive(trait_vectors(pos_hi, neg_hi))
    r_cross = rdm_crossfit(pos_hi, neg_hi, splits_hi)
    b_naive = float(np.mean(r_naive / true_rdm))
    b_cross = float(np.mean(r_cross / true_rdm))
    print(f"[rdm]        naive bias={b_naive:5.2f}x  crossfit bias={b_cross:5.2f}x  "
          f"spearman(naive,true)={_spearman(r_naive, true_rdm):.4f}  "
          f"spearman(cross,true)={_spearman(r_cross, true_rdm):.4f}")
    assert b_naive > 1.5, f"naive RDM should inflate at high noise, got {b_naive:.2f}x"
    assert abs(b_cross - 1) < 0.20, f"cross-fitted RDM should be unbiased, got {b_cross:.2f}x"

    # shared component genuinely cancels in an RDM
    r_shift = rdm_naive(true_V + rng.normal(0, 5.0, (1, H)))
    assert np.allclose(r_shift, true_rdm), "RDM must be invariant to a common vector"
    print("[rdm]        invariance to a common additive vector: exact")

    # Procrustes: an exact rotation must be recovered, and CV must expose a fake one
    Q, _ = np.linalg.qr(rng.normal(0, 1, (H, H)))
    Xc = true_V - true_V.mean(0, keepdims=True)
    Yrot = Xc @ Q.T
    _, e_raw, e_proc = procrustes_lowrank(Yrot, Xc)
    print(f"[procrustes] true rotation: E_raw={e_raw:.3f} -> E_proc={e_proc:.2e}")
    assert e_proc < 1e-6 and e_raw > 0.5

    # 80-cell CV: a real global rotation generalises, unrelated noise does not
    # 80 cells with intrinsic rank 15 embedded in H dimensions. Isotropic noise would make
    # this test unwinnable by construction -- held-out rows would lie mostly outside any
    # basis fitted on training rows -- and real activations are strongly anisotropic, so
    # low intrinsic rank is both the fair and the realistic choice.
    n, intrinsic = 80, 15
    basis_true = np.linalg.qr(rng.normal(0, 1, (H, intrinsic)))[0]
    Xb = rng.normal(0, 1, (n, intrinsic)) @ basis_true.T
    Xb -= Xb.mean(0, keepdims=True)
    Yg = Xb @ Q.T                                    # genuinely one global rotation
    Yn = rng.normal(0, 1, (n, intrinsic)) @ np.linalg.qr(
        rng.normal(0, 1, (H, intrinsic)))[0].T       # unrelated configuration
    Yn -= Yn.mean(0, keepdims=True)
    tr, te = np.arange(0, 64), np.arange(64, 80)
    cg = procrustes_cv(Yg, Xb, tr, te, rank=20)
    cn = procrustes_cv(Yn, Xb, tr, te, rank=20)
    print(f"[procrustes] CV, true global rotation : test_raw={cg['test_raw']:.3f} -> "
          f"test_proc={cg['test_proc']:.3f}")
    print(f"[procrustes] CV, unrelated config     : test_raw={cn['test_raw']:.3f} -> "
          f"test_proc={cn['test_proc']:.3f}")
    assert cg["test_raw"] > 0.5, "sanity: the two arms must differ before alignment"
    assert cg["test_proc"] < 0.2, "a real global rotation must generalise to held-out rows"
    assert cn["test_proc"] > 0.8, "an unrelated configuration must NOT be rescued by CV"

    # the bootstrap must be calibrated: an unbiased point estimate has to land inside its
    # own interval, which is precisely what the naive resample-then-split version breaks
    _bootstrap_calibration(rng, k, H)

    print("\nall self-tests passed")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


if __name__ == "__main__":
    _self_test()

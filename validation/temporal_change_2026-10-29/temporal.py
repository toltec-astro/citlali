"""Experimental fixed-reference temporal test. No pipeline/data access.

Results are evidence, never detector admission. AR(1), conditional fixed
references, OLS and independent wild innovations are explicit assumptions.
"""
from dataclasses import dataclass
import numpy as np

OPTICAL_CUT = 0.10  # Preserved policy identity; not a temporal threshold.


def multiply(a, b):
    return np.einsum("ij,jk->ik", a, b, optimize=False)


def basis(a):
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    rank = np.count_nonzero(s > np.finfo(float).eps * max(a.shape) * s[0])
    return u[:, :rank], rank


def residual(q, y):
    return y - multiply(q, multiply(q.T, y))


@dataclass(frozen=True)
class Design:
    null: np.ndarray
    full: np.ndarray
    contrast: np.ndarray
    names: tuple
    linked: np.ndarray
    excitation: tuple


def prepare(reference, times, blocks):
    """Fixed source-free rows; integer times are original regular-grid slots.

    Missing slots and block boundaries reset the assumed AR process. No short
    block is removed. Rank deficiency returns unavailable, not weak response.
    """
    x, t, b = np.asarray(reference), np.asarray(times), np.asarray(blocks)
    if (x.ndim != 1 or t.shape != x.shape or b.shape != x.shape
            or len(x) < 4 or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(b))
            or np.any(np.diff(t) <= 0) or np.any(np.diff(b) < 0)):
        raise ValueError("finite aligned chronological rows required")
    ids = np.unique(b)
    offsets = (b[:, None] == ids).astype(float)
    q0, rank0 = basis(np.column_stack((offsets, x)))
    if rank0 != len(ids) + 1:
        return None
    patterns = [(b >= k).astype(float) for k in ids[1:]]
    patterns.append((t - t[0]) / (t[-1] - t[0]))
    z = residual(q0, x[:, None] * np.column_stack(patterns))
    zz = np.sum(z*z, axis=0)
    # Only numerical identifiability. No scientific excitation cut invented.
    if np.any(zz <= 128*np.finfo(float).eps*np.sum(x*x)):
        return None
    h = (z / zz).T
    q1, rank1 = basis(np.column_stack((offsets, x, z)))
    if rank1 != len(ids) + 1 + len(patterns) or len(x) <= rank1 + 2:
        return None
    linked = np.r_[False, (np.diff(t) == 1) & (np.diff(b) == 0)]
    excitation = tuple(float(np.sum((x[b == k] - np.mean(x[b == k]))**2))
                       for k in ids)
    return Design(q0, q1, h, tuple([f"step_{k}" for k in ids[1:]] + ["drift"]),
                  linked, excitation)


def wild_errors(q, y, linked, signs):
    """HC2 residuals, fitted AR(1), fixed-amplitude wild innovations.

    No clipping of an unstable fitted AR coefficient; return unavailable.
    Starts/gaps retain their observed residual amplitudes. Conditional on model,
    reference, finite residual amplitudes and estimated dependence.
    """
    r = residual(q, y[:, None])[:, 0]
    leverage = np.sum(q*q, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        return None
    r = r / np.sqrt(1-leverage)
    rows = np.flatnonzero(linked)
    denominator = float(np.sum(r[rows-1]**2))
    if denominator <= np.finfo(float).tiny:
        return None
    rho = float(np.sum(r[rows]*r[rows-1]) / denominator)
    if not np.isfinite(rho) or abs(rho) >= 1:
        return None
    innovation = r.copy()
    innovation[rows] -= rho*r[rows-1]
    e = innovation[:, None] * signs
    for i in range(1, len(y)):
        if linked[i]:
            e[i] += rho*e[i-1]
    return e, rho


def assess(designs, y, *, draws=999, seed=0, confidence=0.99):
    """Maximum search over all supplied panels, steps and drift.

    Simultaneous intervals concern the six individual projection contrasts,
    not independent physical step coefficients or a timewise gain envelope.
    Their true estimands can be calculated from a known synthetic mean.
    """
    y = np.asarray(y, dtype=float)
    if (y.ndim != 1 or not np.all(np.isfinite(y)) or draws < 99
            or not 0 < confidence < 1 or not designs):
        raise ValueError("finite vector, >=99 draws, confidence in (0,1) required")
    if any(d is None for d in designs):
        return {"available": False, "reason": "change_design_unidentifiable"}
    if any(d.null.shape[0] != len(y) for d in designs):
        raise ValueError("design and target lengths differ")
    rng = np.random.default_rng(seed)
    # Shared signs retain target-noise dependence between panel fits.
    signs = rng.choice([-1., 1.], size=(len(y), draws))
    values, null_draws, interval_draws, rho0, rho1 = [], [], [], [], []
    for d in designs:
        null = wild_errors(d.null, y, d.linked, signs)
        full = wild_errors(d.full, y, d.linked, signs)
        if null is None or full is None:
            return {"available": False, "reason": "dependence_or_variance_unavailable"}
        values.extend(multiply(d.contrast, y[:, None])[:, 0])
        null_draws.append(multiply(d.contrast, null[0]))
        interval_draws.append(multiply(d.contrast, full[0]))
        rho0.append(null[1])
        rho1.append(full[1])
    observed = np.asarray(values)
    null = np.vstack(null_draws)
    errors = np.vstack(interval_draws)
    null_scale = np.std(null, axis=1, ddof=1)
    scale = np.std(errors, axis=1, ddof=1)
    if np.any(null_scale <= 0) or np.any(scale <= 0):
        return {"available": False, "reason": "zero_contrast_variance"}
    statistic = float(np.max(np.abs(observed/null_scale)))
    maxima = np.max(np.abs(null/null_scale[:, None]), axis=0)
    p = float((1 + np.count_nonzero(maxima >= statistic))/(draws+1))
    critical = float(np.quantile(np.max(np.abs(errors/scale[:, None]), axis=0),
                                confidence, method="higher"))
    lower, upper = observed-critical*scale, observed+critical*scale
    if not np.all(np.isfinite(np.r_[observed, lower, upper, statistic, p])):
        raise ArithmeticError("nonfinite temporal evidence")
    return {"available": True, "p_value": p, "statistic": statistic,
            "estimate": observed.tolist(), "lower": lower.tolist(),
            "upper": upper.tolist(), "rho_null": rho0, "rho_full": rho1,
            "conditional_confidence": confidence, "draws": draws,
            "optical_cut_unchanged": OPTICAL_CUT,
            "admission": "not_decided", "harm_threshold": None}

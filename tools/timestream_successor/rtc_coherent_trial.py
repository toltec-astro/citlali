"""One offline experimental coherent plan. No accepted RTC plan or selector.

The global fitting projection FAILS five-second support eligibility. Arrays
are immutable; Apply accepts the exact input and an explicitly named overlay.
The existing C++ frozen RTC Apply supplies the independently checked LPF replay.
"""

from dataclasses import dataclass
import hashlib
import numpy as np
from scipy import signal


def readonly(x):
    a = np.array(x, copy=True)
    a.flags.writeable = False
    return a


def fingerprint(x):
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [
        (int(lo), int(hi))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def erode(mask, half):
    out = np.zeros_like(mask, dtype=bool)
    for lo, hi in runs(mask):
        if hi - lo > 2 * half:
            out[lo + half : hi - half if half else hi] = True
    return out


def centered(values, kernel):
    """Centered odd kernel. Callers separately require complete real support."""
    if len(kernel) % 2 != 1:
        raise ValueError("centered kernel must have odd length")
    if not np.isfinite(values).all() or not np.isfinite(kernel).all():
        raise ValueError("unexpected nonfinite filter input")
    return signal.convolve(
        values, np.asarray(kernel)[:, None], mode="same", method="direct"
    )


def native_demod(x, t, frequency, half):
    """Linear Hann local complex demodulation with local weighted DC removed.

    The even 488-sample descriptive D2 estimator is unchanged. This separately
    declared 489-sample centered candidate basis has exact symmetric support.
    No phase interpolation, gap joining, or extra free detector parameters.
    """
    h = np.hanning(2 * half + 1)
    h /= h.sum()
    carrier = np.exp(-2j * np.pi * frequency * t)
    # fftconvolve here computes a finite convolution, not a periodic filter.
    local = signal.fftconvolve(x, h[:, None], mode="same", axes=0)
    demod = signal.fftconvolve(x * carrier[:, None], h[:, None], mode="same", axes=0)
    dc = signal.fftconvolve(carrier, h, mode="same")
    return 2 * (demod - local * dc[:, None])


@dataclass(frozen=True)
class CoherentEvidence:
    input_sha256: str
    val_identity: str
    time_sha256: str
    support_sha256: str
    bases: tuple
    basis_good: tuple
    donor_weights: tuple
    groups: tuple
    training: np.ndarray
    frequency: float
    half: int

    @staticmethod
    def learn(original, time, good, groups, training, frequency, half, val_identity):
        if (
            original.ndim != 3
            or original.shape[2] != 2
            or good.shape != original.shape[:2]
        ):
            raise ValueError("paired native evidence shape mismatch")
        if not np.isfinite(original[good]).all() or not np.isfinite(time).all():
            raise ValueError("unexpected nonfinite admitted original")
        if (
            not val_identity
            or np.any(np.diff(time) <= 0)
            or training.shape != time.shape
        ):
            raise ValueError("missing VAL or native-time/training binding")
        if len(time) < 3 or np.max(np.diff(time)) > 1.001 * np.median(np.diff(time)):
            raise ValueError("this bounded candidate requires one physical native run")
        bases, supports, weights = [], [], []
        for donors in groups:
            if not donors or len(set(donors)) != len(donors):
                raise ValueError("invalid donor group")
            support = erode(np.all(good[:, donors], axis=1), half)
            fit = training & support
            if fit.sum() < 4 * (2 * half + 1):
                raise ValueError("insufficient guarded donor training support")
            z = native_demod(
                np.where(good[:, donors], original[:, donors, 0], 0),
                time,
                frequency,
                half,
            )
            _, _, vh = np.linalg.svd(z[fit], full_matrices=False)
            weight = vh[0].conj()
            q = np.einsum("ij,j->i", z, weight, optimize=False)
            b = q * np.exp(2j * np.pi * frequency * time)
            bases.append(readonly(np.column_stack([b.real, -b.imag])))
            supports.append(readonly(support))
            weights.append(readonly(weight))
        return CoherentEvidence(
            fingerprint(original),
            val_identity,
            fingerprint(time),
            fingerprint(good),
            tuple(bases),
            tuple(supports),
            tuple(weights),
            tuple(tuple(g) for g in groups),
            readonly(training),
            frequency,
            half,
        )


@dataclass(frozen=True)
class CoherentPlan:
    evidence: CoherentEvidence
    original: np.ndarray
    good: np.ndarray
    target_groups: tuple
    coefficients: tuple
    inverse_gram: tuple
    fit_support: tuple
    plan_id: str
    scientific_admission: bool = False
    five_second_projection: bool = False
    production: bool = False

    @staticmethod
    def consider(evidence, original, time, good, target_groups, val_identity, plan_id):
        if (
            evidence.input_sha256 != fingerprint(original)
            or evidence.time_sha256 != fingerprint(time)
            or evidence.support_sha256 != fingerprint(good)
            or evidence.val_identity != val_identity
            or not plan_id
        ):
            raise ValueError("stale exact original/time/support/VAL evidence")
        if len(target_groups) != len(evidence.groups):
            raise ValueError("missing target group")
        seen = set()
        coefficients, inverses, supports = [], [], []
        for gi, targets in enumerate(target_groups):
            if seen.intersection(targets) or set(targets).intersection(
                evidence.groups[gi]
            ):
                raise ValueError("target reused or included among its own donors")
            seen.update(targets)
            b = evidence.bases[gi]
            cs, invs, ts = [], [], []
            for d in targets:
                fit = evidence.training & evidence.basis_good[gi] & good[:, d]
                if fit.sum() < 4 * (2 * evidence.half + 1):
                    raise ValueError("insufficient guarded target training support")
                bt = b[fit]
                gram = np.einsum("ij,ik->jk", bt, bt, optimize=False)
                if np.linalg.cond(gram) > 1e8:
                    raise ValueError("unidentifiable quadrature coupling")
                inverse = np.linalg.inv(gram)
                rhs = np.einsum("ij,ik->jk", bt, original[fit, d], optimize=False)
                cs.append(
                    readonly(np.einsum("ij,jk->ik", inverse, rhs, optimize=False))
                )
                invs.append(readonly(inverse))
                ts.append(readonly(fit))
            coefficients.append(tuple(cs))
            inverses.append(tuple(invs))
            supports.append(tuple(ts))
        return CoherentPlan(
            evidence,
            readonly(original),
            readonly(good),
            tuple(tuple(g) for g in target_groups),
            tuple(coefficients),
            tuple(inverses),
            tuple(supports),
            plan_id,
        )

    def apply(self, original, val_identity, *, injection=None, response="template"):
        if (
            fingerprint(original) != self.evidence.input_sha256
            or val_identity != self.evidence.val_identity
        ):
            raise ValueError(
                "Apply requires exact original pair and VAL; no cumulative replay"
            )
        if response not in ("template", "projection"):
            raise ValueError("unknown frozen response")
        delta = np.zeros_like(original) if injection is None else injection
        if delta.shape != original.shape or not np.isfinite(delta).all():
            raise ValueError("invalid named paired overlay")
        result = np.array(original + delta, copy=True)
        support = self.good.copy()
        for gi, targets in enumerate(self.target_groups):
            b = self.evidence.bases[gi]
            for ti, d in enumerate(targets):
                c = self.coefficients[gi][ti]
                if response == "projection":
                    fit = self.fit_support[gi][ti]
                    rhs = np.einsum("ij,ik->jk", b[fit], delta[fit, d], optimize=False)
                    c = c + np.einsum(
                        "ij,jk->ik", self.inverse_gram[gi][ti], rhs, optimize=False
                    )
                result[:, d] -= np.einsum("ij,jk->ik", b, c, optimize=False)
                support[:, d] &= self.evidence.basis_good[gi]
        if not np.isfinite(result[support]).all():
            raise ValueError("nonfinite coherent Apply output")
        return result, support

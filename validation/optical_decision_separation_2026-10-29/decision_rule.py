"""Unactivated decision prototype; no Citlali imports, I/O or reduction route.

Inputs are dimensionless array-normalized conditional response intervals, in
full-A, full-B, early-A, early-B, late-A, late-B order. Temporal non-overlap is
a conservative review flag, not a calibrated hypothesis test.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Interval:
    lower: float | None
    upper: float | None

    def __post_init__(self):
        if self.lower is None and self.upper is None:
            return
        if (self.lower is None or self.upper is None
                or not isfinite(self.lower) or not isfinite(self.upper)
                or self.lower > self.upper):
            raise ValueError("interval must be unavailable or finite and ordered")

    @property
    def available(self):
        return self.lower is not None


@dataclass(frozen=True)
class Decision:
    optical: str
    temporal: str
    eligible: bool
    reasons: tuple[str, ...]


def consider(checks, threshold=0.10, *, prior_available=True,
             reference_available=True, disturbance_excluded=False):
    """Pure proposal decision; does not change learned evidence or coefficients."""
    if isinstance(threshold, bool) or not isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("response threshold must be finite and in (0, 1]")
    if len(checks) != 6 or any(not isinstance(c, Interval) for c in checks):
        raise ValueError("exactly six ordered Interval values required")
    if not prior_available:
        return Decision("prior_unavailable", "not_assessed", False, ("prior_unavailable",))
    if disturbance_excluded:
        return Decision("not_assessed", "not_assessed", False, ("disturbance_excluded",))
    if not reference_available:
        return Decision("unresolved", "not_assessed", False, ("required_reference_unavailable",))
    full, halves = checks[:2], checks[2:]
    if not all(c.available for c in full):
        return Decision("unresolved", "not_assessed", False, ("full_interval_unavailable",))
    low, high = min(c.lower for c in full), max(c.upper for c in full)
    if low > threshold:
        optical = "responsive"
    elif low > -threshold and high < threshold:
        optical = "weak"
    elif high < -threshold:
        optical = "reversed"
    else:
        optical = "unresolved"

    # Preserve every known temporal cause even if another half is missing.
    reasons = []
    if any(not c.available for c in halves):
        reasons.append("temporal_interval_unavailable")
    if any(c.available and c.upper <= threshold for c in halves):
        reasons.append("subset_response_at_or_below_cut")
    for panel in range(2):
        early, late = checks[2 + panel], checks[4 + panel]
        if early.available and late.available and (
                max(early.lower, late.lower) > min(early.upper, late.upper)):
            reasons.append(f"early_late_nonoverlap_panel_{panel}")
    if any(r.startswith("early_late_nonoverlap") for r in reasons):
        temporal = "difference_requires_review"
    elif "subset_response_at_or_below_cut" in reasons:
        temporal = "subset_below_requirement"
    elif "temporal_interval_unavailable" in reasons:
        temporal = "unassessed"
    elif all(c.lower > threshold for c in halves):
        temporal = "positive_subsets_no_detected_difference"
    else:
        temporal = "subset_uncertainty_only"
    eligible = optical == "responsive" and not reasons
    if optical != "responsive":
        reasons.insert(0, "full_response_" + optical)
    if eligible and temporal == "subset_uncertainty_only":
        reasons.append("eligible_with_temporal_uncertainty")
    return Decision(optical, temporal, eligible, tuple(reasons))

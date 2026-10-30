#pragma once
#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <memory>
#include <optional>
#include <string_view>

namespace citlali::pipeline {
class FruitResidual;
// Explicit experimental ablation; never a fallback for unavailable noise.
// Equal control retains the same variance-availability tests but uses coefficient
// 1 (mJy/nominal-beam)^-2 in cleaning and mapping. This arbitrary scale cancels
// from the map signal and is not measured precision.
enum class PointWeighting { inverse_noise, equal_control };

// Owner-approved empirical POINT policy. These are empirical analysis weights,
// in inverse squared calibrated brightness units, not a covariance model or a
// claim that the resulting map normalization is statistical precision.
inline constexpr std::string_view ptc_inverse_noise_policy =
    "owner-ptc-empirical-inverse-noise-2026-10-29-v1";
struct PtcDetectorNoiseEvidence {
    PointWeighting weighting=PointWeighting::inverse_noise;
    std::size_t eligible_count=0, source_free_count=0, reference_count=0,
        remaining_count=0;
    // Offset is mJy/nominal-beam, slope is dimensionless, variances are
    // (mJy/nominal-beam)^2, and coefficients have the reciprocal units.
    std::optional<double> reference_offset, reference_slope, reference_variance,
        remaining_variance, cleaning_coefficient, map_coefficient;
    std::string reference_reason, remaining_reason;
};
struct PtcInverseNoiseResult {
    // Full input axes, physical mJy/nominal-beam. Excluded storage is zero and
    // carries a nonzero cause; it is never an observed zero.
    PtcApplied cleaned;
    PtcMask effective_eligible;
    std::vector<PtcDetectorNoiseEvidence> detectors;
    // Compact solver columns map to the full detector axis. Prepared/basis
    // values are whitened; their physical units must not be inferred from CAL.
    std::vector<std::size_t> fit_columns;
    PtcPrepared prepared;
    PtcFit fit;
    std::string scope;
};

// One network and physical processing segment, scheduled slots x detectors.
// eligible already contains immutable detector-health and CAL support facts.
// source_free is a separate binary AST training permission, never a MAP veto.
// Reference: median of the other eligible, source-free detector residuals after
// each donor's source-free mean is removed. An offset+slope fit gives full
// residual SSE/(N-2), without clipping, caps, floors, or duration cuts.
// Cleaning whitens with 1/sqrt(v_ref), uses the existing requested PTC solver,
// and returns to calibrated units. v_remaining is ordinary source-free sample
// variance of the cleaned residual, before source restoration; MAP uses
// 1/max(v_ref,v_remaining). Unsupported estimates remain unavailable.
PtcInverseNoiseResult inverse_noise_ptc(const PtcMatrix &values,
    const PtcMask &eligible,const PtcMask &source_free,PtcSolverRequest,
    std::string scope="", PointWeighting weighting=PointWeighting::inverse_noise);

struct PtcInverseNoiseFruitResult {
    PtcInverseNoiseResult noise;
    PtcMatrix total;
    std::shared_ptr<const FruitResidual> residual;
};
// Native composition owns exact subtraction/add-back lineage. Each call learns
// afresh from this immutable CAL-minus-current-model residual; a previous
// cleaned product cannot substitute for that parent.
PtcInverseNoiseFruitResult inverse_noise_fruit(
    std::shared_ptr<const FruitResidual>,const PtcMask &source_free,
    PtcSolverRequest,std::string scope="", PointWeighting weighting=PointWeighting::inverse_noise);
} // namespace citlali::pipeline

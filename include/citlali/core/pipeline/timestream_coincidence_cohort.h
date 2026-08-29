#pragma once

#include <citlali/core/pipeline/timestream_native_sample.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace citlali::pipeline {

// Compact state labels shared by native alignment and the production PTC
// adapter. They describe present state; they are not retained per-cell
// operation history.
enum class CoincidenceCellState {
    mapped_valid,
    mapped_invalid,
    absent,
};

enum class CoincidenceAbsenceReason {
    no_candidate,
    outside_tolerance,
    outside_native_support,
    participant_unavailable,
};

// This value exists only inside an excluded-cell PCA working buffer. It is
// not a detector sample and must never be scattered.
class FinitePcaPlaceholder {
public:
    static FinitePcaPlaceholder checked(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "PCA exclusion placeholder must be finite");
        }
        return FinitePcaPlaceholder{value};
    }

    double value() const noexcept { return value_; }

private:
    explicit FinitePcaPlaceholder(double value) : value_{value} {}
    double value_;
};

enum class PcaCompatibilityHazard : std::uint8_t {
    null_model = 1U << 0U,
    adaptive_selector = 1U << 1U,
    band_limited_marchenko_pastur = 1U << 2U,
};

struct PcaCompatibilityInputs {
    bool null_model_active_for_operation = false;
    bool adaptive_selector_active_for_operation = false;
    bool marchenko_pastur_active_for_operation = false;
    bool marchenko_pastur_band_requested = false;
};

class PcaCompatibilityClassification {
public:
    bool compatible() const noexcept { return hazards_ == 0; }

    bool has(PcaCompatibilityHazard hazard) const noexcept {
        return (hazards_ & static_cast<std::uint8_t>(hazard)) != 0;
    }

private:
    friend PcaCompatibilityClassification classify_pca_compatibility(
        bool, const PcaCompatibilityInputs &);
    explicit PcaCompatibilityClassification(std::uint8_t hazards)
        : hazards_{hazards} {}
    std::uint8_t hazards_ = 0;
};

inline PcaCompatibilityClassification classify_pca_compatibility(
    bool has_excluded_cells, const PcaCompatibilityInputs &inputs) {
    std::uint8_t hazards = 0;
    if (has_excluded_cells && inputs.null_model_active_for_operation) {
        hazards |=
            static_cast<std::uint8_t>(PcaCompatibilityHazard::null_model);
    }
    if (has_excluded_cells &&
        inputs.adaptive_selector_active_for_operation) {
        hazards |= static_cast<std::uint8_t>(
            PcaCompatibilityHazard::adaptive_selector);
    }
    if (has_excluded_cells &&
        inputs.marchenko_pastur_active_for_operation &&
        inputs.marchenko_pastur_band_requested) {
        hazards |= static_cast<std::uint8_t>(
            PcaCompatibilityHazard::band_limited_marchenko_pastur);
    }
    return PcaCompatibilityClassification{hazards};
}

inline void require_pca_compatibility(
    const PcaCompatibilityClassification &classification) {
    if (!classification.compatible()) {
        throw std::logic_error(
            "excluded cohort cells are incompatible with the requested "
            "optional PCA mode; no fallback is selected");
    }
}

}  // namespace citlali::pipeline

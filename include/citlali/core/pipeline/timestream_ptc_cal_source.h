#pragma once

#include <citlali/core/pipeline/timestream_cal_pipeline.h>

namespace citlali::pipeline {

// Exact PTC input binding, not a PTC fit or a scientific-use decision. CAL owns
// values and classification; the retained RTC/ALIGN/AST parents own timing,
// detector identity, support, replacements, coordinates and response history.
// PTC must separately request and resolve each learning/application use.
class PtcCalSource {
public:
    static PtcCalSource bind(std::shared_ptr<const CalAppliedSignal> signal,
                            std::shared_ptr<const ValSnapshot> snapshot) {
        if (!signal || !snapshot ||
            !snapshot->committed_cal_output_facts_handle() ||
            snapshot->committed_cal_output_facts_handle()->signal_handle().get() != signal.get() ||
            snapshot->parent_snapshot_handle().get() != signal->plan_handle()->snapshot_handle().get())
            throw std::invalid_argument(
                "PTC source requires the exact calibrated realization and its CAL output VAL generation");
        return PtcCalSource{std::move(signal), std::move(snapshot)};
    }

    const auto &signal_handle() const noexcept { return signal_; }
    const auto &val_snapshot_handle() const noexcept { return snapshot_; }
    const auto &grid_handle() const noexcept {
        return signal_->plan_handle()->evidence_handle()->source().rtc_terminal_handle()->grid_handle();
    }
    const auto &detector_factors() const noexcept {
        return signal_->plan_handle()->evidence_handle()->factors();
    }
    const auto &opacity_quality() const noexcept {
        return signal_->plan_handle()->evidence_handle()->opacity_quality();
    }
    std::optional<double> value(std::size_t detector, std::size_t slot) const {
        return signal_->value(detector, slot);
    }
    std::uint16_t causes(std::size_t detector, std::size_t slot) const {
        return snapshot_->committed_cal_output_facts_handle()->at(signal_, detector, slot);
    }
    // A stable scheduled slot remains present even when its value is absent.
    // No compact finite-value offset is promoted to a time/sample identity.
    auto occurrence(std::size_t detector, std::size_t slot) const {
        return grid_handle()->occurrence(detector, slot);
    }
    static constexpr std::string_view unit = CalAppliedSignal::unit;
    static constexpr bool ptc_use_admitted = false;
    static constexpr bool r_is_primary_input = false;

private:
    PtcCalSource(std::shared_ptr<const CalAppliedSignal> signal,
                 std::shared_ptr<const ValSnapshot> snapshot)
        : signal_{std::move(signal)}, snapshot_{std::move(snapshot)} {}
    std::shared_ptr<const CalAppliedSignal> signal_;
    std::shared_ptr<const ValSnapshot> snapshot_;
};

} // namespace citlali::pipeline

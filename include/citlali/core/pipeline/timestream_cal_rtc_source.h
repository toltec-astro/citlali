#pragma once

#include <citlali/core/pipeline/timestream_rtc_terminal.h>

namespace citlali::pipeline {

enum class CalRtcAdmissionState : std::uint8_t { not_requested };

// CAL-facing source content, not CAL scientific admission or a calibrated
// product. The future requested CAL operation must admit its factor/atmosphere,
// AST coordinate, response and VAL named-use requirements before producing data.
class CalRtcSource {
public:
    static CalRtcSource bind(std::shared_ptr<const RtcPipelineTerminal> terminal) {
        if (!terminal || !terminal->val_snapshot_handle()->committed_rtc_output_facts_handle() ||
            terminal->val_snapshot_handle()->committed_rtc_output_facts_handle()->grid_handle().get() !=
                terminal->grid_handle().get())
            throw std::invalid_argument("CAL source requires a complete exact RTC terminal");
        return CalRtcSource{std::move(terminal)};
    }
    const auto &rtc_terminal_handle() const noexcept { return terminal_; }
    const auto &val_snapshot_handle() const noexcept { return terminal_->val_snapshot_handle(); }
    std::optional<double> conditioned_x(std::size_t detector, std::size_t slot) const {
        return terminal_->grid_handle()->value(detector, slot, NativeReadoutCoordinate::x);
    }
    static constexpr auto admission = CalRtcAdmissionState::not_requested;
    static constexpr bool calibrated = false, r_is_calibration_input = false;
private:
    explicit CalRtcSource(std::shared_ptr<const RtcPipelineTerminal> terminal) : terminal_{std::move(terminal)} {}
    std::shared_ptr<const RtcPipelineTerminal> terminal_;
};

} // namespace citlali::pipeline

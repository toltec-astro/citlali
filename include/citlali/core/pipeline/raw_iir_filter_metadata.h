#pragma once

namespace citlali::pipeline {

struct RawIirFilterMetadata {
    bool enabled = false;
    double frequency_hz = 0.0;
    int order = 1;
    bool zero_phase = false;
};

template <class IirFilterConfig>
RawIirFilterMetadata raw_iir_filter_metadata(
    const IirFilterConfig &config) {
    if (!config.enabled) {
        return {};
    }
    return {
        true, config.freq_Hz, config.order, config.zero_phase};
}

}  // namespace citlali::pipeline

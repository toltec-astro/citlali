#pragma once

// Included by phdu_rtc_config.h inside namespace citlali::pipeline.

template <class FitsEntry, class LocalResidual, class Logger>
void add_phdu_rtc_local_despike_config(FitsEntry &fits_entry,
                                       const std::string &array_name,
                                       const Logger &logger,
                                       const LocalResidual &local_residual) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("CONFIG.DESPIKE.LOCAL.ENABLED",
               local_residual.enabled,
               "Enable local-residual RTC despike pass");
    add_double_key("CONFIG.DESPIKE.LOCAL.WINDOW_SEC",
                   local_residual.window_sec,
                   "Local-residual despike smoothing window");
    add_double_key("CONFIG.DESPIKE.LOCAL.SIGMA_SCALE",
                   local_residual.sigma_scale,
                   "Local-residual despike raw threshold scale");
    add_double_key("CONFIG.DESPIKE.LOCAL.DELTA_SIGMA_SCALE",
                   local_residual.delta_sigma_scale,
                   "Local-residual despike delta threshold scale");
    hdu.addKey("CONFIG.DESPIKE.LOCAL.EXPAND_WITH_FILTER",
               local_residual.expand_with_filter,
               "Expand local-residual flags by TOD filter window");
    add_double_key("CONFIG.DESPIKE.LOCAL.EVENT_PADDING_SEC",
                   local_residual.event_padding_sec,
                   "Padding around accepted compact local-residual events");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.MAX_ADDED_FLAGGED_FRAC",
        local_residual.max_added_flagged_fraction,
        "Reject local-residual proposals above this added flagged fraction");
    hdu.addKey("CONFIG.DESPIKE.LOCAL.RAW_GATE.ENABLED",
               local_residual.compact_raw_gate.enabled,
               "Enable compact morphology gate for local-residual raw candidates");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_REL_SIGMA_SCALE",
        local_residual.compact_raw_gate.candidate_rel_sigma_scale,
        "Candidate threshold scale relative to the accepted local-residual raw threshold");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_SIGMA_SCALE",
        local_residual.compact_raw_gate.candidate_rel_sigma_scale *
            local_residual.sigma_scale,
        "Effective candidate threshold scale in units of min_spike_sigma for compact local-residual raw gate");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.RAW_GATE.WINDOW_SEC",
        local_residual.compact_raw_gate.window_sec,
        "Window used to score compactness of local-residual raw candidates");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.RAW_GATE.HALF_PEAK_FRAC",
        local_residual.compact_raw_gate.half_peak_frac,
        "Half-peak fraction used to measure local-residual raw candidate width");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_WIDTH_SEC",
        local_residual.compact_raw_gate.max_width_sec,
        "Maximum width allowed for compact local-residual raw candidates");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_STEP_SHIFT_Z",
        local_residual.compact_raw_gate.max_step_shift_z,
        "Maximum allowed pre/post baseline shift for compact local-residual raw candidates");
    hdu.addKey(
        "CONFIG.DESPIKE.LOCAL.DELTA_GATE.ENABLED",
        local_residual.compact_delta_gate.enabled,
        "Enable compact morphology gate for local-residual delta candidates");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.DELTA_GATE.WINDOW_SEC",
        local_residual.compact_delta_gate.window_sec,
        "Window used to score compactness of local-residual delta candidates");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.DELTA_GATE.HALF_PEAK_FRAC",
        local_residual.compact_delta_gate.half_peak_frac,
        "Half-peak fraction used to measure local-residual delta candidate width");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_WIDTH_SEC",
        local_residual.compact_delta_gate.max_width_sec,
        "Maximum width allowed for compact local-residual delta candidates");
    add_double_key(
        "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_STEP_SHIFT_Z",
        local_residual.compact_delta_gate.max_step_shift_z,
        "Maximum allowed pre/post baseline shift for compact local-residual delta candidates");
}


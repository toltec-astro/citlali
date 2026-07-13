#pragma once

// Included by phdu_reduction_config.h inside namespace citlali::pipeline.

template <class FitsEntry, class Logger>
void add_phdu_weight_selection_config(FitsEntry &fits_entry,
                                      const std::string &array_name,
                                      const Logger &logger,
                                      const citlali::config::RawTimeChunkFlaggingConfig
                                          &raw_flagging,
                                      const citlali::config::ProcessedTimeChunkConfig
                                          &processed_config) {
    const auto &flagging = processed_config.flagging;
    const auto &weighting = processed_config.weighting;
    const auto &validation = weighting.validation;
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("CONFIG.WEIGHT.TYPE",
               std::string{citlali::config::to_string(weighting.type)},
               "Weighting scheme");
    add_double_key("CONFIG.INV_VAR.RTC.WTLOW",
                   raw_flagging.lower_tod_inv_var_factor,
                   "RTC lower inv var cutoff");
    add_double_key("CONFIG.INV_VAR.RTC.WTHIGH",
                   raw_flagging.upper_tod_inv_var_factor,
                   "RTC upper inv var cutoff");
    add_double_key("CONFIG.INV_VAR.PTC.WTLOW",
                   flagging.lower_tod_inv_var_factor,
                   "PTC lower inv var cutoff");
    add_double_key("CONFIG.INV_VAR.PTC.WTHIGH",
                   flagging.upper_tod_inv_var_factor,
                   "PTC upper inv var cutoff");
    add_double_key("CONFIG.WEIGHT.PTC.WTLOW",
                   weighting.lower_map_weight_factor,
                   "PTC lower weight cutoff");
    add_double_key("CONFIG.WEIGHT.PTC.WTHIGH",
                   weighting.upper_map_weight_factor,
                   "PTC upper weight cutoff");
    add_double_key("CONFIG.WEIGHT.MEDWTFACTOR",
                   weighting.median_map_weight_factor,
                   "Median weight factor");
    add_double_key("CONFIG.WEIGHT.SRCMASK_ARCSEC",
                   weighting.source_mask_radius_arcsec,
                   "Source mask radius for full-weight variance estimation");
    add_double_key("CONFIG.WEIGHT.HYBRID_MIN",
                   weighting.hybrid_correction_min_factor,
                   "Minimum hybrid residual-variance correction factor");
    add_double_key("CONFIG.WEIGHT.HYBRID_MAX",
                   weighting.hybrid_correction_max_factor,
                   "Maximum hybrid residual-variance correction factor");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.ENABLED",
               validation.enabled,
               "Enable validated detector-weight penalties");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.ACCUM_ITERS",
               validation.accumulation_iters,
               "Fruitloops iterations used to learn penalties");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.APPLY_ITER",
               validation.apply_start_iter,
               "Earliest fruitloops iter applying penalties");
    add_double_key("CONFIG.WEIGHT.VALIDATION.MIN_FACTOR",
                   validation.min_factor,
                   "Minimum validated detector weight factor");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.UPWARD_ENABLED",
               validation.upward_enabled,
               "Allow validated upward weight factors");
    add_double_key("CONFIG.WEIGHT.VALIDATION.UPWARD_MAX",
                   validation.upward_max_factor,
                   "Maximum validated upward weight factor");
    add_double_key("CONFIG.WEIGHT.VALIDATION.UPWARD_POWER",
                   validation.upward_power,
                   "Power for validated upward weight factor");
    add_double_key("CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_BASE",
                   validation.upward_min_base_factor,
                   "Minimum one-sided factor for upward validation");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.UPWARD_REQ_ATM",
               validation.upward_require_atmospheric,
               "Require atmospheric gate for upward factors");
    add_double_key("CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_ATM",
                   validation.upward_min_atmospheric_factor,
                   "Minimum atmospheric factor for upward validation");
}

template <class FitsEntry>
void add_phdu_initial_runtime_config(FitsEntry &fits_entry,
                                     bool verbose_mode,
                                     bool run_polarization,
                                     bool run_despike) {
    auto &hdu = fits_entry.pfits->pHDU();
    hdu.addKey("CONFIG.VERBOSE", verbose_mode, "Reduced in verbose mode");
    hdu.addKey("CONFIG.POLARIZED", run_polarization, "Polarized Obs");
    hdu.addKey("CONFIG.DESPIKED", run_despike, "Despiked");
}

template <class FitsEntry, class RawTimeChunkConfig, class Logger>
void add_phdu_tod_filter_config(FitsEntry &fits_entry,
                                const std::string &array_name,
                                const Logger &logger,
                                const RawTimeChunkConfig &config,
                                bool run_any_tod_filter) {
    auto &hdu = fits_entry.pfits->pHDU();
    const auto iir = raw_iir_filter_metadata(config.iir_filter);
    hdu.addKey("CONFIG.TODFILTERED", run_any_tod_filter, "TOD Filtered");
    hdu.addKey("CONFIG.TODNOTCH", config.filter.notch.enabled,
               "TOD notch enabled");
    hdu.addKey("CONFIG.TODIIRHP", iir.enabled,
               "TOD IIR highpass enabled");
    add_phdu_double_key(fits_entry, array_name, logger,
                        "CONFIG.TODIIRHP.FREQ_HZ",
                        iir.frequency_hz,
                        "TOD IIR highpass cutoff frequency");
    hdu.addKey("CONFIG.TODIIRHP.ORDER", iir.order,
               "TOD IIR highpass cascaded order");
    hdu.addKey("CONFIG.TODIIRHP.ZEROPHASE", iir.zero_phase,
               "TOD IIR highpass forward-backward");
}

template <class FitsEntry, class EdgeGuard, class OuterContext>
void add_phdu_tod_edge_guard_config(FitsEntry &fits_entry,
                                    const EdgeGuard &edge_guard,
                                    OuterContext outer_context_samples) {
    auto &hdu = fits_entry.pfits->pHDU();
    hdu.addKey("CONFIG.TODFILTER.EDGE_GUARD.ENABLED", edge_guard.enabled,
               "TOD filter edge guard enabled");
    hdu.addKey("CONFIG.TODFILTER.EDGE_GUARD.MODE", edge_guard.mode,
               "TOD filter edge guard mode");
    hdu.addKey("CONFIG.TODFILTER.EDGE_GUARD.COMBINE", edge_guard.combine,
               "TOD filter edge guard combine rule");
    hdu.addKey("CONFIG.TODFILTER.EDGE_GUARD.CONTEXT_SAMPLES",
               static_cast<int>(edge_guard.context_samples),
               "TOD filter context samples");
    hdu.addKey("CONFIG.TODFILTER.EDGE_GUARD.GUARD_SAMPLES",
               static_cast<int>(edge_guard.guard_samples),
               "TOD filter guarded samples per edge");
    hdu.addKey("CONFIG.TOD.OUTER_CONTEXT_SAMPLES",
               static_cast<int>(outer_context_samples),
               "TOD loaded outer context samples");
}

template <class FitsEntry, class RawTimeChunkConfig>
void add_phdu_tod_processing_config(FitsEntry &fits_entry,
                                    const RawTimeChunkConfig &config) {
    auto &hdu = fits_entry.pfits->pHDU();
    hdu.addKey("CONFIG.DOWNSAMPLED", config.downsample.enabled,
               "Downsampled");
    hdu.addKey("CONFIG.CALIBRATED", config.flux_calibration_enabled,
               "Calibrated");
    hdu.addKey("CONFIG.EXTINCTION", config.extinction_correction_enabled,
               "Extinction corrected");
    hdu.addKey("CONFIG.EXTINCTION.EXTMODEL",
               config.extinction_model,
               "Extinction model");
}

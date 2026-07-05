#pragma once

// Included by phdu_reduction_config.h inside namespace citlali::pipeline.

template <class FitsEntry, class PtcProc, class RtcProc, class Logger>
void add_phdu_weight_selection_config(FitsEntry &fits_entry,
                                      const std::string &array_name,
                                      const Logger &logger,
                                      const PtcProc &ptcproc,
                                      const RtcProc &rtcproc) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("CONFIG.WEIGHT.TYPE", ptcproc.weighting_type,
               "Weighting scheme");
    add_double_key("CONFIG.INV_VAR.RTC.WTLOW", rtcproc.lower_inv_var_factor,
                   "RTC lower inv var cutoff");
    add_double_key("CONFIG.INV_VAR.RTC.WTHIGH", rtcproc.upper_inv_var_factor,
                   "RTC upper inv var cutoff");
    add_double_key("CONFIG.INV_VAR.PTC.WTLOW", ptcproc.lower_inv_var_factor,
                   "PTC lower inv var cutoff");
    add_double_key("CONFIG.INV_VAR.PTC.WTHIGH", ptcproc.upper_inv_var_factor,
                   "PTC upper inv var cutoff");
    add_double_key("CONFIG.WEIGHT.PTC.WTLOW", ptcproc.lower_weight_factor,
                   "PTC lower weight cutoff");
    add_double_key("CONFIG.WEIGHT.PTC.WTHIGH", ptcproc.upper_weight_factor,
                   "PTC upper weight cutoff");
    add_double_key("CONFIG.WEIGHT.MEDWTFACTOR", ptcproc.med_weight_factor,
                   "Median weight factor");
    add_double_key("CONFIG.WEIGHT.SRCMASK_ARCSEC",
                   ptcproc.source_mask_radius_arcsec,
                   "Source mask radius for full-weight variance estimation");
    add_double_key("CONFIG.WEIGHT.HYBRID_MIN",
                   ptcproc.hybrid_correction_min_factor,
                   "Minimum hybrid residual-variance correction factor");
    add_double_key("CONFIG.WEIGHT.HYBRID_MAX",
                   ptcproc.hybrid_correction_max_factor,
                   "Maximum hybrid residual-variance correction factor");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.ENABLED",
               ptcproc.weight_validation.enabled,
               "Enable validated detector-weight penalties");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.ACCUM_ITERS",
               ptcproc.weight_validation.accumulation_iters,
               "Fruitloops iterations used to learn penalties");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.APPLY_ITER",
               ptcproc.weight_validation.apply_start_iter,
               "Earliest fruitloops iter applying penalties");
    add_double_key("CONFIG.WEIGHT.VALIDATION.MIN_FACTOR",
                   ptcproc.weight_validation.min_factor,
                   "Minimum validated detector weight factor");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.UPWARD_ENABLED",
               ptcproc.weight_validation.upward_enabled,
               "Allow validated upward weight factors");
    add_double_key("CONFIG.WEIGHT.VALIDATION.UPWARD_MAX",
                   ptcproc.weight_validation.upward_max_factor,
                   "Maximum validated upward weight factor");
    add_double_key("CONFIG.WEIGHT.VALIDATION.UPWARD_POWER",
                   ptcproc.weight_validation.upward_power,
                   "Power for validated upward weight factor");
    add_double_key("CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_BASE",
                   ptcproc.weight_validation.upward_min_base_factor,
                   "Minimum one-sided factor for upward validation");
    hdu.addKey("CONFIG.WEIGHT.VALIDATION.UPWARD_REQ_ATM",
               ptcproc.weight_validation.upward_require_atmospheric,
               "Require atmospheric gate for upward factors");
    add_double_key("CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_ATM",
                   ptcproc.weight_validation.upward_min_atmospheric_factor,
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

template <class FitsEntry, class RtcProc, class Logger>
void add_phdu_tod_filter_runtime_config(FitsEntry &fits_entry,
                                        const std::string &array_name,
                                        const Logger &logger,
                                        const RtcProc &rtcproc,
                                        bool run_any_tod_filter) {
    auto &hdu = fits_entry.pfits->pHDU();
    hdu.addKey("CONFIG.TODFILTERED", run_any_tod_filter, "TOD Filtered");
    hdu.addKey("CONFIG.TODNOTCH", rtcproc.run_tod_notch,
               "TOD notch enabled");
    hdu.addKey("CONFIG.TODIIRHP", rtcproc.run_tod_iir_highpass,
               "TOD IIR highpass enabled");
    add_phdu_double_key(fits_entry, array_name, logger,
                        "CONFIG.TODIIRHP.FREQ_HZ",
                        rtcproc.filter.iir_highpass_freq_Hz,
                        "TOD IIR highpass cutoff frequency");
    hdu.addKey("CONFIG.TODIIRHP.ORDER", rtcproc.filter.iir_highpass_order,
               "TOD IIR highpass cascaded order");
    hdu.addKey("CONFIG.TODIIRHP.ZEROPHASE",
               rtcproc.filter.iir_highpass_zero_phase,
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

template <class FitsEntry, class RtcProc>
void add_phdu_tod_processing_config(FitsEntry &fits_entry,
                                    const RtcProc &rtcproc) {
    auto &hdu = fits_entry.pfits->pHDU();
    hdu.addKey("CONFIG.DOWNSAMPLED", rtcproc.run_downsample,
               "Downsampled");
    hdu.addKey("CONFIG.CALIBRATED", rtcproc.run_calibrate,
               "Calibrated");
    hdu.addKey("CONFIG.EXTINCTION", rtcproc.run_extinction,
               "Extinction corrected");
    hdu.addKey("CONFIG.EXTINCTION.EXTMODEL",
               rtcproc.calibration.extinction_model,
               "Extinction model");
}


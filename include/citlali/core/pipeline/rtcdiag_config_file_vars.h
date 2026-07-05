#pragma once

// Included by rtcdiag_layout_config.h inside namespace citlali::pipeline.

template <class RtcProc, class ReductionLearning, class OuterContext>
void add_rtcdiag_file_config_vars(netCDF::NcFile &fo,
                                  const RtcProc &rtcproc,
                                  const ReductionLearning &learning,
                                  bool verbose_mode,
                                  OuterContext outer_context_samples,
                                  double rtc_sample_rate_hz) {
    add_netcdf_var(fo, "RTC_SAMPRATE", rtc_sample_rate_hz);
    add_netcdf_var(fo, "CONFIG.TODFILTERED", rtcproc.run_tod_filter);
    add_netcdf_var(fo, "CONFIG.TODFILTER.FREQ_HIGH_HZ",
                   rtcproc.filter.freq_high_Hz);
    add_netcdf_var(fo, "CONFIG.TODFILTER.FREQ_LOW_HZ",
                   rtcproc.filter.freq_low_Hz);
    add_netcdf_var(fo, "CONFIG.TODFILTER.N_TERMS",
                   rtcproc.filter.n_terms);
    add_tod_filter_edge_guard_config_vars(
        fo, rtcproc.filter_edge_guard, outer_context_samples,
        rtcproc.tod_output_outer_context_samples);

    // Keep a compact provenance subset so rtcdiag is interpretable without the RTC TOD.
    add_netcdf_var(fo, "CONFIG.VERBOSE", verbose_mode);
    add_reduction_learning_config_vars(fo, learning, false);
    add_netcdf_var(fo, "CONFIG.DESPIKED", rtcproc.run_despike);
    add_rtc_local_despike_config_vars(fo, rtcproc.despiker.local_residual);
    add_rtc_event_mask_config_vars(fo, rtcproc);
    add_rtc_line_audit_config_vars(fo, rtcproc.line_audit);
    add_netcdf_var(fo, "CONFIG.INV_VAR.WINDOW_SEC",
                   rtcproc.remove_bad_dets_window_sec);
}

template <class LineAudit>
void add_rtc_line_audit_config_vars_if_absent(
    netCDF::NcFile &fo, const LineAudit &line_audit) {
    if (fo.getVar("CONFIG.RTC.LINE_AUDIT.ENABLED").isNull()) {
        add_rtc_line_audit_config_vars(fo, line_audit);
    }
}


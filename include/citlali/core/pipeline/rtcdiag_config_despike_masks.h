#pragma once

// Included by rtcdiag_layout_config.h inside namespace citlali::pipeline.

template <class LocalResidual>
void add_rtc_local_despike_config_vars(
    netCDF::NcFile &fo, const LocalResidual &local_residual) {
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.ENABLED",
                   local_residual.enabled);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.WINDOW_SEC",
                   local_residual.window_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.SIGMA_SCALE",
                   local_residual.sigma_scale);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_SIGMA_SCALE",
                   local_residual.delta_sigma_scale);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.EXPAND_WITH_FILTER",
                   local_residual.expand_with_filter);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.EVENT_PADDING_SEC",
                   local_residual.event_padding_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.HIGH_SCORE_EVENT_OVERRIDE",
                   local_residual.high_score_event_override);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.MAX_ADDED_FLAGGED_FRAC",
                   local_residual.max_added_flagged_fraction);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.ENABLED",
                   local_residual.compact_raw_gate.enabled);
    add_netcdf_var(
        fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_REL_SIGMA_SCALE",
        local_residual.compact_raw_gate.candidate_rel_sigma_scale);
    add_netcdf_var(
        fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_SIGMA_SCALE",
        local_residual.compact_raw_gate.candidate_rel_sigma_scale *
            local_residual.sigma_scale);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.WINDOW_SEC",
                   local_residual.compact_raw_gate.window_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.HALF_PEAK_FRAC",
                   local_residual.compact_raw_gate.half_peak_frac);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_WIDTH_SEC",
                   local_residual.compact_raw_gate.max_width_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_STEP_SHIFT_Z",
                   local_residual.compact_raw_gate.max_step_shift_z);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.ENABLED",
                   local_residual.compact_delta_gate.enabled);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.WINDOW_SEC",
                   local_residual.compact_delta_gate.window_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.HALF_PEAK_FRAC",
                   local_residual.compact_delta_gate.half_peak_frac);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_WIDTH_SEC",
                   local_residual.compact_delta_gate.max_width_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_STEP_SHIFT_Z",
                   local_residual.compact_delta_gate.max_step_shift_z);
}

template <class StepMask>
void add_rtc_step_mask_config_vars(netCDF::NcFile &fo,
                                   const StepMask &step_mask) {
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.ENABLED",
                   step_mask.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.STEP_WINDOW_SEC",
                   step_mask.step_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.STEP_SCORE_THRESH",
                   step_mask.step_score_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_GOOD_FRAC",
                   step_mask.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_DET_USED",
                   step_mask.min_det_used);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_STEP_DET_FRAC",
                   step_mask.min_step_det_frac);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_ALIGNMENT_FRAC",
                   step_mask.min_alignment_frac);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.CLUSTER_TOL_SEC",
                   step_mask.cluster_tol_sec);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.HALF_WIDTH_SEC",
                   step_mask.mask_half_width_sec);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MAX_FLAGGED_FRAC",
                   step_mask.max_flagged_fraction);
}

template <class ImpulsiveCapture>
void add_rtc_impulsive_capture_config_vars(
    netCDF::NcFile &fo, const ImpulsiveCapture &impulsive_capture) {
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.ENABLED",
                   impulsive_capture.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MIN_GOOD_FRAC",
                   impulsive_capture.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MIN_EVENT_Z",
                   impulsive_capture.min_event_z);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.NEAR_EVENT_Z",
                   impulsive_capture.near_event_z);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MAX_EVENTS",
                   impulsive_capture.max_events_per_network);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.PRE_WINDOW_SEC",
                   impulsive_capture.snippet_pre_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.POST_WINDOW_SEC",
                   impulsive_capture.snippet_post_window_sec);
}

template <class ImpulsiveCoincidence>
void add_rtc_impulsive_coincidence_config_vars(
    netCDF::NcFile &fo, const ImpulsiveCoincidence &impulsive_coincidence) {
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.ENABLED",
                   impulsive_coincidence.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_GOOD_FRAC",
                   impulsive_coincidence.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.EVENT_SCORE_THRESH",
                   impulsive_coincidence.event_score_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_USED",
                   impulsive_coincidence.min_det_used);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_FRAC",
                   impulsive_coincidence.min_impulsive_det_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_ALIGNMENT_FRAC",
                   impulsive_coincidence.min_alignment_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_NETWORKS_ALIGNED",
                   impulsive_coincidence.min_networks_aligned);
    add_netcdf_var(
        fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_OVERRIDE_THRESH",
        impulsive_coincidence.high_score_override_thresh);
    add_netcdf_var(
        fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_MIN_NETWORKS",
        impulsive_coincidence.high_score_min_networks_aligned);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.CLUSTER_TOL_SEC",
                   impulsive_coincidence.cluster_tol_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.PRE_WINDOW_SEC",
                   impulsive_coincidence.mask_pre_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.POST_WINDOW_SEC",
                   impulsive_coincidence.mask_post_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MAX_FLAGGED_FRAC",
                   impulsive_coincidence.max_flagged_fraction);
}

template <class RtcProc>
void add_rtc_event_mask_config_vars(netCDF::NcFile &fo,
                                    const RtcProc &rtcproc) {
    add_rtc_step_mask_config_vars(fo, rtcproc.network_step_mask);
    add_rtc_impulsive_capture_config_vars(fo, rtcproc.impulsive_capture);
    add_rtc_impulsive_coincidence_config_vars(
        fo, rtcproc.impulsive_coincidence);
}


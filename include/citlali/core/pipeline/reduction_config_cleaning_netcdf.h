#pragma once

// Included by reduction_config_netcdf.h inside namespace citlali::pipeline.

template <class WeightCorrPenalty>
void add_weight_corr_penalty_config_vars(
    netCDF::NcFile &fo, const WeightCorrPenalty &penalty) {
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.ENABLED",
                   penalty.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.MIN_GOOD_FRAC",
                   penalty.min_good_frac);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.MIN_OVERLAP",
                   penalty.min_overlap);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.MAX_SAMPLES",
                   penalty.max_samples);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.MAX_PAIRS",
                   penalty.max_pairs);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.FLOOR",
                   penalty.floor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.EXPONENT",
                   penalty.exponent);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.PAIR.ENABLED",
                   penalty.pair_corr.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.PAIR.REF",
                   penalty.pair_corr.ref);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.PAIR.SPAN",
                   penalty.pair_corr.span);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.PAIR.WEIGHT",
                   penalty.pair_corr.weight);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.CM_EL.ENABLED",
                   penalty.cm_el_corr.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.CM_EL.REF",
                   penalty.cm_el_corr.ref);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.CM_EL.SPAN",
                   penalty.cm_el_corr.span);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.CM_EL.WEIGHT",
                   penalty.cm_el_corr.weight);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.ENABLED",
                   penalty.cm_low_mid_ratio.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.REF",
                   penalty.cm_low_mid_ratio.ref);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.SPAN",
                   penalty.cm_low_mid_ratio.span);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.WEIGHT",
                   penalty.cm_low_mid_ratio.weight);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMIN_HZ",
                   penalty.cm_low_mid_ratio.low_band_Hz[0]);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMAX_HZ",
                   penalty.cm_low_mid_ratio.low_band_Hz[1]);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMIN_HZ",
                   penalty.cm_low_mid_ratio.mid_band_Hz[0]);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMAX_HZ",
                   penalty.cm_low_mid_ratio.mid_band_Hz[1]);
}

template <class BusyRowSuppression>
void add_busy_row_suppression_config_vars(
    netCDF::NcFile &fo, const BusyRowSuppression &suppression) {
    add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED",
                   suppression.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.REQUIRE_BUSY_VETO",
                   suppression.require_busy_veto);
    add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_CAND_CLUSTERS",
                   suppression.min_candidate_clusters);
    add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_MAX_RESID_Z",
                   suppression.min_max_unflagged_residual_z);
    add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.FACTOR",
                   suppression.factor);
}

void add_cleaner_mode_config_vars(netCDF::NcFile &fo,
                                  const citlali::config::ProcessedTimeChunkCleanConfig
                                      &clean) {
    add_netcdf_var(fo, "CONFIG.CLEANED", clean.enabled);
    add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.MODESEL",
                                std::string{citlali::config::to_string(
                                    clean.active)});
    add_netcdf_var(fo, "CONFIG.CLEANED.MP.ENABLED",
                   clean.marchenko_pastur.enabled);
    add_netcdf_var(fo, "CONFIG.CLEANED.MP.BANDLOW_HZ",
                   clean.marchenko_pastur.band_low_Hz);
    add_netcdf_var(fo, "CONFIG.CLEANED.MP.BANDHIGH_HZ",
                   clean.marchenko_pastur.band_high_Hz);
    add_netcdf_var(fo, "CONFIG.CLEANED.MP.MAXMODES",
                   clean.marchenko_pastur.max_modes);
}

template <class AdaptiveSelector>
void add_adaptive_cleaner_config_vars(
    netCDF::NcFile &fo, const AdaptiveSelector &selector) {
    const auto adaptive_offsets_joined =
        join_numeric_values(selector.candidate_offsets);
    const auto adaptive_grouping_joined =
        join_string_values(selector.grouping);

    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.ENABLED", selector.enabled);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MIN_GOOD_FRAC",
                   selector.min_good_frac);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MAX_DET", selector.max_det);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MAX_SAMPLES",
                   selector.max_samples);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MAX_PAIRS",
                   selector.max_pairs);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.CLIP_Z", selector.clip_z);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.LOW_WEIGHT",
                   selector.low_weight);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.TAIL_WEIGHT",
                   selector.tail_weight);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.TOPMODE_WEIGHT",
                   selector.topmode_weight);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.REG_WEIGHT",
                   selector.reg_weight);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.LOWMIN_HZ",
                   selector.low_band_Hz[0]);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.LOWMAX_HZ",
                   selector.low_band_Hz[1]);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MIDMIN_HZ",
                   selector.mid_band_Hz[0]);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.MIDMAX_HZ",
                   selector.mid_band_Hz[1]);
    add_netcdf_var<std::string>(
        fo, "CONFIG.CLEANED.ADAPT.CANDIDATE_OFFSETS",
        adaptive_offsets_joined);
    add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.ADAPT.GROUPING",
                                adaptive_grouping_joined);
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.LOG_CANDIDATES",
                   selector.log_candidates);
}

template <class PtcProc, class Calib, class ArrayNameMap>
void add_cleaned_eigen_count_config_vars(netCDF::NcFile &fo,
                                         const PtcProc &ptcproc,
                                         bool cleaning_enabled,
                                         const Calib &calib,
                                         ArrayNameMap &array_name_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        const auto array = calib.arrays(i);
        const auto key = "CONFIG.CLEANED.NEIG_" + array_name_map[array];
        if (cleaning_enabled) {
            add_netcdf_var(fo, key,
                           ptcproc.cleaner.n_eig_to_cut.at(array).sum());
        }
        else {
            add_netcdf_var(fo, key, 0);
        }
    }
}

template <class SecondPassLocal>
void add_ptc_second_pass_config_vars(netCDF::NcFile &fo,
                                     const SecondPassLocal &second_pass) {
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.ENABLED",
                   second_pass.enabled);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_SPIKE_SIGMA",
                   second_pass.min_spike_sigma);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_GOOD_FRAC",
                   second_pass.min_good_frac);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.BASELINE_WINDOW_SEC",
                   second_pass.baseline_window_sec);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.SIGMA_SCALE",
                   second_pass.sigma_scale);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.DELTA_SIGMA_SCALE",
                   second_pass.delta_sigma_scale);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.RAW_CAND_REL_SIGMA_SCALE",
                   second_pass.raw_candidate_rel_sigma_scale);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.RAW_WINDOW_SEC",
                   second_pass.raw_window_sec);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.RAW_HALF_PEAK_FRAC",
                   second_pass.raw_half_peak_frac);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.RAW_MAX_WIDTH_SEC",
                   second_pass.raw_max_width_sec);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.DELTA_WINDOW_SEC",
                   second_pass.delta_window_sec);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.DELTA_HALF_PEAK_FRAC",
                   second_pass.delta_half_peak_frac);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.DELTA_MAX_WIDTH_SEC",
                   second_pass.delta_max_width_sec);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MAX_STEP_SHIFT_Z",
                   second_pass.max_step_shift_z);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.HIGH_SCORE_EVENT_OVERRIDE",
                   second_pass.high_score_event_override);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MERGE_WITHIN_DET_SEC",
                   second_pass.merge_within_detector_sec);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.CLUSTER_EVENTS_SEC",
                   second_pass.cluster_events_sec);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_CLUSTER_DETECTORS",
                   second_pass.min_cluster_detectors);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.HIGH_SCORE_CLUSTER_OVERRIDE",
                   second_pass.high_score_cluster_override);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MAX_AUTO_FLAG_CLUSTERS",
                   second_pass.max_auto_flag_clusters_per_network);
}

template <class PtcProc, class Calib, class ArrayNameMap>
void add_ptc_cleaning_header_config_vars(netCDF::NcFile &fo,
                                         const PtcProc &ptcproc,
                                         const citlali::config::ProcessedTimeChunkConfig
                                             &config,
                                         const Calib &calib,
                                         ArrayNameMap &array_name_map) {
    const auto &clean = config.clean;
    const auto &weighting = config.weighting;
    add_ptc_weight_cutoff_config_vars(fo, ptcproc);
    add_weight_corr_penalty_config_vars(fo, weighting.corr_penalty);
    add_busy_row_suppression_config_vars(
        fo, weighting.busy_row_suppression);
    add_cleaner_mode_config_vars(fo, clean);
    add_adaptive_cleaner_config_vars(fo, clean.adaptive_selector);
    add_ptc_second_pass_config_vars(
        fo, config.flagging.second_pass_local);
    add_cleaned_eigen_count_config_vars(
        fo, ptcproc, clean.enabled, calib, array_name_map);
}

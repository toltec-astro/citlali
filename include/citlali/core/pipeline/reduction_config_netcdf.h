#pragma once

#include <string>

#include <netcdf>

#include <citlali/core/pipeline/string_join.h>
#include <citlali/core/utils/netcdf_io.h>

namespace citlali::pipeline {

template <class WeightValidation>
void add_weight_validation_config_vars(
    netCDF::NcFile &fo, const WeightValidation &weight_validation) {
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.ENABLED",
                   weight_validation.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.ACCUMULATION_ITERS",
                   weight_validation.accumulation_iters);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.APPLY_START_ITER",
                   weight_validation.apply_start_iter);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.MIN_FACTOR",
                   weight_validation.min_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_ENABLED",
                   weight_validation.upward_enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_MAX_FACTOR",
                   weight_validation.upward_max_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_POWER",
                   weight_validation.upward_power);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_BASE_FACTOR",
                   weight_validation.upward_min_base_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_REQUIRE_ATM",
                   weight_validation.upward_require_atmospheric);
    add_netcdf_var(fo, "CONFIG.WEIGHT.VALIDATION.UPWARD_MIN_ATM_FACTOR",
                   weight_validation.upward_min_atmospheric_factor);
    add_netcdf_var<std::string>(
        fo, "CONFIG.WEIGHT.VALIDATION.ATM_GROUPING",
        weight_validation.atmospheric_grouping);
}

template <class PtcProc>
void add_weight_selection_config_vars(netCDF::NcFile &fo,
                                      const PtcProc &ptcproc) {
    add_netcdf_var<std::string>(fo, "CONFIG.WEIGHT.TYPE",
                                ptcproc.weighting_type);
    add_netcdf_var(fo, "CONFIG.WEIGHT.SOURCE_MASK_RADIUS_ARCSEC",
                   ptcproc.source_mask_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.WEIGHT.HYBRID_MIN_FACTOR",
                   ptcproc.hybrid_correction_min_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.HYBRID_MAX_FACTOR",
                   ptcproc.hybrid_correction_max_factor);
    add_weight_validation_config_vars(fo, ptcproc.weight_validation);
}

template <class PtcProc>
void add_ptc_weight_cutoff_config_vars(netCDF::NcFile &fo,
                                       const PtcProc &ptcproc,
                                       bool include_inv_var_window = false) {
    add_netcdf_var(fo, "CONFIG.INV_VAR.PTC.WTLOW",
                   ptcproc.lower_inv_var_factor);
    add_netcdf_var(fo, "CONFIG.INV_VAR.PTC.WTHIGH",
                   ptcproc.upper_inv_var_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTLOW",
                   ptcproc.lower_weight_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.PTC.WTHIGH",
                   ptcproc.upper_weight_factor);
    add_netcdf_var(fo, "CONFIG.WEIGHT.MEDWTFACTOR",
                   ptcproc.med_weight_factor);
    if (include_inv_var_window) {
        add_netcdf_var(fo, "CONFIG.INV_VAR.WINDOW_SEC",
                       ptcproc.remove_bad_dets_window_sec);
    }
}

inline void add_tod_initial_runtime_config_vars(netCDF::NcFile &fo,
                                                bool verbose_mode,
                                                bool run_polarization,
                                                bool run_despike) {
    add_netcdf_var(fo, "CONFIG.VERBOSE", verbose_mode);
    add_netcdf_var(fo, "CONFIG.POLARIZED", run_polarization);
    add_netcdf_var(fo, "CONFIG.DESPIKED", run_despike);
}

template <class RtcProc>
void add_tod_filter_runtime_config_vars(netCDF::NcFile &fo,
                                        const RtcProc &rtcproc,
                                        bool run_any_tod_filter) {
    add_netcdf_var(fo, "CONFIG.TODFILTERED", run_any_tod_filter);
    add_netcdf_var(fo, "CONFIG.TODNOTCH", rtcproc.run_tod_notch);
    add_netcdf_var(fo, "CONFIG.TODIIRHP", rtcproc.run_tod_iir_highpass);
    add_netcdf_var(fo, "CONFIG.TODIIRHP.FREQ_HZ",
                   rtcproc.filter.iir_highpass_freq_Hz);
    add_netcdf_var(fo, "CONFIG.TODIIRHP.ORDER",
                   rtcproc.filter.iir_highpass_order);
    add_netcdf_var(fo, "CONFIG.TODIIRHP.ZEROPHASE",
                   rtcproc.filter.iir_highpass_zero_phase);
}

template <class RtcProc>
void add_tod_processing_config_vars(netCDF::NcFile &fo,
                                    const RtcProc &rtcproc) {
    add_netcdf_var(fo, "CONFIG.DOWNSAMPLED", rtcproc.run_downsample);
    add_netcdf_var(fo, "CONFIG.CALIBRATED", rtcproc.run_calibrate);
    add_netcdf_var(fo, "CONFIG.EXTINCTION", rtcproc.run_extinction);
    add_netcdf_var<std::string>(fo, "CONFIG.EXTINCTION.EXTMODEL",
                                rtcproc.calibration.extinction_model);
}

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
                   penalty.cm_low_mid_ratio.low_min_Hz);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMAX_HZ",
                   penalty.cm_low_mid_ratio.low_max_Hz);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMIN_HZ",
                   penalty.cm_low_mid_ratio.mid_min_Hz);
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMAX_HZ",
                   penalty.cm_low_mid_ratio.mid_max_Hz);
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

template <class PtcProc>
void add_cleaner_mode_config_vars(netCDF::NcFile &fo,
                                  const PtcProc &ptcproc) {
    add_netcdf_var(fo, "CONFIG.CLEANED", ptcproc.run_clean);
    add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.MODESEL",
                                ptcproc.cleaner.active_cleaner_label());
    add_netcdf_var(fo, "CONFIG.CLEANED.MP.ENABLED",
                   ptcproc.cleaner.marchenko_pastur.enabled);
    add_netcdf_var(fo, "CONFIG.CLEANED.MP.BANDLOW_HZ",
                   ptcproc.cleaner.marchenko_pastur.band_low_Hz);
    add_netcdf_var(fo, "CONFIG.CLEANED.MP.BANDHIGH_HZ",
                   ptcproc.cleaner.marchenko_pastur.band_high_Hz);
    add_netcdf_var(fo, "CONFIG.CLEANED.MP.MAXMODES",
                   ptcproc.cleaner.marchenko_pastur.max_modes);
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
                                         const Calib &calib,
                                         ArrayNameMap &array_name_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        const auto array = calib.arrays(i);
        const auto key = "CONFIG.CLEANED.NEIG_" + array_name_map[array];
        if (ptcproc.run_clean) {
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
                                         const Calib &calib,
                                         ArrayNameMap &array_name_map) {
    add_ptc_weight_cutoff_config_vars(fo, ptcproc);
    add_weight_corr_penalty_config_vars(fo, ptcproc.weight_corr_penalty);
    add_busy_row_suppression_config_vars(fo, ptcproc.busy_row_suppression);
    add_cleaner_mode_config_vars(fo, ptcproc);
    add_adaptive_cleaner_config_vars(fo, ptcproc.cleaner.adaptive_selector);
    add_ptc_second_pass_config_vars(fo, ptcproc.second_pass_local);
    add_cleaned_eigen_count_config_vars(fo, ptcproc, calib, array_name_map);
}

template <class PtcProc>
void add_fruit_loops_config_vars(netCDF::NcFile &fo,
                                 const PtcProc &ptcproc) {
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS", ptcproc.run_fruit_loops);
    add_netcdf_var<std::string>(fo, "CONFIG.FRUITLOOPS.PATH",
                                ptcproc.fruit_loops_path);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.S2N",
                   ptcproc.fruit_loops_sig2noise);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.PEAKFRAC",
                   ptcproc.fruit_loops_peak_fraction_limit);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSNR",
                   ptcproc.fruit_loops_local_snr_floor);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_INNER",
                   ptcproc.fruit_loops_local_sigma_inner_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_OUTER",
                   ptcproc.fruit_loops_local_sigma_outer_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_EDGE",
                   ptcproc.fruit_loops_local_sigma_edge_guard_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.LOCALSIG_MINPIX",
                   ptcproc.fruit_loops_local_sigma_min_pixels);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_RAD",
                   ptcproc.fruit_loops_adaptive_support_radius_arcsec);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.ADAPT_SUPPORT_FWHM",
                   ptcproc.fruit_loops_adaptive_support_radius_fwhm);
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.ENABLED",
                   ptcproc.fruit_loops_weight_feedback_enabled);
    add_netcdf_var<std::string>(
        fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.REFERENCE",
        ptcproc.fruit_loops_weight_feedback_reference);
    add_netcdf_var(
        fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.LOW_RELATIVE_WEIGHT",
        ptcproc.fruit_loops_weight_feedback_low_relative_weight);
    add_netcdf_var(
        fo, "CONFIG.FRUITLOOPS.WEIGHT_FEEDBACK.HIGH_RELATIVE_WEIGHT",
        ptcproc.fruit_loops_weight_feedback_high_relative_weight);
}

template <class PtcProc, class Calib, class ArrayNameMap>
void add_fruit_loop_flux_config_vars(netCDF::NcFile &fo,
                                     const PtcProc &ptcproc,
                                     const Calib &calib,
                                     ArrayNameMap &array_name_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        double flux_limit = 0.0;
        if (ptcproc.run_fruit_loops) {
            if (ptcproc.fruit_loops_flux.size() == calib.arrays.size()) {
                flux_limit = ptcproc.fruit_loops_flux(i);
            }
            else if (calib.arrays(i) < ptcproc.fruit_loops_flux.size()) {
                flux_limit = ptcproc.fruit_loops_flux(calib.arrays(i));
            }
        }
        add_netcdf_var(
            fo, "CONFIG.FRUITLOOPS.FLUX_" + array_name_map[calib.arrays(i)],
            flux_limit);
    }
}

template <class PtcProc>
void add_fruit_loop_iteration_config_vars(netCDF::NcFile &fo,
                                          const PtcProc &ptcproc) {
    add_netcdf_var(fo, "CONFIG.FRUITLOOPS.MAXITER",
                   ptcproc.fruit_loops_iters);
}

template <class PtcProc, class Calib, class ArrayNameMap>
void add_fruit_loop_header_config_vars(netCDF::NcFile &fo,
                                       const PtcProc &ptcproc,
                                       const Calib &calib,
                                       ArrayNameMap &array_name_map) {
    add_fruit_loops_config_vars(fo, ptcproc);
    add_fruit_loop_flux_config_vars(fo, ptcproc, calib, array_name_map);
    add_fruit_loop_iteration_config_vars(fo, ptcproc);
}

template <class PtcProc>
void add_ptcdiag_compact_config_vars(netCDF::NcFile &fo,
                                     const PtcProc &ptcproc) {
    add_netcdf_var(fo, "CONFIG.WEIGHT.CORR_PENALTY.ENABLED",
                   ptcproc.weight_corr_penalty.enabled);
    add_netcdf_var(fo, "CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED",
                   ptcproc.busy_row_suppression.enabled);
    add_netcdf_var(fo, "CONFIG.CLEANED", ptcproc.run_clean);
    add_netcdf_var<std::string>(fo, "CONFIG.CLEANED.MODESEL",
                                ptcproc.cleaner.active_cleaner_label());
    add_netcdf_var(fo, "CONFIG.CLEANED.ADAPT.ENABLED",
                   ptcproc.cleaner.adaptive_selector.enabled);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.ENABLED",
                   ptcproc.second_pass_local.enabled);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_SPIKE_SIGMA",
                   ptcproc.second_pass_local.min_spike_sigma);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.HIGH_SCORE_EVENT_OVERRIDE",
                   ptcproc.second_pass_local.high_score_event_override);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MIN_CLUSTER_DETECTORS",
                   ptcproc.second_pass_local.min_cluster_detectors);
    add_netcdf_var(fo, "CONFIG.PTC.SECOND_PASS.MAX_AUTO_FLAG_CLUSTERS",
                   ptcproc.second_pass_local.max_auto_flag_clusters_per_network);
    add_fruit_loops_config_vars(fo, ptcproc);
}

template <class ReductionLearning>
void add_reduction_learning_config_vars(
    netCDF::NcFile &fo, const ReductionLearning &reduction_learning,
    bool include_max_records_per_type = true) {
    const auto &options = reduction_learning.options;
    add_netcdf_var(fo, "CONFIG.LEARNING.ENABLED", options.enabled);
    add_netcdf_var(fo, "CONFIG.LEARNING.DIAGNOSTICS_ENABLED",
                   options.diagnostics_enabled);
    add_netcdf_var(fo, "CONFIG.LEARNING.LEARN_ITERS",
                   options.learn_iters);
    add_netcdf_var(fo, "CONFIG.LEARNING.APPLY_START_ITER",
                   options.apply_start_iter);
    if (include_max_records_per_type) {
        add_netcdf_var(fo, "CONFIG.LEARNING.MAX_RECORDS_PER_TYPE",
                       options.max_records_per_type);
    }
    add_netcdf_var(
        fo, "CONFIG.LEARNING.MAP_PIXEL_OUTLIER_DETECTOR_EXCLUSION_ENABLED",
        options.map_pixel_outlier_detector_exclusion_enabled);
    add_netcdf_var(
        fo, "CONFIG.LEARNING.MAP_PIXEL_OUTLIER_DETECTOR_EXCLUSION_MIN_PIXELS",
        options.map_pixel_outlier_detector_exclusion_min_pixels);
    add_netcdf_var(fo, "CONFIG.LEARNING.BUSY_DETECTOR_EXCLUSION_ENABLED",
                   options.busy_detector_exclusion_enabled);
    add_netcdf_var(fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_ENABLED",
                   options.scan_network_pathology_enabled);
    add_netcdf_var(fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_APPLY_PRE_RTC",
                   options.scan_network_pathology_apply_pre_rtc);
    add_netcdf_var(fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_APPLY_PRE_PTC",
                   options.scan_network_pathology_apply_pre_ptc);
    add_netcdf_var(
        fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_APPLY_PRE_MAPMAKING",
        options.scan_network_pathology_apply_pre_mapmaking);
    add_netcdf_var(fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MIN_CLUSTERS",
                   options.scan_network_pathology_min_candidate_clusters);
    add_netcdf_var(fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MIN_EVENTS",
                   options.scan_network_pathology_min_candidate_events);
    add_netcdf_var(fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MIN_RESID_Z",
                   options.scan_network_pathology_min_max_residual_z);
    add_netcdf_var(fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_SEVERE_EVENTS",
                   options.scan_network_pathology_severe_candidate_events);
    add_netcdf_var(fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_SEVERE_RESID_Z",
                   options.scan_network_pathology_severe_max_residual_z);
    add_netcdf_var(
        fo, "CONFIG.LEARNING.SCAN_NETWORK_PATHOLOGY_MAX_NEW_FLAGGED_FRAC",
        options.scan_network_pathology_max_new_flagged_fraction);
    add_netcdf_var<std::string>(fo, "CONFIG.LEARNING.PHASE",
                                reduction_learning.current_phase_name());
}

template <class EdgeGuard, class OuterContext, class OutputOuterContext>
void add_tod_filter_edge_guard_config_vars(
    netCDF::NcFile &fo, const EdgeGuard &edge_guard,
    OuterContext outer_context_samples,
    OutputOuterContext output_outer_context_samples) {
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.ENABLED",
                   edge_guard.enabled);
    add_netcdf_var<std::string>(fo, "CONFIG.TODFILTER.EDGE_GUARD.MODE",
                                edge_guard.mode);
    add_netcdf_var<std::string>(fo, "CONFIG.TODFILTER.EDGE_GUARD.COMBINE",
                                edge_guard.combine);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.CONTEXT_SAMPLES",
                   edge_guard.context_samples);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.GUARD_SAMPLES",
                   edge_guard.guard_samples);
    add_netcdf_var(fo, "CONFIG.TOD.OUTER_CONTEXT_SAMPLES",
                   outer_context_samples);
    add_netcdf_var(fo, "CONFIG.TOD.OUTPUT_OUTER_CONTEXT_SAMPLES",
                   output_outer_context_samples);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.MIN_SAMPLES",
                   edge_guard.min_samples);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.EXTRA_SAMPLES",
                   edge_guard.extra_samples);
    add_netcdf_var(fo, "CONFIG.TODFILTER.EDGE_GUARD.MAX_SAMPLES",
                   edge_guard.max_samples);
    add_netcdf_var(
        fo, "CONFIG.TODFILTER.EDGE_GUARD.IIR_SETTLE_ATTENUATION",
        edge_guard.iir_settle_attenuation);
}

}  // namespace citlali::pipeline

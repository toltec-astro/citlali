#pragma once

#include <string>

#include <netcdf>

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

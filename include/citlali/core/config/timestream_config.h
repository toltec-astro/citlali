#pragma once

#include <citlali/core/config/timestream_enums.h>

#include <array>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace citlali::config {

struct TodStreamOutputConfig {
    bool enabled = false;
    TodStreamOutputMode mode = TodStreamOutputMode::full;
    int outer_context_samples = 0;
    bool chunk_select_enabled = false;
    std::vector<int> chunks_1based;
    TodOutputSelectionMode selection_mode = TodOutputSelectionMode::indices;
    int selection_n_uniform = 10;
    int selection_n_source_dense = 10;
};

struct TimestreamOutputConfig {
    bool raw_time_chunk_enabled = false;
    bool processed_time_chunk_enabled = false;
    TodOutputType type = TodOutputType::none;
    std::string subdir_name;
    bool write_eigenvalues = false;
    TodStreamOutputConfig raw_time_chunk;
    TodStreamOutputConfig processed_time_chunk;
};

struct TimestreamChunkingConfig {
    std::string mode;
    double value = 0.0;
    bool force = false;
};

struct TimestreamSourceProtectionConfig {
    bool enabled = true;
    bool active = false;
    double radius_arcsec = 20.0;
};

struct RawTimeChunkDespikeCompactRawGateConfig {
    bool enabled = true;
    double candidate_rel_sigma_scale = 1.0;
    double window_sec = 0.18;
    double half_peak_frac = 0.5;
    double max_width_sec = 0.18;
    double max_step_shift_z = 3.0;
};

struct RawTimeChunkDespikeCompactDeltaGateConfig {
    bool enabled = true;
    double window_sec = 0.12;
    double half_peak_frac = 0.5;
    double max_width_sec = 0.10;
    double max_step_shift_z = 3.0;
};

struct RawTimeChunkDespikeLocalResidualConfig {
    bool enabled = false;
    double window_sec = 0.25;
    double sigma_scale = 0.75;
    double delta_sigma_scale = 0.75;
    bool expand_with_filter = false;
    double event_padding_sec = 0.08;
    double high_score_event_override = 20.0;
    double max_added_flagged_fraction = 0.10;
    RawTimeChunkDespikeCompactRawGateConfig compact_raw_gate;
    RawTimeChunkDespikeCompactDeltaGateConfig compact_delta_gate;
};

struct RawTimeChunkDespikeConfig {
    bool enabled = false;
    double min_spike_sigma = 8.0;
    double time_constant_sec = 0.015;
    double window_size = 32.0;
    bool legacy_enabled = true;
    TimestreamSourceProtectionConfig source_protection;
    RawTimeChunkDespikeLocalResidualConfig local_residual;
};

struct RawTimeChunkDownsampleConfig {
    bool enabled = false;
    int factor = 1;
    double downsampled_freq_Hz = 0.0;
};

struct RawTimeChunkFilterNotchConfig {
    bool enabled = false;
    bool zero_phase = true;
    std::vector<double> freqs_Hz;
    std::vector<double> delta_f_Hz;
};

struct RawTimeChunkFilterEdgeGuardConfig {
    bool enabled = false;
    RawTimeChunkFilterEdgeGuardMode mode =
        RawTimeChunkFilterEdgeGuardMode::flag;
    RawTimeChunkFilterEdgeGuardCombine combine =
        RawTimeChunkFilterEdgeGuardCombine::sum;
    int min_samples = 0;
    int extra_samples = 0;
    int max_samples = 128;
    double iir_settle_attenuation = 0.01;
    bool apply_fir = true;
    bool apply_notch = true;
    bool apply_dynamic_notch = true;
    bool apply_iir_highpass = true;
    bool apply_downsample = true;
};

struct RawTimeChunkFilterConfig {
    bool enabled = false;
    double a_gibbs = 50.0;
    double freq_high_Hz = 16.0;
    double freq_low_Hz = 0.0;
    int n_terms = 32;
    RawTimeChunkFilterNotchConfig notch;
    RawTimeChunkFilterEdgeGuardConfig edge_guard;
};

struct RawTimeChunkIirFilterConfig {
    bool enabled = false;
    double freq_Hz = 0.1;
    int order = 1;
    bool zero_phase = false;
};

struct RawTimeChunkNetworkStepMaskConfig {
    bool enabled = false;
    double step_window_sec = 0.5;
    double step_score_thresh = 2.5;
    double min_good_frac = 0.8;
    int min_det_used = 32;
    double min_step_det_frac = 0.05;
    double min_alignment_frac = 0.5;
    double cluster_tol_sec = 0.25;
    double mask_half_width_sec = 0.5;
    double max_flagged_fraction = 0.30;
};

struct RawTimeChunkImpulsiveCaptureConfig {
    bool enabled = false;
    double min_good_frac = 0.8;
    double min_event_z = 6.0;
    double near_event_z = 4.0;
    int max_events_per_network = 3;
    double snippet_pre_window_sec = 0.25;
    double snippet_post_window_sec = 0.25;
};

struct RawTimeChunkImpulsiveCoincidenceConfig {
    bool enabled = false;
    double min_good_frac = 0.8;
    double event_score_thresh = 6.0;
    int min_det_used = 32;
    double min_impulsive_det_frac = 0.05;
    double min_alignment_frac = 0.5;
    int min_networks_aligned = 3;
    double high_score_override_thresh = 0.0;
    int high_score_min_networks_aligned = 0;
    double cluster_tol_sec = 0.03;
    double mask_pre_window_sec = 0.03;
    double mask_post_window_sec = 0.03;
    double max_flagged_fraction = 0.10;
};

struct RawTimeChunkCoherentIqModeObserverConfig {
    bool enabled = false;
    std::vector<std::string> template_paths;
    double candidate_step_score_min = 2.5;
    double candidate_impulsive_score_min = 4.0;
    double candidate_cluster_tolerance_sec = 0.25;
    double pre_window_sec = 0.20;
    double guard_window_sec = 0.05;
    double post_window_sec = 0.20;
    double cross_network_tolerance_sec = 0.35;
    int max_candidates_per_scan_per_network = 8;
    int max_network_event_scores = 20000;
    int progress_interval_scores = 250;
};

struct RawTimeChunkFlaggingConfig {
    double delta_f_min_Hz = 60.e3;
    double lower_tod_inv_var_factor = 0.0;
    double upper_tod_inv_var_factor = 0.0;
    RawTimeChunkNetworkStepMaskConfig network_step_mask;
    RawTimeChunkImpulsiveCaptureConfig impulsive_capture;
    RawTimeChunkImpulsiveCoincidenceConfig impulsive_coincidence;
};

struct RawTimeChunkKernelConfig {
    bool enabled = false;
    std::string filepath;
    std::string type;
    double fwhm_arcsec = 0.0;
    std::vector<std::string> image_ext_names;
};

struct RawTimeChunkAltAzDestripeConfig {
    bool enabled = false;
    std::string grouping = "nw";
    bool fit_time_trend = true;
    bool fit_derivs = true;
    int min_samples = 64;
};

struct RawTimeChunkLineAuditConfig {
    bool enabled = false;
    double line_min_hz = 1.0;
    double line_max_hz = 60.0;
    double segment_sec = 4.0;
    double min_segment_sec = 2.0;
    double overlap_frac = 0.5;
    int continuum_radius_bins = 8;
    double prominence_thresh = 8.0;
    double cm_prominence_thresh = 6.0;
    double min_good_frac = 0.8;
    int min_windows = 2;
    int max_peaks_per_detector = 3;
    int max_det = 128;
    int min_det_for_network = 16;
    double cluster_tol_hz = 0.15;
    double notch_min_detector_frac = 0.10;
    int notch_min_detectors = 8;
    double notch_min_cm_prominence = 10.0;
    double detector_min_prominence = 12.0;
    double detector_min_line_power_frac = 0.10;
    double bad_detector_max_cluster_frac = 0.10;
    bool pre_filter_enabled = true;
    bool post_filter_enabled = false;
    bool post_filter_apply_shared_notches = false;
    bool post_filter_apply_detector_notches = false;
    int post_filter_apply_iterations = 1;
    double post_filter_line_min_hz =
        std::numeric_limits<double>::quiet_NaN();
    double post_filter_line_max_hz =
        std::numeric_limits<double>::quiet_NaN();
    bool ptc_model_protected_enabled = false;
    bool ptc_require_model_subtracted = true;
    bool ptc_apply_fixed_notches = false;
    bool ptc_apply_shared_notches = false;
    bool ptc_apply_detector_notches = false;
    int ptc_apply_iterations = 1;
    double ptc_line_min_hz = std::numeric_limits<double>::quiet_NaN();
    double ptc_line_max_hz = std::numeric_limits<double>::quiet_NaN();
    bool fixed_notch_enabled = false;
    std::vector<double> fixed_notch_freqs_hz;
    std::vector<double> fixed_notch_widths_hz{0.25};
    double fixed_notch_exclusion_half_width_hz = 0.25;
    bool apply_shared_notches = false;
    int apply_min_support_networks = 2;
    double apply_min_detector_frac = 0.90;
    double apply_min_common_mode_prominence = 150.0;
    double apply_width_scale = 1.5;
    double apply_min_width_hz = 0.25;
    double apply_max_width_hz = 1.50;
    int apply_max_notches = 3;
    double apply_cluster_tol_hz = 0.25;
    double detector_notch_min_prominence = 8.0;
    double detector_notch_min_line_power_frac = 0.0;
    int detector_notch_max_notches = 3;
    double detector_notch_width_scale = 1.0;
    double detector_notch_min_width_hz = 0.25;
    double detector_notch_max_width_hz = 1.50;
    int detector_notch_context_samples = 0;
};

struct RawTimeChunkConfig {
    RawTimeChunkDespikeConfig despike;
    RawTimeChunkDownsampleConfig downsample;
    RawTimeChunkFilterConfig filter;
    RawTimeChunkIirFilterConfig iir_filter;
    RawTimeChunkFlaggingConfig flagging;
    RawTimeChunkCoherentIqModeObserverConfig coherent_iq_mode_observer;
    RawTimeChunkKernelConfig kernel;
    RawTimeChunkAltAzDestripeConfig altaz_destripe;
    RawTimeChunkLineAuditConfig line_audit;
    bool flux_calibration_enabled = false;
    bool extinction_correction_enabled = false;
    std::string extinction_model;
};

inline bool raw_time_chunk_filtering_active(
    const RawTimeChunkConfig &config) {
    return config.filter.enabled || config.iir_filter.enabled;
}

struct ProcessedTimeChunkSecondPassLocalConfig {
    bool enabled = false;
    double min_spike_sigma = 8.0;
    double min_good_frac = 0.5;
    double baseline_window_sec = 0.25;
    double sigma_scale = 0.75;
    double delta_sigma_scale = 0.75;
    double raw_candidate_rel_sigma_scale = 1.0;
    double raw_window_sec = 0.18;
    double raw_half_peak_frac = 0.5;
    double raw_max_width_sec = 0.18;
    double delta_window_sec = 0.12;
    double delta_half_peak_frac = 0.5;
    double delta_max_width_sec = 0.10;
    double max_step_shift_z = 3.0;
    double high_score_event_override = 20.0;
    double merge_within_detector_sec = 0.08;
    double cluster_events_sec = 0.08;
    int min_cluster_detectors = 3;
    double high_score_cluster_override = 9.0;
    int max_auto_flag_clusters_per_network = 3;
    bool selective_busy_network_acceptance_enabled = true;
    TimestreamSourceProtectionConfig source_protection;
};

struct ProcessedTimeChunkStandardPcaConfig {
    bool enabled = true;
    double stddev_limit = 0.0;
    int n_calc = 64;
    std::map<std::string, std::vector<int>> n_eig_to_cut;
};

struct ProcessedTimeChunkCorrGroupingConfig {
    bool enabled = false;
    ProcessedTimeChunkCorrGroupingMetric metric =
        ProcessedTimeChunkCorrGroupingMetric::abs;
    double corr_min = 0.6;
    int min_overlap = 300;
    double min_good_frac = 0.8;
    int min_group_size = 10;
    int max_samples = 20000;
    bool clean_residual = true;
};

struct ProcessedTimeChunkNullModelConfig {
    bool enabled = false;
    int n_surrogates = 16;
    double quantile = 0.99;
    double min_good_frac = 0.8;
    int max_modes = 64;
    int max_samples = 20000;
    int seed = 12345;
    std::vector<std::string> grouping;
};

struct ProcessedTimeChunkMarchenkoPasturConfig {
    bool enabled = false;
    double min_good_frac = 0.8;
    int max_modes = 64;
    int max_samples = 20000;
    double band_low_Hz = 0.0;
    double band_high_Hz = 0.0;
    double clip_z = 12.0;
    double bulk_keep_frac = 0.8;
    int q_grid_size = 64;
    std::vector<std::string> grouping;
};

struct ProcessedTimeChunkAdaptiveSelectorConfig {
    bool enabled = false;
    double min_good_frac = 0.7;
    int max_det = 120;
    int max_samples = 1024;
    int max_pairs = 2000;
    int seed = 12345;
    double clip_z = 50.0;
    double low_weight = 1.0;
    double tail_weight = 0.0;
    double topmode_weight = 0.1;
    double reg_weight = 0.3;
    std::array<double, 2> low_band_Hz{0.05, 0.5};
    std::array<double, 2> mid_band_Hz{0.5, 2.0};
    std::vector<int> candidate_offsets{-2, 0, 2, 4};
    std::vector<std::string> grouping;
    bool log_candidates = false;
};

struct ProcessedTimeChunkCleanConfig {
    bool enabled = false;
    ProcessedTimeChunkCleanerMode active = ProcessedTimeChunkCleanerMode::none;
    std::vector<std::string> grouping;
    double mask_radius_arcsec = 0.0;
    double tau = 0.0;
    ProcessedTimeChunkStandardPcaConfig standard_pca;
    ProcessedTimeChunkCorrGroupingConfig corr_grouping;
    ProcessedTimeChunkNullModelConfig null_model;
    ProcessedTimeChunkMarchenkoPasturConfig marchenko_pastur;
    ProcessedTimeChunkAdaptiveSelectorConfig adaptive_selector;
};

struct ProcessedTimeChunkBusyRowSuppressionConfig {
    bool enabled = false;
    bool require_busy_veto = true;
    int min_candidate_clusters = 5;
    double min_max_unflagged_residual_z = 25.0;
    double factor = 0.0;
};

struct ProcessedTimeChunkWeightValidationConfig {
    bool enabled = false;
    int accumulation_iters = 1;
    int apply_start_iter = 1;
    int min_valid_scans = 1;
    double min_factor = 0.1;
    double unvalidated_factor = 1.0;
    bool require_fruitloops_model = true;
    bool transient_ratio_enabled = false;
    double ratio_power = 1.0;
    double transient_ratio_power = 1.0;
    bool upward_enabled = false;
    double upward_max_factor = 1.10;
    double upward_power = 1.0;
    double upward_min_base_factor = 0.95;
    bool upward_require_atmospheric = true;
    double upward_min_atmospheric_factor = 0.9;
    bool atmospheric_correlation_enabled = true;
    ProcessedTimeChunkWeightGrouping atmospheric_grouping =
        ProcessedTimeChunkWeightGrouping::array;
    int atmospheric_min_detectors = 8;
    double atmospheric_ref = 0.0;
    double atmospheric_span = 0.15;
    double atmospheric_power = 1.0;
    double min_good_frac = 0.5;
    int min_overlap = 200;
    int max_samples = 5000;
    bool high_weight_validation_enabled = true;
    bool high_weight_apply_caps = true;
    ProcessedTimeChunkWeightGrouping high_weight_grouping =
        ProcessedTimeChunkWeightGrouping::array;
    int high_weight_min_group_detectors = 20;
    double high_weight_log_robust_z = 6.0;
    double high_weight_max_median_factor = 8.0;
    double high_weight_cap_median_factor = 4.0;
    double high_weight_min_validated_factor = 0.95;
};

struct ProcessedTimeChunkWeightCorrPenaltyTermConfig {
    bool enabled = true;
    double ref = 0.05;
    double span = 0.15;
    double weight = 1.0;
};

struct ProcessedTimeChunkWeightCorrPenaltyBandConfig {
    bool enabled = false;
    double ref = 0.6;
    double span = 2.0;
    double weight = 0.5;
    std::array<double, 2> low_band_Hz{0.05, 0.5};
    std::array<double, 2> mid_band_Hz{0.5, 2.0};
};

struct ProcessedTimeChunkWeightCorrPenaltyConfig {
    bool enabled = false;
    double min_good_frac = 0.7;
    int min_overlap = 200;
    int max_samples = 20000;
    int max_pairs = 4000;
    int seed = 12345;
    double floor = 0.05;
    double exponent = 2.0;
    ProcessedTimeChunkWeightCorrPenaltyTermConfig pair_corr;
    ProcessedTimeChunkWeightCorrPenaltyTermConfig cm_el_corr{
        false, 0.05, 0.25, 0.5};
    ProcessedTimeChunkWeightCorrPenaltyBandConfig cm_low_mid_ratio;
};

struct ProcessedTimeChunkWeightingConfig {
    ProcessedTimeChunkWeightingType type =
        ProcessedTimeChunkWeightingType::full;
    double source_mask_radius_arcsec = 0.0;
    double hybrid_correction_min_factor = 0.5;
    double hybrid_correction_max_factor = 2.0;
    double median_map_weight_factor = 0.0;
    double lower_map_weight_factor = 0.0;
    double upper_map_weight_factor = 0.0;
    ProcessedTimeChunkWeightValidationConfig validation;
    ProcessedTimeChunkWeightCorrPenaltyConfig corr_penalty;
    ProcessedTimeChunkBusyRowSuppressionConfig busy_row_suppression;
};

struct ProcessedTimeChunkFlaggingConfig {
    double lower_tod_inv_var_factor = 0.0;
    double upper_tod_inv_var_factor = 0.0;
    ProcessedTimeChunkSecondPassLocalConfig second_pass_local;
};

struct ProcessedTimeChunkConfig {
    ProcessedTimeChunkCleanConfig clean;
    ProcessedTimeChunkWeightingConfig weighting;
    ProcessedTimeChunkFlaggingConfig flagging;
};

struct FruitLoopsWeightFeedbackConfig {
    bool enabled = false;
    FruitLoopsWeightFeedbackReference reference =
        FruitLoopsWeightFeedbackReference::p95;
    double low_relative_weight = 0.02;
    double high_relative_weight = 0.10;
};

struct FruitLoopsInjectedSourceTestConfig {
    bool enabled = false;
    int start_iteration = 1;
    std::vector<double> array_amplitude_mjy_beam;
};

struct TimestreamFruitLoopsConfig {
    bool enabled = false;
    bool diagnostics_enabled = false;
    bool save_all_iters = false;
    std::string path;
    // A completed reduction directory containing
    // citlali_restart_checkpoint.nc.  Unlike path, this requests an exact
    // continuation of both the fruit-loop map and effective learning state.
    std::string restart_path;
    std::string type;
    FruitLoopsMode mode = FruitLoopsMode::upper;
    double sig2noise_limit = 0.0;
    std::vector<double> array_flux_limit;
    double peak_fraction_limit = 0.0;
    double local_snr_floor = 0.0;
    double local_sigma_inner_radius_arcsec = 10.0;
    double local_sigma_outer_radius_arcsec = 35.0;
    double local_sigma_inner_fwhm = 1.5;
    double local_sigma_outer_fwhm = 4.0;
    double local_sigma_edge_guard_arcsec = 5.0;
    int local_sigma_min_pixels = 50;
    double adaptive_support_radius_arcsec = 12.0;
    double adaptive_support_radius_fwhm = 1.5;
    FruitLoopsSourceCenterMode source_center_mode =
        FruitLoopsSourceCenterMode::automatic;
    FruitLoopsWeightFeedbackConfig weight_feedback;
    FruitLoopsInjectedSourceTestConfig injected_source_test;
    double center_keep_radius_arcsec = 0.0;
    FruitLoopsInterpModeOverride interp_mode_override =
        FruitLoopsInterpModeOverride::automatic;
    bool legacy_center = false;
    bool recompute_weights_after_addback = false;
    int max_iters = 1;
};

struct TimestreamLearningMapPixelOutlierConfig {
    bool diagnostics_enabled = true;
    bool contributor_diagnostics_enabled = false;
    bool targeted_contributor_diagnostics_enabled = false;
    bool detector_exclusion_enabled = false;
    int top_n = 8;
    int targeted_contributor_max_pixels = 32;
    int detector_exclusion_min_pixels = 4;
    double min_abs_z = 8.0;
    double min_n_eff = 4.0;
    double source_radius_arcsec = 30.0;
};

struct TimestreamLearningBusyDetectorConfig {
    bool exclusion_enabled = true;
};

struct TimestreamLearningScanNetworkPathologyConfig {
    bool enabled = true;
    bool apply_pre_rtc = false;
    bool apply_pre_ptc = false;
    bool apply_pre_mapmaking = true;
    int min_candidate_clusters = 4;
    int min_candidate_events = 100;
    double min_max_residual_z = 25.0;
    int severe_candidate_events = 250;
    double severe_max_residual_z = 50.0;
    double max_new_flagged_fraction = 0.35;
};

struct TimestreamLearningConfig {
    bool enabled = false;
    bool diagnostics_enabled = true;
    int learn_iters = 2;
    int apply_start_iter = 2;
    int max_records_per_type = 200000;
    bool apply_sample_masks_enabled = true;
    double apply_max_new_flagged_fraction = 0.02;
    TimestreamLearningMapPixelOutlierConfig map_pixel_outlier;
    TimestreamLearningBusyDetectorConfig busy_detector;
    TimestreamLearningScanNetworkPathologyConfig scan_network_pathology;
};

struct AuxiliaryMeasuredChannelConfig {
    bool enabled = false;
    std::string name = "r";
    TodType source_type = TodType::rs;
    std::string native_unit = "native";
    AuxiliaryMeasuredChannelCalibrationPolicy calibration_policy =
        AuxiliaryMeasuredChannelCalibrationPolicy::native;
    bool apply_primary_linear_transfer = true;
    bool use_for_science_map = false;
    bool diagnostics_enabled = false;
};

struct TimestreamAuxiliaryChannelsConfig {
    AuxiliaryMeasuredChannelConfig quadrature_r;
};

struct TimestreamPolarimetryConfig {
    bool enabled = false;
    PolarimetryGrouping grouping = PolarimetryGrouping::frequency_group;
    PolarimetryHwprPolicy hwpr_policy =
        PolarimetryHwprPolicy::automatic;
};

struct TimestreamConfig {
    bool enabled = true;
    TodType type = TodType::xs;
    TimestreamAuxiliaryChannelsConfig auxiliary_channels;
    TimestreamPolarimetryConfig polarimetry;
    TimestreamOutputConfig output;
    TimestreamChunkingConfig chunking;
    RawTimeChunkConfig raw_time_chunk;
    ProcessedTimeChunkConfig processed_time_chunk;
    TimestreamFruitLoopsConfig fruit_loops;
    TimestreamLearningConfig learning;
};
}  // namespace citlali::config

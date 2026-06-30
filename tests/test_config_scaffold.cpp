#include <citlali/core/config/calibration_config.h>
#include <citlali/core/config/reduction_config.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/observation_preflight.h>
#include <citlali/core/pipeline/output_layout.h>

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

struct FakeLogger {
    template <class... Args>
    void error(const char *, Args &&...) {}

    template <class... Args>
    void info(const char *, Args &&...) {}

    template <class... Args>
    void debug(const char *, Args &&...) {}
};

struct FakeAptColumn {
    double value = 1.0;

    FakeAptColumn &array() { return *this; }

    FakeAptColumn &operator*=(double factor) {
        value *= factor;
        return *this;
    }
};

struct FakeEngine {
    std::string obsnum;
    std::string redu_dir_name = "/tmp/redu01";
    std::string obsnum_dir_name;
    bool run_coadd = false;
    bool run_map_filter = false;
    bool verbose_mode = false;

    struct {
        std::vector<std::string> obsnums;
    } omb;

    struct {
        std::vector<std::string> obsnums;
    } cmb;

    struct {
        std::map<std::string, FakeAptColumn> apt;
    } calib;

    struct {
        double fsmp = 100.0;
        double d_fsmp = -1.0;
    } telescope;

    struct {
        bool run_downsample = false;
        struct {
            int factor = 1;
            double downsampled_freq_Hz = 0.0;
        } downsampler;
        struct {
            double freq_high_Hz = 0.0;
        } filter;
    } rtcproc;
};

struct FakeFlxscaleCorrection {
    double factor = 1.0;
    double value() const { return factor; }
};

struct FakeRawObs {
    const FakeFlxscaleCorrection *correction = nullptr;
    std::string obs_name = "fake_obs";

    const FakeFlxscaleCorrection *flxscale_correction() const {
        return correction;
    }

    const std::string &name() const { return obs_name; }
};

TEST(config_scaffold, formats_config_paths) {
    EXPECT_EQ(citlali::config::format_path({"runtime", "n_threads"}),
              "runtime.n_threads");
    EXPECT_EQ(citlali::config::format_path({}), "<config>");
}

TEST(config_scaffold, validation_report_tracks_errors_and_warnings) {
    citlali::config::ValidationReport report;

    EXPECT_TRUE(report.ok());
    report.add_warning({"runtime", "verbose"}, "example warning");
    EXPECT_TRUE(report.ok());
    EXPECT_EQ(report.warning_count(), 1U);

    report.add_error({"runtime", "n_threads"}, "must be positive");
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 1U);

    auto errors = report.errors();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().path,
              (citlali::config::ConfigPath{"runtime", "n_threads"}));
}

TEST(config_scaffold, parses_existing_runtime_enum_values) {
    EXPECT_EQ(citlali::config::parse_parallel_policy("seq").value(),
              citlali::config::ParallelPolicy::seq);
    EXPECT_EQ(citlali::config::parse_parallel_policy("omp").value(),
              citlali::config::ParallelPolicy::omp);
    EXPECT_FALSE(citlali::config::parse_parallel_policy("threads").has_value());

    EXPECT_EQ(citlali::config::parse_reduction_type("science").value(),
              citlali::config::ReductionType::science);
    EXPECT_EQ(citlali::config::parse_reduction_type("pointing").value(),
              citlali::config::ReductionType::pointing);
    EXPECT_EQ(citlali::config::parse_reduction_type("beammap").value(),
              citlali::config::ReductionType::beammap);
}

TEST(config_scaffold, parses_existing_mapmaking_enum_values) {
    EXPECT_EQ(citlali::config::parse_map_grouping("auto").value(),
              citlali::config::MapGrouping::automatic);
    EXPECT_EQ(citlali::config::parse_map_grouping("detector").value(),
              citlali::config::MapGrouping::detector);
    EXPECT_EQ(citlali::config::parse_map_grouping("nw").value(),
              citlali::config::MapGrouping::network);
    EXPECT_EQ(citlali::config::parse_map_grouping("array").value(),
              citlali::config::MapGrouping::array);
    EXPECT_EQ(citlali::config::parse_map_grouping("fg").value(),
              citlali::config::MapGrouping::frequency_group);

    EXPECT_EQ(citlali::config::parse_map_method("naive").value(),
              citlali::config::MapMethod::naive);
    EXPECT_EQ(citlali::config::parse_map_method("jinc").value(),
              citlali::config::MapMethod::jinc);
    EXPECT_EQ(citlali::config::parse_map_method("maximum_likelihood").value(),
              citlali::config::MapMethod::maximum_likelihood);
}

TEST(config_scaffold, parses_existing_timestream_enum_values) {
    EXPECT_EQ(citlali::config::parse_tod_type("xs").value(),
              citlali::config::TodType::xs);
    EXPECT_EQ(citlali::config::parse_tod_type("rs").value(),
              citlali::config::TodType::rs);
    EXPECT_EQ(citlali::config::parse_tod_type("is").value(),
              citlali::config::TodType::is);
    EXPECT_EQ(citlali::config::parse_tod_type("qs").value(),
              citlali::config::TodType::qs);

    EXPECT_EQ(citlali::config::parse_tod_output_type("none").value(),
              citlali::config::TodOutputType::none);
    EXPECT_EQ(citlali::config::parse_tod_output_type("rtc").value(),
              citlali::config::TodOutputType::rtc);
    EXPECT_EQ(citlali::config::parse_tod_output_type("ptc").value(),
              citlali::config::TodOutputType::ptc);
    EXPECT_EQ(citlali::config::parse_tod_output_type("both").value(),
              citlali::config::TodOutputType::both);

    EXPECT_EQ(citlali::config::parse_tod_stream_output_mode("full").value(),
              citlali::config::TodStreamOutputMode::full);
    EXPECT_EQ(citlali::config::parse_tod_stream_output_mode("mini").value(),
              citlali::config::TodStreamOutputMode::mini);
    EXPECT_EQ(citlali::config::parse_tod_stream_output_mode("full_outer").value(),
              citlali::config::TodStreamOutputMode::full_outer);
    EXPECT_EQ(citlali::config::parse_tod_stream_output_mode("mini_outer").value(),
              citlali::config::TodStreamOutputMode::mini_outer);
    EXPECT_FALSE(citlali::config::parse_tod_stream_output_mode("outer").has_value());

    EXPECT_EQ(citlali::config::parse_tod_output_selection_mode("indices").value(),
              citlali::config::TodOutputSelectionMode::indices);
    EXPECT_EQ(citlali::config::parse_tod_output_selection_mode("all").value(),
              citlali::config::TodOutputSelectionMode::all);
    EXPECT_EQ(citlali::config::parse_tod_output_selection_mode(
                  "uniform_plus_source_crossing").value(),
              citlali::config::TodOutputSelectionMode::uniform_plus_source_crossing);
    EXPECT_FALSE(citlali::config::parse_tod_output_selection_mode("source").has_value());

    EXPECT_EQ(citlali::config::parse_raw_filter_edge_guard_mode("flag").value(),
              citlali::config::RawTimeChunkFilterEdgeGuardMode::flag);
    EXPECT_EQ(citlali::config::parse_raw_filter_edge_guard_mode("none").value(),
              citlali::config::RawTimeChunkFilterEdgeGuardMode::none);
    EXPECT_FALSE(citlali::config::parse_raw_filter_edge_guard_mode("mask").has_value());

    EXPECT_EQ(citlali::config::parse_raw_filter_edge_guard_combine("sum").value(),
              citlali::config::RawTimeChunkFilterEdgeGuardCombine::sum);
    EXPECT_EQ(citlali::config::parse_raw_filter_edge_guard_combine("max").value(),
              citlali::config::RawTimeChunkFilterEdgeGuardCombine::max);
    EXPECT_FALSE(citlali::config::parse_raw_filter_edge_guard_combine("mean").has_value());

    EXPECT_EQ(citlali::config::parse_processed_weighting_type("full").value(),
              citlali::config::ProcessedTimeChunkWeightingType::full);
    EXPECT_EQ(citlali::config::parse_processed_weighting_type("approximate").value(),
              citlali::config::ProcessedTimeChunkWeightingType::approximate);
    EXPECT_EQ(citlali::config::parse_processed_weighting_type("hybrid").value(),
              citlali::config::ProcessedTimeChunkWeightingType::hybrid);
    EXPECT_EQ(citlali::config::parse_processed_weighting_type("validated").value(),
              citlali::config::ProcessedTimeChunkWeightingType::validated);
    EXPECT_EQ(citlali::config::parse_processed_weighting_type("const").value(),
              citlali::config::ProcessedTimeChunkWeightingType::constant);
    EXPECT_FALSE(citlali::config::parse_processed_weighting_type("constant").has_value());

    EXPECT_EQ(citlali::config::parse_processed_weight_grouping("array").value(),
              citlali::config::ProcessedTimeChunkWeightGrouping::array);
    EXPECT_EQ(citlali::config::parse_processed_weight_grouping("nw").value(),
              citlali::config::ProcessedTimeChunkWeightGrouping::network);
    EXPECT_EQ(citlali::config::parse_processed_weight_grouping("all").value(),
              citlali::config::ProcessedTimeChunkWeightGrouping::all);
    EXPECT_FALSE(citlali::config::parse_processed_weight_grouping("fg").has_value());

    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("none").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::none);
    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("standard_pca").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::standard_pca);
    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("null_model").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::null_model);
    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("marchenko_pastur").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::marchenko_pastur);
    EXPECT_EQ(citlali::config::parse_processed_cleaner_mode("adaptive_selector").value(),
              citlali::config::ProcessedTimeChunkCleanerMode::adaptive_selector);
    EXPECT_FALSE(citlali::config::parse_processed_cleaner_mode("pca").has_value());

    EXPECT_EQ(citlali::config::parse_processed_corr_grouping_metric("abs").value(),
              citlali::config::ProcessedTimeChunkCorrGroupingMetric::abs);
    EXPECT_EQ(citlali::config::parse_processed_corr_grouping_metric("signed").value(),
              citlali::config::ProcessedTimeChunkCorrGroupingMetric::signed_metric);
    EXPECT_FALSE(citlali::config::parse_processed_corr_grouping_metric("pearson").has_value());

    EXPECT_EQ(citlali::config::parse_fruit_loops_mode("upper").value(),
              citlali::config::FruitLoopsMode::upper);
    EXPECT_EQ(citlali::config::parse_fruit_loops_mode("lower").value(),
              citlali::config::FruitLoopsMode::lower);
    EXPECT_EQ(citlali::config::parse_fruit_loops_mode("both").value(),
              citlali::config::FruitLoopsMode::both);
    EXPECT_FALSE(citlali::config::parse_fruit_loops_mode("absolute").has_value());

    EXPECT_EQ(citlali::config::parse_fruit_loops_weight_feedback_reference("p95").value(),
              citlali::config::FruitLoopsWeightFeedbackReference::p95);
    EXPECT_EQ(citlali::config::parse_fruit_loops_weight_feedback_reference("median").value(),
              citlali::config::FruitLoopsWeightFeedbackReference::median);
    EXPECT_EQ(citlali::config::parse_fruit_loops_weight_feedback_reference("peak").value(),
              citlali::config::FruitLoopsWeightFeedbackReference::peak);
    EXPECT_FALSE(citlali::config::parse_fruit_loops_weight_feedback_reference("mean").has_value());

    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("auto").value(),
              citlali::config::FruitLoopsInterpModeOverride::automatic);
    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("nearest").value(),
              citlali::config::FruitLoopsInterpModeOverride::nearest);
    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("bilinear").value(),
              citlali::config::FruitLoopsInterpModeOverride::bilinear);
    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("jinc").value(),
              citlali::config::FruitLoopsInterpModeOverride::jinc);
    EXPECT_EQ(citlali::config::parse_fruit_loops_interp_mode_override("trunc").value(),
              citlali::config::FruitLoopsInterpModeOverride::trunc);
    EXPECT_FALSE(citlali::config::parse_fruit_loops_interp_mode_override("legacy_nearest").has_value());
}

TEST(config_scaffold, parses_existing_pointing_enum_values) {
    EXPECT_EQ(citlali::config::parse_pointing_source_strategy("standard").value(),
              citlali::config::PointingSourceStrategy::standard);
    EXPECT_EQ(citlali::config::parse_pointing_source_strategy("psf_preserve").value(),
              citlali::config::PointingSourceStrategy::psf_preserve);

    EXPECT_EQ(citlali::config::parse_fruit_loops_center_mode("auto").value(),
              citlali::config::FruitLoopsCenterMode::automatic);
    EXPECT_EQ(citlali::config::parse_fruit_loops_center_mode("header").value(),
              citlali::config::FruitLoopsCenterMode::header);
    EXPECT_EQ(citlali::config::parse_fruit_loops_center_mode("peak").value(),
              citlali::config::FruitLoopsCenterMode::peak);
    EXPECT_EQ(citlali::config::parse_fruit_loops_center_mode("map_center").value(),
              citlali::config::FruitLoopsCenterMode::map_center);
}

TEST(config_scaffold, parses_existing_map_filter_enum_values) {
    EXPECT_EQ(citlali::config::parse_map_filter_type("wiener_filter").value(),
              citlali::config::MapFilterType::wiener_filter);
    EXPECT_EQ(citlali::config::parse_map_filter_type("convolve").value(),
              citlali::config::MapFilterType::convolve);
    EXPECT_EQ(citlali::config::parse_map_filter_type("destripe").value(),
              citlali::config::MapFilterType::destripe);
    EXPECT_FALSE(citlali::config::parse_map_filter_type("smooth").has_value());

    EXPECT_EQ(citlali::config::parse_map_filter_template_type("kernel").value(),
              citlali::config::MapFilterTemplateType::kernel);
    EXPECT_EQ(citlali::config::parse_map_filter_template_type("gaussian").value(),
              citlali::config::MapFilterTemplateType::gaussian);
    EXPECT_EQ(citlali::config::parse_map_filter_template_type("airy").value(),
              citlali::config::MapFilterTemplateType::airy);
    EXPECT_EQ(citlali::config::parse_map_filter_template_type("highpass").value(),
              citlali::config::MapFilterTemplateType::highpass);

    EXPECT_EQ(citlali::config::parse_map_filter_edge_taper_mode("none").value(),
              citlali::config::MapFilterEdgeTaperMode::none);
    EXPECT_EQ(citlali::config::parse_map_filter_edge_taper_mode("cosine").value(),
              citlali::config::MapFilterEdgeTaperMode::cosine);
}

TEST(config_scaffold, parses_existing_beammap_enum_values) {
    EXPECT_EQ(citlali::config::parse_beammap_detector_weighting_mode("const").value(),
              citlali::config::BeammapDetectorWeightingMode::constant);
    EXPECT_EQ(citlali::config::parse_beammap_detector_weighting_mode("ptc").value(),
              citlali::config::BeammapDetectorWeightingMode::ptc);
    EXPECT_EQ(citlali::config::parse_beammap_detector_weighting_mode("ptc_after_iter0").value(),
              citlali::config::BeammapDetectorWeightingMode::ptc_after_iter0);
    EXPECT_FALSE(citlali::config::parse_beammap_detector_weighting_mode("weights").has_value());
}

TEST(config_scaffold, validates_top_level_config_values) {
    citlali::config::ReductionConfig config;
    EXPECT_TRUE(citlali::config::validate(config).ok());

    config.runtime.n_threads = 0;
    config.timestream.enabled = false;
    config.mapmaking.pixel_size_arcsec = -1.0;
    config.noise.enabled = true;
    config.noise.n_noise_maps = -1;
    config.post_processing.source_fitting.active = true;
    config.post_processing.source_fitting.bounding_box_arcsec = -1.0;
    config.post_processing.map_histogram_n_bins = -1;
    config.pointing.header_max_radius_arcsec = -1.0;
    config.beammap.iteration.max_iterations = 0;

    auto report = citlali::config::validate(config);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 8U);
}

TEST(config_scaffold, validates_timestream_output_selection_values) {
    citlali::config::TimestreamConfig config;
    config.output.raw_time_chunk.outer_context_samples = -1;
    config.output.raw_time_chunk.chunks_1based.push_back(0);
    config.output.raw_time_chunk.selection_mode =
        citlali::config::TodOutputSelectionMode::uniform_plus_source_crossing;
    config.output.raw_time_chunk.selection_n_uniform = 0;
    config.output.raw_time_chunk.selection_n_source_dense = 0;
    config.output.processed_time_chunk.selection_n_uniform = -1;
    config.chunking.value = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 5U);
}

TEST(config_scaffold, validates_timestream_source_protection_values) {
    citlali::config::TimestreamConfig config;
    config.raw_time_chunk.despike.source_protection.radius_arcsec = -1.0;
    config.processed_time_chunk.flagging.second_pass_local
        .source_protection.radius_arcsec = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 2U);
}

TEST(config_scaffold, validates_timestream_despike_local_residual_values) {
    citlali::config::RawTimeChunkDespikeLocalResidualConfig config;
    config.window_sec = -1.0;
    config.sigma_scale = -1.0;
    config.delta_sigma_scale = -1.0;
    config.event_padding_sec = -1.0;
    config.high_score_event_override = -1.0;
    config.max_added_flagged_fraction = 2.0;
    config.compact_raw_gate.candidate_rel_sigma_scale = -1.0;
    config.compact_raw_gate.window_sec = -1.0;
    config.compact_raw_gate.half_peak_frac = 2.0;
    config.compact_raw_gate.max_width_sec = -1.0;
    config.compact_raw_gate.max_step_shift_z = -1.0;
    config.compact_delta_gate.window_sec = -1.0;
    config.compact_delta_gate.half_peak_frac = 2.0;
    config.compact_delta_gate.max_width_sec = -1.0;
    config.compact_delta_gate.max_step_shift_z = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 15U);
}

TEST(config_scaffold, validates_raw_time_chunk_filter_values) {
    citlali::config::RawTimeChunkConfig config;
    config.downsample.enabled = true;
    config.downsample.factor = -1;
    config.downsample.downsampled_freq_Hz = -1.0;
    config.filter.enabled = true;
    config.filter.a_gibbs = -1.0;
    config.filter.freq_low_Hz = 3.0;
    config.filter.freq_high_Hz = 2.0;
    config.filter.n_terms = -1;
    config.filter.notch.enabled = true;
    config.filter.notch.zero_phase = false;
    config.filter.notch.freqs_Hz = {-1.0};
    config.filter.edge_guard.enabled = true;
    config.filter.edge_guard.min_samples = -1;
    config.filter.edge_guard.extra_samples = -1;
    config.filter.edge_guard.max_samples = -1;
    config.filter.edge_guard.iir_settle_attenuation = 2.0;
    config.iir_filter.enabled = true;
    config.iir_filter.freq_Hz = -1.0;
    config.iir_filter.order = 0;
    config.iir_filter.zero_phase = false;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 15U);
}

TEST(config_scaffold, validates_raw_downsample_requires_filter) {
    citlali::config::RawTimeChunkConfig config;
    config.downsample.enabled = true;
    config.filter.enabled = false;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 1U);
}

TEST(config_scaffold, validates_raw_time_chunk_flagging_diagnostics) {
    citlali::config::RawTimeChunkFlaggingConfig config;
    config.network_step_mask.enabled = true;
    config.network_step_mask.step_window_sec = 0.0;
    config.network_step_mask.step_score_thresh = -1.0;
    config.network_step_mask.min_good_frac = 2.0;
    config.network_step_mask.min_det_used = 0;
    config.network_step_mask.min_step_det_frac = -1.0;
    config.network_step_mask.min_alignment_frac = 2.0;
    config.network_step_mask.cluster_tol_sec = -1.0;
    config.network_step_mask.mask_half_width_sec = -1.0;
    config.network_step_mask.max_flagged_fraction = -1.0;
    config.impulsive_capture.enabled = true;
    config.impulsive_capture.min_good_frac = -1.0;
    config.impulsive_capture.min_event_z = -1.0;
    config.impulsive_capture.near_event_z = -1.0;
    config.impulsive_capture.max_events_per_network = 0;
    config.impulsive_capture.snippet_pre_window_sec = -1.0;
    config.impulsive_capture.snippet_post_window_sec = -1.0;
    config.impulsive_coincidence.enabled = true;
    config.impulsive_coincidence.min_good_frac = 2.0;
    config.impulsive_coincidence.event_score_thresh = -1.0;
    config.impulsive_coincidence.min_det_used = 0;
    config.impulsive_coincidence.min_impulsive_det_frac = -1.0;
    config.impulsive_coincidence.min_alignment_frac = 2.0;
    config.impulsive_coincidence.min_networks_aligned = 0;
    config.impulsive_coincidence.high_score_override_thresh = -1.0;
    config.impulsive_coincidence.high_score_min_networks_aligned = -1;
    config.impulsive_coincidence.cluster_tol_sec = -1.0;
    config.impulsive_coincidence.mask_pre_window_sec = -1.0;
    config.impulsive_coincidence.mask_post_window_sec = -1.0;
    config.impulsive_coincidence.max_flagged_fraction = 2.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 27U);
}

TEST(config_scaffold, validates_raw_time_chunk_altaz_destripe_values) {
    citlali::config::RawTimeChunkAltAzDestripeConfig config;
    config.enabled = true;
    config.min_samples = 3;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 1U);
}

TEST(config_scaffold, validates_raw_time_chunk_line_audit_values) {
    citlali::config::RawTimeChunkLineAuditConfig config;
    config.line_min_hz = -1.0;
    config.line_max_hz = -1.0;
    config.segment_sec = 0.0;
    config.min_segment_sec = 0.0;
    config.overlap_frac = 1.0;
    config.continuum_radius_bins = 0;
    config.prominence_thresh = 0.0;
    config.cm_prominence_thresh = 0.0;
    config.min_good_frac = -1.0;
    config.min_windows = 0;
    config.max_peaks_per_detector = 0;
    config.max_det = -1;
    config.min_det_for_network = 0;
    config.cluster_tol_hz = -1.0;
    config.notch_min_detector_frac = 2.0;
    config.notch_min_detectors = 0;
    config.notch_min_cm_prominence = 0.0;
    config.detector_min_prominence = 0.0;
    config.detector_min_line_power_frac = -1.0;
    config.bad_detector_max_cluster_frac = 2.0;
    config.post_filter_apply_iterations = 0;
    config.post_filter_line_min_hz = -1.0;
    config.post_filter_line_max_hz = -1.0;
    config.ptc_apply_iterations = 0;
    config.ptc_line_min_hz = 2.0;
    config.ptc_line_max_hz = 1.0;
    config.fixed_notch_enabled = true;
    config.fixed_notch_widths_hz = {-1.0};
    config.fixed_notch_exclusion_half_width_hz = -1.0;
    config.apply_min_support_networks = 0;
    config.apply_min_detector_frac = -1.0;
    config.apply_min_common_mode_prominence = 0.0;
    config.apply_width_scale = 0.0;
    config.apply_min_width_hz = 2.0;
    config.apply_max_width_hz = 1.0;
    config.apply_max_notches = -1;
    config.apply_cluster_tol_hz = -1.0;
    config.detector_notch_min_prominence = 0.0;
    config.detector_notch_min_line_power_frac = 2.0;
    config.detector_notch_max_notches = -1;
    config.detector_notch_width_scale = 0.0;
    config.detector_notch_min_width_hz = 2.0;
    config.detector_notch_max_width_hz = 1.0;
    config.detector_notch_context_samples = -1;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 41U);
}

TEST(config_scaffold, validates_processed_time_chunk_second_pass_local_values) {
    citlali::config::ProcessedTimeChunkSecondPassLocalConfig config;
    config.min_spike_sigma = -1.0;
    config.min_good_frac = 2.0;
    config.baseline_window_sec = -1.0;
    config.sigma_scale = -1.0;
    config.delta_sigma_scale = -1.0;
    config.raw_candidate_rel_sigma_scale = -1.0;
    config.raw_window_sec = -1.0;
    config.raw_half_peak_frac = -1.0;
    config.raw_max_width_sec = -1.0;
    config.delta_window_sec = -1.0;
    config.delta_half_peak_frac = -1.0;
    config.delta_max_width_sec = -1.0;
    config.max_step_shift_z = -1.0;
    config.high_score_event_override = -1.0;
    config.merge_within_detector_sec = -1.0;
    config.cluster_events_sec = -1.0;
    config.min_cluster_detectors = 0;
    config.high_score_cluster_override = -1.0;
    config.max_auto_flag_clusters_per_network = 0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 19U);
}

TEST(config_scaffold, validates_processed_time_chunk_weighting_values) {
    citlali::config::ProcessedTimeChunkWeightingConfig config;
    config.source_mask_radius_arcsec = -1.0;
    config.hybrid_correction_min_factor = 2.0;
    config.hybrid_correction_max_factor = 1.0;
    config.busy_row_suppression.enabled = true;
    config.busy_row_suppression.min_candidate_clusters = -1;
    config.busy_row_suppression.min_max_unflagged_residual_z = -1.0;
    config.busy_row_suppression.factor = 2.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 5U);
}

TEST(config_scaffold, validates_processed_time_chunk_clean_values) {
    citlali::config::ProcessedTimeChunkCleanConfig config;
    config.enabled = true;
    config.standard_pca.n_calc = -1;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 1U);
}

TEST(config_scaffold, validates_processed_time_chunk_clean_expert_values) {
    citlali::config::ProcessedTimeChunkCleanConfig config;
    config.enabled = true;
    config.corr_grouping.enabled = true;
    config.corr_grouping.corr_min = 2.0;
    config.corr_grouping.min_overlap = 0;
    config.corr_grouping.min_good_frac = -1.0;
    config.corr_grouping.min_group_size = 1;
    config.corr_grouping.max_samples = -1;
    config.null_model.enabled = true;
    config.null_model.n_surrogates = 3;
    config.null_model.quantile = 0.4;
    config.null_model.min_good_frac = 2.0;
    config.null_model.max_modes = -1;
    config.null_model.max_samples = -1;
    config.null_model.seed = -1;
    config.marchenko_pastur.enabled = true;
    config.marchenko_pastur.min_good_frac = 2.0;
    config.marchenko_pastur.max_modes = -1;
    config.marchenko_pastur.max_samples = -1;
    config.marchenko_pastur.band_low_Hz = -1.0;
    config.marchenko_pastur.band_high_Hz = -1.0;
    config.marchenko_pastur.bulk_keep_frac = 0.0;
    config.marchenko_pastur.q_grid_size = 7;
    config.adaptive_selector.enabled = true;
    config.adaptive_selector.min_good_frac = 2.0;
    config.adaptive_selector.max_det = -1;
    config.adaptive_selector.max_samples = -1;
    config.adaptive_selector.max_pairs = -1;
    config.adaptive_selector.seed = -1;
    config.adaptive_selector.low_weight = -1.0;
    config.adaptive_selector.tail_weight = -1.0;
    config.adaptive_selector.topmode_weight = -1.0;
    config.adaptive_selector.reg_weight = -1.0;
    config.adaptive_selector.low_band_Hz = {-1.0, 0.5};
    config.adaptive_selector.mid_band_Hz = {2.0, 1.0};

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 29U);
}

TEST(config_scaffold, validates_processed_time_chunk_weight_validation_values) {
    citlali::config::ProcessedTimeChunkWeightValidationConfig config;
    config.enabled = true;
    config.accumulation_iters = 0;
    config.apply_start_iter = -1;
    config.min_valid_scans = 0;
    config.min_factor = 2.0;
    config.unvalidated_factor = -1.0;
    config.ratio_power = -1.0;
    config.transient_ratio_power = -1.0;
    config.upward_max_factor = 0.5;
    config.upward_power = -1.0;
    config.upward_min_base_factor = 2.0;
    config.upward_min_atmospheric_factor = -1.0;
    config.atmospheric_min_detectors = 1;
    config.atmospheric_ref = 2.0;
    config.atmospheric_span = 0.0;
    config.atmospheric_power = -1.0;
    config.min_good_frac = 2.0;
    config.min_overlap = 1;
    config.max_samples = -1;
    config.high_weight_min_group_detectors = 1;
    config.high_weight_log_robust_z = -1.0;
    config.high_weight_max_median_factor = 0.5;
    config.high_weight_cap_median_factor = 0.5;
    config.high_weight_min_validated_factor = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 23U);
}

TEST(config_scaffold, validates_processed_time_chunk_corr_penalty_values) {
    citlali::config::ProcessedTimeChunkWeightCorrPenaltyConfig config;
    config.enabled = true;
    config.min_good_frac = 2.0;
    config.min_overlap = 1;
    config.max_samples = -1;
    config.max_pairs = -1;
    config.seed = -1;
    config.floor = 2.0;
    config.exponent = -1.0;
    config.pair_corr.span = 0.0;
    config.pair_corr.weight = -1.0;
    config.cm_el_corr.span = 0.0;
    config.cm_el_corr.weight = -1.0;
    config.cm_low_mid_ratio.span = 0.0;
    config.cm_low_mid_ratio.weight = -1.0;
    config.cm_low_mid_ratio.low_band_Hz = {-1.0, 0.5};
    config.cm_low_mid_ratio.mid_band_Hz = {2.0, 1.0};

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 15U);
}

TEST(config_scaffold, validates_fruit_loops_values) {
    citlali::config::TimestreamFruitLoopsConfig config;
    config.enabled = true;
    config.peak_fraction_limit = -1.0;
    config.local_snr_floor = -1.0;
    config.local_sigma_inner_radius_arcsec = -1.0;
    config.local_sigma_outer_radius_arcsec = -1.0;
    config.local_sigma_inner_fwhm = -1.0;
    config.local_sigma_outer_fwhm = -1.0;
    config.local_sigma_edge_guard_arcsec = -1.0;
    config.local_sigma_min_pixels = 0;
    config.adaptive_support_radius_arcsec = -1.0;
    config.adaptive_support_radius_fwhm = -1.0;
    config.weight_feedback.enabled = true;
    config.weight_feedback.low_relative_weight = -1.0;
    config.weight_feedback.high_relative_weight = -2.0;
    config.center_keep_radius_arcsec = -1.0;
    config.max_iters = -1;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 15U);
}

TEST(config_scaffold, validates_timestream_learning_values) {
    citlali::config::TimestreamLearningConfig config;
    config.learn_iters = -1;
    config.apply_start_iter = -1;
    config.max_records_per_type = -1;
    config.apply_max_new_flagged_fraction = -1.0;
    config.map_pixel_outlier.top_n = -1;
    config.map_pixel_outlier.targeted_contributor_max_pixels = -1;
    config.map_pixel_outlier.detector_exclusion_min_pixels = 0;
    config.map_pixel_outlier.min_abs_z = -1.0;
    config.map_pixel_outlier.min_n_eff = -1.0;
    config.map_pixel_outlier.source_radius_arcsec = -1.0;
    config.scan_network_pathology.min_candidate_clusters = -1;
    config.scan_network_pathology.min_candidate_events = -1;
    config.scan_network_pathology.min_max_residual_z = -1.0;
    config.scan_network_pathology.severe_candidate_events = -1;
    config.scan_network_pathology.severe_max_residual_z = -1.0;
    config.scan_network_pathology.max_new_flagged_fraction = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 16U);
}

TEST(config_scaffold, validates_map_filter_config_values) {
    citlali::config::MapFilterConfig config;
    config.enabled = true;
    config.edge_guard.hits_core_fraction = -1.0;
    config.edge_guard.guard_radius_fwhm = -1.0;
    config.edge_guard.taper_min_fraction = 2.0;
    config.denom_rel_tol = 2.0;
    config.tail_frac_tol = -1.0;
    config.max_loops = 0;
    config.denom_check_iters = -1;
    config.max_denom_iters = -1;
    config.template_fwhm_arcsec[""] = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 10U);
}

TEST(config_scaffold, validates_source_finding_config_values) {
    citlali::config::SourceFindingConfig config;
    config.enabled = true;
    config.source_sigma = -1.0;
    config.source_window_arcsec = -1.0;
    config.mode.clear();

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 3U);
}

TEST(config_scaffold, validates_beammap_config_values) {
    citlali::config::BeammapConfig config;
    EXPECT_TRUE(citlali::config::validate(config).ok());

    config.phase_strategy.measurement_start_iter = 0;
    config.rfi_mask.max_flagged_fraction = 1.5;
    config.scan_band_mask.edge_rows = 1;
    config.priors.candidate_top_n = 0;
    config.priors.alignment_common_support_quantile = 0.5;
    config.detector_tod_output.n_uniform = -1;
    config.flagging.max_prior_d2 = -1.0;

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 7U);
}

TEST(config_scaffold, validates_beammap_source_values) {
    citlali::config::BeammapSourceConfig config;
    config.fluxes.push_back(citlali::config::BeammapSourceFluxConfig{"", 0.0, -1.0});

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 3U);
}

TEST(config_scaffold, validates_astrometry_pointing_offsets_values) {
    citlali::config::AstrometryPointingOffsetsConfig config;
    EXPECT_TRUE(citlali::config::validate(
                    citlali::config::AstrometryConfig{}).ok());

    config.enabled = true;
    config.az_arcsec = {1.0, 2.0, 3.0};
    config.alt_arcsec = {1.0};
    config.modified_julian_date = {1.0};

    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 3U);
}

TEST(error_scaffold, preserves_error_code_and_message) {
    auto error = citlali::error::invalid_config("bad config");
    EXPECT_EQ(error.code(), citlali::error::Code::invalid_config);
    EXPECT_STREQ(error.what(), "bad config");
}

TEST(pipeline_preflight, applies_flxscale_correction_when_present) {
    FakeEngine engine;
    engine.calib.apt["flxscale"].value = 2.0;
    FakeFlxscaleCorrection correction{1.5};
    FakeRawObs rawobs{&correction, "obs"};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::apply_flxscale_correction(
        engine, rawobs, logger));
    EXPECT_DOUBLE_EQ(engine.calib.apt["flxscale"].value, 3.0);
}

TEST(pipeline_preflight, skips_absent_flxscale_correction) {
    FakeEngine engine;
    engine.calib.apt["flxscale"].value = 2.0;
    FakeRawObs rawobs;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::apply_flxscale_correction(
        engine, rawobs, logger));
    EXPECT_DOUBLE_EQ(engine.calib.apt["flxscale"].value, 2.0);
}

TEST(pipeline_preflight, rejects_invalid_flxscale_correction) {
    FakeEngine engine;
    engine.calib.apt["flxscale"].value = 2.0;
    FakeFlxscaleCorrection correction{-1.0};
    FakeRawObs rawobs{&correction, "obs"};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::apply_flxscale_correction(
        engine, rawobs, logger));
    EXPECT_DOUBLE_EQ(engine.calib.apt["flxscale"].value, 2.0);
}

TEST(pipeline_preflight, rejects_missing_flxscale_column) {
    FakeEngine engine;
    FakeFlxscaleCorrection correction{1.5};
    FakeRawObs rawobs{&correction, "obs"};
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::apply_flxscale_correction(
        engine, rawobs, logger));
}

TEST(pipeline_preflight, configures_sample_rate_without_downsample) {
    FakeEngine engine;
    engine.telescope.fsmp = 122.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
    EXPECT_DOUBLE_EQ(engine.telescope.d_fsmp, 122.0);
}

TEST(pipeline_preflight, configures_sample_rate_with_downsample_factor) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    engine.rtcproc.run_downsample = true;
    engine.rtcproc.downsampler.factor = 4;
    engine.rtcproc.filter.freq_high_Hz = 10.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
    EXPECT_DOUBLE_EQ(engine.telescope.d_fsmp, 25.0);
}

TEST(pipeline_preflight, derives_downsample_factor_from_frequency) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    engine.rtcproc.run_downsample = true;
    engine.rtcproc.downsampler.factor = 0;
    engine.rtcproc.downsampler.downsampled_freq_Hz = 30.0;
    engine.rtcproc.filter.freq_high_Hz = 10.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_TRUE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
    EXPECT_EQ(engine.rtcproc.downsampler.factor, 3);
    EXPECT_DOUBLE_EQ(engine.telescope.d_fsmp, 100.0 / 3.0);
}

TEST(pipeline_preflight, rejects_invalid_downsample_frequency) {
    FakeEngine engine;
    engine.rtcproc.run_downsample = true;
    engine.rtcproc.downsampler.factor = 0;
    engine.rtcproc.downsampler.downsampled_freq_Hz = 0.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
}

TEST(pipeline_preflight, rejects_downsample_frequency_above_sample_rate) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    engine.rtcproc.run_downsample = true;
    engine.rtcproc.downsampler.factor = 0;
    engine.rtcproc.downsampler.downsampled_freq_Hz = 200.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
}

TEST(pipeline_preflight, rejects_downsample_filter_above_nyquist) {
    FakeEngine engine;
    engine.telescope.fsmp = 100.0;
    engine.rtcproc.run_downsample = true;
    engine.rtcproc.downsampler.factor = 4;
    engine.rtcproc.filter.freq_high_Hz = 20.0;
    auto logger = std::make_shared<FakeLogger>();

    EXPECT_FALSE(citlali::pipeline::configure_effective_sample_rate(
        engine, logger));
}

TEST(pipeline_output_layout, derives_config_copy_destinations) {
    EXPECT_EQ(citlali::pipeline::config_copy_filename("70_reduce.yaml"),
              "70_reduce.yaml");
    EXPECT_EQ(citlali::pipeline::config_copy_filename(
                  "/tmp/redu/70_reduce.yaml"),
              "70_reduce.yaml");
    EXPECT_EQ(citlali::pipeline::config_copy_destination(
                  "/tmp/redu01", "/tmp/redu/70_reduce.yaml"),
              "/tmp/redu01/70_reduce.yaml");
}

TEST(pipeline_output_layout, formats_obsnums_with_legacy_padding) {
    EXPECT_EQ(citlali::pipeline::format_obsnum(42), "000042");
    EXPECT_EQ(citlali::pipeline::format_obsnum(1234567), "1234567");
}

TEST(pipeline_output_layout, configures_observation_output_layout) {
    FakeEngine engine;
    engine.redu_dir_name = "/tmp/redu01";
    engine.omb.obsnums = {"old"};

    citlali::pipeline::configure_observation_output_layout(engine, 42);

    EXPECT_EQ(engine.obsnum, "000042");
    EXPECT_EQ(engine.obsnum_dir_name, "/tmp/redu01/000042/");
    ASSERT_EQ(engine.omb.obsnums.size(), 1U);
    EXPECT_EQ(engine.omb.obsnums.front(), "000042");
    EXPECT_TRUE(engine.cmb.obsnums.empty());
}

TEST(pipeline_output_layout, adds_observation_number_to_coadd_layout) {
    FakeEngine engine;
    engine.run_coadd = true;
    engine.cmb.obsnums = {"000001"};

    citlali::pipeline::configure_observation_output_layout(engine, 42);

    ASSERT_EQ(engine.cmb.obsnums.size(), 2U);
    EXPECT_EQ(engine.cmb.obsnums.back(), "000042");
}

}  // namespace

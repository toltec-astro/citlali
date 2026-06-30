#include <citlali/core/config/reduction_config.h>
#include <citlali/core/error/error.h>

#include <gtest/gtest.h>

namespace {

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

TEST(error_scaffold, preserves_error_code_and_message) {
    auto error = citlali::error::invalid_config("bad config");
    EXPECT_EQ(error.code(), citlali::error::Code::invalid_config);
    EXPECT_STREQ(error.what(), "bad config");
}

}  // namespace

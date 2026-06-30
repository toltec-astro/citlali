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
    config.pointing.header_max_radius_arcsec = -1.0;

    auto report = citlali::config::validate(config);
    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.error_count(), 6U);
}

TEST(error_scaffold, preserves_error_code_and_message) {
    auto error = citlali::error::invalid_config("bad config");
    EXPECT_EQ(error.code(), citlali::error::Code::invalid_config);
    EXPECT_STREQ(error.what(), "bad config");
}

}  // namespace

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_diagnostics_state.h>
#include <citlali/core/pipeline/timestream_output_config_read.h>

#include <gtest/gtest.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>
#include <tula/config/yamlconfig.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace {

struct SelectionLogger {
    template <class... Args>
    void error(const char *, Args &&...) {}

    template <class... Args>
    void info(const char *, Args &&...) {}
};

using ConfigPath = std::vector<std::string>;

bool contains_path(
    const citlali::pipeline::ConfigDiagnosticsState &diagnostics,
    const ConfigPath &path) {
    const auto &invalid = diagnostics.invalid_key_paths();
    return std::find(invalid.begin(), invalid.end(), path) != invalid.end();
}

std::shared_ptr<SelectionLogger> make_selection_logger() {
    if (!spdlog::get("citlali_logger")) {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        spdlog::register_logger(
            std::make_shared<spdlog::logger>("citlali_logger", sink));
    }
    return std::make_shared<SelectionLogger>();
}

}  // namespace

TEST(tod_output_selection_config, records_invalid_chunk_list_atomically) {
    auto config = tula::config::YamlConfig::from_str(R"yaml(
timestream:
  raw_time_chunk:
    output:
      indices: [1, 0, 3]
      selection:
        mode: indices
        n_uniform: 10
        n_source_dense: 10
)yaml");
    citlali::config::TimestreamOutputConfig output;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;
    auto logger = make_selection_logger();

    citlali::pipeline::read_tod_output_selection_config(
        config, true, false, output, diagnostics, logger);

    EXPECT_TRUE(contains_path(
        diagnostics,
        {"timestream", "raw_time_chunk", "output", "indices"}));
    EXPECT_FALSE(output.raw_time_chunk.chunk_select_enabled);
    EXPECT_TRUE(output.raw_time_chunk.chunks_1based.empty());
}

TEST(tod_output_selection_config, records_invalid_selection_count_path) {
    auto config = tula::config::YamlConfig::from_str(R"yaml(
timestream:
  raw_time_chunk:
    output:
      indices: all
      selection:
        mode: indices
        n_uniform: -1
        n_source_dense: 10
)yaml");
    citlali::config::TimestreamOutputConfig output;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;
    auto logger = make_selection_logger();

    citlali::pipeline::read_tod_output_selection_config(
        config, true, false, output, diagnostics, logger);

    EXPECT_TRUE(contains_path(
        diagnostics,
        {"timestream", "raw_time_chunk", "output", "selection",
         "n_uniform"}));
    EXPECT_EQ(output.raw_time_chunk.selection_n_uniform, 10);
}

TEST(tod_output_selection_config, records_impossible_selection_mode_path) {
    auto config = tula::config::YamlConfig::from_str(R"yaml(
timestream:
  raw_time_chunk:
    output:
      indices: all
      selection:
        mode: uniform_plus_source_crossing
        n_uniform: 0
        n_source_dense: 0
)yaml");
    citlali::config::TimestreamOutputConfig output;
    citlali::pipeline::ConfigDiagnosticsState diagnostics;
    auto logger = make_selection_logger();

    citlali::pipeline::read_tod_output_selection_config(
        config, true, false, output, diagnostics, logger);

    EXPECT_TRUE(contains_path(
        diagnostics,
        {"timestream", "raw_time_chunk", "output", "selection", "mode"}));
}

TEST(tod_output_selection_config, rejects_invalid_effective_mode) {
    citlali::config::TodStreamOutputConfig config;
    config.selection_mode =
        static_cast<citlali::config::TodOutputSelectionMode>(-1);
    Eigen::VectorXI scan_to_output;
    Eigen::Index n_output_scans = 0;
    auto logger = make_selection_logger();

    EXPECT_THROW(
        citlali::pipeline::configure_tod_output_stream_selection(
            "raw", true, config, 2, {}, scan_to_output, n_output_scans,
            logger),
        citlali::error::Error);
}

TEST(tod_output_selection_config, rejects_empty_source_crossing_selection) {
    citlali::config::TodStreamOutputConfig config;
    config.selection_mode =
        citlali::config::TodOutputSelectionMode::uniform_plus_source_crossing;
    Eigen::VectorXI scan_to_output;
    Eigen::Index n_output_scans = 0;
    auto logger = make_selection_logger();

    EXPECT_THROW(
        citlali::pipeline::configure_tod_output_stream_selection(
            "raw", true, config, 2, {}, scan_to_output, n_output_scans,
            logger),
        citlali::error::Error);
}

TEST(tod_output_selection_config, rejects_chunk_outside_scan_range) {
    citlali::config::TodStreamOutputConfig config;
    config.chunk_select_enabled = true;
    config.chunks_1based = {1, 3};
    Eigen::VectorXI scan_to_output;
    Eigen::Index n_output_scans = 0;
    auto logger = make_selection_logger();

    EXPECT_THROW(
        citlali::pipeline::configure_tod_output_stream_selection(
            "raw", true, config, 2, {}, scan_to_output, n_output_scans,
            logger),
        citlali::error::Error);
}

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

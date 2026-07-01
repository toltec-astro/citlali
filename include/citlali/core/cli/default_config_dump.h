#pragma once

#include <fmt/core.h>
#include <tula/cli.h>

#include <string_view>

namespace citlali::cli {

inline bool dump_default_config_if_requested(
    int argc, char *argv[], std::string_view git_version,
    std::string_view build_timestamp, std::string_view default_config_content) {
    bool exit_dump_config{false};
    clipp::parse(argc, argv,
                 (clipp::option("--dump_config")
                      .call([&exit_dump_config, git_version, build_timestamp,
                             default_config_content]() {
                          auto preamble = fmt::format(
                              "# Default config.yaml of Citlali {} ({})",
                              git_version, build_timestamp);
                          fmt::print("{}\n{}", preamble,
                                     default_config_content);
                          exit_dump_config = true;
                      }),
                  clipp::any_other()));
    return exit_dump_config;
}

}  // namespace citlali::cli

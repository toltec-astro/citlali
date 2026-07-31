#pragma once

#include <citlali_config/gitversion.h>
#include <kidscpp_config/gitversion.h>
#include <fmt/core.h>
#include <tula/cli.h>
#include <tula/config/flatconfig.h>
#include <tula/config/yamlconfig.h>
#include <tula/grppi.h>
#include <tula/logging.h>

#include <cstdlib>
#include <string>
#include <utility>

namespace citlali::cli {

using RuntimeConfig = tula::config::YamlConfig;

inline std::string citlali_version_string() {
    return fmt::format("{} ({})", CITLALI_GIT_VERSION,
                       CITLALI_BUILD_TIMESTAMP);
}

inline std::string kidscpp_version_string() {
    return fmt::format("kids {} ({})", KIDSCPP_GIT_VERSION,
                       KIDSCPP_BUILD_TIMESTAMP);
}

inline std::string build_provenance_string() {
#ifdef CITLALI_SPACK_DAG_HASH
    return fmt::format("build {} dag={} compiler={}-{} cxx={} {} {}",
                       CITLALI_BUILD_SYSTEM, CITLALI_SPACK_DAG_HASH,
                       CITLALI_CMAKE_CXX_COMPILER_ID,
                       CITLALI_CMAKE_CXX_COMPILER_VERSION,
                       CITLALI_CMAKE_CXX_STANDARD,
                       CITLALI_CMAKE_BUILD_TYPE, CITLALI_BUILD_VARIANTS);
#else
    return "";
#endif
}

inline auto default_cli_log_level_name() {
    auto v = spdlog::level::info;
    if (v < tula::logging::active_level) {
        v = tula::logging::active_level;
    }
    return tula::logging::get_level_name(v);
}

template <class CliConfig>
void apply_cli_log_level(const CliConfig &cli_config) {
    auto log_level_str = cli_config.get_str("log_level");
    auto log_level = spdlog::level::from_str(log_level_str);
    spdlog::set_level(log_level);
    SPDLOG_INFO("reconfigure logger to level={}", log_level_str);
}

inline RuntimeConfig parse_args(int argc, char *argv[]) {
    // disable logger before parse
    spdlog::set_level(spdlog::level::off);
    using namespace tula::cli::clipp_builder;

    // some of the option specs
    auto ver_str = citlali_version_string();
    auto kids_ver_str = kidscpp_version_string();
    auto build_provenance_str = build_provenance_string();
    constexpr auto level_names = tula::logging::active_level_names;
    auto default_level_name = default_cli_log_level_name();
    using ex_config = tula::grppi_utils::ex_config;
    // clang-format off
    auto parse = config_parser<RuntimeConfig, tula::config::FlatConfig>{};
    auto screen = tula::cli::screen{
    // =======================================================================
                      "citlali" , CITLALI_PROJECT_NAME, ver_str,
                                  CITLALI_PROJECT_DESCRIPTION};
    auto [cli, rc, cc] = parse([&](auto &r, auto &c) { return (
    // rc -- runtime config
    // cc -- cli config
    // =======================================================================
    c(p(           "h", "help"), "Print help information and exit."),
    c(p(             "version"), "Print version information and exit."),
    // =======================================================================
    r(             "config_file" , "The path of input config file. "
                                 "Multiple config file are merged in order.",
                                 opt_strs()),
    c(p(          "dump_config"), "Print the default config file to STDOUT."),
    // =======================================================================
              "common options" % g(
    c(p(      "l", "log_level"), "Set the log level.",
                                 default_level_name, list(level_names)),
    r(p(             "grppiex"), "GRPPI execution policy.",
                                 ex_config::default_mode(),
                                 list(ex_config::mode_names_supported())))
    // =======================================================================
    );}, screen, argc, argv);
    // clang-format on
    if (cc.get_typed<bool>("help")) {
        screen.manpage(cli);
        std::exit(EXIT_SUCCESS);
    } else if (cc.get_typed<bool>("version")) {
        screen.version();
        // also print the kids version
        fmt::print("{}\n", kids_ver_str);
        if (!build_provenance_str.empty()) {
            fmt::print("{}\n", build_provenance_str);
        }
        std::exit(EXIT_SUCCESS);
    }
    apply_cli_log_level(cc);
    // pass on the runtime config
    return std::move(rc);
}

}  // namespace citlali::cli

#pragma once

#include <citlali_config/gitversion.h>
#include <kidscpp_config/gitversion.h>
#include <fmt/core.h>
#include <tula/logging.h>

#include <string>

namespace citlali::cli {

inline std::string citlali_version_string() {
    return fmt::format("{} ({})", CITLALI_GIT_VERSION,
                       CITLALI_BUILD_TIMESTAMP);
}

inline std::string kidscpp_version_string() {
    return fmt::format("kids {} ({})", KIDSCPP_GIT_VERSION,
                       KIDSCPP_BUILD_TIMESTAMP);
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

}  // namespace citlali::cli

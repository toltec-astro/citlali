#pragma once

#include <citlali_config/gitversion.h>
#include <kidscpp_config/gitversion.h>
#include <fmt/core.h>

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

}  // namespace citlali::cli

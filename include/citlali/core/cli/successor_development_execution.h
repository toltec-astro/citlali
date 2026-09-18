#pragma once

#include <filesystem>
#include <optional>
#include <string_view>

namespace citlali::cli {

// Application-boundary execution of the existing concrete RTC owners. The
// request file binds native x/r, timing, VAL, motion and explicit finite plans;
// this function does not choose scientific policy from command-line defaults.
enum class RtcInvocation { development_default, comparison };
int run_successor_rtc(const std::filesystem::path &input,
                      const std::filesystem::path &output,
                      RtcInvocation invocation,
                      std::optional<std::filesystem::path> reviewed_selection = {});

inline constexpr std::string_view successor_default_config = R"yaml(# Citlali development default: the implemented Timestream Successor.
# Invoke with: citlali development.yaml
# Current endpoint is RTC-only. CAL/PTC/MAP are not substituted by legacy code.
# Supply the existing exact-bound native x/r and frozen-treatment input record.
schema: citlali-development-v1
input:
  path: /path/to/rtc-input.json
  sha256: REPLACE_WITH_INPUT_FILE_SHA256
output: /path/to/new-output-directory
terminal: rtc-only
)yaml";

} // namespace citlali::cli

#pragma once

#include <filesystem>
#include <optional>
#include <string_view>

namespace citlali::cli {

// Application-boundary execution of the existing concrete RTC owners. The
// request file binds native x/r, timing, VAL, motion and explicit finite plans;
// this function does not choose scientific policy from command-line defaults.
enum class RtcInvocation { development_default, comparison };
enum class DevelopmentTerminal { rtc_only, cal };
int run_successor_rtc(const std::filesystem::path &input,
                      const std::filesystem::path &output,
                      RtcInvocation invocation,
                      std::optional<std::filesystem::path> reviewed_selection = {},
                      DevelopmentTerminal terminal = DevelopmentTerminal::rtc_only);

inline constexpr std::string_view successor_default_config = R"yaml(# Citlali development default: the implemented Timestream Successor.
# Invoke with: citlali development.yaml
# Current endpoint is CAL; unsupported opacity is reported without a fallback.
# PTC/MAP are not yet connected. Use terminal: rtc-only for an explicit early stop.
# Supply the existing exact-bound native x/r and frozen-treatment input record.
schema: citlali-development-v1
input:
  path: /path/to/rtc-input.json
  sha256: REPLACE_WITH_INPUT_FILE_SHA256
output: /path/to/new-output-directory
terminal: cal
)yaml";

} // namespace citlali::cli

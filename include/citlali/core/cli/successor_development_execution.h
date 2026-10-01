#pragma once

#include <filesystem>
#include <optional>
#include <string_view>
#include <string>

namespace citlali::cli {

// Application-boundary execution of the existing concrete RTC owners. The
// request file binds native x/r, timing, VAL, motion and explicit finite plans;
// this function does not choose scientific policy from command-line defaults.
enum class RtcInvocation { development_default, comparison };
enum class DevelopmentTerminal { rtc_only, cal, ptc, prepare };
struct DevelopmentPtcRequest {
    std::string method="observed-als";
    int rank=0; // zero means recover the explicitly configured effective rank
};
int run_successor_saved_map(const std::filesystem::path &input,const std::filesystem::path &output);
// Contract-only mode shares native admission and returns before detector processing.
int run_successor_saved_fruit(const std::filesystem::path &input,const std::filesystem::path &output,bool validate_only=false);
int run_successor_rtc(const std::filesystem::path &input,
                      const std::filesystem::path &output,
                      RtcInvocation invocation,
                      std::optional<std::filesystem::path> reviewed_selection = {},
                      DevelopmentTerminal terminal = DevelopmentTerminal::rtc_only,
                      DevelopmentPtcRequest ptc = {}, unsigned network_workers = 1);

inline constexpr std::string_view successor_default_config = R"yaml(# Citlali development default: the implemented Timestream Successor.
# Invoke with: citlali development.yaml
# terminal: map consumes an exact-bound citlali-map-saved-ptc-v1 input.
# It reuses saved PTC and reconstructs only its exact AST coordinates. FRUIT is deferred.
# Use terminal: cal or rtc-only for an explicit early stop.
# PTC rank/grouping come from the bound effective configuration unless rank is explicit.
# Supply the existing exact-bound native x/r and frozen-treatment input record.
schema: citlali-development-v1
input:
  path: /path/to/rtc-input.json
  sha256: REPLACE_WITH_INPUT_FILE_SHA256
output: /path/to/new-output-directory
terminal: ptc
ptc:
  method: observed-als # explicit comparison: pairwise-covariance
)yaml";

} // namespace citlali::cli

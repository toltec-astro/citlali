#pragma once
#include <string_view>

// Private CLI request gate; no scientific policy or numerical operation.
namespace citlali::cli::detail {
inline bool point_rtc_probe_route_allowed(bool development, bool common,
    bool reference, int obs, int sub, int scan, int network, int array,
    std::string_view authority) {
  if (!development || !common || sub != 0 || scan != 2) return false;
  if (authority == "owner-RTC-step-consequences-2026-09-28")
    return reference && obs == 152430 && network >= 1 && network <= 4 && array == 0;
  if (network != 11 && network != 12) return false;
  if (authority == "owner-RTC-cost-response-2026-09-27") return obs == 152430;
  if (authority == "owner-RTC-step-POINT-2026-09-27")
    return obs == 152420 || obs == 152430;
  return false;
}
}

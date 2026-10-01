#pragma once
#include <string_view>
#include <yaml-cpp/yaml.h>

// Private, bounded acquisition routing only. Scientific owners do not depend on it.
namespace citlali::cli::detail {
inline bool oof_acquisition_scope(int obs, int sub, int scan) {
  return obs >= 152385 && obs <= 152387 && sub == 0 && scan == 1;
}
inline bool oof_request(const YAML::Node &n) {
  return n["development_reference"] &&
      n["development_reference"].as<std::string>() == "oof-cold-m0-v1";
}
inline bool oof_scope(const YAML::Node &n, int obs, int sub, int scan) {
  return oof_request(n) && oof_acquisition_scope(obs, sub, scan) &&
      n["mode"] && n["mode"].as<std::string>() == "OOF" &&
      n["observation"].as<int>() == obs && n["subobservation"] &&
      n["subobservation"].as<int>() == sub && n["scan"] && n["scan"].as<int>() == scan;
}
inline bool oof_network_scope(const YAML::Node &n, int obs, int sub, int scan, int nw) {
  return oof_scope(n, obs, sub, scan) && nw >= 1 && nw <= 4 &&
      n["network"].as<int>() == nw;
}
inline constexpr std::string_view oof_no_mask_authority = "owner-OOF-cold-M0-no-spatial-mask-2026-10-01";
}

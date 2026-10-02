#pragma once
#include <string_view>
#include <string>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

// Private, bounded acquisition routing only. Scientific owners do not depend on it.
namespace citlali::cli::detail {
inline bool oof_acquisition_scope(int obs, int sub, int scan) {
  return obs >= 152385 && obs <= 152387 && sub == 0 && scan == 1;
}
inline bool oof_request(const YAML::Node &n) {
  return n["development_reference"] &&
      (n["development_reference"].as<std::string>() == "oof-cold-m0-v1" ||
       n["development_reference"].as<std::string>() == "oof-a2000-m0-v1");
}
inline bool oof_scope(const YAML::Node &n, int obs, int sub, int scan) {
  return oof_request(n) && oof_acquisition_scope(obs, sub, scan) &&
      n["mode"] && n["mode"].as<std::string>() == "OOF" &&
      n["observation"].as<int>() == obs && n["subobservation"] &&
      n["subobservation"].as<int>() == sub && n["scan"] && n["scan"].as<int>() == scan;
}
inline int oof_array(const YAML::Node &n) {
  return n["development_reference"].as<std::string>() == "oof-a2000-m0-v1" ? 2 : 0;
}
inline bool oof_network_scope(const YAML::Node &n, int obs, int sub, int scan, int nw) {
  return oof_scope(n, obs, sub, scan) &&
      (oof_array(n)==2 ? (nw==11 || nw==12) : (nw>=1 && nw<=4)) &&
      n["network"].as<int>() == nw;
}
// Saved MAP acquisition identity follows the same validated routing as CAL/AST.
inline std::string saved_map_observation_identity(const YAML::Node &n, int obs, int nw, int array) {
  const bool is_oof = oof_request(n);
  if (is_oof && (!oof_network_scope(n,obs,0,1,nw) || array != oof_array(n)))
    throw std::invalid_argument("MAP OOF acquisition or array differs");
  if (!is_oof && ((n["scan"] && n["scan"].as<int>() != 2) ||
      (n["subobservation"] && n["subobservation"].as<int>() != 0) ||
      (n["mode"] && n["mode"].as<std::string>() != "POINT")))
    throw std::invalid_argument("MAP POINT acquisition differs");
  return "obs:" + std::to_string(obs) + (is_oof ? "_0_1" : "_0_2");
}
inline constexpr std::string_view oof_no_mask_authority = "owner-OOF-cold-M0-no-spatial-mask-2026-10-01";
}

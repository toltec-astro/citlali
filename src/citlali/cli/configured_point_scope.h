#pragma once
#include <yaml-cpp/yaml.h>
#include <string>

// Application-boundary scope, not a scientific acceptance decision.
namespace citlali::cli::detail {
inline bool configured_point_request(const YAML::Node &n) {
  return n["development_reference"] &&
      n["development_reference"].as<std::string>() == "configurable-point-v1";
}
inline bool configured_point_scope(const YAML::Node &n, int obs, int sub, int scan) {
  return configured_point_request(n) && obs > 0 && sub == 0 && scan == 2 &&
      n["mode"] && n["mode"].as<std::string>() == "POINT" &&
      n["observation"] && n["observation"].as<int>() == obs &&
      n["subobservation"] && n["subobservation"].as<int>() == sub &&
      n["scan"] && n["scan"].as<int>() == scan;
}
}

#pragma once
// Bounded application binding. Existing RTC rejection owns all support semantics.
#include <yaml-cpp/yaml.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
namespace citlali::cli::detail {
inline constexpr const char *rtc_quarantine_digest =
    "a8edbfecdc297e06fd8625fbd1b435ff1b43bae60de852043cac0fcda995f1c5";
inline std::vector<std::uint32_t> read_rtc_quarantine(const YAML::Node &n,
    int obs,int sub,int scan,int nw,int array,
    const std::vector<std::uint32_t> &channels,
    const std::vector<std::string> &occurrences,
    const std::vector<std::pair<std::int64_t,std::int64_t>> &runs) {
    auto check=[](bool ok){if(!ok)throw std::invalid_argument("RTC frozen whole-run quarantine binding mismatch");};
    check(n["schema"].as<std::string>()=="citlali-rtc-whole-run-quarantine-v1" &&
          n["authority"].as<std::string>()=="owner-RTC-two-stage-2026-09-28" &&
          n["policy"].as<std::string>()=="whole-physical-run-paired-rejection" &&
          !n["automatic_selection"].as<bool>() && !n["production_certified"].as<bool>());
    check(obs==152430 && sub==0 && scan==2 && array==0 && nw>=1 && nw<=4 &&
          n["observation"].as<int>()==obs && n["subobservation"].as<int>()==sub &&
          n["scan"].as<int>()==scan && n["array"].as<std::string>()=="a1100" &&
          n["networks"].as<std::vector<int>>()==std::vector<int>({1,2,3,4}));
    check(n["members"].size()==1 && channels.size()==occurrences.size());
    const auto m=n["members"][0];
    check(m["network"].as<int>()==2 && m["detector"].as<int>()==88 &&
          m["channel"].as<int>()==88 && m["physical_run"].as<int>()==0 &&
          m["native_interval"].as<std::vector<std::int64_t>>()==std::vector<std::int64_t>({0,7821}));
    if(nw!=2)return {};
    check(runs==std::vector<std::pair<std::int64_t,std::int64_t>>({{0,7821}}) &&
          channels.size()>88 && channels[88]==88 &&
          occurrences[88]==m["occurrence"].as<std::string>() && !occurrences[88].empty());
    return {88};
}
} // namespace citlali::cli::detail

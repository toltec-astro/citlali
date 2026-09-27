#pragma once
// Private application-boundary binding only. RTC owners apply the existing FIRs.
#include <citlali/core/pipeline/timestream_native_paired_readout.h>
#include <citlali/core/utils/sha256.h>
#include <yaml-cpp/yaml.h>
#include <bit>
#include <vector>
#include <utility>
#include <stdexcept>
namespace citlali::cli::detail {
inline void validate_point_fixed11(const YAML::Node &cfg,const YAML::Node &selection,
    const YAML::Node &design,int network,double measured_interval,
    const std::vector<std::pair<std::int64_t,std::int64_t>> &runs,
    const std::vector<pipeline::NativeReadoutDetectorBinding> &detectors,
    const std::vector<int> &channels) {
  const auto check=[](bool ok){if(!ok)throw std::invalid_argument("fixed11 experiment binding mismatch");};
  check(selection["schema"].as<std::string>()=="point-fixed11-applicability-v1" &&
    selection["authority"].as<std::string>()=="owner-fixed11-comparison-2026-09-27" &&
    selection["observation"].as<std::vector<int>>()==std::vector<int>{152430,0,2} &&
    selection["tune"].as<std::vector<int>>()==std::vector<int>{152430,0,1} &&
    selection["array"].as<std::string>()=="a2000" && selection["frozen_before_candidate"].as<bool>() &&
    cfg["observation"].as<int>()==152430 && cfg["network"].as<int>()==network &&
    (network==11 || network==12));
  const auto arm=cfg["fixed11"]["arm"].as<std::string>();check(arm=="B-control" || arm=="C-fixed11");
  check(design["schema"].as<std::string>()=="point-fixed11-design-v1" &&
    design["historical_center_hz"].as<double>()==11.006255844577186 &&
    design["requested_width_hz"].as<double>()==1 && design["requested_span_seconds"].as<double>()==3 &&
    design["measured_interval_seconds"].as<double>()==measured_interval &&
    design["factor"].as<int>()==2 && design["phase"].as<int>()==0 && design["paired_coordinates"].as<bool>());
  const auto h=design["finite_notch"].as<std::vector<double>>();
  check(std::endian::native==std::endian::little && h.size()==367);
  utils::Sha256 hash;hash.update(reinterpret_cast<const std::uint8_t*>(h.data()),h.size()*sizeof(double));
  const auto sha=hash.finish();check(sha=="879c4b7978b3ce08ca6844594e8730ba717e459d3e5db387f7a15957eb055fa7" && design["coefficients_sha256"].as<std::string>()==sha);
  if(arm=="C-fixed11")check(cfg["finite_notch"].as<std::vector<double>>()==h &&
    cfg["finite_notch_identity"].as<std::string>()==design["finite_notch_identity"].as<std::string>());
  else check(!cfg["finite_notch"]);
  YAML::Node net;int matches=0;
  for(std::size_t i=0;i<selection["networks"].size();++i)if(selection["networks"][i]["network"].as<int>()==network){net=selection["networks"][i];++matches;}
  check(matches==1 && net["measured_interval_seconds"].as<double>()==measured_interval &&
      net["native_runs"].size()==runs.size() && runs.size()==1);
  for(std::size_t i=0;i<runs.size();++i)check(net["native_runs"][i][0].as<std::int64_t>()==runs[i].first && net["native_runs"][i][1].as<std::int64_t>()==runs[i].second);
  for(const auto *key:{"raw","tune","manifest","telescope","filter_plan"})check(net["input_bindings"][key]["sha256"].as<std::string>()==cfg[key]["sha256"].as<std::string>());
  check(net["detectors"].size()==detectors.size() && channels.size()==detectors.size() && cfg["detectors"].size()==detectors.size());
  for(std::size_t d=0;d<detectors.size();++d){
    const auto n=net["detectors"][d];const auto disposition=n["disposition"].as<std::string>();
    check(disposition=="selected" || disposition=="unselected" || disposition=="uncertain");
    check(n["detector"].as<std::size_t>()==d && n["channel"].as<int>()==channels[d] && n["physical_run"].as<int>()==0 &&
      n["occurrence"].as<std::string>()==detectors[d].detector_occurrence_id &&
      n["array_association"].as<std::string>()==detectors[d].detector_association_record_id &&
      cfg["detectors"][d]["channel"].as<int>()==channels[d] &&
      cfg["detectors"][d]["filter"].as<std::string>()==((arm=="C-fixed11" && disposition=="selected")?"w1-t3":"lowpass"));
  }
}
}

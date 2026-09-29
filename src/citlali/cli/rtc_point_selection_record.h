#pragma once
#include <citlali/core/pipeline/rtc_point_run_selection.h>
#include <yaml-cpp/yaml.h>
namespace citlali::cli::detail {
inline YAML::Node rtc_point_selection_record(const pipeline::RtcPointRunSelectionPlan &p) {
  YAML::Node n;n["schema"]=p.identity;n["parent_identity"]=p.parent_identity();n["pointing_identity"]=p.pointing_identity();
  n["population"]=YAML::Node(YAML::NodeType::Sequence);
  for(const auto &v:p.records()) {
    YAML::Node q;q["detector"]=v.detector;q["physical_run"]=v.run;
    q["native_interval"]=std::vector<std::int64_t>{p.runs()[v.run].first,p.runs()[v.run].past_last};
    q["proposals"]=v.proposals;q["eligible"]=v.eligible;q["contrast_pass"]=v.contrast_pass;q["selected"]=v.selected;
    if(v.selected) {const auto &e=v.event;
      q["event"]["onset"]=e.onset;q["event"]["window"]=std::vector<std::int64_t>{e.first,e.past_last};
      q["event"]["peers"]=e.peers;q["event"]["coherent"]=e.coherent;
      for(int c=0;c<2;++c){const auto key=c?"r":"x";q["event"][key]["contrast"]=e.contrast[c];q["event"][key]["cohort_center"]=e.cohort_center[c];q["event"][key]["cohort_scale"]=e.cohort_scale[c];q["event"][key]["adjacent_max"]=e.adjacent_max[c];}
    }
    n["population"].push_back(q);
  }
  return n;
}
}

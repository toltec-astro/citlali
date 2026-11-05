#pragma once

#include <citlali/core/pipeline/point_optical_response.h>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <string>

namespace citlali::cli::detail {

inline void point_number(YAML::Node node,const char *key,double value) {
    if(std::isfinite(value))node[key]=value;else node[key]=YAML::Null;
}
inline YAML::Node point_interval(const pipeline::PointOpticalInterval &x) {
    YAML::Node n;n["available"]=x.available;point_number(n,"lower",x.lower);point_number(n,"upper",x.upper);
    n["informative_blocks"]=x.informative_blocks;n["resamples"]=x.resamples;n["seed"]=x.seed;n["reason"]=x.reason;return n;
}

inline YAML::Node point_health_header(const pipeline::PointOpticalPlan &plan, const char *schema) {
    YAML::Node record;
    record["schema"]=schema;
    record["method"]=std::string(pipeline::PointOpticalEvidence::method);
    record["uncertainty_limitations"]=std::string(pipeline::PointOpticalEvidence::uncertainty);
    record["bootstrap_seed_policy"]=std::string(pipeline::PointOpticalEvidence::seed_policy);
    record["bootstrap_seed_identity"]=plan.evidence_handle()->seed_identity();
    record["min_relative_response"]=plan.min_relative_response();
    return record;
}

inline YAML::Node point_health_record(const pipeline::PointOpticalPlan &plan) {
    auto record=point_health_header(plan,"citlali-point-optical-health-v2");
    const auto &r=plan.evidence_handle()->request();
    record["bin_seconds"]=r.bin_seconds;record["confidence_level"]=r.confidence_level;record["bootstrap_samples"]=r.bootstrap_samples;
    for(std::size_t ni=0;ni<plan.evidence_handle()->networks().size();++ni) {
        const auto &e=plan.evidence_handle()->networks()[ni];YAML::Node n;
        n["observation_identity"]=e.observation_identity;n["cal_identity"]=e.cal_identity;n["source_mask_identity"]=e.source_mask_identity;
        n["array"]=e.array;n["network"]=e.network;n["detector_ids"]=e.detector_ids;n["reference_networks"]=e.reference_networks;
        n["reference_available"]=e.reference_available;n["cross_array_required"]=e.cross_array_required;
        n["cross_array_corroborated"]=e.cross_array_corroborated;n["reference_reason"]=e.reference_reason;
        point_number(n,"response_normalization",e.response_normalization);
        for(std::size_t di=0;di<e.detectors.size();++di) {
            const auto &d=e.detectors[di];const auto &decision=plan.decisions()[ni][di];YAML::Node v;
            v["detector_id"]=d.detector_id;v["available_samples"]=d.available_samples;v["source_free_samples"]=d.source_free_samples;v["binned_samples"]=d.binned_samples;
            point_number(v,"relative_response",d.relative_response);point_number(v,"correlation",d.correlation);point_number(v,"binned_residual_rms",d.binned_residual_rms);
            v["interval"]=point_interval(d.interval);for(const auto &c:d.checks)v["checks"].push_back(point_interval(c));
            v["state"]=std::string(point_optical_state_name(decision.state));v["reason"]=decision.reason;n["detectors"].push_back(v);
        }
        record["networks"].push_back(n);
    }
    return record;
}

// Build the user report from the frozen plan. Assigning nodes from the full
// YAML document to each new row makes yaml-cpp merge the full node pool.
inline YAML::Node point_decision_record(const pipeline::PointOpticalPlan &plan) {
    // Keep the v1 compact field order as well as its data. This header has
    // only scalar metadata and has no connection to the full evidence tree.
    const auto header=point_health_header(plan,"citlali-point-optical-decisions-v1");
    YAML::Node report;
    for(const auto *key:{"schema","method","min_relative_response","bootstrap_seed_policy","bootstrap_seed_identity","uncertainty_limitations"})
        report[key]=header[key];
    const auto &networks=plan.evidence_handle()->networks();
    for(std::size_t i=0;i<networks.size();++i) {
        const auto &network=networks[i];
        YAML::Node group;
        group["network"]=network.network;
        group["array"]=network.array;
        for(std::size_t j=0;j<network.detectors.size();++j) {
            const auto &detector=network.detectors[j];
            const auto &decision=plan.decisions()[i][j];
            YAML::Node row;
            row["detector_id"]=detector.detector_id;
            row["state"]=std::string(pipeline::point_optical_state_name(decision.state));
            if(!decision.retained() && decision.state!=pipeline::PointOpticalState::prior_unavailable) {
                row["reason"]=decision.reason;
                point_number(row,"relative_response",detector.relative_response);
                row["interval"]=point_interval(detector.interval);
            }
            group["detectors"].push_back(row);
        }
        report["networks"].push_back(group);
    }
    return report;
}

}  // namespace citlali::cli::detail

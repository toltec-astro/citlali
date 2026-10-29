#pragma once
// Private application-boundary checks for persisted configured POINT products.
// Callers authenticate the actual files with their file ledger before passing
// the observed digests here. No scientific decisions or filesystem I/O live here.
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::cli::detail {
namespace point_binding_private {
inline void require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument("POINT " + message);
}
inline bool sha256(std::string_view value) {
    return value.size()==64 && std::all_of(value.begin(),value.end(),[](char c) {
        return (c>='0' && c<='9') || (c>='a' && c<='f');
    });
}
template<class Work> decltype(auto) checked_yaml(Work work) {
    try { return work(); }
    catch (const YAML::Exception &error) {
        throw std::invalid_argument(std::string("POINT malformed persisted binding: ")+error.what());
    }
}
} // namespace point_binding_private

struct PointFeedbackBinding {
    std::string candidate_sha256, bundle_sha256, map_sha256;
    std::string observation, health_identity, array;
    int previous_cycle=-1;
};

inline void validate_point_feedback_provenance(const YAML::Node &feedback,
    const YAML::Node &bundle,const PointFeedbackBinding &expected) {
    using namespace point_binding_private;
    checked_yaml([&] {
        require(sha256(expected.candidate_sha256) && sha256(expected.bundle_sha256) &&
            sha256(expected.map_sha256) && !expected.observation.empty() &&
            expected.health_identity.starts_with("sha256:") && sha256(expected.health_identity.substr(7)) &&
            (expected.array=="a1100" || expected.array=="a1400" || expected.array=="a2000") &&
            expected.previous_cycle>=0,"invalid expected feedback binding");
        require(feedback["schema"].as<std::string>()=="configured-point-base-feedback-v1" &&
            feedback["status"].as<std::string>()=="ready","feedback producer is not a ready configured POINT publication");
        require(feedback["model_sha256"].as<std::string>()==expected.candidate_sha256,
            "feedback candidate digest differs from its producer");
        require(feedback["input_bundle_sha256"].as<std::string>()==expected.bundle_sha256,
            "feedback source bundle digest differs from its producer");
        require(feedback["map_sha256"].as<std::string>()==expected.map_sha256,
            "feedback source map digest differs from its producer");
        require(feedback["observation"].as<std::string>()==expected.observation &&
            feedback["frozen_health_identity"].as<std::string>()==expected.health_identity &&
            feedback["array"].as<std::string>()==expected.array,
            "feedback observation, frozen health or array differs");
        require(feedback["replacement"].as<bool>() && feedback["gain"].as<double>()==1. &&
            feedback["feedback_domain_radius_arcsec"].as<double>()==40. &&
            feedback["frame"].as<std::string>()=="delta-source-AltAz" &&
            feedback["coordinate_units"].as<std::string>()=="arcsec" &&
            feedback["signal_units"].as<std::string>()=="mJy/nominal beam",
            "feedback replacement, frame or units differ");
        require(bundle["schema"].as<std::string>()=="citlali-forward-map-bundle-v1" &&
            bundle["outcome"].as<std::string>()=="success" &&
            bundle["signal_unit"].as<std::string>()=="mJy/beam",
            "feedback source bundle is not a successful calibrated map");
        require(bundle["observation"].as<std::string>()==expected.observation &&
            bundle["frozen_health_identity"].as<std::string>()==expected.health_identity &&
            bundle["array"].as<std::string>()==expected.array,
            "feedback source map observation, frozen health or array differs");
        require(bundle["cycle_index"].as<int>()==expected.previous_cycle,
            "feedback source map is not the preceding cycle");
        require(bundle["files"]["map.fits"].as<std::string>()==expected.map_sha256,
            "feedback source map digest differs from its bundle");
    });
}

struct PointScienceInterval {
    std::int64_t scan, first, past_last; // half-open native row interval
};

inline std::vector<PointScienceInterval> validate_point_processing_binding(
    const YAML::Node &producer,const YAML::Node &input,const YAML::Node &processing,
    const std::string &requested_sha256,const std::string &actual_sha256,std::size_t native_rows) {
    using namespace point_binding_private;
    return checked_yaml([&] {
        require(sha256(requested_sha256) && requested_sha256==actual_sha256 &&
            producer["processing_scans_sha256"].as<std::string>()==actual_sha256,
            "processing support digest differs from the native producer or request");
        require(native_rows>0 && native_rows<=static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
            "processing native grid has an invalid row count");
        const auto decision=input["decision_apply"];
        const bool fresh=bool(decision["processing_request"]);
        require(fresh!=bool(decision["processing_provenance"]),"processing input must name exactly one generation");
        const auto generation_sha=decision[fresh?"processing_request":"processing_provenance"]["sha256"].as<std::string>();
        require(sha256(generation_sha) && processing["generation"].as<std::string>()==
            std::string(fresh?"native-processing-request:sha256:":"sha256:")+generation_sha &&
            processing["generation_kind"].as<std::string>()==
                (fresh?"fresh-realization-from-explicit-request":"verified-existing-processing-generation"),
            "processing generation differs from the bound input");
        require(processing["timing_inputs"].IsSequence() && decision["timing_inputs"].IsSequence() &&
            processing["timing_inputs"].size()>0 &&
            YAML::Dump(processing["timing_inputs"])==YAML::Dump(decision["timing_inputs"]),
            "processing timing inputs differ from the bound input");
        require(processing["scans"].IsSequence(),"processing science scans are absent or malformed");
        std::vector<PointScienceInterval> result;
        std::int64_t prior_scan=-1,prior_end=0;
        for (const auto &scan:processing["scans"]) {
            const auto id=scan["scan"].as<std::int64_t>();
            require(id>=0 && id>prior_scan,"processing science scan IDs are duplicated or unordered");
            prior_scan=id;
            // A valid projected science scan may have no associated native rows.
            if (!scan["science_native"]) continue;
            require(scan["science_native"].IsSequence(),"processing native intervals are malformed");
            for (const auto &range:scan["science_native"]) {
                require(range.IsSequence() && range.size()==2,"processing native interval is malformed");
                const auto first=range[0].as<std::int64_t>(),last=range[1].as<std::int64_t>();
                require(first>=0 && first<last && last<=static_cast<std::int64_t>(native_rows),
                    "processing science interval lies outside the native grid");
                require(first>=prior_end,"processing native intervals overlap or are unordered");
                result.push_back({id,first,last});prior_end=last;
            }
        }
        require(!result.empty(),"processing has no native science support");
        return result;
    });
}

inline void validate_point_health_decision(const YAML::Node &persisted,
    std::string_view state,std::string_view reason) {
    using namespace point_binding_private;
    checked_yaml([&] {
        require(!state.empty() && !reason.empty() && persisted["state"].as<std::string>()==state &&
            persisted["reason"].as<std::string>()==reason,
            "frozen health state or reason differs from its evidence");
    });
}
} // namespace citlali::cli::detail

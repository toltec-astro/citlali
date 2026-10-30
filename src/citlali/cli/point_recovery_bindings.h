#pragma once
// Private, explicit diagnostic boundary. Ordinary POINT publications are never
// relabelled as injected inputs or seeded feedback.
#include "point_weighted_bindings.h"
#include <cmath>
#include <citlali/core/pipeline/fruit_cycle.h>
namespace citlali::cli::detail {
inline void validate_point_recovery(const YAML::Node &n, const std::string &observation,
    const std::string &config, const std::string &health, int cycle) {
    using namespace point_binding_private;
    checked_yaml([&] {
        require(n.IsMap(),"recovery binding is not a map");
        for(const auto &entry:n) {
            const auto k=entry.first.as<std::string>();
            require(k=="schema" || k=="case" || k=="observation" || k=="config_sha256" ||
                k=="health_sha256" || k=="arrays" || k=="weighted_references", "unknown recovery field");
        }
        require(n["schema"].as<std::string>()=="citlali-point-recovery-v1" &&
            n["observation"].as<std::string>()==observation &&
            n["config_sha256"].as<std::string>()==config &&
            n["health_sha256"].as<std::string>()==health,"recovery lineage differs");
        const auto role=n["case"].as<std::string>();
        require(role=="REF" || role=="CORE+" || role=="CORE-" || role=="SHOULDER" || role=="SEED",
            "unknown recovery case");
        require(cycle>=0 && cycle<=10,"recovery cycle exceeds fixed budget");
        require(role!="SEED" || cycle>0,"seed must reuse REF M0");
        require(n["arrays"].IsMap() && n["arrays"].size()==3,"recovery needs three arrays");
        for(const auto *array:{"a1100","a1400","a2000"}) require(bool(n["arrays"][array]),"missing recovery array");
        if(role=="REF") {
            require(n["weighted_references"].IsMap() && n["weighted_references"].size()==3,
                "REF needs three weighted-reference arrays");
            for(const auto *array:{"a1100","a1400","a2000"})
                require(n["weighted_references"][array].IsSequence() && n["weighted_references"][array].size()==2,
                    "REF needs compact and shoulder references");
        } else require(!n["weighted_references"],"weighted-reference verification belongs to REF");
    });
}
// Shared native routing boundary, exercised by tiny matrix tests. Scientific
// addition/validity remains owned by FRUIT; no extra restoration term is added.
inline void apply_point_recovery(const std::string &role, int cycle,
    pipeline::PtcMatrix &original, pipeline::PtcMatrix &operand,
    const pipeline::PtcMask &eligible, const pipeline::PtcMatrix &addition,
    const pipeline::PtcMask &supported) {
    using namespace point_binding_private;
    require(cycle>=0 && cycle<=10,"recovery cycle outside budget");
    require(role=="REF" || role=="CORE+" || role=="CORE-" || role=="SHOULDER" || role=="SEED",
        "unknown recovery intervention");
    require(role!="SEED" || cycle>0,"seed must reuse REF M0");
    if(role=="SEED" && cycle==1)
        operand=pipeline::fruit_injected_parent(operand,eligible,addition,supported);
    else if(role=="CORE+" || role=="CORE-" || role=="SHOULDER")
        original=pipeline::fruit_injected_parent(original,eligible,addition,supported);
}

}

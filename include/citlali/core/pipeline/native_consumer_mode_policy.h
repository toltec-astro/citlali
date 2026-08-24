#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>

#include <stdexcept>

namespace citlali::pipeline {

enum class NativeConsumerRoute {
    legacy_inactive,
    native_required,
    beammap_raw_apt_producer,
    beammap_calibration_table,
};

struct NativeConsumerModeRequest {
    citlali::config::ReductionType reduction_type =
        citlali::config::ReductionType::science;
    citlali::config::MapGrouping map_grouping =
        citlali::config::MapGrouping::automatic;
    bool matched_v2_relation_available = false;
    bool native_carriers_available = false;
};

// The mode decision is intentionally independent of Engine. Beammap can
// never request matched-consumer lineage: detector/automatic is the raw APT
// producer, while all established non-detector groups retain the calibration
// table lane. Science activates only with one complete matched-v2
// relation/carrier pair; partial authority fails closed. Pointing and OOF may
// admit complete matched-v2 calibration values while retaining their shared
// legacy numerical execution family. Native Pointing/OOF execution remains
// unavailable until those mode-specific product contracts are reconstructed.
inline NativeConsumerRoute resolve_native_consumer_route(
    const NativeConsumerModeRequest &request) {
    if (request.reduction_type ==
        citlali::config::ReductionType::beammap) {
        if (request.matched_v2_relation_available) {
            throw std::logic_error(
                "Beammap cannot request matched-v2 native-consumer lineage");
        }
        return citlali::config::is_detector_map_grouping(
                   request.map_grouping) ||
                citlali::config::is_automatic_map_grouping(
                   request.map_grouping)
            ? NativeConsumerRoute::beammap_raw_apt_producer
            : NativeConsumerRoute::beammap_calibration_table;
    }

    if (request.reduction_type ==
        citlali::config::ReductionType::pointing) {
        if (request.matched_v2_relation_available !=
            request.native_carriers_available) {
            throw std::logic_error(
                "native Pointing activation has partial authority");
        }
        return NativeConsumerRoute::legacy_inactive;
    }

    if (request.reduction_type == citlali::config::ReductionType::oof) {
        if (request.matched_v2_relation_available !=
            request.native_carriers_available) {
            throw std::logic_error(
                "native OOF admission has partial authority");
        }
        return NativeConsumerRoute::legacy_inactive;
    }

    if (request.matched_v2_relation_available !=
        request.native_carriers_available) {
        throw std::logic_error(
            "native Science activation has partial authority");
    }
    return request.matched_v2_relation_available
        ? NativeConsumerRoute::native_required
        : NativeConsumerRoute::legacy_inactive;
}

constexpr bool native_consumer_lineage_required(
    NativeConsumerRoute route) noexcept {
    return route == NativeConsumerRoute::native_required;
}

}  // namespace citlali::pipeline

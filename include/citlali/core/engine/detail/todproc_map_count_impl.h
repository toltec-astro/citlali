#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/map_group_indexing.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_map_num() {
    auto &mapmaking_config = citlali::pipeline::mapmaking_config(engine());
    const auto reduction_type =
        citlali::pipeline::runtime_config(engine()).reduction_type;
    const auto requested_grouping = mapmaking_config.grouping;
    mapmaking_config.grouping =
        citlali::pipeline::effective_map_grouping_for_reduction(
            reduction_type, requested_grouping);

    if (citlali::pipeline::detector_map_grouping_disallowed(
            reduction_type, requested_grouping)) {
        logger->warn("mapmaking.grouping=detector is only supported for beammap; defaulting to array for {}",
                     citlali::config::to_string(reduction_type));
    }

    const auto map_grouping_name =
        citlali::pipeline::active_map_grouping_name(engine());
    engine().omb.map_grouping = map_grouping_name;
    engine().cmb.map_grouping = map_grouping_name;
    engine().rtcproc.kernel.map_grouping = map_grouping_name;

    const auto n_maps = citlali::pipeline::apply_polarization_map_count(
        citlali::pipeline::base_map_count_for_grouping(
            mapmaking_config.grouping, engine().calib),
        engine().rtcproc.run_polarization, engine().rtcproc.polarization);

    const auto array_indices =
        citlali::pipeline::map_array_indices_for_grouping(
            mapmaking_config.grouping, engine().calib,
            engine().toltec_io.nw_to_array_map);
    engine().map_indices = citlali::pipeline::make_map_index_state(
        array_indices, n_maps, engine().rtcproc.polarization);
}

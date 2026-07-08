#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/map_group_indexing.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_map_num() {
    auto &mapmaking_config = engine().typed_config.mapmaking;
    const auto reduction_type = engine().typed_config.runtime.reduction_type;
    const auto requested_grouping = mapmaking_config.grouping;
    mapmaking_config.grouping =
        citlali::pipeline::effective_map_grouping_for_reduction(
            reduction_type, requested_grouping);

    if (citlali::pipeline::detector_map_grouping_disallowed(
            reduction_type, requested_grouping)) {
        logger->warn("mapmaking.grouping=detector is only supported for beammap; defaulting to array for {}",
                     citlali::config::to_string(reduction_type));
    }

    engine().map_grouping = std::string{citlali::config::to_string(
        mapmaking_config.grouping)};
    engine().omb.map_grouping = engine().map_grouping;
    engine().cmb.map_grouping = engine().map_grouping;
    engine().rtcproc.kernel.map_grouping = engine().map_grouping;

    engine().n_maps = citlali::pipeline::apply_polarization_map_count(
        citlali::pipeline::base_map_count_for_grouping(
            mapmaking_config.grouping, engine().calib),
        engine().rtcproc.run_polarization, engine().rtcproc.polarization);

    const auto array_indices =
        citlali::pipeline::map_array_indices_for_grouping(
            mapmaking_config.grouping, engine().calib,
            engine().toltec_io.nw_to_array_map);
    citlali::pipeline::populate_map_index_mappings(
        array_indices, engine().n_maps, engine().rtcproc.polarization,
        engine().maps_to_arrays, engine().maps_to_stokes,
        engine().arrays_to_maps);
}

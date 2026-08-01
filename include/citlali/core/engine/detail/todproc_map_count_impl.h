#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/map_group_indexing.h>
#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::calc_map_num() {
    const auto &mapmaking_config =
        citlali::pipeline::mapmaking_config(engine());
    const auto &mapmaking_plan =
        citlali::pipeline::mapmaking_plan(engine());
    const auto reduction_type =
        citlali::pipeline::runtime_reduction_type(engine());
    const auto requested_grouping = mapmaking_plan.requested.grouping;

    if (citlali::pipeline::detector_map_grouping_disallowed(
            reduction_type, requested_grouping)) {
        logger->warn("mapmaking.grouping=detector is only supported for beammap; defaulting to array for {}",
                     citlali::config::to_string(reduction_type));
    }

    const auto map_grouping_name =
        citlali::pipeline::active_map_grouping_name(engine());
    engine().omb.map_grouping = map_grouping_name;
    if (!citlali::pipeline::coadd_outputs_enabled(engine()) ||
        !citlali::pipeline::science_map_v1_profile_available(engine())) {
        // Profiles outside SCI-MAP-001 v1 retain the pre-repair metadata
        // lifecycle. The v1 coadd receives this fact only on atomic admission.
        engine().cmb.map_grouping = map_grouping_name;
    }
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

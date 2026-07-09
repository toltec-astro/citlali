#pragma once

#include <citlali/core/engine/learning.h>
#include <citlali/core/pipeline/interface_sync_state.h>
#include <citlali/core/pipeline/fruit_loop_iteration_state.h>
#include <citlali/core/pipeline/logging_state.h>
#include <citlali/core/pipeline/map_index_state.h>
#include <citlali/core/pipeline/observation_identity_state.h>
#include <citlali/core/pipeline/observation_date_state.h>
#include <citlali/core/pipeline/pointing_offset_state.h>
#include <citlali/core/pipeline/reduction_config_state.h>
#include <citlali/core/pipeline/reduction_output_state.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

struct EngineRuntimeState : public citlali::pipeline::LoggingState,
                            public citlali::pipeline::ReductionConfigState,
                            public citlali::pipeline::ReductionOutputState {
    // TOD alignment products and timing-gap masks
    citlali::pipeline::TimestreamAlignmentState alignment;

    // observation date metadata for each input observation
    citlali::pipeline::ObservationDateState observation_dates;

    // manual interface timing offsets for networks and HWPR
    citlali::pipeline::InterfaceSyncState interface_sync;

    // active observation identity
    citlali::pipeline::ObservationIdentityState observation_identity;

    // map count and per-map index translations
    citlali::pipeline::MapIndexState map_indices;

    // current fruit-loop iteration counter
    citlali::pipeline::FruitLoopRuntimeState iteration;

    // shared state learned across RTC, PTC, and mapmaking phases
    ReductionLearningState learning;

    // manual pointing offsets and optional MJD interpolation anchors
    citlali::pipeline::PointingOffsetState pointing_offsets;

};

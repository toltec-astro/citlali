#pragma once

#include <citlali/core/pipeline/interface_sync_state.h>
#include <citlali/core/pipeline/observation_date_state.h>
#include <citlali/core/pipeline/observation_identity_state.h>
#include <citlali/core/pipeline/pointing_offset_state.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

namespace citlali::pipeline {

struct ObservationRuntimeState {
    // TOD alignment products and timing-gap masks
    TimestreamAlignmentState alignment;

    // observation date metadata for each input observation
    ObservationDateState observation_dates;

    // manual interface timing offsets for networks and HWPR
    InterfaceSyncState interface_sync;

    // active observation identity
    ObservationIdentityState observation_identity;

    // manual pointing offsets and optional MJD interpolation anchors
    PointingOffsetState pointing_offsets;
};

}  // namespace citlali::pipeline

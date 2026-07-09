#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <CCfits/CCfits>
#include <Eigen/Core>
#include <spdlog/spdlog.h>

#include <citlali/core/config/reduction_config.h>
#include <citlali/core/engine/learning.h>
#include <citlali/core/pipeline/interface_sync_state.h>
#include <citlali/core/pipeline/map_fits_output_state.h>
#include <citlali/core/pipeline/map_index_state.h>
#include <citlali/core/pipeline/observation_date_state.h>
#include <citlali/core/pipeline/output_path_state.h>
#include <citlali/core/pipeline/pointing_offset_state.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>
#include <citlali/core/pipeline/tod_output_state.h>
#include <citlali/core/utils/fits_io.h>

struct EngineRuntimeState {
    // type for missing/invalid keys
    using key_vec_t = std::vector<std::vector<std::string>>;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // TOD alignment products and timing-gap masks
    citlali::pipeline::TimestreamAlignmentState alignment;

    // observation date metadata for each input observation
    citlali::pipeline::ObservationDateState observation_dates;

    // reduction, observation, coadd, and timestream output paths
    citlali::pipeline::OutputPathState output_paths;

    // vectors to hold missing/invalid keys
    key_vec_t missing_keys, invalid_keys;

    // manual interface timing offsets for networks and HWPR
    citlali::pipeline::InterfaceSyncState interface_sync;

    // typed config mirror for staged config migration
    citlali::config::ReductionConfig typed_config;

    // obsnum
    std::string obsnum;

    // per-stream TOD output row maps
    citlali::pipeline::TodOutputState tod_outputs;

    // map count and per-map index translations
    citlali::pipeline::MapIndexState map_indices;

    // current fruit loops iteration
    int fruit_iter;

    // shared state learned across RTC, PTC, and mapmaking phases
    ReductionLearningState reduction_learning;

    // manual pointing offsets and optional MJD interpolation anchors
    citlali::pipeline::PointingOffsetState pointing_offsets;

    using map_fits_io_t =
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>>;

    // observation, coadd, filtered, and noise FITS output handles
    citlali::pipeline::MapFitsOutputState<map_fits_io_t::value_type>
        map_fits_outputs;
};

using EngineRunState = EngineRuntimeState;

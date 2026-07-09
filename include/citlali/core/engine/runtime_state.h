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
#include <citlali/core/pipeline/map_index_state.h>
#include <citlali/core/pipeline/output_path_state.h>
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

    // date/time of each obs
    std::vector<std::string> date_obs;

    // reduction, observation, coadd, and timestream output paths
    citlali::pipeline::OutputPathState output_paths;

    // vectors to hold missing/invalid keys
    key_vec_t missing_keys, invalid_keys;

    // manual offsets for nws and hwp
    std::map<std::string, double> interface_sync_offset;

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

    // manual pointing offsets
    std::map<std::string, Eigen::VectorXd> pointing_offsets_arcsec;
    // modified julian dates of pointing offsets
    Eigen::ArrayXd pointing_offsets_modified_julian_date;

    using map_fits_io_t =
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>>;

    // map output files
    map_fits_io_t fits_io_vec, noise_fits_io_vec;
    map_fits_io_t filtered_fits_io_vec, filtered_noise_fits_io_vec;

    // coadded map output files
    map_fits_io_t coadd_fits_io_vec, coadd_noise_fits_io_vec;
    map_fits_io_t filtered_coadd_fits_io_vec, filtered_coadd_noise_fits_io_vec;
};

using EngineRunState = EngineRuntimeState;

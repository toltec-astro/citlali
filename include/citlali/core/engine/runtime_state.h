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
#include <citlali/core/utils/fits_io.h>

struct EngineRuntimeState {
    // type for missing/invalid keys
    using key_vec_t = std::vector<std::vector<std::string>>;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // for timing
    Eigen::VectorXd t_common;
    std::vector<Eigen::VectorXi> masks;
    std::map<Eigen::Index, Eigen::VectorXi> nw_masks;
    std::vector<Eigen::VectorXd> nw_times;

    // date/time of each obs
    std::vector<std::string> date_obs;

    // add extra output for debugging
    bool verbose_mode;

    // time gaps
    std::map<std::string, int> gaps;

    // output directory and optional sub directory name
    std::string output_dir, redu_dir_name;

    // expected sky regime for map interpretation
    std::string map_regime = "unknown";

    // reduction directory number
    int redu_dir_num;

    // obsnum and coadded directory names
    std::string obsnum_dir_name, coadd_dir_name;

    // tod output file name
    std::map<std::string, std::string> tod_filename;

    // vectors to hold missing/invalid keys
    key_vec_t missing_keys, invalid_keys;

    // number of threads
    int n_threads;

    // parallel execution policy
    std::string parallel_policy;

    // number of scans completed
    int n_scans_done;

    // manual offsets for nws and hwp
    std::map<std::string, double> interface_sync_offset;

    // vectors for tod alignment offsets
    std::vector<Eigen::Index> start_indices, end_indices;

    // indices for hwpr alignment offsets
    Eigen::Index hwpr_start_indices, hwpr_end_indices;

    // typed config mirror for staged config migration
    citlali::config::ReductionConfig typed_config;

    // obsnum
    std::string obsnum;

    // write filtered maps as they complete
    bool write_filtered_maps_partial;

    std::string rtcdiag_filename;
    std::string ptcdiag_filename;

    // per-stream TOD output row maps
    Eigen::VectorXI tod_scan_to_output_scan_rtc;
    Eigen::VectorXI tod_scan_to_output_scan_ptc;
    Eigen::Index n_tod_output_scans_rtc = 0;
    Eigen::Index n_tod_output_scans_ptc = 0;

    // map grouping string passed to legacy timestream/mapmaking APIs
    std::string map_grouping;

    // number of maps
    int n_maps;

    // mapping from index in map vector to array index
    Eigen::VectorXI maps_to_arrays, arrays_to_maps;

    // mapping from index in map vector to array index
    Eigen::VectorXI maps_to_stokes;

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

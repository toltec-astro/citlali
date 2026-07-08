#pragma once

// Beammap detector-specific TOD output helpers.

#include <citlali/core/config/config_value.h>
#include <citlali/core/config/runtime_config.h>

#include <Eigen/Core>

#include <algorithm>
#include <filesystem>
#include <string>

namespace beammap_detector_tod_output_helpers {

struct OutputCounts {
    int n_uniform;
    int n_dense;
    Eigen::Index n_slots;
};

inline OutputCounts output_counts(int requested_uniform,
                                  int requested_dense) {
    OutputCounts counts{
        std::max(0, requested_uniform),
        std::max(0, requested_dense),
        0};
    counts.n_slots =
        static_cast<Eigen::Index>(counts.n_uniform + counts.n_dense);
    return counts;
}

template <class Ptcs>
Eigen::Index max_ptc_samples(const Ptcs &ptcs) {
    Eigen::Index n_samples_max = 0;
    for (const auto &ptc : ptcs) {
        n_samples_max =
            std::max<Eigen::Index>(n_samples_max, ptc.scans.data.rows());
    }
    return n_samples_max;
}

struct OutputPaths {
    std::string dir_name;
    std::string filename;
};

inline OutputPaths output_paths(const std::string &obsnum_dir_name,
                                const std::string &subdir_name,
                                bool sim_obs,
                                citlali::config::ReductionType reduction_type,
                                const std::string &obsnum) {
    namespace fs = std::filesystem;
    const std::string reduction_type_name{
        citlali::config::to_string(reduction_type)};
    OutputPaths paths{obsnum_dir_name + "raw/", ""};
    if (citlali::config::has_config_value(subdir_name)) {
        paths.dir_name += subdir_name + "/";
    }
    fs::create_directories(paths.dir_name);
    paths.filename = paths.dir_name + "toltec";
    paths.filename += sim_obs ? "_simu" : "_commissioning";
    paths.filename += "_" + reduction_type_name + "_" + obsnum +
                      "_ptc_detector_tod.nc";
    return paths;
}

} // namespace beammap_detector_tod_output_helpers

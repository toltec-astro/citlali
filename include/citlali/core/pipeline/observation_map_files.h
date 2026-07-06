#pragma once

#include <string>
#include <utility>

#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class RawFitsFiles, class NoiseFitsFiles, class FilteredFitsFiles,
          class FilteredNoiseFitsFiles>
void clear_observation_map_fits_files(RawFitsFiles &raw_fits_files,
                                      NoiseFitsFiles &noise_fits_files,
                                      FilteredFitsFiles &filtered_fits_files,
                                      FilteredNoiseFitsFiles
                                          &filtered_noise_fits_files) {
    raw_fits_files.clear();
    noise_fits_files.clear();
    filtered_fits_files.clear();
    filtered_noise_fits_files.clear();
}

inline std::string raw_observation_map_directory(
    const std::string &obsnum_dir_name) {
    return obsnum_dir_name + "raw/";
}

inline std::string filtered_observation_map_directory(
    const std::string &obsnum_dir_name) {
    return obsnum_dir_name + "filtered/";
}

template <class Engine>
bool should_create_observation_per_obs_outputs(const Engine &engine) {
    return !coadd_outputs_enabled(engine);
}

template <class Engine>
bool should_create_observation_noise_maps(const Engine &engine) {
    return noise_maps_enabled(engine) &&
           noise_realization_outputs_enabled(engine);
}

template <class Engine>
bool should_create_observation_filtered_maps(const Engine &engine) {
    return map_filter_outputs_enabled(engine);
}

template <class Engine>
bool should_create_observation_filtered_noise_maps(const Engine &engine) {
    return should_create_observation_noise_maps(engine);
}

template <class FitsFiles, class MakeFits>
void append_observation_map_fits_file(FitsFiles &fits_files,
                                      const std::string &filename,
                                      MakeFits &&make_fits) {
    fits_files.push_back(make_fits(filename));
}

}  // namespace citlali::pipeline

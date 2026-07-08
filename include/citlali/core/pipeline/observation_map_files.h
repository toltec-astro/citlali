#pragma once

#include <string>
#include <utility>

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/utils/toltec_io.h>

#include <Eigen/Core>

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

template <class RawFitsFiles, class NoiseFitsFiles, class FilteredFitsFiles,
          class FilteredNoiseFitsFiles>
void reset_coadd_map_fits_files(RawFitsFiles &raw_fits_files,
                                NoiseFitsFiles &noise_fits_files,
                                FilteredFitsFiles &filtered_fits_files,
                                FilteredNoiseFitsFiles
                                    &filtered_noise_fits_files) {
    RawFitsFiles().swap(raw_fits_files);
    NoiseFitsFiles().swap(noise_fits_files);
    FilteredFitsFiles().swap(filtered_fits_files);
    FilteredNoiseFitsFiles().swap(filtered_noise_fits_files);
}

inline std::string raw_observation_map_directory(
    const std::string &obsnum_dir_name) {
    return obsnum_dir_name + "raw/";
}

inline std::string filtered_observation_map_directory(
    const std::string &obsnum_dir_name) {
    return obsnum_dir_name + "filtered/";
}

inline std::string raw_coadd_map_directory(
    const std::string &coadd_dir_name) {
    return coadd_dir_name + "raw/";
}

inline std::string filtered_coadd_map_directory(
    const std::string &coadd_dir_name) {
    return coadd_dir_name + "filtered/";
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

template <auto DataType, auto ProductType, auto FilterType, class ToltecIo>
std::string observation_output_filename(
    ToltecIo &toltec_io, const std::string &dir_name,
    citlali::config::ReductionType reduction_type,
    const std::string &array_name, const std::string &obsnum, bool sim_obs) {
    const std::string reduction_type_name{
        citlali::config::to_string(reduction_type)};
    return toltec_io.template create_filename<DataType, ProductType,
                                              FilterType>(
        dir_name, reduction_type_name, array_name, obsnum, sim_obs);
}

template <auto DataType, auto ProductType, auto FilterType, class ToltecIo>
std::string coadd_output_filename(
    ToltecIo &toltec_io, const std::string &dir_name,
    const std::string &array_name, bool sim_obs) {
    return toltec_io.template create_filename<DataType, ProductType,
                                              FilterType>(
        dir_name, "", array_name, "", sim_obs);
}

template <auto DataType, auto ProductType, auto FilterType, class FitsFiles,
          class ToltecIo>
void append_coadd_map_fits_file(FitsFiles &fits_files, ToltecIo &toltec_io,
                                const std::string &dir_name,
                                const std::string &array_name,
                                bool sim_obs) {
    fits_files.emplace_back(
        coadd_output_filename<DataType, ProductType, FilterType>(
            toltec_io, dir_name, array_name, sim_obs));
}

template <auto FilterType, class MapFitsFiles, class NoiseFitsFiles,
          class ToltecIo, class Arrays, class ArrayNameMap>
void append_coadd_array_products(MapFitsFiles &map_fits_files,
                                 NoiseFitsFiles &noise_fits_files,
                                 ToltecIo &toltec_io,
                                 const std::string &dir_name,
                                 const Arrays &arrays,
                                 Eigen::Index n_arrays,
                                 ArrayNameMap &array_name_map, bool sim_obs,
                                 bool write_noise_maps) {
    for (Eigen::Index i = 0; i < n_arrays; ++i) {
        const auto array = arrays[i];
        const std::string array_name = array_name_map[array];
        append_coadd_map_fits_file<engine_utils::toltecIO::toltec,
                                   engine_utils::toltecIO::map, FilterType>(
            map_fits_files, toltec_io, dir_name, array_name, sim_obs);
        if (write_noise_maps) {
            append_coadd_map_fits_file<engine_utils::toltecIO::toltec,
                                       engine_utils::toltecIO::noise,
                                       FilterType>(
                noise_fits_files, toltec_io, dir_name, array_name, sim_obs);
        }
    }
}

}  // namespace citlali::pipeline

#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/observation_coadd_accumulation.h>
#include <citlali/core/pipeline/observation_map_files.h>
#include <citlali/core/pipeline/output_policy.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::coadd() {
    citlali::pipeline::accumulate_observation_into_coadd(
        engine().cmb, engine().omb, engine().n_maps,
        engine().rtcproc.run_kernel);
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::create_coadded_map_files() {
    // clear fits_io vectors
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().coadd_fits_io_vec);
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().coadd_noise_fits_io_vec);
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().filtered_coadd_fits_io_vec);
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().filtered_coadd_noise_fits_io_vec);

    const bool write_noise_maps =
        citlali::pipeline::noise_maps_enabled(engine()) &&
        citlali::pipeline::noise_realization_outputs_enabled(engine());
    const std::string raw_dir = engine().coadd_dir_name + "raw/";
    const std::string filtered_dir = engine().coadd_dir_name + "filtered/";

    // loop through arrays
    for (Eigen::Index i=0; i<engine().calib.n_arrays; ++i) {
        // array index
        auto array = engine().calib.arrays[i];
        // array name
        std::string array_name = engine().toltec_io.array_name_map[array];
        // map filename
        citlali::pipeline::append_coadd_map_fits_file<
            engine_utils::toltecIO::toltec, engine_utils::toltecIO::map,
            engine_utils::toltecIO::raw>(
            engine().coadd_fits_io_vec, engine().toltec_io, raw_dir,
            array_name, engine().telescope.sim_obs);

        // if noise maps requested
        if (write_noise_maps) {
            // noise map filename
            citlali::pipeline::append_coadd_map_fits_file<
                engine_utils::toltecIO::toltec, engine_utils::toltecIO::noise,
                engine_utils::toltecIO::raw>(
                engine().coadd_noise_fits_io_vec, engine().toltec_io,
                raw_dir, array_name, engine().telescope.sim_obs);
        }
    }

    // if map filtering are requested
    if (citlali::pipeline::map_filter_outputs_enabled(engine())) {
        // loop through arrays
        for (Eigen::Index i=0; i<engine().calib.n_arrays; ++i) {
            // array index
            auto array = engine().calib.arrays[i];
            // array name
            std::string array_name = engine().toltec_io.array_name_map[array];
            // filtered map filename
            citlali::pipeline::append_coadd_map_fits_file<
                engine_utils::toltecIO::toltec, engine_utils::toltecIO::map,
                engine_utils::toltecIO::filtered>(
                engine().filtered_coadd_fits_io_vec, engine().toltec_io,
                filtered_dir, array_name, engine().telescope.sim_obs);

            // if noise maps requested
            if (write_noise_maps) {
                // filtered noise map filename
                citlali::pipeline::append_coadd_map_fits_file<
                    engine_utils::toltecIO::toltec,
                    engine_utils::toltecIO::noise,
                    engine_utils::toltecIO::filtered>(
                    engine().filtered_coadd_noise_fits_io_vec,
                    engine().toltec_io, filtered_dir, array_name,
                    engine().telescope.sim_obs);
            }
        }
    }
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::make_index_file(std::string filepath) {
    // get sortedfiles and directories in filepath
    std::set<fs::path> sorted_by_name;
    for (auto &entry : fs::directory_iterator(filepath))
        sorted_by_name.insert(entry);

    // yaml node to store names
    YAML::Node node;
    // data products
    node["description"].push_back("citlali data products");
    // datetime when file is created
    node["date"].push_back(engine_utils::current_date_time());
    // citlali version
    node["citlali_version"].push_back(CITLALI_GIT_VERSION);
    // kids version
    node["kids_version"].push_back(KIDSCPP_GIT_VERSION);
    // tula version
    node["tula_version"].push_back(TULA_GIT_VERSION);

    // call make_index_file recursively if current object is directory
    for (const auto & entry : sorted_by_name) {
        std::string path_string{entry.generic_string()};
        if (fs::is_directory(entry)) {
            make_index_file(path_string);
        }
        node["files/dirs"].push_back(path_string.substr(path_string.find_last_of("/") + 1));
    }
    // output yaml index file
    std::ofstream fout(filepath + "/index.yaml");
    fout << node;
}

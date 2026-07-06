#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/output_policy.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::coadd() {
    // calculate the offset between cmb and omb
    int delta_row = 0.5*(engine().cmb.n_rows - engine().omb.n_rows);
    int delta_col= 0.5*(engine().cmb.n_cols - engine().omb.n_cols);

    // loop through the maps
    for (Eigen::Index i=0; i<engine().n_maps; ++i) {
        // define common block references
        auto cmb_weight_block = engine().cmb.weight.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols);
        auto cmb_signal_block = engine().cmb.signal.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols);

        // update cmb.weight with omb.weight
        cmb_weight_block += engine().omb.weight.at(i);

        // update cmb.signal with omb.signal * omb.weight
        cmb_signal_block += (engine().omb.signal.at(i).array() * engine().omb.weight.at(i).array()).matrix();

        // update cmb.kernel with omb.kernel * omb.weight
        if (engine().rtcproc.run_kernel) {
            auto cmb_kernel_block = engine().cmb.kernel.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols);
            cmb_kernel_block += (engine().omb.kernel.at(i).array() * engine().omb.weight.at(i).array()).matrix();
        }

        // update coverage
        if (!engine().cmb.coverage.empty()) {
            auto cmb_coverage_block = engine().cmb.coverage.at(i).block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols);
            cmb_coverage_block += engine().omb.coverage.at(i);
        }

        if (!engine().cmb.noise.empty() && !engine().omb.noise.empty()) {
            for (Eigen::Index n = 0; n < engine().cmb.n_noise; ++n) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> cmb_noise_matrix(
                    engine().cmb.noise.at(i).data() + n * engine().cmb.n_rows * engine().cmb.n_cols,
                    engine().cmb.n_rows, engine().cmb.n_cols);
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> omb_noise_matrix(
                    engine().omb.noise.at(i).data() + n * engine().omb.n_rows * engine().omb.n_cols,
                    engine().omb.n_rows, engine().omb.n_cols);
                auto cmb_noise_block = cmb_noise_matrix.block(delta_row, delta_col, engine().omb.n_rows, engine().omb.n_cols);
                cmb_noise_block += (omb_noise_matrix.array() * engine().omb.weight.at(i).array()).matrix();
            }
        }
    }
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::create_coadded_map_files() {
    // clear fits_io vectors
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().coadd_fits_io_vec);
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().coadd_noise_fits_io_vec);
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().filtered_coadd_fits_io_vec);
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(engine().filtered_coadd_noise_fits_io_vec);

    // loop through arrays
    for (Eigen::Index i=0; i<engine().calib.n_arrays; ++i) {
        // array index
        auto array = engine().calib.arrays[i];
        // array name
        std::string array_name = engine().toltec_io.array_name_map[array];
        // map filename
        auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::map,
                                                                    engine_utils::toltecIO::raw>(engine().coadd_dir_name + "raw/",
                                                                                                 "", array_name, "",
                                                                                                 engine().telescope.sim_obs);
        // create fits_io class for current array file
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
        // append to fits_io vector
        engine().coadd_fits_io_vec.push_back(std::move(fits_io));

        // if noise maps requested
        if (citlali::pipeline::noise_maps_enabled(engine()) &&
            citlali::pipeline::noise_realization_outputs_enabled(engine())) {
            // noise map filename
            auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::noise,
                                                                        engine_utils::toltecIO::raw>(engine().coadd_dir_name + "raw/",
                                                                                                     "", array_name, "",
                                                                                                     engine().telescope.sim_obs);
            // create fits_io class for current array file
            fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
            // append to fits_io vector
            engine().coadd_noise_fits_io_vec.push_back(std::move(fits_io));
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
            auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::map,
                                                                        engine_utils::toltecIO::filtered>(engine().coadd_dir_name +
                                                                                                          "filtered/","", array_name,
                                                                                                          "", engine().telescope.sim_obs);
            // create fits_io class for current array file
            fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
            // append to fits_io vector
            engine().filtered_coadd_fits_io_vec.push_back(std::move(fits_io));

            // if noise maps requested
            if (citlali::pipeline::noise_maps_enabled(engine()) &&
                citlali::pipeline::noise_realization_outputs_enabled(engine())) {
                // filtered noise map filename
                auto filename = engine().toltec_io.template create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::noise,
                                                                            engine_utils::toltecIO::filtered>(engine().coadd_dir_name +
                                                                                                              "filtered/","", array_name,
                                                                                                              "", engine().telescope.sim_obs);
                // create fits_io class for current array file
                fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);
                // append to fits_io vector
                engine().filtered_coadd_noise_fits_io_vec.push_back(std::move(fits_io));
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

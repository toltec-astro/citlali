#pragma once

// Implementation detail included by pointing.h.

#include <citlali/core/pipeline/output_policy.h>

template <mapmaking::MapType map_type>
void Pointing::output() {
    const std::string reduction_type_name{
        citlali::config::to_string(typed_config.runtime.reduction_type)};
    // pointer to map buffer
    mapmaking::MapBuffer* mb = nullptr;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = nullptr;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = nullptr;

    // directory name
    std::string dir_name;

    // matrix to hold pointing fit values and errors (n_params + 2 for array and S/N)
    Eigen::MatrixXf ppt_table(n_maps, 2 * map_fitter.n_params + 2);

    // determine pointers and directory name based on map_type
    if constexpr (map_type == mapmaking::RawObs || map_type == mapmaking::FilteredObs) {
        mb = &omb;
        dir_name = obsnum_dir_name + (map_type == mapmaking::RawObs ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawObs) ? &fits_io_vec : &filtered_fits_io_vec;
        n_io = (map_type == mapmaking::RawObs) ? &noise_fits_io_vec : &filtered_noise_fits_io_vec;

        // filename for ppt table
        auto ppt_filename = toltec_io.create_filename<engine_utils::toltecIO::ppt, engine_utils::toltecIO::map,
                                                      (map_type == mapmaking::RawObs ? engine_utils::toltecIO::raw : engine_utils::toltecIO::filtered)>
                            (dir_name, reduction_type_name, "", obsnum,
                             telescope.sim_obs);

        // add array and S/N to ppt
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            ppt_table(i, 0) = maps_to_arrays(i);
            double map_std_dev = engine_utils::calc_std_dev(mb->signal[i]);
            ppt_table(i, 2 * map_fitter.n_params + 1) = params(i, 0) / map_std_dev;
        }

        Eigen::Index j = 0;
        // populate ppt with fitted parameters and errors
        for (Eigen::Index i = 1; i < 2 * map_fitter.n_params; i += 2) {
            ppt_table.col(i) = params.col(j).cast<float>();
            ppt_table.col(i + 1) = perrors.col(j).cast<float>();
            j++;
        }

        // write ppt
        to_ecsv_from_matrix(ppt_filename, ppt_table, ppt_header, ppt_meta);

        if constexpr (map_type == mapmaking::RawObs) {
            // write stats file
            write_stats();
            if (run_tod_output && !tod_filename.empty()) {
                // add tod header information
                add_tod_header(mb);
            }
        }
    } else if constexpr (map_type == mapmaking::RawCoadd || map_type == mapmaking::FilteredCoadd) {
        mb = &cmb;
        dir_name = coadd_dir_name + (map_type == mapmaking::RawCoadd ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawCoadd) ? &coadd_fits_io_vec : &filtered_coadd_fits_io_vec;
        n_io = (map_type == mapmaking::RawCoadd) ? &coadd_noise_fits_io_vec : &filtered_coadd_noise_fits_io_vec;
    }

    if (citlali::pipeline::mapmaking_outputs_enabled(*this)) {
        if (!f_io->empty()) {
            {
                // progress bar
                tula::logging::progressbar pb(
                    [&](const auto &msg) { logger->info("{}", msg); }, 100, "output progress ");

                for (Eigen::Index i=0; i<f_io->size(); i++) {
                    // add primary hdu
                    add_phdu(f_io, mb, i);

                    if (!mb->noise.empty() && !n_io->empty()) {
                        add_phdu(n_io, mb, i);
                    }
                }

                Eigen::Index k = 0;

                for (Eigen::Index i=0; i<n_maps; i++) {
                    // update progress bar
                    pb.count(n_maps, 1);
                    write_maps(f_io,n_io,mb,i);

                    Eigen::Index map_index = arrays_to_maps(i);

                    // check if we move from one file to the next
                    // if so go back to first hdu layer
                    if (i>0) {
                        if (map_index > arrays_to_maps(i-1)) {
                            k = 0;
                        }
                    }
                    // get current hdu extension name
                    std::string extname = f_io->at(map_index).hdus.at(k)->name();
                    // see if this is a signal extension
                    std::size_t found = extname.find("signal");

                    // find next signal extension
                    while (found==std::string::npos && k<f_io->at(map_index).hdus.size()) {
                        k = k + 1;
                        // get current hdu extension name
                        extname = f_io->at(map_index).hdus.at(k)->name();
                        // see if this is a signal extension
                        found = extname.find("signal");
                    }

                    // add ppt table
                    for (Eigen::Index j = 0; j < ppt_header.size(); ++j) {
                        const auto& key = ppt_header[j];
                        try {
                            f_io->at(map_index).hdus.at(k)->addKey("POINTING." + key, ppt_table(i, j), key + " (" + ppt_header_units[key] + ")");
                        } catch (...) {
                            f_io->at(map_index).hdus.at(k)->addKey("POINTING." + key, 0, key + " (" + ppt_header_units[key] + ")");
                        }
                    }
                    try {
                        f_io->at(map_index).hdus.at(k)->addKey("POINTING.fit_enabled",
                                                               static_cast<int>(typed_config.pointing.fit_gaussian),
                                                               "Gaussian fit enabled");
                    } catch (...) {
                        f_io->at(map_index).hdus.at(k)->addKey("POINTING.fit_enabled", 0,
                                                               "Gaussian fit enabled");
                    }
                    try {
                        f_io->at(map_index).hdus.at(k)->addKey("POINTING.fit_valid",
                                                               static_cast<int>(fit_valid(i)),
                                                               "Gaussian fit valid");
                    } catch (...) {
                        f_io->at(map_index).hdus.at(k)->addKey("POINTING.fit_valid", 0,
                                                               "Gaussian fit valid");
                    }
                    try {
                        f_io->at(map_index).hdus.at(k)->addKey("POINTING.source_strategy",
                                                               std::string(citlali::config::to_string(
                                                                   typed_config.pointing.source_strategy)),
                                                               "Pointing source strategy");
                    } catch (...) {}
                    try {
                        f_io->at(map_index).hdus.at(k)->addKey("POINTING.source_center_mode",
                                                               std::string(citlali::config::to_string(
                                                                   typed_config.pointing.fruitloops_center_mode)),
                                                               "Fruit loops source center mode");
                    } catch (...) {}
                    ++k; // Move to next extension
                }
            }

            logger->info("maps have been written to:");
            for (const auto& file: *f_io) {
                logger->info("{}.fits", file.filepath);
            }
        }

        // clear fits file vectors to ensure its closed.
        f_io->clear();
        n_io->clear();

        // write psd and histogram files
        logger->debug("writing psds");
        write_psd<map_type>(mb, dir_name);
        logger->debug("writing histograms");
        write_hist<map_type>(mb, dir_name);
        logger->debug("writing map diagnostics");
        write_mapdiag<map_type>(mb, dir_name);

        // write source table
        if (citlali::pipeline::source_finding_outputs_enabled(*this)) {
            logger->debug("writing source table");
            write_sources<map_type>(mb, dir_name);
        }
    }
}

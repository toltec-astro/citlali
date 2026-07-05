#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

template <mapmaking::MapType map_type>
void Beammap::write_beammap_map_products(
    mapmaking::MapBuffer *mb,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
    const std::string &dir_name) {
    if (!run_mapmaking) {
        return;
    }

    namespace fs = std::filesystem;

    bool split_by_flag_mode = false;
    if constexpr (map_type == mapmaking::RawObs) {
        split_by_flag_mode = (map_grouping == "detector") && beammap_split_fits_by_flag;
        if (split_by_flag_mode && beammap_split_flag_values.empty()) {
            logger->warn("beammap.split_fits_by_flag enabled but no flag_values specified; using standard map output");
            split_by_flag_mode = false;
        }
    }

    // wiener filtered maps write before this and are deleted from the vector.
    if (!f_io->empty()) {
        auto write_standard_maps = [&]() {
            // progress bar
            tula::logging::progressbar pb(
                [&](const auto &msg) { logger->info("{}", msg); }, 100, "output progress ");

            for (Eigen::Index i=0; i<f_io->size(); ++i) {
                // get the array for the given map
                // add primary hdu
                logger->debug("adding primary header to file {}",i);
                add_phdu(f_io, mb, i);

                if (!mb->noise.empty() && !n_io->empty()) {
                    logger->debug("adding primary header to noise file {}",i);
                    add_phdu(n_io, mb, i);
                }
            }

            logger->debug("done adding primary headers");

            // write the maps
            Eigen::Index k = 0;
            Eigen::Index step = 2;

            if (!mb->kernel.empty()) {
                step++;
                timestream::log_kernel_map_diag(
                    logger,
                    "map output " + dir_name + " before write",
                    mb->kernel);
            }
            if (!mb->coverage.empty()) {
                step++;
            }

            // write the maps
            for (Eigen::Index i=0; i<n_maps; ++i) {
                // update progress bar
                pb.count(n_maps, 1);
                logger->debug("adding map");
                write_maps(f_io,n_io,mb,i);

                if (map_grouping=="detector") {
                    if constexpr (map_type == mapmaking::RawObs) {
                        // get the array for the given map
                        Eigen::Index map_index = arrays_to_maps(i);

                        // check if we move from one file to the next
                        // if so go back to first hdu layer
                        if (i>0) {
                            if (map_index > arrays_to_maps(i-1)) {
                                k = 0;
                            }
                        }

                        // add apt table
                        logger->debug("adding beammap header keys");
                        for (auto const& key: calib.apt_header_keys) {
                            if (key!="flag2") {
                                try {
                                    f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, calib.apt[key](i), key
                                                                          + " (" + calib.apt_header_units[key] + ")");
                                } catch(...) {
                                    f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, 0.0, key
                                                                           + " (" + calib.apt_header_units[key] + ")");
                                }
                            }
                            else {
                                f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, flag2(i), key
                                                                       + " (" + calib.apt_header_units[key] + ")");
                            }
                        }
                        // increment hdu layer
                        k = k + step;
                    }
                }
            }

            logger->info("maps have been written to:");
            for (Eigen::Index i=0; i<f_io->size(); ++i) {
                logger->info("{}.fits",f_io->at(i).filepath);
            }
        };

        if (split_by_flag_mode) {
            if (!mb->kernel.empty()) {
                timestream::log_kernel_map_diag(
                    logger,
                    "map output " + dir_name + " split-by-flag before write",
                    mb->kernel);
            }
            std::set<int> split_values(beammap_split_flag_values.begin(), beammap_split_flag_values.end());
            Eigen::Index n_selected_maps = 0;
            for (Eigen::Index i = 0; i < n_maps; ++i) {
                const int det_flag = static_cast<int>(std::lround(calib.apt["flag"](i)));
                if (split_values.count(det_flag) > 0) {
                    n_selected_maps++;
                }
            }

            if (n_selected_maps <= 0) {
                logger->warn("beammap split_fits_by_flag selected no detector maps; using standard map output");
                write_standard_maps();
            }
            else {
                std::vector<std::string> base_filepaths;
                base_filepaths.reserve(f_io->size());
                for (const auto &fio : *f_io) {
                    base_filepaths.push_back(fio.filepath);
                }

                std::vector<std::string> base_noise_filepaths;
                base_noise_filepaths.reserve(n_io->size());
                for (const auto &nio : *n_io) {
                    base_noise_filepaths.push_back(nio.filepath);
                }

                // close and remove the default unsplit files before writing split outputs
                f_io->clear();
                n_io->clear();
                for (const auto &path : base_filepaths) {
                    const auto fits_path = path + ".fits";
                    try {
                        if (fs::exists(fits_path)) {
                            fs::remove(fits_path);
                        }
                    }
                    catch (const std::exception &e) {
                        logger->warn("unable to remove unsplit beammap file {}: {}", fits_path, e.what());
                    }
                }
                for (const auto &path : base_noise_filepaths) {
                    const auto fits_path = path + ".fits";
                    try {
                        if (fs::exists(fits_path)) {
                            fs::remove(fits_path);
                        }
                    }
                    catch (const std::exception &e) {
                        logger->warn("unable to remove unsplit beammap noise file {}: {}", fits_path, e.what());
                    }
                }

                using split_io_t = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>;

                for (const auto flag_value : beammap_split_flag_values) {
                    Eigen::Index n_flag_maps = 0;
                    for (Eigen::Index i = 0; i < n_maps; ++i) {
                        const int det_flag = static_cast<int>(std::lround(calib.apt["flag"](i)));
                        if (det_flag == flag_value) {
                            n_flag_maps++;
                        }
                    }

                    if (n_flag_maps <= 0) {
                        logger->warn("beammap split_fits_by_flag: no detector maps found with flag={}; skipping", flag_value);
                        continue;
                    }

                    std::string split_suffix = "_flag" + std::to_string(flag_value);
                    if (flag_value == 0) {
                        split_suffix += "_good";
                    }
                    else if (flag_value == 1) {
                        split_suffix += "_bad";
                    }

                    std::vector<split_io_t> split_f_io_vec;
                    std::vector<split_io_t> split_n_io_vec;
                    split_f_io_vec.reserve(base_filepaths.size());
                    for (const auto &path : base_filepaths) {
                        split_f_io_vec.emplace_back(path + split_suffix);
                    }
                    if (!mb->noise.empty()) {
                        split_n_io_vec.reserve(base_noise_filepaths.size());
                        for (const auto &path : base_noise_filepaths) {
                            split_n_io_vec.emplace_back(path + split_suffix);
                        }
                    }

                    auto split_f_io = &split_f_io_vec;
                    auto split_n_io = &split_n_io_vec;

                    tula::logging::progressbar pb(
                        [&](const auto &msg) { logger->info("{}", msg); }, 100,
                        "output progress (flag=" + std::to_string(flag_value) + ") ");

                    for (Eigen::Index i = 0; i < split_f_io->size(); ++i) {
                        logger->debug("adding primary header to split file {} flag={}", i, flag_value);
                        add_phdu(split_f_io, mb, i);
                        split_f_io->at(i).pfits->pHDU().addKey("BEAMMAP.SPLIT_BY", "flag",
                                                                "Beammap detector split criterion");
                        split_f_io->at(i).pfits->pHDU().addKey("BEAMMAP.SPLIT_VALUE", flag_value,
                                                                "Beammap detector flag value in this file");

                        if (!mb->noise.empty()) {
                            logger->debug("adding primary header to split noise file {} flag={}", i, flag_value);
                            add_phdu(split_n_io, mb, i);
                            split_n_io->at(i).pfits->pHDU().addKey("BEAMMAP.SPLIT_BY", "flag",
                                                                    "Beammap detector split criterion");
                            split_n_io->at(i).pfits->pHDU().addKey("BEAMMAP.SPLIT_VALUE", flag_value,
                                                                    "Beammap detector flag value in this file");
                        }
                    }

                    Eigen::Index step = 2;
                    if (!mb->kernel.empty()) {
                        step++;
                    }
                    if (!mb->coverage.empty()) {
                        step++;
                    }

                    std::vector<Eigen::Index> hdu_layer(split_f_io->size(), 0);

                    for (Eigen::Index i = 0; i < n_maps; ++i) {
                        const int det_flag = static_cast<int>(std::lround(calib.apt["flag"](i)));
                        if (det_flag != flag_value) {
                            continue;
                        }

                        pb.count(n_flag_maps, 1);
                        logger->debug("adding split map for detector {} flag={}", i, flag_value);
                        write_maps(split_f_io, split_n_io, mb, i);

                        if (map_grouping == "detector") {
                            if constexpr (map_type == mapmaking::RawObs) {
                                const Eigen::Index map_index = arrays_to_maps(i);
                                const Eigen::Index k = hdu_layer.at(map_index);

                                logger->debug("adding split beammap header keys");
                                for (auto const &key : calib.apt_header_keys) {
                                    if (key != "flag2") {
                                        try {
                                            split_f_io->at(map_index).hdus.at(k)->addKey(
                                                "BEAMMAP." + key, calib.apt[key](i),
                                                key + " (" + calib.apt_header_units[key] + ")");
                                        }
                                        catch (...) {
                                            split_f_io->at(map_index).hdus.at(k)->addKey(
                                                "BEAMMAP." + key, 0.0,
                                                key + " (" + calib.apt_header_units[key] + ")");
                                        }
                                    }
                                    else {
                                        split_f_io->at(map_index).hdus.at(k)->addKey(
                                            "BEAMMAP." + key, flag2(i),
                                            key + " (" + calib.apt_header_units[key] + ")");
                                    }
                                }
                                hdu_layer.at(map_index) = hdu_layer.at(map_index) + step;
                            }
                        }
                    }

                    logger->info("beammap split maps (flag={}) have been written to:", flag_value);
                    for (Eigen::Index i = 0; i < split_f_io->size(); ++i) {
                        logger->info("{}.fits", split_f_io->at(i).filepath);
                    }
                }
            }
        }
        else {
            write_standard_maps();
        }
    }

    // clear fits file vectors to ensure its closed.
    f_io->clear();
    n_io->clear();

    if (map_grouping!="detector") {
        // write psd and histogram files
        logger->debug("writing psds");
        write_psd<map_type>(mb, dir_name);
        logger->debug("writing histograms");
        write_hist<map_type>(mb, dir_name);
        logger->debug("writing map diagnostics");
        write_mapdiag<map_type>(mb, dir_name);
    }
}

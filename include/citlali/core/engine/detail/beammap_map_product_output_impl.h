#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_map_product_headers.h>
#include <citlali/core/engine/detail/beammap_map_product_split_helpers.h>
#include <citlali/core/pipeline/map_output_debug_breadcrumb.h>

template <mapmaking::MapType map_type>
void Beammap::write_beammap_map_products(
    mapmaking::MapBuffer *mb,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
    const std::string &dir_name) {
    if (!run_mapmaking) {
        return;
    }

    const bool detector_grouping =
        typed_config.mapmaking.grouping ==
        citlali::config::MapGrouping::detector;
    bool split_by_flag_mode = false;
    if constexpr (map_type == mapmaking::RawObs) {
        split_by_flag_mode = detector_grouping && beammap_split_fits_by_flag;
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

            if (!mb->kernel.empty()) {
                timestream::log_kernel_map_diag(
                    logger,
                    "map output " + dir_name + " before write",
                    mb->kernel);
            }

            // write the maps
            for (Eigen::Index i=0; i<n_maps; ++i) {
                // update progress bar
                pb.count(n_maps, 1);
                logger->debug("adding map");
                const Eigen::Index signal_hdu_index = write_maps(f_io,n_io,mb,i);

                if (detector_grouping) {
                    if constexpr (map_type == mapmaking::RawObs) {
                        // get the array for the given map
                        Eigen::Index map_index = arrays_to_maps(i);

                        // add apt table
                        logger->debug("adding beammap header keys");
                        citlali::pipeline::update_map_output_debug_breadcrumb(
                            "beammap-detector-header",
                            f_io->at(map_index).filepath.c_str(), i, map_index,
                            -1, -1, signal_hdu_index,
                            static_cast<Eigen::Index>(f_io->at(map_index).hdus.size()));
                        beammap_map_product_headers::add_detector_header_keys(
                            f_io->at(map_index).hdus.at(signal_hdu_index), calib, flag2, i);
                        citlali::pipeline::reset_map_output_debug_breadcrumb();
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
            const Eigen::Index n_selected_maps =
                beammap_map_product_split_helpers::count_maps_with_any_flag(
                    calib.apt["flag"], n_maps, beammap_split_flag_values);

            if (n_selected_maps <= 0) {
                logger->warn("beammap split_fits_by_flag selected no detector maps; using standard map output");
                write_standard_maps();
            }
            else {
                const auto base_filepaths =
                    beammap_map_product_split_helpers::filepaths(*f_io);
                const auto base_noise_filepaths =
                    beammap_map_product_split_helpers::filepaths(*n_io);

                // close and remove the default unsplit files before writing split outputs
                f_io->clear();
                n_io->clear();
                beammap_map_product_split_helpers::remove_fits_files(
                    base_filepaths, "map", logger);
                beammap_map_product_split_helpers::remove_fits_files(
                    base_noise_filepaths, "noise", logger);

                using split_io_t = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>;

                for (const auto flag_value : beammap_split_flag_values) {
                    const Eigen::Index n_flag_maps =
                        beammap_map_product_split_helpers::count_maps_with_flag(
                            calib.apt["flag"], n_maps, flag_value);

                    if (n_flag_maps <= 0) {
                        logger->warn("beammap split_fits_by_flag: no detector maps found with flag={}; skipping", flag_value);
                        continue;
                    }

                    const std::string split_suffix =
                        beammap_map_product_split_helpers::split_suffix(
                            flag_value);

                    auto split_f_io_vec =
                        beammap_map_product_split_helpers::make_split_io<split_io_t>(
                            base_filepaths, split_suffix);
                    std::vector<split_io_t> split_n_io_vec;
                    if (!mb->noise.empty()) {
                        split_n_io_vec =
                            beammap_map_product_split_helpers::make_split_io<split_io_t>(
                                base_noise_filepaths, split_suffix);
                    }

                    auto split_f_io = &split_f_io_vec;
                    auto split_n_io = &split_n_io_vec;

                    tula::logging::progressbar pb(
                        [&](const auto &msg) { logger->info("{}", msg); }, 100,
                        "output progress (flag=" + std::to_string(flag_value) + ") ");

                    for (Eigen::Index i = 0; i < split_f_io->size(); ++i) {
                        logger->debug("adding primary header to split file {} flag={}", i, flag_value);
                        add_phdu(split_f_io, mb, i);
                        beammap_map_product_split_helpers::add_split_primary_header(
                            *split_f_io, i, flag_value);

                        if (!mb->noise.empty()) {
                            logger->debug("adding primary header to split noise file {} flag={}", i, flag_value);
                            add_phdu(split_n_io, mb, i);
                            beammap_map_product_split_helpers::add_split_primary_header(
                                *split_n_io, i, flag_value);
                        }
                    }

                    for (Eigen::Index i = 0; i < n_maps; ++i) {
                        const int det_flag = static_cast<int>(std::lround(calib.apt["flag"](i)));
                        if (det_flag != flag_value) {
                            continue;
                        }

                        pb.count(n_flag_maps, 1);
                        logger->debug("adding split map for detector {} flag={}", i, flag_value);
                        const Eigen::Index signal_hdu_index = write_maps(split_f_io, split_n_io, mb, i);

                        if (detector_grouping) {
                            if constexpr (map_type == mapmaking::RawObs) {
                                const Eigen::Index map_index = arrays_to_maps(i);

                                logger->debug("adding split beammap header keys");
                                citlali::pipeline::update_map_output_debug_breadcrumb(
                                    "beammap-split-detector-header",
                                    split_f_io->at(map_index).filepath.c_str(),
                                    i, map_index, -1, -1, signal_hdu_index,
                                    static_cast<Eigen::Index>(split_f_io->at(map_index).hdus.size()),
                                    flag_value);
                                beammap_map_product_headers::add_detector_header_keys(
                                    split_f_io->at(map_index).hdus.at(signal_hdu_index), calib, flag2, i);
                                citlali::pipeline::reset_map_output_debug_breadcrumb();
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

    if (!detector_grouping) {
        // write psd and histogram files
        logger->debug("writing psds");
        write_psd<map_type>(mb, dir_name);
        logger->debug("writing histograms");
        write_hist<map_type>(mb, dir_name);
        logger->debug("writing map diagnostics");
        write_mapdiag<map_type>(mb, dir_name);
    }
}

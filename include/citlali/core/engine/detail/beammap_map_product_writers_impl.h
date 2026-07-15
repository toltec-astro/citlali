#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

template <mapmaking::MapType map_type>
void Beammap::write_standard_beammap_map_entries(
    mapmaking::MapBuffer *mb,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
    citlali::pipeline::StageProfileCollector &stage_profile,
    const std::string &dir_name,
    bool detector_grouping) {
    (void)stage_profile;
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100,
        "output progress ");

    const auto profile_scope =
        citlali::pipeline::profile_stage(stage_profile,
            "beammap.map_output.write_maps", logger,
            "dir=" + dir_name +
                " maps=" + std::to_string(map_indices.n_maps));
    for (Eigen::Index i=0; i<map_indices.n_maps; ++i) {
        pb.count(map_indices.n_maps, 1);
        logger->debug("adding map");
        const Eigen::Index signal_hdu_index =
            write_maps(f_io, n_io, mb, i);

        maybe_add_beammap_detector_map_header<map_type>(
            f_io, i, signal_hdu_index, detector_grouping,
            "beammap-detector-header");
    }
}

template <mapmaking::MapType map_type>
void Beammap::write_split_beammap_flag_maps(
    mapmaking::MapBuffer *mb,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *split_f_io,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *split_n_io,
    citlali::pipeline::StageProfileCollector &stage_profile,
    const std::string &dir_name,
    bool detector_grouping,
    int flag_value,
    Eigen::Index n_flag_maps) {
    (void)stage_profile;
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100,
        "output progress (flag=" + std::to_string(flag_value) + ") ");

    const auto profile_scope =
        citlali::pipeline::profile_stage(stage_profile,
            "beammap.map_output.split_write_maps", logger,
            "dir=" + dir_name +
                " flag=" + std::to_string(flag_value) +
                " maps=" + std::to_string(n_flag_maps));
    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        const int det_flag =
            beammap_map_product_split_helpers::detector_flag(
                calib.apt["flag"], i);
        if (det_flag != flag_value) {
            continue;
        }

        pb.count(n_flag_maps, 1);
        logger->debug("adding split map for detector {} flag={}", i,
                      flag_value);
        const Eigen::Index signal_hdu_index =
            write_maps(split_f_io, split_n_io, mb, i);

        maybe_add_beammap_detector_map_header<map_type>(
            split_f_io, i, signal_hdu_index, detector_grouping,
            "beammap-split-detector-header", flag_value);
    }
}

template <mapmaking::MapType map_type>
bool Beammap::should_split_beammap_maps_by_flag(
    bool detector_grouping,
    const citlali::config::BeammapSplitFitsByFlagConfig &split_config) {
    bool split_by_flag_mode = false;
    if constexpr (map_type == mapmaking::RawObs) {
        split_by_flag_mode = detector_grouping && split_config.enabled;
        if (split_by_flag_mode && split_config.flag_values.empty()) {
            logger->warn("beammap.split_fits_by_flag enabled but no flag_values specified; using standard map output");
            split_by_flag_mode = false;
        }
    }
    return split_by_flag_mode;
}

template <mapmaking::MapType map_type>
void Beammap::write_beammap_non_detector_map_diagnostics(
    mapmaking::MapBuffer *mb,
    const std::string &dir_name,
    bool detector_grouping) {
    if (detector_grouping) {
        return;
    }

    logger->debug("writing psds");
    write_psd<map_type>(mb, dir_name);
    logger->debug("writing histograms");
    write_hist<map_type>(mb, dir_name);
    logger->debug("writing map diagnostics");
    write_mapdiag<map_type>(mb, dir_name);
}

template <mapmaking::MapType map_type>
void Beammap::write_standard_beammap_map_products(
    mapmaking::MapBuffer *mb,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
    citlali::pipeline::StageProfileCollector &stage_profile,
    const std::string &dir_name,
    bool detector_grouping) {
    add_beammap_map_primary_headers(
        mb, f_io, n_io, stage_profile,
        "beammap.map_output.primary_headers",
        "dir=" + dir_name);
    logger->debug("done adding primary headers");

    if (!mb->kernel.empty()) {
        timestream::log_kernel_map_diag(
            logger, "map output " + dir_name + " before write",
            mb->kernel);
    }

    write_standard_beammap_map_entries<map_type>(
        mb, f_io, n_io, stage_profile, dir_name, detector_grouping);

    beammap_map_product_split_helpers::log_output_filepaths(logger, *f_io);
}

template <mapmaking::MapType map_type>
void Beammap::write_split_beammap_map_products(
    mapmaking::MapBuffer *mb,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
    citlali::pipeline::StageProfileCollector &stage_profile,
    const std::string &dir_name,
    bool detector_grouping,
    const std::vector<int> &flag_values) {
    if (!mb->kernel.empty()) {
        timestream::log_kernel_map_diag(
            logger, "map output " + dir_name + " split-by-flag before write",
            mb->kernel);
    }
    const Eigen::Index n_selected_maps =
        beammap_map_product_split_helpers::count_maps_with_any_flag(
            calib.apt["flag"], map_indices.n_maps, flag_values);

    if (n_selected_maps <= 0) {
        logger->warn("beammap split_fits_by_flag selected no detector maps; using standard map output");
        write_standard_beammap_map_products<map_type>(
            mb, f_io, n_io, stage_profile, dir_name, detector_grouping);
        return;
    }

    const auto output_files =
        prepare_split_beammap_map_output_files(f_io, n_io);

    using split_io_t =
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>;

    for (const auto flag_value : flag_values) {
        const Eigen::Index n_flag_maps =
            beammap_map_product_split_helpers::count_maps_with_flag(
                calib.apt["flag"], map_indices.n_maps, flag_value);

        if (n_flag_maps <= 0) {
            logger->warn("beammap split_fits_by_flag: no detector maps found with flag={}; skipping", flag_value);
            continue;
        }

        const std::string split_suffix =
            beammap_map_product_split_helpers::split_suffix(flag_value);

        auto split_f_io_vec =
            beammap_map_product_split_helpers::make_split_io<split_io_t>(
                output_files.base_filepaths, split_suffix);
        std::vector<split_io_t> split_n_io_vec;
        if (!mb->noise.empty()) {
            split_n_io_vec =
                beammap_map_product_split_helpers::make_split_io<split_io_t>(
                    output_files.base_noise_filepaths, split_suffix);
        }

        auto split_f_io = &split_f_io_vec;
        auto split_n_io = &split_n_io_vec;

        add_beammap_map_primary_headers(
            mb, split_f_io, split_n_io, stage_profile,
            "beammap.map_output.split_primary_headers",
            "dir=" + dir_name + " flag=" + std::to_string(flag_value),
            flag_value);

        write_split_beammap_flag_maps<map_type>(
            mb, split_f_io, split_n_io, stage_profile, dir_name,
            detector_grouping,
            flag_value, n_flag_maps);

        beammap_map_product_split_helpers::log_split_output_filepaths(
            logger, *split_f_io, flag_value);
    }
}

template <mapmaking::MapType map_type>
void Beammap::write_beammap_map_products(
    mapmaking::MapBuffer *mb,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
    citlali::pipeline::StageProfileCollector &stage_profile,
    const std::string &dir_name) {
    const auto &mapmaking_config = citlali::pipeline::mapmaking_config(*this);
    const auto &beammap_config = citlali::pipeline::beammap_config(*this);

    if (!citlali::config::mapmaking_active(mapmaking_config)) {
        return;
    }

    const bool detector_grouping =
        citlali::config::is_detector_map_grouping(
            mapmaking_config.grouping);
    const auto &split_config = beammap_config.split_fits_by_flag;
    const bool split_by_flag_mode =
        should_split_beammap_maps_by_flag<map_type>(
            detector_grouping, split_config);

    // wiener filtered maps write before this and are deleted from the vector.
    if (!f_io->empty()) {
        if (split_by_flag_mode) {
            write_split_beammap_map_products<map_type>(
                mb, f_io, n_io, stage_profile, dir_name, detector_grouping,
                split_config.flag_values);
        }
        else {
            write_standard_beammap_map_products<map_type>(
                mb, f_io, n_io, stage_profile, dir_name, detector_grouping);
        }
    }

    // clear fits file vectors to ensure its closed.
    f_io->clear();
    n_io->clear();

    write_beammap_non_detector_map_diagnostics<map_type>(
        mb, dir_name, detector_grouping);
}

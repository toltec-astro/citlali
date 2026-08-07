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
    const std::vector<Eigen::Index> &detector_indices) {
    (void)stage_profile;
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100,
        "output progress (flag=" + std::to_string(flag_value) + ") ");

    const auto profile_scope =
        citlali::pipeline::profile_stage(stage_profile,
            "beammap.map_output.split_write_maps", logger,
            "dir=" + dir_name +
                " flag=" + std::to_string(flag_value) +
                " maps=" + std::to_string(detector_indices.size()));
    for (const auto i : detector_indices) {
        pb.count(detector_indices.size(), 1);
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
    std::vector<std::vector<Eigen::Index>> detector_indices_by_flag;
    detector_indices_by_flag.reserve(flag_values.size());
    std::size_t n_selected_maps = 0;
    for (const auto flag_value : flag_values) {
        detector_indices_by_flag.push_back(
            beammap_map_product_split_helpers::map_indices_with_flag(
                calib.apt["flag"], map_indices.n_maps, flag_value));
        n_selected_maps += detector_indices_by_flag.back().size();
    }

    if (n_selected_maps == 0) {
        logger->warn("beammap split_fits_by_flag selected no detector maps; using standard map output");
        write_standard_beammap_map_products<map_type>(
            mb, f_io, n_io, stage_profile, dir_name, detector_grouping);
        return;
    }

    const auto output_files =
        prepare_split_beammap_map_output_files(f_io, n_io);

    using split_io_t =
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>;

    for (std::size_t split_index = 0;
         split_index < flag_values.size(); ++split_index) {
        const int flag_value = flag_values[split_index];
        const auto &detector_indices =
            detector_indices_by_flag[split_index];

        if (detector_indices.empty()) {
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
            flag_value, detector_indices);

        beammap_map_product_split_helpers::log_split_output_filepaths(
            logger, *split_f_io, flag_value);

        const auto published_data_paths =
            citlali::pipeline::noise_fits_output_paths(*split_f_io);
        const auto published_noise_paths =
            citlali::pipeline::noise_fits_output_paths(*split_n_io);
        split_f_io->clear();
        split_n_io->clear();

        auto &run_noise_plan = citlali::pipeline::noise_plan(*this);
        citlali::pipeline::record_noise_fits_members(
            run_noise_plan, published_data_paths, published_noise_paths,
            citlali::pipeline::noise_data_fits_have_package_join(
                run_noise_plan, false, *mb));
    }

    constexpr bool is_coadd =
        map_type == mapmaking::RawCoadd ||
        map_type == mapmaking::FilteredCoadd;
    constexpr bool is_filtered =
        map_type == mapmaking::FilteredObs ||
        map_type == mapmaking::FilteredCoadd;
    citlali::pipeline::record_noise_selected_map_output_stage(
        citlali::pipeline::noise_plan(*this), is_coadd, is_filtered,
        *mb, n_selected_maps);
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
    const bool map_output_started = !f_io->empty();
    if (map_output_started) {
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

    const auto published_data_paths =
        citlali::pipeline::noise_fits_output_paths(*f_io);
    const auto published_noise_paths =
        citlali::pipeline::noise_fits_output_paths(*n_io);

    // clear fits file vectors to ensure its closed.
    f_io->clear();
    n_io->clear();

    if (map_output_started && !published_data_paths.empty()) {
        constexpr bool is_coadd =
            map_type == mapmaking::RawCoadd ||
            map_type == mapmaking::FilteredCoadd;
        constexpr bool is_filtered =
            map_type == mapmaking::FilteredObs ||
            map_type == mapmaking::FilteredCoadd;
        auto &run_noise_plan = citlali::pipeline::noise_plan(*this);
        citlali::pipeline::record_noise_map_output_publication(
            run_noise_plan, is_coadd, is_filtered, *mb,
            published_data_paths, published_noise_paths);
    }

    write_beammap_non_detector_map_diagnostics<map_type>(
        mb, dir_name, detector_grouping);
}

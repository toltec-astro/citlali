#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_detector_tod_selection.h>
#include <citlali/core/engine/detail/beammap_detector_tod_netcdf_helpers.h>
#include <citlali/core/engine/detail/beammap_detector_tod_output_helpers.h>

void Beammap::write_detector_specific_ptc_tod(int output_iter) {
    if (!beammap_detector_tod_output_enabled) {
        return;
    }
    const Eigen::Index n_scans = telescope.scan_indices.cols();
    if (n_scans <= 0) {
        logger->error("cannot write detector-specific PTC TOD: no scans");
        std::exit(EXIT_FAILURE);
    }
    if (typed_config.mapmaking.grouping !=
        citlali::config::MapGrouping::detector) {
        logger->warn(
            "beammap.detector_tod_output requires detector map grouping; skipping detector-specific PTC TOD");
        return;
    }
    const auto output_counts = beammap_detector_tod_output_helpers::output_counts(
        beammap_detector_tod_output_n_uniform,
        beammap_detector_tod_output_n_source_dense);
    const int n_uniform = output_counts.n_uniform;
    const int n_dense = output_counts.n_dense;
    const Eigen::Index n_slots = output_counts.n_slots;
    if (n_slots <= 0) {
        logger->warn("beammap.detector_tod_output requested with no output slots; skipping");
        return;
    }
    const Eigen::Index n_samples_max =
        beammap_detector_tod_output_helpers::max_ptc_samples(ptcs);
    if (n_samples_max <= 0) {
        logger->warn("beammap.detector_tod_output has no PTC samples to write; skipping");
        return;
    }

    std::vector<Eigen::Index> uniform_scans =
        beammap_detector_tod_selection::uniform_scan_indices(n_uniform, n_scans);

    auto [sampled_indices, sampled_scan] =
        beammap_detector_tod_selection::sampled_scan_samples(
            telescope.scan_indices, telescope.tel_data, n_scans);
    const Eigen::Index n_sampled = static_cast<Eigen::Index>(sampled_indices.size());
    if (n_sampled <= 0) {
        logger->warn("beammap.detector_tod_output cannot sample telescope pointing; skipping");
        return;
    }

    std::map<std::string, Eigen::VectorXd> sampled_tel_data =
        beammap_detector_tod_selection::sample_tel_data(
            telescope.tel_data, sampled_indices);
    std::map<std::string, Eigen::VectorXd> pointing_offsets;
    pointing_offsets["az"] =
        beammap_detector_tod_selection::sample_pointing_offset(
            pointing_offsets_arcsec, "az", sampled_indices);
    pointing_offsets["alt"] =
        beammap_detector_tod_selection::sample_pointing_offset(
            pointing_offsets_arcsec, "alt", sampled_indices);

    const int fill_int = -2147483647;
    const double fill_double = std::numeric_limits<double>::quiet_NaN();
    const float fill_float = std::numeric_limits<float>::quiet_NaN();
    const signed char fill_flag = static_cast<signed char>(-1);
    const auto total_det_slots =
        static_cast<std::size_t>(calib.n_dets) * static_cast<std::size_t>(n_slots);
    std::vector<int> slot_scan_index(total_det_slots, fill_int);
    std::vector<int> slot_kind(total_det_slots, fill_int);
    std::vector<int> slot_n_samples(total_det_slots, fill_int);
    std::vector<int> slot_inner_start(total_det_slots, fill_int);
    std::vector<int> slot_inner_end(total_det_slots, fill_int);
    std::vector<int> slot_outer_start(total_det_slots, fill_int);
    std::vector<int> slot_outer_end(total_det_slots, fill_int);
    std::vector<double> slot_source_distance_arcsec(total_det_slots, fill_double);
    std::vector<int> det_center_scan_index(static_cast<std::size_t>(calib.n_dets), fill_int);
    std::vector<double> det_center_distance_arcsec(static_cast<std::size_t>(calib.n_dets), fill_double);
    std::vector<double> det_fit_x_arcsec(static_cast<std::size_t>(calib.n_dets), fill_double);
    std::vector<double> det_fit_y_arcsec(static_cast<std::size_t>(calib.n_dets), fill_double);
    std::vector<int> det_fit_good(static_cast<std::size_t>(calib.n_dets), 0);
    Eigen::Index n_det_fit_positions = 0;
    Eigen::Index n_det_fallback_positions = 0;
    std::map<Eigen::Index, Eigen::Index> center_scan_counts;
    std::vector<double> center_distances;
    center_distances.reserve(static_cast<std::size_t>(calib.n_dets));

    for (Eigen::Index det = 0; det < calib.n_dets; ++det) {
        bool used_fit = false;
        auto [x_arcsec, y_arcsec] = beammap_detector_tod_selection::detector_source_position(
            det, good_fits, params, calib.apt["x_t"], calib.apt["y_t"],
            omb.pixel_size_rad, omb.n_cols, omb.n_rows, used_fit);
        det_fit_x_arcsec[static_cast<std::size_t>(det)] = x_arcsec;
        det_fit_y_arcsec[static_cast<std::size_t>(det)] = y_arcsec;
        det_fit_good[static_cast<std::size_t>(det)] = used_fit ? 1 : 0;
        if (used_fit) {
            n_det_fit_positions++;
        }
        else if (std::isfinite(x_arcsec) && std::isfinite(y_arcsec)) {
            n_det_fallback_positions++;
        }
        std::vector<double> distances_arcsec;
        const Eigen::Index center_scan =
            beammap_detector_tod_selection::scan_distances_for_detector_source(
                det, x_arcsec, y_arcsec, n_scans, n_sampled, sampled_scan,
                sampled_tel_data, calib.apt["x_t"], calib.apt["y_t"],
                telescope.pixel_axes, pointing_offsets,
                typed_config.mapmaking.grouping, distances_arcsec);
        det_center_scan_index[static_cast<std::size_t>(det)] = static_cast<int>(center_scan + 1);
        center_scan_counts[center_scan]++;
        if (center_scan >= 0 && center_scan < n_scans &&
            std::isfinite(distances_arcsec[static_cast<std::size_t>(center_scan)])) {
            det_center_distance_arcsec[static_cast<std::size_t>(det)] =
                distances_arcsec[static_cast<std::size_t>(center_scan)];
            center_distances.push_back(distances_arcsec[static_cast<std::size_t>(center_scan)]);
        }

        Eigen::Index slot = 0;
        for (const auto scan_index : uniform_scans) {
            beammap_detector_tod_selection::fill_slot_scan_metadata(
                det, slot, n_slots, scan_index, n_scans, 1,
                telescope.scan_indices, ptcs, distances_arcsec,
                slot_scan_index, slot_kind, slot_n_samples, slot_inner_start,
                slot_inner_end, slot_outer_start, slot_outer_end,
                slot_source_distance_arcsec);
            slot++;
        }
        for (const auto scan_index : beammap_detector_tod_selection::dense_scan_window(center_scan, n_dense, n_scans)) {
            beammap_detector_tod_selection::fill_slot_scan_metadata(
                det, slot, n_slots, scan_index, n_scans, 2,
                telescope.scan_indices, ptcs, distances_arcsec,
                slot_scan_index, slot_kind, slot_n_samples, slot_inner_start,
                slot_inner_end, slot_outer_start, slot_outer_end,
                slot_source_distance_arcsec);
            slot++;
        }
    }

    const std::string center_scan_summary =
        beammap_detector_tod_selection::format_center_scan_counts(
            center_scan_counts);

    double median_center_distance_arcsec = std::numeric_limits<double>::quiet_NaN();
    if (!center_distances.empty()) {
        Eigen::Map<Eigen::VectorXd> dist_vec(
            center_distances.data(),
            static_cast<Eigen::Index>(center_distances.size()));
        median_center_distance_arcsec = tula::alg::median(dist_vec);
    }

    const auto output_paths = beammap_detector_tod_output_helpers::output_paths(
        obsnum_dir_name, beammap_detector_tod_output_subdir_name,
        telescope.sim_obs, redu_type, obsnum);
    const std::string &filename = output_paths.filename;

    logger->info(
        "writing detector-specific PTC TOD iter={} file={} n_dets={} n_slots={} n_uniform={} n_source_dense={} fit_positions={} fallback_positions={} median_center_distance_arcsec={} top_center_scans={}",
        output_iter,
        filename,
        calib.n_dets,
        n_slots,
        n_uniform,
        n_dense,
        n_det_fit_positions,
        n_det_fallback_positions,
        median_center_distance_arcsec,
        center_scan_summary);

    write_netcdf_atomic(filename, [&](netCDF::NcFile &fo) {
        netCDF::NcDim n_tod_output_type_dim = fo.addDim("n_tod_output_type", 1);
        netCDF::NcVar tod_output_type_var =
            fo.addVar("tod_output_type", netCDF::ncString, n_tod_output_type_dim);
        const std::vector<size_t> tod_output_type_index = {0};
        std::string tod_output_type_name = "ptc_detector_tod";
        tod_output_type_var.putVar(tod_output_type_index, tod_output_type_name);

        netCDF::NcVar obsnum_v = fo.addVar("obsnum", netCDF::ncInt);
        obsnum_v.putAtt("units", "N/A");
        int obsnum_int = std::stoi(obsnum);
        obsnum_v.putVar(&obsnum_int);
        add_netcdf_var<std::string>(fo, "SOURCE", telescope.source_name);
        add_netcdf_var<std::string>(fo, "PROJID", telescope.project_id);
        add_netcdf_var<std::string>(fo, "GOAL", redu_type);
        add_netcdf_var<std::string>(fo, "OBSGOAL", telescope.obs_goal);
        add_netcdf_var<std::string>(fo, "TYPE", tod_type);
        add_netcdf_var<std::string>(fo, "PIPELINE", "CITLALI");
        add_netcdf_var<std::string>(fo, "VERSION", CITLALI_GIT_VERSION);
        add_netcdf_var<std::string>(fo, "KIDS", KIDSCPP_GIT_VERSION);
        add_netcdf_var<std::string>(fo, "TULA", TULA_GIT_VERSION);
        add_netcdf_var(fo, "SourceRa", telescope.tel_header["Header.Source.Ra"](0));
        add_netcdf_var(fo, "SourceDec", telescope.tel_header["Header.Source.Dec"](0));
        add_netcdf_var(fo, "PTC_SAMPRATE", processed_time_chunk_fs_hz());
        add_netcdf_var(fo, "FRUITLOOPS_ITER", output_iter);
        add_netcdf_var(fo, "CONFIG.BEAMMAP.DETECTOR_TOD.N_UNIFORM", n_uniform);
        add_netcdf_var(fo, "CONFIG.BEAMMAP.DETECTOR_TOD.N_SOURCE_DENSE", n_dense);

        netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);
        netCDF::NcDim n_slots_dim = fo.addDim("n_detector_tod_slots", n_slots);
        netCDF::NcDim n_samples_dim = fo.addDim("n_samples_max", n_samples_max);
        std::vector<netCDF::NcDim> det_dims = {n_dets_dim};
        std::vector<netCDF::NcDim> det_slot_dims = {n_dets_dim, n_slots_dim};
        std::vector<netCDF::NcDim> data_dims = {n_dets_dim, n_slots_dim, n_samples_dim};
        std::vector<std::size_t> det_slot_chunks = {1, static_cast<std::size_t>(n_slots)};
        std::vector<std::size_t> data_chunks = {
            1, 1, static_cast<std::size_t>(n_samples_max)};

        namespace tod_nc = beammap_detector_tod_netcdf_helpers;

        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_uid", "detector UID along n_dets",
            tod_nc::apt_int_values(calib.apt, "uid", calib.n_dets, fill_int));
        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_array", "array index along n_dets",
            tod_nc::apt_int_values(calib.apt, "array", calib.n_dets, fill_int));
        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_network", "network index along n_dets",
            tod_nc::apt_int_values(calib.apt, "nw", calib.n_dets, fill_int));
        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_fit_good",
            "1 when the source-crossing scan was centered on a good fit, else 0",
            det_fit_good);
        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_source_center_scan_index",
            "1-based full-observation scan with closest approach to the fitted detector source position",
            det_center_scan_index);
        tod_nc::put_detector_double(
            fo, det_dims, "detector_tod_source_center_distance_arcsec", "arcsec",
            "closest sampled distance from source in detector source-center scan",
            det_center_distance_arcsec);
        tod_nc::put_detector_double(
            fo, det_dims, "detector_tod_fit_x_t_arcsec", "arcsec",
            "fitted detector source x_t used for dense scan selection",
            det_fit_x_arcsec);
        tod_nc::put_detector_double(
            fo, det_dims, "detector_tod_fit_y_t_arcsec", "arcsec",
            "fitted detector source y_t used for dense scan selection",
            det_fit_y_arcsec);

        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_slot_kind",
            "slot kind: 1=uniform over full raster, 2=dense around detector source crossing",
            slot_kind);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_scan_index",
            "1-based full-observation scan index selected for this detector/slot",
            slot_scan_index);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_n_samples",
            "number of PTC samples populated for this detector/slot",
            slot_n_samples);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_scan_inner_start_sample",
            "raw inner-scan start sample from telescope scan definition",
            slot_inner_start);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_scan_inner_end_sample",
            "raw inner-scan end sample from telescope scan definition",
            slot_inner_end);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_scan_outer_start_sample",
            "raw outer-scan start sample from telescope scan definition",
            slot_outer_start);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_scan_outer_end_sample",
            "raw outer-scan end sample from telescope scan definition",
            slot_outer_end);
        tod_nc::put_slot_double(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_source_distance_arcsec",
            "arcsec",
            "closest sampled source distance for this detector on this selected scan",
            slot_source_distance_arcsec);

        netCDF::NcVar signal_v = fo.addVar("signal", netCDF::ncFloat, data_dims);
        signal_v.putAtt("units", omb.sig_unit);
        signal_v.putAtt("comment", "PTC signal for each detector-specific selected scan; unused samples are NaN");
        set_netcdf_chunking_and_compression(signal_v, data_chunks, 1);

        netCDF::NcVar flags_v = fo.addVar("flags", netCDF::ncByte, data_dims);
        flags_v.putAtt("units", "N/A");
        flags_v.putAtt("comment", "0=good, 1=flagged, -1=unused sample");
        set_netcdf_chunking_and_compression(flags_v, data_chunks, 1);

        tod_nc::put_detector_tod_signal_flags(
            signal_v, flags_v, ptcs, slot_scan_index, calib.n_dets, n_slots,
            n_samples_max, fill_float, fill_flag);
    });
}

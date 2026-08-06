#pragma once

// Beammap mapmaking stage implementation detail.
// Include this only after Beammap has been declared.

void Beammap::populate_beammap_maps(
    citlali::config::MapGrouping mapmaking_grouping,
    citlali::config::MapMethod mapmaking_method,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps,
    bool update_progress) {
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100,
        "PTC progress ");
    const bool make_noise_maps =
        citlali::pipeline::noise_maps_enabled(*this);
    prepare_beammap_direction_selection();

    mapmaking::MapBuffer disabled_coadd{"direction_cmb_disabled"};
    auto populate_scan = [&](Eigen::Index scan_vec_idx,
                             mapmaking::MapBuffer &map_buffer,
                             bool populate_coadd) {
        auto &ptc = ptcs[static_cast<std::size_t>(scan_vec_idx)];
        auto &scan_apt =
            calib_scans[static_cast<std::size_t>(scan_vec_idx)].apt;
        auto &coadd_buffer = populate_coadd ? cmb : disabled_coadd;
        const bool run_omb = true;
        if (citlali::config::is_detector_map_grouping(mapmaking_grouping)) {
            if (citlali::config::is_naive_map_method(mapmaking_method)) {
                naive_mm.populate_maps_naive_parallel(
                    ptc, map_buffer, coadd_buffer, ptc.map_indices.data,
                    telescope.pixel_axes, scan_apt, telescope.d_fsmp,
                    run_omb, make_noise_maps, active_maps);
            }
            else if (citlali::config::is_jinc_map_method(mapmaking_method)) {
                citlali::pipeline::log_beammap_jinc_preflight(
                    ptc, calib.apt["array"], map_buffer, jinc_mm, logger);
                jinc_mm.populate_maps_jinc_parallel(
                    ptc, map_buffer, coadd_buffer, ptc.map_indices.data,
                    telescope.pixel_axes, scan_apt, telescope.d_fsmp,
                    run_omb, make_noise_maps, active_maps);
            }
            return;
        }
        citlali::pipeline::populate_naive_or_jinc_maps(
            mapmaking_method, naive_mm, jinc_mm, ptc, map_buffer,
            coadd_buffer, ptc.map_indices.data, telescope.pixel_axes,
            scan_apt, telescope.d_fsmp, run_omb, make_noise_maps);
    };

    auto populate_selected_buffers = [&](Eigen::Index scan_vec_idx) {
        const auto selected =
            beammap_direction_buffer_selection(scan_vec_idx);
        if (selected.standard) {
            populate_scan(scan_vec_idx, omb, true);
        }
        if (selected.left) {
            populate_scan(
                scan_vec_idx, beammap_direction_products.left, false);
        }
        if (selected.right) {
            populate_scan(
                scan_vec_idx, beammap_direction_products.right, false);
        }
    };

    if (citlali::config::is_detector_map_grouping(mapmaking_grouping)) {
        for (std::size_t scan_vec_idx = 0; scan_vec_idx < ptcs.size(); ++scan_vec_idx) {
            populate_selected_buffers(
                static_cast<Eigen::Index>(scan_vec_idx));
            if (update_progress) {
                pb.count(telescope.scan_indices.cols(), 1);
            }
        }
        return;
    }

    grppi::map(tula::grppi_utils::dyn_ex(map_parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
        populate_selected_buffers(i);
        if (update_progress) {
            pb.count(telescope.scan_indices.cols(), 1);
        }
        return 0;
    });
}

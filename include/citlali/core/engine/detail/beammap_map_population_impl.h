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

    if (citlali::config::is_detector_map_grouping(mapmaking_grouping)) {
        bool run_omb = true;
        for (std::size_t scan_vec_idx = 0; scan_vec_idx < ptcs.size(); ++scan_vec_idx) {
            if (!beammap_direction_scan_selected(
                    static_cast<Eigen::Index>(scan_vec_idx))) {
                if (update_progress) {
                    pb.count(telescope.scan_indices.cols(), 1);
                }
                continue;
            }
            auto &ptc = ptcs[scan_vec_idx];
            auto &scan_apt = calib_scans[scan_vec_idx].apt;
            if (citlali::config::is_naive_map_method(mapmaking_method)) {
                naive_mm.populate_maps_naive_parallel(
                    ptc, omb, cmb, ptc.map_indices.data, telescope.pixel_axes,
                    scan_apt, telescope.d_fsmp, run_omb, make_noise_maps,
                    active_maps);
            }
            else if (citlali::config::is_jinc_map_method(mapmaking_method)) {
                citlali::pipeline::log_beammap_jinc_preflight(
                    ptc, calib.apt["array"], omb, jinc_mm, logger);
                jinc_mm.populate_maps_jinc_parallel(
                    ptc, omb, cmb, ptc.map_indices.data, telescope.pixel_axes,
                    scan_apt, telescope.d_fsmp, run_omb, make_noise_maps,
                    active_maps);
            }
            if (update_progress) {
                pb.count(telescope.scan_indices.cols(), 1);
            }
        }
        return;
    }

    grppi::map(tula::grppi_utils::dyn_ex(map_parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
        if (!beammap_direction_scan_selected(i)) {
            if (update_progress) {
                pb.count(telescope.scan_indices.cols(), 1);
            }
            return 0;
        }
        bool run_omb = true;
        citlali::pipeline::populate_naive_or_jinc_maps(
            mapmaking_method, naive_mm, jinc_mm, ptcs[i], omb, cmb,
            ptcs[i].map_indices.data, telescope.pixel_axes,
            calib_scans[i].apt, telescope.d_fsmp, run_omb,
            make_noise_maps);
        if (update_progress) {
            pb.count(telescope.scan_indices.cols(), 1);
        }
        return 0;
    });
}

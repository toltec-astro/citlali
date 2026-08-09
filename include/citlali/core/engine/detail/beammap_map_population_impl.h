#pragma once

// Beammap mapmaking stage implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/beammap_map_population.h>

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

    citlali::pipeline::populate_beammap_maps_production(
        mapmaking_grouping, mapmaking_method, map_parallel_policy, jinc_mm,
        naive_mm, omb, cmb, ptcs, calib_scans, scan_in_vec, scan_out_vec,
        telescope, calib.apt["array"], make_noise_maps, active_maps, logger,
        [&] {
            if (update_progress) {
                pb.count(telescope.scan_indices.cols(), 1);
            }
        });
}

#pragma once

#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/beammap_mapmaking_policy.h>

#include <cstddef>

namespace citlali::pipeline {

template <class JincMapmaker, class NaiveMapmaker, class MapBuffer,
          class PtcScans, class CalibScans, class ScanInput,
          class ScanOutput, class Telescope, class Logger,
          class ArrayIdentity, class Progress>
void populate_beammap_maps_production(
    citlali::config::MapGrouping grouping,
    citlali::config::MapMethod method, const std::string &outer_policy,
    JincMapmaker &jinc_mm, NaiveMapmaker &naive_mm, MapBuffer &omb,
    MapBuffer &cmb, PtcScans &ptcs, CalibScans &calib_scans,
    ScanInput &scan_in, ScanOutput &scan_out, Telescope &telescope,
    const ArrayIdentity &array_identity, bool make_noise_maps,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps,
    const Logger &logger, Progress &&progress) {
    if (citlali::config::is_detector_map_grouping(grouping)) {
        for (std::size_t scan_index = 0; scan_index < ptcs.size();
             ++scan_index) {
            auto &ptc = ptcs[scan_index];
            auto &scan_apt = calib_scans[scan_index].apt;
            if (citlali::config::is_naive_map_method(method)) {
                naive_mm.populate_maps_naive_parallel(
                    ptc, omb, cmb, ptc.map_indices.data,
                    telescope.pixel_axes, scan_apt, telescope.d_fsmp, true,
                    make_noise_maps, active_maps);
            }
            else if (citlali::config::is_jinc_map_method(method)) {
                log_beammap_jinc_preflight(
                    ptc, array_identity, omb, jinc_mm, logger);
                jinc_mm.populate_maps_jinc_parallel(
                    ptc, omb, cmb, ptc.map_indices.data,
                    telescope.pixel_axes, scan_apt, telescope.d_fsmp, true,
                    make_noise_maps, active_maps);
            }
            progress();
        }
        return;
    }

    grppi::map(
        tula::grppi_utils::dyn_ex(outer_policy), scan_in, scan_out,
        [&](auto scan_index) {
            populate_naive_or_jinc_maps(
                method, naive_mm, jinc_mm, ptcs[scan_index], omb, cmb,
                ptcs[scan_index].map_indices.data, telescope.pixel_axes,
                calib_scans[scan_index].apt, telescope.d_fsmp, true,
                make_noise_maps);
            progress();
            return 0;
        });
}

}  // namespace citlali::pipeline

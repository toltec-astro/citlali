#pragma once

// Pointing map-population implementation detail.
// Include this only after Pointing has been declared.

#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_run_context.h>

template <class CalibScan>
void Pointing::populate_pointing_final_maps(
    TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    CalibScan &calib_scan,
    Eigen::VectorXI &map_indices,
    const std::string &map_grouping,
    citlali::config::MapMethod mapmaking_method,
    bool make_maps,
    bool make_noise_maps,
    const citlali::pipeline::NativeScienceProjection *
        native_projection,
    citlali::pipeline::OrderedWriter *jinc_merge_order) {
    if (!make_maps) {
        return;
    }

    bool run_omb = true;
    const bool run_noise_fruit =
        citlali::pipeline::should_populate_final_noise_maps(
            make_noise_maps,
            citlali::pipeline::fruit_loops_config(*this).enabled,
            !ptcproc.tod_mb.signal.empty());
    if (!native_projection) {
        apply_learned_mapmaking_detector_exclusions(
            ptcdata, calib_scan);
    }
    logger->info("populating maps");
    if (native_projection) {
        citlali::pipeline::populate_naive_or_jinc_maps_native(
            mapmaking_method, naive_mm, jinc_mm, ptcdata, omb, cmb,
            map_indices, telescope.pixel_axes, calib_scan.apt,
            telescope.d_fsmp, run_omb, run_noise_fruit,
            *native_projection, jinc_merge_order);
    }
    else {
        citlali::pipeline::populate_naive_or_jinc_maps(
            mapmaking_method, naive_mm, jinc_mm, ptcdata, omb, cmb,
            map_indices, telescope.pixel_axes, calib_scan.apt,
            telescope.d_fsmp, run_omb, run_noise_fruit,
            jinc_merge_order);
    }
}

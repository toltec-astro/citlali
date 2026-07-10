#pragma once

// Lali fruit-loop per-scan implementation detail.
// Include this only after Lali has been declared.

#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/timestream_run_context.h>

template <class CalibScan>
void Lali::maybe_subtract_lali_fruitloop_model(
    TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    CalibScan &calib_scan,
    Eigen::VectorXI &map_indices,
    const std::string &map_grouping,
    const citlali::pipeline::FruitLoopWeightPolicy &fruit_weight_policy) {
    // if running fruit loops and a map has been read in
    if (!fruit_weight_policy.use_noise_weights) {
        return;
    }

    logger->info("subtracting map from tod");
    // subtract map
    ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(
        ptcproc.tod_mb, ptcdata, calib_scan, map_indices,
        telescope.pixel_axes, map_grouping);
}

template <class CalibScan>
void Lali::run_lali_fruitloop_noise_pass(
    TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    CalibScan &calib_scan,
    Eigen::VectorXI &map_indices,
    const std::string &map_grouping,
    citlali::config::MapMethod mapmaking_method,
    bool make_maps,
    bool make_noise_maps,
    const citlali::pipeline::FruitLoopWeightPolicy &fruit_weight_policy) {
    // if running fruit loops and a map has been read in
    if (!fruit_weight_policy.use_noise_weights) {
        return;
    }

    // calculate weights
    logger->info("calculating weights for scan {} (fruit loops noise-only pass)",
                 ptcdata.index.data + 1);
    ptcproc.calc_weights(ptcdata, calib_scan.apt, telescope, true);

    // reset weights to median
    calib_scan = ptcproc.reset_weights(ptcdata, calib_scan, map_grouping);

    if (make_maps && make_noise_maps) {
        // populate noise maps only
        bool run_omb = false;
        logger->info("populating noise maps");
        citlali::pipeline::populate_naive_or_jinc_maps(
            mapmaking_method, naive_mm, jinc_mm, ptcdata, omb, cmb,
            map_indices, telescope.pixel_axes, calib_scan.apt,
            telescope.d_fsmp, run_omb, make_noise_maps);
    }
    logger->info("adding map to tod");
    // add map back
    ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(
        ptcproc.tod_mb, ptcdata, calib_scan, map_indices,
        telescope.pixel_axes, map_grouping);
}

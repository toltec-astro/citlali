#pragma once

// Pointing fruit-loop per-scan implementation detail.
// Include this only after Pointing has been declared.

#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/timestream_run_context.h>

template <class CalibScan>
void Pointing::maybe_subtract_pointing_fruitloop_model(
    TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    CalibScan &calib_scan,
    Eigen::VectorXI &map_indices,
    const std::string &map_grouping,
    const citlali::pipeline::FruitLoopWeightPolicy &fruit_weight_policy) {
    // if running fruit loops and a map has been read in
    if (!fruit_weight_policy.use_noise_weights) {
        return;
    }

    timestream::log_kernel_matrix_diag(
        logger, "ptc before fruitloops map subtraction", ptcdata.kernel.data, ptcdata.index.data);
    logger->info("subtracting map from tod");
    // subtract map
    ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(
        ptcproc.tod_mb, ptcdata, calib_scan, map_indices,
        telescope.pixel_axes, map_grouping);
    timestream::log_kernel_matrix_diag(
        logger, "ptc after fruitloops map subtraction", ptcdata.kernel.data, ptcdata.index.data);
}

template <class CalibScan>
void Pointing::run_pointing_fruitloop_noise_pass(
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

    // populate maps
    if (make_maps) {
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
    timestream::log_kernel_matrix_diag(
        logger, "ptc after fruitloops map addback", ptcdata.kernel.data, ptcdata.index.data);
}

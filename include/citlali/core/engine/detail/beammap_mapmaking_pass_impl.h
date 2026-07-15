#pragma once

// Beammap mapmaking stage implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/beammap_provenance_lifecycle.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>

void Beammap::normalize_beammap_maps_after_pass(
    const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps,
    const std::string &profile_context) {
    logger->info("normalizing maps");
    const auto normalize_profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.mapmaking.normalize", logger, profile_context);
    if (citlali::pipeline::raw_kernel_enabled(*this) &&
        !omb.grid_weight.empty()) {
        timestream::log_kernel_map_diag(
            logger,
            "beammap iter " + std::to_string(current_iter) + " before normalize",
            omb.kernel,
            active_maps,
            &omb.grid_weight);
    }
    omb.normalize_maps(active_maps);
    if (citlali::pipeline::raw_kernel_enabled(*this)) {
        timestream::log_kernel_map_diag(
            logger,
            "beammap iter " + std::to_string(current_iter) + " after normalize",
            omb.kernel,
            active_maps);
    }
    citlali::pipeline::log_beammap_normalize_support_summary(
        omb, calib, current_iter, logger);
}

template <class RandomBits, class Generator>
void Beammap::run_beammap_mapmaking_pass(bool update_progress,
                                         RandomBits &rands,
                                         Generator &eng,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    const auto &mapmaking = citlali::pipeline::mapmaking_config(*this);
    const auto mapmaking_grouping = mapmaking.grouping;
    const auto mapmaking_method = mapmaking.method;
    const auto active_maps =
        citlali::pipeline::select_unconverged_beammap_maps(
            mapmaking_grouping, converged, map_indices.n_maps, logger);
    const auto *active_maps_ptr = active_maps.ptr();

    std::ostringstream context;
    context << "iter=" << current_iter
            << " phase=" << beammap_iter_phase_name(current_iter)
            << " update_progress=" << (update_progress ? 1 : 0)
            << " grouping=" << static_cast<int>(mapmaking_grouping)
            << " method=" << static_cast<int>(mapmaking_method)
            << " active_maps=" << active_maps.n_active_maps << "/" << map_indices.n_maps;
    const auto profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.mapmaking.pass", logger, context.str());

    {
        const auto reset_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.mapmaking.reset_buffers", logger, context.str());
        citlali::pipeline::ensure_jinc_grid_weight_maps(
            mapmaking_method, omb, map_indices.n_maps, logger);

        citlali::pipeline::reset_beammap_mapmaking_buffers(
            omb, ptcs, map_indices.n_maps,
            citlali::pipeline::raw_kernel_enabled(*this),
            citlali::pipeline::noise_maps_enabled(*this),
            omb.randomize_dets, calib.n_dets, active_maps_ptr, rands,
            eng);
    }

    logger->info("running mapmaking");

    {
        const auto populate_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.mapmaking.populate", logger, context.str());

        populate_beammap_maps(
            mapmaking_grouping, mapmaking_method, active_maps_ptr,
            update_progress);
    }

    normalize_beammap_maps_after_pass(active_maps_ptr, context.str());
    citlali::pipeline::record_beammap_mapmaking_pass_completed_if_available(
        *this);
}

template <class RandomBits, class Generator>
void Beammap::run_beammap_mapmaking_stage(bool locator_iter,
                                          bool measurement_iter,
                                          bool detector_grouping,
                                          RandomBits &rands,
                                          Generator &eng,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    logger->info("starting mapmaking");

    if (!citlali::pipeline::mapmaking_enabled(*this)) {
        return;
    }

    const auto &scan_band_config =
        citlali::pipeline::beammap_config(*this).scan_band_mask;
    run_beammap_mapmaking_pass(true, rands, eng, stage_profile);

    if (scan_band_config.enabled && detector_grouping && locator_iter) {
        auto scan_band_summary = apply_scan_band_mask(omb);
        if (scan_band_summary.n_samples_flagged > 0) {
            logger->info(
                "beammap scan-band mask summary: flagged {} samples in {} rows across {} detectors ({} rejected by max_flagged_fraction={:.4f}); rebuilding maps",
                scan_band_summary.n_samples_flagged,
                scan_band_summary.n_rows_flagged,
                scan_band_summary.n_det_flagged,
                scan_band_summary.n_det_rejected,
                scan_band_config.max_flagged_fraction);
            run_beammap_mapmaking_pass(false, rands, eng, stage_profile);
        }
        else {
            logger->info(
                "beammap scan-band mask summary: no edge bands flagged ({} detectors rejected by max_flagged_fraction={:.4f})",
                scan_band_summary.n_det_rejected,
                scan_band_config.max_flagged_fraction);
        }
    }

    fit_beammap_maps(detector_grouping, measurement_iter, stage_profile);
}

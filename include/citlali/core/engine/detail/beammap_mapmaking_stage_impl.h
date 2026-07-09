#pragma once

// Beammap mapmaking stage implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/beammap_mapmaking_policy.h>
#include <citlali/core/pipeline/beammap_normalize_support_logging.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <sstream>

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

    if (citlali::config::is_detector_map_grouping(mapmaking_grouping)) {
        bool run_omb = true;
        for (std::size_t scan_vec_idx = 0; scan_vec_idx < ptcs.size(); ++scan_vec_idx) {
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

void Beammap::prepare_beammap_iteration_state(bool rerun_source_aware_rtc,
                                              bool measurement_iter,
                                              bool first_measurement_iter,
                                              bool detector_grouping) {
    ptcs = ptcs0;
    calib_scans = calib_scans0;

    const auto &rfi_config = typed_config.beammap.rfi_mask;
    if (rfi_config.enabled && detector_grouping &&
        rfi_mask_samples_flagged.size() == calib.n_dets &&
        rfi_mask_scans_flagged.size() == calib.n_dets) {
        rfi_mask_samples_flagged.setZero();
        rfi_mask_scans_flagged.setZero();
    }

    const bool skip_centered_kernel_map_feedback = rerun_source_aware_rtc;
    ptcproc.fruit_loops_kernel_feedback_enabled = !skip_centered_kernel_map_feedback;
    if (skip_centered_kernel_map_feedback) {
        logger->info(
            "beammap detector kernel map feedback disabled on iter {} while building the first source-aware kernel map",
            current_iter);
    }

    // copy previous-iteration maps for source-aperture convergence tests
    const auto &iteration_config = typed_config.beammap.iteration;
    if (citlali::pipeline::mapmaking_enabled(*this) &&
        iteration_config.tolerance > 0.0 &&
        measurement_iter) {
        omb_copy.signal = omb.signal;
        omb_copy.weight = omb.weight;
    }

    if (ptcproc.run_fruit_loops) {
        if (first_measurement_iter && !omb.noise.empty()) {
            omb.calc_median_rms();
        }
        if (measurement_iter) {
            ptcproc.configure_fruit_loops_adaptive_gate(
                omb, calib, citlali::pipeline::active_map_grouping_name(*this),
                false);
        }
    }
}

template <class RandomBits, class Generator>
void Beammap::run_beammap_mapmaking_pass(bool update_progress,
                                         RandomBits &rands,
                                         Generator &eng) {
    const auto mapmaking_grouping = typed_config.mapmaking.grouping;
    const auto mapmaking_method = typed_config.mapmaking.method;
    const auto active_maps =
        citlali::pipeline::select_unconverged_beammap_maps(
            mapmaking_grouping, converged, n_maps, logger);
    const auto *active_maps_ptr = active_maps.ptr();

    std::ostringstream context;
    context << "iter=" << current_iter
            << " phase=" << beammap_iter_phase_name(current_iter)
            << " update_progress=" << (update_progress ? 1 : 0)
            << " grouping=" << static_cast<int>(mapmaking_grouping)
            << " method=" << static_cast<int>(mapmaking_method)
            << " active_maps=" << active_maps.n_active_maps << "/" << n_maps;
    const auto profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.mapmaking.pass", logger, context.str());

    {
        const auto reset_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.mapmaking.reset_buffers", logger, context.str());
        citlali::pipeline::ensure_jinc_grid_weight_maps(
            mapmaking_method, omb, n_maps, logger);

        citlali::pipeline::reset_beammap_mapmaking_buffers(
            omb, ptcs, n_maps, rtcproc.run_kernel,
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

    logger->info("normalizing maps");
    {
        const auto normalize_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.mapmaking.normalize", logger, context.str());
        if (rtcproc.run_kernel && !omb.grid_weight.empty()) {
            timestream::log_kernel_map_diag(
                logger,
                "beammap iter " + std::to_string(current_iter) + " before normalize",
                omb.kernel,
                active_maps_ptr,
                &omb.grid_weight);
        }
        omb.normalize_maps(active_maps_ptr);
        if (rtcproc.run_kernel) {
            timestream::log_kernel_map_diag(
                logger,
                "beammap iter " + std::to_string(current_iter) + " after normalize",
                omb.kernel,
                active_maps_ptr);
        }
        citlali::pipeline::log_beammap_normalize_support_summary(
            omb, calib, current_iter, logger);
    }
}

template <class RandomBits, class Generator>
void Beammap::run_beammap_mapmaking_stage(bool locator_iter,
                                          bool measurement_iter,
                                          bool detector_grouping,
                                          RandomBits &rands,
                                          Generator &eng) {
    logger->info("starting mapmaking");

    if (!citlali::pipeline::mapmaking_enabled(*this)) {
        return;
    }

    const auto &scan_band_config = typed_config.beammap.scan_band_mask;
    run_beammap_mapmaking_pass(true, rands, eng);

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
            run_beammap_mapmaking_pass(false, rands, eng);
        }
        else {
            logger->info(
                "beammap scan-band mask summary: no edge bands flagged ({} detectors rejected by max_flagged_fraction={:.4f})",
                scan_band_summary.n_det_rejected,
                scan_band_config.max_flagged_fraction);
        }
    }

    fit_beammap_maps(detector_grouping, measurement_iter);
}

template <class KidsProc, class RawObs>
bool Beammap::maybe_run_beammap_source_aware_rtc(KidsProc &kidsproc,
                                                 RawObs &rawobs,
                                                 bool first_measurement_iter,
                                                 bool detector_grouping) {
    configure_detector_source_centers_from_previous_fit();

    const bool detector_kernel_source_centers_active =
        detector_grouping &&
        rtcproc.run_kernel &&
        rtcproc.kernel.has_source_centers();
    const bool rerun_source_aware_rtc =
        first_measurement_iter && detector_kernel_source_centers_active;
    if (!rerun_source_aware_rtc) {
        return false;
    }

    logger->info(
        "beammap iter {} rerunning RTC with previous-fit detector source centers; regular RTC TOD output disabled for this internal pass",
        current_iter);
    const auto profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.rtc.source_aware_rerun", logger,
            "iter=" + std::to_string(current_iter));
    timestream_pipeline(kidsproc, rawobs, false);
    return true;
}

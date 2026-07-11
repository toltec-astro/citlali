#pragma once

// Beammap PTC cleaning implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/beammap_mapmaking_policy.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/stage_profile.h>

bool Beammap::subtract_beammap_model_for_ptc_scan(int scan_index,
                                                  bool measurement_iter) {
    if (!citlali::pipeline::mapmaking_enabled(*this) || !measurement_iter) {
        return false;
    }
    auto map_grouping =
        citlali::pipeline::active_map_grouping_name(*this);
    if (!citlali::pipeline::fruit_loops_config(*this).enabled) {
        logger->info("subtracting gaussian from tod");
        ptcproc.add_gaussian<timestream::TCProc::SourceType::NegativeGaussian>(
            ptcs[scan_index], params, telescope.pixel_axes, map_grouping,
            calib.apt, omb.pixel_size_rad, omb.n_rows, omb.n_cols);
        return false;
    }

    logger->info("subtracting map from tod");
    ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(
        omb, ptcs[scan_index], calib, ptcs[scan_index].map_indices.data,
        telescope.pixel_axes, map_grouping);
    return true;
}

void Beammap::restore_beammap_model_for_ptc_scan(int scan_index,
                                                 bool measurement_iter) {
    if (!citlali::pipeline::mapmaking_enabled(*this) || !measurement_iter) {
        return;
    }
    auto map_grouping =
        citlali::pipeline::active_map_grouping_name(*this);
    if (!citlali::pipeline::fruit_loops_config(*this).enabled) {
        logger->info("adding gaussian to tod");
        ptcproc.add_gaussian<timestream::TCProc::SourceType::Gaussian>(
            ptcs[scan_index], params, telescope.pixel_axes, map_grouping,
            calib.apt, omb.pixel_size_rad, omb.n_rows, omb.n_cols);
        return;
    }

    logger->info("adding map to tod");
    ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(
        omb, ptcs[scan_index], calib, ptcs[scan_index].map_indices.data,
        telescope.pixel_axes, map_grouping);
}

void Beammap::remove_bad_beammap_dets_for_scan(int scan_index,
                                               bool locator_iter,
                                               bool detector_grouping) {
    // For detector-grouped beammaps, keep the locator pass permissive so
    // bright-source scans are less likely to be rejected before we have
    // any source-location estimate to feed back into later iterations.
    if (detector_grouping && locator_iter) {
        logger->info(
            "skipping remove_bad_dets on beammap locator iter {} for detector scan {}",
            current_iter, ptcs[scan_index].index.data + 1);
        return;
    }

    auto map_grouping =
        citlali::pipeline::active_map_grouping_name(*this);
    calib_scans[scan_index] = ptcproc.remove_bad_dets(
        ptcs[scan_index], calib_scans[scan_index], map_grouping);
}

void Beammap::apply_beammap_ptc_scan_weights(int scan_index,
                                             bool measurement_iter,
                                             bool detector_grouping) {
    auto map_grouping =
        citlali::pipeline::active_map_grouping_name(*this);
    if (detector_grouping) {
        const auto &rfi_config =
            citlali::pipeline::beammap_config(*this).rfi_mask;
        auto rfi_summary = apply_rfi_sample_mask(ptcs[scan_index]);
        if (rfi_config.enabled) {
            if (rfi_summary.n_samples_flagged > 0 ||
                rfi_summary.n_det_rejected > 0) {
                logger->info(
                    "beammap rfi mask scan {}: masked {} samples across {}/{} detectors ({} rejected by max_flagged_fraction={:.4f})",
                    ptcs[scan_index].index.data + 1,
                    rfi_summary.n_samples_flagged,
                    rfi_summary.n_det_flagged,
                    rfi_summary.n_det_candidates,
                    rfi_summary.n_det_rejected,
                    rfi_config.max_flagged_fraction);
            }
            else {
                logger->debug(
                    "beammap rfi mask scan {}: no samples masked",
                    ptcs[scan_index].index.data + 1);
            }
        }

        const auto detector_weighting_mode =
            citlali::pipeline::beammap_config(*this)
                .detector_weighting_mode;
        const auto detector_weighting_mode_name =
            citlali::config::to_string(detector_weighting_mode);
        const bool use_ptc_weights =
            citlali::pipeline::use_beammap_detector_ptc_weights(
                detector_weighting_mode, measurement_iter);
        if (use_ptc_weights) {
            logger->info(
                "calculating detector-mode PTC weights for scan {} (mode={})",
                ptcs[scan_index].index.data + 1,
                detector_weighting_mode_name);
            ptcproc.calc_weights(ptcs[scan_index], calib_scans[scan_index].apt, telescope);
            calib_scans[scan_index] = ptcproc.reset_weights(
                ptcs[scan_index], calib_scans[scan_index], map_grouping);
        }
        else {
            // Constant weights remain the safest default for bright beammaps.
            ptcs[scan_index].weights.data.resize(ptcs[scan_index].scans.data.cols());
            ptcs[scan_index].weights.data.setOnes();
        }
        return;
    }

    logger->info("calculating weights for scan {}", ptcs[scan_index].index.data + 1);
    ptcproc.calc_weights(ptcs[scan_index], calib_scans[scan_index].apt, telescope);
    calib_scans[scan_index] = ptcproc.reset_weights(
        ptcs[scan_index], calib_scans[scan_index], map_grouping);
}

void Beammap::process_beammap_ptc_scan(
    int scan_index, bool locator_iter, bool measurement_iter,
    bool detector_grouping,
    const std::shared_ptr<std::mutex> &ptc_line_audit_mutex) {
    const bool model_subtracted_for_ptc_line_audit =
        subtract_beammap_model_for_ptc_scan(scan_index, measurement_iter);

    {
        std::lock_guard<std::mutex> lock(*ptc_line_audit_mutex);
        apply_model_protected_ptc_line_audit(
            ptcs[scan_index], calib_scans[scan_index],
            model_subtracted_for_ptc_line_audit);
    }

    logger->info("processed time chunk processing for scan {}", scan_index + 1);
    auto map_grouping =
        citlali::pipeline::active_map_grouping_name(*this);
    ptcproc.run(
        ptcs[scan_index], ptcs[scan_index], calib_scans[scan_index],
        telescope.pixel_axes, map_grouping);

    restore_beammap_model_for_ptc_scan(scan_index, measurement_iter);
    remove_bad_beammap_dets_for_scan(scan_index, locator_iter, detector_grouping);
    apply_beammap_ptc_scan_weights(scan_index, measurement_iter, detector_grouping);

    logger->debug("calculating stats");
    diagnostics.calc_stats(ptcs[scan_index]);
}

void Beammap::run_beammap_ptc_cleaning_pass(bool locator_iter,
                                            bool measurement_iter,
                                            bool detector_grouping) {
    auto ptc_line_audit_mutex = std::make_shared<std::mutex>();

    const auto profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.ptc.cleaning", logger,
            "iter=" + std::to_string(current_iter) +
                " phase=" + beammap_iter_phase_name(current_iter));
    grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
        process_beammap_ptc_scan(
            i, locator_iter, measurement_iter, detector_grouping,
            ptc_line_audit_mutex);
        return 0;
    });
}

#pragma once

// Beammap PTC output implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_ptc_product_output_helpers.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/stage_profile.h>

void Beammap::clear_beammap_ptc_diagnostics() {
    for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
        ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
    }
}

void Beammap::write_beammap_ptc_chunk_summaries(
    int output_iter,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    if (!citlali::pipeline::verbose_runtime_enabled(*this)) {
        return;
    }
    const auto profile_scope =
        citlali::pipeline::profile_stage(stage_profile,
            "beammap.ptc_output.chunk_summaries", logger,
            "iter=" + std::to_string(output_iter));
    logger->debug(
        "writing chunk summaries for beammap PTC iteration {}",
        output_iter);
    for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
        write_chunk_summary(ptcs[i]);
    }
}

void Beammap::write_beammap_ptc_diag_sidecar(
    int output_iter,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    if (output_paths.ptcdiag_filename.empty()) {
        return;
    }
    const auto profile_scope =
        citlali::pipeline::profile_stage(stage_profile,
            "beammap.ptc_output.diag_sidecar", logger,
            "iter=" + std::to_string(output_iter) +
                " scans=" + std::to_string(telescope.scan_indices.cols()));
    logger->info(
        "writing ptc diagnostics sidecar chunks for beammap iteration {}",
        output_iter);
    for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
        ptcproc.append_diag_to_netcdf(
            ptcs[i], output_paths.ptcdiag_filename, calib_scans[i],
            ptcs[i].index.data);
        if (!citlali::pipeline::processed_tod_output_files_available(*this)) {
            ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
        }
    }
}

void Beammap::write_beammap_processed_ptc_tod(
    int output_iter,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    if (!citlali::pipeline::processed_tod_output_files_available(*this)) {
        return;
    }
    const auto profile_scope =
        citlali::pipeline::profile_stage(stage_profile,
            "beammap.ptc_output.processed_tod", logger,
            "iter=" + std::to_string(output_iter) +
                " scans=" + std::to_string(telescope.scan_indices.cols()));
    logger->info(
        "writing processed time chunk for beammap iteration {}",
        output_iter);
    beammap_ptc_product_output_helpers::update_ptc_tod_fruitloops_iter(
        output_paths.tod_filename, output_iter, logger);
    const auto map_grouping =
        citlali::pipeline::active_map_grouping_name(*this);
    for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
        const auto ptc_scan_row = tod_output_scan_row(
            i, citlali::config::TodOutputStream::ptc);
        if (ptc_scan_row < 0) {
            continue;
        }
        ptcproc.append_to_netcdf(
            ptcs[i], output_paths.tod_filename["ptc"], map_grouping,
            telescope.pixel_axes, ptcs[i].pointing_offsets_arcsec.data,
            calib_scans[i], true, ptc_scan_row,
            citlali::pipeline::processed_tod_mini_output(*this));
        ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
    }
}

void Beammap::write_beammap_detector_ptc_tod_stage(
    int output_iter,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    const auto profile_scope =
        citlali::pipeline::profile_stage(stage_profile,
            "beammap.ptc_output.detector_tod", logger,
            "iter=" + std::to_string(output_iter));
    write_detector_specific_ptc_tod(output_iter);
}

void Beammap::write_beammap_ptc_products(
    int output_iter,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    const auto total_profile_scope =
        citlali::pipeline::profile_stage(stage_profile,
            "beammap.ptc_output.total", logger,
            "iter=" + std::to_string(output_iter));
    write_beammap_ptc_chunk_summaries(output_iter, stage_profile);
    write_beammap_ptc_diag_sidecar(output_iter, stage_profile);
    write_beammap_processed_ptc_tod(output_iter, stage_profile);
    write_beammap_detector_ptc_tod_stage(output_iter, stage_profile);
}

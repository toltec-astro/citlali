#pragma once

// Beammap PTC output implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_ptc_product_output_helpers.h>
#include <citlali/core/pipeline/stage_profile.h>

void Beammap::clear_beammap_ptc_diagnostics() {
    for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
        ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
    }
}

void Beammap::write_beammap_ptc_products(int output_iter) {
    const auto total_profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.ptc_output.total", logger,
            "iter=" + std::to_string(output_iter));
    if (verbose_mode) {
        const auto profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.ptc_output.chunk_summaries", logger,
                "iter=" + std::to_string(output_iter));
        logger->debug(
            "writing chunk summaries for beammap PTC iteration {}",
            output_iter);
        for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
            write_chunk_summary(ptcs[i]);
        }
    }
    if (!ptcdiag_filename.empty()) {
        const auto profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.ptc_output.diag_sidecar", logger,
                "iter=" + std::to_string(output_iter) +
                    " scans=" + std::to_string(telescope.scan_indices.cols()));
        logger->info(
            "writing ptc diagnostics sidecar chunks for beammap iteration {}",
            output_iter);
        for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
            ptcproc.append_diag_to_netcdf(
                ptcs[i], ptcdiag_filename, calib_scans[i],
                ptcs[i].index.data);
            if (!(run_tod_output && run_tod_output_ptc &&
                  !tod_filename.empty())) {
                ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
            }
        }
    }
    if (run_tod_output && run_tod_output_ptc && !tod_filename.empty()) {
        const auto profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.ptc_output.processed_tod", logger,
                "iter=" + std::to_string(output_iter) +
                    " scans=" + std::to_string(telescope.scan_indices.cols()));
        logger->info(
            "writing processed time chunk for beammap iteration {}",
            output_iter);
        beammap_ptc_product_output_helpers::update_ptc_tod_fruitloops_iter(
            tod_filename, output_iter, logger);
        for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
            const auto ptc_scan_row = tod_output_scan_row(i, "ptc");
            if (ptc_scan_row < 0) {
                continue;
            }
            ptcproc.append_to_netcdf(
                ptcs[i], tod_filename["ptc"], map_grouping,
                telescope.pixel_axes, ptcs[i].pointing_offsets_arcsec.data,
                calib_scans[i], true, ptc_scan_row);
            ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
        }
    }
    {
        const auto profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.ptc_output.detector_tod", logger,
                "iter=" + std::to_string(output_iter));
        write_detector_specific_ptc_tod(output_iter);
    }
}

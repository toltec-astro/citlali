#pragma once

// Beammap PTC output implementation detail.
// Include this only after Beammap has been declared.

void Beammap::clear_beammap_ptc_diagnostics() {
    for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
        ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
    }
}

void Beammap::write_beammap_ptc_products(int output_iter) {
    if (verbose_mode) {
        logger->debug(
            "writing chunk summaries for beammap PTC iteration {}",
            output_iter);
        for (Eigen::Index i = 0; i < telescope.scan_indices.cols(); ++i) {
            write_chunk_summary(ptcs[i]);
        }
    }
    if (!ptcdiag_filename.empty()) {
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
        logger->info(
            "writing processed time chunk for beammap iteration {}",
            output_iter);
        auto ptc_filename_it = tod_filename.find("ptc");
        if (ptc_filename_it != tod_filename.end() &&
            !ptc_filename_it->second.empty()) {
            try {
                netCDF::NcFile ptc_tod_file(
                    ptc_filename_it->second, netCDF::NcFile::write);
                netCDF::NcVar fruit_iter_var =
                    ptc_tod_file.getVar("FRUITLOOPS_ITER");
                if (!fruit_iter_var.isNull()) {
                    fruit_iter_var.putVar(&output_iter);
                }
                else {
                    logger->warn("PTC TOD file {} has no FRUITLOOPS_ITER variable",
                                 ptc_filename_it->second);
                }
            } catch (const std::exception &e) {
                logger->warn(
                    "failed to update PTC TOD FRUITLOOPS_ITER in {}: {}",
                    ptc_filename_it->second, e.what());
            }
        }
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
    write_detector_specific_ptc_tod(output_iter);
}

#pragma once

// Included by stats_netcdf.h inside namespace citlali::pipeline.

template <class Calib, class Diagnostics, class Cleaner, class Logger>
void add_stats_file_outputs(netCDF::NcFile &fo, int obsnum,
                            const Calib &calib, Diagnostics &diagnostics,
                            const Cleaner &cleaner, const Logger &logger,
                            const std::string &signal_unit,
                            Eigen::Index n_stats_chunks,
                            double eigenvalue_fill_value) {
    add_obsnum_var(fo, obsnum);

    const auto stats_dims =
        add_stats_dims(fo, calib.n_dets, calib.n_arrays, n_stats_chunks);
    const auto det_stats_header_units = detector_stats_units(signal_unit);
    const auto grp_stats_header_units = group_stats_units(signal_unit);

    add_detector_stats_vars(
        fo, diagnostics, stats_dims.det_stat, det_stats_header_units);
    add_group_stats_vars(
        fo, diagnostics, stats_dims.grp_stat, grp_stats_header_units);
    add_stats_apt_double_vars(fo, calib, stats_dims.n_dets);
    add_stats_adc_snap_vars(fo, calib, diagnostics.adc_snap_data);
    add_stats_eigenvalue_outputs_if_needed(
        fo, diagnostics, cleaner, logger, eigenvalue_fill_value);
}


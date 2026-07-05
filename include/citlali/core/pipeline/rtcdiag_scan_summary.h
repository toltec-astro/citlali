#pragma once

// Included by rtcdiag_netcdf.h inside namespace citlali::pipeline.

inline double rtcdiag_percentile_sorted(
    const std::vector<double> &sorted_values, double pct) {
    if (sorted_values.empty()) {
        return rtcdiag_fill_double();
    }
    if (sorted_values.size() == 1) {
        return sorted_values.front();
    }
    pct = std::min(100.0, std::max(0.0, pct));
    const double pos =
        (pct / 100.0) * static_cast<double>(sorted_values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}

inline void add_rtcdiag_scan_double(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment, netCDF::NcDim n_scans_dim,
    const std::vector<std::size_t> &scan_chunks,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, n_scans_dim);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, scan_chunks, 1);
    v.putVar(values.data());
}

inline void add_rtcdiag_scan_array_double(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment,
    const std::vector<netCDF::NcDim> &scan_array_dims,
    const std::vector<std::size_t> &scan_array_chunks,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, scan_array_dims);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, scan_array_chunks, 1);
    v.putVar(values.data());
}

struct RtcDiagScanDoubleValues {
    const std::vector<double> &scan_duration_s;
    const std::vector<double> &scan_speed_p50_arcsec_s;
    const std::vector<double> &scan_speed_p95_arcsec_s;
    const std::vector<double> &scan_speed_p995_arcsec_s;
};

template <class AddDouble>
void add_rtcdiag_scan_summary_vars(
    const AddDouble &add_double, const RtcDiagScanDoubleValues &values) {
    add_double("scan_duration_s", "s",
               "inner scan duration used for scan-speed diagnostics",
               values.scan_duration_s);
    add_double("scan_speed_altaz_p50_arcsec_s", "arcsec/s",
               "per-scan median boresight speed in the delta-source altaz frame",
               values.scan_speed_p50_arcsec_s);
    add_double("scan_speed_altaz_p95_arcsec_s", "arcsec/s",
               "per-scan 95th percentile boresight speed in the delta-source altaz frame",
               values.scan_speed_p95_arcsec_s);
    add_double("scan_speed_altaz_p995_arcsec_s", "arcsec/s",
               "per-scan robust peak (99.5th percentile) boresight speed in the delta-source altaz frame",
               values.scan_speed_p995_arcsec_s);
}

struct RtcDiagScanArrayDoubleValues {
    const std::vector<double> &source_power_half_bandwidth_hz;
    const std::vector<double> &tod_lowpass_to_source_power_half_ratio;
};

template <class AddDouble>
void add_rtcdiag_scan_array_summary_vars(
    const AddDouble &add_double,
    const RtcDiagScanArrayDoubleValues &values) {
    add_double("scan_source_power_half_bandwidth_hz", "Hz",
               "Gaussian compact-source temporal power half-bandwidth from scan_speed_altaz_p995_arcsec_s and array mean FWHM",
               values.source_power_half_bandwidth_hz);
    add_double("scan_tod_lowpass_to_source_power_half_ratio", "N/A",
               "configured RTC FIR low-pass cutoff divided by scan_source_power_half_bandwidth_hz; values much larger than 1 indicate extra high-frequency noise admitted relative to compact-source half-power bandwidth",
               values.tod_lowpass_to_source_power_half_ratio);
}

struct RtcDiagScanSummaryData {
    std::vector<double> scan_duration_s;
    std::vector<double> scan_speed_p50_arcsec_s;
    std::vector<double> scan_speed_p95_arcsec_s;
    std::vector<double> scan_speed_p995_arcsec_s;
};

template <class Telescope, class Logger>
RtcDiagScanSummaryData calculate_rtcdiag_scan_summary(
    const Telescope &telescope, Eigen::Index n_scans,
    std::size_t n_scan_values, double rad_to_arcsec, double fill_double,
    const Logger &logger) {
    RtcDiagScanSummaryData values{
        std::vector<double>(n_scan_values, fill_double),
        std::vector<double>(n_scan_values, fill_double),
        std::vector<double>(n_scan_values, fill_double),
        std::vector<double>(n_scan_values, fill_double)};
    constexpr double max_tel_sample_step_s = 0.1;
    constexpr double max_pointing_step_rad = 0.01;

    const auto tel_time_it = telescope.tel_data.find("TelTime");
    const auto az_it = telescope.tel_data.find("az_phys");
    const auto alt_it = telescope.tel_data.find("alt_phys");
    const bool has_telescope_motion_data =
        tel_time_it != telescope.tel_data.end() &&
        az_it != telescope.tel_data.end() &&
        alt_it != telescope.tel_data.end();
    if (!has_telescope_motion_data) {
        logger->warn(
            "rtcdiag scan-speed diagnostics skipped: missing TelTime, "
            "az_phys, or alt_phys telescope data");
        return values;
    }

    const auto &tel_time = tel_time_it->second;
    const auto &az_phys = az_it->second;
    const auto &alt_phys = alt_it->second;
    const Eigen::Index n_tel =
        std::min({tel_time.size(), az_phys.size(), alt_phys.size()});
    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        const auto scan_index = static_cast<std::size_t>(scan);
        const Eigen::Index start =
            std::max<Eigen::Index>(0, telescope.scan_indices(0, scan));
        const Eigen::Index stop =
            std::min<Eigen::Index>(n_tel - 1,
                                   telescope.scan_indices(1, scan));
        const bool has_valid_scan_bounds =
            stop > start && start >= 0 && stop < n_tel;
        if (!has_valid_scan_bounds) {
            continue;
        }
        const double duration = tel_time(stop) - tel_time(start);
        if (std::isfinite(duration) && duration > 0.0) {
            values.scan_duration_s[scan_index] = duration;
        }
        const auto n_scan_samples = std::max<Eigen::Index>(stop - start, 0);
        std::vector<double> speed_arcsec_s;
        speed_arcsec_s.reserve(static_cast<std::size_t>(n_scan_samples));
        for (Eigen::Index i = start; i < stop; ++i) {
            const double dt = tel_time(i + 1) - tel_time(i);
            const double daz = az_phys(i + 1) - az_phys(i);
            const double dalt = alt_phys(i + 1) - alt_phys(i);
            if (!std::isfinite(dt) || !std::isfinite(daz) ||
                !std::isfinite(dalt) || dt <= 0.0 ||
                dt > max_tel_sample_step_s ||
                std::abs(daz) > max_pointing_step_rad ||
                std::abs(dalt) > max_pointing_step_rad) {
                continue;
            }
            speed_arcsec_s.push_back(
                std::hypot(daz, dalt) / dt * rad_to_arcsec);
        }
        if (!speed_arcsec_s.empty()) {
            std::sort(speed_arcsec_s.begin(), speed_arcsec_s.end());
            values.scan_speed_p50_arcsec_s[scan_index] =
                rtcdiag_percentile_sorted(speed_arcsec_s, 50.0);
            values.scan_speed_p95_arcsec_s[scan_index] =
                rtcdiag_percentile_sorted(speed_arcsec_s, 95.0);
            values.scan_speed_p995_arcsec_s[scan_index] =
                rtcdiag_percentile_sorted(speed_arcsec_s, 99.5);
        }
    }
    return values;
}

inline void add_rtcdiag_scan_summary_outputs(
    netCDF::NcFile &fo, netCDF::NcDim n_scans_dim,
    const std::vector<std::size_t> &scan_chunks,
    const RtcDiagScanSummaryData &values) {
    auto add_scan_double = [&](const std::string &name,
                               const std::string &units,
                               const std::string &comment,
                               const std::vector<double> &data) {
        add_rtcdiag_scan_double(
            fo, name, units, comment, n_scans_dim, scan_chunks, data);
    };
    add_rtcdiag_scan_summary_vars(
        add_scan_double,
        {values.scan_duration_s,
         values.scan_speed_p50_arcsec_s,
         values.scan_speed_p95_arcsec_s,
         values.scan_speed_p995_arcsec_s});
}

struct RtcDiagScanArraySummaryData {
    std::vector<double> source_power_half_bandwidth_hz;
    std::vector<double> tod_lowpass_to_source_power_half_ratio;
};

template <class Calib, class RtcProc>
RtcDiagScanArraySummaryData calculate_rtcdiag_scan_array_summary(
    const Calib &calib, const RtcProc &rtcproc,
    const std::vector<double> &scan_speed_p995_arcsec_s,
    Eigen::Index n_scans, std::size_t n_array_values,
    std::size_t n_scan_array_values, double pi_value, double fwhm_to_std,
    double fill_double) {
    RtcDiagScanArraySummaryData values{
        std::vector<double>(n_scan_array_values, fill_double),
        std::vector<double>(n_scan_array_values, fill_double)};

    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        const auto scan_index = static_cast<std::size_t>(scan);
        const double speed = scan_speed_p995_arcsec_s[scan_index];
        if (!std::isfinite(speed) || speed <= 0.0) {
            continue;
        }
        for (Eigen::Index arr_i = 0; arr_i < calib.n_arrays; ++arr_i) {
            const Eigen::Index array = calib.arrays(arr_i);
            const auto fwhm_it = calib.array_fwhms.find(array);
            if (fwhm_it == calib.array_fwhms.end()) {
                continue;
            }
            const double fwhm_arcsec =
                0.5 * (std::get<0>(fwhm_it->second) +
                       std::get<1>(fwhm_it->second));
            if (!std::isfinite(fwhm_arcsec) || fwhm_arcsec <= 0.0) {
                continue;
            }
            const double f_half_hz =
                (std::sqrt(std::log(2.0)) /
                 (2.0 * pi_value * fwhm_arcsec * fwhm_to_std)) *
                speed;
            const auto flat_i = scan_index * n_array_values +
                                static_cast<std::size_t>(arr_i);
            values.source_power_half_bandwidth_hz[flat_i] = f_half_hz;
            const bool has_lowpass_ratio =
                rtcproc.run_tod_filter &&
                rtcproc.filter.freq_high_Hz > 0.0 && f_half_hz > 0.0;
            if (has_lowpass_ratio) {
                values.tod_lowpass_to_source_power_half_ratio[flat_i] =
                    rtcproc.filter.freq_high_Hz / f_half_hz;
            }
        }
    }
    return values;
}

inline void add_rtcdiag_scan_array_summary_outputs(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &scan_array_dims,
    const std::vector<std::size_t> &scan_array_chunks,
    const RtcDiagScanArraySummaryData &values) {
    auto add_scan_array_double = [&](const std::string &name,
                                     const std::string &units,
                                     const std::string &comment,
                                     const std::vector<double> &data) {
        add_rtcdiag_scan_array_double(
            fo, name, units, comment, scan_array_dims, scan_array_chunks,
            data);
    };
    add_rtcdiag_scan_array_summary_vars(
        add_scan_array_double,
        {values.source_power_half_bandwidth_hz,
         values.tod_lowpass_to_source_power_half_ratio});
}


#pragma once

// Included by rtcdiag_netcdf.h inside namespace citlali::pipeline.

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
    const std::vector<double> &scan_speed_max_arcsec_s;
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
    add_double("scan_speed_altaz_max_arcsec_s", "arcsec/s",
               "maximum valid in-scan boresight speed after the one-telescope-sample boundary guard; this is the Stage A sampling authority",
               values.scan_speed_max_arcsec_s);
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
    const std::vector<double> &beam_major_fwhm_arcsec;
    const std::vector<double> &beam_minor_fwhm_arcsec;
    const std::vector<double> &beam_position_angle_rad;
    const std::vector<double> &limiting_projected_fwhm_arcsec;
    const std::vector<double> &limiting_speed_arcsec_s;
};

template <class AddDouble>
void add_rtcdiag_scan_array_summary_vars(
    const AddDouble &add_double,
    const RtcDiagScanArrayDoubleValues &values) {
    add_double("scan_source_power_half_bandwidth_hz", "Hz",
               "Gaussian compact-source temporal power half-bandwidth from the limiting valid speed and scan-projected elliptical array beam",
               values.source_power_half_bandwidth_hz);
    add_double("scan_tod_lowpass_to_source_power_half_ratio", "N/A",
               "configured RTC FIR low-pass cutoff divided by scan_source_power_half_bandwidth_hz; values much larger than 1 indicate extra high-frequency noise admitted relative to compact-source half-power bandwidth",
               values.tod_lowpass_to_source_power_half_ratio);
    add_double("scan_array_beam_major_fwhm_arcsec", "arcsec",
               "array mean major-axis beam FWHM from the admitted APT beam authority",
               values.beam_major_fwhm_arcsec);
    add_double("scan_array_beam_minor_fwhm_arcsec", "arcsec",
               "array mean minor-axis beam FWHM from the admitted APT beam authority",
               values.beam_minor_fwhm_arcsec);
    add_double("scan_array_beam_position_angle_rad", "rad",
               "array mean beam position angle used for scan projection",
               values.beam_position_angle_rad);
    add_double("scan_array_limiting_projected_fwhm_arcsec", "arcsec",
               "scan-projected beam FWHM at the valid interval with the shortest beam crossing time",
               values.limiting_projected_fwhm_arcsec);
    add_double("scan_array_limiting_speed_arcsec_s", "arcsec/s",
               "valid telescope speed at the interval with the shortest scan-projected beam crossing time",
               values.limiting_speed_arcsec_s);
}

struct RtcDiagScanSummaryData {
    std::vector<double> scan_duration_s;
    std::vector<double> scan_speed_max_arcsec_s;
    std::vector<double> scan_speed_p50_arcsec_s;
    std::vector<double> scan_speed_p95_arcsec_s;
    std::vector<double> scan_speed_p995_arcsec_s;
    std::vector<RtcSamplingScanMotion> scan_motion;
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
        std::vector<double>(n_scan_values, fill_double),
        std::vector<double>(n_scan_values, fill_double),
        std::vector<RtcSamplingScanMotion>(n_scan_values)};
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
        const auto motion = calculate_rtc_sampling_scan_motion(
            tel_time, az_phys, alt_phys, start, stop, rad_to_arcsec,
            max_tel_sample_step_s, max_pointing_step_rad, 1);
        values.scan_motion[scan_index] = motion;
        if (std::isfinite(motion.duration_s)) {
            values.scan_duration_s[scan_index] = motion.duration_s;
        }
        if (!motion.intervals.empty()) {
            values.scan_speed_max_arcsec_s[scan_index] =
                motion.speed_max_arcsec_s;
            values.scan_speed_p50_arcsec_s[scan_index] =
                motion.speed_p50_arcsec_s;
            values.scan_speed_p95_arcsec_s[scan_index] =
                motion.speed_p95_arcsec_s;
            values.scan_speed_p995_arcsec_s[scan_index] =
                motion.speed_p995_arcsec_s;
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
         values.scan_speed_max_arcsec_s,
         values.scan_speed_p50_arcsec_s,
         values.scan_speed_p95_arcsec_s,
         values.scan_speed_p995_arcsec_s});
}

struct RtcDiagScanArraySummaryData {
    std::vector<double> source_power_half_bandwidth_hz;
    std::vector<double> tod_lowpass_to_source_power_half_ratio;
    std::vector<double> beam_major_fwhm_arcsec;
    std::vector<double> beam_minor_fwhm_arcsec;
    std::vector<double> beam_position_angle_rad;
    std::vector<double> limiting_projected_fwhm_arcsec;
    std::vector<double> limiting_speed_arcsec_s;
    std::vector<int> candidate_factors;
    std::vector<double> fir_coefficients;
    std::vector<int> candidate_status;
    std::vector<double> candidate_output_sample_rate_hz;
    std::vector<double> candidate_output_nyquist_hz;
    std::vector<double> candidate_samples_per_fwhm;
    std::vector<double> candidate_beam_peak_attenuation_fraction;
    std::vector<double> candidate_beam_half_power_fir_attenuation_db;
    std::vector<double> candidate_beam_broadening_fraction;
    std::vector<double> candidate_astronomical_alias_power_ratio;
    std::vector<double> candidate_fir_stopband_rejection_db;
    std::vector<double> candidate_fir_transition_margin_hz;
    std::vector<double> candidate_fir_raw_group_delay_s;
    std::vector<double> candidate_software_group_delay_s;
};

template <class Calib, class RawTimeChunkConfig>
RtcDiagScanArraySummaryData calculate_rtcdiag_scan_array_summary(
    const Calib &calib, const RawTimeChunkConfig &raw_config,
    const std::vector<RtcSamplingScanMotion> &scan_motion,
    double native_sample_rate_hz,
    Eigen::Index n_scans, std::size_t n_array_values,
    std::size_t n_scan_array_values, double fill_double) {
    RtcDiagScanArraySummaryData values;
    auto make_scan_array_values = [&]() {
        return std::vector<double>(n_scan_array_values, fill_double);
    };
    values.source_power_half_bandwidth_hz = make_scan_array_values();
    values.tod_lowpass_to_source_power_half_ratio = make_scan_array_values();
    values.beam_major_fwhm_arcsec = make_scan_array_values();
    values.beam_minor_fwhm_arcsec = make_scan_array_values();
    values.beam_position_angle_rad = make_scan_array_values();
    values.limiting_projected_fwhm_arcsec = make_scan_array_values();
    values.limiting_speed_arcsec_s = make_scan_array_values();

    values.candidate_factors = rtc_sampling_supported_factors(
        native_sample_rate_hz, raw_config.filter.freq_high_Hz);
    if (raw_config.filter.enabled) {
        timestream::Filter filter;
        filter.a_gibbs = raw_config.filter.a_gibbs;
        filter.freq_low_Hz = raw_config.filter.freq_low_Hz;
        filter.freq_high_Hz = raw_config.filter.freq_high_Hz;
        filter.n_terms = raw_config.filter.n_terms;
        filter.make_filter(native_sample_rate_hz);
        values.fir_coefficients.assign(
            filter.filter.data(), filter.filter.data() + filter.filter.size());
    }
    else {
        values.fir_coefficients = {1.0};
    }

    const std::size_t candidate_count = values.candidate_factors.size();
    const std::size_t table_size =
        n_scan_array_values * candidate_count;
    values.candidate_status.assign(table_size, -1);
    auto make_candidate_values = [&]() {
        return std::vector<double>(table_size, fill_double);
    };
    values.candidate_output_sample_rate_hz = make_candidate_values();
    values.candidate_output_nyquist_hz = make_candidate_values();
    values.candidate_samples_per_fwhm = make_candidate_values();
    values.candidate_beam_peak_attenuation_fraction = make_candidate_values();
    values.candidate_beam_half_power_fir_attenuation_db =
        make_candidate_values();
    values.candidate_beam_broadening_fraction = make_candidate_values();
    values.candidate_astronomical_alias_power_ratio = make_candidate_values();
    values.candidate_fir_stopband_rejection_db = make_candidate_values();
    values.candidate_fir_transition_margin_hz = make_candidate_values();
    values.candidate_fir_raw_group_delay_s = make_candidate_values();
    values.candidate_software_group_delay_s = make_candidate_values();

    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        const auto scan_index = static_cast<std::size_t>(scan);
        if (scan_index >= scan_motion.size() ||
            scan_motion[scan_index].intervals.empty()) {
            continue;
        }
        for (Eigen::Index arr_i = 0; arr_i < calib.n_arrays; ++arr_i) {
            const Eigen::Index array = calib.arrays(arr_i);
            const auto fwhm_it = calib.array_fwhms.find(array);
            if (fwhm_it == calib.array_fwhms.end()) {
                continue;
            }
            const auto pa_it = calib.array_pas.find(array);
            if (pa_it == calib.array_pas.end()) {
                continue;
            }
            const auto flat_i = scan_index * n_array_values +
                                static_cast<std::size_t>(arr_i);
            const auto projected = calculate_rtc_sampling_projected_beam(
                scan_motion[scan_index], std::get<0>(fwhm_it->second),
                std::get<1>(fwhm_it->second), pa_it->second);
            values.beam_major_fwhm_arcsec[flat_i] =
                projected.major_fwhm_arcsec;
            values.beam_minor_fwhm_arcsec[flat_i] =
                projected.minor_fwhm_arcsec;
            values.beam_position_angle_rad[flat_i] =
                projected.position_angle_rad;
            values.limiting_projected_fwhm_arcsec[flat_i] =
                projected.limiting_projected_fwhm_arcsec;
            values.limiting_speed_arcsec_s[flat_i] =
                projected.limiting_speed_arcsec_s;
            if (!std::isfinite(projected.temporal_sigma_s) ||
                projected.temporal_sigma_s <= 0.0) {
                continue;
            }
            const double f_half_hz =
                std::sqrt(std::log(2.0)) /
                (2.0 * rtc_sampling_pi * projected.temporal_sigma_s);
            values.source_power_half_bandwidth_hz[flat_i] = f_half_hz;
            const bool has_lowpass_ratio =
                raw_config.filter.enabled &&
                raw_config.filter.freq_high_Hz > 0.0 && f_half_hz > 0.0;
            if (has_lowpass_ratio) {
                values.tod_lowpass_to_source_power_half_ratio[flat_i] =
                    raw_config.filter.freq_high_Hz / f_half_hz;
            }

            for (std::size_t candidate_i = 0;
                 candidate_i < candidate_count; ++candidate_i) {
                const std::size_t table_i =
                    flat_i * candidate_count + candidate_i;
                const auto metrics = calculate_rtc_sampling_candidate_metrics(
                    values.candidate_factors[candidate_i],
                    native_sample_rate_hz, raw_config.filter.freq_high_Hz,
                    values.fir_coefficients, projected.temporal_sigma_s);
                values.candidate_status[table_i] = 0;
                values.candidate_output_sample_rate_hz[table_i] =
                    metrics.output_sample_rate_hz;
                values.candidate_output_nyquist_hz[table_i] =
                    metrics.output_nyquist_hz;
                values.candidate_samples_per_fwhm[table_i] =
                    metrics.samples_per_fwhm;
                values.candidate_beam_peak_attenuation_fraction[table_i] =
                    metrics.beam_peak_attenuation_fraction;
                values.candidate_beam_half_power_fir_attenuation_db[table_i] =
                    metrics.beam_half_power_fir_attenuation_db;
                values.candidate_beam_broadening_fraction[table_i] =
                    metrics.beam_broadening_fraction;
                values.candidate_astronomical_alias_power_ratio[table_i] =
                    metrics.astronomical_alias_power_ratio;
                values.candidate_fir_stopband_rejection_db[table_i] =
                    metrics.fir_stopband_rejection_db;
                values.candidate_fir_transition_margin_hz[table_i] =
                    metrics.fir_transition_margin_hz;
                values.candidate_fir_raw_group_delay_s[table_i] =
                    metrics.fir_raw_group_delay_s;
                values.candidate_software_group_delay_s[table_i] =
                    metrics.software_group_delay_s;
            }
        }
    }
    return values;
}

inline void add_rtc_sampling_candidate_double(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment, const std::vector<netCDF::NcDim> &dims,
    const std::vector<std::size_t> &chunks,
    const std::vector<double> &values) {
    auto var = fo.addVar(name, netCDF::ncDouble, dims);
    var.putAtt("units", units);
    var.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(var, chunks, 1);
    var.putVar(values.data());
}

inline void add_rtc_sampling_candidate_int(
    netCDF::NcFile &fo, const std::string &name, const std::string &comment,
    const std::vector<netCDF::NcDim> &dims,
    const std::vector<std::size_t> &chunks, const std::vector<int> &values) {
    auto var = fo.addVar(name, netCDF::ncInt, dims);
    var.putAtt("units", "N/A");
    var.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(var, chunks, 1);
    var.putVar(values.data());
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
         values.tod_lowpass_to_source_power_half_ratio,
         values.beam_major_fwhm_arcsec,
         values.beam_minor_fwhm_arcsec,
         values.beam_position_angle_rad,
         values.limiting_projected_fwhm_arcsec,
         values.limiting_speed_arcsec_s});

    add_netcdf_var(
        fo, "RTC_SAMPLING_METRICS_NOTICE",
        std::string{"This is a metrics-only diagnostic. No candidate was selected and no RTC behavior was changed."});
    add_netcdf_var(
        fo, "RTC_SAMPLING_TIME_GRID_SEMANTICS",
        std::string{"assigned compatibility grid; physical detector integration-event semantics and absolute timing placement are unavailable"});
    add_netcdf_var(
        fo, "RTC_SAMPLING_CANDIDATE_NOTES",
        std::string{"unranked metrics only; every candidate uses the same exact current configured FIR coefficients and differs only by integer phase-zero decimation factor; no candidate-specific FIR redesign, tolerance, recommendation, or selection is applied"});
    if (values.candidate_factors.empty()) {
        return;
    }

    const auto candidate_dim = fo.addDim(
        "n_rtc_sampling_candidates", values.candidate_factors.size());
    const std::vector<netCDF::NcDim> candidate_dims = {candidate_dim};
    const std::vector<std::size_t> candidate_chunks = {
        values.candidate_factors.size()};
    add_rtc_sampling_candidate_int(
        fo, "rtc_sampling_candidate_factor",
        "unranked integer factors admitted by the current configured-FIR Nyquist rule",
        candidate_dims, candidate_chunks, values.candidate_factors);

    const auto fir_dim = fo.addDim(
        "n_rtc_sampling_fir_coefficients", values.fir_coefficients.size());
    auto fir_var = fo.addVar(
        "rtc_sampling_realized_fir_coefficients", netCDF::ncDouble, fir_dim);
    fir_var.putAtt("units", "N/A");
    fir_var.putAtt(
        "comment",
        "exact centered FIR coefficients used by every candidate; [1] is the explicit identity response when RTC FIR filtering is disabled");
    fir_var.putVar(values.fir_coefficients.data());

    auto table_dims = scan_array_dims;
    table_dims.push_back(candidate_dim);
    auto table_chunks = scan_array_chunks;
    table_chunks.push_back(values.candidate_factors.size());
    add_rtc_sampling_candidate_int(
        fo, "rtc_sampling_candidate_status",
        "0=computed; -1=required scan motion or beam metadata unavailable; status is availability only, never a safe/unsafe or ranking label",
        table_dims, table_chunks, values.candidate_status);
    auto add_candidate_double = [&](const std::string &name,
                                    const std::string &units,
                                    const std::string &comment,
                                    const std::vector<double> &data) {
        add_rtc_sampling_candidate_double(
            fo, name, units, comment, table_dims, table_chunks, data);
    };
    add_candidate_double(
        "rtc_sampling_candidate_output_sample_rate_hz", "Hz",
        "native detector sample rate divided by candidate factor",
        values.candidate_output_sample_rate_hz);
    add_candidate_double(
        "rtc_sampling_candidate_output_nyquist_hz", "Hz",
        "candidate output Nyquist frequency",
        values.candidate_output_nyquist_hz);
    add_candidate_double(
        "rtc_sampling_candidate_samples_per_fwhm", "sample/FWHM",
        "candidate output samples across the shortest valid scan-projected beam crossing time",
        values.candidate_samples_per_fwhm);
    add_candidate_double(
        "rtc_sampling_candidate_beam_peak_attenuation_fraction", "N/A",
        "one minus the exact centered FIR response to the limiting Gaussian beam profile; no acceptance threshold is applied",
        values.candidate_beam_peak_attenuation_fraction);
    add_candidate_double(
        "rtc_sampling_candidate_beam_half_power_fir_attenuation_db", "dB",
        "exact FIR attenuation at the limiting Gaussian beam temporal power half-frequency relative to FIR DC gain",
        values.candidate_beam_half_power_fir_attenuation_db);
    add_candidate_double(
        "rtc_sampling_candidate_beam_broadening_fraction", "N/A",
        "fractional FWHM change of the exact zero-phase FIR-convolved limiting Gaussian beam profile",
        values.candidate_beam_broadening_fraction);
    add_candidate_double(
        "rtc_sampling_candidate_astronomical_alias_power_ratio", "N/A",
        "integrated phase-zero decimator alias power divided by desired compact-source power over the output baseband",
        values.candidate_astronomical_alias_power_ratio);
    add_candidate_double(
        "rtc_sampling_candidate_fir_stopband_rejection_db", "dB",
        "worst exact FIR rejection from output Nyquist through native Nyquist relative to DC; +inf for factor 1 with no alias band",
        values.candidate_fir_stopband_rejection_db);
    add_candidate_double(
        "rtc_sampling_candidate_fir_transition_margin_hz", "Hz",
        "output Nyquist minus the configured FIR high-frequency edge",
        values.candidate_fir_transition_margin_hz);
    add_candidate_double(
        "rtc_sampling_candidate_fir_raw_group_delay_s", "s",
        "uncentered FIR support delay before the existing centered convolution placement",
        values.candidate_fir_raw_group_delay_s);
    add_candidate_double(
        "rtc_sampling_candidate_software_group_delay_s", "s",
        "realized software group delay after centered convolution placement; zero for the exact symmetric FIR family",
        values.candidate_software_group_delay_s);
}

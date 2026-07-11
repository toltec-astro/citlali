#pragma once

// Pointing timestream output implementation detail.
// Include this only after Pointing has been declared.

#include <citlali/core/pipeline/stage_profile.h>
#include <citlali/core/pipeline/timestream_output_context.h>

template <class CalibScan>
bool Pointing::write_pointing_rtc_outputs(
    TCData<TCDataKind::RTC, Eigen::MatrixXd> &rtcdata,
    TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    TCData<TCDataKind::RTC, Eigen::MatrixXd> &rtc_outer_output,
    CalibScan &calib_scan,
    const citlali::pipeline::TimestreamOutputFlags &output_flags,
    const citlali::pipeline::TimestreamOutputWriters &output_writers,
    Eigen::Index rtc_scan_row,
    bool write_this_rtc,
    const std::string &map_grouping) {
    if (output_flags.write_rtcdiag) {
        if (!output_writers.write_when_ready(
            output_writers.rtcdiag, ptcdata.index.data, [&] {
                logger->info("writing rtc diagnostics sidecar chunk");
                auto profile_scope = citlali::pipeline::profile_stage(
                    "timestream.rtcdiag.write_chunk", logger,
                    "scan=" + std::to_string(static_cast<long long>(
                                  ptcdata.index.data + 1)));
                rtcproc.append_diag_to_netcdf(
                    ptcdata, output_paths.rtcdiag_filename, calib_scan,
                    ptcdata.index.data);
            })) {
            return false;
        }
    }

    // write rtc timestreams
    if (write_this_rtc) {
        if (!output_writers.write_when_ready(
            output_writers.rtc, rtc_scan_row, [&] {
                if (citlali::pipeline::raw_tod_outer_output(*this)) {
                    logger->info("writing outer raw time chunk");
                    auto profile_scope = citlali::pipeline::profile_stage(
                        "timestream.rtc_output.write_chunk", logger,
                        "scan=" + std::to_string(static_cast<long long>(
                                      rtcdata.index.data + 1)));
                    rtcproc.append_to_netcdf(
                        rtc_outer_output, output_paths.tod_filename["rtc"],
                        map_grouping, telescope.pixel_axes,
                        rtc_outer_output.pointing_offsets_arcsec.data, calib,
                        false, rtc_scan_row);
                }
                else {
                    logger->info("writing raw time chunk");
                    auto profile_scope = citlali::pipeline::profile_stage(
                        "timestream.rtc_output.write_chunk", logger,
                        "scan=" + std::to_string(static_cast<long long>(
                                      rtcdata.index.data + 1)));
                    rtcproc.append_to_netcdf(
                        ptcdata, output_paths.tod_filename["rtc"], map_grouping,
                        telescope.pixel_axes,
                        ptcdata.pointing_offsets_arcsec.data, calib, false,
                        rtc_scan_row);
                }
            })) {
            return false;
        }
    }
    if (output_flags.write_rtc || output_flags.write_rtcdiag) {
        rtcproc.clear_cached_diagnostics(ptcdata.index.data);
    }
    return true;
}

template <class CalibScan>
bool Pointing::write_pointing_ptc_outputs(
    TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    CalibScan &calib_scan,
    const citlali::pipeline::TimestreamOutputFlags &output_flags,
    const citlali::pipeline::TimestreamOutputWriters &output_writers,
    const std::string &map_grouping) {
    // write ptc timestreams
    if (output_flags.write_ptcdiag) {
        if (!output_writers.write_when_ready(
            output_writers.ptcdiag, ptcdata.index.data, [&] {
                logger->info("writing ptc diagnostics sidecar chunk");
                auto profile_scope = citlali::pipeline::profile_stage(
                    "timestream.ptcdiag.write_chunk", logger,
                    "scan=" + std::to_string(static_cast<long long>(
                                  ptcdata.index.data + 1)));
                ptcproc.append_diag_to_netcdf(
                    ptcdata, output_paths.ptcdiag_filename, calib_scan,
                    ptcdata.index.data);
            })) {
            return false;
        }
    }

    const auto ptc_scan_row = tod_output_scan_row(
        ptcdata.index.data, citlali::config::TodOutputStream::ptc);
    if (output_flags.write_ptc && ptc_scan_row >= 0) {
        if (!output_writers.write_when_ready(
            output_writers.ptc, ptc_scan_row, [&] {
                logger->info("writing processed time chunk");
                auto profile_scope = citlali::pipeline::profile_stage(
                    "timestream.ptc_output.write_chunk", logger,
                    "scan=" + std::to_string(static_cast<long long>(
                                  ptcdata.index.data + 1)));
                ptcproc.append_to_netcdf(
                    ptcdata, output_paths.tod_filename["ptc"], map_grouping,
                    telescope.pixel_axes, ptcdata.pointing_offsets_arcsec.data,
                    calib_scan, false, ptc_scan_row);
            })) {
            return false;
        }
    }
    if (output_flags.write_ptc || output_flags.write_ptcdiag) {
        ptcproc.clear_cached_diagnostics(ptcdata.index.data);
    }
    return true;
}

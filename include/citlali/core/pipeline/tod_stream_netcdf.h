#pragma once

#include <string>
#include <vector>

#include <netcdf>

#include <citlali/core/pipeline/output_netcdf_metadata.h>
#include <citlali/core/pipeline/ptcdiag_netcdf.h>
#include <citlali/core/pipeline/rtcdiag_netcdf.h>

namespace citlali::pipeline {

template <class Calib, class RtcProc>
void add_rtc_tod_stream_diagnostic_outputs(
    netCDF::NcFile &fo, const Calib &calib, const RtcProc &rtcproc,
    const TodPreparedLayout &tod_layout, double fsmp,
    double downsampled_fsmp) {
    const int fill_int = rtcdiag_fill_int();
    const double fill_double = rtcdiag_fill_double();
    const double stream_sample_rate =
        rtc_tod_stream_sample_rate(rtcproc, fsmp, downsampled_fsmp);
    add_rtcdiag_tod_stream_diag(
        fo, calib, rtcproc, tod_layout.dims.n_scans,
        tod_layout.dims.n_dets, tod_layout.stream.n_output_scans,
        stream_sample_rate, fill_int, fill_double);
}

template <class Calib, class PtcProc>
void add_ptc_tod_stream_weight_and_diagnostic_outputs(
    netCDF::NcFile &fo, const Calib &calib, const PtcProc &ptcproc,
    const TodPreparedLayout &tod_layout, const std::string &signal_unit) {
    const std::vector<netCDF::NcDim> weight_dims = {
        tod_layout.dims.n_scans, tod_layout.dims.n_dets};
    add_ptc_weights_var(fo, weight_dims, signal_unit);

    const int fill_int = ptcdiag_fill_int();
    const double fill_double = ptcdiag_fill_double();
    add_ptcdiag_tod_optional_diag(
        fo, calib, ptcproc, tod_layout.dims.signal,
        tod_layout.chunking.mode, tod_layout.chunking.sizes,
        tod_layout.dims.n_scans, tod_layout.dims.n_dets,
        tod_layout.stream.n_output_scans, fill_int, fill_double);
}

}  // namespace citlali::pipeline

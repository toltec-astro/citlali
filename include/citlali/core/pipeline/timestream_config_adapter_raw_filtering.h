#pragma once

#include <citlali/core/config/timestream_config.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace citlali::pipeline {

template <class RtcProc>
void adapt_raw_filtering_config_one_way(
    const citlali::config::RawTimeChunkConfig &raw, RtcProc &rtcproc,
    double arcsec_to_rad, double fwhm_to_std) {
    const auto &kernel = raw.kernel;
    rtcproc.run_kernel = kernel.enabled;
    rtcproc.kernel.filepath = kernel.filepath;
    rtcproc.kernel.type = kernel.type;
    rtcproc.kernel.fwhm_rad = kernel.fwhm_arcsec * arcsec_to_rad;
    rtcproc.kernel.sigma_rad = rtcproc.kernel.fwhm_rad * fwhm_to_std;
    rtcproc.kernel.img_ext_names =
        kernel.enabled && kernel.type == "fits"
            ? kernel.image_ext_names
            : std::vector<std::string>{};

    const auto &filter = raw.filter;
    rtcproc.run_tod_filter = filter.enabled;
    rtcproc.filter.a_gibbs = filter.a_gibbs;
    rtcproc.filter.freq_low_Hz = filter.freq_low_Hz;
    rtcproc.filter.freq_high_Hz = filter.freq_high_Hz;
    rtcproc.filter.n_terms = filter.enabled ? filter.n_terms : 0;
    rtcproc.run_tod_notch = filter.enabled && filter.notch.enabled;
    rtcproc.filter.notch_zero_phase = filter.notch.zero_phase;
    rtcproc.filter.w0s = filter.notch.freqs_Hz;
    rtcproc.filter.qs.clear();
    if (rtcproc.run_tod_notch) {
        rtcproc.filter.qs.reserve(filter.notch.freqs_Hz.size());
        for (std::size_t index = 0; index < filter.notch.freqs_Hz.size();
             ++index) {
            const auto width = filter.notch.delta_f_Hz.size() == 1
                                   ? filter.notch.delta_f_Hz.front()
                                   : filter.notch.delta_f_Hz.at(index);
            rtcproc.filter.qs.push_back(
                filter.notch.freqs_Hz[index] / width);
        }
    }

    const auto &iir = raw.iir_filter;
    rtcproc.run_tod_iir_highpass = iir.enabled;
    rtcproc.filter.iir_highpass_freq_Hz = iir.enabled ? iir.freq_Hz : 0.0;
    rtcproc.filter.iir_highpass_order = iir.enabled ? iir.order : 1;
    rtcproc.filter.iir_highpass_zero_phase =
        iir.enabled && iir.zero_phase;

    const auto &downsample = raw.downsample;
    rtcproc.run_downsample = downsample.enabled;
    rtcproc.downsampler.factor = downsample.factor;
    rtcproc.downsampler.downsampled_freq_Hz =
        downsample.downsampled_freq_Hz;

    const auto &guard = filter.edge_guard;
    rtcproc.filter_edge_guard.enabled = guard.enabled;
    rtcproc.filter_edge_guard.mode =
        std::string(citlali::config::to_string(guard.mode));
    rtcproc.filter_edge_guard.combine =
        std::string(citlali::config::to_string(guard.combine));
    rtcproc.filter_edge_guard.min_samples = guard.min_samples;
    rtcproc.filter_edge_guard.extra_samples = guard.extra_samples;
    rtcproc.filter_edge_guard.max_samples = guard.max_samples;
    rtcproc.filter_edge_guard.iir_settle_attenuation =
        guard.iir_settle_attenuation;
    rtcproc.filter_edge_guard.apply_fir = guard.apply_fir;
    rtcproc.filter_edge_guard.apply_notch = guard.apply_notch;
    rtcproc.filter_edge_guard.apply_dynamic_notch =
        guard.apply_dynamic_notch;
    rtcproc.filter_edge_guard.apply_iir_highpass =
        guard.apply_iir_highpass;
    rtcproc.filter_edge_guard.apply_downsample = guard.apply_downsample;

    rtcproc.run_calibrate = raw.flux_calibration_enabled;
    rtcproc.run_extinction = raw.extinction_correction_enabled;
    rtcproc.altaz_destripe.enabled = raw.altaz_destripe.enabled;
    rtcproc.altaz_destripe.grouping = raw.altaz_destripe.grouping;
    rtcproc.altaz_destripe.fit_time_trend =
        raw.altaz_destripe.fit_time_trend;
    rtcproc.altaz_destripe.fit_derivs = raw.altaz_destripe.fit_derivs;
    rtcproc.altaz_destripe.min_samples = raw.altaz_destripe.min_samples;
}

}  // namespace citlali::pipeline

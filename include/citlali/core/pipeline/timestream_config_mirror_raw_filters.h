#pragma once

// Included by timestream_config_mirror.h inside namespace citlali::pipeline.

template <class KernelConfig, class RtcProc>
void mirror_raw_kernel_config(KernelConfig &target, const RtcProc &rtcproc,
                              double radians_to_arcsec) {
    target.enabled = rtcproc.run_kernel;
    if (!rtcproc.run_kernel) {
        return;
    }
    target.filepath = rtcproc.kernel.filepath;
    target.type = rtcproc.kernel.type;
    target.fwhm_arcsec = rtcproc.kernel.fwhm_rad * radians_to_arcsec;
    target.image_ext_names = rtcproc.kernel.img_ext_names;
}

template <class DownsampleConfig, class RtcProc>
void mirror_raw_downsample_config(DownsampleConfig &target,
                                  const RtcProc &rtcproc) {
    target.enabled = rtcproc.run_downsample;
    if (!rtcproc.run_downsample) {
        return;
    }
    target.factor = rtcproc.downsampler.factor;
    target.downsampled_freq_Hz = rtcproc.downsampler.downsampled_freq_Hz;
}

template <class AltazDestripeConfig, class RtcProc>
void mirror_raw_altaz_destripe_config(AltazDestripeConfig &target,
                                      const RtcProc &rtcproc) {
    target.enabled = rtcproc.altaz_destripe.enabled;
    target.grouping = rtcproc.altaz_destripe.grouping;
    target.fit_time_trend = rtcproc.altaz_destripe.fit_time_trend;
    target.fit_derivs = rtcproc.altaz_destripe.fit_derivs;
    target.min_samples =
        static_cast<int>(rtcproc.altaz_destripe.min_samples);
}

template <class RawTimeChunkConfig, class RtcProc>
void mirror_raw_correction_flags(RawTimeChunkConfig &target,
                                 const RtcProc &rtcproc) {
    target.flux_calibration_enabled = rtcproc.run_calibrate;
    target.extinction_correction_enabled = rtcproc.run_extinction;
    target.extinction_model = rtcproc.run_extinction
                                  ? rtcproc.calibration.extinction_model
                                  : "N/A";
}

template <class FilterConfig, class RtcProc>
void mirror_raw_filter_config(FilterConfig &target, const RtcProc &rtcproc) {
    target.enabled = rtcproc.run_tod_filter;
    if (!rtcproc.run_tod_filter) {
        return;
    }

    target.a_gibbs = rtcproc.filter.a_gibbs;
    target.freq_low_Hz = rtcproc.filter.freq_low_Hz;
    target.freq_high_Hz = rtcproc.filter.freq_high_Hz;
    target.n_terms = static_cast<int>(rtcproc.filter.n_terms);
    target.notch.enabled = rtcproc.run_tod_notch;
    if (!rtcproc.run_tod_notch) {
        return;
    }

    target.notch.zero_phase = rtcproc.filter.notch_zero_phase;
    target.notch.freqs_Hz = rtcproc.filter.w0s;
    target.notch.delta_f_Hz.clear();
    target.notch.delta_f_Hz.reserve(rtcproc.filter.qs.size());
    for (std::size_t i = 0; i < rtcproc.filter.qs.size(); ++i) {
        const auto center_Hz =
            i < rtcproc.filter.w0s.size() ? rtcproc.filter.w0s[i] : 0.0;
        target.notch.delta_f_Hz.push_back(
            rtcproc.filter.qs[i] > 0.0 ? center_Hz / rtcproc.filter.qs[i]
                                        : 0.0);
    }
}

template <class IirFilterConfig, class RtcProc>
void mirror_raw_iir_filter_config(IirFilterConfig &target,
                                  const RtcProc &rtcproc) {
    target.enabled = rtcproc.run_tod_iir_highpass;
    if (!rtcproc.run_tod_iir_highpass) {
        // Match the legacy effective state instead of exposing typed defaults.
        target.freq_Hz = 0.0;
        target.order = 1;
        target.zero_phase = false;
        return;
    }

    target.freq_Hz = rtcproc.filter.iir_highpass_freq_Hz;
    target.order = rtcproc.filter.iir_highpass_order;
    target.zero_phase = rtcproc.filter.iir_highpass_zero_phase;
}

template <class EdgeGuardConfig, class FilterEdgeGuard>
void mirror_raw_filter_edge_guard_config(EdgeGuardConfig &target,
                                         const FilterEdgeGuard &source) {
    target.enabled = source.enabled;
    if (auto parsed =
            citlali::config::parse_raw_filter_edge_guard_mode(source.mode)) {
        target.mode = *parsed;
    }
    if (auto parsed =
            citlali::config::parse_raw_filter_edge_guard_combine(
                source.combine)) {
        target.combine = *parsed;
    }
    target.min_samples = static_cast<int>(source.min_samples);
    target.extra_samples = static_cast<int>(source.extra_samples);
    target.max_samples = static_cast<int>(source.max_samples);
    target.iir_settle_attenuation = source.iir_settle_attenuation;
    target.apply_fir = source.apply_fir;
    target.apply_notch = source.apply_notch;
    target.apply_dynamic_notch = source.apply_dynamic_notch;
    target.apply_iir_highpass = source.apply_iir_highpass;
    target.apply_downsample = source.apply_downsample;
}

#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_config_read_helpers.h>

#include <array>
#include <string_view>
#include <tuple>

namespace citlali::pipeline {

inline constexpr std::array<std::string_view, 40>
    raw_filtering_request_paths{
        "timestream.raw_time_chunk.kernel.enabled",
        "timestream.raw_time_chunk.kernel.filepath",
        "timestream.raw_time_chunk.kernel.type",
        "timestream.raw_time_chunk.kernel.fwhm_arcsec",
        "timestream.raw_time_chunk.kernel.image_ext_names",
        "timestream.raw_time_chunk.filter.enabled",
        "timestream.raw_time_chunk.filter.a_gibbs",
        "timestream.raw_time_chunk.filter.freq_low_Hz",
        "timestream.raw_time_chunk.filter.freq_high_Hz",
        "timestream.raw_time_chunk.filter.n_terms",
        "timestream.raw_time_chunk.filter.notch.enabled",
        "timestream.raw_time_chunk.filter.notch.zero_phase",
        "timestream.raw_time_chunk.filter.notch.freqs_Hz",
        "timestream.raw_time_chunk.filter.notch.delta_f_Hz",
        "timestream.raw_time_chunk.filter.edge_guard.enabled",
        "timestream.raw_time_chunk.filter.edge_guard.mode",
        "timestream.raw_time_chunk.filter.edge_guard.combine",
        "timestream.raw_time_chunk.filter.edge_guard.min_samples",
        "timestream.raw_time_chunk.filter.edge_guard.extra_samples",
        "timestream.raw_time_chunk.filter.edge_guard.max_samples",
        "timestream.raw_time_chunk.filter.edge_guard.iir_settle_attenuation",
        "timestream.raw_time_chunk.filter.edge_guard.apply_fir",
        "timestream.raw_time_chunk.filter.edge_guard.apply_notch",
        "timestream.raw_time_chunk.filter.edge_guard.apply_dynamic_notch",
        "timestream.raw_time_chunk.filter.edge_guard.apply_iir_highpass",
        "timestream.raw_time_chunk.filter.edge_guard.apply_downsample",
        "timestream.raw_time_chunk.IIR_filter.enabled",
        "timestream.raw_time_chunk.IIR_filter.freq_Hz",
        "timestream.raw_time_chunk.IIR_filter.order",
        "timestream.raw_time_chunk.IIR_filter.zero_phase",
        "timestream.raw_time_chunk.downsample.enabled",
        "timestream.raw_time_chunk.downsample.factor",
        "timestream.raw_time_chunk.downsample.downsampled_freq_Hz",
        "timestream.raw_time_chunk.altaz_destripe.enabled",
        "timestream.raw_time_chunk.altaz_destripe.grouping",
        "timestream.raw_time_chunk.altaz_destripe.fit_time_trend",
        "timestream.raw_time_chunk.altaz_destripe.fit_derivs",
        "timestream.raw_time_chunk.altaz_destripe.min_samples",
        "timestream.raw_time_chunk.flux_calibration.enabled",
        "timestream.raw_time_chunk.extinction_correction.enabled",
    };

template <class Config, class Diagnostics>
void read_raw_kernel_request_config(
    Config &config, citlali::config::RawTimeChunkKernelConfig &kernel,
    Diagnostics &diagnostics) {
    auto key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "kernel", name};
    };
    read_optional_raw_request_value(
        config, key("enabled"), kernel.enabled, diagnostics);
    read_optional_raw_request_value(
        config, key("filepath"), kernel.filepath, diagnostics);
    read_optional_raw_request_value(
        config, key("type"), kernel.type, diagnostics);
    read_optional_raw_request_value(
        config, key("fwhm_arcsec"), kernel.fwhm_arcsec, diagnostics,
        {}, {0.0});
    read_optional_raw_request_value(
        config, key("image_ext_names"), kernel.image_ext_names,
        diagnostics);
}

template <class Config, class Diagnostics>
void read_raw_filter_request_config(
    Config &config, citlali::config::RawTimeChunkFilterConfig &filter,
    Diagnostics &diagnostics) {
    auto key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "filter", name};
    };
    read_optional_raw_request_value(
        config, key("enabled"), filter.enabled, diagnostics);
    read_optional_raw_request_value(
        config, key("a_gibbs"), filter.a_gibbs, diagnostics);
    read_optional_raw_request_value(
        config, key("freq_low_Hz"), filter.freq_low_Hz, diagnostics,
        {}, {0.0});
    read_optional_raw_request_value(
        config, key("freq_high_Hz"), filter.freq_high_Hz, diagnostics,
        {}, {0.0});
    read_optional_raw_request_value(
        config, key("n_terms"), filter.n_terms, diagnostics, {}, {0});

    auto &notch = filter.notch;
    auto notch_key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "filter", "notch", name};
    };
    read_optional_raw_request_value(
        config, notch_key("enabled"), notch.enabled, diagnostics);
    read_optional_raw_request_value(
        config, notch_key("zero_phase"), notch.zero_phase, diagnostics);
    read_optional_raw_request_value(
        config, notch_key("freqs_Hz"), notch.freqs_Hz, diagnostics);
    read_optional_raw_request_value(
        config, notch_key("delta_f_Hz"), notch.delta_f_Hz, diagnostics);

    auto &guard = filter.edge_guard;
    auto guard_key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "filter", "edge_guard",
            name};
    };
    read_optional_raw_request_value(
        config, guard_key("enabled"), guard.enabled, diagnostics);
    std::string mode{citlali::config::to_string(guard.mode)};
    read_optional_parsed_mirrored_config_value(
        config, guard_key("mode"), mode, guard.mode,
        citlali::config::parse_raw_filter_edge_guard_mode, diagnostics,
        {"flag", "none"});
    std::string combine{citlali::config::to_string(guard.combine)};
    read_optional_parsed_mirrored_config_value(
        config, guard_key("combine"), combine, guard.combine,
        citlali::config::parse_raw_filter_edge_guard_combine, diagnostics,
        {"sum", "max"});
    read_optional_raw_request_value(
        config, guard_key("min_samples"), guard.min_samples, diagnostics,
        {}, {0});
    read_optional_raw_request_value(
        config, guard_key("extra_samples"), guard.extra_samples,
        diagnostics, {}, {0});
    read_optional_raw_request_value(
        config, guard_key("max_samples"), guard.max_samples, diagnostics,
        {}, {0});
    read_optional_raw_request_value(
        config, guard_key("iir_settle_attenuation"),
        guard.iir_settle_attenuation, diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, guard_key("apply_fir"), guard.apply_fir, diagnostics);
    read_optional_raw_request_value(
        config, guard_key("apply_notch"), guard.apply_notch, diagnostics);
    read_optional_raw_request_value(
        config, guard_key("apply_dynamic_notch"),
        guard.apply_dynamic_notch, diagnostics);
    read_optional_raw_request_value(
        config, guard_key("apply_iir_highpass"),
        guard.apply_iir_highpass, diagnostics);
    read_optional_raw_request_value(
        config, guard_key("apply_downsample"), guard.apply_downsample,
        diagnostics);
}

template <class Config, class Diagnostics>
void read_raw_iir_filter_request_config(
    Config &config, citlali::config::RawTimeChunkIirFilterConfig &filter,
    Diagnostics &diagnostics) {
    auto key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "IIR_filter", name};
    };
    read_optional_raw_request_value(
        config, key("enabled"), filter.enabled, diagnostics);
    read_optional_raw_request_value(
        config, key("freq_Hz"), filter.freq_Hz, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("order"), filter.order, diagnostics, {}, {1});
    read_optional_raw_request_value(
        config, key("zero_phase"), filter.zero_phase, diagnostics);
}

template <class Config, class Diagnostics>
void read_raw_downsample_request_config(
    Config &config, citlali::config::RawTimeChunkDownsampleConfig &downsample,
    Diagnostics &diagnostics) {
    auto key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "downsample", name};
    };
    read_optional_raw_request_value(
        config, key("enabled"), downsample.enabled, diagnostics);
    read_optional_raw_request_value(
        config, key("factor"), downsample.factor, diagnostics, {}, {0});
    read_optional_raw_request_value(
        config, key("downsampled_freq_Hz"),
        downsample.downsampled_freq_Hz, diagnostics);
}

template <class Config, class Diagnostics>
void read_raw_altaz_destripe_request_config(
    Config &config, citlali::config::RawTimeChunkAltAzDestripeConfig &altaz,
    Diagnostics &diagnostics) {
    auto key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "altaz_destripe", name};
    };
    read_optional_raw_request_value(
        config, key("enabled"), altaz.enabled, diagnostics);
    read_optional_raw_request_value(
        config, key("grouping"), altaz.grouping, diagnostics,
        {"nw", "network", "array", "all"});
    read_optional_raw_request_value(
        config, key("fit_time_trend"), altaz.fit_time_trend, diagnostics);
    read_optional_raw_request_value(
        config, key("fit_derivs"), altaz.fit_derivs, diagnostics);
    read_optional_raw_request_value(
        config, key("min_samples"), altaz.min_samples, diagnostics,
        {}, {4});
}

template <class Config, class Diagnostics>
void read_raw_correction_request_config(
    Config &config, citlali::config::RawTimeChunkConfig &raw,
    Diagnostics &diagnostics) {
    read_optional_raw_request_value(
        config,
        std::tuple{"timestream", "raw_time_chunk", "flux_calibration",
                   "enabled"},
        raw.flux_calibration_enabled, diagnostics);
    read_optional_raw_request_value(
        config,
        std::tuple{"timestream", "raw_time_chunk",
                   "extinction_correction", "enabled"},
        raw.extinction_correction_enabled, diagnostics);
}

template <class Config, class Diagnostics>
void read_raw_filtering_request_config(
    Config &config, citlali::config::RawTimeChunkConfig &raw,
    Diagnostics &diagnostics) {
    read_raw_kernel_request_config(config, raw.kernel, diagnostics);
    read_raw_filter_request_config(config, raw.filter, diagnostics);
    read_raw_iir_filter_request_config(config, raw.iir_filter, diagnostics);
    read_raw_downsample_request_config(config, raw.downsample, diagnostics);
    read_raw_altaz_destripe_request_config(
        config, raw.altaz_destripe, diagnostics);
    read_raw_correction_request_config(config, raw, diagnostics);
}

}  // namespace citlali::pipeline

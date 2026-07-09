#pragma once

#include <citlali/core/config/coadd_config.h>
#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/runtime_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool timestream_processing_enabled(const Engine &engine) {
    return engine.typed_config.timestream.enabled;
}

template <class Engine>
bool tod_output_enabled(const Engine &engine) {
    return citlali::config::is_tod_output_enabled(
        engine.typed_config.timestream.output.type);
}

template <class Engine>
bool raw_tod_output_enabled(const Engine &engine) {
    return citlali::config::tod_output_includes_rtc(
        engine.typed_config.timestream.output.type);
}

template <class Engine>
bool processed_tod_output_enabled(const Engine &engine) {
    return citlali::config::tod_output_includes_ptc(
        engine.typed_config.timestream.output.type);
}

template <class Engine>
bool tod_output_files_available(const Engine &engine) {
    return tod_output_enabled(engine) && !engine.output_paths.tod_filename.empty();
}

template <class Engine>
bool raw_tod_output_files_available(const Engine &engine) {
    return raw_tod_output_enabled(engine) && !engine.output_paths.tod_filename.empty();
}

template <class Engine>
bool processed_tod_output_files_available(const Engine &engine) {
    return processed_tod_output_enabled(engine) && !engine.output_paths.tod_filename.empty();
}

template <class Engine>
bool mapmaking_enabled(const Engine &engine) {
    return citlali::config::mapmaking_active(engine.typed_config.mapmaking);
}

template <class Engine>
bool coadd_enabled(const Engine &engine) {
    return citlali::config::coadd_active(engine.typed_config.coadd);
}

template <class Engine>
bool noise_maps_enabled(const Engine &engine) {
    return citlali::config::noise_maps_active(engine.typed_config.noise);
}

template <class Engine>
bool noise_realization_outputs_enabled(const Engine &engine) {
    return citlali::config::noise_realization_outputs_active(
        engine.typed_config.noise);
}

template <class Engine>
bool noise_product_outputs_enabled(const Engine &engine) {
    return citlali::config::noise_product_outputs_active(
        engine.typed_config.noise);
}

template <class Engine>
bool empirical_noise_weights_enabled(const Engine &engine) {
    return citlali::config::empirical_noise_weights_active(
        engine.typed_config.noise);
}

template <class Engine>
bool empirical_weight_calibration_enabled(const Engine &engine) {
    return noise_product_outputs_enabled(engine) &&
           noise_maps_enabled(engine) &&
           empirical_noise_weights_enabled(engine);
}

template <class Engine>
bool map_filter_enabled(const Engine &engine) {
    return citlali::config::map_filtering_active(
        engine.typed_config.post_processing);
}

template <class Engine>
bool source_finding_enabled(const Engine &engine) {
    return citlali::config::source_finding_active(
        engine.typed_config.post_processing);
}

template <class Engine>
bool mapmaking_outputs_enabled(const Engine &engine) {
    return mapmaking_enabled(engine);
}

template <class Engine>
bool coadd_outputs_enabled(const Engine &engine) {
    return coadd_enabled(engine);
}

template <class Engine>
bool map_filter_outputs_enabled(const Engine &engine) {
    return map_filter_enabled(engine);
}

template <class Engine>
bool source_finding_outputs_enabled(const Engine &engine) {
    return source_finding_enabled(engine);
}

template <class Engine>
bool should_write_filtered_outputs(const Engine &engine) {
    return map_filter_outputs_enabled(engine);
}

template <class Engine>
bool filtered_maps_written_during_filtering(const Engine &engine) {
    return engine.typed_config.runtime.reduction_type ==
           citlali::config::ReductionType::science;
}

template <class Engine>
bool should_calculate_filtered_noise_products(const Engine &engine) {
    return noise_product_outputs_enabled(engine) &&
           noise_maps_enabled(engine) &&
           !filtered_maps_written_during_filtering(engine);
}

template <class Engine>
bool should_find_filtered_sources(const Engine &engine) {
    return source_finding_outputs_enabled(engine);
}

template <class Engine>
bool should_write_iteration_coadd_outputs(const Engine &engine) {
    return coadd_outputs_enabled(engine);
}

}  // namespace citlali::pipeline

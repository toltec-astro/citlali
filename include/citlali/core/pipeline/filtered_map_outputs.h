#pragma once

#include <citlali/core/pipeline/map_diagnostics.h>
#include <citlali/core/pipeline/map_filtering.h>
#include <citlali/core/pipeline/map_noise_products.h>
#include <citlali/core/pipeline/map_output.h>
#include <citlali/core/pipeline/map_source_finding.h>
#include <citlali/core/pipeline/noise_weight_policy.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/post_processing_provenance_lifecycle.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class Engine>
bool should_calculate_filtered_map_noise_products(const Engine &engine) {
    return should_calculate_filtered_noise_products(engine);
}

template <class Engine>
bool should_find_filtered_map_sources(const Engine &engine) {
    return should_find_filtered_sources(engine);
}

template <class Engine>
bool filtered_map_written_during_filtering(const Engine &engine) {
    return filtered_maps_written_during_filtering(engine);
}

template <class Engine>
bool filtered_map_noise_products_apply_empirical_weights(
    const Engine &engine) {
    return filtered_noise_products_apply_empirical_weights(engine);
}

template <auto FilteredMap, class Engine, class MapBuffer, class Logger>
void filter_maps(Engine &engine, MapBuffer &map_buffer,
                 StageProfileCollector &stage_profile,
                 const Logger &logger, const char *log_message) {
    (void)stage_profile;
    const auto profile_scope =
        profile_stage(stage_profile, "map.filter", logger, log_message);
    run_wiener_filter_with_log<FilteredMap>(
        engine, map_buffer, logger, log_message);
}

template <class Engine, class MapBuffer, class Logger>
void calculate_filtered_map_noise_products_if_needed(
    Engine &engine, MapBuffer &map_buffer,
    StageProfileCollector &stage_profile, const Logger &logger,
    const char *log_message) {
    calculate_map_noise_products_if_needed(
        map_buffer, should_calculate_filtered_map_noise_products(engine),
        filtered_map_noise_products_apply_empirical_weights(engine),
        stage_profile, logger, log_message);
}

template <class MapBuffer, class Logger>
void calculate_filtered_map_diagnostics(
    MapBuffer &map_buffer, StageProfileCollector &stage_profile,
    const Logger &logger, const char *psd_log_message,
    const char *histogram_log_message) {
    calculate_map_diagnostics(
        map_buffer, stage_profile, logger, psd_log_message,
        histogram_log_message);
}

template <auto FilteredMap, class Engine, class MapBuffer, class Logger>
void find_filtered_map_sources_if_needed(
    Engine &engine, MapBuffer &map_buffer,
    StageProfileCollector &stage_profile, const Logger &logger,
    const char *log_message, PostProcessingMapContext context) {
    (void)stage_profile;
    const auto profile_scope =
        profile_stage(stage_profile, "map.source_finding", logger, log_message);
    const auto cardinality = find_map_sources_if_needed<FilteredMap>(
        engine, map_buffer, logger, should_find_filtered_map_sources(engine),
        log_message);
    if (cardinality.has_value()) {
        record_post_processing_catalog_fits_completed_if_available(
            engine, context,
            cardinality->attempt_count, cardinality->valid_count);
    }
}

template <auto FilteredMap, class Engine, class Logger>
void output_filtered_maps_if_needed(
    Engine &engine, StageProfileCollector &stage_profile,
    const Logger &logger, const char *output_log_message,
    const char *skip_log_message) {
    output_map_if_needed<FilteredMap>(
        engine, stage_profile, logger,
        !filtered_map_written_during_filtering(engine),
        output_log_message, skip_log_message);
}

}  // namespace citlali::pipeline

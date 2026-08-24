#pragma once

#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>

#include <cstddef>
#include <stdexcept>

namespace citlali::pipeline {

inline bool post_processing_source_fitting_required(
    citlali::config::ReductionType reduction_type,
    const citlali::config::PostProcessingConfig &requested) {
    return citlali::config::is_pointing_family_reduction_type(reduction_type) ||
           citlali::config::is_beammap_reduction_type(reduction_type) ||
           requested.map_filtering.enabled ||
           requested.source_finding.enabled;
}

struct PostProcessingEffectiveResolutionRecord {
    citlali::config::ReductionType reduction_type =
        citlali::config::ReductionType::science;
    bool mapmaking_enabled = false;
    bool coadd_enabled = false;
    bool map_filtering_requested = false;
    bool map_filtering_effective = false;
    bool map_filtering_disabled_by_mapmaking = false;
    bool source_finding_requested = false;
    bool source_finding_effective = false;
    bool source_finding_disabled_by_mapmaking = false;
    bool source_fitting_required_by_reduction = false;
    bool source_fitting_required_by_map_filtering = false;
    bool source_fitting_required_by_source_finding = false;
    bool source_fitting_effective = false;
    bool source_fitting_disabled_by_mapmaking = false;
};

struct PostProcessingFitCardinality {
    std::size_t context_count = 0;
    std::size_t attempt_count = 0;
    std::size_t valid_count = 0;
};

struct PostProcessingMapContextRealizedState {
    std::size_t filter_context_count = 0;
    std::size_t filtered_map_count = 0;
    std::size_t source_finding_context_count = 0;
    std::size_t detected_source_count = 0;
    std::size_t source_table_write_count = 0;
    std::size_t source_table_row_count = 0;
    PostProcessingFitCardinality catalog_fits;
};

struct PostProcessingRealizedState {
    bool reduction_completed = false;
    PostProcessingMapContextRealizedState observation;
    PostProcessingMapContextRealizedState coadd;
    PostProcessingFitCardinality pointing_raw_fits;
    PostProcessingFitCardinality pointing_filtered_fits;
    PostProcessingFitCardinality beammap_fits;
    bool outputs_completed = false;
};

struct PostProcessingExecutionPlan {
    bool initialized = false;
    citlali::config::PostProcessingConfig requested;
    citlali::config::PostProcessingConfig effective;
    PostProcessingEffectiveResolutionRecord effective_resolution;
    PostProcessingRealizedState realized;

    void reset_from_request(
        const citlali::config::PostProcessingConfig &request,
        citlali::config::ReductionType reduction_type,
        bool mapmaking_enabled, bool coadd_enabled) {
        initialized = true;
        requested = request;
        effective = request;

        const bool fitting_required_by_reduction =
            citlali::config::is_pointing_family_reduction_type(reduction_type) ||
            citlali::config::is_beammap_reduction_type(reduction_type);
        const bool fitting_required =
            post_processing_source_fitting_required(
                reduction_type, request);
        if (!mapmaking_enabled) {
            citlali::config::set_map_filtering_enabled(effective, false);
            citlali::config::set_source_finding_enabled(effective, false);
        }
        citlali::config::set_source_fitting_active(
            effective, mapmaking_enabled && fitting_required);

        effective_resolution = PostProcessingEffectiveResolutionRecord{
            reduction_type,
            mapmaking_enabled,
            coadd_enabled,
            request.map_filtering.enabled,
            effective.map_filtering.enabled,
            request.map_filtering.enabled && !mapmaking_enabled,
            request.source_finding.enabled,
            effective.source_finding.enabled,
            request.source_finding.enabled && !mapmaking_enabled,
            fitting_required_by_reduction,
            request.map_filtering.enabled,
            request.source_finding.enabled,
            effective.source_fitting.active,
            fitting_required && !mapmaking_enabled,
        };
        realized = {};
    }

    void begin_iteration() {
        if (!initialized) {
            throw std::logic_error(
                "post-processing plan is not initialized");
        }
        realized = {};
    }
};

}  // namespace citlali::pipeline

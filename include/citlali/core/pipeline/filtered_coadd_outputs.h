#pragma once

#include <citlali/core/pipeline/filtered_map_outputs.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <auto FilteredCoaddMap, class Engine, class Logger>
bool filter_coadd_maps(Engine &engine,
                       StageProfileCollector &stage_profile,
                       const Logger &logger) {
    if constexpr (requires { engine.cmb.freeze_raw_science_parent(); }) {
        engine.cmb.freeze_raw_science_parent();
    }
    filter_maps<FilteredCoaddMap>(
        engine, engine.cmb, stage_profile, logger, "filtering coadded maps");
    record_post_processing_filter_completed_if_available(
        engine, PostProcessingMapContext::coadd);
    return filtered_map_written_during_filtering(engine);
}

template <class Engine, class Logger>
void calculate_filtered_coadd_noise_products_if_needed(
    Engine &engine, StageProfileCollector &stage_profile,
    const Logger &logger) {
    calculate_filtered_map_noise_products_if_needed(
        engine, engine.cmb, stage_profile, logger,
        "calculating filtered coadd empirical noise products");
}

template <class Engine, class Logger>
void calculate_filtered_coadd_map_diagnostics(Engine &engine,
                                              StageProfileCollector &stage_profile,
                                              const Logger &logger) {
    calculate_filtered_map_diagnostics(
        engine.cmb, stage_profile, logger,
        "calculating filtered coadded map psds",
        "calculating filtered coadded map histograms");
}

template <auto FilteredCoaddMap, class Engine, class Logger>
void find_filtered_coadd_sources_if_needed(Engine &engine,
                                           StageProfileCollector &stage_profile,
                                           const Logger &logger) {
    find_filtered_map_sources_if_needed<FilteredCoaddMap>(
        engine, engine.cmb, stage_profile, logger,
        "finding filtered coadded map sources",
        PostProcessingMapContext::coadd);
}

template <auto FilteredCoaddMap, class Engine, class Logger>
void output_filtered_coadd_maps_if_needed(Engine &engine,
                                          StageProfileCollector &stage_profile,
                                          const Logger &logger,
                                          bool published_during_filtering) {
    output_map_if_needed<FilteredCoaddMap>(
        engine, stage_profile, logger, !published_during_filtering,
        "outputting filtered coadded files",
        "filtered coadded files atomically published during Wiener "
        "filtering; skipping duplicate post-filter output stage");
}

template <auto FilteredCoaddMap, class TodProc, class Logger>
void write_filtered_coadd_outputs(TodProc &todproc,
                                  StageProfileCollector &stage_profile,
                                  const Logger &logger) {
    auto &engine = todproc.engine();
    (void)stage_profile;
    const auto profile_scope =
        profile_stage(stage_profile, "filtered_coadd.outputs", logger);

    const bool published_during_filtering =
        filter_coadd_maps<FilteredCoaddMap>(engine, stage_profile, logger);
    calculate_filtered_coadd_noise_products_if_needed(
        engine, stage_profile, logger);
    calculate_filtered_coadd_map_diagnostics(
        engine, stage_profile, logger);
    find_filtered_coadd_sources_if_needed<FilteredCoaddMap>(
        engine, stage_profile, logger);
    output_filtered_coadd_maps_if_needed<FilteredCoaddMap>(
        engine, stage_profile, logger, published_during_filtering);
}

template <auto FilteredCoaddMap, class TodProc, class Logger>
void write_filtered_coadd_outputs_if_needed(TodProc &todproc,
                                            StageProfileCollector &stage_profile,
                                            const Logger &logger) {
    auto &engine = todproc.engine();

    if (should_write_filtered_outputs(engine)) {
        write_filtered_coadd_outputs<FilteredCoaddMap>(
            todproc, stage_profile, logger);
    }
}

}  // namespace citlali::pipeline

#pragma once

#include <citlali/core/pipeline/coherent_iq_mode_sidecar.h>
#include <citlali/core/pipeline/observation_fruit_loop_map_loading.h>
#include <citlali/core/pipeline/observation_output_execution.h>
#include <citlali/core/pipeline/observation_pipeline.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <utility>

namespace citlali::pipeline {

template <class PublishCanonicalPackage, class WriteLinkedProducts>
auto publish_canonical_package_before_linked_products(
    PublishCanonicalPackage &&publish_canonical_package,
    WriteLinkedProducts &&write_linked_products) {
    auto package = std::forward<PublishCanonicalPackage>(
        publish_canonical_package)();
    std::forward<WriteLinkedProducts>(write_linked_products)();
    return package;
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class RawObs, class Logger>
void run_reduction_observation_pipeline(TodProc &todproc, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        StageProfileCollector &stage_profile,
                                        const Logger &logger) {
    auto &engine = todproc.engine();

    load_observation_fruit_loop_maps_if_needed<IsBeammap>(engine, logger);
    setup_and_run_observation_pipeline(
        engine, kidsproc, rawobs, stage_profile, logger);
    finalize_complete_calibration_product_identity_if_available(engine);
    write_coherent_iq_mode_sidecar_if_requested(
        engine, rawobs, logger);
    const auto raw_provenance_path =
        publish_canonical_package_before_linked_products(
            [&]() {
                auto path = publish_completed_raw_timestream_provenance<
                    IsBeammap>(engine);
                if (path) {
                    logger->info(
                        "canonical calibration package published before linked products: {}",
                        path->string());
                }
                return path;
            },
            [&]() {
                write_observation_outputs_and_accumulate<
                    RawObsMap, FilteredObsMap, FitMaps>(
                    todproc, stage_profile, logger);
            });
    (void)raw_provenance_path;
}

}  // namespace citlali::pipeline

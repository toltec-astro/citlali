#pragma once

#include <citlali/core/pipeline/coherent_iq_mode_sidecar.h>
#include <citlali/core/pipeline/observation_fruit_loop_map_loading.h>
#include <citlali/core/pipeline/observation_output_execution.h>
#include <citlali/core/pipeline/observation_pipeline.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>
#include <citlali/core/pipeline/rtcdiag_netcdf.h>
#include <citlali/core/pipeline/runtime_policy.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class RawObs, class Logger>
void run_reduction_observation_pipeline(TodProc &todproc, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        StageProfileCollector &stage_profile,
                                        const Logger &logger) {
    auto &engine = todproc.engine();

    try {
        load_observation_fruit_loop_maps_if_needed<IsBeammap>(engine, logger);
        setup_and_run_observation_pipeline(
            engine, kidsproc, rawobs, stage_profile, logger);
        write_coherent_iq_mode_sidecar_if_requested(
            engine, rawobs, logger);
        write_observation_outputs_and_accumulate<RawObsMap, FilteredObsMap,
                                                 FitMaps>(
            todproc, stage_profile, logger);
        std::optional<RtcdiagRawInputManifest> rtcdiag_raw_manifest;
        std::optional<RtcdiagSuccessorProductIdentity> rtcdiag_identity;
        if (!engine.output_paths.rtcdiag_filename.empty()) {
            if constexpr (requires {
                              rawobs.data_items();
                              rawobs.name();
                              engine.telescope.obs_goal;
                          }) {
                rtcdiag_raw_manifest =
                    make_rtcdiag_raw_input_manifest(rawobs);
                rtcdiag_identity = rtcdiag_successor_identity_for_mode(
                    rtcdiag_successor_mode(
                        runtime_reduction_type(engine),
                        engine.telescope.obs_goal));
            }
            else {
                throw std::logic_error(
                    "rtcdiag product identity/raw membership unavailable");
            }
        }
        const auto raw_provenance_path =
            publish_completed_raw_timestream_provenance<IsBeammap>(engine);
        if (raw_provenance_path) {
            logger->info("raw timestream provenance sidecar: {}",
                         raw_provenance_path->string());
        }
        if (!engine.output_paths.rtcdiag_filename.empty()) {
            engine.output_paths.rtcdiag_filename =
                finalize_rtcdiag_successor_staging(
                    engine.output_paths.rtcdiag_filename,
                    *rtcdiag_raw_manifest, *rtcdiag_identity);
            logger->info("complete RTC diagnostics published atomically: {}",
                         engine.output_paths.rtcdiag_filename);
        }
    }
    catch (...) {
        cleanup_netcdf_atomic_staging(
            engine.output_paths.rtcdiag_filename);
        throw;
    }
}

}  // namespace citlali::pipeline

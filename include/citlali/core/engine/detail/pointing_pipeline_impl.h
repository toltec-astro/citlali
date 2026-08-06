#pragma once

// Implementation detail included by pointing.h.

#include <citlali/core/pipeline/map_diagnostics.h>
#include <citlali/core/pipeline/noise_execution_plan.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_scan_generation.h>

template <class KidsProc, class RawObs>
void Pointing::pipeline(
    KidsProc &kidsproc, RawObs &rawobs,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    using input_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;
    const auto output_flags =
        citlali::pipeline::standard_timestream_output_flags(*this);
    const auto output_writers =
        citlali::pipeline::make_timestream_output_writers(output_flags);
    const auto output_expectations =
        citlali::pipeline::standard_timestream_output_expectations(
            *this, output_flags);
    const bool make_noise_maps =
        citlali::pipeline::noise_maps_enabled(*this);
    std::optional<citlali::pipeline::NoiseAssignmentContext> noise_context;
    if (make_noise_maps) {
        noise_context.emplace(
            citlali::pipeline::make_noise_assignment_context(
                observation_identity.obsnum, iteration.fruit_iter,
                "ordinary_mapmaking", static_cast<int>(omb.n_noise),
                static_cast<std::size_t>(telescope.scan_indices.cols()),
                static_cast<std::size_t>(calib.n_dets),
                omb.randomize_dets));
    }

    // progress bar
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100, "citlali progress ");
    citlali::pipeline::ScanCursor scan_cursor(telescope.scan_indices.cols());

    // grppi generator function. gets time chunk data from files sequentially and passes them to grppi::farm
    grppi::pipeline(
        tula::grppi_utils::dyn_ex(
            citlali::pipeline::runtime_parallel_policy_name(*this)),
        [&]() -> std::optional<input_t> {
            const auto next_scan = scan_cursor.next();
            if (!next_scan.has_value()) {
                return {};
            }
            const Eigen::Index scan = *next_scan;
            pb.count(telescope.scan_indices.cols(), 1);

            TCData<TCDataKind::RTC, Eigen::MatrixXd> rtcdata;
            const Eigen::Index scan_length =
                citlali::pipeline::initialize_rtc_scan(
                    rtcdata, telescope, scan);

            // populate noise matrix
            if (noise_context) {
                citlali::pipeline::populate_noise_map_signs(
                    rtcdata, omb, calib, true, *noise_context,
                    static_cast<std::size_t>(scan));
            }

            citlali::pipeline::populate_rtc_scan_samples(
                rtcdata, kidsproc, rawobs, scan, telescope, alignment.start_indices,
                alignment.end_indices, alignment.common_time, alignment.network_times, alignment.masks,
                citlali::config::timing_gap_interpolation_active(
                    citlali::pipeline::effective_runtime_values(*this)),
                scan_length, calib.n_dets,
                citlali::pipeline::timestream_config(*this).type);

            return rtcdata;
        },
        run(kidsproc, output_flags, output_writers, stage_profile));

    output_writers.rethrow_if_failed();
    output_writers.verify_complete(output_expectations);
    if (noise_context) {
        citlali::pipeline::record_noise_assignment_completed(
            citlali::pipeline::noise_plan(*this), *noise_context);
    }

    if (citlali::pipeline::mapmaking_enabled(*this)) {
        // normalize maps
        logger->info("normalizing maps");
        omb.normalize_maps();
        citlali::pipeline::calculate_map_diagnostics(
            omb, stage_profile, logger, "calculating map psd",
            "calculating map histogram");

        // fit maps
        fit_maps(citlali::pipeline::PointingFitStage::raw_observation);
    }
}

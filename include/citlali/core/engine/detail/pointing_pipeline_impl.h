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
    // declare random number generator
    boost::random::mt19937 eng{citlali::pipeline::noise_random_seed};

    // boost random number generator (0,1)
    boost::random::uniform_int_distribution<> rands{0,1};

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
            citlali::pipeline::populate_noise_map_signs(
                rtcdata, omb, calib,
                citlali::pipeline::noise_maps_enabled(*this),
                rands, eng);

            citlali::pipeline::populate_rtc_scan_samples(
                rtcdata, kidsproc, rawobs, scan, telescope, alignment,
                citlali::config::timing_gap_interpolation_active(
                    citlali::pipeline::effective_runtime_values(*this)),
                scan_length, calib.n_dets,
                citlali::pipeline::timestream_config(*this).type);

            return rtcdata;
        },
        run(kidsproc, output_flags, output_writers, stage_profile));

    output_writers.rethrow_if_failed();
    output_writers.verify_complete(output_expectations);

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

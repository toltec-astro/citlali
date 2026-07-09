#pragma once

// Implementation detail included by lali.h.

#include <citlali/core/pipeline/map_diagnostics.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/timestream_scan_generation.h>

void Lali::setup() {
    // run obsnum setup
    obsnum_setup();
}

template <class KidsProc, class RawObs>
void Lali::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    using tuple_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;

    // declare random number generator
    boost::random::mt19937 eng;

    // boost random number generator (0,1)
    boost::random::uniform_int_distribution<> rands{0,1};

    // progress bar
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100, "citlali progress ");

    // grppi generator function. gets time chunk data from files sequentially and passes them to grppi::farm
    grppi::pipeline(
        tula::grppi_utils::dyn_ex(
            citlali::pipeline::runtime_parallel_policy_name(*this)),
        [&]() -> std::optional<tuple_t> {
            // variable to hold current scan
            static int scan = 0;
            // loop through scans
            while (scan < telescope.scan_indices.cols()) {
                // update progress bar
                pb.count(telescope.scan_indices.cols(), 1);

                TCData<TCDataKind::RTC, Eigen::MatrixXd> rtcdata;
                const Eigen::Index scan_length =
                    citlali::pipeline::initialize_rtc_scan(
                        rtcdata, telescope, scan);

                // populate noise matrix (do outside of parallelized region for thread safety)
                citlali::pipeline::populate_noise_map_signs(
                    rtcdata, omb, calib,
                    citlali::pipeline::noise_maps_enabled(*this),
                    rands, eng);

                citlali::pipeline::populate_rtc_scan_samples(
                    rtcdata, kidsproc, rawobs, scan, telescope, alignment.start_indices,
                    alignment.end_indices, alignment.common_time, alignment.network_times, alignment.masks,
                    citlali::config::timing_gap_interpolation_active(
                        typed_config.runtime),
                    scan_length, calib.n_dets, typed_config.timestream.type);

                // increment scan
                scan++;
                // return rtcdata
                return rtcdata;
            }
            // reset scan to zero for each obs
            scan = 0;
            return {};
        },

        // run the farm
        run());

    if (citlali::pipeline::mapmaking_enabled(*this)) {
        // normalize maps
        logger->info("normalizing maps");
        if (typed_config.mapmaking.method !=
            citlali::config::MapMethod::maximum_likelihood) {
            if (rtcproc.run_polarization) {
                omb.normalize_polarized_maps();
            }
            else {
                omb.normalize_maps();
            }
        }
        citlali::pipeline::calculate_map_diagnostics(
            omb, logger, "calculating map psd",
            "calculating map histogram");

        // write map summary
        if (citlali::pipeline::verbose_runtime_enabled(*this)) {
            write_map_summary(omb);
        }
    }
}

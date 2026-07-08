#pragma once

// Implementation detail included by pointing.h.

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/timestream_scan_context.h>

template <class KidsProc, class RawObs>
void Pointing::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    using input_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;
    // initialize number of completed scans
    n_scans_done = 0;

    // declare random number generator
    boost::random::mt19937 eng;

    // boost random number generator (0,1)
    boost::random::uniform_int_distribution<> rands{0,1};

    // progress bar
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100, "citlali progress ");

    // grppi generator function. gets time chunk data from files sequentially and passes them to grppi::farm
    grppi::pipeline(tula::grppi_utils::dyn_ex(parallel_policy),
        [&]() -> std::optional<input_t> {
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

                // populate noise matrix
                citlali::pipeline::populate_noise_map_signs(
                    rtcdata, omb, calib,
                    citlali::pipeline::noise_maps_enabled(*this),
                    rands, eng);

                citlali::pipeline::populate_rtc_scan_samples(
                    rtcdata, kidsproc, rawobs, scan, telescope, start_indices,
                    end_indices, t_common, nw_times, masks, interp_over_gaps,
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
        run(kidsproc));

    if (citlali::pipeline::mapmaking_enabled(*this)) {
        // normalize maps
        logger->info("normalizing maps");
        omb.normalize_maps();
        // calculate map psds
        logger->info("calculating map psd");
        omb.calc_map_psd();
        // calculate map histograms
        logger->info("calculating map histogram");
        omb.calc_map_hist();
        // calculate mean error
        omb.calc_median_err();
        // calculate mean rms
        omb.calc_median_rms();

        // fit maps
        fit_maps();
    }
}

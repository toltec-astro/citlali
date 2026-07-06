#pragma once

// Implementation detail included by pointing.h.

#include <citlali/core/pipeline/output_policy.h>

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

                // create rtcdata
                TCData<TCDataKind::RTC, Eigen::MatrixXd> rtcdata;
                // get scan indices
                rtcdata.scan_indices.data = telescope.scan_indices.col(scan);
                // current scan
                rtcdata.index.data = scan;

                // populate noise matrix
                if (citlali::pipeline::noise_maps_enabled(*this)) {
                    if (omb.randomize_dets) {
                        rtcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(omb.n_noise, calib.n_dets)
                                                 .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                    } else {
                        rtcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(omb.n_noise)
                                                 .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                    }
                }

                // current length of outer scans
                Eigen::Index sl = rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;

                // get raw tod from files
                if (!interp_over_gaps) {
                    rtcdata.scans.data = kidsproc.populate_rtc_from_rawobs(rawobs, scan, telescope.scan_indices,
                                                                           start_indices, end_indices,
                                                                           sl, calib.n_dets, tod_type);
                }
                else {
                    auto scan_rawobs = kidsproc.load_rawobs_gaps(rawobs, scan, telescope.scan_indices, start_indices,
                                                                 t_common, nw_times, 1 / (2 * telescope.fsmp));
                    rtcdata.scans.data = kidsproc.populate_rtc_gaps(scan_rawobs, t_common, nw_times, masks, scan, 1 / (2 * telescope.fsmp),
                                                                telescope.scan_indices, sl, calib.n_dets, tod_type);
                    std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>().swap(scan_rawobs);
                }

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

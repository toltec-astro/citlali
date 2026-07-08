#pragma once

// Implementation detail included by lali.h.

#include <citlali/core/pipeline/output_policy.h>

void Lali::setup() {
    // run obsnum setup
    obsnum_setup();
}

template <class KidsProc, class RawObs>
void Lali::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    using tuple_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;

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
        [&]() -> std::optional<tuple_t> {
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

                // populate noise matrix (do outside of parallelized region for thread safety)
                if (citlali::pipeline::noise_maps_enabled(*this)) {
                    if (omb.randomize_dets) {
                        // n_noise x n_dets
                        rtcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(omb.n_noise, calib.n_dets)
                                    .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                    } else {
                        // n_noise
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
                                                                           sl, calib.n_dets,
                                                                           typed_config.timestream.type);
                }
                else {
                    // vector to store kids data
                    auto scan_rawobs = kidsproc.load_rawobs_gaps(rawobs, scan, telescope.scan_indices, start_indices,
                                                                 t_common, nw_times, 1 / (2 * telescope.fsmp));
                    rtcdata.scans.data = kidsproc.populate_rtc_gaps(scan_rawobs, t_common, nw_times, masks, scan, 1 / (2 * telescope.fsmp),
                                                                telescope.scan_indices, sl, calib.n_dets,
                                                                typed_config.timestream.type);
                    // try and clear input vector
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

        // write map summary
        if (verbose_mode) {
            write_map_summary(omb);
        }
    }
}

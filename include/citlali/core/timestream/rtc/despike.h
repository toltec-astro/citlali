#pragma once

#include <boost/random.hpp>
#include <boost/random/random_device.hpp>

#include <Eigen/Core>

#include <tula/logging.h>

#include <tula/algorithm/mlinterp/mlinterp.hpp>
#include <citlali/core/utils/utils.h>

namespace timestream {

class Despiker {
public:
    // spike sigma, time constant, sample rate
    double min_spike_sigma, time_constant_sec, window_size;
    double fsmp;

    // for window size
    bool run_filter = false;

    // size of region to merge flags
    int size = 10;

    // standard deviation limit
    double sigest_lim = 0;//1e-8;

    // grouping for replacing spikes
    std::string grouping;

    // use all detectors in replacing flagged scans
    bool use_all_det = true;

    // the main despiking routine
    template <typename DerivedA, typename DerivedB, typename apt_t>
    void despike(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB> &, apt_t &);

    // replace flagged data with interpolation
    template<typename DerivedA, typename DerivedB, typename apt_t>
    void replace_spikes(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB>&, apt_t &,
                        Eigen::Index);

private:
    // this function loops through the delta timestream values and finds the
    // spikes setting the corresponding flags to zero and calculating n_spikes
    template <typename DerivedA, typename DerivedB, typename DerivedC>
    void spike_finder(Eigen::DenseBase<DerivedA> &flags,
                      Eigen::DenseBase<DerivedB> &delta,
                      Eigen::DenseBase<DerivedC> &diff, int &n_spikes,
                      double &cutoff) {

        Eigen::Index n_pts = flags.size();

        // set flag matrix to zero if scan delta is above the cutoff
        flags.segment(1,n_pts - 2) =
            (diff.segment(1,n_pts - 2).array() > cutoff).select(0, flags.segment(1,n_pts - 2));

        // set corresponding delta values to zero
        delta.segment(1,n_pts - 2) =
            (diff.segment(1,n_pts - 2).array() > cutoff).select(0, delta.segment(1,n_pts - 2));

        // update difference vector and cutoff
        diff = abs(delta.derived().array() - delta.mean());
        cutoff = min_spike_sigma * engine_utils::calc_std_dev(delta);
    }

    template <typename Derived>
    auto make_window(Eigen::DenseBase<Derived> &spike_loc, int n_spikes,
                     Eigen::Index n_pts) {
        // window first index, last index, and size
        int win_index_0, win_index_1, win_size;

        // find biggest spike-free window
        // first deal with the harder case of multiple spikes
        // remember that there are n_spikes + 1 possible windows
        // only do if more than one spike
        if (n_spikes > 1) {
            Eigen::Matrix<int, Eigen::Dynamic, 1> delta_spike_loc(n_spikes + 1);
            // first element is the distance to first spike
            delta_spike_loc(0) = spike_loc(0);
            // last element is the distance to last spike
            delta_spike_loc(n_spikes) = n_pts - spike_loc(n_spikes - 1);
            // populate the delta spike location vector
            delta_spike_loc.segment(1,n_spikes - 1) =
                spike_loc.tail(n_spikes - 1) - spike_loc.head(n_spikes - 1);

            // get the maximum and index of the delta spike locations
            Eigen::Index mx_window_index;
            delta_spike_loc.maxCoeff(&mx_window_index);

            SPDLOG_INFO("delta_spike_loc {} fsmp {} mx_window_index {}",delta_spike_loc,fsmp,mx_window_index);
            SPDLOG_INFO("spike_loc(mx_window_index) {} spike_loc(mx_window_index-1) {}",spike_loc(mx_window_index),
                        spike_loc(mx_window_index-1));

            // set the starting and ending indices for the window
            if (mx_window_index == 0) {
                win_index_0 = 0;
                win_index_1 = spike_loc(1);// - fsmp;
                SPDLOG_INFO("a");
            }
            else {
                if (mx_window_index == n_spikes) {
                    win_index_0 = spike_loc(n_spikes - 1);
                    win_index_1 = n_pts;
                    SPDLOG_INFO("b");
                }
                // leave a 2 second region after the spike beginning the
                // window and a 1 second region before the spike ending the window
                else {
                    win_index_0 = spike_loc(mx_window_index - 1);// + 2 * fsmp;
                    win_index_1 = spike_loc(mx_window_index);// - fsmp;
                    SPDLOG_INFO("c");
                }
            }
        }
        else {
            if (n_pts - spike_loc(0) > spike_loc(0)) {
                win_index_0 = spike_loc(0) + 2 * fsmp;
                win_index_1 = n_pts - 1;
                SPDLOG_INFO("d");
            }
            else {
                win_index_0 = 0;
                win_index_1 = spike_loc(0) - fsmp;
                SPDLOG_INFO("e");
            }
        }

        SPDLOG_INFO("win_index_0 {}", win_index_0);
        SPDLOG_INFO("win_index_1 {}", win_index_1);

        // limit the maximum window size
        if ((win_index_1 - win_index_0 - 1) / fsmp > size) {
            win_index_1 = win_index_0 + size * fsmp + 1;
        }

        win_size = win_index_1 - win_index_0 - 1;

        if (win_index_0 > win_index_1) {
            SPDLOG_ERROR("despike failed: win_index_0 > win_index_1");
            SPDLOG_ERROR("scan size {}", n_pts);
            SPDLOG_ERROR("spike_loc {}", spike_loc.derived());
            SPDLOG_ERROR("win_index_0 {}", win_index_0);
            SPDLOG_ERROR("win_index_1 {}", win_index_1);
            SPDLOG_ERROR("win_size {}", win_size);

            std::exit(EXIT_FAILURE);
        }

        return std::tuple<int, int, int>(win_index_0, win_index_1, win_size);
    }
};

template <typename DerivedA, typename DerivedB, typename apt_t>
void Despiker::despike(Eigen::DenseBase<DerivedA> &scans,
                       Eigen::DenseBase<DerivedB> &flags,
                       apt_t &apt) {
    Eigen::Index n_pts = scans.rows();
    Eigen::Index n_dets = scans.cols();

    // loop through detectors
    for (Eigen::Index det=0; det<n_dets; det++) {
        // only run if detector is good
        if (apt["flag"](det)) {
            // get detector's flags
            Eigen::Matrix<bool, Eigen::Dynamic, 1> det_flags = flags.col(det);

            // total number of spikes
            int n_spikes = 0;

            // array of delta's between adjacent points
            Eigen::VectorXd delta = scans.col(det).tail(n_pts - 1) - scans.col(det).head(n_pts - 1);

            // minimum amplitude of spike
            double cutoff = min_spike_sigma * engine_utils::calc_std_dev(delta);

            // mean subtracted delta array
            Eigen::VectorXd diff = abs(delta.array() - delta.mean());

            // run the spike finder,
            spike_finder(det_flags, delta, diff, n_spikes, cutoff);

            // variable to control spike_finder while loop
            bool new_spikes_found = ((diff.segment(1,n_pts - 2).array() > cutoff).count() > 0) ? 1 : 0;

            // keep despiking recursively to remove effects on the mean from large spikes
            while (new_spikes_found) {
                // if no new spikes found, set new_found to zero to end while loop
                new_spikes_found = ((diff.segment(1,n_pts - 2).array() > cutoff).count() > 0) ? 1 : 0;

                // only run if there are spikes
                if (new_spikes_found) {
                    spike_finder(det_flags, delta, diff, n_spikes, cutoff);
                }
            }

            // count up the number of spikes
            n_spikes = (det_flags.head(n_pts - 1).array() == 0).count();

            // if there are other spikes within set number of samples after a spike, set only the
            // center value to be a spike
            for (Eigen::Index i = 0; i < n_pts; i++) {
                if (det_flags(i) == 0) {
                    int size_loop = size;
                    // check the size of the region to set un_flagged if a flag is found.
                    if ((n_pts - i - 1) < size_loop) {
                        SPDLOG_INFO("rng {} {} {}", (n_pts - i - 1), size,i );
                        size_loop = n_pts - i - 1;
                    }

                    // count up the flags in the region
                    int c = (det_flags.segment(i + 1, size_loop).array() == 0).count();

                    // if flags are found
                    if (c > 0) {
                        // remove those flags from the total count
                        n_spikes -= c;
                        // set region to un_flagged
                        det_flags.segment(i + 1, size_loop).setOnes();

                        // is this a bug?  if n_pts - i <= size/2, i + size/2 >= n_pts
                        // for now, let's limit it to i + size/2 < n_pts since the
                        // start and end of the scan are not used due to
                        // filtering
                        if ((i + size_loop/2) < n_pts) {
                            det_flags(i + size_loop/2) = 0;
                        }
                    }

                    // increment so we go to the next sample region
                    i = i + size_loop - 1;
                }
            }

            // now loop through if spikes were found
            if (n_spikes > 0) {
                // recount spikes
                n_spikes = (det_flags.head(n_pts - 1).array() == 0).count();

                SPDLOG_INFO("n_spikes 3 {} {}", n_spikes, det);

                // vector for spike indices
                Eigen::Matrix<int, Eigen::Dynamic, 1> spike_loc(n_spikes);
                // amplitude of spike
                Eigen::VectorXd spike_vals(n_spikes);

                // populate scan location and amplitude vectors
                int count = 0;
                for (Eigen::Index i = 0; i < n_pts - 1; i++) {
                    if (det_flags(i) == 0) {
                        spike_loc(count) = i + 1;
                        spike_vals(count) = scans(det,i+1) - scans(det,i);
                        count++;
                    }
                }

                SPDLOG_INFO("spike_loc {}", spike_loc);
                SPDLOG_INFO("spike_vals {}", spike_vals);
                SPDLOG_INFO("count {}", count);

                for (Eigen::Index i=0;i<spike_loc.size();i++) {
                    std::cerr << spike_loc(i) << "\n";
                }

                // get the largest window that is without spikes
                auto [win_index_0, win_index_1, win_size] =
                    make_window(spike_loc, n_spikes, n_pts);

                // create a sub-array with values from the largest spike-free window
                Eigen::VectorXd sub_vals =
                    scans.col(det).segment(win_index_0, win_size);

                // copy of the sub-array for smoothing
                Eigen::VectorXd smoothed_sub_vals = Eigen::VectorXd::Zero(win_size);

                // smooth the sub-array with a box-car filter
                engine_utils::smooth_boxcar(sub_vals, smoothed_sub_vals, size);

                // estimate the standard deviation
                sub_vals -= smoothed_sub_vals;
                auto sigest = engine_utils::calc_std_dev(sub_vals);
                if (sigest < sigest_lim) {
                    sigest = sigest_lim;
                }

                // calculate the decay length of all the spikes
                Eigen::VectorXd decay_length = -fsmp * time_constant_sec *
                                               ((sigest / spike_vals.array()).abs()).log();

                // if a decay length is less than 6, set it to 6
                decay_length = (decay_length.array() < 6.).select(6., decay_length);


                // exit if the decay length is too large
                if ((decay_length.array() > size * fsmp).any()) {
                    SPDLOG_INFO("decay length is longer than {} * fsmp.  mission "
                                "failed, we'll get em next time.",size);
                    std::exit(EXIT_FAILURE);
                }

                // due to filtering later, we flag a region with the size of the filter
                // around each spike
                if (run_filter) {
                    // half of the total despike window - 1
                    int decay_window = (window_size - 1) / 2;

                    for (Eigen::Index i=0; i<n_spikes; i++) {
                        if (spike_loc(i) - decay_window >= 0 &&
                            spike_loc(i) + decay_window + 1 < n_pts) {
                            det_flags
                                .segment(spike_loc(i) - decay_window, 2*window_size + 1)
                                .setZero();
                        }
                    }
                }

                // if lowpassing/highpassing skipped, use the decay length instead
                else {
                    for (Eigen::Index i=0; i<n_spikes; i++) {
                        if (spike_loc(i) - decay_length(i) >= 0 &&
                            spike_loc(i) + decay_length(i) + 1 < n_pts) {
                            det_flags
                                .segment(spike_loc(i) - decay_length(i), 2*decay_length(i) + 1)
                                .setZero();
                        }
                    }
                }

            } // end of "if (n_spikes > 0)" loop
            flags.col(det) = det_flags;
        } // end of apt["flag"] loop
    } // end of "for (Eigen::Index det = 0; det < n_dets; det++)" loop
}

template<typename DerivedA, typename DerivedB, typename apt_t>
void Despiker::replace_spikes(Eigen::DenseBase<DerivedA> &scans, Eigen::DenseBase<DerivedB> &flags,
                              apt_t &apt, Eigen::Index start_det) {

    // declare random number generator
    thread_local boost::random::mt19937 eng;

    Eigen::Index n_dets = flags.cols();
    Eigen::Index n_pts = flags.rows();

    // figure out if there are any flag-free detectors
    Eigen::Index n_flagged = 0;

    // if spike_free(detector) == 0, it contains a spike
    // otherwise none found
    auto spike_free = flags.colwise().minCoeff();
    n_flagged = n_dets - spike_free.template cast<int>().sum();

    SPDLOG_INFO("has spikes {}", spike_free);
    SPDLOG_INFO("n_flagged {}", n_flagged);

    for (Eigen::Index det = 0; det < n_dets; det++) {
        if (apt["flag"](det + start_det)) {
            if (!spike_free(det)) {
                // condition flags so that if there is a spike we can make
                // one long flagged or un-flagged region.
                // first do spikes from 0 to 1
                for (Eigen::Index j = 1; j < n_pts - 1; j++) {
                    if (flags(j, det) == 1 && flags(j - 1, det) == 0 && flags(j + 1, det) == 0) {
                        flags(j, det) = 0;
                    }
                }
                // now do spikes from 1 to 0
                for (Eigen::Index j = 1; j < n_pts - 1; j++) {
                    if (flags(j, det) == 0 && flags(j - 1, det) == 1 && flags(j + 1, det) == 1) {
                        flags(j, det) = 1;
                    }
                }
                // and the first and last samples
                flags(0, det) = flags(1, det);
                flags(n_pts - 1, det) = flags(n_pts - 2, det);

                // count up the number of flagged regions of data in the scan
                Eigen::Index n_flagged_regions = 0;

                if (flags(n_pts - 1, det) == 0) {
                    n_flagged_regions++;
                }

                n_flagged_regions
                    += ((flags.col(det).tail(n_pts - 1) - flags.col(det).head(n_pts - 1)).array() > 0)
                           .count()/ 2;
                if (n_flagged_regions == 0) {
                    break;
                }

                SPDLOG_INFO("n_flagged_regions {}",n_flagged_regions);

                // find the start and end index for each flagged region
                Eigen::Matrix<int, Eigen::Dynamic, 1> si_flags(n_flagged_regions);
                Eigen::Matrix<int, Eigen::Dynamic, 1> ei_flags(n_flagged_regions);

                si_flags.setConstant(-1);
                ei_flags.setConstant(-1);

                int count = 0;
                Eigen::Index j = 0;

                while (j < n_pts) {
                    if (flags(j, det) == 0) {
                        int jstart = j;
                        int samp_count = 0;

                        while (flags(j, det) == 0 && j <= n_pts - 1) {
                            samp_count++;
                            j++;
                        }
                        if (samp_count > 1) {
                            si_flags(count) = jstart;
                            ei_flags(count) = j - 1;
                            count++;
                        } else {
                            j++;
                        }
                    } else {
                        j++;
                    }
                }

                SPDLOG_INFO("count {}", count);

                // now loop on the number of flagged regions for the fix
                Eigen::VectorXd xx(2);
                Eigen::VectorXd yy(2);
                Eigen::Matrix<Eigen::Index, 1, 1> tn_pts;
                tn_pts << 2;

                for (Eigen::Index j = 0; j < n_flagged_regions; j++) {
                    // determine the linear baseline for flagged region
                    //but use flat level if flagged at endpoints
                    Eigen::Index n_flags = ei_flags(j) - si_flags(j);
                    Eigen::VectorXd lin_offset(n_flags);

                    if (si_flags(j) == 0) {
                        lin_offset.setConstant(scans(ei_flags(j) + 1, det));
                    }

                    else if (ei_flags(j) == n_pts - 1) {
                        lin_offset.setConstant(scans(si_flags(j) - 1, det));
                    }

                    else {
                        // linearly interpolate between the before and after good samples
                        xx(0) = si_flags(j) - 1;
                        xx(1) = ei_flags(j) + 1;
                        yy(0) = scans(si_flags(j) - 1, det);
                        yy(1) = scans(ei_flags(j) + 1, det);

                        Eigen::VectorXd xlin_offset =
                            Eigen::VectorXd::LinSpaced(n_flags, si_flags(j), si_flags(j) + n_flags - 1);

                        mlinterp::interp(tn_pts.data(), n_flags, yy.data(), lin_offset.data(), xx.data(),
                                         xlin_offset.data());
                        SPDLOG_INFO("xlin_offset {}", xlin_offset);
                    }

                    SPDLOG_INFO("xx {}", xx);
                    SPDLOG_INFO("yy {}", yy);
                    SPDLOG_INFO("lin_offset {}", lin_offset);

                    // all non-flagged detectors repeat for all detectors without spikes
                    // count up spike-free detectors and store their values
                    int det_count = 0;
                    if (use_all_det) {
                        det_count = (apt["flag"].segment(start_det,n_dets).array()==1).count();
                    }
                    else {
                        for (Eigen::Index ii=0;ii<n_dets;ii++) {
                            if(spike_free(ii) && apt["flag"](ii + start_det)) {
                                det_count++;
                            }
                        }
                        //det_count = spike_free.template cast<int>().sum();
                    }

                    SPDLOG_INFO("det_count {}", det_count);

                    Eigen::MatrixXd detm(n_flags, det_count);
                    detm.setConstant(-99);
                    Eigen::VectorXd res(det_count);

                    SPDLOG_INFO("si {}", si_flags);
                    int c = 0;
                    for (Eigen::Index ii = 0; ii < n_dets; ii++) {
                        if ((spike_free(ii) || use_all_det) && apt["flag"](ii + start_det)) {
                            detm.col(c) =
                                scans.block(si_flags(j), ii, n_flags, 1);
                            res(c) = apt["responsivity"](c + start_det);
                            c++;
                        }
                    }

                    detm.transposeInPlace();

                    SPDLOG_INFO("detm {}", detm);

                    // for each of these go through and redo the offset
                    Eigen::MatrixXd lin_offset_others(det_count, n_flags);

                    // first sample in scan is flagged so offset is flat
                    // with the value of the last sample in the flagged region
                    if (si_flags(j) == 0) {
                        lin_offset_others = detm.col(0).replicate(1, n_flags);
                    }

                    // last sample in scan is flagged so offset is flat
                    // with the value of the first sample in the flagged region
                    else if (ei_flags(j) == n_pts - 1) {
                        lin_offset_others = detm.col(n_flags - 1).replicate(1, n_flags);
                    }

                    else {
                        Eigen::VectorXd tmp_vec(n_flags);
                        Eigen::VectorXd xlin_offset =
                            Eigen::VectorXd::LinSpaced(n_flags, si_flags(j), si_flags(j) + n_flags - 1);

                        xx(0) = si_flags(j) - 1;
                        xx(1) = ei_flags(j) + 1;
                        // do we need this loop?
                        for (Eigen::Index ii = 0; ii < det_count; ii++) {
                            yy(0) = detm(ii, 0);
                            yy(1) = detm(ii, n_flags - 1);

                            mlinterp::interp(tn_pts.data(), n_flags, yy.data(), tmp_vec.data(), xx.data(),
                                             xlin_offset.data());
                            lin_offset_others.row(ii) = tmp_vec;
                        }

                        SPDLOG_INFO("xlin_offset {}", xlin_offset);
                    }

                    SPDLOG_INFO("xx {}", xx);
                    SPDLOG_INFO("yy {}", yy);
                    SPDLOG_INFO("lin_offset_others {}", lin_offset_others);

                    detm.noalias() = detm - lin_offset_others;

                    SPDLOG_INFO("detm {}", detm);

                    // scale det by responsivities and average to make sky model
                    Eigen::VectorXd sky_model = Eigen::VectorXd::Zero(n_flags);

                    //sky_model = sky_model.array() + (detm.array().colwise() / res.array()).rowwise().sum();
                    //sky_model /= det_count;

                    for (Eigen::Index ii=0; ii<det_count; ii++) {
                        for (Eigen::Index l=0; l<n_flags; l++) {
                            sky_model(l) += detm(ii,l)/res(ii);
                        }
                    }

                    sky_model = sky_model/det_count;

                    SPDLOG_INFO("sky_model {}",sky_model);

                    Eigen::VectorXd std_dev_ff = Eigen::VectorXd::Zero(det_count);

                    for (Eigen::Index ii = 0; ii < det_count; ii++) {
                        Eigen::VectorXd tmp_vec = detm.row(ii).array() / res.array() - sky_model.transpose().array();

                        double tmp_mean = tmp_vec.mean();

                        std_dev_ff(ii) = (tmp_vec.array() - tmp_mean).pow(2).sum();
                        std_dev_ff(ii) = (n_flags == 1.) ? std_dev_ff(ii) / n_flags
                                                        : std_dev_ff(ii) / (n_flags - 1.);

                    }

                    SPDLOG_INFO("std_dev_ff {}",std_dev_ff);

                    double mean_std_dev = (std_dev_ff.array().sqrt()).sum() / det_count;

                    // add noise to the fake signal
                    mean_std_dev *= apt["responsivity"](det + start_det); // not used

                    // boost random number generator
                    boost::random::normal_distribution<> rands{0, mean_std_dev};

                    Eigen::VectorXd error =
                        Eigen::VectorXd::Zero(n_flags).unaryExpr([&](double dummy){return rands(eng);});

                    SPDLOG_INFO("error {}", error);

                    // the noiseless fake data is then the sky model plus the
                    // flagged detectors linear offset
                    Eigen::VectorXd fake = sky_model.array() * apt["responsivity"](det + start_det) + lin_offset.array() + error.array();

                    SPDLOG_INFO("fake {}", fake);

                    SPDLOG_INFO("mean std dev {}", mean_std_dev);

                    scans.col(det).segment(si_flags(j), n_flags) = fake;
                } // flagged regions
            } // if it has spikes
        } // apt flag
    } // main detector loop
}

} // namespace timestream

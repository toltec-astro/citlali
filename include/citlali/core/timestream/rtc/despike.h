#pragma once

#include <Eigen/Core>

#include <tula/logging.h>

#include <tula/algorithm/mlinterp/mlinterp.hpp>
#include <citlali/core/utils/utils.h>

namespace timestream {

class Despiker {
public:
    // spike sigma, time constant, sample rate
    double sigma, time_constant, fsmp;
    // despike window
    int despike_window;
    bool run_filter = false;

    // the main despiking routine
    template <typename DerivedA, typename DerivedB>
    void despike(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB> &);

    // replace flagged data with interpolation
    template<typename DerivedA, typename DerivedB, typename DerivedC>
    void replace_spikes(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB>&, Eigen::DenseBase<DerivedC>&);

private:
    // this function loops through the delta timestream values and finds the
    // spikes setting the corresponding flags to zero and calculating nspikes
    template <typename DerivedA, typename DerivedB, typename DerivedC>
    void spike_finder(Eigen::DenseBase<DerivedA> const &f_,
                     Eigen::DenseBase<DerivedB> &delta,
                     Eigen::DenseBase<DerivedC> &diff, int &nspikes,
                     double &cutoff) {
        // since the flags we pass in here is a row (a reference) we need to
        // const_cast it
        Eigen::DenseBase<DerivedA> &flags =
            const_cast<Eigen::DenseBase<DerivedA> &>(f_);

        Eigen::Index npts = flags.size();

        // set flag matrix to zero if scan delta is nsigmaSpike times larger than
        // the standard deviaton
        flags.tail(npts - 2) =
            (diff.tail(npts - 2).array() > cutoff).select(0, flags.tail(npts - 2));

        // set corresponding delta values to zero
        delta.tail(npts - 2) =
            (diff.tail(npts - 2).array() > cutoff).select(0, delta.tail(npts - 2));

        // count up the number of spikes
        nspikes = (flags.tail(npts - 2).array() == 0).count();

        // update difference vector and cutoff
        diff = abs(delta.derived().array() - delta.mean());
        cutoff = sigma * engine_utils::stddev(delta);

    }

    template <typename Derived>
    auto make_window(Eigen::DenseBase<Derived> &spike_loc, int nspikes,
                    Eigen::Index npts) {
        // window first index, last index, and size
        int win_index_0, win_index_1, win_size;

        // find biggest spike-free window
        // first deal with the harder case of multiple spikes
        // remember that there are nspikes + 1 possible windows
        // only do if more than one spike
        if (nspikes > 1) {
            Eigen::Matrix<int, Eigen::Dynamic, 1> deltaspike_loc(nspikes + 1);
            // first element is the distance to first spike
            deltaspike_loc(0) = spike_loc(0);
            // last element is the distance to last spike
            deltaspike_loc(nspikes) = npts - spike_loc(nspikes - 1);
            // populate the delta spike location vector
            deltaspike_loc.segment(1, nspikes - 1) =
                spike_loc.tail(nspikes - 1) - spike_loc.head(nspikes - 1);

            // get the maximum and index of the delta spike locations
            Eigen::Index mx_window_index;
            deltaspike_loc.maxCoeff(&mx_window_index);

            // set the starting and ending indices for the window
            // leave a 2 second pad after the spike beginning the
            // window and a 1 second pad before the spike ending the window
            if (mx_window_index == 0) {
                win_index_0 = 0;
                win_index_1 = spike_loc(1) - fsmp;
            }
            else {
                if (mx_window_index == nspikes) {
                    win_index_0 = spike_loc(nspikes - 1);
                    win_index_1 = npts;
                }
                else {
                    win_index_0 = spike_loc(mx_window_index - 1) + 2 * fsmp;
                    win_index_1 = spike_loc(mx_window_index) - fsmp;
                }
            }
        }
        else {
            if (npts - spike_loc(0) > spike_loc(0)) {
                win_index_0 = spike_loc(0) + 2 * fsmp;
                win_index_1 = npts - 1;
            }
            else {
            win_index_0 = 0;
            win_index_1 = spike_loc(0) - fsmp;
            }
        }

        // limit the maximum window size to 10 samples
        if ((win_index_1 - win_index_0 - 1) / fsmp > 10.) {
            win_index_1 = win_index_0 + 10 * fsmp + 1;
        }

        win_size = win_index_1 - win_index_0 - 1;

        if (win_index_0 > win_index_1) {
            SPDLOG_ERROR("despike failed: win_index_0 > win_index_1");
            SPDLOG_ERROR("scan size {}", npts);
            SPDLOG_ERROR("spike_loc {}", spike_loc.derived());
            SPDLOG_ERROR("win_index_0 {}", win_index_0);
            SPDLOG_ERROR("win_index_1 {}", win_index_1);
            SPDLOG_ERROR("win_size {}", win_size);

            std::exit(EXIT_FAILURE);
        }

        return std::tuple<int, int, int>(win_index_0, win_index_1, win_size);
    }
  };

template <typename DerivedA, typename DerivedB>
void Despiker::despike(Eigen::DenseBase<DerivedA> &scans,
                     Eigen::DenseBase<DerivedB> &flags) {
    Eigen::Index npts = scans.rows();
    Eigen::Index ndet = scans.cols();

    // loop through detectors
    for (Eigen::Index det = 0; det < ndet; det++) {
        // total number of spikes
        int nspikes = 0;

        // array of delta's between adjacent points
        Eigen::VectorXd delta(npts - 1);
        delta = scans.col(det).tail(npts - 1) - scans.col(det).head(npts - 1);

        // minimum amplitude of spike
        auto cutoff = sigma * engine_utils::stddev(delta);

        // mean subtracted delta array
        Eigen::VectorXd diff = abs(delta.array() - delta.mean());

        // run the spike finder,
        spike_finder(flags.col(det), delta, diff, nspikes, cutoff);

        // variable to control spike_finder while loop
        bool new_found = 1;

        // keep despiking recursively to remove effects on the mean from large
        // spikes
        while (new_found == 1) {
            // if no new spikes found, set new_found to zero to end while loop
            new_found = ((diff.tail(npts - 2).array() > cutoff).count() > 0) ? 1 : 0;

            // only run if there are spikes
            if (new_found == 1) {
                spike_finder(flags.col(det), delta, diff, nspikes, cutoff);
            }
        }

        // if there are other spikes within 10 samples after a spike, set only the
        // center value to be a spike
        for (Eigen::Index i = 0; i < npts; i++) {
            if (flags.col(det)(i) == 0) {
                int size = 10;
                // check the size of the region to set un_flagged if a flag is found.
                // defaults to 10 samples if there's room before the end
                if ((npts - i) < size) {
                    size = npts - i - 1;
                }

                // count up the flags in the region
                auto c = (flags.col(det).segment(i + 1, size).array() == 0).count();

                // if flags are found
                if (c > 0) {
                    // remove those flags from the total count
                    nspikes -= c;
                    // set region to un_flagged
                    flags.col(det).segment(i + 1, size).setOnes();

                    // is this a bug?  if npts - i <= 5, i + 5 >= npts
                    // for now, let's limit it to i + 5 < npts since the
                    // start and end of the scan are not used due to
                    // filtering
                    if (i + 5 < npts) {
                        flags.col(det)(i + 5) = 0;
                    }
                }

                // increment so we go to the next 10 sample region
                i = i + 9;
            }
        }

        // now loop through if spikes were found
        if (nspikes > 0) {
            // recount spikes
            nspikes = (flags.col(det).head(npts - 1).array() == 0).count();

            // vector for spike indices
            Eigen::Matrix<int, Eigen::Dynamic, 1> spike_loc(nspikes);
            // amplitude of spike
            Eigen::VectorXd spike_vals(nspikes);

            // populate scan location and amplitude vectors
            int count = 0;
            for (Eigen::Index i = 0; i < npts - 1; i++) {
                if (flags.col(det)(i) == 0) {
                    spike_loc(count) = i + 1;
                    spike_vals(count) = scans.col(det)(i + 1) - scans.col(det)(i);
                    count++;
                }
            }

            // get the largest window that is without spikes
            auto [win_index_0, win_index_1, win_size] =
                make_window(spike_loc, nspikes, npts);

            // create a sub-array with values from the largest spike-free window
            Eigen::VectorXd sub_vals =
                scans.col(det).segment(win_index_0, win_size);

            // copy of the sub-array for smoothing
            Eigen::VectorXd smoothed_sub_vals = Eigen::VectorXd::Zero(win_size);

            // smooth the sub-array with a box-car filter (10 element is hardcoded)
            engine_utils::smooth(sub_vals, smoothed_sub_vals, 10);

            // estimate the standard deviation
            sub_vals -= smoothed_sub_vals;
            auto sigest = engine_utils::stddev(sub_vals);
            if (sigest < 1.e-8) {
                sigest = 1.e-8;
            }

            // calculate the decay length of all the spikes
            Eigen::VectorXd decay_length = -fsmp * time_constant *
                                        ((sigest / spike_vals.array()).abs()).log();

            // if a decay length is less than 6, set it to 6
            decay_length = (decay_length.array() < 6.).select(6., decay_length);

            // exit if the decay length is too large
            if ((decay_length.array() > 10. * fsmp).any()) {
                SPDLOG_INFO("decay length is longer than 10 * fsmp.  mission "
                            "failed, we'll get em next time.");
                std::exit(EXIT_FAILURE);
            }

            // due to filtering later, we flag a region with the size of the filter
            // around each spike
            if (run_filter) {

                // half of the total despike window - 1
                int decay_window = (despike_window - 1) / 2;

                for (Eigen::Index i = 0; i < nspikes; i++) {
                    if (spike_loc(i) - decay_window >= 0 &&
                        spike_loc(i) + decay_window + 1 < npts) {
                        flags.col(det)
                        .segment(spike_loc(i) - decay_window, 2*despike_window + 1)
                        .setZero();
                    }
                }
            }

            // if lowpassing/highpassing skipped, use the decay length instead
            else {
                for (Eigen::Index i = 0; i < nspikes; i++) {
                    if (spike_loc(i) - decay_length(i) >= 0 &&
                        spike_loc(i) + decay_length(i) + 1 < npts) {
                        flags.col(det)
                        .segment(spike_loc(i) - decay_length(i), 2*decay_length(i) + 1)
                        .setZero();
                    }
                }
            }

        } // end of "if (nspikes > 0)" loop
    }   // end of "for (Eigen::Index det = 0; det < ndet; det++)" loop
}

template<typename DerivedA, typename DerivedB, typename DerivedC>
void Despiker::replace_spikes(Eigen::DenseBase<DerivedA> &scans, Eigen::DenseBase<DerivedB> &flags,
                           Eigen::DenseBase<DerivedC>&responsivity) {

      // use all detectors in replacing flagged scans
      bool use_all_det = 1;
      Eigen::Index ndet = flags.cols();
      Eigen::Index npts = flags.rows();

      // figure out if there are any flag-free detectors
      Eigen::Index n_flagged = 0;

      // if has_spikes(detector) == 0, it contains a spike
      // otherwise none found
      auto has_spikes = flags.colwise().minCoeff();
      n_flagged = ndet - has_spikes.sum();

      for (Eigen::Index det = 0; det < ndet; det++) {
          if (!has_spikes(det)) {
              // condition flags so that if there is a spike we can make
              // one long flagged or un-flagged region.
              // first do spikes from 0 to 1
              for (Eigen::Index j = 1; j < npts - 1; j++) {
                  if (flags(j, det) == 1 && flags(j - 1, det) == 0 && flags(j + 1, det) == 0) {
                      flags(j, det) = 0;
                  }
              }
              // now do spikes from 1 to 0
              for (Eigen::Index j = 1; j < npts - 1; j++) {
                  if (flags(j, det) == 0 && flags(j - 1, det) == 1 && flags(j + 1, det) == 1) {
                      flags(j, det) = 1;
                  }
              }
              // and the first and last samples
              flags(0, det) = flags(1, det);
              flags(npts - 1, det) = flags(npts - 2, det);

              // count up the number of flagged regions of data in the scan
              Eigen::Index n_flagged_regions = 0;

              if (flags(npts - 1, det) == 0) {
                  n_flagged_regions++;
              }

              n_flagged_regions
                  += ((flags.col(det).tail(npts - 1) - flags.col(det).head(npts - 1)).array() > 0)
                         .count()/ 2;
              if (n_flagged_regions == 0) {
                  break;
              }

              // find the start and end index for each flagged region
              Eigen::Matrix<int, Eigen::Dynamic, 1> si_flags(n_flagged_regions);
              Eigen::Matrix<int, Eigen::Dynamic, 1> ei_flags(n_flagged_regions);

              si_flags.setConstant(-1);
              ei_flags.setConstant(-1);

              int count = 0;
              Eigen::Index j = 0;

              while (j < npts) {
                  if (flags(j, det) == 0) {
                      int jstart = j;
                      int samp_count = 0;

                      while (flags(j, det) == 0 && j <= npts - 1) {
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

              // now loop on the number of flagged regions for the fix
              Eigen::VectorXd xx(2);
              Eigen::VectorXd yy(2);
              Eigen::Matrix<Eigen::Index, 1, 1> tnpts;
              tnpts << 2;

              for (Eigen::Index j = 0; j < n_flagged_regions; j++) {
                  // determine the linear baseline for flagged region
                  //but use flat level if flagged at endpoints
                  Eigen::Index nflags = ei_flags(j) - si_flags(j);
                  Eigen::VectorXd lin_offset(nflags);

                  if (si_flags(j) == 0) {
                      lin_offset.setConstant(scans(ei_flags(j) + 1, det));
                  }

                  else if (ei_flags(j) == npts - 1) {
                      lin_offset.setConstant(scans(si_flags(j) - 1, det));
                  }

                  else {
                      // linearly interpolate between the before
                      // and after good samples
                      xx(0) = si_flags(j) - 1;
                      xx(1) = ei_flags(j) + 1;
                      yy(0) = scans(si_flags(j) - 1, det);
                      yy(1) = scans(ei_flags(j) + 1, det);

                      Eigen::VectorXd xlin_offset =
                              Eigen::VectorXd::LinSpaced(nflags, si_flags(j), si_flags(j) + nflags - 1);

                      mlinterp::interp(tnpts.data(), nflags, yy.data(), lin_offset.data(), xx.data(),
                                       xlin_offset.data());
                  }

                  // all non-flagged detectors
                  // repeat for all detectors without spikes
                  // count up spike-free detectors and store their values
                  int det_count = 0;
                  if (use_all_det) {
                      det_count = ndet;
                  }
                  else {
                      det_count = has_spikes.sum();
                  }

                  Eigen::MatrixXd detm(det_count, nflags);
                  Eigen::VectorXd res(det_count);
                  int c = 0;
                  for (Eigen::Index ii = 0; ii < ndet; ii++) {
                      if (has_spikes(ii) || use_all_det) {
                          detm.row(c) =
                                  Eigen::VectorXd::LinSpaced(nflags, si_flags(j), si_flags(j) + nflags - 1);
                          res(c) = responsivity(c);
                          c++;
                      }
                  }

                  // for each of these go through and redo the offset bit
                  Eigen::MatrixXd lin_offset_others(det_count, nflags);

                  // first sample in scan is flagged so offset is flat
                  // with the value of the last sample in the flagged region
                  if (si_flags(j) == 0) {
                      lin_offset_others = detm.col(0).replicate(1, nflags);
                  }

                  // last sample in scan is flagged so offset is flat
                  // with the value of the first sample in the flagged region
                  else if (ei_flags(j) == npts - 1) {
                      lin_offset_others = detm.col(nflags - 1).replicate(1, nflags);
                  }

                  else {
                      Eigen::VectorXd tmp_vec(nflags);
                      Eigen::VectorXd xlin_offset =
                              Eigen::VectorXd::LinSpaced(nflags, si_flags(j), si_flags(j) + nflags - 1);

                      xx(0) = si_flags(j) - 1;
                      xx(1) = ei_flags(j) + 1;
                      // do we need this loop?
                      for (Eigen::Index ii = 0; ii < det_count; ii++) {
                          yy(0) = detm(ii, 0);
                          yy(0) = detm(ii, nflags - 1);

                          mlinterp::interp(tnpts.data(), nflags, yy.data(), tmp_vec.data(), xx.data(),
                                           xlin_offset.data());
                          lin_offset_others.row(ii) = tmp_vec;
                      }
                  }

                  detm = detm - lin_offset_others;

                  // scale det by responsivities and average to make sky model
                  Eigen::VectorXd sky_model = Eigen::VectorXd::Zero(nflags);

                  sky_model = sky_model.array() + (detm.array().colwise() / res.array()).rowwise().sum();
                  sky_model /= det_count;

                  Eigen::VectorXd std_dev_ff = Eigen::VectorXd::Zero(det_count);
                  double tmp_mean;
                  Eigen::VectorXd tmp_vec(nflags);

                  for (Eigen::Index ii = 0; ii < det_count; ii++) {
                      tmp_vec = (detm.array().colwise() / res.array() - sky_model.array());
                      tmp_mean = tmp_vec.mean();
                      std_dev_ff(ii) = (tmp_vec.array() - tmp_mean).pow(2).sum();

                      std_dev_ff(ii) = (nflags == 1.) ? std_dev_ff(ii) / nflags
                                                    : std_dev_ff(ii) / (nflags - 1.);
                  }

                  double mean_std_dev = (std_dev_ff.array().sqrt()).sum() / det_count;

                  // the noiseless fake data is then the sky model plus the
                  // flagged detectors linear offset
                  Eigen::VectorXd fake(nflags);
                  fake = sky_model.array() * responsivity(det) + lin_offset.array();

                  // add noise to the fake signal
                  mean_std_dev *= responsivity(det); // not used
                  scans.col(det).segment(si_flags(j), nflags) = fake;
               }
          }
     }
}

} // namespace

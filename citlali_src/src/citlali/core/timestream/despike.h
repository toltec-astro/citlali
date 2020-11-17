#pragma once

namespace timestream {

class Despiker {
public:
  Despiker(const double nss, const double tc, const double sr, const int dw)
      : nsigmaSpikes(nss), timeconstant(tc), samplerate(sr), despikewindow(dw) {
  }

  // despike parameters
  const double nsigmaSpikes, timeconstant, samplerate;
  const int despikewindow;
  bool useAllDet = 1;

  // The main despiking routine
  template <typename DerivedA, typename DerivedB>
  void despike(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB> &);

  // Replace flagged data with interpolation
  template<typename DerivedA, typename DerivedB, typename DerivedC>
  void replaceSpikes(Eigen::DenseBase<DerivedA>&, Eigen::DenseBase<DerivedB>&, Eigen::DenseBase<DerivedC>&);

private:
  // This function loops through the delta timestream values and finds the
  // spikes setting the corresponding flags to zero and calculating nspikes
  template <typename DerivedA, typename DerivedB, typename DerivedC>
  void spikefinder(Eigen::DenseBase<DerivedA> const &f_,
                   Eigen::DenseBase<DerivedB> &delta,
                   Eigen::DenseBase<DerivedC> &diff, int &nspikes,
                   double cutoff) {
    // Since the flags we pass in here is a row (a reference) we need to
    // const_cast it
    Eigen::DenseBase<DerivedA> &flags =
        const_cast<Eigen::DenseBase<DerivedA> &>(f_);

    Eigen::Index npts = flags.size();

    // Set flag matrix to zero if scan delta is nsigmaSpike times larger than
    // the standard deviaton
    flags.tail(npts - 2) =
        (diff.tail(npts - 2).array() > cutoff).select(0, flags.tail(npts - 2));

    // Set corresponding delta values to zero
    delta.tail(npts - 2) =
        (diff.tail(npts - 2).array() > cutoff).select(0, delta.tail(npts - 2));

    // Count up the number of spikes
    nspikes = (flags.tail(npts - 2).array() == 0).count();
  }

  template <typename Derived>
  auto makewindow(Eigen::DenseBase<Derived> &spikeLoc, int nspikes,
                  Eigen::Index npts) {
    // Window first index, last index, and size
    int winIndex0, winIndex1, winSize;

    // Only do if more than one spike
    if (nspikes > 1) {
      Eigen::Matrix<int, Eigen::Dynamic, 1> deltaSpikeLoc(nspikes + 1);
      // First element is the distance to first spike
      deltaSpikeLoc(0) = spikeLoc(0);
      // Last element is the distance to last spike
      deltaSpikeLoc(nspikes) = npts - spikeLoc(nspikes - 1);
      // Populate the delta spike location vector
      deltaSpikeLoc.segment(1, nspikes - 1) =
          spikeLoc.tail(nspikes - 1) - spikeLoc.head(nspikes - 1);

      // Get the maximum and index of the delta spike locations
      Eigen::Index mxWindowIndex;
      auto mxWindow = deltaSpikeLoc.maxCoeff(&mxWindowIndex);

      if (mxWindowIndex == 0) {
        winIndex0 = 0;
        winIndex1 = spikeLoc(1) - samplerate;
      } else {
        if (mxWindowIndex == nspikes) {
          winIndex0 = spikeLoc(nspikes - 1);
          winIndex1 = npts;
        } else {
          winIndex0 = spikeLoc(mxWindowIndex - 1) + 2 * samplerate;
          winIndex1 = spikeLoc(mxWindowIndex) - 1 * samplerate;
        }
      }
    } else {
      if (npts - spikeLoc(0) > spikeLoc(0)) {
        winIndex0 = spikeLoc(0) + 2 * samplerate;
        winIndex1 = npts - 1;
      } else {
        winIndex0 = 0;
        winIndex1 = spikeLoc(0) - samplerate;
      }
    }
    // Limit the maximum window size to 10 samples
    if ((winIndex1 - winIndex0 - 1) / samplerate > 10.) {
      winIndex1 = winIndex0 + 10 * samplerate + 1;
    }

    winSize = winIndex1 - winIndex0 - 1;

    return std::tuple<int, int, int>(winIndex0, winIndex1, winSize);
  }
};

template <typename DerivedA, typename DerivedB>
void Despiker::despike(Eigen::DenseBase<DerivedA> &scans,
                       Eigen::DenseBase<DerivedB> &flags) {
  Eigen::Index npts = scans.rows();
  Eigen::Index ndetectors = scans.cols();

  // Loop through detectors
  for (Eigen::Index det = 0; det < ndetectors; det++) {
    // Total number of spikes
    int nspikes = 0;

    // Array of delta's between adjacent points
    Eigen::VectorXd delta(npts - 1);
    delta = scans.col(det).tail(npts - 1) - scans.col(det).head(npts - 1);

    // Minimum amplitude of spike
    auto cutoff = nsigmaSpikes * timestream_utils::stddev(delta);

    // Mean subtracted delta array
    auto diff = abs(delta.array() - delta.mean());

    // Run the spike finder,
    spikefinder(flags.col(det), delta, diff, nspikes, cutoff);

    // Variable to control spikefinder while loop
    bool newfound = 1;

    // Keep despiking recursively to remove effects on the mean from large
    // spikes
    while (newfound == 1) {
      // If no new spikes found, set newfound to zero to end while loop
      newfound = ((diff.tail(npts - 2).array() > cutoff).count() > 0) ? 1 : 0;

      // Only run if there are spikes
      if (newfound == 1) {
        spikefinder(flags.col(det), delta, diff, nspikes, cutoff);
      }
    }

    // If there are other spikes within 10 samples after a spike, set only the
    // center value to be a spike
    for (int i = 0; i < npts; i++) {
      if (flags.col(det)(i) == 0) {
        int size = 10;
        // Check the size of the region to set unflagged if a flag is found.
        // Defaults to 10 samples if there's room before the end
        if ((npts - i) < size) {
          size = npts - i - 1;
        }

        // Count up the flags in the region
        auto c = (flags.col(det).segment(i + 1, size).array() == 0).count();

        // If flags are found
        if (c > 0) {
          // Remove those flags from the total count
          nspikes -= c;
          // Set region to unflagged
          flags.col(det).segment(i + 1, size).setOnes();

          // Is this a bug?  if npts - i <= 5, i + 5 >= npts
          // For now, let's limit it to i + 5 < npts since the
          // start and end of the scan are not used due to
          // filtering
          if (i + 5 < npts) {
            flags.col(det)(i + 5) = 0;
          }
        }

        // Increment so we go to the next 10 sample region
        i = i + 9;
      }
    }

    // Now loop through if spikes were found
    if (nspikes > 0) {
      nspikes = (flags.col(det).head(npts - 1).array() == 0).count();

      // Vector for spike indices
      Eigen::Matrix<int, Eigen::Dynamic, 1> spikeLoc(nspikes);
      // Amplitude of spike
      Eigen::VectorXd spikeVals(nspikes);

      // Populate scan location and amplitude vectors
      int count = 0;
      for (int i = 0; i < npts - 1; i++) {
        if (flags.col(det)(i) == 0) {
          spikeLoc(count) = i + 1;
          spikeVals(count) = scans.col(det)(i + 1) - scans.col(det)(i);
          count++;
        }
      }

      // Get the largest window that is without spikes
      auto [winIndex0, winIndex1, winSize] =
          makewindow(spikeLoc, nspikes, npts);

      // Create a sub-array with values from the largest spike-free window
      Eigen::VectorXd subVals =
          scans.col(det).segment(winIndex0, winIndex1 - winIndex0 - 1);

      // Copy of th sub-array for smoothing
      Eigen::VectorXd smoothedSubVals = Eigen::VectorXd::Zero(winSize);

      // Smooth the sub-array with a box-car filter
      timestream_utils::smooth(subVals, smoothedSubVals, 10);

      // Estimate the standard deviation
      subVals -= smoothedSubVals;
      auto sigest = timestream_utils::stddev(subVals);
      if (sigest < 1.e-8) {
        sigest = 1.e-8;
      }

      // Calculate the decay length of all the spikes
      Eigen::VectorXd decayLength = -samplerate * timeconstant *
                                    ((sigest / spikeVals.array()).abs()).log();

      // If a decay length is less than 6, set it to 6
      decayLength = (decayLength.array() < 6).select(6, decayLength);

      // Exit if the decay length is too large
      if ((decayLength.array() > 10. * samplerate).any()) {
        SPDLOG_INFO("Decay length is longer than 10 * samplerate.  Mission "
                    "failed, we'll get "
                    "em next time.");
        exit(1);
      }

      // Half of the total despike window - 1
      auto decayWindow = (despikewindow - 1) / 2;

      // Due to filtering later, we flag a region with the size of the filter
      // around each spike
      for (int i = 0; i < nspikes; i++) {
        if (spikeLoc(i) - decayWindow - 1 >= 0 &&
            spikeLoc(i) + decayWindow < npts) {
          flags.col(det)
              .segment(spikeLoc(i) - decayWindow - 1, despikewindow)
              .setZero();
        }
      }

    } // end of "if (nspikes > 0)" loop
  }   // end of "for (Eigen::Index det = 0; det < ndetectors; det++)" loop
}

template<typename DerivedA, typename DerivedB, typename DerivedC>
void Despiker::replaceSpikes(Eigen::DenseBase<DerivedA> &scans, Eigen::DenseBase<DerivedB> &flags,
                             Eigen::DenseBase<DerivedC>&responsivity)
{
    Eigen::Index ndetectors = flags.cols();
    Eigen::Index npts = flags.rows();

    // Figure out if there are any flag-free detectors
    Eigen::Index nFlagged = 0;

    // If hasSpikes(detector) == 0, it contains a spike
    // otherwise none found
    auto hasSpikes = flags.colwise().minCoeff();
    nFlagged = ndetectors - hasSpikes.sum();

    for (Eigen::Index det = 0; det < ndetectors; det++) {
        if (!hasSpikes(det)) {
            // Condition flags so that if there is a spike we can make
            //one long flagged or unflagged region.
            //first do spikes from 0 to 1
            for (Eigen::Index j = 1; j < npts - 1; j++) {
                if (flags(j, det) == 1 && flags(j - 1, det) == 0 && flags(j + 1, det) == 0) {
                    flags(j, det) = 0;
                }
            }
            //now spikes from 1 to 0
            for (Eigen::Index j = 1; j < npts - 1; j++) {
                if (flags(j, det) == 0 && flags(j - 1, det) == 1 && flags(j + 1, det) == 1) {
                    flags(j, det) = 1;
                }
            }
            //and the first and last samples
            flags(0, det) = flags(1, det);
            flags(npts - 1, det) = flags(npts - 2, det);

            // Count up the number of flagged regions of data in the scan
            Eigen::Index nFlaggedRegions = 0;

            if (flags(npts - 1, det) == 0) {
                nFlaggedRegions++;
            }

            nFlaggedRegions
                += ((flags.col(det).tail(npts - 1) - flags.col(det).head(npts - 1)).array() > 0)
                       .count()
                   / 2;
            if (nFlaggedRegions == 0) {
                break;
            }

            // Find the start and end index for each flagged region
            Eigen::Matrix<int, Eigen::Dynamic, 1> siFlags(nFlaggedRegions);
            Eigen::Matrix<int, Eigen::Dynamic, 1> eiFlags(nFlaggedRegions);

            siFlags.setConstant(-1);
            eiFlags.setConstant(-1);

            int count = 0;
            Eigen::Index j = 0;

            while (j < npts) {
                if (flags(j, det) == 0) {
                    int jstart = j;
                    int sampcount = 0;

                    while (flags(j, det) == 0 && j <= npts - 1) {
                        sampcount++;
                        j++;
                    }
                    if (sampcount > 1) {
                        siFlags(count) = jstart;
                        eiFlags(count) = j - 1;
                        count++;
                    } else {
                        j++;
                    }
                } else {
                    j++;
                }
            }

            // Now loop on the number of flagged regions for the fix
            Eigen::VectorXd xx(2);
            Eigen::VectorXd yy(2);
            Eigen::Matrix<Eigen::Index, 1, 1> tnpts;
            tnpts << 2;

            for (int j = 0; j < nFlaggedRegions; j++) {
                // Determine the linear baseline for flagged region
                //but use flat level if flagged at endpoints
                Eigen::Index nFlags = eiFlags(j) - siFlags(j);
                Eigen::VectorXd linOffset(nFlags);

                if (siFlags(j) == 0) {
                    linOffset.setConstant(scans(eiFlags(j) + 1, det));
                } else if (eiFlags(j) == npts - 1) {
                    linOffset.setConstant(scans(siFlags(j) - 1, det));
                } else {
                    // Linearly interpolate between the before
                    //and after good samples
                    xx(0) = siFlags(j) - 1;
                    xx(1) = eiFlags(j) + 1;
                    yy(0) = scans(siFlags(j) - 1, det);
                    yy(1) = scans(eiFlags(j) + 1, det);

                    Eigen::VectorXd xLinOffset = Eigen::VectorXd::LinSpaced(nFlags,
                                                                            siFlags(j),
                                                                            siFlags(j) + nFlags - 1);

                    mlinterp::interp(tnpts.data(),
                                     nFlags,
                                     yy.data(),
                                     linOffset.data(),
                                     xx.data(),
                                     xLinOffset.data());
                }

                // All non-flagged detectors
                // Repeat for all detectors without spikes
                // Count up spike-free detectors and store their values

                int detCount = 0;
                if (useAllDet) {
                    detCount = ndetectors;
                } else {
                    detCount = hasSpikes.sum();
                }

                Eigen::MatrixXd detm(detCount, nFlags);
                Eigen::VectorXd res(detCount);
                int c = 0;
                for (int ii = 0; ii < ndetectors; ii++) {
                    if (hasSpikes(ii) || useAllDet) {
                        detm.row(c) = Eigen::VectorXd::LinSpaced(nFlags,
                                                                 siFlags(j),
                                                                 siFlags(j) + nFlags - 1);
                        res(c) = 1;
                        c++;
                    }
                }

                // For each of these go through and redo the offset bit
                Eigen::MatrixXd linOffsetOthers(detCount, nFlags);

                if (siFlags(j) == 0) {
                    // First sample in scan is flagged so offset is flat
                    // with the value of the last sample in the flagged region

                    linOffsetOthers = detm.col(0).replicate(1, nFlags);

                } else if (eiFlags(j) == npts - 1) {
                    // Last sample in scan is flagged so offset is flat
                    // with the value of the first sample in the flagged region
                    linOffsetOthers = detm.col(nFlags - 1).replicate(1, nFlags);

                } else {
                    Eigen::VectorXd tmpVec(nFlags);
                    Eigen::VectorXd xLinOffset = Eigen::VectorXd::LinSpaced(nFlags,
                                                                            siFlags(j),
                                                                            siFlags(j) + nFlags - 1);

                    xx(0) = siFlags(j) - 1;
                    xx(1) = eiFlags(j) + 1;
                    // Do we need this loop?
                    for (int ii = 0; ii < detCount; ii++) {
                        yy(0) = detm(ii, 0);
                        yy(0) = detm(ii, nFlags - 1);

                        mlinterp::interp(tnpts.data(),
                                         nFlags,
                                         yy.data(),
                                         tmpVec.data(),
                                         xx.data(),
                                         xLinOffset.data());
                        linOffsetOthers.row(ii) = tmpVec;

                    }
                }

                detm = detm - linOffsetOthers;

                // Scale det by responsivities and average to make sky model
                Eigen::VectorXd skyModel = Eigen::VectorXd::Zero(nFlags);

                skyModel = skyModel.array() + (detm.array().colwise() / res.array()).rowwise().sum();
                skyModel /= detCount;

                Eigen::VectorXd stdDevFF = Eigen::VectorXd::Zero(detCount);
                double tmpMean;
                Eigen::VectorXd tmpVec(nFlags);

                for (int ii = 0; ii < detCount; ii++) {
                    tmpVec = (detm.array().colwise() / res.array() - skyModel.array());
                    tmpMean = tmpVec.mean();
                    stdDevFF(ii) = (tmpVec.array() - tmpMean).pow(2).sum();

                    stdDevFF(ii) = (nFlags == 1.) ? stdDevFF(ii) / nFlags
                                                  : stdDevFF(ii) / (nFlags - 1.);
                }

                double meanStdDev = (stdDevFF.array().sqrt()).sum() / detCount;

                // The noiseless fake data is then the sky model plus the
                // flagged detectors linear offset
                Eigen::VectorXd fake(nFlags);
                fake = skyModel.array() * responsivity(det) + linOffset.array();

                // Add noise to the fake signal
                meanStdDev *= responsivity(det); // not used
                scans.col(det).segment(siFlags(j), nFlags) = fake;
             }
        }
    }
}

} // namespace timestream

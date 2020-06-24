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

  // The main despiking routine
  template <typename DerivedA, typename DerivedB>
  void despike(Eigen::DenseBase<DerivedA> &, Eigen::DenseBase<DerivedB> &);

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

} // namespace timestream

#include <Eigen/Core>
#include <optional>

#include <citlali/core/timestream/ptc/sensitivity.h>

namespace internal {

std::tuple<Eigen::Index, Eigen::Index, double> stat(Eigen::Index scanlength, double fsmp) {
    // make an even number of data points by rounding-down
    Eigen::Index npts = scanlength;
    if (npts % 2 == 1)
        npts--;
    // prepare containers in frequency domain
    Eigen::Index nfreqs = npts / 2 + 1; // number of one sided freq bins
    double dfreq = fsmp / npts;

    return {npts, nfreqs, dfreq};
}

Eigen::VectorXd freq(Eigen::Index npts, Eigen::Index nfreqs, double dfreq) {
    return dfreq * Eigen::VectorXd::LinSpaced(nfreqs, 0, npts / 2);
}

} // namespace internal

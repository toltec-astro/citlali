#pragma once

// Beammap detector table fit-QC vector helpers.

#include <Eigen/Core>

#include <cmath>

#include <citlali/core/engine/detail/beammap_detector_table_common_vectors.h>

namespace beammap_detector_table_vectors {

template <class Params, class PErrors, class MapBuffer, class MapRmsFunc>
FitQCSignalVectors fit_qc_signal_vectors(const Params &params,
                                         const PErrors &perrors,
                                         const MapBuffer &omb,
                                         Eigen::Index n_dets,
                                         MapRmsFunc map_rms_for_detector) {
    FitQCSignalVectors vectors{
        Eigen::VectorXd::Zero(n_dets),
        Eigen::VectorXd::Zero(n_dets),
        Eigen::VectorXd::Zero(n_dets),
        Eigen::VectorXd::Zero(n_dets)};
    for (Eigen::Index i = 0; i < n_dets; ++i) {
        const double amp = params(i, 0);
        const double amp_err = perrors(i, 0);
        const double rms = map_rms_for_detector(i);
        const double npos =
            static_cast<double>((omb.weight[i].array() > 0.0).count());
        vectors.n_weight_pos(i) = npos;
        if (std::isfinite(rms) && rms > 0.0) {
            vectors.map_rms(i) = rms;
            if (std::isfinite(amp)) {
                vectors.map_sig2noise(i) = amp / rms;
            }
        }
        if (std::isfinite(amp) && std::isfinite(amp_err) && amp_err > 0.0) {
            vectors.fit_sig2noise(i) = amp / amp_err;
        }
    }
    return vectors;
}

} // namespace beammap_detector_table_vectors

#pragma once

#include <Eigen/Core>

namespace timestream {

class Calibration {
public:
    template <typename Derived, class calib_t>
    void calibrate_tod(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, calib_t &);
    void calc_tau();

};

template <typename Derived, class calib_t>
void Calibration::calibrate_tod(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &det_indices,
                                Eigen::DenseBase<Derived> &array_indices, calib_t &calib) {

    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
        Eigen::Index det_index = det_indices(i);
        Eigen::Index array_index = array_indices(i);

        in.scans.data.col(i) = in.scans.data.col(i)*calib.apt["flxscale"](det_index)*calib.flux_conversion_factor(array_index);
    }
}

} // namespace timestream

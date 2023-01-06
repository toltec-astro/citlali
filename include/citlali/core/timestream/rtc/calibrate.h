#pragma once

#include <map>
#include <string>
#include <Eigen/Core>

#include <citlali/core/utils/constants.h>

namespace timestream {

class Calibration {
public:
    // extinction model
    std::string extinction_model;

    // coefficients for transmission ratios fit to order 4 polynomial
    std::map<std::string,Eigen::VectorXd> tx_ratio_coeff;

    void setup() {
        tx_ratio_coeff["a1100"].resize(5);
        tx_ratio_coeff["a1400"].resize(5);
        tx_ratio_coeff["a2000"].resize(5);

        // am_q25
        if (extinction_model == "am_q25") {
            tx_ratio_coeff["a1100"] << -7.09414810e-09,  1.73486174e-06, -1.63050022e-04,  7.23047174e-03,
                                       8.44501602e-01;
            tx_ratio_coeff["a1400"] << 1.48670310e-09, -3.62353172e-07,  3.38625356e-05, -1.48802341e-03,
                                       1.03135867e+00;
            tx_ratio_coeff["a2000"] << 9.22610521e-09, -2.24323135e-06,  2.08778592e-04, -9.11422313e-03,
                                       1.18925417e+00;
        }

        // am_q50
        if (extinction_model == "am_q50") {
            tx_ratio_coeff["a1100"] << -1.13658979e-08, 2.78526091e-06, -2.62689766e-04, 1.17142504e-02,
                                        7.44943260e-01;
            tx_ratio_coeff["a1400"] << 2.42686787e-09, -5.91310593e-07, 5.52294555e-05, -2.42486849e-03,
                                       1.05100321e+00;
            tx_ratio_coeff["a2000"] << 1.88481829e-08, -4.57135777e-06, 4.23676405e-04, -1.83705149e-02,
                                       1.37564570e+00;
        }

        // am_q75
        if (extinction_model == "am_q75") {
            tx_ratio_coeff["a1100"] << -1.79196947e-08,  4.40909853e-06, -4.18703056e-04,  1.88754214e-02,
                                       5.79135815e-01;
            tx_ratio_coeff["a1400"] << 4.25816993e-09, -1.03687212e-06,  9.67458504e-05, -4.24066113e-03,
                                       1.08886782e+00;
            tx_ratio_coeff["a2000"] << 4.06780486e-08, -9.82328968e-06,  9.03824568e-04, -3.87289340e-02,
                                       1.77085296e+00;
        }
    }

    template <typename DerivedA, typename DerivedB>
    auto tau_polynomial(Eigen::DenseBase<DerivedA> &coeff, Eigen::DenseBase<DerivedB> &elev) {

        return coeff(0)*pow(elev.derived().array(),4) + coeff(1)*pow(elev.derived().array(),3) +
               coeff(2)*pow(elev.derived().array(),2) + coeff(3)*pow(elev.derived().array(),1) +
               coeff(4);
    }

    template <typename Derived>
    auto calc_tau(Eigen::DenseBase<Derived> &, double);

    template <typename Derived, class calib_t, typename tau_t>
    void calibrate_tod(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &,
                       Eigen::DenseBase<Derived> &, calib_t &, tau_t &);
};

template <typename Derived>
auto Calibration::calc_tau(Eigen::DenseBase<Derived> &elev, double tau_225_GHz) {

    // tau at toltec freqs
    std::map<int,Eigen::VectorXd> tau_freq;

    // zenith angle
    auto cz = cos(pi/2 - elev.derived().array());

    // David Tholen’s approximation for airmass
    //auto A = sqrt(235225.0*cz*cz + 970.0 + 1.0) - 485*cz;

    auto secz = 1. / cos(cz);
    auto A = secz * (1. - 0.0012 * (pow(secz.array(), 2) - 1.));

    // observed 225 GHz tau
    auto obs_tau_i = A*tau_225_GHz;

    // 225 GHz transmission
    auto tx = (-obs_tau_i).exp();

    // a1100
    auto tx_a1100 = tau_polynomial(tx_ratio_coeff["a1100"],elev).array()*tx.array();
    tau_freq[0] = -(tx_a1100.array().log())/A.array();

    // a1400
    auto tx_a1400 = tau_polynomial(tx_ratio_coeff["a1400"],elev)*tx.array();
    tau_freq[1] = -(tx_a1400.array().log())/A.array();

    // a2000
    auto tx_a2000 = tau_polynomial(tx_ratio_coeff["a2000"],elev)*tx.array();
    tau_freq[2] = -(tx_a2000.array().log())/A.array();

    return tau_freq;
}

template <typename Derived, class calib_t, typename tau_t>
void Calibration::calibrate_tod(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &det_indices,
                                Eigen::DenseBase<Derived> &array_indices, calib_t &calib, tau_t &tau_freq) {

    // loop through detectors
    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
        // current detector index in apt table
        Eigen::Index det_index = det_indices(i);
        // current array index in apt table
        Eigen::Index array_index = array_indices(i);

        // factor = flux conversion factor / exp(-tau_freq)
        auto factor = calib.flux_conversion_factor(array_index)/(-tau_freq[array_index]).array().exp();

        // flux calibration factor for sens
        in.fcf.data(i) = factor.mean()*calib.apt["flxscale"](det_index);

        // data x flxscale x factor
        in.scans.data.col(i) = in.scans.data.col(i).array()*factor.array()*calib.apt["flxscale"](det_index);
    }
}

} // namespace timestream

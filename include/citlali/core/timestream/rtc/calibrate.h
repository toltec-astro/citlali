#pragma once

#include <map>
#include <string>
#include <Eigen/Core>

#include <citlali/core/utils/constants.h>
#include <citlali/core/timestream/timestream.h>

namespace timestream {

class Calibration {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // extinction model
    std::string extinction_model;

    // coefficients for transmission ratios fit to order 4 polynomial
    std::map<std::string,Eigen::VectorXd> tx_ratio_coeff;

    std::map<std::string, double> tx_225_zenith = {
        {"am_q25",0.9500275},
        {"am_q50",0.9142065},
        {"am_q75",0.8515054},
        {"am_q95",0.7337698},
    };

    void setup(double tau_225_zenith) {
        // cos of zenith angle
        auto cz = cos(pi/2 - 80.0*DEG_TO_RAD);
        // 1/cos(zenith angle)
        auto secz = 1. / cz;
        // airmass
        auto A = secz * (1. - 0.0012 * (pow(secz, 2) - 1.));

        // tau at 225 GHz at 80deg from atm model and airmass
        Eigen::VectorXd tau_225_calc(tx_225_zenith.size());

        // calc tau at 225 GHz at 80deg from each model
        int i = 0;
        for (const auto &[key,val]: tx_225_zenith) {
            tau_225_calc(i) = -log(val)/A;
            i++;
        }

        // difference between telescope tau and calculated tau at 225 GHz
        double tau_225_diff = abs(tau_225_zenith - tau_225_calc(0));
        // set initial model to am_q25
        extinction_model = "am_q25";

        // find model with closest tau to telescope tau and use that model
        // for extinction correction
        i = 0;
        for (const auto &[key,val]: tx_225_zenith) {
            if (abs(tau_225_zenith - tau_225_calc(i)) < tau_225_diff) {
                extinction_model = key;
            }
            i++;
        }

        // allocate transmission coefficients (order 4 polynomial)
        tx_ratio_coeff["a1100"].resize(5);
        tx_ratio_coeff["a1400"].resize(5);
        tx_ratio_coeff["a2000"].resize(5);

        // am_q25
        if (extinction_model=="am_q25") {
            tx_ratio_coeff["a1100"] << -0.07645234, 0.32631179, -0.53526165,  0.41427551,  0.8445016;
            tx_ratio_coeff["a1400"] << 0.01602193, -0.06815535,  0.11116415, -0.08525746,  1.03135867;
            tx_ratio_coeff["a2000"] << 0.09942805, -0.42193151,  0.68537969, -0.52220652,  1.18925417;
        }

        // am_q50
        else if (extinction_model=="am_q50") {
            tx_ratio_coeff["a1100"] << -0.12248821,  0.52388237, -0.86235963,  0.67117711,  0.74494326;
            tx_ratio_coeff["a1400"] << 0.02615391, -0.11122017,  0.18130761, -0.13893473,  1.05100321;
            tx_ratio_coeff["a2000"] << 0.20312343, -0.85983102,  1.39084759, -1.05255297,  1.3756457;
        }

        // am_q75
        else if (extinction_model=="am_q75") {
            tx_ratio_coeff["a1100"] << -0.19311728,  0.82931153, -1.37452105,  1.08148198,  0.57913582;
            tx_ratio_coeff["a1400"] << 0.04588952, -0.19502626,  0.31759789, -0.24297199,  1.08886782;
            tx_ratio_coeff["a2000"] << 0.43837991, -1.84767188,  2.96708103, -2.21900446,  1.77085296;
        }

        // am_q95
        else if (extinction_model=="am_q95") {
            tx_ratio_coeff["a1100"] << -0.6973269, 2.7011668, -3.93198542, 2.65377595, 0.129337;
            tx_ratio_coeff["a1400"] << 0.2933151, -1.10126494, 1.52262009, -0.9424269, 1.25622249;
            tx_ratio_coeff["a2000"] << 4.87342355, -17.94456945,  24.02708919, -14.06116984, 4.35136574;
        }
    }

    // polynomial fit to transmission ratio as a function of elevation (radians)
    template <typename DerivedA, typename DerivedB>
    auto tau_polynomial(Eigen::DenseBase<DerivedA> &coeff, Eigen::DenseBase<DerivedB> &elev) {

        // p(x) = a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
        return coeff(0)*pow(elev.derived().array(),4) + coeff(1)*pow(elev.derived().array(),3) +
               coeff(2)*pow(elev.derived().array(),2) + coeff(3)*pow(elev.derived().array(),1) +
               coeff(4);
    }

    // calculate tau for each toltec band for given elevations
    template <typename Derived>
    auto calc_tau(Eigen::DenseBase<Derived> &, double);

    // run flux calibration on the timestreams
    template <TCDataKind tcdata_kind, typename Derived, class calib_t>
    void calibrate_tod(TCData<tcdata_kind, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &,
                       Eigen::DenseBase<Derived> &, calib_t &);

    // run extinction correction on the timestreams
    template <TCDataKind tcdata_kind, typename Derived, class calib_t, typename tau_t>
    void extinction_correction(TCData<tcdata_kind, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &,
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

    auto secz = 1. / cz;
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

template <TCDataKind tcdata_kind, typename Derived, class calib_t>
void Calibration::calibrate_tod(TCData<tcdata_kind, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &det_indices,
                                Eigen::DenseBase<Derived> &array_indices, calib_t &calib) {

    // resize fcf
    in.fcf.data.setOnes(in.scans.data.cols());

    // loop through detectors
    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
        // current detector index in apt table
        Eigen::Index det_index = det_indices(i);
        // current array index in apt table
        Eigen::Index array_index = array_indices(i);

        // flux conversion factor for non-mJy/beam units
        in.fcf.data(i) = calib.flux_conversion_factor(array_index);

        // data x flxscale x factor
        in.scans.data.col(i) = in.scans.data.col(i).array()*in.fcf.data(i)*calib.apt["flxscale"](det_index);
    }
}

template <TCDataKind tcdata_kind, typename Derived, class calib_t, typename tau_t>
void Calibration::extinction_correction(TCData<tcdata_kind, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &det_indices,
                                        Eigen::DenseBase<Derived> &array_indices, calib_t &calib, tau_t &tau_freq) {

    // loop through detectors
    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
        // current detector index in apt table
        Eigen::Index det_index = det_indices(i);
        // current array index in apt table
        Eigen::Index array_index = array_indices(i);

        // factor = 1 / exp(-tau_freq)
        auto factor = 1/(-tau_freq[array_index]).array().exp();

        // apply extinction correction to fcf
        in.fcf.data(i) = (in.fcf.data(i)*factor).mean();

        // data x factor
        in.scans.data.col(i) = in.scans.data.col(i).array()*factor.array();
    }
}

} // namespace timestream

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
                tau_225_diff = abs(tau_225_zenith - tau_225_calc(i));
            }
            i++;
        }

        // allocate transmission coefficients (order 6 polynomial)
        tx_ratio_coeff["a1100"].resize(7);
        tx_ratio_coeff["a1400"].resize(7);
        tx_ratio_coeff["a2000"].resize(7);

        // am_q25
        if (extinction_model=="am_q25") {
            tx_ratio_coeff["a1100"] << -0.12008024,  0.72422015, -1.81734478,  2.45313012, -1.92159695,  0.86918801, 0.78604295;
            tx_ratio_coeff["a1400"] << 0.02619509, -0.15757661, 0.39400473, -0.52912696, 0.411213, -0.18360141, 1.04398466;
            tx_ratio_coeff["a2000"] << 0.16726241, -1.00436302,  2.50507317, -3.35219659, 2.59080373, -1.14622096, 1.26931683;
        }

        // am_q50
        else if (extinction_model=="am_q50") {
            tx_ratio_coeff["a1100"] << -0.18770822,  1.13390437, -2.85173457,  3.8617083,  -3.03996805,  1.38624303,
                                           0.65300169;
            tx_ratio_coeff["a1400"] << 0.04292884, -0.25817762,  0.64533115, -0.86622214,  0.67267823, -0.29996916,
                                           1.07167603;
            tx_ratio_coeff["a2000"] << 0.35178447, -2.10859714,  5.24620825, -6.9952531,   5.37645792, -2.35675076,
                                           1.54286813;
        }

        // am_q75
        else if (extinction_model=="am_q75") {
            tx_ratio_coeff["a1100"] << -0.28189529,  1.70842347, -4.31606883,  5.88248549, -4.67702093,  2.16747228,
                                           0.4393435;
            tx_ratio_coeff["a1400"] << 0.07581154, -0.45574885,  1.13852458, -1.52697451,  1.18425865, -0.5269455,
                                           1.12531753;
            tx_ratio_coeff["a2000"] << 0.79869908,  -4.77265095, 11.82393401, -15.67007557,  11.93052031,
                                           -5.14788907,   2.14595898;
        }

        // am_q95
        else if (extinction_model=="am_q95") {
            tx_ratio_coeff["a1100"] <<  -1.21882233,   6.67068453, -14.96466875,  17.78045563, -12.10288687,
                                           4.76050807,  -0.06765066;
            tx_ratio_coeff["a1400"] << 0.76090502, -4.05867663,  8.78487281, -9.90872343,  6.2198602,  -2.13790165,
                                           1.3668983;
            tx_ratio_coeff["a2000"] << 16.0063036,   -84.30325144, 179.28096414, -197.05751682, 118.73627425,
                                           -37.99279818, 6.55457576;
        }
    }

    // polynomial fit to transmission ratio as a function of elevation (radians)
    template <typename DerivedA, typename DerivedB>
    auto tau_polynomial(Eigen::DenseBase<DerivedA> &coeff, Eigen::DenseBase<DerivedB> &elev) {

        // p(x) = a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
        /*return coeff(0)*pow(elev.derived().array(),4) + coeff(1)*pow(elev.derived().array(),3) +
               coeff(2)*pow(elev.derived().array(),2) + coeff(3)*pow(elev.derived().array(),1) +
               coeff(4);*/

        // p(x) = a6*x^6 + a5*x^5 + a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
        return coeff(0)*pow(elev.derived().array(),6) + coeff(1)*pow(elev.derived().array(),5) +
               coeff(2)*pow(elev.derived().array(),4) + coeff(3)*pow(elev.derived().array(),3) +
               coeff(4)*pow(elev.derived().array(),2) + coeff(5)*pow(elev.derived().array(),1) +
               coeff(6);
    }

    // calculate tau for each toltec band for given elevations
    template <typename Derived>
    auto calc_tau(Eigen::DenseBase<Derived> &, double);

    // run flux calibration on the timestreams
    template <TCDataKind tcdata_kind, typename Derived, class calib_t>
    void calibrate_tod(TCData<tcdata_kind, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &,
                       Eigen::DenseBase<Derived> &, calib_t &);

    // run extinction correction on the timestreams
    template <TCDataKind tcdata_kind, typename Derived, typename tau_t>
    void extinction_correction(TCData<tcdata_kind, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &,
                       Eigen::DenseBase<Derived> &, tau_t &);
};

template <typename Derived>
auto Calibration::calc_tau(Eigen::DenseBase<Derived> &elev, double tau_225_GHz) {

    // tau at toltec freqs
    std::map<int,Eigen::VectorXd> tau_freq;

    // zenith angle
    auto cz = cos(pi/2 - elev.derived().array());
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

    // loop through detectors
    for (Eigen::Index i=0; i<in.scans.data.cols(); ++i) {
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

template <TCDataKind tcdata_kind, typename Derived, typename tau_t>
void Calibration::extinction_correction(TCData<tcdata_kind, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &det_indices,
                                        Eigen::DenseBase<Derived> &array_indices, tau_t &tau_freq) {

    // loop through detectors
    for (Eigen::Index i=0; i<in.scans.data.cols(); ++i) {
        // current detector index in apt table
        Eigen::Index det_index = det_indices(i);
        // current array index in apt table
        Eigen::Index array_index = array_indices(i);

        // factor = 1 / exp(-tau_freq)
        auto factor = 1./(-tau_freq[array_index]).array().exp();

        // apply extinction correction to fcf
        in.fcf.data(i) = (in.fcf.data(i)*factor).mean();

        // data x factor
        in.scans.data.col(i) = in.scans.data.col(i).array()*factor.array();
    }
}
} // namespace timestream

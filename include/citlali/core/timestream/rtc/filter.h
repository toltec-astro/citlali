#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/SpecialFunctions>

#include <cmath>
#include <boost/math/special_functions/bessel.hpp>

#include <tula/logging.h>

#include <citlali/core/utils/constants.h>

namespace timestream {

class Filter {
public:
    double a_gibbs, freq_low_Hz, freq_high_Hz;

    Eigen::VectorXd filter;
    Eigen::Index n_terms;

    std::vector<double> w0s, qs;
    // one biquad per notch
    std::vector<Eigen::VectorXd> notch_a_vec, notch_b_vec;

    void make_filter(double);
    void make_notch_filter(double);

    template <typename Derived>
    void convolve(Eigen::DenseBase<Derived> &);

    template <typename Derived>
    void iir(Eigen::DenseBase<Derived> &);
};

void Filter::make_filter(double fsmp) {
    auto logger = spdlog::get("citlali_logger");

    // calculate nyquist frequency
    double nyquist = fsmp / 2.;
    // scale cutoffs to Nyquist frequency (names: freq_low_Hz=f_low, freq_high_Hz=f_high)
    auto f_low = freq_low_Hz / nyquist;
    auto f_high = freq_high_Hz / nyquist;

    // enforce ordering to avoid negative DC term
    if (f_high < f_low) {
        std::swap(f_low, f_high);
        if (logger) {
            logger->warn("swapping freq_low/freq_high to enforce f_low <= f_high ({} <= {})", f_low*nyquist, f_high*nyquist);
        }
    }

    // check if upper frequency limit (lowpass)
    // is larger than lower frequency limit (highpass)
    // (names are kept as-is, see comment above)

    // determine alpha parameter based on Gibbs factor
    double alpha;

    if (a_gibbs < 21.0) {
        alpha = 0.0;
    }
    else if (a_gibbs > 50.0) {
        alpha = 0.1102 * (a_gibbs - 8.7);
    }
    else {
        alpha = 0.5842 * std::pow(a_gibbs - 21.0, 0.4) + 0.07886 * (a_gibbs - 21.0);
    }

    // argument for bessel function
    Eigen::VectorXd arg = Eigen::VectorXd::LinSpaced(n_terms, 1, n_terms);
    arg = alpha * (1.0 - (arg / n_terms).cwiseAbs2().array()).sqrt();

    // calculate the coefficients from bessel functions.
    double i_0_alpha = boost::math::cyl_bessel_i(0, alpha);

    Eigen::VectorXd coef = arg.unaryExpr([i_0_alpha](double x) {
        return boost::math::cyl_bessel_i(0, x) / i_0_alpha;
    });

    // generate time array
   Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_terms, 1, n_terms) * pi;

    // multiply coefficients by time array trig functions
    coef = coef.array()*(sin(t.array()*f_high) - sin(t.array()*f_low)) /
           t.array();

    // populate the filter vector
    filter.resize(2 * n_terms + 1);
    filter.setZero();
    filter.head(n_terms) = coef.reverse();
    filter(n_terms) = f_high - f_low;
    filter.tail(n_terms) = coef;

    // normalize overall gain to avoid amplitude scaling
    double filter_sum = filter.sum();
    if (filter_sum != 0.) {
        filter /= filter_sum;
    } else if (logger) {
        logger->warn("filter sum is zero; skipping normalization");
    }
}

void Filter::make_notch_filter(double fsmp) {
    auto logger = spdlog::get("citlali_logger");
    notch_a_vec.clear();
    notch_b_vec.clear();

    for (Eigen::Index i=0; i<w0s.size(); i++) {
        double w0 = w0s[i];
        double Q = qs[i];
        if (Q <= 0.) {
            if (logger) {
                logger->warn("invalid notch Q {} at index {}; skipping", Q, i);
            }
            continue;
        }
        if (w0 <= 0. || w0 >= fsmp/2.) {
            if (logger) {
                logger->warn("invalid notch center freq {} Hz (fsmp {}), skipping", w0, fsmp);
            }
            continue;
        }
        w0 = 2*w0/fsmp;

        // Get bandwidth
        double bw = w0/Q;

        // Normalize inputs
        bw = bw*pi;
        w0 = w0*pi;

        // Compute -3dB attenuation
        double gb = 1/sqrt(2);

        //if ftype == "notch":
            // Compute beta: formula 11.3.4 (p.575) from reference [1]
        double beta = (sqrt(1.0-pow(gb,2.0))/gb)*tan(bw/2.0);
        //elif ftype == "peak":
            // Compute beta: formula 11.3.19 (p.579) from reference [1]
          //  beta = (gb/np.sqrt(1.0-gb**2.0))*np.tan(bw/2.0)
        //else:
          //  raise ValueError("Unknown ftype.")

        // Compute gain: formula 11.3.6 (p.575) from reference [1]
        double gain = 1.0/(1.0+beta);

        // Compute numerator b and denominator a
        // formulas 11.3.7 (p.575) and 11.3.21 (p.579)
        // from reference [1]
        //if ftype == "notch":
        Eigen::VectorXd b(3);
        b << 1.0, -2.0*cos(w0), 1.0;
        b = gain*b;
        //b = gain*np.array([1.0, -2.0*np.cos(w0), 1.0]);
        //else:
        //double b = (1.0-gain)*np.array([1.0, 0.0, -1.0]);
        Eigen::VectorXd a(3);
        a << 1.0, -2.0*gain*cos(w0), (2.0*gain-1.0);
        //double a = np.array([1.0, -2.0*gain*np.cos(w0), (2.0*gain-1.0)])

        notch_a_vec.push_back(a);
        notch_b_vec.push_back(b);
    }
}

template <typename Derived>
void Filter::convolve(Eigen::DenseBase<Derived> &in) {
    // array to tell which dimension to do the convolution over
    Eigen::array<ptrdiff_t, 1> dims{0};

    // map the Eigen Matrices to Tensors to work with the Eigen::Tensor
    // convolution method
    Eigen::TensorMap<Eigen::Tensor<double, 2>> in_tensor(in.derived().data(),
                                                         in.rows(), in.cols());
    Eigen::TensorMap<Eigen::Tensor<double, 1>> filter_tensor(filter.data(),
                                                             filter.size());
    // convolve
    Eigen::Tensor<double, 2> out_tensor(
        in_tensor.dimension(0) - filter_tensor.dimension(0) + 1, in.cols());

    // run the tensor convolution
    out_tensor = in_tensor.convolve(filter_tensor, dims);

    // replace the scan data with the filtered data through an Eigen::Map
    // the first and last n_terms samples are not overwritten; copy boundary values to avoid mixing filtered/unfiltered edges
    in.block(n_terms, 0, out_tensor.dimension(0),
             in.cols()) =
        Eigen::Map<Eigen::MatrixXd>(out_tensor.data(), out_tensor.dimension(0),
                                    out_tensor.dimension(1));
    // pad edges with nearest filtered value to avoid raw/unfiltered samples
    for (Eigen::Index c=0; c<in.cols(); ++c) {
        double first_val = in(n_terms, c);
        double last_val = in(n_terms + out_tensor.dimension(0) - 1, c);
        in.block(0, c, n_terms, 1).setConstant(first_val);
        in.block(n_terms + out_tensor.dimension(0), c, n_terms, 1).setConstant(last_val);
    }
}

template <typename Derived>
void Filter::iir(Eigen::DenseBase<Derived> &in) {

    Derived out(in.rows(),in.cols());
    out.setZero();

    for (Eigen::Index i=0; i < in.cols(); ++i) {
        // cascade all notches sequentially
        Eigen::VectorXd stage_in = in.col(i);
        Eigen::VectorXd stage_out(in.rows());
        for (std::size_t sec = 0; sec < notch_a_vec.size(); ++sec) {
            const auto& a = notch_a_vec[sec];
            const auto& b = notch_b_vec[sec];

            double x_2 = 0.;
            double x_1 = 0.;
            double y_2 = 0.;
            double y_1 = 0.;
            for (Eigen::Index j=0; j<stage_in.size(); ++j) {
                stage_out(j) = a(0) * stage_in(j) + a(1) * x_1 + a(2) * x_2
                               + b(1) * y_1 + b(2) * y_2;
                x_2 = x_1;
                x_1 = stage_in(j);
                y_2 = y_1;
                y_1 = stage_out(j);
            }
            stage_in.swap(stage_out);
        }
        out.col(i) = stage_in;
    }

    in = std::move(out);
}

} // namespace timestream

#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/SpecialFunctions>

#include <cmath>
#include <boost/math/special_functions/bessel.hpp>

#include <citlali/core/utils/constants.h>

namespace timestream {

class Filter {
public:
    double a_gibbs, freq_low_Hz, freq_high_Hz;

    Eigen::VectorXd filter;
    Eigen::Index n_terms;

    std::vector<double> w0s, qs;
    Eigen::VectorXd notch_a, notch_b;

    void make_filter(double);
    void make_notch_filter(double);

    template <typename Derived>
    void convolve(Eigen::DenseBase<Derived> &);

    template <typename Derived>
    void iir(Eigen::DenseBase<Derived> &);
};

void Filter::make_filter(double fsmp) {
    // calculate nyquist frequency
    double nyquist = fsmp / 2.;
    // scale upper frequency cutoff to Nyquist frequency
    auto f_low = freq_low_Hz / nyquist;
    // scale lower frequency cutoff to Nyquist frequency
    auto f_high = freq_high_Hz / nyquist;

    // check if upper frequency limit (lowpass)
    // is larger than lower frequency limit (highpass)
    double f_stop = (f_high < f_low) ? 1. : 0.;

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
    filter(n_terms) = f_high - f_low - f_stop;
    filter.tail(n_terms) = coef;

    // normalize with sum
    //double filter_sum = filter.sum();
    //filter = filter.array() / filter_sum;
}

void Filter::make_notch_filter(double fsmp) {
    for (Eigen::Index i=0; i<w0s.size(); i++) {
        double w0 = w0s[i];
        double Q = qs[i];
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

        notch_a = a;//.push_back(a);
        notch_b = b;//.push_back(b);
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
    // the first and last nterms samples are not overwritten
    in.block(n_terms, 0, out_tensor.dimension(0),
             in.cols()) =
        Eigen::Map<Eigen::MatrixXd>(out_tensor.data(), out_tensor.dimension(0),
                                    out_tensor.dimension(1));
}

template <typename Derived>
void Filter::iir(Eigen::DenseBase<Derived> &in) {

    Derived out(in.rows(),in.cols());
    out.setZero();

    for (Eigen::Index i=0; i < in.cols(); ++i) {
        double x_2 = 0.;
        double x_1 = 0.;
        double y_2 = 0.;
        double y_1 = 0.;
        for (Eigen::Index j=0; j<in.rows(); ++j) {
            out(j,i) = notch_a(0) * in(j,i) + notch_a(1) * x_1 + notch_a(2) * x_2
                        + notch_b(1) * y_1 + notch_b(2) * y_2;
            x_2 = x_1;
            x_1 = in(j,i);
            y_2 = y_1;
            y_1 = out(j,i);
        }
    }

    in = std::move(out);
}

} // namespace timestream

#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cmath>
#include <boost/math/special_functions/bessel.hpp>

#include <citlali/core/utils/constants.h>

namespace timestream {

class Filter {
public:
    double a_gibbs, freq_low_Hz, freq_high_Hz;

    Eigen::VectorXd filter;
    Eigen::Index n_terms;

    void make_filter(double);

    template <typename Derived>
    void convolve(Eigen::DenseBase<Derived> &);
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

    if (a_gibbs <= 21.) {
        alpha = 0.;
    }
    if (a_gibbs >= 50) {
        alpha = 0.1102 * (a_gibbs - 8.7);
    }
    if (a_gibbs > 21. && a_gibbs < 50) {
        alpha = 0.5842 * std::pow(a_gibbs - 21., 0.4) + 0.07886 * (a_gibbs - 21.);
    }

    // argument for bessel function
    Eigen::VectorXd arg(n_terms);
    arg.setLinSpaced(n_terms, 1, n_terms);
    arg = alpha * sqrt(1. - (arg / n_terms).cwiseAbs2().array());

    // calculate the coefficients from bessel functions.  a loop appears to
    // be required here.
    Eigen::VectorXd coef(n_terms);

    for (Eigen::Index i = 0; i < n_terms; i++) {
        coef(i) = boost::math::cyl_bessel_i(0, arg(i)) /
                  boost::math::cyl_bessel_i(0, alpha);
    }

    // generate time array
    Eigen::VectorXd t(n_terms);
    t.setLinSpaced(n_terms, 1, n_terms);
    t = t*pi;

    // multiply coefficients by time array trig functions
    coef = coef.array()*(sin(t.array()*f_high) - sin(t.array()*f_low)) /
           t.array();

    // populate the filter vector
    filter.setZero(2 * n_terms + 1);
    filter.head(n_terms) = coef.reverse();
    filter(n_terms) = f_high - f_low - f_stop;
    filter.tail(n_terms) = coef;

    // normalize with sum
    double filter_sum = filter.sum();
    filter = filter.derived().array() / filter_sum;
}

template <typename Derived>
void Filter::convolve(Eigen::DenseBase<Derived> &in) {
    // array to tell which dimension to do the convolution over
    Eigen::array<ptrdiff_t, 1> dims{0};

    // map the Eigen Matrices to Tensors to work with the Eigen::Tensor
    // convolution method
    Eigen::TensorMap<Eigen::Tensor<double, 2>> in_tensor(in.derived().data(),
                                                         in.rows(), in.cols());
    Eigen::TensorMap<Eigen::Tensor<double, 1>> fvec_tensor(filter.data(),
                                                           filter.size());
    // convolve
    Eigen::Tensor<double, 2> out_tensor(
        in_tensor.dimension(0) - fvec_tensor.dimension(0) + 1, in.cols());

    // run the tensor convolution
    out_tensor = in_tensor.convolve(fvec_tensor, dims);

    // replace the scan data with the filtered data through an Eigen::Map
    // the first and last nterms samples are not overwritten
    in.block((fvec_tensor.size() - 1) / 2, 0, out_tensor.dimension(0),
             in.cols()) =
        Eigen::Map<Eigen::MatrixXd>(out_tensor.data(), out_tensor.dimension(0),
                                    out_tensor.dimension(1));
}

} // namespace timestream

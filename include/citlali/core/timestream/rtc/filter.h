#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/SpecialFunctions>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <boost/math/special_functions/bessel.hpp>

#include <citlali/core/utils/constants.h>

namespace timestream {

class Filter {
public:
    double a_gibbs, freq_low_Hz, freq_high_Hz;
    double iir_highpass_freq_Hz = 0.0;
    int iir_highpass_order = 1;
    bool iir_highpass_zero_phase = false;
    bool notch_zero_phase = true;

    Eigen::VectorXd filter;
    Eigen::Index n_terms;

    std::vector<double> w0s, qs;
    std::vector<Eigen::VectorXd> notch_a, notch_b;

    void make_filter(double);
    void make_notch_filter(double);
    static Eigen::Index notch_settle_samples_for_width(double, double, double);
    Eigen::Index notch_settle_samples(double, double) const;

    template <typename Derived>
    void convolve(Eigen::DenseBase<Derived> &);

    template <typename Derived>
    void iir(Eigen::DenseBase<Derived> &);

    template <typename Derived>
    void iir_highpass(Eigen::DenseBase<Derived> &, double);

    Eigen::Index iir_highpass_settle_samples(double) const;
};

inline void Filter::make_filter(double fsmp) {
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

inline void Filter::make_notch_filter(double fsmp) {
    notch_a.clear();
    notch_b.clear();
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

        notch_a.push_back(a);
        notch_b.push_back(b);
    }
}

inline Eigen::Index Filter::notch_settle_samples_for_width(
    double fsmp, double width_Hz, double attenuation) {
    if (fsmp <= 0.0 || width_Hz <= 0.0) {
        return 0;
    }
    if (!(attenuation > 0.0 && attenuation < 1.0)) {
        attenuation = 0.01;
    }

    const double bw = 2.0 * pi * width_Hz / fsmp;
    const double beta = std::tan(bw / 2.0);
    if (!std::isfinite(beta) || beta <= 0.0) {
        return 0;
    }
    const double gain = 1.0 / (1.0 + beta);
    const double radius2 = 2.0 * gain - 1.0;
    if (!(radius2 > 0.0)) {
        return 1;
    }
    const double radius = std::sqrt(radius2);
    if (!(radius > 0.0 && radius < 1.0)) {
        return 0;
    }
    const double n_samples = std::log(attenuation) / std::log(radius);
    if (!std::isfinite(n_samples) || n_samples <= 0.0) {
        return 0;
    }
    return static_cast<Eigen::Index>(std::ceil(n_samples));
}

inline Eigen::Index Filter::notch_settle_samples(double fsmp, double attenuation) const {
    Eigen::Index total = 0;
    const auto n = std::min(w0s.size(), qs.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (w0s[i] <= 0.0 || qs[i] <= 0.0) {
            continue;
        }
        total += notch_settle_samples_for_width(fsmp, w0s[i] / qs[i], attenuation);
    }
    return total;
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

    if (notch_a.empty() || notch_b.empty() ||
        in.rows() == 0 || in.cols() == 0) {
        return;
    }

    auto apply_once = [&](const Eigen::VectorXd &a, const Eigen::VectorXd &b, auto &in_arr, auto &out_arr) {
        out_arr.setZero();
        for (Eigen::Index i=0; i < in_arr.cols(); ++i) {
            const double x_init = in_arr(0, i);
            const double b_sum = b.sum();
            const double a_sum = a.sum();
            const double y_init =
                (std::isfinite(x_init) && std::isfinite(b_sum) &&
                 std::isfinite(a_sum) && std::abs(a_sum) > 0.0)
                    ? (b_sum / a_sum) * x_init
                    : 0.0;
            double x_2 = x_init;
            double x_1 = x_init;
            double y_2 = y_init;
            double y_1 = y_init;
            for (Eigen::Index j=0; j<in_arr.rows(); ++j) {
                // Direct-form I with a0 assumed 1.0
                out_arr(j,i) = b(0) * in_arr(j,i) + b(1) * x_1 + b(2) * x_2
                            - a(1) * y_1 - a(2) * y_2;
                x_2 = x_1;
                x_1 = in_arr(j,i);
                y_2 = y_1;
                y_1 = out_arr(j,i);
            }
        }
        in_arr = out_arr;
    };

    auto make_odd_extension = [](const auto &arr, Eigen::Index pad) {
        Eigen::MatrixXd extended(arr.rows() + 2 * pad, arr.cols());
        extended.block(pad, 0, arr.rows(), arr.cols()) = arr;
        if (pad <= 0) {
            return extended;
        }
        for (Eigen::Index j = 0; j < arr.cols(); ++j) {
            const double first = arr(0, j);
            const double last = arr(arr.rows() - 1, j);
            for (Eigen::Index i = 0; i < pad; ++i) {
                extended(pad - 1 - i, j) = 2.0 * first - arr(i + 1, j);
                extended(pad + arr.rows() + i, j) =
                    2.0 * last - arr(arr.rows() - 2 - i, j);
            }
        }
        return extended;
    };

    Eigen::MatrixXd out(in.rows(), in.cols());

    for (std::size_t k = 0; k < notch_a.size(); ++k) {
        const auto &a = notch_a[k];
        const auto &b = notch_b[k];
        if (!notch_zero_phase) {
            apply_once(a, b, in.derived(), out);
            continue;
        }

        const Eigen::Index filter_order =
            std::max<Eigen::Index>(a.size(), b.size());
        const Eigen::Index pad =
            std::min<Eigen::Index>(in.rows() - 1,
                                   std::max<Eigen::Index>(0, 3 * filter_order));
        if (pad <= 0) {
            apply_once(a, b, in.derived(), out);
            for (Eigen::Index i = 0; i < in.cols(); ++i) {
                in.derived().col(i).reverseInPlace();
            }
            apply_once(a, b, in.derived(), out);
            for (Eigen::Index i = 0; i < in.cols(); ++i) {
                in.derived().col(i).reverseInPlace();
            }
            continue;
        }

        auto padded = make_odd_extension(in.derived(), pad);
        Eigen::MatrixXd padded_out(padded.rows(), padded.cols());
        apply_once(a, b, padded, padded_out);
        for (Eigen::Index i = 0; i < padded.cols(); ++i) {
            padded.col(i).reverseInPlace();
        }
        apply_once(a, b, padded, padded_out);
        for (Eigen::Index i = 0; i < padded.cols(); ++i) {
            padded.col(i).reverseInPlace();
        }
        in.derived() = padded.block(pad, 0, in.rows(), in.cols());
    }
}

template <typename Derived>
void Filter::iir_highpass(Eigen::DenseBase<Derived> &in, double fsmp) {

    if (in.rows() == 0 || in.cols() == 0) {
        return;
    }
    if (fsmp <= 0.0 || iir_highpass_freq_Hz <= 0.0 || iir_highpass_order <= 0) {
        return;
    }

    const double dt = 1.0 / fsmp;
    const double rc = 1.0 / (2.0 * pi * iir_highpass_freq_Hz);
    const double alpha = rc / (rc + dt);

    auto apply_once = [&](auto &arr) {
        for (Eigen::Index i = 0; i < arr.cols(); ++i) {
            double x_1 = arr(0, i);
            double y_1 = 0.0;
            arr(0, i) = 0.0;
            for (Eigen::Index j = 1; j < arr.rows(); ++j) {
                const double x_0 = arr(j, i);
                const double y_0 = alpha * (y_1 + x_0 - x_1);
                arr(j, i) = y_0;
                x_1 = x_0;
                y_1 = y_0;
            }
        }
    };

    for (int k = 0; k < iir_highpass_order; ++k) {
        apply_once(in.derived());
    }

    if (iir_highpass_zero_phase) {
        for (Eigen::Index i = 0; i < in.cols(); ++i) {
            in.derived().col(i).reverseInPlace();
        }
        for (int k = 0; k < iir_highpass_order; ++k) {
            apply_once(in.derived());
        }
        for (Eigen::Index i = 0; i < in.cols(); ++i) {
            in.derived().col(i).reverseInPlace();
        }
    }
}

inline Eigen::Index Filter::iir_highpass_settle_samples(double fsmp) const {
    if (fsmp <= 0.0 || iir_highpass_freq_Hz <= 0.0 || iir_highpass_order <= 0) {
        return 0;
    }

    // Drop five RC time constants per stage to suppress IIR startup transients at scan edges.
    const double tau_sec = 1.0 / (2.0 * pi * iir_highpass_freq_Hz);
    const double settle_samples =
        5.0 * tau_sec * fsmp * static_cast<double>(std::max(1, iir_highpass_order));
    return static_cast<Eigen::Index>(std::ceil(std::max(0.0, settle_samples)));
}

} // namespace timestream

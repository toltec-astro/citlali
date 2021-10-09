#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/bessel_prime.hpp>
#include <boost/math/special_functions/round.hpp>
#include <cmath>
#include <iostream>


namespace timestream {

class Filter {
public:
  Eigen::VectorXd fvec;

  void makefilter(const double, const double, const double, const int,
                  const double);

  template <typename Derived>
  void convolveFilter(Eigen::DenseBase<Derived> &in) {
    // Array to tell which dimension to do the convolution over
    Eigen::array<ptrdiff_t, 1> dims{0};

    // Map the Eigen Matrices to Tensors to work with the Eigen::Tensor
    // convolution method
    Eigen::TensorMap<Eigen::Tensor<double, 2>> inTensor(in.derived().data(),
                                                        in.rows(), in.cols());
    Eigen::TensorMap<Eigen::Tensor<double, 1>> fvecTensor(fvec.data(),
                                                          fvec.size());

    // New Tensor of size in.size() - fVec.size() + 1 to store output of
    // convolve
    Eigen::Tensor<double, 2> outTensor(
        inTensor.dimension(0) - fvecTensor.dimension(0) + 1, in.cols());

    // Run the tensor convolution
    outTensor = inTensor.convolve(fvecTensor, dims);

    // Replace the scan data with the filtered data through an Eigen::Map
    // The first and last nterms samples are not overwritten
    in.block((fvecTensor.size() - 1) / 2, 0, outTensor.dimension(0),
             in.cols()) =
        Eigen::Map<Eigen::MatrixXd>(outTensor.data(), outTensor.dimension(0),
                                    outTensor.dimension(1));
  }
};

void Filter::makefilter(const double fLow_, const double fHigh_,
                        const double aGibbs, const int nterms,
                        const double samplerate) {

  // Calculate Nyquist frequency
  double nyquist = samplerate / 2.;
  // Scale upper frequency cutoff to Nyquist frequency
  double fHigh = fLow_ / nyquist;
  // Scale lower frequency cutoff to Nyquist frequency
  double fLow = fHigh_ / nyquist;

  // Check if upper frequency limit (lowpass)
  // is larger than lower frequency limit (highpass)
  double fStop = (fHigh < fLow) ? 1. : 0.;

  // Determine alpha parameter based on Gibbs factor
  double alpha;

  if (aGibbs <= 21.) {
    alpha = 0.;
  }
  if (aGibbs >= 50) {
    alpha = 0.1102 * (aGibbs - 8.7);
  }
  if (aGibbs > 21. && aGibbs < 50) {
    alpha = 0.5842 * pow(aGibbs - 21., 0.4) + 0.07886 * (aGibbs - 21.);
  }

  // Argument for bessel function
  Eigen::VectorXd arg(nterms);
  arg.setLinSpaced(nterms, 1, nterms);
  arg = alpha * sqrt(1. - (arg / nterms).cwiseAbs2().array());

  // Calculate the coefficients from bessel functions.  Note a loop appears to
  // be required here.
  Eigen::VectorXd coef(nterms);
  for (int i = 0; i < nterms; i++) {
    coef[i] = boost::math::cyl_bessel_i(0, arg[i]) /
              boost::math::cyl_bessel_i(0, alpha);
  }

  // Generate time array
  Eigen::VectorXd t(nterms);
  t.setLinSpaced(nterms, 1, nterms);
  t = t * pi;

  // Multiply coefficients by time array trig functions
  coef = coef.array() * (sin(t.array() * fHigh) - sin(t.array() * fLow)) /
         t.array();

  // Populate the filter vector
  fvec.resize(2 * nterms + 1);
  fvec.head(nterms) = coef.reverse();
  fvec[nterms] = fHigh - fLow - fStop;
  fvec.tail(nterms) = coef;

  // Normalize with sum
  fvec = fvec.derived().array() / fvec.sum();
}

} // namespace timestream

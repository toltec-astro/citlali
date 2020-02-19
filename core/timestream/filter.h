#pragma once

//#define _STDCPP_MATH_SPEC_FUNCS__201003L
//#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/CXX11/Tensor>

#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/math/special_functions/bessel_prime.hpp>
#include <boost/math/constants/constants.hpp>

#include <cmath>
#include <iostream>

namespace timestream {

namespace internal {

//class filter


//Filter function
template <typename DerivedA>
void df(Eigen::DenseBase<DerivedA> &dft, const double flow,
        const double fhigh, const double agibbs, const int nterms){

    //Checks if upper frequency limit (lowpass) is larger than lower frequency limit (highpass)
    double fStop = (fhigh < flow) ? 1. : 0.;

    //Determine alpha parameter based on Gibbs factor
    double alpha;

    if(agibbs <= 21.){
        alpha = 0.;
    }
    if(agibbs >= 50){
        alpha = 0.1102*(agibbs-8.7);
    }
    if(agibbs > 21. && agibbs < 50){
        alpha = 0.5842*pow(agibbs-21.,0.4) + 0.07886*(agibbs-21.);
    }

    //Argument for bessel function
    Eigen::VectorXd arg(nterms);
    arg.setLinSpaced(nterms,1,nterms);
    arg = alpha*sqrt(1.-(arg/nterms).cwiseAbs2().array());

    //Calculate the coefficients from bessel functions.  Note a loop appears to be required here.
    Eigen::VectorXd coef(nterms);
    for(int i=0;i<nterms;i++){
        coef[i] = boost::math::cyl_bessel_i(0,arg[i]) / boost::math::cyl_bessel_i(0,alpha);
    }

    //Generate time array
    Eigen::VectorXd t(nterms);
    t.setLinSpaced(nterms,1,nterms);
    t = t*boost::math::constants::pi<double>();

    //Multiply coefficients by time array trig functions
    coef = coef.array()*(sin(t.array()*fhigh)-sin(t.array()*flow))/t.array();

    //Populate the filter vector
    dft.head(nterms) = coef.reverse();
    dft[nterms] = fhigh-flow-fStop;
    dft.tail(nterms) = coef;

    //Normalize with sum
    dft = dft.derived().array()/dft.sum();
}

} //namespace internal

//Convoltuion code using Eigen FFT
template <typename DerivedA>
void filter(Eigen::DenseBase<DerivedA> &in_scans,
            const double HighpassFilterKnee, const double LowpassFilterKnee,
            const double agibbs,const int nterms, const int samplerate){

    //Calculate Nyquist frequency
    double nyquist = samplerate/2.;
    //Scale upper frequency cutoff to Nyquist frequency
    double fhigh = LowpassFilterKnee/nyquist;
    //Scale lower frequency cutoff to Nyquist frequency
    double flow = HighpassFilterKnee/nyquist;

    //Array to store Filter terms
    Eigen::VectorXd dft(2*nterms + 1);
    //Generate filter
    internal::df(dft, flow, fhigh, agibbs, nterms);

    //Array to tell which dimension to do the convolution over
    Eigen::array<ptrdiff_t, 1> dims{0};

    //Map the Eigen Matrices to Tensors to work with the Eigen Tensor convolution method
    Eigen::TensorMap<Eigen::Tensor<double, 2>> tis(in_scans.derived().data(), in_scans.rows(),in_scans.cols());
    Eigen::TensorMap<Eigen::Tensor<double, 1>> tdft(dft.data(), dft.size());

    //New Tensor to store output.  Note size is input_size - filter_size + 1
    Eigen::Tensor<double, 2> out(tis.dimension(0) - tdft.dimension(0) + 1, in_scans.cols());

    //Carry out the convolution
    out = tis.convolve(tdft,dims);

    //Map back to Eigen Matrix
    in_scans.block(nterms,0,tis.dimension(0) - tdft.dimension(0) + 1,in_scans.cols()) = Eigen::Map<Eigen::MatrixXd> (out.data(),out.dimension(0), out.dimension(1));
}
} //namespace

#pragma once
#include <Eigen/Dense>

//static double pi = boost::math::constants::pi<double>();

namespace timestream {

template <typename DerivedA, typename DerivedB>
void makeKernelTimestream(Eigen::DenseBase<DerivedA> &scans,
                          Eigen::DenseBase<DerivedB> &lat,
                          Eigen::DenseBase<DerivedB> &lon,
                          const double beamSigAz, const double beamSigEl){

  Eigen::VectorXd dist = (lat.derived().array().square() + lon.derived().array().square()).sqrt();

  double sigma = (beamSigAz+beamSigEl)/2./3600./360.*pi;

  scans = (-0.5*(dist/sigma).array().pow(2)).exp();

  //If the source is more than 3 beam sigmas away, call it 0.
    for(int i=0;i<dist.size();i++){
    if(dist(i) > 3.*sigma){
      scans(i) = 0;
    }
  }


}
} //namespace

#pragma once

#include <Eigen/Core>


class Histogram {
public:
    Eigen::VectorXd hist_vals, hist_bins;

    template<typename Derived>
    void calc_hist(Eigen::DenseBase<Derived> &, const int);
};


template<typename Derived>
void Histogram::calc_hist(Eigen::DenseBase<Derived> &in, const int nbins) {
    double min = in.minCoeff();
    double max = in.maxCoeff();

  // force the histogram to be symmetric about 0.
  double rg = (abs(min) > abs(max)) ? abs(min) : abs(max);

  hist_bins = Eigen::VectorXd::LinSpaced(nbins,-rg,rg);

  for (int i=0; i<nbins-1; i++){
      hist_vals(i) = ((in.derived().array() >= hist_bins(i)) && (in.derived().array() < hist_bins(i+1))).count();
  }
}

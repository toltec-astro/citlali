#pragma once

#include <Eigen/Core>

#include <citlali/core/utils/utils.h>


class Histogram {
public:
    double cov_cut;
    Eigen::VectorXd hist_vals, hist_bins;

    template<typename Derived>
    void calc_hist(Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, const int);
};


template<typename Derived>
void Histogram::calc_hist(Eigen::DenseBase<Derived> &in, Eigen::DenseBase<Derived> &wt, const int nbins) {

    auto weight_threshold = engine_utils::find_weight_threshold(wt, cov_cut);
    auto [cut_row_range, cut_col_range] = engine_utils::set_coverage_cut_ranges(wt, weight_threshold);

    // get rows and cols
    Eigen::Index nr = cut_row_range(1) - cut_row_range(0) + 1;
    Eigen::Index nc = cut_col_range(1) - cut_col_range(0) + 1;


    Eigen::MatrixXd im(nr,nc);
    for (Eigen::Index i=0; i<nc; i++) {
        for (Eigen::Index j=0; j<nr; j++) {
            Eigen::Index r = cut_row_range(0)+j;
            Eigen::Index c = cut_col_range(0)+i;
            im(j,i) = in(r,c);
        }
    }


    double min = in.minCoeff();
    double max = in.maxCoeff();

    // force the histogram to be symmetric about 0.
    double rg = (abs(min) > abs(max)) ? abs(min) : abs(max);

    hist_bins = Eigen::VectorXd::LinSpaced(nbins,-rg,rg);
    hist_vals.setZero(nbins);

    for (int i=0; i<nbins-1; i++) {
          hist_vals(i) = ((im.array() >= hist_bins(i)) && (im.array() < hist_bins(i+1))).count();
    }

    SPDLOG_INFO("hist_bins {}", hist_bins);
    SPDLOG_INFO("hist_vals {}", hist_vals);

}

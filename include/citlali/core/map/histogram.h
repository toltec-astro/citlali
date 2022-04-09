#pragma once

#include <Eigen/Core>

#include <citlali/core/utils/utils.h>


class Histogram {
public:
    double cov_cut;

    int nbins = 25;
    Eigen::VectorXd hist_vals, hist_bins;

    template<typename DerivedA, typename DerivedB>
    void calc_hist(Eigen::DenseBase<DerivedA> &, Eigen::DenseBase<DerivedB> &);
};


template<typename DerivedA, typename DerivedB>
void Histogram::calc_hist(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &wt) {

    auto weight_threshold = engine_utils::find_weight_threshold(wt, cov_cut);
    auto [cut_row_range, cut_col_range] = engine_utils::set_coverage_cut_ranges(wt, weight_threshold);

    // get rows and cols
    Eigen::Index nr = cut_row_range(1) - cut_row_range(0) + 1;
    Eigen::Index nc = cut_col_range(1) - cut_col_range(0) + 1;

    auto im = in.block(cut_row_range(0), cut_col_range(0), nr, nc);

    double min = im.minCoeff();
    double max = im.maxCoeff();

    // force the histogram to be symmetric about 0.
    double rg = (abs(min) > abs(max)) ? abs(min) : abs(max);

    hist_bins = Eigen::VectorXd::LinSpaced(nbins,-rg,rg);
    hist_vals.setZero(nbins);

    for (int i=0; i<nbins-1; i++) {
          hist_vals(i) = ((im.array() >= hist_bins(i)) && (im.array() < hist_bins(i+1))).count();
    }
}

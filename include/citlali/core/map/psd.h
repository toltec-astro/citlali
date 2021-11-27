#pragma once

#include <Eigen/Core>


class PSD {
public:
    template <typename Derived>
    void calc_map_psd(Eigen::DenseBase<Derived> &);
    double cov_cut;
    Eigen::VectorXd cut_x_range, cut_y_range;
};

template <typename Derived>
void PSD::calc_map_psd(Eigen::DenseBase<Derived> &in) {

    //find_weight_thresh();
    //set_coverage_cut_ranges();

    // make sure coverage cut map has an even number
    // of rows and columns
    int nx = cut_x_range(1)-cut_x_range(0)+1;
    int ny = cut_y_range(1)-cut_y_range(0)+1;
    int cxr0 = cut_x_range(0);
    int cyr0 = cut_y_range(0);
    int cxr1 = cut_x_range(1);
    int cyr1 = cut_y_range(1);
    if (nx % 2 == 1) {
      cxr1 = cut_x_range(1)-1;
      nx--;
    }
    if (ny % 2 == 1) {
      cyr1 = cut_y_range(1)-1;
      ny--;
    }

}

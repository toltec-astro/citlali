#pragma once

#include <vector>
#include <tuple>

#include <Eigen/Core>

class Array {
public:
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic,1> array_indices;
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic,1> nw_indices;
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic,1> map_indices;
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic,1> det_indices;

    template <class C>
    void calc_indices(C &calib_data) {

    }

};

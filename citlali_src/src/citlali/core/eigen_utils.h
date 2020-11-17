#pragma once

#include <Eigen/Core>

namespace eigen_utils {

// Row Major MatrixXd
using RowMatrixXd = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
// Row Major MatrixXcd
using RowMatrixXcd = Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

// Sorts an Eigen Vector and returns an std::vector of tuples of (value, index)
template<typename Derived>
std::vector<std::tuple<double, int>> sorter(Eigen::DenseBase<Derived> &vec){

    std::vector<std::tuple<double, int>> vis;
    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);

    for(int i=0; i<vec.size(); i++){
        std::tuple<double, double> vec_and_val(vec(i), indices(i));
        vis.push_back(vec_and_val);
    }

    std::sort(vis.begin(), vis.end(),
              [&](const std::tuple<double, int>& a, const std::tuple<double, int>& b) -> bool{
                  return std::get<0>(a) < std::get<0>(b);
              });

    return vis;
}


template<typename Derived, typename IndexType>
void shift(Eigen::DenseBase<Derived> &vec, IndexType n)
{
    int nx = vec.size();
    Eigen::DenseBase<Derived> vec2(nx);
    vec2.LinSpaced(nx, n % nx, (nx + n) % nx);

    for (int i = 0; i < nx; i++) {
        int ti = (i + n) % nx;
        int shifti = (ti < 0) ? nx + ti : ti;
        vec2(shifti) = vec(i);
    }
    vec = vec2;
}

template<typename Derived, typename IndexType>
void shift(Eigen::DenseBase<Derived> &mat, IndexType n1, IndexType n2)
{
    int nx = mat.nrows();
    int ny = mat.ncols();
    Eigen::DenseBase<Derived> mat2(nx, ny);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            int ti = (i + n1) % nx;
            int tj = (j + n2) % ny;
            int shifti = (ti < 0) ? nx + ti : ti;
            int shiftj = (tj < 0) ? ny + tj : tj;
            mat2(shifti, shiftj) = mat(i, j);
        }

        mat = mat2;
}

} // namespace

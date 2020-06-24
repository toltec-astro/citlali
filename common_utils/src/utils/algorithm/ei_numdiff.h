#pragma once

#include <Eigen/Core>


namespace alg {

template <typename DerivedA, typename DerivedB, typename DerivedC>
void gradient(const Eigen::DenseBase<DerivedA> &ydata_,
              const Eigen::DenseBase<DerivedB> &xdata_,
              Eigen::DenseBase<DerivedC> const &dy_dx_) {
    const auto n = ydata_.size();
    auto &ydata = ydata_.derived().array();
    auto &xdata = xdata_.derived().array();
    auto &dy_dx = const_cast<Eigen::DenseBase<DerivedC> &>(dy_dx_).derived();

    // check uniform dx
    auto dx = (xdata.tail(n - 1) - xdata.head(n - 1)).array().eval();
    auto dn = n - 1;
    // handle center
    if ((dx == dx.coeff(0)).all()) {
        // uniform
        dy_dx.segment(1, n - 2) =
            (ydata.tail(n - 2) - ydata.head(n - 2)) / (2. * dx.coeff(0));
    } else {
        // see
        // https://github.com/numpy/numpy/blob/master/numpy/lib/function_base.py
        auto dx1 = dx.head(dn - 1);
        auto dx2 = dx.tail(dn - 1);
        auto a = -(dx2) / (dx1 * (dx1 + dx2));
        auto b = (dx2 - dx1) / (dx1 * dx2);
        auto c = dx1 / (dx2 * (dx1 + dx2));
        // 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
        dy_dx.segment(1, n - 2) = a * ydata.head(n - 2) +
            b * ydata.segment(1, n - 2) + c * ydata.tail(n - 2);
    }
    // edge
    dy_dx.coeffRef(0) = (ydata.coeff(1) - ydata.coeff(0)) / dx.coeff(0);
    dy_dx.coeffRef(n - 1) =
        (ydata.coeff(n - 1) - ydata.coeff(n - 2)) / dx.coeff(dn - 1);
}

template <typename DerivedA, typename DerivedB, typename DerivedC>
auto gradient(const Eigen::DenseBase<DerivedA> &ydata,
              const Eigen::DenseBase<DerivedB> &xdata) {
    Eigen::Matrix<typename DerivedA::Scalar, DerivedA::RowsAtCompileTime,
                  DerivedA::ColsAtCompileTime>
        dy_dx(ydata.size());
    gradient(ydata.derived(), xdata.derived(), dy_dx);
    return dy_dx;
}


} // namespace alg
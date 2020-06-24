#pragma once

#include <Eigen/Dense>
#include "../logging.h"

namespace alg {

/**
 * @brief Fit polynomial to 2-D data vectors.
 * @param x The x values.
 * @param y The y values.
 * @param order The order. Default is 1 (linear fit)
 * @param det If specified, it will hold the transformation matrix
 * @return A tuple of two elements:
 *  - 0: The polynomial coefficients [x0, x1, ...] such that xi is for x^i.
 *  - 1: The residual vector. Same size as x and y.
 *
 */
template <typename DerivedA, typename DerivedB>
auto polyfit(const Eigen::DenseBase<DerivedA> &x,
                       const Eigen::DenseBase<DerivedB> &y, int order = 1,
                       Eigen::MatrixXd *det = nullptr) {
    auto s = x.size();
    Eigen::MatrixXd det_;
    if (!det) {
        det = &det_;
    }
    det->resize(s, order + 1);
    for (auto i = 0; i < order + 1; ++i) {
        det->col(i) = x.derived().array().pow(i);
    }
    // SPDLOG_TRACE("det: {}", *det);

    // xscale (1, s, s^2, ...)
    Eigen::Index max;
    x.derived().array().abs().maxCoeff(&max);
    Eigen::VectorXd xscale{order + 1};
    for (auto i = 0; i < order + 1; ++i) {
        xscale.coeffRef(i) = pow(x(max), i);
    }
    // SPDLOG_TRACE("xscale: {}", xscale);

    // scale det before the fit
    Eigen::MatrixXd det_scaled{s, order + 1};
    for (auto i = 0; i < order + 1; ++i) {
        det_scaled.col(i) = det->col(i) / xscale(i);
    }
    // SPDLOG_TRACE("det scaled: {}", det_scaled);
    // fit with scaled y
    auto yscale  = y(0);
    Eigen::VectorXd pol_scaled = det_scaled.colPivHouseholderQr().solve(
        y.derived() / yscale);
    // restore the scaling for pol
    Eigen::VectorXd pol = pol_scaled.cwiseQuotient(xscale) * yscale;
    Eigen::VectorXd res = y.derived() - (*det) * pol;
    // SPDLOG_TRACE("res: {}", res);
    return std::make_tuple(std::move(pol), std::move(res));
}

}  // namespace alg

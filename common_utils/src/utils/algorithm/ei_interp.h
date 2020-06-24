#include "../eigen.h"
#include "mlinterp/mlinterp.hpp"
#include <Eigen/Core>
#include <vector>

namespace alg {

template <typename DerivedA, typename DerivedB, typename DerivedC,
          typename DerivedD>
void interp(const Eigen::DenseBase<DerivedA> &xp_,
            const Eigen::DenseBase<DerivedB> &yp_,
            const Eigen::DenseBase<DerivedC> &x_,
            Eigen::DenseBase<DerivedD> const &y_) {
    using Eigen::Index;
    auto &xp = xp_.derived();
    auto &yp = yp_.derived();
    auto &x = x_.derived();
    auto &y = const_cast<Eigen::DenseBase<DerivedD> &>(y_).derived();
    // make sure the data are continuous
    if (!(eigen_utils::is_contiguous(xp) && eigen_utils::is_contiguous(yp) &&
          eigen_utils::is_contiguous(x) && eigen_utils::is_contiguous(y))) {
        throw std::runtime_error(
            "interp does not work with non-contiguous data");
    }
    std::vector<Index> nd{xp.size()};
    mlinterp::interp(nd.data(), x.size(), yp.data(), y.data(), xp.data(),
                     x.data());
}

template <typename A, typename B, typename C>
auto interp(A &&xp, B &&yp, C &&x) {
    typename std::decay_t<B>::PlainObject y(x.size());
    interp(FWD(xp), FWD(yp), FWD(x), y);
    return y;
}

} // namespace alg
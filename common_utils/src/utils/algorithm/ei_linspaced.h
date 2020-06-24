#pragma once
#include "../logging.h"
#include "../eigen.h"
#include "../meta.h"

namespace alg {

template <typename Derived, typename... Args>
void fill_linspaced(Eigen::DenseBase<Derived> const &m_, Args... args) {
    using Eigen::Index;
    auto &m = const_cast<Eigen::DenseBase<Derived> &>(m_).derived();
    auto set_linspaced = [&](auto&& m) {
        if constexpr ((sizeof...(args)) == 0) {
            m.setLinSpaced(m.size(), 0, m.size() - 1);
        } else {
            m.setLinSpaced(m.size(), args...);
        }
    };
    if constexpr (Derived::IsVectorAtCompileTime) {
        SPDLOG_TRACE("fill vector");
        set_linspaced(m);
    } else if constexpr (eigen_utils::is_plain_v<Derived>) {
        // check continugous
        assert(eigen_utils::is_contiguous(m));
        SPDLOG_TRACE("fill matrix");
        // make a map and assign to it
        set_linspaced(typename eigen_utils::type_traits<Derived>::VecMap(m.data(), m.size()));
    } else {
        SPDLOG_TRACE("fill matrix expr");
        typename Derived::PlainObject tmp{m.rows(), m.cols()};
        fill_linspaced(tmp, args...);
        m = std::move(tmp);
    }
}

template <typename Scalar> auto arange(Scalar start, Scalar stop, Scalar step) {
    using Eigen::Index;
    Index size =
        Index((stop - std::numeric_limits<Scalar>::epsilon() - start) / step) +
        1;
    return Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::LinSpaced(
        size, start, start + (size - 1) * step);
}

}  // namespace alg

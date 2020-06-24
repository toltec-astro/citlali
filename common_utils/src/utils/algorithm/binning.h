#pragma once

#include <Eigen/Core>

namespace alg {

template <bool conserve_sum = true, typename Derived>
auto upsample(const Eigen::DenseBase<Derived> &vector, Eigen::Index n) {
    static_assert(std::is_floating_point_v<typename Derived::Scalar>,
                  "EXPECT FLOATING POINT");
    static_assert(Derived::IsVectorAtCompileTime, "EXPECT VECTOR");
    auto output_size = vector.size() * n;
    typename Derived::PlainObject output(output_size);
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
        if constexpr (conserve_sum) {
            output.segment(i, n).array() = vector.coeff(i) / n;
        } else {
            output.segment(i, n).array() = vector.coeff(i);
        }
    }
}

} // namespace alg
#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include "../container.h"

namespace alg {

enum class BorderMode { Mirror, Nearest };

template <BorderMode mode = BorderMode::Nearest, typename Derived>
auto borderpad1d(const Eigen::DenseBase<Derived> &vector, Eigen::Index size) {
    auto n = vector.size();
    auto n0 = (size - 1) / 2;
    assert(n > size);
    assert(size > 1);
    // make copy
    typename Derived::PlainObject output(n + size - 1);
    output.segment(n0, n) = vector;
    if (n0 > 0) {
        if constexpr (mode == BorderMode::Mirror) {
            // --- n0--- -- n ---------  size - 1 - n0
            // c   b   | a   bc  d     | c      b          |
            // 0 n0-1   n0     n0+n-1   n0 + n  n+size-2
            output.head(n0).reverse() = output.segment(n0 + 1, n0);
            output.tail(size - 1 - n0).reverse() =
                output.segment(n0 + n - 1 - (size - 1 - n0), size - 1 - n0);
        }
        if constexpr (mode == BorderMode::Nearest) {
            output.head(n0) = output.coeff(n0);
            output.tail(size - 1 - n0) = output.coeff(n0 + n - 1);
        }
    }
    return output;
}

template <typename DerivedA, typename DerivedB>
auto convolve1d(const Eigen::DenseBase<DerivedA> &vector,
                const Eigen::DenseBase<DerivedB> &kernel) {
    using ScalarA = typename DerivedA::Scalar;
    using ScalarB = typename DerivedB::Scalar;
    const auto data = Eigen::TensorMap<Eigen::Tensor<ScalarA, 1>>(
        const_cast<ScalarA *>(vector.derived().data()), vector.size());
    const auto ker = Eigen::TensorMap<Eigen::Tensor<ScalarB, 1>>(
        const_cast<ScalarB *>(kernel.derived().data()), kernel.size());
    Eigen::array<ptrdiff_t, 1> dims({0});
    typename DerivedA::PlainObject output(vector.size() - ker.size() + 1);
    Eigen::TensorMap<Eigen::Tensor<ScalarA, 1>>(output.data(), output.size()) =
        data.convolve(ker, dims);
    return output;
}

template <typename DerivedA, typename F, typename DerivedB>
void convolve1d(const Eigen::DenseBase<DerivedA> &vector_, F &&func,
                Eigen::Index size, Eigen::DenseBase<DerivedB> const &output_) {
    using Scalar = typename DerivedA::Scalar;
    const auto &vector = vector_.derived();
    auto &output = const_cast<Eigen::DenseBase<DerivedB> &>(output_).derived();
    auto data = Eigen::TensorMap<Eigen::Tensor<const Scalar, 1>>(
        vector.data(), {vector.size()});
    Eigen::Tensor<Scalar, 2> patches_ =
        data.extract_patches(Eigen::array<std::ptrdiff_t, 1>{size});
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        patches(patches_.data(), patches_.dimension(0), patches_.dimension(1));
    if (output.size() == 0) {
        output.resize(patches.cols());
    }
    if (patches.cols() != output.size()) {
        throw std::runtime_error("output data has incorrect dimension");
    }
    auto icols = container_utils::index(patches.cols());
    std::for_each(icols.begin(), icols.end(), [&](auto i) {
        if constexpr (std::is_invocable_v<F, decltype(patches)>) {
            output.coeffRef(i) = FWD(func)(patches.col(i));
        } else if constexpr (std::is_invocable_v<F, decltype(patches),
                                                 Eigen::Index>) {
            output.coeffRef(i) = FWD(func)(patches.col(i), i);
        }
    });
}

template <typename A, typename F>
auto convolve1d(A &&vector, F &&func, Eigen::Index size) {
    typename std::decay_t<A>::PlainObject output;
    convolve1d(FWD(vector), FWD(func), size, output);
    return output;
}

}  // namespace alg
#pragma once

#include "../eigen.h"
#include <type_traits>

namespace alg {

auto shape = [](const auto &m) { return std::make_pair(m.rows(), m.cols()); };

auto span = [](const auto &v) { return v.coeff(v.size() - 1) - v.coeff(0); };

auto step = [](const auto &v, Eigen::Index i = 0) {
    return v.coeff(i + 1) - v.coeff(i);
};

template <typename T1, typename T2>
auto windowindex_fixedsize(T1 i, T1 size, T2 window) {
    auto il = i - (window - 1) / 2;
    auto ir1 = il + window;
    if (il < 0) {
        il = 0;
        ir1 = window;
    }
    if (ir1 > size) {
        ir1 = size;
        il = ir1 - window;
        assert(il >= 0);
    }
    return std::make_pair(il, ir1);
}

/**
 * @brief Return argmin.
 */
template <typename Derived, typename = std::enable_if_t<
                                std::is_arithmetic_v<typename Derived::Scalar>>>
auto argmin(const Eigen::DenseBase<Derived> &m) {
    static_assert(Derived::IsVectorAtCompileTime, "REQUIRES VECTOR TYPE");
    Eigen::Index imin = 0;
    m.minCoeff(&imin);
    return imin;
}

/**
 * @brief Return argmax.
 */
template <
    typename Derived,
    typename = std::enable_if_t<std::is_arithmetic_v<typename Derived::Scalar>>>
auto argmax(const Eigen::DenseBase<Derived> &m) {
    static_assert(Derived::IsVectorAtCompileTime, "REQUIRES VECTOR TYPE");
    Eigen::Index imax = 0;
    m.maxCoeff(&imax);
    return imax;
}

/**
 * @brief Return arg nearest.
 */
template <typename Derived, typename = std::enable_if_t<
                                std::is_arithmetic_v<typename Derived::Scalar>>>
auto argeq(const Eigen::DenseBase<Derived> &m, typename Derived::Scalar v) {
    static_assert(Derived::IsVectorAtCompileTime, "REQUIRES VECTOR TYPE");
    auto imin = argmin((m.derived().array() - v).abs());
    return std::make_pair(imin, m.coeff(imin) - v);
}

/**
 * @brief Return slice that has value range winin given range.
 */
template <
    typename Derived,
    typename = std::enable_if_t<std::is_arithmetic_v<typename Derived::Scalar>>>
auto argwithin(const Eigen::DenseBase<Derived> &m,
               typename Derived::Scalar vmin, typename Derived::Scalar vmax) {
    static_assert(Derived::IsVectorAtCompileTime, "REQUIRES VECTOR TYPE");
    auto [il, epsl] = argeq(m.derived(), vmin);
    if (epsl < 0) {
        ++il;
    }
    auto [ir, epsr] = argeq(m.derived(), vmax);
    if (epsr > 0) {
        --ir;
    }
    return std::make_pair(il, ir + 1);
}

/**
 * @brief Return mean.
 * This promote the type to double for shorter types.
 */
template <typename Derived, typename = std::enable_if_t<
                                std::is_arithmetic_v<typename Derived::Scalar>>>
auto mean(const Eigen::DenseBase<Derived> &m) {
    // promote size to Double for better precision
    auto size = static_cast<double>(m.size());
    return m.sum() / size;
}

/**
 * @brief Return mean and stddev.
 * @param m The vector for which the mean and stddev are calculated.
 * @param ddof Delta degree of freedom. See numpy.std.
 * @note If \p m is an expression, it is evaluated twice.
 */
template <typename Derived, typename = std::enable_if_t<
                                std::is_arithmetic_v<typename Derived::Scalar>>>
auto meanstd(const Eigen::DenseBase<Derived> &m, int ddof = 0) {
    auto mean_ = mean(m.derived());
    auto size = static_cast<double>(m.size());
    auto std =
        std::sqrt((m.derived().array().template cast<decltype(mean_)>() - mean_)
                      .square()
                      .sum() /
                  (size - ddof));
    return std::make_pair(mean_, std);
}

/**
 * @brief Return median
 * @note Promotes to double.
 */
template <typename Derived, typename = std::enable_if_t<
                                std::is_arithmetic_v<typename Derived::Scalar>>>
auto median(const Eigen::DenseBase<Derived> &m) {
    // copy to a std vector for sort
    auto v = eigen_utils::tostd(m);
    auto n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    if (v.size() % 2) {
        return v[n] * 1.0; // promote to double
    }
    // even sized vector -> average the two middle values
    auto max_it = std::max_element(v.begin(), v.begin() + n);
    return (*max_it + v[n]) / 2.0;
}

/**
 * @brief Return median absolute deviation
 */
template <typename Derived> auto medmad(const Eigen::DenseBase<Derived> &m) {
    auto med = median(m);
    return std::make_pair(
        med,
        median(
            (m.derived().array().template cast<decltype(med)>() - med).abs()));
}

/**
 * @brief Return median with nan excluded
 * @note Promotes to double.
 */
template <
    typename Derived,
    typename = std::enable_if_t<std::is_arithmetic_v<typename Derived::Scalar>>>
auto nanmedian(const Eigen::DenseBase<Derived> &m) {
    // copy to a std vector for sort
    using Scalar = typename Derived::Scalar;
    std::vector<Scalar> v;
    v.reserve(m.size());
    for (Eigen::Index i = 0; i < m.size(); ++i) {
        if (!std::isnan(m.coeff(i))) {
            v.push_back(m.coeff(i));
        }
    }
    if (v.size() == 0) {
        return std::numeric_limits<Scalar>::quiet_NaN();
    }
    auto n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    if (v.size() % 2) {
        return v[n] * 1.0; // promote to double
    }
    // even sized vector -> average the two middle values
    auto max_it = std::max_element(v.begin(), v.begin() + n);
    const double medf = 2.0;
    return (*max_it + v[n]) / medf;
}

/**
 * @brief Return median absolute deviation
 */
template <typename Derived> auto nanmedmad(const Eigen::DenseBase<Derived> &m) {
    auto med = nanmedian(m);
    return std::make_pair(
        med,
        nanmedian(
            (m.derived().array().template cast<decltype(med)>() - med).abs()));
}

} // namespace alg

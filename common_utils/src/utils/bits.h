#pragma once

#include "meta.h"
#include <tuple>
#include <type_traits>

namespace bits_utils {

/// @brief Same as the linux routine fls
/// fls(0) = 0; fls(1) = 1; fls(111) = 3
template <typename T, REQUIRES_(std::is_integral<T>)>
constexpr auto fls(T v) {
    std::size_t n = 0;
    while (v > 0) {
        ++n;
        v >>= 1;
    }
    return n;
}

/// @brief Return the count of set bits
template <typename T, REQUIRES_(std::is_integral<T>)>
constexpr auto count(T v) {
    std::size_t n = 0;
    while (v > 0) {
        if ((v & 1) > 0) {
            ++n;
        }
        v >>= 1;
    }
    return n;
}

/// @brief Decompose the set bits to an integral sequence
template <auto v, REQUIRES_(std::is_integral<decltype(v)>)>
constexpr auto decompose() {
    using T = decltype(v);
    if constexpr (v == 0) {
        return std::integer_sequence<T>{};
    } else {
        constexpr auto n = fls(v);
        return meta::apply_const_sequence(
            [](auto... i) {
                return std::apply(
                    [](auto j0, auto... j) {
                        return std::integer_sequence<
                            typename decltype(j0)::value_type, j0,
                            decltype(j)::value...>{};
                    },
                    std::tuple_cat(std::conditional_t<
                                   (((1 << decltype(i)::value) & v) > 0),
                                   std::tuple<std::integral_constant<
                                       T, (1 << decltype(i)::value)>>,
                                   std::tuple<>>{}...));
            },
            std::make_index_sequence<n>{});
    }
};

} // namespace bits_utils
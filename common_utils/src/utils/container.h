#pragma once
#include "eigen.h"
#include "eigeniter.h"
#include "logging.h"
#include "meta.h"
#include <regex>

namespace container_utils {

/// @brief Populate data using data in existing container.
/// Optioanally, when \p func is provided, the elements are mapped using
/// \p func.
template <typename In, typename Out, typename... F,
          typename = std::enable_if_t<meta::is_iterable<In>::value>,
          typename = std::enable_if_t<
              meta::is_sized<Out>::value ||
              meta::is_iterator<Out>::value
              >
          >
void populate(In &&in, Out& out, F &&... func) {
    if constexpr (
                meta::is_sized<Out>::value
                ) {
        if constexpr (
                meta::has_resize<Out, decltype(in.size())>::value) {
            // resize if available
            out.resize(in.size());
        }
        // check size match
        assert(out.size() == in.size());
    }
    // figure out output for transform
    auto out_iter = [&]() {
        if constexpr (eigen_utils::is_eigen_v<Out> && eigen_utils::type_traits<Out>::value) {
            // eigen type
            return eigeniter::EigenIter(out, 0);
        } else if constexpr (meta::is_iterator<Out>::value) {
            return out;
        } else {
            static_assert(meta::always_false<Out>::value,
                          "NOT ABLE TO GET OUTPUT ITERATOR");
        }
    }();
    // to handle custom transform function
    auto run_with_iter = [&](const auto &begin, const auto &end) {
        // no func provided, run a copy with implicit conversion
        if constexpr ((sizeof...(F)) == 0) {
            // SPDLOG_TRACE("use implicit conversion");
            std::copy(begin, end, out_iter);
            // run transform with func
        } else {
            std::transform(begin, end, out_iter, FWD(func)...);
        };
    };
    // handle l- and r- values differently for input
    if constexpr (std::is_lvalue_reference_v<In>) {
        // run with normal iter
        SPDLOG_TRACE("use copy iterator");
        run_with_iter(in.begin(), in.end());
    } else {
        // run with move iter
        SPDLOG_TRACE("use move iterator");
        run_with_iter(std::make_move_iterator(in.begin()),
                      std::make_move_iterator(in.end()));
    }
}

/// @brief Create data from data in existing container. Makes use
/// of \ref populate.
template <typename Out, typename In, typename... F,
          typename = std::enable_if_t<meta::is_iterable<In>::value>,
          typename = std::enable_if_t<std::is_default_constructible_v<Out>>
          >
Out create(In &&in, F &&... func) {
    Out out;
    if constexpr (meta::is_instance<Out, std::vector>::value) {
        // reserve for vector
        // SPDLOG_TRACE("reserve vector{}", in.size());
        out.reserve(in.size());
    }
    if constexpr (eigen_utils::is_plain_v<Out>) {
        SPDLOG_TRACE("create eigen");
        populate(FWD(in), out, FWD(func)...);
        return out;
    } else if constexpr (meta::is_iterable<Out>::value) {
        SPDLOG_TRACE("create stl");
        // populate with iterator
        auto out_iter = [&]() {
            if constexpr (meta::has_push_back<Out>::value) {
                SPDLOG_TRACE("use back_inserter");
                return std::back_inserter(out);
            } else if constexpr (meta::has_insert<Out>::value) {
                SPDLOG_TRACE("use inserter");
                return std::inserter(out, out.end());
            } else {
                static_assert(meta::always_false<Out>::value,
                              "NOT ABLE TO GET OUTPUT ITERATOR");
            }
        }();
        populate(FWD(in), out_iter, FWD(func)...);
        return out;
    } else {
        static_assert(meta::always_false<Out>::value,
                      "NOT KNOW HOW TO POPULATE OUT TYPE"
                      );
    }
}

/// @brief Returns true if \p v ends with \p ending.
template <typename T, typename U> bool startswith(const T &v, const U &prefix) {
    if (prefix.size() > v.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), v.begin());
}

/// @brief Returns true if \p v ends with \p ending.
template<typename T, typename U>
bool endswith(const T &v, const U &ending) {
    if (ending.size() > v.size()) {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), v.rbegin());
}

/// @brief Rearrange nested vector to a flat one *in place*.
template <typename T> void ravel(std::vector<std::vector<T>> &v) {
    std::size_t total_size = 0;
    for (const auto &sub : v)
        total_size += sub.size(); // I wish there was a transform_accumulate
    std::vector<T> result;
    result.reserve(total_size);
    for (const auto &sub : v)
        result.insert(result.end(), std::make_move_iterator(sub.begin()),
                      std::make_move_iterator(sub.end()));
    v = std::move(result);
}

/// @brief Returns the index of element in vector.
template<typename T>
std::optional<typename std::vector<T>::difference_type> indexof(const std::vector<T>& vec, const T& v) {
    auto it = std::find(vec.begin(), vec.end(), v);
    if (it == vec.end()) {
        return std::nullopt;
    }
    return std::distance(vec.begin(), it);
}

/// @brief Create [index, element] like enumerate for container.
template<typename T>
std::unordered_map<std::size_t, typename T::value_type> unordered_enumerate(const T& v) {
    std::unordered_map<std::size_t, typename T::value_type> ret;
    std::size_t i = 0;
    for (auto it = v.begin(); it != v.cend(); ++it) {
        ret.insert({i, *it});
        ++i;
    }
    return ret;
}

/// @brief Create [index, element] like enumerate for container.
template<typename T>
std::vector<std::pair<std::size_t, typename T::value_type>> enumerate(const T& v) {
    std::vector<std::pair<std::size_t, typename T::value_type>> ret;
    for (std::size_t i = 0; i < v.size(); ++i) {
        ret.emplace_back(std::piecewise_construct,
                         std::forward_as_tuple(i),
                         std::forward_as_tuple(v[i]));
    }
    return ret;
}

/// @brief Create index sequence for container.
template<typename Index, typename=std::enable_if_t<std::is_integral_v<Index>>>
auto index(Index size) {
    std::vector<Index> ret(size);
    eigen_utils::asvec(ret).setLinSpaced(size, 0, size -1);
    return ret;
}

/// @brief Create index sequence for container.
template<typename T, typename=std::enable_if_t<meta::is_sized<T>::value>>
auto index(const T& v) {
    return index(v.size());
}

template <typename T>
using Slice = std::tuple<std::optional<T>, std::optional<T>, std::optional<T>>;
using IndexSlice = Slice<Eigen::Index>;

/// @brief Parse python-like slice string.
template <typename T = Eigen::Index>
auto parse_slice(const std::string &slice_str) {
    Slice<T> result;
    auto &[start, stop, step] = result;
    std::size_t istart{0};
    std::size_t istop{0};
    std::size_t istep{0};
    std::string value_pattern;
    if constexpr (std::is_integral_v<T>) {
        value_pattern = "[-+]?[0-9]+";
        istart = 1;
        istop = 3;
        istep = 4;
    } else {
        value_pattern = "[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?";
        istart = 1;
        istop = 4;
        istep = 6;
    }
    std::string pattern = fmt::format("^({})?(:)?({})?:?({})?$", value_pattern,
                                      value_pattern, value_pattern);
    std::regex re_slice{pattern};
    SPDLOG_TRACE("checking slice str {} with {}", slice_str, pattern);
    std::smatch match;
    T var;
    if (std::regex_match(slice_str, match, re_slice)) {
        if (match[istart].matched) {
            std::istringstream iss;
            iss.str(match[istart].str());
            iss >> var;
            start = var;
        }
        if (match[istop].matched) {
            std::istringstream iss;
            iss.str(match[istop].str());
            iss >> var;
            stop = var;
        }
        if (match[istep].matched) {
            std::istringstream iss;
            iss.str(match[istep].str());
            iss >> var;
            step = var;
        }
    }
    SPDLOG_TRACE("parsed slice {}", result);
    return result;
}

//                                              start stop step len
template <typename T> using BoundedSlice = std::tuple<T, T, T, T>;

/// @brief Convert slice to indices
template <typename T, typename N,
          REQUIRES_V(std::is_integral_v<T> &&std::is_integral_v<N>)>
auto to_indices(Slice<T> slice, N n) {
    BoundedSlice<T> result{
        std::get<0>(slice).value_or(0),
        std::get<1>(slice).value_or(n),
        std::get<2>(slice).value_or(1),
        0,
    };
    auto &[start, stop, step, size] = result;
    if (start < 0) {
        start += n;
    }
    if (stop < 0) {
        start += n;
    }
    if (stop >= n) {
        stop = n;
    }
    size = (stop - start) / step + (((stop - start) % step) ? 1 : 0);
    return result;
}

// std::vector<T>&& src - src MUST be an rvalue reference
// std::vector<T> src - src MUST NOT, but MAY be an rvalue reference
template <typename T>
inline void append(std::vector<T> source, std::vector<T> &destination) {
    if (destination.empty())
        destination = std::move(source);
    else
        destination.insert(std::end(destination),
                           std::make_move_iterator(std::begin(source)),
                           std::make_move_iterator(std::end(source)));
}

} // namespace container_utils

namespace fmt {

template <typename T, typename Char>
struct formatter<container_utils::Slice<T>, Char>
    : fmt_utils::nullspec_formatter_base {

    template <typename FormatContext>
    auto format(const container_utils::Slice<T> &slice, FormatContext &ctx) {
        auto it = ctx.out();
        const auto &[start, stop, step] = slice;
        return format_to(it, "[{}:{}:{}]", start, stop, step);
    }
};

template <typename T, typename Char>
struct formatter<container_utils::BoundedSlice<T>, Char>
    : fmt_utils::nullspec_formatter_base {

    template <typename FormatContext>
    auto format(const container_utils::BoundedSlice<T> &slice,
                FormatContext &ctx) {
        auto it = ctx.out();
        const auto &[start, stop, step, size] = slice;
        return format_to(it, "[{}:{}:{}]({})", start, stop, step, size);
    }
};

} // namespace fmt

#pragma once
#include "../eigen.h"
#include "../meta.h"
#include <array>
#include <fmt/format.h>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
#if __has_include(<charconv>)
#include <charconv>
#else
namespace std {
template <typename Scalar>
void from_chars(char *begin, char *end, Scalar &dest) {
    std::stringstream ss;
    for (auto it = begin; it != end; ++it) {
        ss << *it;
    }
    ss >> dest;
}
} // namespace std
#endif
#else
#include <charconv>
#endif

namespace fmt_utils {

template <typename T> struct scalar_traits {
    using type = typename std::decay_t<T>;
    constexpr static bool value = std::is_arithmetic_v<type>;
};

/// @brief Pretty print Eigen types.
/// @param s output stream.
/// @param m_ input data.
/// @param fmt format spec.
/// @param max_rows maximum number of rows to be printed for 2-D data.
/// @param max_cols maximum number of cols to be printed for 2-D data.
/// @param max_size maximum number of items to be printed in case of 1-D data.
template <typename OStream, typename Derived>
OStream &pprint_matrix(OStream &s, const Eigen::DenseBase<Derived> &m_,
                       const Eigen::IOFormat &fmt, int max_rows, int max_cols,
                       int max_size) {
    auto &&m = m_.derived();
    if (m.size() == 0) {
        s << fmt.matPrefix << fmt.matSuffix;
        return s;
    }
    // minimum size to print;
    if (max_rows < 3)
        max_rows = 3;
    if (max_cols < 3)
        max_cols = 3;
    if (max_size < 3)
        max_size = 3;
    // vector case
    if (m.cols() == 1 || m.rows() == 1) {
        max_rows = max_size;
        max_cols = max_size;
    }

    std::streamsize explicit_precision;
    if (fmt.precision == Eigen::StreamPrecision) {
        explicit_precision = 0;
    } else {
        explicit_precision = fmt.precision;
    }
    std::streamsize old_precision = 0;
    if (explicit_precision)
        old_precision = s.precision(explicit_precision);

    // iterate-and-apply F to m items from head and tail of container with n
    // items in total
    auto partial_apply = [](auto n, auto m, auto &&F) {
        if (n <= m) {
            for (decltype(n) i = 0; i < n; ++i)
                FWD(F)(i);
        } else {
            for (decltype(n) i = 0; i < m / 2; ++i) {
                FWD(F)(i);
            }
            for (decltype(n) i = n - m / 2 - 1; i < n; ++i) {
                FWD(F)(i);
            }
        }
    };
    bool align_cols = !(fmt.flags & Eigen::DontAlignCols);
    Eigen::Index width = 0;
    if (align_cols) {
        partial_apply(m.rows(), max_rows - 1, [&](auto i) {
            partial_apply(m.cols(), max_cols - 1, [&](auto j) {
                std::stringstream sstr;
                sstr.copyfmt(s);
                sstr << m.coeff(i, j);
                width = std::max(width, decltype(i)(sstr.str().length()));
            });
        });
    }
    std::string ellipsis = "...";
    std::string fmt_ellipsis = "...";
    if (width > 0) {
        auto width_ = SIZET(width);
        if (width <= 3) {
            fmt_ellipsis = std::string(width_, '.');
        } else {
            fmt_ellipsis = std::string(width_, ' ');
            fmt_ellipsis.replace((width_ - ellipsis.size() + 1) / 2,
                                 ellipsis.size(), ellipsis);
        }
    }
    s << fmt.matPrefix;
    Eigen::Index oi = -1; // use this to detect jump of coeffs, at which point
                          // "..." is inserted
    partial_apply(m.rows(), max_rows, [&](auto i) {
        decltype(i) oj = -1;
        partial_apply(m.cols(), max_cols, [&](auto j) {
            if (j == 0) {
                // if (i > 0) s << fmt.rowSpacer;
                s << fmt.rowPrefix;
            } else {
                s << fmt.coeffSeparator;
            }
            if (j > oj + 1) {
                s << ellipsis;
            } else {
                if (width > 0) {
                    s.width(width);
                }
                if (i > oi + 1) {
                    s << fmt_ellipsis;
                } else {
                    s << m.coeff(i, j);
                }
            }
            if (j == m.cols() - 1) {
                s << fmt.rowSuffix;
                if (i < m.rows() - 1) {
                    s << fmt.rowSeparator;
                }
            }
            oj = j;
        });
        oi = i;
    });
    s << fmt.matSuffix;
    if (explicit_precision) {
        s.precision(old_precision);
    }
    return s;
}

/// @brief The default pprint format.
struct pformat {
    using Index = Eigen::Index;
    static auto format(Index rows, Index cols) {
        using Eigen::DontAlignCols;
        using Eigen::IOFormat;
        using Eigen::StreamPrecision;
        // as flat vector
        if (cols == 1 || rows == 1) {
            return IOFormat(StreamPrecision, DontAlignCols, ", ", ", ", "", "",
                            "[", "]");
        }
        // as flat vector of vector
        if (cols < 3) {
            return IOFormat(StreamPrecision, DontAlignCols, ", ", " ", "[", "]",
                            "[", "]");
        }
        // as matrix
        return IOFormat(StreamPrecision, 0, ", ", "\n ", "[", "]", "[\n ", "]");
    }
};

template <typename T, typename Format = pformat> struct pprint {
    using Ref =
        typename Eigen::internal::ref_selector<typename std::conditional<
            fmt_utils::scalar_traits<T>::value,
            // T is deduced to be scalar type, use Eigen Map
            Eigen::Map<
                const Eigen::Matrix<typename fmt_utils::scalar_traits<T>::type,
                                    Eigen::Dynamic, Eigen::Dynamic>>,
            // T is deduced to be Eigen type, use T
            T>::type>::type;
    using Index = Eigen::Index;
    /**
     * @brief Pretty print data held by Eigen types.
     */
    template <typename U = T,
              typename = std::enable_if_t<!fmt_utils::scalar_traits<U>::value>>
    pprint(const Eigen::DenseBase<T> &m)
        : matrix(m.derived()), format(Format::format(m.rows(), m.cols())) {}
    /**
     * @name Pretty print raw data buffer as Eigen::Map.
     * @{
     */
    /// The size of matrix data \p m is given by \p nrows and \p ncols.
    pprint(const T *const data, Index nrows, Index ncols)
        : matrix(Ref(data, nrows, ncols)),
          format(Format::format(nrows, ncols)) {}
    /// The size of vector data \p v is given by \p size.
    pprint(const T *const data, Eigen::Index size) : pprint(data, size, 1) {}
    /** @}*/
    /**
     * @brief Pretty print data held by std vector
     */
    template <typename U = T, typename Allocator=std::allocator<T>,
              typename = std::enable_if_t<fmt_utils::scalar_traits<U>::value>>
    pprint(const std::vector<T, Allocator> &vec) : pprint(vec.data(), vec.size()) {}

    /// This is need to handle VectorBlock specialty
    template <typename U = T,
              typename = std::enable_if_t<
                  eigen_utils::internal::is_vector_block<U>::value>>
    pprint(const T &m)
        : matrix(m), format(Format::format(m.rows(), m.cols())) {}

    template <typename, typename, typename> friend struct fmt::formatter;

protected:
    Ref matrix;
    Eigen::IOFormat format;
};

} // namespace fmt_utils

namespace fmt {

template <typename T, typename Format>
struct formatter<fmt_utils::pprint<T, Format>> {

    using Index = Eigen::Index;

    Index max_rows = 5;
    Index max_cols = 5;
    Index max_size = 10;

    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it == end)
            return it;
        const char end_token = '}'; // the end of parse
        const std::array arg_tokens{'r', 'c', 's'};
        auto is_arg_token = [&](const auto &it) {
            for (auto c : arg_tokens) {
                if (*it == c)
                    return true;
            }
            return false;
        };
        // call this when previous item is token,
        // the sucessful arg is set to dest
        auto get_token_arg = [&](auto &it, auto &dest) {
            std::vector<char> ss;
            // read until another token or end
            while (!is_arg_token(it) && *it != end_token && it != end) {
                ss.push_back(*it);
                ++it;
            }
            // handle token arg
            if (ss.empty()) {
                // use actual size
                dest = -1;
            } else {
                // parse
                std::from_chars(ss.data(), ss.data() + ss.size(), dest);
            }
        };
        /// e.g 'r9c10s11'
        do {
            switch (*it) {
            case 'r': {
                get_token_arg(++it, max_rows);
                break;
            }
            case 'c': {
                get_token_arg(++it, max_cols);
                break;
            }
            case 's': {
                get_token_arg(++it, max_size);
                break;
            }
            default: { // ignore unrecongnized
                break;
            }
            }
        } while (*it != end_token && it != end);
        return it;
    }

    template <typename FormatContext>
    auto format(const fmt_utils::pprint<T, Format> &pp, FormatContext &ctx)
        -> decltype(ctx.out()) {
        auto it = ctx.out();
        if (pp.matrix.size() == 0) {
            return format_to(it, "(empty)");
        }
        // SPDLOG_TRACE("max_rows {}", max_rows);
        // SPDLOG_TRACE("max_cols {}", max_cols);
        // SPDLOG_TRACE("max_size {}", max_size);
        // shape
        it = format_to(it, "({},{})", pp.matrix.rows(), pp.matrix.cols());
        // omit content if any of the parsed args is zero
        if ((max_rows == 0) || (max_cols == 0) || (max_size == 0)) {
            return format_to(it, "[...]");
        }
        // dynamic size if args is not specified (-1)
        if (max_rows < 0)
            max_rows = pp.matrix.rows();
        if (max_cols < 0)
            max_cols = pp.matrix.cols();
        if (max_size < 0)
            max_size = pp.matrix.size();
        // pprint content
        std::stringstream ss;
        return format_to(it, "{}",
                         fmt_utils::pprint_matrix(ss, pp.matrix, pp.format,
                                                  max_rows, max_cols, max_size)
                             .str());
    }
};

template <typename Derived, typename Char>
struct formatter<Derived, Char,
                 std::enable_if_t<eigen_utils::is_eigen_v<Derived>>>
    : formatter<fmt_utils::pprint<Derived>> {};

template <typename T, typename Char, typename... Rest>
struct formatter<std::vector<T, Rest...>, Char,
                 std::enable_if_t<fmt_utils::scalar_traits<T>::value>>
    : formatter<fmt_utils::pprint<T>> {};

template <typename T, std::size_t size, typename Char>
struct formatter<std::array<T, size>, Char,
                 std::enable_if_t<fmt_utils::scalar_traits<T>::value>>
    : formatter<fmt_utils::pprint<T>> {
    template <typename FormatContext>
    auto format(const std::array<T, size> &arr, FormatContext &ctx)
        -> decltype(ctx.out()) {
        return formatter<fmt_utils::pprint<T>>::format(
            fmt_utils::pprint{arr.data(), static_cast<Eigen::Index>(arr.size())}, ctx);
    }
};

} // namespace fmt

#pragma once
#include "utils.h"
#include <fmt/format.h>
#include <optional>
#include <variant>
// list of stl containers is from https://stackoverflow.com/a/31105859/1824372
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fmt_utils {

// specialize a type for all of the STL containers.
namespace internal {

// list of stl containers is from https://stackoverflow.com/a/31105859/1824372
template <typename T> struct is_stl_container : std::false_type {};
template <typename T, std::size_t N>
struct is_stl_container<std::array<T, N>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::vector<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::deque<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::list<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::forward_list<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::set<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::multiset<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::map<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::multimap<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::unordered_set<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::unordered_multiset<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::unordered_map<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::unordered_multimap<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::stack<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::queue<Args...>> : std::true_type {};
template <typename... Args>
struct is_stl_container<std::priority_queue<Args...>> : std::true_type {};

} // namespace internal

template <typename T>
using is_stl_container = internal::is_stl_container<std::decay_t<T>>;

} // namespace fmt_utils

namespace fmt {

template <typename T> struct formatter<std::optional<T>> : formatter<T> {
    template <typename FormatContext>
    auto format(const std::optional<T> &opt, FormatContext &ctx)
        -> decltype(ctx.out()) {
        if (opt) {
            return formatter<T>::format(opt.value(), ctx);
        }
        return format_to(ctx.out(), "(nullopt)");
    }
};

template <typename T, typename U>
struct formatter<std::pair<T, U>> : fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const std::pair<T, U> &p, FormatContext &ctx)
        -> decltype(ctx.out()) {
        return format_to(ctx.out(), "{{{}: {}}}", p.first, p.second);
    }
};

template <>
struct formatter<std::monostate, char, void>
    : fmt_utils::nullspec_formatter_base {
    template <typename FormatContext>
    auto format(const std::monostate &, FormatContext &ctx)
        -> decltype(ctx.out()) {
        return format_to(ctx.out(), "<undef>");
    }
};

template <typename T, typename... Rest>
struct formatter<std::variant<T, Rest...>, char, void>
    : fmt_utils::charspec_formatter_base<'l', '0', 's'> {
    // 0: value
    // s: value (t)
    // l: value (type) (default)
    template <typename FormatContext>
    auto format(const std::variant<T, Rest...> &v, FormatContext &ctx)
        -> decltype(ctx.out()) {
        auto it = ctx.out();
        auto spec = spec_handler();
        std::visit(
            [&](auto &&arg) {
                std::string f;
                using A = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<A, std::string> ||
                              fmt_utils::is_c_str<A>::value) {
                    f = "\"{}\"";
                } else {
                    f = "{}";
                }
                // format value
                it = format_to(it, f, arg);
                if (spec == '0') {
                    return;
                }
                // format with type code
                std::string t;
                f = " ({})";
                if constexpr (std::is_same_v<A, std::monostate>) {
                    t = "undef";
                } else if constexpr (std::is_same_v<A, bool>) {
                    t = "bool";
                } else if constexpr (std::is_same_v<A, int>) {
                    t = "int";
                } else if constexpr (std::is_same_v<A, double>) {
                    t = "doub";
                } else if constexpr (std::is_same_v<A, std::string> ||
                                     fmt_utils::is_c_str<A>::value) {
                    t = "str";
                } else {
                    // fallback to rtti type id
                    t = typeid(A).name();
                    // static_assert(meta::always_false<T>::value, "NOT KNOWN
                    // TYPE");
                }
                switch (spec) {
                case 's': {
                    it = format_to(it, f, t[0]);
                    return;
                }
                case 'l': {
                    it = format_to(it, f, t);
                    return;
                }
                }
                return;
            },
            v);
        return it;
    }
};

} // namespace fmt

namespace std {

/// Provide output stream operator for all containers
/// this will be superseded by any formatter specialization
template <typename OStream, typename T, typename... Rest,
          template <typename...> typename U,
          typename = std::enable_if_t<
              // stl containers
              fmt_utils::is_stl_container<U<T, Rest...>>::value>>
OStream &operator<<(OStream &os, const U<T, Rest...> &cont) {
    os << "{";
    bool sep = false;
    for (const auto &v : cont) {
        if (sep) {
            os << ", ";
        }
        sep = true;
        os << fmt::format("{}", v);
    }
    os << "}";
    return os;
}

/// specialize for std::array
template <typename OStream, typename T, std::size_t size>
OStream &operator<<(OStream &os, const std::array<T, size> &cont) {
    os << "{";
    bool sep = false;
    for (const auto &v : cont) {
        if (sep) {
            os << ", ";
        }
        sep = true;
        os << fmt::format("{}", v);
    }
    os << "}";
    return os;
}

} // namespace std

#pragma once
#include <string>
#include <type_traits>
#include <algorithm>
#include <cassert>
#include <array>

namespace fmt_utils {

template <class T>
struct is_c_str
    : std::integral_constant<
          bool, std::is_same_v<char const *, typename std::decay_t<T>> ||
                    std::is_same_v<char *, typename std::decay_t<T>>> {};

template <typename String>
auto remove_space(String &&str_) {
    String str{std::forward<String>(str_)};
    str.erase(std::remove_if(str.begin(), str.end(),
                             [](const auto &c) { return std::isspace(c); }),
              str.end());
    return str;
}

template <typename Int>
constexpr auto bitwidth(Int v) {
    std::size_t digits = 0;
    for (; v > 0; v >>= 1)
        digits++;
    return digits;
}

/// @brief Convert unsigned integer type to string with given base.
template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto itoa(T n, unsigned base) {
    assert(base >= 2 && base <= 64);
    const char digits[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-";
    std::string retval;
    while (n != 0) {
        retval += digits[n % base];
        n /= base;
    }
    std::reverse(retval.begin(), retval.end());
    return retval;
};

namespace internal {

/// @brief Parse single charactor format spec.
template <char... _charset>
struct charspec {

    charspec(char value_) : value{value_} {}

    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it == end) {
            return it;
        }
        // only check the first
        bool valid = false;
        auto spec = *it++;
        for (auto c : charset) {
            if (spec == c) {
                valid = true;
                break;
            }
        }
        if (valid) {
            value = spec;
        }
        return it;
    }
    constexpr auto operator()() const { return value; }

private:
    char value;
    std::array<char, sizeof...(_charset)> charset{_charset...};
};

} // namespace internal

/// @brief Formatter base class that handles single char format spec.
template <char default_char, char... charset>
struct charspec_formatter_base {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
        return spec_handler.parse(ctx);
    }

protected:
    internal::charspec<default_char, charset...> spec_handler{default_char};
};

/// @brief Formatter base class that ignores the format spec.
struct nullspec_formatter_base {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
};

} // namespace fmt_utils

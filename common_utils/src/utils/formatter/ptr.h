#pragma once
#include "utils.h"
#include <fmt/format.h>

namespace fmt_utils {

/// @brief Convert pointer to the memory address
template <typename T> struct ptr {
    using value_t = std::uintptr_t;
    const value_t value;
    ptr(const T *p) : value(reinterpret_cast<value_t>(p)) {}
};

} // namespace fmt_utils

namespace fmt {

template <typename T> struct formatter<fmt_utils::ptr<T>>
: fmt_utils::charspec_formatter_base<'z', 'x', 'y'> {
    // x: base16, i.e., hex
    // y: base32
    // z: base64 (default)
    template <typename FormatContext>
    auto format(const fmt_utils::ptr<T> ptr, FormatContext &ctx)
        -> decltype(ctx.out()) {
        auto it = ctx.out();
        auto spec = spec_handler();
        switch (spec) {
        case 'x':
            return format_to(it, "{:x}", ptr.value);
        case 'y':
            return format_to(it, "{}", fmt_utils::itoa(ptr.value, 32));
        case 'z':
            return format_to(it, "{}", fmt_utils::itoa(ptr.value, 64));
        }
        return it;
    }
};

} // namespace fmt

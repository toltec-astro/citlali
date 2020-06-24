#pragma once

#include "utils.h"
#include <fmt/format.h>

namespace fmt {

template <> struct formatter<std::byte> : fmt_utils::charspec_formatter_base<'i', 'x'> {
    // x: base16, i.e., hex
    // i: int
    template <typename FormatContext>
    auto format(const std::byte& byte, FormatContext &ctx)
        -> decltype(ctx.out()) {
        auto it = ctx.out();
        auto spec = spec_handler();
        auto value = std::to_integer<int>(byte);
        switch (spec) {
        case 'x':
            return format_to(it, "{:x}", value);
        case 'i':
            return format_to(it, "{}", value);
        }
        return it;
    }
};

} // namespace fmt
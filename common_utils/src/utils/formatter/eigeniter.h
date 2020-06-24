#pragma once
#include "../eigeniter.h"
#include "utils.h"
#include "ptr.h"

namespace fmt {

template <typename T> struct formatter<eigeniter::EigenIter<T>>
    : fmt_utils::charspec_formatter_base<'l', 's'> {
    // s: short fmt
    // l: long fmt
    template <typename FormatContext>
    auto format(const eigeniter::EigenIter<T> &ei, FormatContext &ctx)
        -> decltype(ctx.out()) {
        using fmt_utils::ptr;
        auto it = ctx.out();
        switch (spec_handler()) {
        case 'l': {
            auto [nrows, ncols, outer, inner, outer_stride, inner_stride] = ei.internals();
            return format_to(it, "[{}]({} rc=({}, {}) oi=({}, {}) stride=({}, {}))", ei.n,
                             ptr(ei.data), nrows, ncols, outer, inner, outer_stride, inner_stride);
        }
        case 's':
            return format_to(it, "[{}]({})", ei.n, ptr(ei.data));
        }
        return it;
    }
};

} // namespace fmt

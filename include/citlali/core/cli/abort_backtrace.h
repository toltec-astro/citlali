#pragma once

#if defined(__linux__)
#include <citlali/core/mapmaking/jinc_debug_breadcrumb.h>
#include <citlali/core/pipeline/map_output_debug_breadcrumb.h>

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <unistd.h>
#endif

namespace citlali::cli {

#if defined(__linux__)
inline void abort_backtrace_handler(int sig) {
    void *frames[128];
    int n = ::backtrace(frames, static_cast<int>(sizeof(frames) / sizeof(frames[0])));
    const char msg[] =
        "\n[citlali] fatal signal received; stack trace follows:\n";
    const ssize_t nw = ::write(STDERR_FILENO, msg, sizeof(msg) - 1);
    if (nw < 0) {
        // best-effort only in signal context
    }
    const auto &map_crumb = pipeline::get_map_output_debug_breadcrumb();
    if (map_crumb.valid) {
        std::fprintf(stderr,
                     "[citlali] map output breadcrumb: stage=%s map_i=%lld map_index=%lld "
                     "stokes=%lld array=%lld hdu_index=%lld hdu_count=%lld flag=%d file=%s\n",
                     map_crumb.stage,
                     map_crumb.map_i,
                     map_crumb.map_index,
                     map_crumb.stokes_index,
                     map_crumb.array_index,
                     map_crumb.hdu_index,
                     map_crumb.hdu_count,
                     map_crumb.flag_value,
                     map_crumb.filepath);
    }
    const auto &crumb = mapmaking::get_jinc_debug_breadcrumb();
    if (crumb.valid) {
        std::fprintf(stderr,
                     "[citlali] jinc breadcrumb: stage=%s det_col=%lld det_uid=%d sample=%lld map_index=%lld array=%lld "
                     "pixel=(%d,%d) subpix=%d map_block=[%d:%d,%d:%d] jinc_offset=(%d,%d) size=%dx%d\n",
                     crumb.stage,
                     crumb.det_col,
                     crumb.det_uid,
                     crumb.sample,
                     crumb.map_index,
                     crumb.array_index,
                     crumb.pixel_row,
                     crumb.pixel_col,
                     crumb.subpix_idx,
                     crumb.lower_row,
                     crumb.upper_row,
                     crumb.lower_col,
                     crumb.upper_col,
                     crumb.jinc_lower_row,
                     crumb.jinc_lower_col,
                     crumb.size_rows,
                     crumb.size_cols);
    }
    ::backtrace_symbols_fd(frames, n, STDERR_FILENO);
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}

inline void install_abort_backtrace_handler() {
    ::signal(SIGABRT, abort_backtrace_handler);
    ::signal(SIGBUS, abort_backtrace_handler);
    ::signal(SIGFPE, abort_backtrace_handler);
    ::signal(SIGILL, abort_backtrace_handler);
    ::signal(SIGSEGV, abort_backtrace_handler);
}
#else
inline void install_abort_backtrace_handler() {}
#endif

}  // namespace citlali::cli

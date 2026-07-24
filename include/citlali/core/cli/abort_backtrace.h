#pragma once

#if defined(__linux__)
#include <citlali/core/mapmaking/jinc_debug_breadcrumb.h>
#include <citlali/core/pipeline/map_output_debug_breadcrumb.h>

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <execinfo.h>
#include <initializer_list>
#include <unistd.h>
#endif

namespace citlali::cli {

#if defined(__linux__)
inline char *append_signal_literal(char *out, const char *end,
                                   const char *text) {
    while (out != end && *text != '\0') {
        *out++ = *text++;
    }
    return out;
}

inline char *append_signal_unsigned(char *out, const char *end,
                                    std::uintptr_t value,
                                    unsigned int base = 10) {
    char reversed[2 * sizeof(value) + 1];
    std::size_t size = 0;
    do {
        const auto digit = static_cast<unsigned int>(value % base);
        reversed[size++] =
            static_cast<char>(digit < 10 ? '0' + digit : 'a' + digit - 10);
        value /= base;
    } while (value != 0 && size < sizeof(reversed));
    while (out != end && size > 0) {
        *out++ = reversed[--size];
    }
    return out;
}

inline char *append_signal_signed(char *out, const char *end, int value) {
    if (value < 0 && out != end) {
        *out++ = '-';
    }
    const auto magnitude =
        value < 0 ? static_cast<std::uintptr_t>(-(value + 1)) + 1
                  : static_cast<std::uintptr_t>(value);
    return append_signal_unsigned(out, end, magnitude);
}

inline void write_signal_fault_record(int sig, const siginfo_t *info) {
    char message[256];
    char *out = message;
    const char *end = message + sizeof(message);
    out = append_signal_literal(out, end, "\n[citlali] fatal signal=");
    out = append_signal_signed(out, end, sig);
    out = append_signal_literal(out, end, " code=");
    out = append_signal_signed(out, end, info == nullptr ? 0 : info->si_code);
    out = append_signal_literal(out, end, " address=0x");
    out = append_signal_unsigned(
        out, end,
        reinterpret_cast<std::uintptr_t>(
            info == nullptr ? nullptr : info->si_addr),
        16);
    out = append_signal_literal(out, end, "\n");
    const ssize_t nw =
        ::write(STDERR_FILENO, message, static_cast<std::size_t>(out - message));
    if (nw < 0) {
        // best-effort only in signal context
    }
}

inline void abort_backtrace_handler(int sig, siginfo_t *info, void *) {
    // Emit the fault code and address before calling backtrace(), which is not
    // guaranteed to survive a damaged stack or executable mapping.
    write_signal_fault_record(sig, info);
    void *frames[128];
    int n = ::backtrace(frames, static_cast<int>(sizeof(frames) / sizeof(frames[0])));
    const char msg[] = "[citlali] stack trace follows:\n";
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
    struct sigaction action {};
    action.sa_handler = SIG_DFL;
    ::sigemptyset(&action.sa_mask);
    ::sigaction(sig, &action, nullptr);
    ::raise(sig);
}

inline void install_abort_backtrace_handler() {
    struct sigaction action {};
    action.sa_sigaction = abort_backtrace_handler;
    ::sigemptyset(&action.sa_mask);
    action.sa_flags = SA_SIGINFO | SA_RESETHAND;
    for (const int signal :
         {SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV}) {
        ::sigaction(signal, &action, nullptr);
    }
}
#else
inline void install_abort_backtrace_handler() {}
#endif

}  // namespace citlali::cli

#pragma once

#include <CCfits/CCfits>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <exception>

namespace citlali::cli {

inline int report_unhandled_fits_error(const CCfits::FitsError &error) {
    SPDLOG_CRITICAL("Unhandled CCfits::FitsError: {}", error.message());
    return EXIT_FAILURE;
}

inline int report_unhandled_std_exception(const std::exception &error) {
    SPDLOG_CRITICAL("Unhandled exception: {}", error.what());
    return EXIT_FAILURE;
}

inline int report_unhandled_unknown_exception() {
    SPDLOG_CRITICAL("Unhandled non-standard exception");
    return EXIT_FAILURE;
}

template <class RunMain>
int run_with_exception_reporting(RunMain &&run_main) {
    try {
        return run_main();
    } catch (const CCfits::FitsError &e) {
        return report_unhandled_fits_error(e);
    } catch (const std::exception &e) {
        return report_unhandled_std_exception(e);
    } catch (...) {
        return report_unhandled_unknown_exception();
    }
}

}  // namespace citlali::cli

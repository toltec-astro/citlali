#pragma once

#include <citlali/core/compat/kidscpp_raw_timestream.h>

namespace citlali::cli {

template <class Logger>
void log_kids_data_spec(const Logger &logger) {
    logger->info("use KIDs data spec: {}",
                 citlali::compat::kidscpp::data_spec);
}

}  // namespace citlali::cli

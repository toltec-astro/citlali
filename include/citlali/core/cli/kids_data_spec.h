#pragma once

#include <kids/core/kidsdata.h>

namespace citlali::cli {

template <class Logger>
void log_kids_data_spec(const Logger &logger) {
    logger->info("use KIDs data spec: {}", predefs::kidsdata::name);
}

}  // namespace citlali::cli

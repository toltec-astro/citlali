#pragma once

#include <kids/toltec/toltec.h>

namespace citlali::cli {

template <class Logger>
void log_kids_data_spec(const Logger &logger) {
    logger->info("use KIDs data spec: {}", kids::toltec::name);
}

}  // namespace citlali::cli

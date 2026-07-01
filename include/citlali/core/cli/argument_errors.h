#pragma once

#include <ostream>

namespace citlali::cli {

inline void report_missing_config_file_argument(std::ostream &os) {
    os << "Invalid argument. Type --help for usage.\n";
}

}  // namespace citlali::cli

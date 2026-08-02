#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace citlali::utils {

inline std::int64_t round_half_up_slot(double coordinate) {
    if (!std::isfinite(coordinate)) {
        throw std::invalid_argument("slot coordinate must be finite");
    }
    const double rounded = std::floor(coordinate + 0.5);
    const double inclusive_lower =
        static_cast<double>(std::numeric_limits<std::int64_t>::min());
    // `double(INT64_MAX)` rounds to 2^63, which is already outside int64.
    // Compare against that value as an exclusive upper bound before casting.
    const double exclusive_upper = -inclusive_lower;
    if (rounded < inclusive_lower || rounded >= exclusive_upper) {
        throw std::overflow_error("slot identity exceeds int64 range");
    }
    return static_cast<std::int64_t>(rounded);
}

}  // namespace citlali::utils

#pragma once

#include <cstddef>
#include <stdexcept>

namespace citlali::engine_detail {

inline void require_gap_stream_cardinality(std::size_t kids_count,
                                           std::size_t time_count) {
    if (kids_count != time_count) {
        throw std::runtime_error(
            "rawobs KIDs and detector-time cardinalities differ");
    }
}

}  // namespace citlali::engine_detail

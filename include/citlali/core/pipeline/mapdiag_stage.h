#pragma once

#include <string>

#include <citlali/core/mapmaking/map.h>

namespace citlali::pipeline {

template <mapmaking::MapType map_t>
std::string mapdiag_stage_name() {
    if constexpr (map_t == mapmaking::FilteredObs) {
        return "filtered_obs";
    }
    else if constexpr (map_t == mapmaking::RawCoadd) {
        return "raw_coadd";
    }
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        return "filtered_coadd";
    }
    else {
        return "raw_obs";
    }
}

}  // namespace citlali::pipeline

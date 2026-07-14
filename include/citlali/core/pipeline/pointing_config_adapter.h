#pragma once

#include <citlali/core/config/pointing_config.h>

#include <string>

namespace citlali::pipeline {

template <class PtcProcessor>
void adapt_pointing_config_one_way(
    const citlali::config::PointingConfig &effective,
    PtcProcessor &processor) {
    processor.fruit_loops_source_center_mode =
        std::string(citlali::config::to_string(
            effective.fruitloops_center_mode));
    processor.fruit_loops_header_center_max_radius_arcsec =
        effective.header_max_radius_arcsec;
    processor.fruit_loops_header_center_require_coverage =
        effective.header_require_coverage;
}

}  // namespace citlali::pipeline

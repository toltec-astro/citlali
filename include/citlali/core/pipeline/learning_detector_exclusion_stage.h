#pragma once

#include <citlali/core/config/timestream_enums.h>

#include <string_view>

namespace citlali::pipeline {

inline bool map_pixel_outlier_detector_exclusion_applies_at_stage(
    citlali::config::MapPixelOutlierDetectorExclusionApplication application,
    std::string_view stage) {
    const bool is_cleaning =
        stage == "pre_rtc_detector_exclusion" ||
        stage == "pre_ptc_detector_exclusion";
    const bool is_mapmaking =
        stage == "pre_mapmaking_detector_exclusion";
    switch (application) {
        case citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_cleaning:
            return is_cleaning;
        case citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_mapmaking:
            return is_mapmaking;
    }
    return false;
}

}  // namespace citlali::pipeline

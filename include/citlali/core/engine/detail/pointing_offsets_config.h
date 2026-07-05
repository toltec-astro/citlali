#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace citlali::engine_detail {

inline std::string normalized_pointing_axis_name(std::string axis_name) {
    std::transform(axis_name.begin(), axis_name.end(), axis_name.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return axis_name;
}

template <class PointingOffsetMap, class MjdValues, class TypedOffsets>
void mirror_typed_pointing_offsets(
    const PointingOffsetMap &pointing_offsets_arcsec,
    const MjdValues &pointing_offsets_modified_julian_date,
    TypedOffsets &typed_offsets) {
    typed_offsets.enabled = true;
    const auto &az_offsets = pointing_offsets_arcsec.at("az");
    typed_offsets.az_arcsec.assign(
        az_offsets.data(), az_offsets.data() + az_offsets.size());
    const auto &alt_offsets = pointing_offsets_arcsec.at("alt");
    typed_offsets.alt_arcsec.assign(
        alt_offsets.data(), alt_offsets.data() + alt_offsets.size());
    typed_offsets.modified_julian_date.assign(
        pointing_offsets_modified_julian_date.data(),
        pointing_offsets_modified_julian_date.data() +
            pointing_offsets_modified_julian_date.size());
}

}  // namespace citlali::engine_detail

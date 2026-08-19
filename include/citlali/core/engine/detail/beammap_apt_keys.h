#pragma once

#include <string_view>

namespace beammap_apt_keys {

inline constexpr std::string_view flag() {
    return "flag";
}

inline constexpr std::string_view flag2() {
    return "flag2";
}

inline constexpr std::string_view kids_flag() {
    return "kids_flag";
}

inline bool is_flag(std::string_view key) {
    return key == flag();
}

inline bool is_flag2(std::string_view key) {
    return key == flag2();
}

inline bool is_legacy_observation_match_field(std::string_view key) {
    return key == "fr" || key == "f_out" || key == "Qr";
}

} // namespace beammap_apt_keys

#pragma once

#include <string_view>

namespace beammap_apt_keys {

inline constexpr std::string_view flag2() {
    return "flag2";
}

inline bool is_flag2(std::string_view key) {
    return key == flag2();
}

} // namespace beammap_apt_keys

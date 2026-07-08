#pragma once

#include <string_view>

namespace citlali::config {

inline constexpr std::string_view null_config_value() {
    return "null";
}

inline bool is_null_config_value(std::string_view value) {
    return value == null_config_value();
}

inline bool has_config_value(std::string_view value) {
    return !is_null_config_value(value);
}

inline bool is_empty_or_null_config_value(std::string_view value) {
    return value.empty() || is_null_config_value(value);
}

inline bool has_nonempty_config_value(std::string_view value) {
    return !is_empty_or_null_config_value(value);
}

}  // namespace citlali::config

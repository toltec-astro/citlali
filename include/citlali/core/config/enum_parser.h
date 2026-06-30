#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

namespace citlali::config {

template <typename Enum>
struct EnumName {
    Enum value;
    std::string_view name;
};

template <typename Enum, std::size_t N>
std::optional<Enum> parse_enum(
    std::string_view value,
    const std::array<EnumName<Enum>, N> &names) {
    for (const auto &entry : names) {
        if (entry.name == value) {
            return entry.value;
        }
    }
    return std::nullopt;
}

template <typename Enum, std::size_t N>
std::string_view enum_name(
    Enum value,
    const std::array<EnumName<Enum>, N> &names,
    std::string_view fallback = "unknown") {
    for (const auto &entry : names) {
        if (entry.value == value) {
            return entry.name;
        }
    }
    return fallback;
}

}  // namespace citlali::config

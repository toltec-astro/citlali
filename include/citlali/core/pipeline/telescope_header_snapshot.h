#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace citlali::pipeline::sci_align {

enum class TelescopeHeaderNumericType {
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float32,
    float64,
};

struct TelescopeHeaderDimensionSnapshot {
    std::string name;
    std::size_t size = 0;
};

using TelescopeHeaderNumericValues =
    std::variant<std::vector<signed char>,
                 std::vector<unsigned char>,
                 std::vector<short>,
                 std::vector<unsigned short>,
                 std::vector<int>,
                 std::vector<unsigned int>,
                 std::vector<long long>,
                 std::vector<unsigned long long>,
                 std::vector<float>,
                 std::vector<double>>;

struct TelescopeHeaderSnapshot {
    TelescopeHeaderNumericType type = TelescopeHeaderNumericType::float64;
    std::vector<TelescopeHeaderDimensionSnapshot> dimensions;
    TelescopeHeaderNumericValues values = std::vector<double>{};
    std::optional<std::string> units;
};

inline std::size_t telescope_header_element_count(
    const TelescopeHeaderSnapshot &snapshot) {
    if (snapshot.dimensions.empty()) {
        return 1;
    }
    if (snapshot.dimensions.size() != 1 ||
        snapshot.dimensions.front().name.empty() ||
        snapshot.dimensions.front().size == 0) {
        throw std::invalid_argument(
            "telescope header snapshot must preserve a scalar or nonempty vector shape");
    }
    return snapshot.dimensions.front().size;
}

inline std::size_t telescope_header_value_count(
    const TelescopeHeaderSnapshot &snapshot) {
    return std::visit([](const auto &values) { return values.size(); },
                      snapshot.values);
}

inline TelescopeHeaderNumericType telescope_header_value_type(
    const TelescopeHeaderSnapshot &snapshot) {
    return std::visit(
        [](const auto &values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            if constexpr (std::is_same_v<Value, signed char>) {
                return TelescopeHeaderNumericType::int8;
            }
            else if constexpr (std::is_same_v<Value, unsigned char>) {
                return TelescopeHeaderNumericType::uint8;
            }
            else if constexpr (std::is_same_v<Value, short>) {
                return TelescopeHeaderNumericType::int16;
            }
            else if constexpr (std::is_same_v<Value, unsigned short>) {
                return TelescopeHeaderNumericType::uint16;
            }
            else if constexpr (std::is_same_v<Value, int>) {
                return TelescopeHeaderNumericType::int32;
            }
            else if constexpr (std::is_same_v<Value, unsigned int>) {
                return TelescopeHeaderNumericType::uint32;
            }
            else if constexpr (std::is_same_v<Value, long long>) {
                return TelescopeHeaderNumericType::int64;
            }
            else if constexpr (std::is_same_v<Value, unsigned long long>) {
                return TelescopeHeaderNumericType::uint64;
            }
            else if constexpr (std::is_same_v<Value, float>) {
                return TelescopeHeaderNumericType::float32;
            }
            else {
                return TelescopeHeaderNumericType::float64;
            }
        },
        snapshot.values);
}

inline void validate_telescope_header_snapshot(
    const TelescopeHeaderSnapshot &snapshot, const std::string &name) {
    const auto expected_count = telescope_header_element_count(snapshot);
    if (telescope_header_value_count(snapshot) != expected_count) {
        throw std::invalid_argument(
            "telescope header snapshot '" + name +
            "' has a value/shape cardinality mismatch");
    }
    if (telescope_header_value_type(snapshot) != snapshot.type) {
        throw std::invalid_argument(
            "telescope header snapshot '" + name +
            "' has a native type/value representation mismatch");
    }
}

inline std::vector<double> telescope_header_legacy_double_view(
    const TelescopeHeaderSnapshot &snapshot, const std::string &name) {
    validate_telescope_header_snapshot(snapshot, name);
    std::vector<double> result;
    result.reserve(telescope_header_value_count(snapshot));
    std::visit(
        [&](const auto &values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            for (const auto value : values) {
                if constexpr (std::is_integral_v<Value>) {
                    constexpr long double max_exact_integer_double =
                        9007199254740991.0L;
                    const long double exact = static_cast<long double>(value);
                    if (exact < -max_exact_integer_double ||
                        exact > max_exact_integer_double) {
                        throw std::invalid_argument(
                            "telescope header snapshot '" + name +
                            "' cannot be represented losslessly by the legacy double compatibility view");
                    }
                }
                result.push_back(static_cast<double>(value));
            }
        },
        snapshot.values);
    return result;
}

}  // namespace citlali::pipeline::sci_align

#pragma once

#include <citlali/core/pipeline/telescope_header_snapshot.h>
#include <citlali/core/utils/netcdf_io.h>

#include <Eigen/Core>
#include <netcdf>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::engine_detail::sci_align_netcdf {

// This selected-fixture fingerprint identifies only the compatible legacy
// column layout.  It does not establish physical producer timestamp semantics.
inline constexpr std::string_view legacy_toltec_ts_long_name =
    "ClockTime (sec), PpsCount (pps ticks), ClockCount (clock ticks), "
    "PacketCount (packet ticks), PpsTime (clock ticks), "
    "ClockTimeNanoSec (nsec)";

inline Eigen::Index require_eigen_index_size(
    std::size_t size, const std::string &name) {
    if (size > static_cast<std::size_t>(
                   std::numeric_limits<Eigen::Index>::max())) {
        throw DataIOError{"netCDF variable '" + name +
                          "' exceeds the Eigen index range"};
    }
    return static_cast<Eigen::Index>(size);
}

inline Eigen::Index require_eigen_matrix_rows(
    std::size_t rows, std::size_t columns, const std::string &name) {
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<Eigen::Index>::max());
    if (columns == 0 || rows > maximum / columns) {
        throw DataIOError{"netCDF variable '" + name +
                          "' exceeds the Eigen matrix index range"};
    }
    return static_cast<Eigen::Index>(rows);
}

inline netCDF::NcVar require_variable(const netCDF::NcFile &file,
                                     const std::string &name) {
    auto variable = file.getVar(name);
    if (variable.isNull()) {
        throw DataIOError{"required netCDF variable '" + name +
                          "' is absent"};
    }
    return variable;
}

inline void require_type(
    const netCDF::NcVar &variable, const std::string &name,
    std::initializer_list<netCDF::NcType::ncType> permitted) {
    const auto actual = variable.getType().getTypeClass();
    if (std::find(permitted.begin(), permitted.end(), actual) !=
        permitted.end()) {
        return;
    }
    std::ostringstream expected;
    bool first = true;
    for (const auto type : permitted) {
        if (!first) {
            expected << ",";
        }
        expected << netCDF::NcType(static_cast<nc_type>(type))
                        .getTypeClassName();
        first = false;
    }
    throw DataIOError{
        "netCDF variable '" + name + "' has incompatible type " +
        variable.getType().getTypeClassName() + "; expected " +
        expected.str()};
}

inline void require_scalar(const netCDF::NcVar &variable,
                           const std::string &name) {
    if (variable.getDimCount() != 0) {
        throw DataIOError{"netCDF variable '" + name +
                          "' must be a rank-0 scalar"};
    }
}

inline std::size_t require_scalar_or_nonempty_vector(
    const netCDF::NcVar &variable, const std::string &name) {
    const auto dimensions = variable.getDims();
    if (dimensions.empty()) {
        return 1;
    }
    if (dimensions.size() == 1 && dimensions.front().getSize() > 0) {
        return dimensions.front().getSize();
    }
    throw DataIOError{
        "netCDF variable '" + name +
        "' must be a rank-0 scalar or nonempty rank-1 vector"};
}

inline netCDF::NcDim require_nonempty_vector(
    const netCDF::NcVar &variable, const std::string &name) {
    const auto dimensions = variable.getDims();
    if (dimensions.size() != 1 || dimensions.front().getSize() == 0) {
        throw DataIOError{"netCDF variable '" + name +
                          "' must be a nonempty rank-1 vector"};
    }
    return dimensions.front();
}

inline void require_vector_on_dimension(
    const netCDF::NcVar &variable, const std::string &name,
    const netCDF::NcDim &expected_dimension) {
    const auto dimensions = variable.getDims();
    if (dimensions.size() != 1 ||
        dimensions.front().getId() != expected_dimension.getId() ||
        dimensions.front().getSize() != expected_dimension.getSize()) {
        throw DataIOError{
            "netCDF variable '" + name +
            "' must be a rank-1 vector on the authoritative '" +
            expected_dimension.getName() + "' dimension"};
    }
}

inline void require_matrix_shape(const netCDF::NcVar &variable,
                                 const std::string &name,
                                 std::size_t second_dimension_size) {
    const auto dimensions = variable.getDims();
    if (dimensions.size() != 2 || dimensions[0].getSize() == 0 ||
        dimensions[1].getSize() != second_dimension_size) {
        throw DataIOError{
            "netCDF variable '" + name +
            "' must have nonempty exact shape (time," +
            std::to_string(second_dimension_size) + ")"};
    }
}

inline void require_exact_text_attribute(
    const netCDF::NcVar &variable, const std::string &name,
    const std::string &attribute_name, std::string_view expected) {
    const auto attributes = variable.getAtts();
    const auto attribute = attributes.find(attribute_name);
    if (attribute == attributes.end() ||
        attribute->second.getType().getTypeClass() !=
            netCDF::NcType::nc_CHAR) {
        throw DataIOError{
            "netCDF variable '" + name + "' lacks required text " +
            attribute_name + " compatibility fingerprint"};
    }
    std::string actual;
    attribute->second.getValues(actual);
    if (actual != expected) {
        throw DataIOError{
            "netCDF variable '" + name + "' has incompatible " +
            attribute_name + " compatibility fingerprint"};
    }
}

inline void require_legacy_toltec_timing_schema(
    const netCDF::NcVar &timestamps, const netCDF::NcVar &in_phase,
    const netCDF::NcVar &quadrature) {
    constexpr std::string_view timestamp_name = "Data.Toltec.Ts";
    constexpr std::string_view in_phase_name = "Data.Toltec.Is";
    constexpr std::string_view quadrature_name = "Data.Toltec.Qs";

    require_type(in_phase, std::string(in_phase_name),
                 {netCDF::NcType::nc_INT});
    require_type(quadrature, std::string(quadrature_name),
                 {netCDF::NcType::nc_INT});
    const auto in_phase_dimensions = in_phase.getDims();
    const auto quadrature_dimensions = quadrature.getDims();
    if (in_phase_dimensions.size() != 2 ||
        in_phase_dimensions[0].getSize() == 0 ||
        in_phase_dimensions[1].getSize() == 0) {
        throw DataIOError{
            "netCDF variable 'Data.Toltec.Is' must be a nonempty rank-2 "
            "integer detector matrix"};
    }
    if (quadrature_dimensions.size() != 2 ||
        quadrature_dimensions[0].getSize() == 0 ||
        quadrature_dimensions[1].getSize() == 0) {
        throw DataIOError{
            "netCDF variable 'Data.Toltec.Qs' must be a nonempty rank-2 "
            "integer detector matrix"};
    }
    if (in_phase_dimensions[0].getId() !=
            quadrature_dimensions[0].getId() ||
        in_phase_dimensions[0].getSize() !=
            quadrature_dimensions[0].getSize() ||
        in_phase_dimensions[1].getId() !=
            quadrature_dimensions[1].getId() ||
        in_phase_dimensions[1].getSize() !=
            quadrature_dimensions[1].getSize()) {
        throw DataIOError{
            "Data.Toltec.Is and Data.Toltec.Qs must share the same "
            "authoritative time and detector dimensions"};
    }

    require_type(timestamps, std::string(timestamp_name),
                 {netCDF::NcType::nc_INT});
    require_matrix_shape(timestamps, std::string(timestamp_name), 6);
    const auto timestamp_dimensions = timestamps.getDims();
    const auto &time_dimension = in_phase_dimensions[0];
    if (time_dimension.getName() != "time" ||
        timestamp_dimensions[0].getId() != time_dimension.getId() ||
        timestamp_dimensions[0].getSize() != time_dimension.getSize()) {
        throw DataIOError{
            "netCDF variable 'Data.Toltec.Ts' must share the required "
            "Data.Toltec.Is/Data.Toltec.Qs 'time' dimension"};
    }
    require_exact_text_attribute(
        timestamps, std::string(timestamp_name), "long_name",
        legacy_toltec_ts_long_name);
}

inline std::string read_fixed_width_text(const netCDF::NcVar &variable,
                                         const std::string &name,
                                         std::size_t expected_size) {
    require_type(variable, name, {netCDF::NcType::nc_CHAR});
    const auto dimensions = variable.getDims();
    if (dimensions.size() != 1 ||
        dimensions.front().getSize() != expected_size) {
        throw DataIOError{
            "netCDF variable '" + name + "' must have exact char[" +
            std::to_string(expected_size) + "] shape"};
    }
    std::vector<char> buffer(expected_size);
    variable.getVar(buffer.data());

    auto padding = buffer.end();
    for (auto cursor = buffer.begin(); cursor != buffer.end(); ++cursor) {
        const bool is_padding = *cursor == ' ' || *cursor == '\0';
        if (padding == buffer.end()) {
            if (is_padding) {
                padding = cursor;
            }
        }
        else if (!is_padding) {
            throw DataIOError{
                "netCDF variable '" + name +
                "' has non-padding data after fixed-width text padding"};
        }
    }
    return std::string(buffer.begin(), padding);
}

inline void require_units(
    const netCDF::NcVar &variable, const std::string &name,
    std::initializer_list<std::string_view> permitted) {
    const auto attributes = variable.getAtts();
    const auto units = attributes.find("units");
    if (units == attributes.end() ||
        units->second.getType().getTypeClass() !=
            netCDF::NcType::nc_CHAR) {
        throw DataIOError{"netCDF variable '" + name +
                          "' lacks an authoritative text units attribute"};
    }
    std::string actual;
    units->second.getValues(actual);
    if (std::find(permitted.begin(), permitted.end(), actual) !=
        permitted.end()) {
        return;
    }
    std::ostringstream expected;
    bool first = true;
    for (const auto value : permitted) {
        if (!first) {
            expected << ",";
        }
        expected << value;
        first = false;
    }
    throw DataIOError{"netCDF variable '" + name + "' has units '" +
                      actual + "'; expected " + expected.str()};
}

inline citlali::pipeline::sci_align::TelescopeHeaderSnapshot
read_numeric_telescope_header(const netCDF::NcVar &variable,
                              const std::string &name) {
    using citlali::pipeline::sci_align::TelescopeHeaderNumericType;
    using citlali::pipeline::sci_align::TelescopeHeaderSnapshot;

    const auto element_count =
        require_scalar_or_nonempty_vector(variable, name);
    (void)require_eigen_index_size(element_count, name);
    TelescopeHeaderSnapshot snapshot;
    for (const auto &dimension : variable.getDims()) {
        snapshot.dimensions.push_back(
            {dimension.getName(), dimension.getSize()});
    }

    const auto attributes = variable.getAtts();
    const auto units = attributes.find("units");
    if (units != attributes.end()) {
        if (units->second.getType().getTypeClass() !=
            netCDF::NcType::nc_CHAR) {
            throw DataIOError{"netCDF variable '" + name +
                              "' has a non-text units attribute"};
        }
        std::string value;
        units->second.getValues(value);
        snapshot.units = std::move(value);
    }

    auto read_values = [&](auto value_tag,
                           TelescopeHeaderNumericType type) {
        using Value = decltype(value_tag);
        std::vector<Value> values(element_count);
        variable.getVar(values.data());
        snapshot.type = type;
        snapshot.values = std::move(values);
    };
    switch (variable.getType().getTypeClass()) {
        case netCDF::NcType::nc_BYTE:
            read_values(static_cast<signed char>(0),
                        TelescopeHeaderNumericType::int8);
            break;
        case netCDF::NcType::nc_UBYTE:
            read_values(static_cast<unsigned char>(0),
                        TelescopeHeaderNumericType::uint8);
            break;
        case netCDF::NcType::nc_SHORT:
            read_values(short{}, TelescopeHeaderNumericType::int16);
            break;
        case netCDF::NcType::nc_USHORT:
            read_values(static_cast<unsigned short>(0),
                        TelescopeHeaderNumericType::uint16);
            break;
        case netCDF::NcType::nc_INT:
            read_values(int{}, TelescopeHeaderNumericType::int32);
            break;
        case netCDF::NcType::nc_UINT:
            read_values(static_cast<unsigned int>(0),
                        TelescopeHeaderNumericType::uint32);
            break;
        case netCDF::NcType::nc_INT64:
            read_values(static_cast<long long>(0),
                        TelescopeHeaderNumericType::int64);
            break;
        case netCDF::NcType::nc_UINT64:
            read_values(static_cast<unsigned long long>(0),
                        TelescopeHeaderNumericType::uint64);
            break;
        case netCDF::NcType::nc_FLOAT:
            read_values(float{}, TelescopeHeaderNumericType::float32);
            break;
        case netCDF::NcType::nc_DOUBLE:
            read_values(double{}, TelescopeHeaderNumericType::float64);
            break;
        default:
            throw DataIOError{"netCDF variable '" + name +
                              "' is not an atomic numeric telescope header"};
    }

    try {
        citlali::pipeline::sci_align::validate_telescope_header_snapshot(
            snapshot, name);
    }
    catch (const std::invalid_argument &error) {
        throw DataIOError{error.what()};
    }
    return snapshot;
}

}  // namespace citlali::engine_detail::sci_align_netcdf

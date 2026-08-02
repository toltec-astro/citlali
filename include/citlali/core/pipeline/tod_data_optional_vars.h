#pragma once

// Included by tod_output_data_vars.h inside namespace citlali::pipeline.

inline void add_tod_hwpr_var(netCDF::NcFile &fo, netCDF::NcDim n_pts_dim) {
    netCDF::NcVar hwpr_v = fo.addVar("hwpr", netCDF::ncDouble, n_pts_dim);
    hwpr_v.putAtt("units", "rad");
}

inline void add_tod_hwpr_var_if_requested(netCDF::NcFile &fo,
                                          bool run_polarization,
                                          bool run_hwpr,
                                          netCDF::NcDim n_pts_dim) {
    if (run_polarization && run_hwpr) {
        add_tod_hwpr_var(fo, n_pts_dim);
    }
}

inline netCDF::NcType telescope_header_netcdf_type(
    sci_align::TelescopeHeaderNumericType type) {
    using NumericType = sci_align::TelescopeHeaderNumericType;
    switch (type) {
        case NumericType::int8:
            return netCDF::ncByte;
        case NumericType::uint8:
            return netCDF::ncUbyte;
        case NumericType::int16:
            return netCDF::ncShort;
        case NumericType::uint16:
            return netCDF::ncUshort;
        case NumericType::int32:
            return netCDF::ncInt;
        case NumericType::uint32:
            return netCDF::ncUint;
        case NumericType::int64:
            return netCDF::ncInt64;
        case NumericType::uint64:
            return netCDF::ncUint64;
        case NumericType::float32:
            return netCDF::ncFloat;
        case NumericType::float64:
            return netCDF::ncDouble;
    }
    throw DataIOError{"unsupported native telescope header numeric type"};
}

inline std::vector<netCDF::NcDim> telescope_header_output_dimensions(
    netCDF::NcFile &fo,
    const sci_align::TelescopeHeaderSnapshot &snapshot,
    const std::string &name) {
    std::vector<netCDF::NcDim> result;
    result.reserve(snapshot.dimensions.size());
    for (const auto &dimension : snapshot.dimensions) {
        auto output_dimension = fo.getDim(dimension.name);
        if (output_dimension.isNull()) {
            output_dimension = fo.addDim(dimension.name, dimension.size);
        }
        else if (output_dimension.getSize() != dimension.size) {
            throw DataIOError{
                "required telescope header output '" + name +
                "' conflicts with existing dimension '" +
                dimension.name + "'"};
        }
        result.push_back(output_dimension);
    }
    return result;
}

template <class TelescopeHeaderSnapshots>
void add_telescope_header_vars(netCDF::NcFile &fo,
                               const TelescopeHeaderSnapshots &snapshots) {
    for (const auto &[key, snapshot] : snapshots) {
        try {
            sci_align::validate_telescope_header_snapshot(snapshot, key);
        }
        catch (const std::invalid_argument &error) {
            throw DataIOError{error.what()};
        }
        if (!fo.getVar(key).isNull()) {
            throw DataIOError{
                "required telescope header output variable '" + key +
                "' already exists"};
        }
        const auto dimensions =
            telescope_header_output_dimensions(fo, snapshot, key);
        auto variable = dimensions.empty()
                            ? fo.addVar(key,
                                        telescope_header_netcdf_type(
                                            snapshot.type))
                            : fo.addVar(key,
                                        telescope_header_netcdf_type(
                                            snapshot.type),
                                        dimensions);
        if (snapshot.units.has_value()) {
            variable.putAtt("units", *snapshot.units);
        }
        std::visit(
            [&](const auto &values) {
                variable.putVar(values.data());
            },
            snapshot.values);
    }
}

#pragma once

// Included by tod_output_data_vars.h inside namespace citlali::pipeline.

template <class AptTable, class AptUnits>
void add_tod_apt_table_vars(netCDF::NcFile &fo, const AptTable &apt,
                            const AptUnits &apt_header_units,
                            netCDF::NcDim n_dets_dim) {
    for (const auto &item : apt) {
        netCDF::NcVar apt_v =
            fo.addVar("apt_" + item.first, netCDF::ncDouble, n_dets_dim);
        const auto units_it = apt_header_units.find(item.first);
        const std::string units =
            (units_it == apt_header_units.end()) ? "" : units_it->second;
        apt_v.putAtt("units", units);
    }
}

template <class TelescopeData>
void add_telescope_data_vars(
    netCDF::NcFile &fo, const TelescopeData &tel_data,
    netCDF::NcDim n_pts_dim, netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    for (const auto &item : tel_data) {
        netCDF::NcVar tel_data_v =
            fo.addVar(item.first, netCDF::ncDouble, n_pts_dim);
        tel_data_v.putAtt("units", "rad");
        set_tod_var_chunking(tel_data_v, chunk_mode, chunk_sizes);
    }
}

template <class PointingOffsets, class Logger>
void add_tod_pointing_offset_vars(
    netCDF::NcFile &fo, const PointingOffsets &pointing_offsets_arcsec,
    const Logger &logger, netCDF::NcDim n_pts_dim,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    for (const auto &item : pointing_offsets_arcsec) {
        logger->info("pointing_offsets_arcsec.second {} {}", item.first,
                     item.second);
        netCDF::NcVar offsets_v = fo.addVar(
            "pointing_offset_" + item.first, netCDF::ncDouble, n_pts_dim);
        offsets_v.putAtt("units", "arcsec");
        set_tod_var_chunking(offsets_v, chunk_mode, chunk_sizes);
    }
}

template <class AptTable, class AptUnits, class TelescopeData,
          class PointingOffsets, class Logger>
void add_tod_static_metadata_vars(
    netCDF::NcFile &fo, const AptTable &apt, const AptUnits &apt_header_units,
    const TelescopeData &tel_data, const PointingOffsets &pointing_offsets,
    const Logger &logger, netCDF::NcDim n_dets_dim, netCDF::NcDim n_pts_dim,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    add_tod_apt_table_vars(fo, apt, apt_header_units, n_dets_dim);
    add_telescope_data_vars(fo, tel_data, n_pts_dim, chunk_mode,
                            chunk_sizes);
    add_tod_pointing_offset_vars(
        fo, pointing_offsets, logger, n_pts_dim, chunk_mode, chunk_sizes);
}


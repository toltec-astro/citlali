#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include <netcdf>

#include <citlali/core/utils/netcdf_io.h>

namespace citlali::pipeline {

inline void add_tod_output_type_label(netCDF::NcFile &fo,
                                      const std::string &label) {
    netCDF::NcDim dim = fo.addDim("n_tod_output_type", 1);
    netCDF::NcVar var = fo.addVar("tod_output_type", netCDF::ncString, dim);
    const std::vector<std::size_t> index = {0};
    std::string value = label;
    var.putVar(index, value);
}

inline void add_obsnum_var(netCDF::NcFile &fo, int obsnum) {
    netCDF::NcVar var = fo.addVar("obsnum", netCDF::ncInt);
    var.putAtt("units", "N/A");
    var.putVar(&obsnum);
}

inline void add_source_radec_vars(netCDF::NcFile &fo, double source_ra,
                                  double source_dec) {
    netCDF::NcVar source_ra_v = fo.addVar("SourceRa", netCDF::ncDouble);
    source_ra_v.putAtt("units", "rad");
    source_ra_v.putVar(&source_ra);

    netCDF::NcVar source_dec_v = fo.addVar("SourceDec", netCDF::ncDouble);
    source_dec_v.putAtt("units", "rad");
    source_dec_v.putVar(&source_dec);
}

inline void add_observation_identity_vars(netCDF::NcFile &fo, int obsnum,
                                          double source_ra,
                                          double source_dec) {
    add_obsnum_var(fo, obsnum);
    add_source_radec_vars(fo, source_ra, source_dec);
}

inline void add_pipeline_identity_vars(
    netCDF::NcFile &fo, const std::string &citlali_version,
    const std::string &kids_version, const std::string &tula_version,
    const std::string &project_id, const std::string &reduction_goal,
    const std::string &obs_goal, const std::string &tod_type) {
    add_netcdf_var<std::string>(fo, "INSTRUME", "TolTEC");
    add_netcdf_var<std::string>(fo, "TELESCOP", "LMT");
    add_netcdf_var<std::string>(fo, "PIPELINE", "CITLALI");
    add_netcdf_var<std::string>(fo, "VERSION", citlali_version);
    add_netcdf_var<std::string>(fo, "KIDS", kids_version);
    add_netcdf_var<std::string>(fo, "TULA", tula_version);
    add_netcdf_var<std::string>(fo, "PROJID", project_id);
    add_netcdf_var<std::string>(fo, "GOAL", reduction_goal);
    add_netcdf_var<std::string>(fo, "OBSGOAL", obs_goal);
    add_netcdf_var<std::string>(fo, "TYPE", tod_type);
}

inline void add_tod_map_geometry_vars(
    netCDF::NcFile &fo, const std::string &map_grouping,
    const std::string &map_method, double exposure_time,
    const std::string &radec_system, double tangent_ra, double tangent_dec,
    double mean_el_deg, double mean_az_deg, double mean_pa_deg) {
    add_netcdf_var<std::string>(fo, "GROUPING", map_grouping);
    add_netcdf_var<std::string>(fo, "METHOD", map_method);
    add_netcdf_var(fo, "EXPTIME", exposure_time);
    add_netcdf_var<std::string>(fo, "RADESYS", radec_system);
    add_netcdf_var(fo, "TAN_RA", tangent_ra);
    add_netcdf_var(fo, "TAN_DEC", tangent_dec);
    add_netcdf_var(fo, "MEAN_EL", mean_el_deg);
    add_netcdf_var(fo, "MEAN_AZ", mean_az_deg);
    add_netcdf_var(fo, "MEAN_PA", mean_pa_deg);
}

template <class Arrays, class FwhmMap, class PositionAngleMap,
          class ArrayNameMap>
void add_array_beam_geometry_vars(netCDF::NcFile &fo, const Arrays &arrays,
                                  FwhmMap &array_fwhms,
                                  PositionAngleMap &array_pas,
                                  ArrayNameMap &array_name_map,
                                  double rad_to_deg,
                                  double pa_quadrature_offset_rad) {
    for (const auto &arr: arrays) {
        const auto &fwhm = array_fwhms[arr];
        const auto &name = array_name_map[arr];
        if (std::get<0>(fwhm) >= std::get<1>(fwhm)) {
            add_netcdf_var(fo, "BMAJ_" + name, std::get<0>(fwhm));
            add_netcdf_var(fo, "BMIN_" + name, std::get<1>(fwhm));
            add_netcdf_var(fo, "BPA_" + name, array_pas[arr]*rad_to_deg);
        }
        else {
            add_netcdf_var(fo, "BMAJ_" + name, std::get<1>(fwhm));
            add_netcdf_var(fo, "BMIN_" + name, std::get<0>(fwhm));
            add_netcdf_var(fo, "BPA_" + name,
                           (array_pas[arr] + pa_quadrature_offset_rad)*
                               rad_to_deg);
        }
    }
}

inline void add_tod_scan_index_placeholders(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &raw_scans_dims,
    const std::vector<netCDF::NcDim> &scans_dims,
    netCDF::NcDim n_scans_dim, std::size_t n_output_scans,
    std::size_t n_raw_scan_indices, bool tod_output_outer, int fill_value) {
    netCDF::NcVar raw_scan_indices_v =
        fo.addVar("raw_scan_indices", netCDF::ncInt, raw_scans_dims);
    raw_scan_indices_v.putAtt("units", "N/A");
    raw_scan_indices_v.putAtt(
        "comment",
        tod_output_outer
            ? "indices in output timebase: inner_start, inner_end, outer_start, outer_end"
            : "indices in output timebase; outer=inner (output stores inner scans only)");
    std::vector<int> raw_scan_init(n_output_scans * n_raw_scan_indices,
                                   fill_value);
    raw_scan_indices_v.putVar(raw_scan_init.data());

    netCDF::NcVar scan_indices_v =
        fo.addVar("scan_indices", netCDF::ncInt, scans_dims);
    scan_indices_v.putAtt("units", "N/A");
    std::vector<int> scan_init(n_output_scans * 2, fill_value);
    scan_indices_v.putVar(scan_init.data());

    netCDF::NcVar output_scan_index_v =
        fo.addVar("output_scan_index", netCDF::ncInt, n_scans_dim);
    output_scan_index_v.putAtt("units", "N/A");
    output_scan_index_v.putAtt(
        "comment", "1-based original scan index from the full observation");
    std::vector<int> output_scan_init(n_output_scans, fill_value);
    output_scan_index_v.putVar(output_scan_init.data());
}

template <class AddInt, class AddDouble>
void add_tod_filter_edge_guard_scan_vars(const AddInt &add_int,
                                         const AddDouble &add_double) {
    add_int("tod_filter_edge_guard_pre_samples",
            "samples flagged at the start of this output scan by the TOD filter edge guard");
    add_int("tod_filter_edge_guard_post_samples",
            "samples flagged at the end of this output scan by the TOD filter edge guard");
    add_int("tod_filter_edge_guard_flagged_samples",
            "detector-samples flagged by the TOD filter edge guard");
    add_double("tod_filter_edge_guard_flagged_frac", "N/A",
               "fraction of time samples guarded at this output scan edge");
}

}  // namespace citlali::pipeline

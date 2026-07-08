#pragma once

// Included by tod_output_reduction_metadata.h inside namespace citlali::pipeline.

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

template <class Arrays, class FwhmMap, class PositionAngleMap,
          class ArrayNameMap>
void add_tod_identity_geometry_vars(
    netCDF::NcFile &fo, const std::string &citlali_version,
    const std::string &kids_version, const std::string &tula_version,
    const std::string &project_id, citlali::config::ReductionType reduction_type,
    const std::string &obs_goal, citlali::config::TodType tod_type,
    bool run_hwpr, citlali::config::MapGrouping map_grouping,
    citlali::config::MapMethod map_method,
    double exposure_time, const std::string &pixel_axes, double tangent_ra,
    double tangent_dec, double mean_el_deg, double mean_az_deg,
    double mean_pa_deg, const Arrays &arrays, FwhmMap &array_fwhms,
    PositionAngleMap &array_pas, ArrayNameMap &array_name_map,
    double rad_to_deg, double pa_quadrature_offset_rad,
    const std::string &signal_unit) {
    const std::string reduction_type_name{
        citlali::config::to_string(reduction_type)};
    const std::string tod_type_name{citlali::config::to_string(tod_type)};
    const std::string map_grouping_name{
        citlali::config::to_string(map_grouping)};
    const std::string map_method_name{citlali::config::to_string(map_method)};

    add_pipeline_identity_vars(
        fo, citlali_version, kids_version, tula_version, project_id,
        reduction_type_name, obs_goal, tod_type_name);
    add_netcdf_var(fo, "HWPR", run_hwpr);
    add_tod_map_geometry_vars(
        fo, map_grouping_name, map_method_name, exposure_time, pixel_axes,
        tangent_ra, tangent_dec, mean_el_deg, mean_az_deg, mean_pa_deg);
    add_array_beam_geometry_vars(
        fo, arrays, array_fwhms, array_pas, array_name_map, rad_to_deg,
        pa_quadrature_offset_rad);
    add_tod_signal_unit_var(fo, signal_unit);
}

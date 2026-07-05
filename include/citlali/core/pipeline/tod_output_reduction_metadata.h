#pragma once

// TOD output NetCDF metadata implementation detail.
// Include this only from output_netcdf_metadata.h inside citlali::pipeline.

inline void add_unit_conversion_basis_vars(netCDF::NcFile &fo) {
    add_netcdf_var<std::string>(fo, "UNITCONV.UK_CONVENTION",
                                "Rayleigh-Jeans brightness temperature");
    add_netcdf_var<std::string>(
        fo, "UNITCONV.UK_BASIS",
        "monochromatic array center frequency; mJy/beam uses Gaussian beam solid angle to Jy/sr");
}

inline void add_unit_conversion_array_vars(
    netCDF::NcFile &fo, const std::string &array_name,
    const std::string &signal_unit, double mjy_sr_to_mjy_beam,
    double mjy_beam_to_uk, double mjy_beam_to_jy_pixel) {
    if (signal_unit == "mJy/beam") {
        add_netcdf_var(fo, "to_mJy_beam_" + array_name, 1);
        add_netcdf_var(fo, "to_MJy_sr_" + array_name,
                       1/mjy_sr_to_mjy_beam);
        add_netcdf_var(fo, "to_uK_" + array_name, mjy_beam_to_uk);
        add_netcdf_var(fo, "to_Jy_pixel_" + array_name,
                       mjy_beam_to_jy_pixel);
    }
    else if (signal_unit == "MJy/sr") {
        add_netcdf_var(fo, "to_mJy_beam_" + array_name,
                       mjy_sr_to_mjy_beam);
        add_netcdf_var(fo, "to_MJy_sr_" + array_name, 1);
        add_netcdf_var(fo, "to_uK_" + array_name,
                       mjy_sr_to_mjy_beam*mjy_beam_to_uk);
        add_netcdf_var(fo, "to_Jy_pixel_" + array_name,
                       mjy_sr_to_mjy_beam*mjy_beam_to_jy_pixel);
    }
    else if (signal_unit == "uK") {
        add_netcdf_var(fo, "to_mJy_beam_" + array_name, 1/mjy_beam_to_uk);
        add_netcdf_var(fo, "to_MJy_sr_" + array_name,
                       1/mjy_beam_to_uk/mjy_sr_to_mjy_beam);
        add_netcdf_var(fo, "to_uK_" + array_name, 1);
        add_netcdf_var(fo, "to_Jy_pixel_" + array_name,
                       (1/mjy_beam_to_uk)*mjy_beam_to_jy_pixel);
    }
    else if (signal_unit == "Jy/pixel") {
        add_netcdf_var(fo, "to_mJy_beam_" + array_name,
                       1/mjy_beam_to_jy_pixel);
        add_netcdf_var(fo, "to_MJy_sr_" + array_name,
                       (1/mjy_beam_to_jy_pixel)/mjy_sr_to_mjy_beam);
        add_netcdf_var(fo, "to_uK_" + array_name,
                       mjy_beam_to_uk/mjy_beam_to_jy_pixel);
        add_netcdf_var(fo, "to_Jy_pixel_" + array_name, 1);
    }
}

template <class Calib, class ToltecIo>
void add_tod_unit_conversion_vars(
    netCDF::NcFile &fo, Calib &calib, ToltecIo &toltec_io,
    const std::string &signal_unit, double pixel_size_rad,
    double mjy_sr_to_mjy_arcsec, double fwhm_to_std, double arcsec_to_rad,
    double pi_value) {
    add_unit_conversion_basis_vars(fo);
    for (const auto &array_id: calib.arrays) {
        const auto array_name = toltec_io.array_name_map[array_id];
        const auto fwhm =
            mean_beam_fwhm_arcsec(calib.array_fwhms[array_id]);
        const auto mjy_beam_to_uk = engine_utils::mJy_beam_to_uK(
            1, toltec_io.array_freq_map[array_id], fwhm);
        const auto beam_area_sr =
            gaussian_beam_area_sr(
                fwhm, fwhm_to_std, arcsec_to_rad, pi_value);
        const auto mjy_beam_to_jy_pixel =
            mjy_beam_to_jy_pixel_factor(beam_area_sr, pixel_size_rad);

        add_unit_conversion_array_vars(
            fo, array_name, signal_unit,
            calib.array_beam_areas[array_id] * mjy_sr_to_mjy_arcsec,
            mjy_beam_to_uk, mjy_beam_to_jy_pixel);
    }
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

template <class Arrays, class FwhmMap, class PositionAngleMap,
          class ArrayNameMap>
void add_tod_identity_geometry_vars(
    netCDF::NcFile &fo, const std::string &citlali_version,
    const std::string &kids_version, const std::string &tula_version,
    const std::string &project_id, const std::string &reduction_goal,
    const std::string &obs_goal, const std::string &tod_type, bool run_hwpr,
    const std::string &map_grouping, const std::string &map_method,
    double exposure_time, const std::string &pixel_axes, double tangent_ra,
    double tangent_dec, double mean_el_deg, double mean_az_deg,
    double mean_pa_deg, const Arrays &arrays, FwhmMap &array_fwhms,
    PositionAngleMap &array_pas, ArrayNameMap &array_name_map,
    double rad_to_deg, double pa_quadrature_offset_rad,
    const std::string &signal_unit) {
    add_pipeline_identity_vars(
        fo, citlali_version, kids_version, tula_version, project_id,
        reduction_goal, obs_goal, tod_type);
    add_netcdf_var(fo, "HWPR", run_hwpr);
    add_tod_map_geometry_vars(
        fo, map_grouping, map_method, exposure_time, pixel_axes, tangent_ra,
        tangent_dec, mean_el_deg, mean_az_deg, mean_pa_deg);
    add_array_beam_geometry_vars(
        fo, arrays, array_fwhms, array_pas, array_name_map, rad_to_deg,
        pa_quadrature_offset_rad);
    add_tod_signal_unit_var(fo, signal_unit);
}

template <class Arrays, class ShapeParams, class ArrayNameMap>
void add_jinc_shape_config_vars(netCDF::NcFile &fo, const Arrays &arrays,
                                ShapeParams &shape_params,
                                ArrayNameMap &array_name_map,
                                double r_max) {
    add_netcdf_var(fo, "JINC_R", r_max);
    for (const auto &arr: arrays) {
        const auto &name = array_name_map[arr];
        add_netcdf_var(fo, "JINC_A_" + name, shape_params[arr][0]);
        add_netcdf_var(fo, "JINC_B_" + name, shape_params[arr][1]);
        add_netcdf_var(fo, "JINC_C_" + name, shape_params[arr][2]);
    }
}

template <class Arrays, class ShapeParams, class ArrayNameMap>
void add_jinc_shape_config_vars_if_needed(
    netCDF::NcFile &fo, const std::string &map_method, const Arrays &arrays,
    ShapeParams &shape_params, ArrayNameMap &array_name_map, double r_max) {
    if (map_method == "jinc") {
        add_jinc_shape_config_vars(
            fo, arrays, shape_params, array_name_map, r_max);
    }
}

template <class TauByFrequency, class Calib, class ArrayNameMap>
void add_mean_tau_vars(netCDF::NcFile &fo, const TauByFrequency &tau_freq,
                       const Calib &calib, ArrayNameMap &array_name_map) {
    decltype(calib.arrays.size()) i = 0;
    for (auto const& [key, val] : tau_freq) {
        add_netcdf_var(
            fo, "MEAN_TAU_" + array_name_map[calib.arrays(i)], val[0]);
        i++;
    }
}

template <class Calib, class ArrayNameMap>
void add_zero_mean_tau_vars(netCDF::NcFile &fo, const Calib &calib,
                            ArrayNameMap &array_name_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        add_netcdf_var(
            fo, "MEAN_TAU_" + array_name_map[calib.arrays(i)], 0.);
    }
}

template <class Rtcproc, class TelescopeData, class Calib, class ArrayNameMap>
void add_tod_mean_tau_vars(netCDF::NcFile &fo, Rtcproc &rtcproc,
                           TelescopeData &tel_data, double tau_225_ghz,
                           const Calib &calib, ArrayNameMap &array_name_map) {
    if (rtcproc.run_extinction) {
        Eigen::VectorXd tau_el(1);
        tau_el << tel_data["TelElAct"].mean();
        auto tau_freq = rtcproc.calibration.calc_tau(tau_el, tau_225_ghz);
        add_mean_tau_vars(fo, tau_freq, calib, array_name_map);
    }
    else {
        add_zero_mean_tau_vars(fo, calib, array_name_map);
    }
}

template <class Arrays, class ArrayNameMap, class FluxMap>
void add_beammap_source_flux_vars(netCDF::NcFile &fo, const Arrays &arrays,
                                  ArrayNameMap &array_name_map,
                                  FluxMap &flux_mjy_beam,
                                  FluxMap &flux_mjy_sr) {
    for (const auto &arr: arrays) {
        const auto &name = array_name_map[arr];
        add_netcdf_var(fo, "HEADER.SOURCE.FLUX_MJYPERBEAM_" + name,
                       flux_mjy_beam[name]);
        add_netcdf_var(fo, "HEADER.SOURCE.FLUX_MJYPERSR_" + name,
                       flux_mjy_sr[name]);
    }
}

inline void add_beammap_tuning_vars(
    netCDF::NcFile &fo, double iter_tolerance,
    double convergence_radius_arcsec, int iter_max, bool phase_split_enabled,
    int locator_iter, int measurement_start_iter, bool is_derotated) {
    add_netcdf_var(fo, "BEAMMAP.ITER_TOLERANCE", iter_tolerance);
    add_netcdf_var(fo, "BEAMMAP.CONVERGENCE_RADIUS_ARCSEC",
                   convergence_radius_arcsec);
    add_netcdf_var(fo, "BEAMMAP.ITER_MAX", iter_max);
    add_netcdf_var(fo, "BEAMMAP.PHASE_SPLIT_ENABLED",
                   phase_split_enabled);
    add_netcdf_var(fo, "BEAMMAP.LOCATOR_ITER", locator_iter);
    add_netcdf_var(fo, "BEAMMAP.MEASUREMENT_START_ITER",
                   measurement_start_iter);
    add_netcdf_var(fo, "BEAMMAP.IS_DEROTATED", is_derotated);
}

inline void add_beammap_reference_vars(netCDF::NcFile &fo, int det_index,
                                       double ref_x_t, double ref_y_t) {
    add_netcdf_var(fo, "BEAMMAP.REF_DET_INDEX", det_index);
    add_netcdf_var(fo, "BEAMMAP.REF_X_T", ref_x_t);
    add_netcdf_var(fo, "BEAMMAP.REF_Y_T", ref_y_t);
}

template <class Calib, class ArrayNameMap, class FluxMap, class ReferenceDet>
void add_beammap_tod_header_vars(
    netCDF::NcFile &fo, Calib &calib, ArrayNameMap &array_name_map,
    FluxMap &flux_mjy_beam, FluxMap &flux_mjy_sr, double iter_tolerance,
    double convergence_radius_arcsec, int iter_max,
    bool phase_split_enabled, int locator_iter, int measurement_start_iter,
    bool is_derotated, bool subtract_reference,
    const ReferenceDet &reference_det) {
    add_beammap_source_flux_vars(
        fo, calib.arrays, array_name_map, flux_mjy_beam, flux_mjy_sr);
    add_beammap_tuning_vars(
        fo, iter_tolerance, convergence_radius_arcsec, iter_max,
        phase_split_enabled, locator_iter, measurement_start_iter,
        is_derotated);

    int ref_det_index = -99;
    double ref_x_t = -99.0;
    double ref_y_t = -99.0;
    if (subtract_reference) {
        const auto reference_values =
            beammap_reference_header_values(calib, reference_det);
        ref_det_index = reference_values.det_index;
        ref_x_t = reference_values.x_t;
        ref_y_t = reference_values.y_t;
    }
    add_beammap_reference_vars(fo, ref_det_index, ref_x_t, ref_y_t);
}

inline void add_oof_telescope_vars(netCDF::NcFile &fo, double m2x_microns,
                                   double m2y_microns,
                                   double m2z_microns) {
    add_netcdf_var(fo, "OOF_T", 3.0);
    add_netcdf_var(fo, "OOF_M2X", m2x_microns);
    add_netcdf_var(fo, "OOF_M2Y", m2y_microns);
    add_netcdf_var(fo, "OOF_M2Z", m2z_microns);
    add_netcdf_var(fo, "OOF_RO", 25.);
    add_netcdf_var(fo, "OOF_RI", 1.65);
}

template <class MapBuffer, class Calib, class ArrayNameMap,
          class WavelengthMap>
void add_oof_array_vars(netCDF::NcFile &fo, const MapBuffer &mb,
                        const std::string &reduction_type,
                        bool run_mapmaking, const Calib &calib,
                        ArrayNameMap &array_name_map,
                        WavelengthMap &array_wavelength_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        double rms = 0.0;
        if (reduction_type != "beammap" && run_mapmaking) {
            rms = std::pow(mb->median_err(i), 0.5);
        }
        const auto array = calib.arrays(i);
        const auto &name = array_name_map[array];
        add_netcdf_var(fo, "OOF_RMS_" + name, rms);
        add_netcdf_var(fo, "OOF_W_" + name,
                       array_wavelength_map[array]/1000.);
        add_netcdf_var(fo, "OOF_ID_" + name,
                       static_cast<int>(array_wavelength_map[array]*1000));
    }
}

template <class TelescopeHeader, class MapBuffer, class Calib,
          class ArrayNameMap, class WavelengthMap>
void add_oof_header_vars_if_observed(
    netCDF::NcFile &fo, bool simulated_observation,
    TelescopeHeader &tel_header, const MapBuffer &mb,
    const std::string &reduction_type, bool run_mapmaking,
    const Calib &calib, ArrayNameMap &array_name_map,
    WavelengthMap &array_wavelength_map) {
    if (simulated_observation) {
        return;
    }

    add_oof_telescope_vars(
        fo, tel_header["Header.M2.XReq"](0) / 1000. * 1e6,
        tel_header["Header.M2.YReq"](0) / 1000. * 1e6,
        tel_header["Header.M2.ZReq"](0) / 1000. * 1e6);
    add_oof_array_vars(
        fo, mb, reduction_type, run_mapmaking, calib, array_name_map,
        array_wavelength_map);
}

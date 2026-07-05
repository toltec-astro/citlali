#pragma once

// Included by tod_output_reduction_metadata.h inside namespace citlali::pipeline.

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


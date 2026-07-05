#pragma once

#include <citlali/core/pipeline/phdu_extinction.h>
#include <citlali/core/pipeline/phdu_observation_metadata.h>
#include <citlali/core/pipeline/phdu_oof.h>
#include <citlali/core/pipeline/phdu_reduction_config.h>

namespace citlali::engine_detail {

template <class FitsEntry, class MapBuffer, class Calib, class ToltecIo,
          class ArrayId, class Logger>
void add_phdu_unit_conversion_section(
    FitsEntry &fits_entry, const MapBuffer &mb, Calib &calib,
    ToltecIo &toltec_io, const ArrayId &array_id,
    const std::string &array_name, bool run_calibrate,
    double fwhm_to_std, double arcsec_to_rad, double pi_value,
    double mjy_sr_to_mjy_asec, const Logger &logger) {
    logger->debug("adding unit conversions");

    const auto unit_conversion =
        citlali::pipeline::phdu_unit_conversion_factors(
            calib.array_fwhms[array_id], mb->pixel_size_rad, fwhm_to_std,
            arcsec_to_rad, pi_value);
    auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(
        1, toltec_io.array_freq_map[array_id],
        unit_conversion.mean_fwhm_arcsec);

    citlali::pipeline::add_phdu_unit_conversion_config(
        fits_entry, array_name, logger, run_calibrate, mb->sig_unit,
        calib.array_beam_areas[array_id] * mjy_sr_to_mjy_asec,
        mJy_beam_to_uK, unit_conversion.mjy_beam_to_jy_pixel);
}

template <class FitsEntry, class MapBuffer, class Telescope, class Calib,
          class Logger>
void add_phdu_identity_geometry_section(
    FitsEntry &fits_entry, const MapBuffer &mb, const Telescope &telescope,
    const Calib &calib, const std::string &array_name,
    const std::string &citlali_version, const std::string &kids_version,
    const std::string &tula_version, const std::string &reduction_type,
    const std::string &tod_type, const std::string &map_grouping,
    const std::string &map_method, double rad_to_deg,
    const Logger &logger) {
    citlali::pipeline::add_phdu_pipeline_identity_keys(
        fits_entry, telescope.source_name, calib.run_hwpr, array_name,
        citlali_version, kids_version, tula_version, telescope.project_id,
        reduction_type, telescope.obs_goal, tod_type, map_grouping,
        map_method);

    const double source_ra =
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.Source.Ra", 0.0, logger);
    const double source_dec =
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.Source.Dec", 0.0, logger);
    citlali::pipeline::add_phdu_map_geometry_keys(
        fits_entry, array_name, logger, mb->exposure_time,
        telescope.pixel_axes, source_ra, source_dec,
        rad_to_deg * citlali::pipeline::telescope_data_mean(
                         telescope.tel_data, "TelElAct", 0.0, logger),
        rad_to_deg * citlali::pipeline::telescope_data_mean(
                         telescope.tel_data, "TelAzAct", 0.0, logger),
        rad_to_deg * citlali::pipeline::telescope_data_mean(
                         telescope.tel_data, "ActParAng", 0.0, logger));
}

template <class FitsEntry, class MapBuffer, class RtcProc, class Telescope,
          class Calib, class ToltecIo, class ArrayId, class Logger>
void add_phdu_extinction_apt_oof_section(
    FitsEntry &fits_entry, const MapBuffer &mb, RtcProc &rtcproc,
    const Telescope &telescope, const Calib &calib, ToltecIo &toltec_io,
    Eigen::Index map_index, const ArrayId &array_id,
    const std::string &array_name, const std::string &reduction_type,
    const Logger &logger) {
    logger->debug("adding extinction");
    const double mean_tau = citlali::pipeline::phdu_mean_tau(
        rtcproc, telescope, calib, map_index, logger);
    citlali::pipeline::add_phdu_double_key(
        fits_entry, array_name, logger, "MEAN_TAU", mean_tau,
        "mean tau (" + array_name + ")");

    citlali::pipeline::add_phdu_apt_key_if_single_observation(
        fits_entry, mb->obsnums, calib.apt_filepath, logger);

    const double rms = citlali::pipeline::phdu_oof_rms(
        mb, map_index, reduction_type, array_name, fits_entry.filepath,
        logger);

    citlali::pipeline::add_phdu_oof_keys_if_observed(
        fits_entry, array_name, logger, telescope.sim_obs, rms,
        mb->sig_unit, toltec_io.array_wavelength_map[array_id] / 1000.,
        static_cast<int>(toltec_io.array_wavelength_map[array_id] * 1000),
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.M2.XReq", 0.0, logger) /
            1000. * 1e6,
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.M2.YReq", 0.0, logger) /
            1000. * 1e6,
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.M2.ZReq", 0.0, logger) /
            1000. * 1e6);
}

}  // namespace citlali::engine_detail

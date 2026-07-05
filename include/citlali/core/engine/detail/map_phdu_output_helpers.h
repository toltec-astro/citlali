#pragma once

#include <citlali/core/pipeline/phdu_observation_metadata.h>
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

}  // namespace citlali::engine_detail

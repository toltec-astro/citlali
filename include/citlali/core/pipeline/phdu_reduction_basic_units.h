#pragma once

// Included by phdu_reduction_config.h inside namespace citlali::pipeline.

template <class Arrays, class Index, class ArrayId>
double phdu_fruit_loop_flux_limit(
    const citlali::config::TimestreamFruitLoopsConfig &config,
    const Arrays &arrays, Index i, const ArrayId &array_id) {
    double flux_limit = 0.0;
    if (config.enabled) {
        if (config.array_flux_limit.size() == arrays.size()) {
            flux_limit = config.array_flux_limit[i];
        }
        else if (array_id < config.array_flux_limit.size()) {
            flux_limit = config.array_flux_limit[array_id];
        }
    }
    return flux_limit;
}

template <class FitsEntry, class Logger>
void add_phdu_unit_conversion_config(FitsEntry &fits_entry,
                                     const std::string &array_name,
                                     const Logger &logger,
                                     bool run_calibrate,
                                     const std::string &signal_unit,
                                     double mjy_sr_to_mjy_beam,
                                     double mjy_beam_to_uk,
                                     double mjy_beam_to_jy_pixel) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("UKCONV", "RJ",
               "uK convention: Rayleigh-Jeans brightness temperature");
    hdu.addKey("UKBASIS", "Jy/sr",
               "uK basis: monochromatic intensity per steradian");

    if (!run_calibrate) {
        hdu.addKey("to_mJy/beam", "N/A", "Conversion to mJy/beam");
        hdu.addKey("to_MJy/sr", "N/A", "Conversion to MJy/sr");
        hdu.addKey("to_uK", "N/A", "Conversion to uK");
        hdu.addKey("to_Jy/pixel", "N/A", "Conversion to Jy/pixel");
        return;
    }

    if (signal_unit == "mJy/beam") {
        hdu.addKey("to_mJy/beam", 1, "Conversion to mJy/beam");
        add_double_key("to_MJy/sr", 1/mjy_sr_to_mjy_beam,
                       "Conversion to MJy/sr");
        add_double_key("to_uK", mjy_beam_to_uk,
                       "Conversion to Rayleigh-Jeans uK");
        add_double_key("to_Jy/pixel", mjy_beam_to_jy_pixel,
                       "Conversion to Jy/pixel");
    }
    else if (signal_unit == "MJy/sr") {
        add_double_key("to_mJy/beam", mjy_sr_to_mjy_beam,
                       "Conversion to mJy/beam");
        hdu.addKey("to_MJy/sr", 1, "Conversion to MJy/sr");
        add_double_key("to_uK", mjy_sr_to_mjy_beam*mjy_beam_to_uk,
                       "Conversion to Rayleigh-Jeans uK");
        add_double_key("to_Jy/pixel",
                       mjy_sr_to_mjy_beam*mjy_beam_to_jy_pixel,
                       "Conversion to Jy/pixel");
    }
    else if (signal_unit == "uK") {
        add_double_key("to_mJy/beam", 1/mjy_beam_to_uk,
                       "Conversion to mJy/beam");
        add_double_key("to_MJy/sr", 1/mjy_beam_to_uk/mjy_sr_to_mjy_beam,
                       "Conversion to MJy/sr");
        hdu.addKey("to_uK", 1, "Conversion to Rayleigh-Jeans uK");
        add_double_key("to_Jy/pixel",
                       (1/mjy_beam_to_uk)*mjy_beam_to_jy_pixel,
                       "Conversion to Jy/pixel");
    }
    else if (signal_unit == "Jy/pixel") {
        add_double_key("to_mJy/beam", 1/mjy_beam_to_jy_pixel,
                       "Conversion to mJy/beam");
        add_double_key("to_MJy/sr",
                       (1/mjy_beam_to_jy_pixel)/mjy_sr_to_mjy_beam,
                       "Conversion to MJy/sr");
        add_double_key("to_uK", mjy_beam_to_uk/mjy_beam_to_jy_pixel,
                       "Conversion to Rayleigh-Jeans uK");
        hdu.addKey("to_Jy/pixel", 1, "Conversion to Jy/pixel");
    }
}

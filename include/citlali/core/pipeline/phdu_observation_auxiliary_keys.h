#pragma once

// Included by phdu_observation_metadata.h inside namespace citlali::pipeline.

template <class FitsEntry>
void add_phdu_auxiliary_scalar_keys(FitsEntry &fits_entry,
                                    const std::string &signal_unit,
                                    double sample_rate_hz,
                                    int fruit_loop_iter) {
    auto &hdu = fits_entry.pfits->pHDU();
    hdu.addKey("BUNIT", signal_unit, "bunit");
    hdu.addKey("SAMPRATE", sample_rate_hz, "sample rate (Hz)");
    hdu.addKey("FRUITLOOPS_ITER", fruit_loop_iter,
               "Current fruit loops iteration");
}

template <class FitsEntry>
void add_phdu_apt_key(FitsEntry &fits_entry, const std::string &apt_name) {
    fits_entry.pfits->pHDU().addKey("APT", apt_name, "APT table used");
}

template <class FitsEntry, class Obsnums, class Logger>
void add_phdu_apt_key_if_single_observation(
    FitsEntry &fits_entry, const Obsnums &obsnums,
    const std::string &apt_filepath, const Logger &logger) {
    if (!phdu_has_single_observation(obsnums)) {
        return;
    }
    const auto apt_name = apt_table_header_name(apt_filepath, logger);
    add_phdu_apt_key(fits_entry, apt_name);
}

template <class FitsEntry, class ShapeValues, class Logger>
void add_phdu_jinc_shape_keys(FitsEntry &fits_entry,
                              const std::string &array_name,
                              const Logger &logger, double r_max,
                              const ShapeValues &shape_values) {
    add_phdu_double_key(fits_entry, array_name, logger, "JINC_R", r_max,
                        "Jinc filter R_max");
    add_phdu_double_key(fits_entry, array_name, logger, "JINC_A",
                        shape_values[0], "Jinc filter param a");
    add_phdu_double_key(fits_entry, array_name, logger, "JINC_B",
                        shape_values[1], "Jinc filter param b");
    add_phdu_double_key(fits_entry, array_name, logger, "JINC_C",
                        shape_values[2], "Jinc filter param c");
}

template <class FitsEntry, class ShapeMap, class ArrayId, class Logger>
void add_phdu_jinc_shape_keys_if_needed(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, const std::string &map_method,
    double r_max, ShapeMap &shape_params, const ArrayId &array_id) {
    if (map_method != "jinc") {
        return;
    }
    logger->debug("adding jinc params");
    add_phdu_jinc_shape_keys(
        fits_entry, array_name, logger, r_max, shape_params[array_id]);
}


#pragma once

// Included by phdu_observation_metadata.h inside namespace citlali::pipeline.

template <class FitsEntry, class HeaderValues, class Logger>
void add_phdu_telescope_header_keys(FitsEntry &fits_entry,
                                    const std::string &array_name,
                                    const Logger &logger,
                                    const HeaderValues &tel_header) {
    for (auto const& [key, val] : tel_header) {
        if (val.size() < 1 || !std::isfinite(val(0))) {
            logger->warn("skipping tel_header '{}' due to empty/non-finite value",
                         key);
            continue;
        }
        logger->debug("adding {}: {}", key, val);
        add_phdu_double_key(fits_entry, array_name, logger, key, val(0), key);
    }
}

template <class FitsEntry, class Obsnums, class HeaderValues, class Logger>
void add_phdu_telescope_header_keys_if_single_observation(
    FitsEntry &fits_entry, const Obsnums &obsnums,
    const std::string &array_name, const Logger &logger,
    const HeaderValues &tel_header) {
    if (!phdu_has_single_observation(obsnums)) {
        return;
    }
    logger->debug("adding tel params");
    add_phdu_telescope_header_keys(fits_entry, array_name, logger,
                                   tel_header);
}


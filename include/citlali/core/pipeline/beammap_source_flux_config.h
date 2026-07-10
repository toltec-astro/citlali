#pragma once

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <Eigen/Core>

#include <cmath>
#include <tuple>

namespace citlali::pipeline {

template <class Config, class SourceConfig, class Diagnostics>
void read_beammap_source_identity_config(Config &config,
                                         SourceConfig &source_config,
                                         Diagnostics &diagnostics) {
    read_config_value(
        config, source_config.name, diagnostics,
        std::tuple{"beammap_source", "name"});
    read_config_value(
        config, source_config.ra_deg, diagnostics,
        std::tuple{"beammap_source", "ra_deg"});
    read_config_value(
        config, source_config.dec_deg, diagnostics,
        std::tuple{"beammap_source", "dec_deg"});
}

template <class Config, class FluxMap, class SourceConfig>
void read_beammap_source_fluxes(Config &config, FluxMap &fluxes_mjy_beam,
                                SourceConfig &source_config) {
    const Eigen::Index n_fluxes =
        config.get_node(std::tuple{"beammap_source", "fluxes"}).size();

    for (Eigen::Index i = 0; i < n_fluxes; ++i) {
        const auto array =
            config.get_str(std::tuple{"beammap_source", "fluxes", i,
                                      "array_name"});
        const auto flux = config.template get_typed<double>(
            std::tuple{"beammap_source", "fluxes", i, "value_mJy"});
        const auto uncertainty_mjy = config.template get_typed<double>(
            std::tuple{"beammap_source", "fluxes", i, "uncertainty_mJy"});

        fluxes_mjy_beam[array] = flux;
        source_config.fluxes.push_back(
            citlali::config::BeammapSourceFluxConfig{
                array, flux, uncertainty_mjy});
    }
}

template <class FluxMap, class ArrayNameMap, class Logger>
bool validate_beammap_source_fluxes(const FluxMap &fluxes_mjy_beam,
                                    const ArrayNameMap &array_name_map,
                                    const Logger &logger) {
    bool valid = true;
    for (const auto &entry : array_name_map) {
        const auto &array_name = entry.second;
        const auto flux_it = fluxes_mjy_beam.find(array_name);
        if (flux_it == fluxes_mjy_beam.end()) {
            logger->error(
                "beammap reductions require a positive source flux for {}; no beammap_source.fluxes entry was found",
                array_name);
            valid = false;
            continue;
        }
        const double flux = flux_it->second;
        if (!std::isfinite(flux) || flux <= 0.0) {
            logger->error(
                "beammap reductions require positive finite source fluxes; {} value_mJy={}",
                array_name, flux);
            valid = false;
        }
    }
    return valid;
}

}  // namespace citlali::pipeline

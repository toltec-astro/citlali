#pragma once

#include <citlali/core/config/calibration_config_validation.h>
#include <citlali/core/error/error.h>

#include <Eigen/Core>

#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <utility>

namespace citlali::pipeline {

struct BeammapPhotometryObservationConfig {
    citlali::config::BeammapPhotometryConfig photometry;
    std::map<std::string, double> fluxes_mjy_beam;
};

template <class Config, class FluxMap, class PhotometryConfig>
void read_beammap_source_fluxes(Config &config, FluxMap &fluxes_mjy_beam,
                                PhotometryConfig &photometry) {
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
        photometry.fluxes.push_back(
            citlali::config::BeammapArrayFluxConfig{
                array, flux, uncertainty_mjy});
    }
}

template <class Config>
BeammapPhotometryObservationConfig read_beammap_photometry_config(
    Config &config) {
    BeammapPhotometryObservationConfig observation;
    read_beammap_source_fluxes(
        config, observation.fluxes_mjy_beam, observation.photometry);
    return observation;
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

template <class ArrayNameMap, class Logger>
void require_valid_beammap_source_fluxes(
    const BeammapPhotometryObservationConfig &observation,
    const ArrayNameMap &array_name_map, const Logger &logger) {
    citlali::config::ValidationReport report;
    citlali::config::validate(observation.photometry, report);
    if (!report.ok()) {
        logger->error(
            "invalid beammap_source flux configuration:\n{}",
            report.format_for_cli());
        throw citlali::error::invalid_config(
            "invalid beammap_source flux configuration");
    }
    if (!validate_beammap_source_fluxes(
            observation.fluxes_mjy_beam, array_name_map, logger)) {
        throw citlali::error::invalid_config(
            "invalid beammap_source flux configuration");
    }
}

template <class PhotometryConfig, class FluxMap>
void install_beammap_photometry_config(
    BeammapPhotometryObservationConfig observation,
    PhotometryConfig &photometry, FluxMap &fluxes_mjy_beam,
    FluxMap &fluxes_mjy_sr) {
    photometry = std::move(observation.photometry);
    fluxes_mjy_beam = std::move(observation.fluxes_mjy_beam);
    fluxes_mjy_sr.clear();
}

}  // namespace citlali::pipeline

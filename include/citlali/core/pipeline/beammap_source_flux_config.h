#pragma once

#include <citlali/core/config/beammap_config.h>

#include <Eigen/Core>

#include <tuple>

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline

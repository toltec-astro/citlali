#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

namespace citlali::engine_detail {

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

}  // namespace citlali::engine_detail

template<typename CT>
void Engine::get_photometry_config(CT &config) {
    auto &source_config = citlali::pipeline::beammap_config(*this).source;
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    source_config = citlali::config::BeammapSourceConfig{};

    // beammap source name
    citlali::engine_detail::read_config_value(
        config, source_config.name, config_diag,
        std::tuple{"beammap_source","name"});
    // beammap source ra
    citlali::engine_detail::read_config_value(
        config, source_config.ra_deg, config_diag,
        std::tuple{"beammap_source","ra_deg"});

    // beammap source dec
    citlali::engine_detail::read_config_value(
        config, source_config.dec_deg, config_diag,
        std::tuple{"beammap_source","dec_deg"});

    // get source fluxes
    citlali::engine_detail::read_beammap_source_fluxes(
        config, source_flux_mJy_beam, source_config);

    if (citlali::pipeline::runtime_config(*this).reduction_type ==
        citlali::config::ReductionType::beammap) {
        bool valid_flux_config = true;
        for (auto const& entry : toltec_io.array_name_map) {
            const auto &arr_name = entry.second;
            auto flux_it = source_flux_mJy_beam.find(arr_name);
            if (flux_it == source_flux_mJy_beam.end()) {
                logger->error(
                    "beammap reductions require a positive source flux for {}; no beammap_source.fluxes entry was found",
                    arr_name);
                valid_flux_config = false;
                continue;
            }
            const double flux = flux_it->second;
            if (!std::isfinite(flux) || flux <= 0.0) {
                logger->error(
                    "beammap reductions require positive finite source fluxes; {} value_mJy={}",
                    arr_name, flux);
                valid_flux_config = false;
            }
        }
        if (!valid_flux_config) {
            std::exit(EXIT_FAILURE);
        }
    }
}

#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/citlali_config_read.h>
#include <citlali/core/engine/detail/mapmaking_activation_policy.h>
#include <citlali/core/engine/detail/source_protection_activation.h>

template<typename CT>
void Engine::get_citlali_config(CT &config) {
    // interface key names
    const std::vector<std::string> interface_keys = {
        "toltec0",
        "toltec1",
        "toltec2",
        "toltec3",
        "toltec4",
        "toltec5",
        "toltec6",
        "toltec7",
        "toltec8",
        "toltec9",
        "toltec10",
        "toltec11",
        "toltec12",
        "hwpr"
    };
    // initialize all offsets explicitly to zero
    for (const auto &key : interface_keys) {
        interface_sync_offset[key] = 0.0;
    }

    //  get interface offsets
    if (config.has(std::tuple{"interface_sync_offset"})) {
        auto interface_node = config.get_node(std::tuple{"interface_sync_offset"});
        std::set<std::string> configured_keys;
        // parse each list entry by key name so YAML order does not matter
        for (Eigen::Index i=0; i<interface_node.size(); ++i) {
            bool found_key = false;
            for (const auto &key : interface_keys) {
                if (config.has(std::tuple{"interface_sync_offset", i, key})) {
                    auto offset = config.template get_typed<double>(std::tuple{"interface_sync_offset", i, key});
                    if (configured_keys.find(key) != configured_keys.end()) {
                        logger->warn("interface_sync_offset for {} specified multiple times; using last value", key);
                    }
                    interface_sync_offset[key] = offset;
                    configured_keys.insert(key);
                    found_key = true;
                }
            }
            if (!found_key) {
                logger->warn("interface_sync_offset entry {} does not contain a recognized interface key; ignoring entry", i);
            }
        }
        for (const auto &key : interface_keys) {
            if (configured_keys.find(key) == configured_keys.end()) {
                logger->warn("interface_sync_offset missing {}; using 0.0 s", key);
            }
        }
    }

    typed_runtime_config = get_runtime_config(config);
    if (!typed_runtime_config.interp_over_gaps) {
        logger->error("runtime.interp_over_gaps=false is unsupported; set runtime.interp_over_gaps: true");
        std::exit(EXIT_FAILURE);
    }

    /* get timestream config */
    get_timestream_config(config);
    citlali::engine_detail::apply_source_protection_activation(
        redu_type, rtcproc, ptcproc, typed_timestream_config, logger);

    /* get mapmaking config */
    typed_post_processing_config = citlali::config::PostProcessingConfig{};
    get_mapmaking_config(config);

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return citlali::engine_detail::config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before);
    };

    citlali::engine_detail::read_post_processing_activation_config(
        config, run_map_filter, run_source_finder,
        typed_post_processing_config, missing_keys, invalid_keys);

    // map fitter options if in pointing or beammap mode or if map filtering or source finding are enabled
    citlali::engine_detail::read_source_fitting_config(
        config, redu_type, run_map_filter, run_source_finder, map_fitter,
        omb.pixel_size_rad, ASEC_TO_RAD, typed_post_processing_config,
        missing_keys, invalid_keys);

    /* get wiener filter config */
    if (run_map_filter) {
        // needs map fitter config
        get_map_filter_config(config);
    }

    // get source finder config options
    if (run_source_finder) {
        // minimum found source sigma
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.source_sigma, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_finding","source_sigma"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_finding.source_sigma =
                    omb.source_sigma;
            }
        }
        // window around source to exclude other sources
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.source_window_rad, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_finding","source_window_arcsec"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_finding.source_window_arcsec =
                    omb.source_window_rad;
            }
        }
        // search map, negative of map, or both
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.source_finder_mode, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_finding","mode"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_finding.mode =
                    omb.source_finder_mode;
            }
        }

        // convert source window to radians
        omb.source_window_rad =
            citlali::pipeline::source_window_arcsec_to_rad(
                omb.source_window_rad, ASEC_TO_RAD);

        citlali::pipeline::mirror_source_finding_config_to_coadd(
            omb, cmb, run_coadd);
    }

    /* get pointing config */
    if (redu_type=="pointing") {
        get_pointing_config(config);
    }

    /* get beammap config */
    if (redu_type=="beammap") {
        // needs redu_type config
        get_beammap_config(config);
    }

    // disable map related keys if map-making is disabled
    citlali::engine_detail::disable_map_products_if_mapmaking_disabled(
        run_mapmaking, run_coadd, run_noise, run_map_filter,
        run_source_finder, typed_coadd_config, typed_noise_config,
        typed_post_processing_config, beammap_iter_max,
        typed_beammap_config);
}

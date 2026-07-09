#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/beammap_config_loading.h>

template<typename CT>
void Engine::get_beammap_config(CT &config) {
    logger->info("getting beammap config options");
    auto &beammap_config = typed_config.beammap;
    const auto beammap_core_config =
        citlali::pipeline::read_beammap_core_config(
            config, missing_keys, invalid_keys, logger);
    citlali::pipeline::sync_beammap_core_controls(
        *this, beammap_core_config);

    const auto beammap_fitting_config =
        citlali::pipeline::read_beammap_fitting_config(
            config, missing_keys, invalid_keys);

    const auto beammap_scan_band_mask_config =
        citlali::pipeline::read_beammap_scan_band_mask_config(
            config, missing_keys, invalid_keys);

    const auto beammap_split_fits_config =
        citlali::pipeline::read_beammap_split_fits_config(
            config, missing_keys, invalid_keys, logger);
    citlali::pipeline::sync_beammap_map_controls(
        *this, beammap_fitting_config, beammap_scan_band_mask_config,
        beammap_split_fits_config, map_fitter);

    const auto beammap_priors_config =
        citlali::pipeline::read_beammap_priors_config(
            config, missing_keys, invalid_keys, logger);
    citlali::pipeline::sync_beammap_priors_controls(
        *this, beammap_priors_config);

    const auto flagging_vectors =
        citlali::pipeline::read_beammap_flagging_vectors(
            config, missing_keys, invalid_keys, toltec_io.array_name_map.size());
    beammap_flag_max_prior_d2 = flagging_vectors.max_prior_d2;

    citlali::pipeline::assign_beammap_array_flag_limits(
        toltec_io.array_name_map, flagging_vectors.lower_fwhm_arcsec,
        flagging_vectors.upper_fwhm_arcsec,
        flagging_vectors.lower_sig2noise,
        flagging_vectors.upper_sig2noise,
        flagging_vectors.max_dist_arcsec, flagging_vectors.network_robust_z,
        lower_fwhm_arcsec, upper_fwhm_arcsec, lower_sig2noise,
        upper_sig2noise, max_dist_arcsec, network_robust_z);

    std::vector<double> sens_factors_vec;
    std::vector<double> sens_psd_limits_Hz_vec;
    citlali::pipeline::read_beammap_sensitivity_config(
        config, invalid_keys, lower_sens_factor, upper_sens_factor,
        sens_psd_limits_Hz, sens_factors_vec, sens_psd_limits_Hz_vec);

    // Beammap PTC TOD/diagnostics are written after the convergence decision.
    // The default is the actual last attempted iteration, including early
    // convergence, so the saved PTC reflects the final cleaning state.
    beammap_tod_output_iter =
        citlali::pipeline::default_beammap_tod_output_iter();

    const auto beammap_detector_tod_output_config =
        citlali::pipeline::read_beammap_detector_tod_output_config(
            config, missing_keys, invalid_keys);
    citlali::pipeline::sync_beammap_detector_tod_output_controls(
        *this, beammap_detector_tod_output_config);

    citlali::pipeline::reset_beammap_config_mirror(beammap_config);
    citlali::pipeline::mirror_beammap_core_config(
        beammap_config, beammap_core_config, beammap_fitting_config,
        beammap_scan_band_mask_config, beammap_split_fits_config);
    citlali::pipeline::mirror_beammap_priors_config(
        beammap_config, beammap_priors_config);
    citlali::pipeline::mirror_beammap_output_and_flagging_config(
        beammap_config, beammap_detector_tod_output_config, flagging_vectors,
        sens_factors_vec, sens_psd_limits_Hz_vec);
}

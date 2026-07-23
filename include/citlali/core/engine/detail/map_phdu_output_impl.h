#pragma once

// Engine FITS map output implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/map_phdu_output_helpers.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template <typename fits_io_type, class map_buffer_t>
void Engine::add_phdu(fits_io_type &fits_io, map_buffer_t &mb, Eigen::Index i) {
    citlali::pipeline::require_phdu_output_slots(
        i, static_cast<Eigen::Index>(fits_io->size()),
        static_cast<Eigen::Index>(calib.arrays.size()), logger);

    const auto array_id = citlali::pipeline::phdu_array_id(calib.arrays, i);

    // array name
    std::string name = citlali::pipeline::phdu_array_name(
        toltec_io.array_name_map, array_id);
    auto &fits_entry = fits_io->at(i);
    const auto reduction_type =
        citlali::pipeline::runtime_reduction_type(*this);
    const auto &beammap_settings = citlali::pipeline::beammap_config(*this);
    const auto &mapmaking_settings =
        citlali::pipeline::mapmaking_config(*this);
    const auto &raw_timestream_settings =
        citlali::pipeline::raw_time_chunk_config(*this);
    const auto &beammap_iteration_config = beammap_settings.iteration;
    const auto &beammap_phase_config = beammap_settings.phase_strategy;
    const auto &beammap_reference_config = beammap_settings.reference;

    try {
    citlali::engine_detail::add_phdu_unit_conversion_section(
        fits_entry, mb, calib, toltec_io, array_id, name,
        raw_timestream_settings.flux_calibration_enabled,
        FWHM_TO_STD, ASEC_TO_RAD, pi,
        MJY_SR_TO_mJY_ASEC, logger);

    citlali::engine_detail::add_phdu_beammap_observation_section(
        fits_entry, mb, name, logger, reduction_type,
        source_flux_mJy_beam, source_flux_MJy_Sr,
        beammap_iteration_config, beammap_phase_config,
        beammap_reference_config, calib, observation_dates.date_obs);

    logger->debug("adding obs info");

    citlali::engine_detail::add_phdu_identity_geometry_section(
        fits_entry, mb, telescope, calib, name, CITLALI_GIT_VERSION,
        KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
        reduction_type, citlali::pipeline::timestream_config(*this).type,
        mapmaking_settings.grouping, mapmaking_settings.method,
        RAD_TO_DEG, logger);

    logger->debug("adding beamsizes");

    // add beamsizes
    citlali::pipeline::add_phdu_beam_geometry_keys(
        fits_entry, name, logger, calib.array_fwhms[array_id],
        calib.array_pas[array_id], RAD_TO_DEG, pi/2);

    citlali::pipeline::add_phdu_auxiliary_scalar_keys(
        fits_entry, mb->sig_unit, telescope.fsmp, iteration.fruit_iter);

    // add jinc shape params
    citlali::pipeline::add_phdu_jinc_shape_keys_if_needed(
        fits_entry, name, logger, mapmaking_settings.method, jinc_mm.r_max,
        jinc_mm.shape_params, array_id);

    citlali::engine_detail::add_phdu_extinction_apt_oof_section(
        fits_entry, mb, rtcproc, telescope, calib, toltec_io, i, array_id,
        name, reduction_type,
        raw_timestream_settings.extinction_correction_enabled, logger);

    citlali::engine_detail::add_phdu_tod_runtime_config_section(
        fits_entry, name, logger,
        citlali::pipeline::verbose_runtime_enabled(*this),
        citlali::pipeline::polarimetry_config(*this).enabled,
        raw_timestream_settings,
        citlali::pipeline::processed_time_chunk_config(*this), rtcproc,
        telescope.outer_scans_chunk);

    citlali::engine_detail::add_phdu_ptc_learning_config_section(
        fits_entry, name, logger, ptcproc, calib, learning, i,
        array_id, mb->sig_unit,
        citlali::pipeline::processed_time_chunk_config(*this),
        citlali::pipeline::fruit_loops_config(*this),
        citlali::pipeline::pointing_config(*this), reduction_type);

    citlali::engine_detail::add_phdu_pointing_telescope_header_section(
        fits_entry, mb, telescope, name, logger, reduction_type,
        citlali::pipeline::pointing_config(*this));
    } catch (const CCfits::FitsException &e) {
        throw std::runtime_error(
            citlali::pipeline::phdu_write_error_message(
                name, fits_io->at(i).filepath, e.message()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            citlali::pipeline::phdu_write_error_message(
                name, fits_io->at(i).filepath, e.what()));
    }
}

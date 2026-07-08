#pragma once

// Engine FITS map output implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/map_phdu_output_helpers.h>

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
    const std::string reduction_type_name{
        citlali::config::to_string(typed_config.runtime.reduction_type)};
    const std::string tod_type_name{
        citlali::config::to_string(typed_config.timestream.type)};
    const std::string map_grouping_name{
        citlali::config::to_string(typed_config.mapmaking.grouping)};
    const std::string map_method_name{
        citlali::config::to_string(typed_config.mapmaking.method)};
    const auto &beammap_iteration_config = typed_config.beammap.iteration;
    const auto &beammap_phase_config = typed_config.beammap.phase_strategy;
    const auto &beammap_reference_config = typed_config.beammap.reference;

    try {
    citlali::engine_detail::add_phdu_unit_conversion_section(
        fits_entry, mb, calib, toltec_io, array_id, name,
        rtcproc.run_calibrate, FWHM_TO_STD, ASEC_TO_RAD, pi,
        MJY_SR_TO_mJY_ASEC, logger);

    citlali::engine_detail::add_phdu_beammap_observation_section(
        fits_entry, mb, name, logger, reduction_type_name,
        beammap_fluxes_mJy_beam,
        beammap_fluxes_MJy_Sr, beammap_iteration_config.tolerance,
        beammap_iteration_config.convergence_radius_arcsec,
        beammap_iteration_config.max_iterations, beammap_phase_config.enabled,
        beammap_phase_config.locator_iter,
        beammap_phase_config.measurement_start_iter,
        beammap_reference_config.derotate,
        beammap_reference_config.subtract_reference_detector, calib,
        static_cast<Eigen::Index>(beammap_reference_config.reference_detector),
        date_obs);

    logger->debug("adding obs info");

    citlali::engine_detail::add_phdu_identity_geometry_section(
        fits_entry, mb, telescope, calib, name, CITLALI_GIT_VERSION,
        KIDSCPP_GIT_VERSION, TULA_GIT_VERSION, reduction_type_name,
        tod_type_name, map_grouping_name, map_method_name, RAD_TO_DEG,
        logger);

    logger->debug("adding beamsizes");

    // add beamsizes
    citlali::pipeline::add_phdu_beam_geometry_keys(
        fits_entry, name, logger, calib.array_fwhms[array_id],
        calib.array_pas[array_id], RAD_TO_DEG, pi/2);

    citlali::pipeline::add_phdu_auxiliary_scalar_keys(
        fits_entry, mb->sig_unit, telescope.fsmp, fruit_iter);

    // add jinc shape params
    citlali::pipeline::add_phdu_jinc_shape_keys_if_needed(
        fits_entry, name, logger, typed_config.mapmaking.method, jinc_mm.r_max,
        jinc_mm.shape_params, array_id);

    citlali::engine_detail::add_phdu_extinction_apt_oof_section(
        fits_entry, mb, rtcproc, telescope, calib, toltec_io, i, array_id,
        name, reduction_type_name, logger);

    citlali::engine_detail::add_phdu_tod_runtime_config_section(
        fits_entry, name, logger, verbose_mode, rtcproc, ptcproc,
        telescope.outer_scans_chunk);

    citlali::engine_detail::add_phdu_ptc_learning_config_section(
        fits_entry, name, logger, ptcproc, calib, reduction_learning, i,
        array_id, mb->sig_unit);

    citlali::engine_detail::add_phdu_pointing_telescope_header_section(
        fits_entry, mb, telescope, name, logger, reduction_type_name,
        pointing_source_strategy, pointing_fit_gaussian_enabled,
        pointing_fruitloops_center_mode,
        pointing_header_center_max_radius_arcsec,
        pointing_header_center_require_coverage);
    } catch (const CCfits::FitsError &e) {
        throw std::runtime_error(
            citlali::pipeline::phdu_write_error_message(
                name, fits_io->at(i).filepath, e.message()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            citlali::pipeline::phdu_write_error_message(
                name, fits_io->at(i).filepath, e.what()));
    }
}

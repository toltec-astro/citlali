#pragma once

// Beammap implementation detail. Include only after Beammap is declared.

void Beammap::finalize_beammap_directional_product(
    citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    calculate_beammap_detector_sensitivities(omb.parallel_policy);
    populate_beammap_detector_fit_apt_columns();
    populate_beammap_mask_diagnostic_apt_columns();
    log_beammap_final_bound_summary();
    set_apt_flags();
    process_apt();
    apply_final_network_position_flags();
    update_final_prior_match_diagnostics();
    write_beammap_final_prior_diagnostics_to_apt();
    refresh_beammap_final_calibration_products();
}

citlali::engine_detail::beammap::DirectionalProduct
Beammap::fit_beammap_directional_product(
    citlali::config::BeammapDirectionMode mode,
    mapmaking::MapBuffer &direction_buffer,
    const engine::Calib &common_calib,
    const citlali::engine_detail::beammap::DirectionalProduct &standard_state,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    using citlali::engine_detail::beammap::ObservationMapBufferTransaction;
    using citlali::engine_detail::beammap::capture_product_state;
    using citlali::engine_detail::beammap::clone_product_calib;
    using citlali::engine_detail::beammap::restore_product_state;

    restore_product_state(*this, standard_state);
    calib = clone_product_calib(common_calib);
    ObservationMapBufferTransaction map_transaction{omb, direction_buffer};

    reset_beammap_fit_buffers();
    p0 = standard_state.params;
    perror0 = standard_state.perrors;
    good_fits = standard_state.good_fits;
    converged.setConstant(map_indices.n_maps, false);
    converge_iter = standard_state.converge_iter;
    calib.apt_meta["beammap_direction_mode"] =
        std::string{citlali::config::to_string(mode)};
    calib.apt_meta["beammap_direction_fit"] =
        "existing Beammap fit/QC pipeline on the final directional map buffer";

    fit_beammap_maps(true, true, stage_profile, false);
    finalize_beammap_directional_product(stage_profile);
    const auto product = capture_product_state(*this, mode);

    write_detector_table_outputs();
    logger->info(
        "beammap direction_mode=all finalized {} maps and APT from shared processed PTC chunks",
        citlali::config::to_string(mode));
    return product;
}

void Beammap::build_beammap_all_directional_products(
    const engine::Calib &common_calib,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    using Mode = citlali::config::BeammapDirectionMode;
    using citlali::engine_detail::beammap::ProductStateTransaction;
    using citlali::engine_detail::beammap::restore_product_state;

    if (!citlali::pipeline::beammap_direction_mode_is_all(
            citlali::pipeline::beammap_config(*this).direction_mode)) {
        return;
    }
    if (!beammap_direction_products.buffers_initialized) {
        throw std::logic_error(
            "beammap direction_mode=all lacks directional map buffers");
    }

    ProductStateTransaction transaction{*this};
    const auto standard_state = transaction.saved();
    beammap_direction_products.left_product =
        fit_beammap_directional_product(
            Mode::left, beammap_direction_products.left, common_calib,
            standard_state, stage_profile);
    beammap_direction_products.right_product =
        fit_beammap_directional_product(
            Mode::right, beammap_direction_products.right, common_calib,
            standard_state, stage_profile);
    restore_product_state(*this, standard_state);
    transaction.restore();
}

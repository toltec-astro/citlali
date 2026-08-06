#pragma once

// Beammap implementation detail. Include only after Beammap is declared.

#include <citlali/core/pipeline/raw_map_outputs.h>
#include <citlali/core/pipeline/filtered_map_outputs.h>

void Beammap::create_beammap_product_map_files(
    citlali::config::BeammapDirectionMode mode,
    bool include_raw, bool include_filtered) {
    const auto raw_dir =
        citlali::pipeline::raw_observation_map_directory(
            output_paths.obsnum_dir_name);
    const auto filtered_dir =
        citlali::pipeline::filtered_observation_map_directory(
            output_paths.obsnum_dir_name);
    if (include_raw) {
        map_fits_outputs.obs.clear();
        map_fits_outputs.obs_noise.clear();
    }
    if (include_filtered) {
        map_fits_outputs.filtered_obs.clear();
        map_fits_outputs.filtered_obs_noise.clear();
    }
    const bool write_noise =
        citlali::pipeline::should_create_observation_noise_maps(*this);
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        const auto array = calib.arrays[i];
        const auto array_name = toltec_io.array_name_map[array];
        if (include_raw) {
            auto filename =
                citlali::pipeline::beammap_direction_product_filename(
                    citlali::pipeline::observation_output_filename<
                        engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::map,
                        engine_utils::toltecIO::raw>(
                            toltec_io, raw_dir,
                            citlali::pipeline::runtime_reduction_type(*this),
                            array_name, observation_identity.obsnum,
                            telescope.sim_obs),
                    mode);
            map_fits_outputs.obs.emplace_back(filename);
            if (write_noise) {
                auto noise_filename =
                    citlali::pipeline::beammap_direction_product_filename(
                        citlali::pipeline::observation_output_filename<
                            engine_utils::toltecIO::toltec,
                            engine_utils::toltecIO::noise,
                            engine_utils::toltecIO::raw>(
                                toltec_io, raw_dir,
                                citlali::pipeline::runtime_reduction_type(*this),
                                array_name, observation_identity.obsnum,
                                telescope.sim_obs),
                        mode);
                map_fits_outputs.obs_noise.emplace_back(noise_filename);
            }
        }
        if (include_filtered) {
            auto filename =
                citlali::pipeline::beammap_direction_product_filename(
                    citlali::pipeline::observation_output_filename<
                        engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::map,
                        engine_utils::toltecIO::filtered>(
                            toltec_io, filtered_dir,
                            citlali::pipeline::runtime_reduction_type(*this),
                            array_name, observation_identity.obsnum,
                            telescope.sim_obs),
                    mode);
            map_fits_outputs.filtered_obs.emplace_back(filename);
            if (write_noise) {
                auto noise_filename =
                    citlali::pipeline::beammap_direction_product_filename(
                        citlali::pipeline::observation_output_filename<
                            engine_utils::toltecIO::toltec,
                            engine_utils::toltecIO::noise,
                            engine_utils::toltecIO::filtered>(
                                toltec_io, filtered_dir,
                                citlali::pipeline::runtime_reduction_type(*this),
                                array_name, observation_identity.obsnum,
                                telescope.sim_obs),
                        mode);
                map_fits_outputs.filtered_obs_noise.emplace_back(
                    noise_filename);
            }
        }
    }
}

void Beammap::write_beammap_directional_raw_product(
    const citlali::engine_detail::beammap::DirectionalProduct &product,
    mapmaking::MapBuffer &map_buffer,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    using citlali::engine_detail::beammap::ProductStateTransaction;
    using citlali::engine_detail::beammap::restore_product_state;

    ProductStateTransaction transaction{*this};
    restore_product_state(*this, product);
    citlali::pipeline::calculate_unfiltered_map_noise_products_if_needed(
        *this, map_buffer, stage_profile, logger, true,
        "calculating directional raw obs empirical noise products");
    create_beammap_product_map_files(product.mode, true, false);
    map_buffer.calc_median_err();
    write_beammap_map_products<mapmaking::RawObs>(
        &map_buffer, &map_fits_outputs.obs,
        &map_fits_outputs.obs_noise, stage_profile,
        output_paths.obsnum_dir_name + "raw/");
    transaction.restore();
}

void Beammap::write_beammap_directional_filtered_product(
    const citlali::engine_detail::beammap::DirectionalProduct &product,
    mapmaking::MapBuffer &map_buffer,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    using citlali::engine_detail::beammap::ProductStateTransaction;
    using citlali::engine_detail::beammap::restore_product_state;

    ProductStateTransaction transaction{*this};
    restore_product_state(*this, product);
    create_beammap_product_map_files(product.mode, false, true);
    citlali::pipeline::filter_maps<mapmaking::FilteredObs>(
        *this, map_buffer, stage_profile, logger,
        "filtering directional obs maps");
    citlali::pipeline::calculate_filtered_map_noise_products_if_needed(
        *this, map_buffer, stage_profile, logger,
        "calculating directional filtered obs empirical noise products");
    citlali::pipeline::calculate_filtered_map_diagnostics(
        map_buffer, stage_profile, logger,
        "calculating directional filtered obs map psds",
        "calculating directional filtered obs map histograms");
    write_beammap_map_products<mapmaking::FilteredObs>(
        &map_buffer, &map_fits_outputs.filtered_obs,
        &map_fits_outputs.filtered_obs_noise, stage_profile,
        output_paths.obsnum_dir_name + "filtered/");
    transaction.restore();
}

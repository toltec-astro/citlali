#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_omb(map_extent_t &map_extent, map_coord_t &map_coord) {
    auto& omb = engine().omb;
    const auto &mapmaking_settings =
        citlali::pipeline::mapmaking_config(engine());
    const bool allocate_science_products =
        citlali::pipeline::science_map_v1_profile_available(
            mapmaking_settings.method, mapmaking_settings.grouping,
            engine().rtcproc.run_polarization);
    const std::string science_product_absence_reason =
        citlali::pipeline::science_map_v1_profile_absence_reason(
            mapmaking_settings.method, mapmaking_settings.grouping,
            engine().rtcproc.run_polarization);

    citlali::pipeline::clear_map_matrix_products(omb);
    citlali::pipeline::apply_observation_map_geometry(
        omb, map_extent, map_coord);
    citlali::pipeline::allocate_map_matrices(
        omb, engine().map_indices.n_maps,
        mapmaking_settings.method == citlali::config::MapMethod::jinc,
        citlali::pipeline::raw_kernel_enabled(engine()),
        mapmaking_settings.method == citlali::config::MapMethod::jinc ||
            mapmaking_settings.grouping != citlali::config::MapGrouping::detector,
        allocate_science_products, science_product_absence_reason,
        mapmaking_settings.method == citlali::config::MapMethod::jinc);
    if (mapmaking_settings.method == citlali::config::MapMethod::jinc) {
        auto &provenance = omb.jinc_products.provenance;
        const auto &plan = citlali::pipeline::mapmaking_plan(engine());
        if (!plan.initialized ||
            plan.effective.method != citlali::config::MapMethod::jinc ||
            engine().jinc_mm.resolved_arrays.empty() ||
            engine().map_indices.maps_to_arrays.size() !=
                engine().map_indices.n_maps) {
            throw std::logic_error(
                "JINC observation allocation requires a resolved typed plan and map identity");
        }
        provenance.requested_digest =
            mapmaking::jinc_filter_config_digest(
                plan.requested.jinc_filter);
        provenance.effective_digest =
            mapmaking::jinc_filter_config_digest(
                plan.effective.jinc_filter);
        provenance.requested_r_max = plan.requested.jinc_filter.r_max;
        provenance.effective_r_max = plan.effective.jinc_filter.r_max;
        provenance.requested_subpixel_n =
            plan.requested.jinc_filter.subpixel_n;
        provenance.effective_subpixel_n =
            plan.effective.jinc_filter.subpixel_n;
        provenance.requested_shape_params =
            plan.requested.jinc_filter.shape_params;
        provenance.effective_shape_params =
            plan.effective.jinc_filter.shape_params;
        for (const auto &resolved : engine().jinc_mm.resolved_arrays) {
            bool selected = false;
            for (Eigen::Index slot = 0;
                 slot < engine().map_indices.maps_to_arrays.size(); ++slot) {
                if (engine().map_indices.maps_to_arrays(slot) ==
                    resolved.array_id) {
                    selected = true;
                    break;
                }
            }
            if (selected) {
                mapmaking::validate_jinc_resolved_array(resolved);
                provenance.resolved_arrays.push_back(resolved);
            }
        }
        for (Eigen::Index slot = 0;
             slot < engine().map_indices.maps_to_arrays.size(); ++slot) {
            const auto array_id =
                engine().map_indices.maps_to_arrays(slot);
            const bool resolved = std::any_of(
                provenance.resolved_arrays.begin(),
                provenance.resolved_arrays.end(), [&](const auto &record) {
                    return record.array_id == array_id;
                });
            if (!resolved) {
                throw std::logic_error(
                    "JINC selected map array lacks a resolved stable identity");
            }
        }
        const auto bool_text = [](bool value) {
            return value ? std::string{"true"} : std::string{"false"};
        };
        const auto vector_text = [](const auto &values) {
            std::ostringstream stream;
            for (const auto &value : values) {
                if (stream.tellp() > 0) {
                    stream << ',';
                }
                if constexpr (std::is_floating_point_v<
                                  std::remove_cv_t<std::remove_reference_t<
                                      decltype(value)>>>) {
                    stream << mapmaking::jinc_double_hex(value);
                }
                else {
                    stream << value;
                }
            }
            return stream.str();
        };
        const auto &rtc = engine().rtcproc;
        const auto &kernel = rtc.kernel;
        const bool kernel_enabled =
            citlali::pipeline::raw_kernel_enabled(engine());
        provenance.kernel_template_identity =
            mapmaking::jinc_kernel_template_identity(
                kernel, kernel_enabled);

        const auto &ptc = engine().ptcproc;
        const auto &raw_plan =
            citlali::pipeline::raw_timestream_plan(engine());
        const auto &processed_plan =
            citlali::pipeline::processed_timestream_plan(engine());
        if (!raw_plan.initialized || !processed_plan.initialized) {
            throw std::logic_error(
                "JINC processing identity requires initialized typed timestream plans");
        }
        const auto &raw = raw_plan.effective;
        const auto &processed =
            processed_plan.effective.processed_time_chunk;
        std::vector<std::pair<std::string, std::string>> processing_facts{
            {"kernel_enabled", bool_text(raw.kernel.enabled)},
            {"despike_enabled", bool_text(raw.despike.enabled)},
            {"temporal_fir_enabled", bool_text(raw.filter.enabled)},
            {"configured_notch_enabled",
             bool_text(raw.filter.enabled && raw.filter.notch.enabled)},
            {"iir_highpass_enabled", bool_text(raw.iir_filter.enabled)},
            {"downsample_enabled", bool_text(raw.downsample.enabled)},
            {"ptc_clean_enabled", bool_text(processed.clean.enabled)},
            {"fir_coefficients_digest", mapmaking::jinc_matrix_digest(rtc.filter.filter)},
            {"notch_centers_hz", vector_text(rtc.filter.w0s)},
            {"notch_q", vector_text(rtc.filter.qs)},
            {"iir_highpass_hz", mapmaking::jinc_double_hex(rtc.filter.iir_highpass_freq_Hz)},
            {"iir_highpass_order", std::to_string(rtc.filter.iir_highpass_order)},
            {"iir_highpass_zero_phase", bool_text(rtc.filter.iir_highpass_zero_phase)},
            {"notch_zero_phase", bool_text(rtc.filter.notch_zero_phase)},
            {"ptc_stddev_limit", mapmaking::jinc_double_hex(ptc.cleaner.stddev_limit)},
            {"ptc_tau", mapmaking::jinc_double_hex(ptc.cleaner.tau)},
            {"ptc_n_calc", std::to_string(ptc.cleaner.n_calc)},
            {"ptc_grouping", vector_text(ptc.cleaner.grouping)},
            {"flags_and_masks", "kernel-and-signal-share-realized-eligibility-v1"},
        };
        for (const auto &[group, cuts] : ptc.cleaner.n_eig_to_cut) {
            processing_facts.emplace_back(
                "ptc_n_eig_to_cut_group_" + std::to_string(group),
                vector_text(cuts));
        }
        provenance.processing_configuration_identity =
            mapmaking::jinc_realization_identity_digest(
                "actual-enabled-processing-operators-v1", processing_facts);
        provenance.processing_realization_identity =
            provenance.processing_configuration_identity;
        if (!std::isfinite(engine().telescope.d_fsmp) ||
            engine().telescope.d_fsmp <= 0.0) {
            throw std::logic_error(
                "JINC coverage requires a finite positive realized sample frequency");
        }
        provenance.coverage_sample_frequency_identity =
            "effective-processed-timestream-sample-rate-telescope-d_fsmp-v1";
        provenance.coverage_sample_frequency_hz =
            engine().telescope.d_fsmp;
    }
    citlali::pipeline::allocate_polarization_pointing_matrices(
        omb, engine().map_indices.n_maps,
        static_cast<Eigen::Index>(
            engine().rtcproc.polarization.stokes_params.size()),
        engine().rtcproc.run_polarization);
}

// allocate the coadded map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_cmb() {
    auto& cmb = engine().cmb;
    const auto &mapmaking_settings =
        citlali::pipeline::mapmaking_config(engine());
    const bool allocate_science_products =
        citlali::pipeline::science_map_v1_profile_available(
            mapmaking_settings.method, mapmaking_settings.grouping,
            engine().rtcproc.run_polarization);
    const std::string science_product_absence_reason =
        citlali::pipeline::science_map_v1_profile_absence_reason(
            mapmaking_settings.method, mapmaking_settings.grouping,
            engine().rtcproc.run_polarization);

    citlali::pipeline::clear_map_matrix_products(cmb);
    citlali::pipeline::allocate_map_matrices(
        cmb, engine().map_indices.n_maps, false,
        citlali::pipeline::raw_kernel_enabled(engine()),
        mapmaking_settings.grouping != citlali::config::MapGrouping::detector,
        allocate_science_products, science_product_absence_reason, false);
    citlali::pipeline::allocate_polarization_pointing_matrices(
        cmb, engine().map_indices.n_maps,
        static_cast<Eigen::Index>(
            engine().rtcproc.polarization.stokes_params.size()),
        engine().rtcproc.run_polarization);
}

template <class EngineType>
template <class map_buffer_t>
void TimeOrderedDataProc<EngineType>::allocate_nmb(map_buffer_t &nmb) {
    // clear noise map buffer
    std::vector<Eigen::Tensor<double,3>>().swap(nmb.noise);

    const double nmb_size_gb =
        8.0 * static_cast<double>(nmb.n_rows) * static_cast<double>(nmb.n_cols) *
        static_cast<double>(engine().map_indices.n_maps) * static_cast<double>(nmb.n_noise) / 1e9;
    engine().logger->info("allocating {} noise realization cube: rows={} cols={} maps={} n_noise={} estimated_size={:.2f} GB",
                          nmb.name, static_cast<long long>(nmb.n_rows),
                          static_cast<long long>(nmb.n_cols),
                          static_cast<long long>(engine().map_indices.n_maps),
                          static_cast<long long>(nmb.n_noise),
                          nmb_size_gb);

    // resize noise maps (n_maps, [n_rows, n_cols, n_noise])
    for (Eigen::Index i=0; i<engine().map_indices.n_maps; ++i) {
        nmb.noise.emplace_back(nmb.n_rows, nmb.n_cols, nmb.n_noise);
        nmb.noise.at(i).setZero();
    }
}

// coadd maps

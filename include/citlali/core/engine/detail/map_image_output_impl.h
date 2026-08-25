#pragma once

// Engine FITS map output implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/map_image_output_helpers.h>
#include <citlali/core/pipeline/map_output_debug_breadcrumb.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/science_map_identity.h>

#include <type_traits>

template <typename fits_io_type, class map_buffer_t>
Eigen::Index Engine::write_maps(fits_io_type &fits_io, fits_io_type &noise_fits_io, map_buffer_t &mb, Eigen::Index i) {
    citlali::pipeline::require_map_data_slots(
        i, static_cast<Eigen::Index>(mb->signal.size()),
        static_cast<Eigen::Index>(mb->weight.size()), logger);
    citlali::pipeline::require_primary_map_image_shapes(
        mb->signal, mb->weight, i, mb->n_rows, mb->n_cols, logger);
    citlali::pipeline::require_map_wcs_cardinality(mb->wcs, 4, logger);
    if (mb->jinc_products.initialized &&
        !mapmaking::jinc_processing_provenance_complete(
            mb->jinc_products.provenance)) {
        citlali::pipeline::fail_required_output(
            logger,
            "JINC product publication requires completed actual processing provenance");
    }
    const bool is_filtered_output =
        citlali::pipeline::is_filtered_map_output(
            fits_io, map_fits_outputs.filtered_obs,
            map_fits_outputs.filtered_coadd);

    const mapmaking::ScienceMapBundleIdentity *typed_identity = nullptr;
    const mapmaking::ScienceMapSlotIdentity *typed_slot = nullptr;
    const bool frozen_raw_parent_available =
        static_cast<bool>(mb->raw_science_parent);
    if (is_filtered_output && !frozen_raw_parent_available &&
        citlali::pipeline::science_map_any_product_available(
            mb->science_products)) {
        citlali::pipeline::fail_required_output(
            logger,
            "filtered science-map output lacks an immutable raw-parent snapshot");
    }
    const auto &science_products =
        is_filtered_output && frozen_raw_parent_available
            ? *mb->raw_science_parent
            : mb->science_products;
    citlali::pipeline::require_science_map_output_profile_authority(
        science_products, mb->signal.size(), mb->n_rows, mb->n_cols,
        logger);
    std::string raw_parent_digest;
    if (science_products.initialized &&
        i >= 0 &&
        i < static_cast<Eigen::Index>(science_products.realized.size())) {
        const auto science_slot = static_cast<std::size_t>(i);
        const auto &record = science_products.realized[science_slot];
        const bool any_product_available = std::any_of(
            record.product_available.begin(), record.product_available.end(),
            [](bool available) { return available; });
        if (any_product_available) {
            if (!science_products.identity_admitted ||
                !science_products.bundle_identity ||
                science_products.bundle_identity->rows != mb->n_rows ||
                science_products.bundle_identity->cols != mb->n_cols ||
                science_products.bundle_identity->ordered_slots.size() !=
                    mb->signal.size() ||
                science_slot >=
                    science_products.bundle_identity->ordered_slots.size() ||
                !mapmaking::science_map_realized_product_facts_match(
                    science_products, science_slot) ||
                !citlali::pipeline::science_map_available_product_planes_have_shape(
                    science_products, i, mb->n_rows, mb->n_cols) ||
                (is_filtered_output
                     ? record.raw_parent_digest.empty()
                     : record.raw_parent_digest !=
                           mapmaking::science_map_raw_parent_digest(
                               *mb, science_slot)) ||
                science_slot >= science_products.retained_exposure.size() ||
                (!is_filtered_output &&
                 (science_slot >= mb->coverage.size() ||
                  !citlali::pipeline::science_map_planes_bitwise_equal(
                      mb->coverage[science_slot],
                      science_products.retained_exposure[science_slot])))) {
                citlali::pipeline::fail_required_output(
                    logger,
                    fmt::format(
                        "science-map output bundle is stale or incomplete for map index {}",
                        static_cast<long long>(i)));
            }
            typed_identity = &*science_products.bundle_identity;
            typed_slot = &typed_identity->ordered_slots[science_slot];
            raw_parent_digest = record.raw_parent_digest;
            const auto current_slot =
                citlali::pipeline::science_map_slot_identity(*this, i);
            const bool slot_matches =
                current_slot.ordered_slot == typed_slot->ordered_slot &&
                current_slot.grouping == typed_slot->grouping &&
                current_slot.group_identity == typed_slot->group_identity &&
                current_slot.array_identity == typed_slot->array_identity &&
                current_slot.stokes_identity == typed_slot->stokes_identity &&
                mapmaking::science_map_exact_double_equal(
                    current_slot.frequency_hz, typed_slot->frequency_hz);
            const double current_epoch =
                citlali::pipeline::science_map_source_epoch(
                    telescope.tel_header);
            const auto current_response =
                citlali::pipeline::science_map_response_identity(
                    rtcproc.kernel,
                    citlali::pipeline::raw_kernel_enabled(*this));
            if (!slot_matches || mb->map_grouping != typed_identity->grouping ||
                mb->sig_unit != typed_identity->signal_unit ||
                telescope.pixel_axes !=
                    typed_identity->wcs.coordinate_frame ||
                !mapmaking::science_map_exact_double_equal(
                    current_epoch, typed_identity->wcs.source_epoch) ||
                current_response != typed_identity->response_identity ||
                typed_identity->wcs.axis_types.size() != 2 ||
                typed_identity->wcs.axis_units.size() != 2 ||
                typed_identity->wcs.pixel_scale.size() != 2 ||
                typed_identity->wcs.reference_world.size() != 2 ||
                typed_identity->wcs.reference_pixel.size() != 2 ||
                mb->wcs.ctype.size() < 4 || mb->wcs.naxis.size() < 2) {
                citlali::pipeline::fail_required_output(
                    logger,
                    fmt::format(
                        "science-map output identity differs from the admitted bundle for map index {}",
                        static_cast<long long>(i)));
            }
        }
    }

    // get name for extension layer
    std::string map_name = get_map_name(i);

    const auto write_indices =
        citlali::pipeline::map_write_indices(
            i, map_indices.arrays_to_maps, map_indices.maps_to_stokes, map_indices.maps_to_arrays);
    const Eigen::Index map_index = write_indices.map_index;
    const Eigen::Index stokes_index = write_indices.stokes_index;
    const Eigen::Index array_id = write_indices.array_id;
    if (typed_identity && typed_slot) {
        Eigen::Index expected_file_index = 0;
        for (std::size_t slot_index = 1;
             slot_index <= typed_slot->ordered_slot; ++slot_index) {
            const auto previous =
                typed_identity->ordered_slots[slot_index - 1].array_identity;
            const auto current =
                typed_identity->ordered_slots[slot_index].array_identity;
            if (current > previous) {
                ++expected_file_index;
            }
            else if (current < previous) {
                expected_file_index = 0;
            }
        }
        if (map_index != expected_file_index ||
            array_id != typed_slot->array_identity ||
            stokes_index != typed_slot->stokes_identity) {
            citlali::pipeline::fail_required_output(
                logger,
                fmt::format(
                    "science-map output file/array/Stokes routing differs from admitted slot {}",
                    typed_slot->ordered_slot));
        }
    }
    citlali::pipeline::require_map_write_index_slots(
        i, map_index, static_cast<Eigen::Index>(fits_io->size()),
        stokes_index,
        static_cast<Eigen::Index>(rtcproc.polarization.stokes_params.size()),
        array_id, logger);
    const bool is_beammap =
        citlali::pipeline::runtime_reduction_type(*this) ==
        citlali::config::ReductionType::beammap;
    const bool empirical_weight_calibration =
        citlali::pipeline::empirical_weight_calibration_enabled(*this);
    const bool empirical_noise_products_expected =
        citlali::pipeline::noise_maps_enabled(*this) &&
        citlali::pipeline::noise_product_outputs_enabled(*this);
    const bool noise_realization_outputs_expected =
        citlali::pipeline::noise_maps_enabled(*this) &&
        citlali::pipeline::noise_realization_outputs_enabled(*this) &&
        mb->n_noise > 0;
    const bool coadd_product =
        citlali::pipeline::science_map_coadd_output_product(
            science_products);
    if (!coadd_product &&
        (i < 0 || i >= mb->median_err.size())) {
        mb->calc_median_err();
    }
    if (!coadd_product &&
        (i < 0 || i >= mb->median_err.size())) {
        citlali::pipeline::fail_required_output(
            logger,
            fmt::format(
                "observation map index {} lacks the required legacy median coefficient-scale diagnostic",
                static_cast<long long>(i)));
    }
    if (empirical_noise_products_expected && !coadd_product &&
        (!citlali::pipeline::has_map_image_slot(
             mb->weight_formal, i, mb->n_rows, mb->n_cols) ||
         !citlali::pipeline::has_map_image_slot(
             mb->noise_variance, i, mb->n_rows, mb->n_cols) ||
         !citlali::pipeline::has_map_image_slot(
             mb->sig2noise_pixel, i, mb->n_rows, mb->n_cols) ||
         (is_filtered_output &&
          (!citlali::pipeline::has_map_image_slot(
               mb->point_source_uncertainty, i, mb->n_rows, mb->n_cols) ||
           !citlali::pipeline::has_map_image_slot(
               mb->sig2noise_point_source, i, mb->n_rows,
               mb->n_cols))))) {
        citlali::pipeline::fail_required_output(
            logger,
            fmt::format(
                "empirical map-output inventory is incomplete for map index {}",
                static_cast<long long>(i)));
    }
    const bool write_raw_kernel =
        citlali::pipeline::raw_kernel_enabled(*this);
    using array_fwhm_type = typename std::decay_t<
        decltype(calib.array_fwhms)>::mapped_type;
    const array_fwhm_type *raw_kernel_array_fwhm = nullptr;
    if (write_raw_kernel) {
        if (!citlali::pipeline::has_map_image_slot(
                mb->kernel, i, mb->n_rows, mb->n_cols)) {
            citlali::pipeline::fail_required_output(
                logger,
                fmt::format(
                    "required kernel output is absent for map index {}",
                    static_cast<long long>(i)));
        }
        raw_kernel_array_fwhm =
            &citlali::pipeline::require_array_fwhm_for_id(
                calib.array_fwhms, array_id, logger);
    }
    const bool write_noise_realizations =
        citlali::pipeline::should_write_noise_maps(mb->noise,
                                                   noise_fits_io);
    if (noise_realization_outputs_expected || write_noise_realizations) {
        citlali::pipeline::require_noise_map_write_slots(
            mb->noise, noise_fits_io, map_index, i, logger);
        citlali::pipeline::require_noise_map_tensor_shape(
            mb->noise, i, mb->n_rows, mb->n_cols, mb->n_noise, logger);
    }
    const auto first_hdu_index =
        static_cast<Eigen::Index>(fits_io->at(map_index).hdus.size());
    struct MapOutputBreadcrumbReset {
        ~MapOutputBreadcrumbReset() {
            citlali::pipeline::reset_map_output_debug_breadcrumb();
        }
    } map_output_breadcrumb_reset;
    citlali::pipeline::update_map_output_debug_breadcrumb(
        "write-maps", fits_io->at(map_index).filepath.c_str(), i, map_index,
        stokes_index, array_id, first_hdu_index, first_hdu_index);

    const double source_epoch = typed_identity
        ? typed_identity->wcs.source_epoch
        : citlali::pipeline::wcs_source_epoch_or_default(
              telescope.tel_header, logger);

    if (typed_identity && typed_slot) {
        // One-way projection from the full-precision admitted authority into
        // the legacy float WCS adapter used by the FITS writer.
        for (std::size_t axis = 0; axis < 2; ++axis) {
            mb->wcs.ctype[axis] = typed_identity->wcs.axis_types[axis];
            mb->wcs.cunit[axis] = typed_identity->wcs.axis_units[axis];
            mb->wcs.cdelt[axis] = static_cast<float>(
                typed_identity->wcs.pixel_scale[axis]);
            mb->wcs.crval[axis] = static_cast<float>(
                typed_identity->wcs.reference_world[axis]);
            mb->wcs.crpix[axis] = static_cast<float>(
                typed_identity->wcs.reference_pixel[axis]);
        }
        mb->wcs.naxis[0] = static_cast<int>(typed_identity->cols);
        mb->wcs.naxis[1] = static_cast<int>(typed_identity->rows);
    }

    // update wcs ctypes for frequency and stokes params
    if (typed_slot) {
        mb->wcs.crval[2] = static_cast<float>(typed_slot->frequency_hz);
        mb->wcs.crval[3] = static_cast<float>(typed_slot->stokes_identity);
    }
    else {
        citlali::pipeline::assign_map_wcs_spectral_axes(
            mb->wcs, toltec_io.array_freq_map, array_id, stokes_index);
    }
    const std::string &stokes_suffix = rtcproc.polarization.stokes_params[stokes_index];

    try {
        auto add_map_hdu_with_wcs = [&](const std::string &hdu_name,
                                        auto &data) {
            citlali::pipeline::add_map_hdu_with_wcs(
                fits_io->at(map_index), hdu_name, data, mb->wcs,
                source_epoch);
        };

        citlali::pipeline::add_primary_map_image_hdus(
            fits_io->at(map_index), mb, i, map_name, stokes_suffix, mb->wcs,
            source_epoch, empirical_weight_calibration,
            empirical_noise_products_expected, is_beammap, coadd_product,
            logger,
            is_filtered_output, raw_parent_digest);

        // kernel map
        if (write_raw_kernel) {
            citlali::pipeline::add_kernel_map_image_hdu(
                fits_io->at(map_index), mb, i, map_name, stokes_suffix,
                rtcproc.kernel, *raw_kernel_array_fwhm, mb->wcs,
                source_epoch, RAD_TO_ASEC, logger);
        }

        /* coverage bool and signal-to-noise maps */
        citlali::pipeline::add_coverage_support_image_hdus(
            fits_io->at(map_index), mb, i, map_name, stokes_suffix, mb->wcs,
            source_epoch, is_filtered_output,
            empirical_noise_products_expected, coadd_product, logger);

        // write noise maps
        if (write_noise_realizations) {
            const double median_rms =
                citlali::pipeline::map_median_rms_or_zero_logged(
                    mb->median_rms, i, map_name,
                    noise_fits_io->at(map_index).filepath, logger);
            citlali::pipeline::add_noise_realization_image_hdus(
                noise_fits_io->at(map_index), mb, i, map_name, stokes_suffix,
                mb->wcs, source_epoch, median_rms);
        }
        if (mb->jinc_products.initialized) {
            if (i < 0 ||
                i >= static_cast<Eigen::Index>(
                    mb->jinc_products.formal_support.size())) {
                citlali::pipeline::fail_required_output(
                    logger,
                    "JINC product publication lacks an authoritative formal-support slot");
            }
            auto &provenance = mb->jinc_products.provenance;
            const std::string product_scope =
                fmt::format("{}_map_slot_{}",
                            is_filtered_output ? "filtered" : "raw",
                            static_cast<long long>(i));
            const std::string output_file =
                fits_io->at(map_index).filepath;
            auto join = [&](std::string product_identity,
                            std::string hdu_name,
                            const auto &plane) {
                mapmaking::record_jinc_product_join(
                    provenance,
                    mapmaking::JincProductJoin{
                        std::move(product_identity), product_scope,
                        output_file, std::move(hdu_name),
                        mapmaking::jinc_matrix_digest(plane)});
            };
            join("jinc-finalized-signal-N-over-C",
                 citlali::pipeline::signal_map_hdu_name(
                     map_name, stokes_suffix),
                 mb->signal[static_cast<std::size_t>(i)]);
            if (empirical_weight_calibration) {
                join("jinc-empirical-working-weight",
                     citlali::pipeline::weight_map_hdu_name(
                         map_name, stokes_suffix),
                     mb->weight[static_cast<std::size_t>(i)]);
                join("jinc-conditional-formal-mapmaker-weight-C2-over-Q",
                     citlali::pipeline::formal_weight_map_hdu_name(
                         map_name, stokes_suffix),
                     mb->weight_formal[static_cast<std::size_t>(i)]);
            }
            else {
                join("jinc-conditional-formal-mapmaker-weight-C2-over-Q",
                     citlali::pipeline::weight_map_hdu_name(
                         map_name, stokes_suffix),
                     mb->weight[static_cast<std::size_t>(i)]);
            }
            join("jinc-coefficient-squared-effective-integration-time",
                 citlali::pipeline::coverage_map_hdu_name(
                     map_name, stokes_suffix),
                 mb->coverage[static_cast<std::size_t>(i)]);
            join("jinc-authoritative-formal-support",
                 citlali::pipeline::coverage_mask_map_hdu_name(
                     map_name, stokes_suffix),
                 mb->jinc_products.formal_support[
                     static_cast<std::size_t>(i)]);
            if (write_raw_kernel) {
                join("jinc-processing-filtered-template-response-K-over-C",
                     citlali::pipeline::kernel_map_hdu_name(
                         map_name, stokes_suffix),
                     mb->kernel[static_cast<std::size_t>(i)]);
            }
        }
        return first_hdu_index;
    } catch (const CCfits::FitsException &e) {
        throw std::runtime_error(
            citlali::pipeline::map_write_error_message(
                map_name, i, fits_io->at(map_index).filepath, mb->noise,
                noise_fits_io, map_index, e.message()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            citlali::pipeline::map_write_error_message(
                map_name, i, fits_io->at(map_index).filepath, mb->noise,
                noise_fits_io, map_index, e.what()));
    }
}

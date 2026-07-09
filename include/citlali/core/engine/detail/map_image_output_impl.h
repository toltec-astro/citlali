#pragma once

// Engine FITS map output implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/map_image_output_helpers.h>
#include <citlali/core/pipeline/map_output_debug_breadcrumb.h>
#include <citlali/core/pipeline/output_policy.h>

template <typename fits_io_type, class map_buffer_t>
Eigen::Index Engine::write_maps(fits_io_type &fits_io, fits_io_type &noise_fits_io, map_buffer_t &mb, Eigen::Index i) {
    citlali::pipeline::require_map_data_slots(
        i, static_cast<Eigen::Index>(mb->signal.size()),
        static_cast<Eigen::Index>(mb->weight.size()), logger);

    // get name for extension layer
    std::string map_name = get_map_name(i);

    const auto write_indices =
        citlali::pipeline::map_write_indices(
            i, map_indices.arrays_to_maps, map_indices.maps_to_stokes, map_indices.maps_to_arrays);
    const Eigen::Index map_index = write_indices.map_index;
    const Eigen::Index stokes_index = write_indices.stokes_index;
    const Eigen::Index array_id = write_indices.array_id;
    citlali::pipeline::require_map_write_index_slots(
        i, map_index, static_cast<Eigen::Index>(fits_io->size()),
        stokes_index,
        static_cast<Eigen::Index>(rtcproc.polarization.stokes_params.size()),
        array_id, logger);
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

    const double source_epoch =
        citlali::pipeline::wcs_source_epoch_or_default(telescope.tel_header,
                                                       logger);

    // update wcs ctypes for frequency and stokes params
    citlali::pipeline::assign_map_wcs_spectral_axes(
        mb->wcs, toltec_io.array_freq_map, array_id, stokes_index);
    const std::string &stokes_suffix = rtcproc.polarization.stokes_params[stokes_index];

    try {
        auto add_map_hdu_with_wcs = [&](const std::string &hdu_name,
                                        auto &data) {
            citlali::pipeline::add_map_hdu_with_wcs(
                fits_io->at(map_index), hdu_name, data, mb->wcs,
                source_epoch);
        };

        const bool is_beammap =
            typed_config.runtime.reduction_type ==
            citlali::config::ReductionType::beammap;
        const bool empirical_weight_calibration =
            citlali::pipeline::empirical_weight_calibration_enabled(*this);
        citlali::pipeline::add_primary_map_image_hdus(
            fits_io->at(map_index), mb, i, map_name, stokes_suffix, mb->wcs,
            source_epoch, empirical_weight_calibration, is_beammap, logger);

        // kernel map
        if (rtcproc.run_kernel) {
            const auto &array_fwhm =
                citlali::pipeline::require_array_fwhm_for_id(
                    calib.array_fwhms, array_id, logger);
            citlali::pipeline::add_kernel_map_image_hdu(
                fits_io->at(map_index), mb, i, map_name, stokes_suffix,
                rtcproc.kernel, array_fwhm, mb->wcs,
                source_epoch, RAD_TO_ASEC, logger);
        }

        /* coverage bool and signal-to-noise maps */
        const bool is_filtered_output =
            citlali::pipeline::is_filtered_map_output(
                fits_io, map_fits_outputs.filtered_obs, map_fits_outputs.filtered_coadd);
        citlali::pipeline::add_coverage_support_image_hdus(
            fits_io->at(map_index), mb, i, map_name, stokes_suffix, mb->wcs,
            source_epoch, is_filtered_output, logger);

        // write noise maps
        if (citlali::pipeline::should_write_noise_maps(mb->noise,
                                                       noise_fits_io)) {
            citlali::pipeline::require_noise_map_write_slots(
                mb->noise, noise_fits_io, map_index, i, logger);
            const double median_rms =
                citlali::pipeline::map_median_rms_or_zero_logged(
                    mb->median_rms, i, map_name,
                    noise_fits_io->at(map_index).filepath, logger);
            citlali::pipeline::add_noise_realization_image_hdus(
                noise_fits_io->at(map_index), mb, i, map_name, stokes_suffix,
                mb->wcs, source_epoch, median_rms);
        }
        return first_hdu_index;
    } catch (const CCfits::FitsError &e) {
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

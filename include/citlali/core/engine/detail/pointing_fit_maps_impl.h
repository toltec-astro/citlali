#pragma once

// Implementation detail included by pointing.h.

#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/pointing_execution_plan.h>
#include <citlali/core/pipeline/post_processing_provenance_lifecycle.h>

void Pointing::fit_maps(citlali::pipeline::PointingFitStage stage) {
    fit_valid.setZero(map_indices.n_maps);

    if (!citlali::pipeline::pointing_config(*this).fit_gaussian) {
        logger->info("pointing Gaussian map fitting disabled");
        params.setZero(map_indices.n_maps, map_fitter.n_params);
        perrors.setZero(map_indices.n_maps, map_fitter.n_params);
        citlali::pipeline::record_pointing_fit_results(
            citlali::pipeline::pointing_plan(*this), stage, 0, 0);
        citlali::pipeline::record_post_processing_pointing_fits_completed(
            citlali::pipeline::post_processing_plan(*this),
            stage == citlali::pipeline::PointingFitStage::filtered_observation,
            0, 0);
        return;
    }

    // fit maps
    logger->info("fitting maps");
    double init_row = -99;
    double init_col = -99;

    // Run pointing fits sequentially. Parallel Ceres covariance work has shown
    // allocator instability on some systems.
    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        auto array = map_indices.maps_to_arrays(i);
        // init fwhm in pixels
        double init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/omb.pixel_size_rad;
        auto [map_params, map_perror, good_fit] =
            map_fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(omb.signal[i], omb.weight[i], init_fwhm, init_row, init_col);
        params.row(i) = map_params;
        perrors.row(i) = map_perror;

        if (good_fit) {
            fit_valid(i) = 1;
            // rescale fit params from pixel to on-sky units
            params(i,1) = RAD_TO_ASEC*omb.pixel_size_rad*(params(i,1) - (omb.n_cols - 1)/2.0);
            params(i,2) = RAD_TO_ASEC*omb.pixel_size_rad*(params(i,2) - (omb.n_rows - 1)/2.0);
            params(i,3) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params(i,3));
            params(i,4) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params(i,4));

            // rescale fit errors from pixel to on-sky units
            perrors(i,1) = RAD_TO_ASEC*omb.pixel_size_rad*(perrors(i,1));
            perrors(i,2) = RAD_TO_ASEC*omb.pixel_size_rad*(perrors(i,2));
            perrors(i,3) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors(i,3));
            perrors(i,4) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors(i,4));

            // if in radec calculate absolute pointing
            if (citlali::config::is_radec_map_pixel_axes(
                    telescope.pixel_axes)) {
                Eigen::VectorXd lat(1), lon(1);
                lat << params(i,2)*ASEC_TO_RAD;
                lon << params(i,1)*ASEC_TO_RAD;

                auto [adec, ara] = engine_utils::tangent_to_abs(lat, lon, omb.wcs.crval[0]*DEG_TO_RAD, omb.wcs.crval[1]*DEG_TO_RAD);

                params(i,1) = ara(0)*RAD_TO_DEG;
                params(i,2) = adec(0)*RAD_TO_DEG;

                perrors(i,1) = perrors(i,1)*ASEC_TO_DEG;
                perrors(i,2) = perrors(i,2)*ASEC_TO_DEG;
            }
        }
        if (citlali::pipeline::fruit_loops_config(*this)
                .diagnostics_enabled) {
            const double map_rms =
                engine_utils::calc_std_dev(omb.signal[i]);
            const double amplitude_over_map_rms =
                std::isfinite(map_rms) && map_rms > 0.0
                    ? params(i, 0) / map_rms
                    : std::numeric_limits<double>::quiet_NaN();
            logger->info(
                "fruit_loop_diag kind=final_map iteration={} stage={} "
                "array={} map={} "
                "fit_valid={} amplitude={:.17g} a_fwhm_arcsec={:.17g} "
                "b_fwhm_arcsec={:.17g} amplitude_over_map_rms={:.17g} "
                "x_t={:.17g} y_t={:.17g}",
                iteration.fruit_iter,
                citlali::pipeline::pointing_fit_stage_name(stage), array, i,
                good_fit, params(i, 0), params(i, 3), params(i, 4),
                amplitude_over_map_rms, params(i, 1), params(i, 2));
        }
    }
    citlali::pipeline::record_pointing_fit_results(
        citlali::pipeline::pointing_plan(*this), stage,
        static_cast<std::size_t>(map_indices.n_maps),
        static_cast<std::size_t>((fit_valid.array() != 0).count()));
    citlali::pipeline::record_post_processing_pointing_fits_completed(
        citlali::pipeline::post_processing_plan(*this),
        stage == citlali::pipeline::PointingFitStage::filtered_observation,
        static_cast<std::size_t>(map_indices.n_maps),
        static_cast<std::size_t>((fit_valid.array() != 0).count()));
}

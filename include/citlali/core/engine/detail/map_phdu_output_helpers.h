#pragma once

#include <citlali/core/timestream/calibration_product.h>

#include <citlali/core/pipeline/phdu_beammap.h>
#include <citlali/core/pipeline/phdu_extinction.h>
#include <citlali/core/pipeline/phdu_observation_metadata.h>
#include <citlali/core/pipeline/phdu_oof.h>
#include <citlali/core/pipeline/phdu_reduction_config.h>
#include <citlali/core/pipeline/phdu_rtc_config.h>

namespace citlali::engine_detail {

template <class FitsEntry, class MapBuffer, class Calib, class ToltecIo,
          class ArrayId, class Logger>
void add_phdu_unit_conversion_section(
    FitsEntry &fits_entry, const MapBuffer &mb, Calib &calib,
    ToltecIo &toltec_io, const ArrayId &array_id,
    const std::string &array_name, bool run_calibrate,
    double fwhm_to_std, double arcsec_to_rad, double pi_value,
    double mjy_sr_to_mjy_asec, const Logger &logger) {
    logger->debug("adding unit conversions");

    const auto unit_conversion =
        citlali::pipeline::phdu_unit_conversion_factors(
            calib.array_fwhms[array_id], mb->pixel_size_rad, fwhm_to_std,
            arcsec_to_rad, pi_value);
    auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(
        1, toltec_io.array_freq_map[array_id],
        unit_conversion.mean_fwhm_arcsec);

    citlali::pipeline::add_phdu_unit_conversion_config(
        fits_entry, array_name, logger, run_calibrate, mb->sig_unit,
        calib.array_beam_areas[array_id] * mjy_sr_to_mjy_asec,
        mJy_beam_to_uK, unit_conversion.mjy_beam_to_jy_pixel);
}

template <class FitsEntry, class MapBuffer, class FluxMap, class Calib,
          class DateObs, class Logger>
void add_phdu_beammap_observation_section(
    FitsEntry &fits_entry, const MapBuffer &mb, const std::string &array_name,
    const Logger &logger, citlali::config::ReductionType reduction_type,
    FluxMap &beammap_fluxes_mjy_beam, FluxMap &beammap_fluxes_mjy_sr,
    const citlali::config::BeammapIterationConfig &iteration_config,
    const citlali::config::BeammapPhaseStrategyConfig &phase_config,
    const citlali::config::BeammapReferenceConfig &reference_config,
    Calib &calib, const DateObs &date_obs) {
    citlali::pipeline::add_phdu_beammap_keys_if_needed(
        fits_entry, array_name, logger, reduction_type,
        beammap_fluxes_mjy_beam, beammap_fluxes_mjy_sr,
        iteration_config, phase_config, reference_config, calib);

    logger->debug("adding obsnums");
    citlali::pipeline::add_phdu_obsnum_keys(fits_entry, mb->obsnums);
    citlali::pipeline::add_phdu_date_obs_keys(
        fits_entry, mb->obsnums, date_obs);
}

template <class FitsEntry, class MapBuffer, class Telescope, class Calib,
          class Logger>
void add_phdu_identity_geometry_section(
    FitsEntry &fits_entry, const MapBuffer &mb, const Telescope &telescope,
    const Calib &calib, const std::string &array_name,
    const std::string &citlali_version, const std::string &kids_version,
    const std::string &tula_version,
    citlali::config::ReductionType reduction_type,
    citlali::config::TodType tod_type,
    citlali::config::MapGrouping map_grouping,
    citlali::config::MapMethod map_method, double rad_to_deg,
    const Logger &logger) {
    citlali::pipeline::add_phdu_pipeline_identity_keys(
        fits_entry, telescope.source_name, calib.run_hwpr, array_name,
        citlali_version, kids_version, tula_version, telescope.project_id,
        reduction_type, telescope.obs_goal, tod_type, map_grouping,
        map_method);

    const double source_ra =
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.Source.Ra", 0.0, logger);
    const double source_dec =
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.Source.Dec", 0.0, logger);
    citlali::pipeline::add_phdu_map_geometry_keys(
        fits_entry, array_name, logger, mb->exposure_time,
        telescope.pixel_axes, source_ra, source_dec,
        rad_to_deg * citlali::pipeline::telescope_data_mean(
                         telescope.tel_data, "TelElAct", 0.0, logger),
        rad_to_deg * citlali::pipeline::telescope_data_mean(
                         telescope.tel_data, "TelAzAct", 0.0, logger),
        rad_to_deg * citlali::pipeline::telescope_data_mean(
                         telescope.tel_data, "ActParAng", 0.0, logger));
}

template <class FitsEntry, class MapBuffer, class RtcProc, class Telescope,
          class Calib, class ToltecIo, class ArrayId, class Logger>
void add_phdu_extinction_apt_oof_section(
    FitsEntry &fits_entry, const MapBuffer &mb, RtcProc &rtcproc,
    const Telescope &telescope, const Calib &calib, ToltecIo &toltec_io,
    Eigen::Index map_index, const ArrayId &array_id,
    const std::string &array_name, citlali::config::ReductionType reduction_type,
    bool extinction_enabled, const Logger &logger) {
    logger->debug("adding extinction");
    const double mean_tau = citlali::pipeline::phdu_mean_tau(
        extinction_enabled, rtcproc, telescope, calib, map_index, logger);
    citlali::pipeline::add_phdu_double_key(
        fits_entry, array_name, logger, "MEAN_TAU", mean_tau,
        "mean tau (" + array_name + ")");

    auto &hdu = fits_entry.pfits->pHDU();
    const auto requested_alpha =
        rtcproc.calibration.requested_reference_spectral_index_alpha();
    hdu.addKey("CAL.ALPHA.REQUESTED_AVAILABLE",
               requested_alpha.has_value(),
               "Reference spectral index explicitly requested");
    if (requested_alpha) {
        hdu.addKey("CAL.ALPHA.REQUESTED", *requested_alpha,
                   "Requested reference spectral index alpha");
    }
    hdu.addKey(
        "CAL.ALPHA.EFFECTIVE",
        rtcproc.calibration.effective_reference_spectral_index_alpha(),
        "Effective reference spectral index alpha");
    hdu.addKey(
        "CAL.ALPHA.REALIZED",
        rtcproc.calibration.effective_reference_spectral_index_alpha(),
        "Realized reference spectral index alpha");
    hdu.addKey(
        "CAL.ALPHA.DEFAULT_APPLIED",
        rtcproc.calibration.reference_spectral_index_default_applied(),
        "Alpha zero supplied by omission default");
    hdu.addKey("CAL.OPERATOR_ID",
               std::string{rtcproc.calibration.operator_id()},
               "Atmosphere operator identity");
    hdu.addKey("CAL.OPERATOR_CONTRACT_SHA256",
               std::string{rtcproc.calibration.operator_contract_sha256()},
               "Atmosphere operator contract digest");
    hdu.addKey("CAL.NODE_TABLE_SHA256",
               std::string{rtcproc.calibration.operator_nodes_sha256()},
               "Atmosphere operator node digest");
    hdu.addKey("CAL.PASSBAND_SET_ID",
               std::string{rtcproc.calibration.passband_set_id()},
               "Passband provenance identity");
    hdu.addKey("CAL.REFERENCE_PROFILE_ID",
               std::string{rtcproc.calibration.reference_profile_id()},
               "Reference atmosphere profile identity");
    hdu.addKey("CAL.QUALITY_REGIME",
               rtcproc.calibration.calibration_quality_regime,
               "Calibration quality regime metadata");
    hdu.addKey("CAL.VALID", rtcproc.calibration.calibration_valid,
               "Calibration validity");
    hdu.addKey("CAL.VALIDITY_REASON",
               rtcproc.calibration.calibration_validity_reason,
               "Calibration validity reason");
    const auto &product = rtcproc.calibration.product;
    hdu.addKey("CAL.PRODUCT_SCHEMA", std::string{product.schema_version},
               "Complete calibration product schema");
    hdu.addKey("CAL.VALIDITY_DETAIL", product.validity_detail,
               "Calibration validity detail");
    hdu.addKey("CAL.TARGET_UNIT", product.target_unit,
               "Admitted production unit");
    hdu.addKey("CAL.PHOTOMETRY_POLICY", std::string{product.photometry_policy},
               "Admitted point-source photometry policy");
    hdu.addKey("CAL.FACTOR_COMPOSITION", std::string{product.factor_composition},
               "Applied signal factor composition");
    hdu.addKey("CAL.FACTOR_PROVENANCE", std::string{product.factor_provenance},
               "Factor units, sources, recipients, and exclusions");
    hdu.addKey("CAL.COMPATIBILITY_FCF_SEMANTICS",
               std::string{product.compatibility_fcf_semantics},
               "Compatibility fcf contents and exclusions");
    hdu.addKey("CAL.WEIGHT_RECIPIENT_SEMANTICS",
               std::string{product.weight_recipient_semantics},
               "Conditional weight recipient rules");
    hdu.addKey("CAL.COMPACT_COVARIANCE_STATE",
               std::string{product.compact_covariance_state},
               "Persisted nuisance covariance availability");
    hdu.addKey("CAL.APT_ARTIFACT_SHA256", product.apt_artifact_sha256,
               "Exact selected APT artifact digest");
    hdu.addKey("CAL.ACQUISITION_BINDING_SHA256",
               product.acquisition_binding_sha256,
               "Observation-local acquisition binding digest");
    hdu.addKey("CAL.ACQUISITION_BINDING_MODE", product.acquisition_binding_mode,
               "APT/acquisition binding mode");
    hdu.addKey("CAL.ACQUISITION_KEY_SCHEMA", product.acquisition_key_schema,
               "APT/acquisition key schema");
    hdu.addKey("CAL.RESPONSE_IDENTITY", product.response_identity,
               "Originating and realized response identity");
    hdu.addKey("CAL.CONDITIONAL_VARIANCE_TRANSFER",
               std::string{product.conditional_variance_transfer},
               "Conditional variance scaling rule");
    hdu.addKey("CAL.CONDITIONAL_INVERSE_VARIANCE_TRANSFER",
               std::string{product.conditional_inverse_variance_transfer},
               "Conditional inverse-variance scaling rule");
    hdu.addKey("CAL.PRECISION_LIMITATION",
               std::string{product.precision_limitation},
               "Limits of conditional precision products");
    hdu.addKey("CAL.NUISANCE_STATES",
               timestream::calibration_nuisance_state_summary(product),
               "Nuisance availability and correlation scopes");
    const auto minimum_total_multiplier =
        timestream::minimum_total_signal_multiplier(product);
    const auto maximum_total_multiplier =
        timestream::maximum_total_signal_multiplier(product);
    const bool total_multiplier_extrema_available =
        std::isfinite(minimum_total_multiplier) &&
        std::isfinite(maximum_total_multiplier);
    hdu.addKey("CAL.TOTAL_MULTIPLIER_EXTREMA_AVAILABLE",
               total_multiplier_extrema_available,
               "Admitted total signal multiplier extrema are available");
    if (total_multiplier_extrema_available) {
        hdu.addKey("CAL.MINIMUM_TOTAL_MULTIPLIER",
                   minimum_total_multiplier,
                   "Minimum admitted total signal multiplier");
        hdu.addKey("CAL.MAXIMUM_TOTAL_MULTIPLIER",
                   maximum_total_multiplier,
                   "Maximum admitted total signal multiplier");
    }
    const bool tau225_available =
        std::isfinite(rtcproc.calibration.realized_tau225);
    hdu.addKey("CAL.TAU225_AVAILABLE", tau225_available,
               "Realized tau225 is available");
    if (tau225_available) {
        hdu.addKey("CAL.TAU225", rtcproc.calibration.realized_tau225,
                   "Realized 225 GHz zenith-opacity request");
    }
    const bool reduction_max_tau225_available =
        std::isfinite(rtcproc.calibration.reduction_maximum_tau225);
    hdu.addKey("CAL.REDUCTION_MAX_TAU225_AVAILABLE",
               reduction_max_tau225_available,
               "Reduction maximum tau225 is available");
    if (reduction_max_tau225_available) {
        hdu.addKey("CAL.REDUCTION_MAX_TAU225",
                   rtcproc.calibration.reduction_maximum_tau225,
                   "Maximum supported tau225 in this reduction");
    }
    hdu.addKey("CAL.REDUCTION_QUALITY_REGIME",
               rtcproc.calibration.reduction_calibration_quality_regime,
               "Reduction-level calibration quality regime");
    hdu.addKey("CAL.TAU_FRAME",
               std::string{"line_of_sight_at_sample_elevation"},
               "Optical-depth coordinate frame");
    hdu.addKey("CAL.X_REF", 0.0,
               "Top-of-atmosphere calibration airmass pivot");

    citlali::pipeline::add_phdu_apt_key_if_single_observation(
        fits_entry, mb->obsnums, calib.apt_filepath, logger);

    const double rms = citlali::pipeline::phdu_oof_rms(
        mb, map_index, reduction_type, array_name, fits_entry.filepath,
        logger);

    citlali::pipeline::add_phdu_oof_keys_if_observed(
        fits_entry, array_name, logger, telescope.sim_obs, rms,
        mb->sig_unit, toltec_io.array_wavelength_map[array_id] / 1000.,
        static_cast<int>(toltec_io.array_wavelength_map[array_id] * 1000),
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.M2.XReq", 0.0, logger) /
            1000. * 1e6,
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.M2.YReq", 0.0, logger) /
            1000. * 1e6,
        citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, "Header.M2.ZReq", 0.0, logger) /
            1000. * 1e6);
}

template <class FitsEntry, class RawTimeChunkConfig,
          class ProcessedTimeChunkConfig, class RtcProc,
          class OuterContext, class Logger>
void add_phdu_tod_runtime_config_section(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, bool verbose_mode, bool polarimetry_enabled,
    const RawTimeChunkConfig &raw_config,
    const ProcessedTimeChunkConfig &processed_config,
    const RtcProc &rtcproc,
    OuterContext outer_context_samples) {
    logger->debug("adding config params");
    const bool run_any_tod_filter =
        citlali::config::raw_time_chunk_filtering_active(raw_config);
    citlali::pipeline::add_phdu_initial_runtime_config(
        fits_entry, verbose_mode, polarimetry_enabled,
        raw_config.despike.enabled);
    citlali::pipeline::add_phdu_rtc_local_despike_config(
        fits_entry, array_name, logger, rtcproc.despiker.local_residual);
    citlali::pipeline::add_phdu_tod_filter_config(
        fits_entry, array_name, logger, raw_config, run_any_tod_filter);
    citlali::pipeline::add_phdu_tod_edge_guard_config(
        fits_entry, rtcproc.filter_edge_guard, outer_context_samples);
    citlali::pipeline::add_phdu_tod_processing_config(
        fits_entry, raw_config);
    citlali::pipeline::add_phdu_weight_selection_config(
        fits_entry, array_name, logger, raw_config.flagging,
        processed_config);
    citlali::pipeline::add_phdu_rtc_event_mask_config(
        fits_entry, array_name, logger, raw_config.flagging);
}

template <class FitsEntry, class PtcProc, class Calib, class LearningState,
          class ArrayId, class Logger>
void add_phdu_ptc_learning_config_section(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, PtcProc &ptcproc, const Calib &calib,
    const LearningState &reduction_learning, Eigen::Index map_index,
    const ArrayId &array_id, const std::string &signal_unit,
    const citlali::config::ProcessedTimeChunkConfig &processed_config,
    const citlali::config::TimestreamFruitLoopsConfig &fruit_config,
    const citlali::config::PointingConfig &pointing_config,
    citlali::config::ReductionType reduction_type) {
    citlali::pipeline::add_phdu_reduction_learning_config(
        fits_entry, array_name, logger, reduction_learning);
    citlali::pipeline::add_phdu_weight_corr_penalty_config(
        fits_entry, array_name, logger,
        processed_config.weighting.corr_penalty);
    citlali::pipeline::add_phdu_busy_row_suppression_config(
        fits_entry, array_name, logger,
        processed_config.weighting.busy_row_suppression);
    const auto n_eig_removed =
        processed_config.clean.enabled
            ? ptcproc.cleaner.n_eig_to_cut[array_id].sum()
            : 0;
    citlali::pipeline::add_phdu_cleaner_config(
        fits_entry, array_name, logger, processed_config.clean,
        n_eig_removed);

    const double fruit_loops_flux_limit =
        citlali::pipeline::phdu_fruit_loop_flux_limit(
            fruit_config, calib.arrays, map_index, array_id);
    citlali::pipeline::add_phdu_fruit_loops_config(
        fits_entry, array_name, logger, fruit_config, pointing_config,
        reduction_type, fruit_loops_flux_limit, signal_unit);
}

template <class FitsEntry, class MapBuffer, class Telescope, class Logger>
void add_phdu_pointing_telescope_header_section(
    FitsEntry &fits_entry, const MapBuffer &mb, const Telescope &telescope,
    const std::string &array_name, const Logger &logger,
    citlali::config::ReductionType reduction_type,
    const citlali::config::PointingConfig &pointing_config) {
    citlali::pipeline::add_phdu_pointing_config_if_needed(
        fits_entry, array_name, logger, reduction_type, pointing_config);

    citlali::pipeline::add_phdu_telescope_header_keys_if_single_observation(
        fits_entry, mb->obsnums, array_name, logger, telescope.tel_header);
}

}  // namespace citlali::engine_detail

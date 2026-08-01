#pragma once

// Engine member-function implementations split from engine.h.
// Include this only after Engine has been declared.

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/config_value.h>
#include <citlali/core/pipeline/observation_setup_validation.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/pipeline/raw_timestream_observation_shadow.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/science_map_identity.h>
#include <citlali/core/pipeline/stage_profile.h>
#include <citlali/core/pipeline/timestream_output_provenance.h>

#include <stdexcept>

namespace citlali::engine_detail {

template <class EngineT>
void setup_observation_extinction(EngineT &engine) {
    if (citlali::pipeline::raw_extinction_correction_enabled(engine)) {
        // get atm model
        engine.rtcproc.calibration.setup(engine.telescope.tau_225_GHz);

        engine.logger->info(
            "using {} model for extinction correction",
            engine.rtcproc.calibration.extinction_model);

        // check tau (may be unnecessary now)
        if (!engine.telescope.sim_obs) {
            Eigen::VectorXd tau_el(1);
            // get mean elevation
            tau_el << engine.telescope.tel_data["TelElAct"].mean();
            // get tau at mean elevation for each band
            auto tau_freq = engine.rtcproc.calibration.calc_tau(
                tau_el, engine.telescope.tau_225_GHz);
            // loop through and make sure average tau is not negative (implies wrong model)
            for (auto const& [key, val] : tau_freq) {
                citlali::pipeline::require_nonnegative_extinction_tau(
                    val[0], engine.toltec_io.array_name_map[key]);
            }
        }
    }
    else {
        engine.rtcproc.calibration.extinction_model = "N/A";
    }

    if constexpr (citlali::pipeline::has_raw_timestream_plan_v<EngineT>) {
        auto &plan = citlali::pipeline::raw_timestream_plan(engine);
        if (plan.initialized) {
            const auto shadow =
                citlali::pipeline::complete_raw_timestream_extinction_shadow(
                    plan, engine.telescope.tau_225_GHz,
                    engine.rtcproc.calibration.tx_225_zenith,
                    engine.rtcproc.run_extinction,
                    engine.rtcproc.calibration.extinction_model);
            if (!shadow.exact) {
                engine.logger->error(
                    "typed raw extinction shadow differs from legacy state: {}",
                    shadow.diagnostic());
                throw std::runtime_error(
                    "typed raw extinction shadow parity failure");
            }
        }
    }
}

template <class EngineT>
void validate_observation_polarization_inputs(EngineT &engine) {
    // make sure there are matched fg's in apt if reducing in polarized mode
    if (engine.rtcproc.run_polarization) {
        citlali::pipeline::require_polarization_frequency_groups(
            (engine.calib.apt["fg"].array() == -1).all());
    }
}

template <class EngineT>
void setup_observation_timestream_processors(EngineT &engine) {
    // setup kernel
    if (citlali::pipeline::raw_kernel_enabled(engine)) {
        engine.rtcproc.kernel.setup(engine.map_indices.n_maps);
    }

    // set despiker sample rate
    engine.rtcproc.despiker.fsmp = engine.telescope.fsmp;
    // set processed timestream sample rate for optional adaptive PTC mode selection
    engine.ptcproc.cleaner.sample_rate_Hz = engine.telescope.d_fsmp;

    // if filter is requested, make it here
    if (citlali::pipeline::raw_fir_filter_enabled(engine)) {
        engine.rtcproc.filter.make_filter(engine.telescope.fsmp);
        if (citlali::pipeline::raw_notch_filter_enabled(engine)) {
            engine.rtcproc.filter.make_notch_filter(engine.telescope.fsmp);
        }
    }
    if (citlali::pipeline::raw_iir_filter_enabled(engine)) {
        const double nyquist_Hz = engine.telescope.fsmp / 2.0;
        citlali::pipeline::require_iir_below_nyquist_hz(
            citlali::pipeline::raw_iir_filter_below_nyquist(engine),
            citlali::pipeline::raw_iir_filter_frequency_hz(engine),
            nyquist_Hz);
    }
}

template <class EngineT>
void setup_observation_map_wcs(EngineT &engine) {
    const bool retain_legacy_coadd_metadata =
        citlali::pipeline::coadd_outputs_enabled(engine) &&
        !citlali::pipeline::science_map_v1_profile_available(engine);
    // set map wcs crvals to source ra/dec
    if (citlali::config::is_radec_map_pixel_axes(engine.telescope.pixel_axes)) {
        engine.omb.wcs.crval[0] =
            engine.telescope.tel_header["Header.Source.Ra"](0)*RAD_TO_DEG;
        engine.omb.wcs.crval[1] =
            engine.telescope.tel_header["Header.Source.Dec"](0)*RAD_TO_DEG;

        if (retain_legacy_coadd_metadata) {
            engine.cmb.wcs.crval[0] = engine.omb.wcs.crval[0];
            engine.cmb.wcs.crval[1] = engine.omb.wcs.crval[1];
        }

    }

    // set map wcs crvals to source l/b
    else if (citlali::config::is_galactic_map_pixel_axes(
                 engine.telescope.pixel_axes)) {
        engine.omb.wcs.crval[0] =
            engine.telescope.tel_header["Header.Source.L"](0)*RAD_TO_DEG;
        engine.omb.wcs.crval[1] =
            engine.telescope.tel_header["Header.Source.B"](0)*RAD_TO_DEG;

        if (retain_legacy_coadd_metadata) {
            engine.cmb.wcs.crval[0] = engine.omb.wcs.crval[0];
            engine.cmb.wcs.crval[1] = engine.omb.wcs.crval[1];
        }

    }

    citlali::pipeline::configure_observation_science_map_identity(engine);
}

template <class EngineT>
void setup_observation_tod_output_files(
    EngineT &engine, citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    {
        auto profile_scope = citlali::pipeline::profile_stage(stage_profile,
            "observation.setup.tod_output_selection", engine.logger);
        engine.setup_tod_output_chunk_selection();
    }
    const auto &tod_output_subdir_name =
        citlali::pipeline::timestream_config(engine).output.subdir_name;
    // create output subdirectory if requested
    if (citlali::config::has_config_value(tod_output_subdir_name)) {
        auto profile_scope = citlali::pipeline::profile_stage(stage_profile,
            "observation.setup.tod_output_directory", engine.logger);
        fs::create_directories(
            engine.output_paths.obsnum_dir_name + "raw/" + tod_output_subdir_name);
    }
    // create timestream files
    if (citlali::pipeline::tod_output_enabled(engine)) {
        // make rtc tod output file
        if (citlali::pipeline::raw_tod_output_enabled(engine)) {
            auto profile_scope = citlali::pipeline::profile_stage(stage_profile,
                "observation.setup.create_rtc_tod_file", engine.logger);
            engine.template create_tod_files<
                engine_utils::toltecIO::rtc_timestream>();
        }
        // make ptc tod output file
        if (citlali::pipeline::processed_tod_output_enabled(engine)) {
            auto profile_scope = citlali::pipeline::profile_stage(stage_profile,
                "observation.setup.create_ptc_tod_file", engine.logger);
            engine.template create_tod_files<
                engine_utils::toltecIO::ptc_timestream>();
        }
    }
    // don't calculate any eigenvalues
    else if (!engine.diagnostics.write_evals) {
        engine.ptcproc.cleaner.n_calc = 0;
    }

    citlali::pipeline::write_timestream_output_provenance_file(engine);
    engine.logger->info(
        "timestream output provenance sidecar: {}",
        citlali::pipeline::timestream_output_provenance_path(
            engine.output_paths.obsnum_dir_name)
            .string());
}

template <class EngineT>
void create_observation_diagnostic_files(
    EngineT &engine, citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    {
        auto profile_scope = citlali::pipeline::profile_stage(stage_profile,
            "observation.setup.create_rtcdiag_file", engine.logger);
        engine.create_rtcdiag_file();
    }
    {
        auto profile_scope = citlali::pipeline::profile_stage(stage_profile,
            "observation.setup.create_ptcdiag_file", engine.logger);
        engine.create_ptcdiag_file();
    }
}

template <class EngineT>
void log_observation_cli_summary(
    EngineT &engine, citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    // output basic info for obs reduction to command line
    {
        auto profile_scope = citlali::pipeline::profile_stage(stage_profile,
            "observation.setup.cli_summary", engine.logger);
        engine.cli_summary();
    }
}

template <class EngineT>
void setup_observation_stats_buffers(
    EngineT &engine, citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    // set up per-det stats file values
    {
        auto profile_scope = citlali::pipeline::profile_stage(stage_profile,
            "observation.setup.stats_buffers", engine.logger);
        for (const auto &stat: engine.diagnostics.det_stats_header) {
            engine.diagnostics.stats[stat].setZero(
                engine.calib.n_dets, engine.telescope.scan_indices.cols());
        }
        // set up per-group stats file values
        for (const auto &stat: engine.diagnostics.grp_stats_header) {
            engine.diagnostics.stats[stat].setZero(
                engine.calib.n_arrays, engine.telescope.scan_indices.cols());
        }
    }
    // clear stored eigenvalues
    std::map<Eigen::Index, std::vector<std::vector<Eigen::VectorXd>>>().swap(
        engine.diagnostics.evals);
}

}  // namespace citlali::engine_detail

void Engine::obsnum_setup(
    citlali::pipeline::StageProfileCollector &stage_profile) {
    citlali::engine_detail::setup_observation_extinction(*this);
    citlali::engine_detail::validate_observation_polarization_inputs(*this);
    citlali::engine_detail::setup_observation_timestream_processors(*this);
    citlali::engine_detail::setup_observation_map_wcs(*this);
    citlali::engine_detail::setup_observation_tod_output_files(
        *this, stage_profile);
    citlali::engine_detail::create_observation_diagnostic_files(
        *this, stage_profile);
    citlali::engine_detail::log_observation_cli_summary(
        *this, stage_profile);
    citlali::engine_detail::setup_observation_stats_buffers(
        *this, stage_profile);
}

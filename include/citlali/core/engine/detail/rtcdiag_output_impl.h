#pragma once

// Engine diagnostic output implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/obsnum_format.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <regex>
#include <stdexcept>

void Engine::create_rtcdiag_file() {
    const auto rtcdiag_final_path =
        citlali::pipeline::diagnostic_output_netcdf_filename<
            engine_utils::toltecIO::toltec,
            engine_utils::toltecIO::rtcdiag,
            engine_utils::toltecIO::raw>(
            toltec_io, output_paths.obsnum_dir_name,
            citlali::pipeline::timestream_config(*this).output.subdir_name,
            citlali::pipeline::runtime_reduction_type(*this),
            observation_identity.obsnum, telescope.sim_obs);

    output_paths.rtcdiag_filename = write_netcdf_staging(
        rtcdiag_final_path, [&](netCDF::NcFile &fo) {

    const int fill_int = citlali::pipeline::rtcdiag_fill_int();
    const double fill_double = citlali::pipeline::rtcdiag_fill_double();
    const Eigen::Index n_scans = telescope.scan_indices.cols();
    const double rtc_fsmp =
        citlali::pipeline::rtc_tod_stream_sample_rate(
            citlali::pipeline::raw_time_chunk_config(*this)
                .downsample.enabled,
            telescope.fsmp, telescope.d_fsmp);
    const auto &raw_plan = citlali::pipeline::raw_timestream_plan(*this);
    const auto &realized_rtc = this->rtcproc;
    // Supported diagnostics are explicitly total-intensity. They do not
    // inspect legacy HWPR flags, enum spellings, files, or hardware presence.
    const citlali::pipeline::RtcSamplingHwprState rtc_sampling_hwpr{
        citlali::pipeline::RtcSamplingHwprState::AnalysisMode::total_intensity};
    citlali::pipeline::RtcSamplingCadenceState rtc_sampling_cadence;
    rtc_sampling_cadence.requested_factor =
        raw_plan.requested.downsample.enabled
            ? raw_plan.requested.downsample.factor : 1;
    rtc_sampling_cadence.requested_output_hz =
        raw_plan.requested.downsample.enabled &&
                raw_plan.requested.downsample.factor <= 0
            ? raw_plan.requested.downsample.downsampled_freq_Hz
            : telescope.fsmp /
                  std::max(1, rtc_sampling_cadence.requested_factor);
    if (raw_plan.observation.has_value()) {
        const auto &observation = *raw_plan.observation;
        rtc_sampling_cadence.effective_native_hz =
            observation.native_sample_rate_hz.value_or(
                std::numeric_limits<double>::quiet_NaN());
        rtc_sampling_cadence.effective_output_hz =
            observation.effective_sample_rate_hz.value_or(
                std::numeric_limits<double>::quiet_NaN());
        rtc_sampling_cadence.effective_factor =
            observation.downsample_factor.value_or(1);
    }
    rtc_sampling_cadence.requested_factor =
        citlali::pipeline::rtc_sampling_requested_factor(
            rtc_sampling_cadence.effective_native_hz,
            rtc_sampling_cadence.requested_factor,
            rtc_sampling_cadence.requested_output_hz);
    citlali::pipeline::update_rtc_sampling_requested_effective_consistency(
        rtc_sampling_cadence);
    rtc_sampling_cadence.realized_downsample_enabled =
        realized_rtc.run_downsample;
    rtc_sampling_cadence.realized_factor =
        rtc_sampling_cadence.realized_downsample_enabled
            ? realized_rtc.downsampler.factor : 1;
    citlali::pipeline::RtcSamplingFilterState rtc_sampling_filter;
    rtc_sampling_filter.requested_enabled = raw_plan.requested.filter.enabled;
    rtc_sampling_filter.effective_enabled = raw_plan.effective.filter.enabled;
    rtc_sampling_filter.realized_enabled = realized_rtc.run_tod_filter;
    rtc_sampling_filter.requested_a_gibbs = raw_plan.requested.filter.a_gibbs;
    rtc_sampling_filter.effective_a_gibbs = raw_plan.effective.filter.a_gibbs;
    rtc_sampling_filter.requested_low_hz = raw_plan.requested.filter.freq_low_Hz;
    rtc_sampling_filter.effective_low_hz = raw_plan.effective.filter.freq_low_Hz;
    rtc_sampling_filter.requested_high_hz = raw_plan.requested.filter.freq_high_Hz;
    rtc_sampling_filter.effective_high_hz = raw_plan.effective.filter.freq_high_Hz;
    rtc_sampling_filter.requested_n_terms = raw_plan.requested.filter.n_terms;
    rtc_sampling_filter.effective_n_terms = raw_plan.effective.filter.n_terms;
    if (rtc_sampling_filter.realized_enabled) {
        rtc_sampling_filter.realized_a_gibbs = realized_rtc.filter.a_gibbs;
        rtc_sampling_filter.realized_low_hz = realized_rtc.filter.freq_low_Hz;
        rtc_sampling_filter.realized_high_hz = realized_rtc.filter.freq_high_Hz;
        rtc_sampling_filter.realized_n_terms = realized_rtc.filter.n_terms;
        if (realized_rtc.filter.filter.size() > 0) {
            rtc_sampling_filter.realized_coefficients.assign(
                realized_rtc.filter.filter.data(),
                realized_rtc.filter.filter.data() +
                    realized_rtc.filter.filter.size());
        }
    }

    const auto &source_support = alignment.rtc_sampling_source_motion;
    if (!citlali::pipeline::rtc_sampling_source_matches_active_observation(
            alignment) ||
        source_support.observation_obsnum != observation_identity.obsnum) {
        throw std::runtime_error(
            "rtcdiag Stage A source motion is not bound to the active observation");
    }
    if constexpr (citlali::pipeline::has_astrometry_plan_v<Engine>) {
        const auto &plan = citlali::pipeline::astrometry_plan(*this);
        if (!plan.active_observation_index ||
            *plan.active_observation_index != source_support.observation_index ||
            source_support.observation_index >= plan.observations.size() ||
            citlali::pipeline::format_obsnum(
                plan.observations[source_support.observation_index].obsnum) !=
                source_support.observation_obsnum) {
            throw std::runtime_error(
                "rtcdiag Stage A source motion differs from the production observation index or obsnum");
        }
    }
    citlali::pipeline::measure_rtc_sampling_realized_cadence_from_source(
        rtc_sampling_cadence, source_support);
    const std::string full_commit{CITLALI_GIT_COMMIT_FULL};
    if (!std::regex_match(full_commit, std::regex{"[0-9a-f]{40}"}) ||
        CITLALI_GIT_WORKTREE_DIRTY != 0) {
        throw std::runtime_error(
            "rtcdiag successor requires a clean exact 40-hex Citlali source identity");
    }

    citlali::pipeline::add_diagnostic_file_identity_vars(
        fo, "rtcdiag", std::stoi(observation_identity.obsnum),
        telescope.tel_header["Header.Source.Ra"](0),
        telescope.tel_header["Header.Source.Dec"](0));

    const auto rtcdiag_dims =
        citlali::pipeline::add_rtcdiag_dims(
            fo, n_scans, calib.n_dets, calib.n_arrays, calib.n_nws);

    citlali::pipeline::add_diagnostic_output_scan_index(
        fo, rtcdiag_dims.n_scans, n_scans, fill_int);

    citlali::pipeline::add_rtcdiag_array_ids(
        fo, calib, rtcdiag_dims.n_arrays, fill_int);

    const auto scan_summary =
        citlali::pipeline::calculate_rtcdiag_scan_summary(
            telescope, source_support,
            rtc_sampling_hwpr, n_scans, rtcdiag_dims.n_scan_values,
            fill_double, fill_int, logger);
    citlali::pipeline::add_rtcdiag_scan_summary_outputs(
        fo, rtcdiag_dims.n_scans, rtcdiag_dims.scan_chunks, scan_summary);

    const auto scan_array_summary =
        citlali::pipeline::calculate_rtcdiag_scan_array_summary(
            calib, rtc_sampling_filter, telescope, scan_summary,
            rtc_sampling_hwpr, rtc_sampling_cadence, n_scans,
            rtcdiag_dims.n_array_values,
            rtcdiag_dims.n_scan_array_values, fill_double, fill_int,
            source_support.intervals.size());
    citlali::pipeline::add_rtcdiag_scan_array_summary_outputs(
        fo, rtcdiag_dims.scan_array, rtcdiag_dims.scan_array_chunks,
        scan_array_summary, rtc_sampling_hwpr, rtc_sampling_cadence,
        source_support,
        std::string{citlali::pipeline::rtcdiag_raw_input_manifest_reference},
        full_commit);

    citlali::pipeline::add_rtcdiag_network_ids(
        fo, calib, rtcdiag_dims.n_nws, fill_int);

    citlali::pipeline::add_pipeline_identity_vars(
        fo, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
        telescope.project_id,
        citlali::pipeline::runtime_reduction_type(*this),
        telescope.obs_goal, citlali::pipeline::timestream_config(*this).type);
    add_netcdf_var(fo, "SAMPRATE", telescope.fsmp);
    citlali::pipeline::add_rtcdiag_file_config_vars(
        fo, rtcproc, citlali::pipeline::raw_time_chunk_config(*this),
        learning,
        citlali::pipeline::verbose_runtime_enabled(*this),
        telescope.outer_scans_chunk,
        citlali::pipeline::raw_tod_outer_context_samples(*this), rtc_fsmp);

    citlali::pipeline::add_rtcdiag_apt_double_vars(
        fo, calib, rtcdiag_dims.n_dets);

    citlali::pipeline::add_rtcdiag_standard_detector_outputs(
        fo, rtcdiag_dims.det, rtcdiag_dims.det_chunks,
        rtcdiag_dims.n_det_values, fill_int, fill_double);

    citlali::pipeline::add_rtcdiag_standard_network_outputs(
        fo, rtcdiag_dims.nw, rtcdiag_dims.nw_chunks,
        rtcdiag_dims.n_nw_values, fill_int, fill_double);

    citlali::pipeline::add_rtcdiag_impulsive_capture_file_outputs_if_needed(
        fo,
        citlali::pipeline::raw_time_chunk_config(*this)
            .flagging.impulsive_capture,
        rtcdiag_dims.n_scans,
        rtcdiag_dims.n_nws, n_scans, calib.n_nws, rtc_fsmp, fill_int,
        fill_double);

    });
}

#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/beammap_mapmaking_policy.h>
#include <citlali/core/pipeline/beammap_normalize_support_logging.h>
#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <sstream>

bool Beammap::subtract_beammap_model_for_ptc_scan(int scan_index,
                                                  bool measurement_iter) {
    if (!run_mapmaking || !measurement_iter) {
        return false;
    }
    if (!ptcproc.run_fruit_loops) {
        logger->info("subtracting gaussian from tod");
        ptcproc.add_gaussian<timestream::TCProc::SourceType::NegativeGaussian>(
            ptcs[scan_index], params, telescope.pixel_axes, map_grouping,
            calib.apt, omb.pixel_size_rad, omb.n_rows, omb.n_cols);
        return false;
    }

    logger->info("subtracting map from tod");
    ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(
        omb, ptcs[scan_index], calib, ptcs[scan_index].map_indices.data,
        telescope.pixel_axes, map_grouping);
    return true;
}

void Beammap::restore_beammap_model_for_ptc_scan(int scan_index,
                                                 bool measurement_iter) {
    if (!run_mapmaking || !measurement_iter) {
        return;
    }
    if (!ptcproc.run_fruit_loops) {
        logger->info("adding gaussian to tod");
        ptcproc.add_gaussian<timestream::TCProc::SourceType::Gaussian>(
            ptcs[scan_index], params, telescope.pixel_axes, map_grouping,
            calib.apt, omb.pixel_size_rad, omb.n_rows, omb.n_cols);
        return;
    }

    logger->info("adding map to tod");
    ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(
        omb, ptcs[scan_index], calib, ptcs[scan_index].map_indices.data,
        telescope.pixel_axes, map_grouping);
}

void Beammap::remove_bad_beammap_dets_for_scan(int scan_index,
                                               bool locator_iter,
                                               bool detector_grouping) {
    // For detector-grouped beammaps, keep the locator pass permissive so
    // bright-source scans are less likely to be rejected before we have
    // any source-location estimate to feed back into later iterations.
    if (detector_grouping && locator_iter) {
        logger->info(
            "skipping remove_bad_dets on beammap locator iter {} for detector scan {}",
            current_iter, ptcs[scan_index].index.data + 1);
        return;
    }

    calib_scans[scan_index] = ptcproc.remove_bad_dets(
        ptcs[scan_index], calib_scans[scan_index], map_grouping);
}

void Beammap::apply_beammap_ptc_scan_weights(int scan_index,
                                             bool measurement_iter,
                                             bool detector_grouping) {
    if (detector_grouping) {
        auto rfi_summary = apply_rfi_sample_mask(ptcs[scan_index]);
        if (beammap_rfi_mask_enabled) {
            if (rfi_summary.n_samples_flagged > 0 ||
                rfi_summary.n_det_rejected > 0) {
                logger->info(
                    "beammap rfi mask scan {}: masked {} samples across {}/{} detectors ({} rejected by max_flagged_fraction={:.4f})",
                    ptcs[scan_index].index.data + 1,
                    rfi_summary.n_samples_flagged,
                    rfi_summary.n_det_flagged,
                    rfi_summary.n_det_candidates,
                    rfi_summary.n_det_rejected,
                    beammap_rfi_mask_max_flagged_fraction);
            }
            else {
                logger->debug(
                    "beammap rfi mask scan {}: no samples masked",
                    ptcs[scan_index].index.data + 1);
            }
        }

        const bool use_ptc_weights =
            citlali::pipeline::use_beammap_detector_ptc_weights(
                beammap_detector_weighting_mode, measurement_iter);
        if (use_ptc_weights) {
            logger->info(
                "calculating detector-mode PTC weights for scan {} (mode={})",
                ptcs[scan_index].index.data + 1, beammap_detector_weighting_mode);
            ptcproc.calc_weights(ptcs[scan_index], calib_scans[scan_index].apt, telescope);
            calib_scans[scan_index] = ptcproc.reset_weights(
                ptcs[scan_index], calib_scans[scan_index], map_grouping);
        }
        else {
            // Constant weights remain the safest default for bright beammaps.
            ptcs[scan_index].weights.data.resize(ptcs[scan_index].scans.data.cols());
            ptcs[scan_index].weights.data.setOnes();
        }
        return;
    }

    logger->info("calculating weights for scan {}", ptcs[scan_index].index.data + 1);
    ptcproc.calc_weights(ptcs[scan_index], calib_scans[scan_index].apt, telescope);
    calib_scans[scan_index] = ptcproc.reset_weights(
        ptcs[scan_index], calib_scans[scan_index], map_grouping);
}

void Beammap::process_beammap_ptc_scan(
    int scan_index, bool locator_iter, bool measurement_iter,
    bool detector_grouping,
    const std::shared_ptr<std::mutex> &ptc_line_audit_mutex) {
    const bool model_subtracted_for_ptc_line_audit =
        subtract_beammap_model_for_ptc_scan(scan_index, measurement_iter);

    {
        std::lock_guard<std::mutex> lock(*ptc_line_audit_mutex);
        apply_model_protected_ptc_line_audit(
            ptcs[scan_index], calib_scans[scan_index],
            model_subtracted_for_ptc_line_audit);
    }

    logger->info("processed time chunk processing for scan {}", scan_index + 1);
    ptcproc.run(
        ptcs[scan_index], ptcs[scan_index], calib_scans[scan_index],
        telescope.pixel_axes, map_grouping);

    restore_beammap_model_for_ptc_scan(scan_index, measurement_iter);
    remove_bad_beammap_dets_for_scan(scan_index, locator_iter, detector_grouping);
    apply_beammap_ptc_scan_weights(scan_index, measurement_iter, detector_grouping);

    logger->debug("calculating stats");
    diagnostics.calc_stats(ptcs[scan_index]);
}

void Beammap::run_beammap_ptc_cleaning_pass(bool locator_iter,
                                            bool measurement_iter,
                                            bool detector_grouping) {
    auto ptc_line_audit_mutex = std::make_shared<std::mutex>();

    const auto profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.ptc.cleaning", logger,
            "iter=" + std::to_string(current_iter) +
                " phase=" + beammap_iter_phase_name(current_iter));
    grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
        process_beammap_ptc_scan(
            i, locator_iter, measurement_iter, detector_grouping,
            ptc_line_audit_mutex);
        return 0;
    });
}

void Beammap::populate_beammap_maps(
    citlali::config::MapGrouping mapmaking_grouping,
    citlali::config::MapMethod mapmaking_method,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps,
    bool update_progress) {
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100,
        "PTC progress ");

    if (mapmaking_grouping == citlali::config::MapGrouping::detector) {
        bool run_omb = true;
        for (std::size_t scan_vec_idx = 0; scan_vec_idx < ptcs.size(); ++scan_vec_idx) {
            auto &ptc = ptcs[scan_vec_idx];
            auto &scan_apt = calib_scans[scan_vec_idx].apt;
            if (mapmaking_method == citlali::config::MapMethod::naive) {
                naive_mm.populate_maps_naive_parallel(
                    ptc, omb, cmb, ptc.map_indices.data, telescope.pixel_axes,
                    scan_apt, telescope.d_fsmp, run_omb, run_noise,
                    active_maps);
            }
            else if (mapmaking_method == citlali::config::MapMethod::jinc) {
                citlali::pipeline::log_beammap_jinc_preflight(
                    ptc, calib.apt["array"], omb, jinc_mm, logger);
                jinc_mm.populate_maps_jinc_parallel(
                    ptc, omb, cmb, ptc.map_indices.data, telescope.pixel_axes,
                    scan_apt, telescope.d_fsmp, run_omb, run_noise,
                    active_maps);
            }
            if (update_progress) {
                pb.count(telescope.scan_indices.cols(), 1);
            }
        }
        return;
    }

    grppi::map(tula::grppi_utils::dyn_ex(map_parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
        bool run_omb = true;
        citlali::pipeline::populate_naive_or_jinc_maps(
            mapmaking_method, naive_mm, jinc_mm, ptcs[i], omb, cmb,
            ptcs[i].map_indices.data, telescope.pixel_axes,
            calib_scans[i].apt, telescope.d_fsmp, run_omb, run_noise);
        if (update_progress) {
            pb.count(telescope.scan_indices.cols(), 1);
        }
        return 0;
    });
}

void Beammap::prepare_beammap_iteration_state(bool rerun_source_aware_rtc,
                                              bool measurement_iter,
                                              bool first_measurement_iter,
                                              bool detector_grouping) {
    ptcs = ptcs0;
    calib_scans = calib_scans0;

    if (beammap_rfi_mask_enabled && detector_grouping &&
        rfi_mask_samples_flagged.size() == calib.n_dets &&
        rfi_mask_scans_flagged.size() == calib.n_dets) {
        rfi_mask_samples_flagged.setZero();
        rfi_mask_scans_flagged.setZero();
    }

    const bool skip_centered_kernel_map_feedback = rerun_source_aware_rtc;
    ptcproc.fruit_loops_kernel_feedback_enabled = !skip_centered_kernel_map_feedback;
    if (skip_centered_kernel_map_feedback) {
        logger->info(
            "beammap detector kernel map feedback disabled on iter {} while building the first source-aware kernel map",
            current_iter);
    }

    // copy previous-iteration maps for source-aperture convergence tests
    if (run_mapmaking && beammap_iter_tolerance > 0.0 && measurement_iter) {
        omb_copy.signal = omb.signal;
        omb_copy.weight = omb.weight;
    }

    if (ptcproc.run_fruit_loops) {
        if (first_measurement_iter && !omb.noise.empty()) {
            omb.calc_median_rms();
        }
        if (measurement_iter) {
            ptcproc.configure_fruit_loops_adaptive_gate(omb, calib, map_grouping, false);
        }
    }
}

template <class RandomBits, class Generator>
void Beammap::run_beammap_mapmaking_pass(bool update_progress,
                                         RandomBits &rands,
                                         Generator &eng) {
    const auto mapmaking_grouping = typed_config.mapmaking.grouping;
    const auto mapmaking_method = typed_config.mapmaking.method;
    const auto active_maps =
        citlali::pipeline::select_unconverged_beammap_maps(
            mapmaking_grouping, converged, n_maps, logger);
    const auto *active_maps_ptr = active_maps.ptr();

    std::ostringstream context;
    context << "iter=" << current_iter
            << " phase=" << beammap_iter_phase_name(current_iter)
            << " update_progress=" << (update_progress ? 1 : 0)
            << " grouping=" << static_cast<int>(mapmaking_grouping)
            << " method=" << static_cast<int>(mapmaking_method)
            << " active_maps=" << active_maps.n_active_maps << "/" << n_maps;
    const auto profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.mapmaking.pass", logger, context.str());

    {
        const auto reset_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.mapmaking.reset_buffers", logger, context.str());
        citlali::pipeline::ensure_jinc_grid_weight_maps(
            mapmaking_method, omb, n_maps, logger);

        citlali::pipeline::reset_beammap_mapmaking_buffers(
            omb, ptcs, n_maps, rtcproc.run_kernel, run_noise,
            omb.randomize_dets, calib.n_dets, active_maps_ptr, rands,
            eng);
    }

    logger->info("running mapmaking");

    {
        const auto populate_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.mapmaking.populate", logger, context.str());

        populate_beammap_maps(
            mapmaking_grouping, mapmaking_method, active_maps_ptr,
            update_progress);
    }

    logger->info("normalizing maps");
    {
        const auto normalize_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.mapmaking.normalize", logger, context.str());
        if (rtcproc.run_kernel && !omb.grid_weight.empty()) {
            timestream::log_kernel_map_diag(
                logger,
                "beammap iter " + std::to_string(current_iter) + " before normalize",
                omb.kernel,
                active_maps_ptr,
                &omb.grid_weight);
        }
        omb.normalize_maps(active_maps_ptr);
        if (rtcproc.run_kernel) {
            timestream::log_kernel_map_diag(
                logger,
                "beammap iter " + std::to_string(current_iter) + " after normalize",
                omb.kernel,
                active_maps_ptr);
        }
        citlali::pipeline::log_beammap_normalize_support_summary(
            omb, calib, current_iter, logger);
    }
}

template <class RandomBits, class Generator>
void Beammap::run_beammap_mapmaking_stage(bool locator_iter,
                                          bool measurement_iter,
                                          bool detector_grouping,
                                          RandomBits &rands,
                                          Generator &eng) {
    logger->info("starting mapmaking");

    if (!run_mapmaking) {
        return;
    }

    run_beammap_mapmaking_pass(true, rands, eng);

    if (beammap_scan_band_mask_enabled && detector_grouping && locator_iter) {
        auto scan_band_summary = apply_scan_band_mask(omb);
        if (scan_band_summary.n_samples_flagged > 0) {
            logger->info(
                "beammap scan-band mask summary: flagged {} samples in {} rows across {} detectors ({} rejected by max_flagged_fraction={:.4f}); rebuilding maps",
                scan_band_summary.n_samples_flagged,
                scan_band_summary.n_rows_flagged,
                scan_band_summary.n_det_flagged,
                scan_band_summary.n_det_rejected,
                beammap_scan_band_mask_max_flagged_fraction);
            run_beammap_mapmaking_pass(false, rands, eng);
        }
        else {
            logger->info(
                "beammap scan-band mask summary: no edge bands flagged ({} detectors rejected by max_flagged_fraction={:.4f})",
                scan_band_summary.n_det_rejected,
                beammap_scan_band_mask_max_flagged_fraction);
        }
    }

    fit_beammap_maps(detector_grouping, measurement_iter);
}

template <class KidsProc, class RawObs>
bool Beammap::maybe_run_beammap_source_aware_rtc(KidsProc &kidsproc,
                                                 RawObs &rawobs,
                                                 bool first_measurement_iter,
                                                 bool detector_grouping) {
    configure_detector_source_centers_from_previous_fit();

    const bool detector_kernel_source_centers_active =
        detector_grouping &&
        rtcproc.run_kernel &&
        rtcproc.kernel.has_source_centers();
    const bool rerun_source_aware_rtc =
        first_measurement_iter && detector_kernel_source_centers_active;
    if (!rerun_source_aware_rtc) {
        return false;
    }

    logger->info(
        "beammap iter {} rerunning RTC with previous-fit detector source centers; regular RTC TOD output disabled for this internal pass",
        current_iter);
    const auto profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.rtc.source_aware_rerun", logger,
            "iter=" + std::to_string(current_iter));
    timestream_pipeline(kidsproc, rawobs, false);
    return true;
}

void Beammap::require_beammap_fit_map_geometry(Eigen::Index map_index) const {
    if (omb.signal[map_index].rows() != omb.n_rows ||
        omb.signal[map_index].cols() != omb.n_cols ||
        omb.weight[map_index].rows() != omb.n_rows ||
        omb.weight[map_index].cols() != omb.n_cols) {
        logger->error(
            "beammap fit map={} geometry mismatch: signal={}x{} weight={}x{} expected={}x{}",
            map_index, omb.signal[map_index].rows(),
            omb.signal[map_index].cols(), omb.weight[map_index].rows(),
            omb.weight[map_index].cols(), omb.n_rows, omb.n_cols);
        std::exit(EXIT_FAILURE);
    }
}

void Beammap::log_beammap_fit_map_stats(Eigen::Index map_index) const {
    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    const Eigen::Index n_pix = sig.size();
    const Eigen::Index sig_finite = sig.array().isFinite().count();
    const Eigen::Index wt_finite = wt.array().isFinite().count();
    const Eigen::Index wt_pos = (wt.array() > 0.0).count();
    logger->debug(
        "beammap fit map={} stats: sig_finite={}/{} wt_finite={}/{} wt_pos={}/{} sig[min,max]=({:.6g}, {:.6g}) wt[min,max]=({:.6g}, {:.6g})",
        map_index, sig_finite, n_pix, wt_finite, n_pix, wt_pos, n_pix,
        sig.minCoeff(), sig.maxCoeff(), wt.minCoeff(), wt.maxCoeff());
}

bool Beammap::prepare_beammap_fit_map(Eigen::Index map_index) {
    if (has_beammap_prior_diagnostics()) {
        reset_beammap_prior_diagnostics(map_index);
    }

    const Eigen::Index n_weight_pos =
        (omb.weight[map_index].array() > 0.0).count();
    if (n_weight_pos < map_fitter.n_params) {
        logger->warn(
            "beammap fit map={} skipped: insufficient weighted pixels ({})",
            map_index, n_weight_pos);
        clear_beammap_fit_result(map_index);
        return false;
    }
    return true;
}

double Beammap::beammap_init_fwhm_pix(Eigen::Index map_index) {
    const auto array = maps_to_arrays(map_index);
    return toltec_io.array_fwhm_arcsec[array] * ASEC_TO_RAD /
           omb.pixel_size_rad;
}

void Beammap::reset_beammap_fit_diagnostics(Eigen::Index map_index) {
    fit_diag_init_params.row(map_index).setZero();
    fit_diag_lower_limits.row(map_index).setZero();
    fit_diag_upper_limits.row(map_index).setZero();
    fit_diag_hit_lower.row(map_index).setZero();
    fit_diag_hit_upper.row(map_index).setZero();
    fit_diag_bound_code(map_index) = 0;
    fit_diag_bound_nhit(map_index) = 0;
}

void Beammap::clear_beammap_fit_result(Eigen::Index map_index) {
    params.row(map_index).setZero();
    perrors.row(map_index).setZero();
    reset_beammap_fit_diagnostics(map_index);
    good_fits(map_index) = false;
}

void Beammap::restore_converged_beammap_fit_result(Eigen::Index map_index) {
    params.row(map_index) = p0.row(map_index);
    perrors.row(map_index) = perror0.row(map_index);
}

bool Beammap::has_beammap_prior_diagnostics() const {
    return prior_diag_values.rows() == n_maps &&
           prior_diag_values.cols() == n_prior_diag_cols;
}

void Beammap::reset_beammap_prior_diagnostics(Eigen::Index map_index) {
    prior_diag_values.row(map_index).setConstant(
        std::numeric_limits<double>::quiet_NaN());
    prior_diag_values(map_index, prior_init_mode_col) = -1.0;
    prior_diag_values(map_index, prior_used_col) = 0.0;
    prior_diag_values(map_index, prior_fallback_blind_col) = 0.0;
    prior_diag_values(map_index, prior_no_candidate_reason_col) = 0.0;
    prior_diag_values(map_index, prior_slot_index_col) = -1.0;
}

bool Beammap::beammap_prior_allows_peak_switch(Eigen::Index map_index,
                                               double prev_row, double prev_col,
                                               Eigen::Index peak_row,
                                               Eigen::Index peak_col) {
    const int array_int = static_cast<int>(maps_to_arrays(map_index));
    const int nw_int = static_cast<int>(std::lround(calib.apt["nw"](map_index)));
    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    const double col0 = static_cast<double>(omb.n_cols - 1) / 2.0;
    const double row0 = static_cast<double>(omb.n_rows - 1) / 2.0;
    const double derot_elev_rad = get_prior_derot_elev_rad();
    const double prior_max_d2 = effective_prior_max_d2();

    auto prior_compatible = [&](double row, double col, double &d2_out) {
        const double x_raw = pix_to_arcsec * (col - col0);
        const double y_raw = pix_to_arcsec * (row - row0);
        double x_prior = std::numeric_limits<double>::quiet_NaN();
        double y_prior = std::numeric_limits<double>::quiet_NaN();
        d2_out = std::numeric_limits<double>::infinity();
        int slot_index = -1;
        if (!observed_to_prior_frame(array_int, x_raw, y_raw, derot_elev_rad,
                                     x_prior, y_prior, nullptr, nullptr, true)) {
            return false;
        }
        if (!match_prior_slot(array_int, nw_int, x_prior, y_prior,
                              d2_out, slot_index)) {
            return false;
        }
        static_cast<void>(slot_index);
        return prior_max_d2 <= 0.0 || d2_out <= prior_max_d2;
    };

    double prev_prior_d2 = std::numeric_limits<double>::infinity();
    double peak_prior_d2 = std::numeric_limits<double>::infinity();
    const bool prev_prior_ok = prior_compatible(prev_row, prev_col, prev_prior_d2);
    const bool peak_prior_ok = prior_compatible(
        static_cast<double>(peak_row), static_cast<double>(peak_col),
        peak_prior_d2);
    const bool prior_allows_switch = peak_prior_ok || !prev_prior_ok;
    if (!prior_allows_switch) {
        logger->debug(
            "beammap fit map={} kept previous init over stronger weighted peak because prior d2 prev={} peak={} max_d2={}",
            map_index, prev_prior_d2, peak_prior_d2, prior_max_d2);
    }
    return prior_allows_switch;
}

Beammap::BeammapPreviousFitInit Beammap::choose_previous_beammap_fit_init(
    Eigen::Index map_index, bool measurement_iter, bool can_try_prior,
    double init_fwhm) {
    BeammapPreviousFitInit result;

    if (!(measurement_iter &&
          good_fits(map_index) &&
          p0.cols() > 2 &&
          std::isfinite(p0(map_index, 0)) && p0(map_index, 0) > 0.0 &&
          std::isfinite(p0(map_index, 1)) &&
          std::isfinite(p0(map_index, 2)))) {
        return result;
    }

    const double prev_col = p0(map_index, 1);
    const double prev_row = p0(map_index, 2);
    Eigen::Index prev_row_i =
        static_cast<Eigen::Index>(std::llround(prev_row));
    Eigen::Index prev_col_i =
        static_cast<Eigen::Index>(std::llround(prev_col));
    bool prev_seed_valid = false;
    if (prev_row_i >= 0 && prev_row_i < omb.signal[map_index].rows() &&
        prev_col_i >= 0 && prev_col_i < omb.signal[map_index].cols()) {
        const double seed_w = omb.weight[map_index](prev_row_i, prev_col_i);
        const double seed_s = omb.signal[map_index](prev_row_i, prev_col_i);
        prev_seed_valid = std::isfinite(seed_w) && seed_w > 0.0 &&
                          std::isfinite(seed_s) && seed_s > 0.0;
        if (prev_seed_valid) {
            Eigen::Index peak_row = -1;
            Eigen::Index peak_col = -1;
            double peak_snr = -std::numeric_limits<double>::infinity();
            if (find_map_weighted_peak(map_index, peak_row, peak_col, peak_snr) &&
                peak_row >= 0 && peak_col >= 0 && std::isfinite(peak_snr)) {
                const double prev_snr = seed_s * std::sqrt(seed_w);
                const double dr = static_cast<double>(peak_row) - prev_row;
                const double dc = static_cast<double>(peak_col) - prev_col;
                const double dist_pix = std::sqrt(dr * dr + dc * dc);
                const double min_switch_dist_pix = std::max(1.0, init_fwhm);
                constexpr double min_switch_snr_ratio = 1.25;
                const bool prior_allows_switch =
                    !can_try_prior ||
                    beammap_prior_allows_peak_switch(
                        map_index, prev_row, prev_col, peak_row, peak_col);
                if (std::isfinite(prev_snr) &&
                    peak_snr > min_switch_snr_ratio * prev_snr &&
                    dist_pix > min_switch_dist_pix &&
                    prior_allows_switch) {
                    prev_seed_valid = false;
                    result.rejected_by_peak = true;
                    logger->debug(
                        "beammap fit map={} rejected previous init: current weighted peak row={} col={} snr={} is {} pix from previous row={} col={} snr={}",
                        map_index, peak_row, peak_col, peak_snr, dist_pix,
                        prev_row, prev_col, prev_snr);
                }
            }
        }
    }
    if (prev_seed_valid) {
        result.valid = true;
        result.col = prev_col;
        result.row = prev_row;
    }
    else {
        logger->debug(
            "beammap fit map={} rejected previous init at row={} col={} due to invalid/no-weight/non-positive seed pixel",
            map_index, prev_row, prev_col);
    }
    return result;
}

void Beammap::record_beammap_prior_init_mode(
    Eigen::Index map_index, const BeammapFitInitSelection &init_selection) {
    if (!has_beammap_prior_diagnostics()) {
        return;
    }
    if (init_selection.from_previous) {
        prior_diag_values(map_index, prior_init_mode_col) = 1.0;
    }
    else if (init_selection.from_prior) {
        prior_diag_values(map_index, prior_init_mode_col) = 2.0;
    }
    else {
        prior_diag_values(map_index, prior_init_mode_col) = 0.0;
    }
}

Beammap::BeammapFitInitSelection Beammap::choose_beammap_fit_init(
    Eigen::Index map_index, bool measurement_iter, bool can_try_prior,
    double init_fwhm, BeammapFitIterationStats &fit_stats) {
    BeammapFitInitSelection selection;

    const auto prev_init = choose_previous_beammap_fit_init(
        map_index, measurement_iter, can_try_prior, init_fwhm);
    if (prev_init.rejected_by_peak) {
        fit_stats.prev_rejected_by_peak++;
    }
    if (prev_init.valid) {
        selection.col = prev_init.col;
        selection.row = prev_init.row;
        selection.from_previous = true;
        selection.mode = BeammapFitInitMode::Previous;
        fit_stats.init_prev++;
        record_beammap_prior_init_mode(map_index, selection);
        return selection;
    }

    if (can_try_prior) {
        if (choose_prior_guided_init(map_index, selection.row, selection.col)) {
            selection.from_prior = true;
            selection.mode = BeammapFitInitMode::Prior;
            fit_stats.init_prior++;
        }
        else if (!beammap_priors_fallback_blind) {
            if (has_beammap_prior_diagnostics()) {
                prior_diag_values(map_index, prior_init_mode_col) = -1.0;
            }
            logger->warn(
                "beammap fit map={} skipped: no prior-guided init candidate and fallback_blind=false",
                map_index);
            fit_stats.init_skip++;
            selection.skip_fit = true;
            return selection;
        }
        else if (has_beammap_prior_diagnostics()) {
            prior_diag_values(map_index, prior_fallback_blind_col) = 1.0;
        }
    }

    if (!selection.from_prior) {
        fit_stats.init_blind++;
    }
    record_beammap_prior_init_mode(map_index, selection);
    return selection;
}

const char *Beammap::beammap_fit_init_mode_name(
    BeammapFitInitMode init_mode) const {
    switch (init_mode) {
        case BeammapFitInitMode::Previous:
            return "previous";
        case BeammapFitInitMode::Prior:
            return "prior";
        case BeammapFitInitMode::Blind:
            return "blind";
    }
    return "unknown";
}

Beammap::BeammapFitAttemptFlags Beammap::beammap_fit_attempt_flags(
    const engine_utils::mapFitter::FitDiagnostics &fit_diag) const {
    BeammapFitAttemptFlags flags;
    if (fit_diag.valid &&
        fit_diag.init_params.size() > 0 &&
        fit_diag.lower_limits.size() > 0 &&
        fit_diag.upper_limits.size() > 0) {
        const double init_amp = fit_diag.init_params(0);
        const double amp_low = fit_diag.lower_limits(0);
        const double amp_high = fit_diag.upper_limits(0);
        flags.init_amp_zero =
            std::isfinite(init_amp) && std::abs(init_amp) <= 1e-12;
        flags.amp_bounds_zero =
            std::isfinite(amp_low) && std::isfinite(amp_high) &&
            std::abs(amp_high - amp_low) <= 1e-12;
    }
    return flags;
}

void Beammap::record_beammap_fit_attempt_stats(
    BeammapFitIterationStats &fit_stats, BeammapFitInitMode init_mode,
    bool good_fit, bool init_amp_zero, bool amp_bounds_zero) {
    switch (init_mode) {
        case BeammapFitInitMode::Previous:
            fit_stats.attempt_prev++;
            if (!good_fit) {
                fit_stats.fail_prev++;
            }
            if (init_amp_zero) {
                fit_stats.init_amp_zero_prev++;
            }
            if (amp_bounds_zero) {
                fit_stats.amp_bounds_zero_prev++;
            }
            break;
        case BeammapFitInitMode::Prior:
            fit_stats.attempt_prior++;
            if (!good_fit) {
                fit_stats.fail_prior++;
            }
            if (init_amp_zero) {
                fit_stats.init_amp_zero_prior++;
            }
            if (amp_bounds_zero) {
                fit_stats.amp_bounds_zero_prior++;
            }
            break;
        case BeammapFitInitMode::Blind:
            fit_stats.attempt_blind++;
            if (!good_fit) {
                fit_stats.fail_blind++;
            }
            if (init_amp_zero) {
                fit_stats.init_amp_zero_blind++;
            }
            if (amp_bounds_zero) {
                fit_stats.amp_bounds_zero_blind++;
            }
            break;
    }
}

bool Beammap::has_complete_beammap_fit_diagnostics(
    const engine_utils::mapFitter::FitDiagnostics &fit_diag) const {
    return fit_diag.valid &&
           fit_diag.init_params.size() == map_fitter.n_params &&
           fit_diag.lower_limits.size() == map_fitter.n_params &&
           fit_diag.upper_limits.size() == map_fitter.n_params &&
           fit_diag.hit_lower.size() == map_fitter.n_params &&
           fit_diag.hit_upper.size() == map_fitter.n_params;
}

void Beammap::record_beammap_fit_diagnostics(
    Eigen::Index map_index,
    const engine_utils::mapFitter::FitDiagnostics &fit_diag,
    BeammapFitIterationStats &fit_stats) {
    if (!has_complete_beammap_fit_diagnostics(fit_diag)) {
        reset_beammap_fit_diagnostics(map_index);
        return;
    }

    fit_diag_init_params.row(map_index) = fit_diag.init_params.transpose();
    fit_diag_lower_limits.row(map_index) = fit_diag.lower_limits.transpose();
    fit_diag_upper_limits.row(map_index) = fit_diag.upper_limits.transpose();
    fit_diag_hit_lower.row(map_index) = fit_diag.hit_lower.transpose();
    fit_diag_hit_upper.row(map_index) = fit_diag.hit_upper.transpose();

    int bound_code = 0;
    int bound_nhit = 0;
    for (int p = 0; p < map_fitter.n_params; ++p) {
        const bool hit_low = fit_diag.hit_lower(p) != 0;
        const bool hit_high = fit_diag.hit_upper(p) != 0;
        if (hit_low) {
            bound_code |= (1 << (2 * p));
            fit_stats.bound_low(p)++;
            bound_nhit++;
        }
        if (hit_high) {
            bound_code |= (1 << (2 * p + 1));
            fit_stats.bound_high(p)++;
            bound_nhit++;
        }
    }
    fit_diag_bound_code(map_index) = bound_code;
    fit_diag_bound_nhit(map_index) = bound_nhit;
    if (bound_nhit > 0) {
        fit_stats.bound_any++;
    }
}

void Beammap::log_beammap_fit_iteration_stats(
    const BeammapFitIterationStats &fit_stats) {
    logger->info("beammap init summary (iter {}): previous={} prior={} blind={} skipped={} prev_rejected_by_peak={}",
                 current_iter, fit_stats.init_prev, fit_stats.init_prior, fit_stats.init_blind, fit_stats.init_skip,
                 fit_stats.prev_rejected_by_peak);
    logger->info(
        "beammap fit diagnostics (iter {}): prev fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{} | "
        "prior fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{} | "
        "blind fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{}",
        current_iter,
        fit_stats.fail_prev, fit_stats.attempt_prev, fit_stats.init_amp_zero_prev, fit_stats.attempt_prev,
        fit_stats.amp_bounds_zero_prev, fit_stats.attempt_prev,
        fit_stats.fail_prior, fit_stats.attempt_prior, fit_stats.init_amp_zero_prior, fit_stats.attempt_prior,
        fit_stats.amp_bounds_zero_prior, fit_stats.attempt_prior,
        fit_stats.fail_blind, fit_stats.attempt_blind, fit_stats.init_amp_zero_blind, fit_stats.attempt_blind,
        fit_stats.amp_bounds_zero_blind, fit_stats.attempt_blind);

    if (map_fitter.n_params >= 6) {
        logger->info(
            "beammap fit bound summary (iter {}): any_hit={}/{} amp(lo/hi)={}/{} x(lo/hi)={}/{} y(lo/hi)={}/{} a(lo/hi)={}/{} b(lo/hi)={}/{} angle(lo/hi)={}/{}",
            current_iter, fit_stats.bound_any, n_maps,
            fit_stats.bound_low(0), fit_stats.bound_high(0),
            fit_stats.bound_low(1), fit_stats.bound_high(1),
            fit_stats.bound_low(2), fit_stats.bound_high(2),
            fit_stats.bound_low(3), fit_stats.bound_high(3),
            fit_stats.bound_low(4), fit_stats.bound_high(4),
            fit_stats.bound_low(5), fit_stats.bound_high(5));
    }
    else {
        logger->info("beammap fit bound summary (iter {}): any_hit={}/{}",
                     current_iter, fit_stats.bound_any, n_maps);
    }
    logger->info("number of good fits {}/{}", static_cast<long long>(good_fits.cast<int>().sum()), n_maps);
}

void Beammap::fit_beammap_maps(bool detector_grouping, bool measurement_iter) {
    BeammapFitIterationStats fit_stats(map_fitter.n_params);

    logger->info("fitting maps");
    logger->info("beammap fit diagnostics enabled");
    if (beammap_priors_enabled && beammap_soft_priors_loaded &&
        detector_grouping) {
        update_prior_frame_estimates();
    }
    // Run beammap fits sequentially. This avoids allocator/covariance instability
    // observed with parallel Ceres fits on some systems.
    {
    const auto fit_profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.fit_maps", logger,
            "iter=" + std::to_string(current_iter) +
                " phase=" + beammap_iter_phase_name(current_iter));
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        logger->debug("beammap fit checkpoint: map={} begin converged={}", i, converged(i));

        require_beammap_fit_map_geometry(i);
        log_beammap_fit_map_stats(i);

        // only fit if not converged
        if (!converged(i)) {
            if (!prepare_beammap_fit_map(i)) {
                continue;
            }

            const double init_fwhm = beammap_init_fwhm_pix(i);
            const bool can_try_prior =
                beammap_priors_enabled && beammap_soft_priors_loaded &&
                detector_grouping;
            const auto init_selection = choose_beammap_fit_init(
                i, measurement_iter, can_try_prior, init_fwhm, fit_stats);
            if (init_selection.skip_fit) {
                clear_beammap_fit_result(i);
                continue;
            }
            logger->debug("beammap fit map={} init mode={} row={:.3f} col={:.3f}",
                          i, beammap_fit_init_mode_name(init_selection.mode),
                          init_selection.row, init_selection.col);
            // fit the maps
            logger->debug("beammap fit checkpoint: map={} call fit_to_gaussian", i);
            engine_utils::mapFitter::FitDiagnostics fit_diag;
            auto [det_params, det_perror, good_fit] =
                map_fitter.fit_to_gaussian<engine_utils::mapFitter::beammap>(omb.signal[i], omb.weight[i],
                                                                             init_fwhm, init_selection.row,
                                                                             init_selection.col, &fit_diag);
            logger->debug("beammap fit checkpoint: map={} fit_to_gaussian returned good_fit={}", i, good_fit);

            if (!(det_params.array().isFinite().all() && det_perror.array().isFinite().all())) {
                det_params.setZero();
                det_perror.setZero();
                good_fit = false;
            }

            params.row(i) = det_params;
            perrors.row(i) = det_perror;
            good_fits(i) = good_fit;

            const auto fit_flags = beammap_fit_attempt_flags(fit_diag);
            record_beammap_fit_attempt_stats(
                fit_stats, init_selection.mode, good_fit, fit_flags.init_amp_zero,
                fit_flags.amp_bounds_zero);
            record_beammap_fit_diagnostics(i, fit_diag, fit_stats);
        }
        // otherwise keep value from previous iteration
        else {
            restore_converged_beammap_fit_result(i);
        }

        logger->debug("beammap fit checkpoint: map={} end good_fit={}", i, good_fits(i));
    }
    }

    log_beammap_fit_iteration_stats(fit_stats);
}

bool Beammap::update_beammap_convergence_state() {
    if (!has_completed_beammap_measurement_iter(current_iter)) {
        return false;
    }

    // only do convergence test if tolerance is above zero, otherwise run all iterations
    if (run_mapmaking && beammap_iter_tolerance > 0) {
        // loop through maps and check if it is converged
        logger->info("checking convergence in fitted-source aperture radius={:.3f} arcsec",
                     beammap_convergence_radius_arcsec);
        const auto convergence_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.convergence", logger,
                "iter=" + std::to_string(current_iter) +
                    " radius_arcsec=" +
                    std::to_string(beammap_convergence_radius_arcsec));
        Eigen::VectorXd convergence_delta =
            Eigen::VectorXd::Constant(n_maps, std::numeric_limits<double>::quiet_NaN());
        grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            if (!converged(i)) {
                const double delta = calc_beammap_convergence_delta(i);
                convergence_delta(i) = delta;
                if (std::isfinite(delta) && delta <= beammap_iter_tolerance) {
                    // set as converged
                    converged(i) = true;
                    // set convergence iteration
                    converge_iter(i) = current_iter;
                }
            }
            return 0;
        });

        Eigen::Index n_delta_finite = 0;
        Eigen::Index n_delta_invalid = 0;
        double max_delta = 0.0;
        for (Eigen::Index i = 0; i < convergence_delta.size(); ++i) {
            if (std::isfinite(convergence_delta(i))) {
                n_delta_finite++;
                max_delta = std::max(max_delta, convergence_delta(i));
            }
            else if (!converged(i)) {
                n_delta_invalid++;
            }
        }

        logger->info(
            "{} maps converged on iter {} (finite_metrics={} invalid_metrics={} max_delta={})",
            (converged.array() == true).count(), current_iter,
            n_delta_finite, n_delta_invalid, max_delta);

        // stop if all maps converged
        if ((converged.array() == true).all()) {
            logger->info("all maps converged");
            return true;
        }
    }
    else {
        logger->info("bypassing convergence check");
    }
    return false;
}

bool Beammap::advance_beammap_iteration_state() {
    bool keep_going = true;

    // increment loop iteration
    current_iter++;

    if (current_iter < beammap_iter_max) {
        // check if all detectors are converged
        if ((converged.array() == true).all()) {
            logger->info("all maps converged");
            keep_going = false;
        }
        else if (update_beammap_convergence_state()) {
            keep_going = false;
        }

        // set previous iteration fits to current iteration fits
        p0 = params;
        perror0 = perrors;
    }
    else {
        logger->info("max iteration reached");
        keep_going = false;
    }

    return keep_going;
}

void Beammap::write_or_clear_beammap_ptc_products_for_iter(int completed_iter,
                                                           bool keep_going) {
    const bool beammap_iter_is_final = !keep_going;
    const bool write_beammap_ptc_this_iter =
        (beammap_tod_output_iter < 0 && beammap_iter_is_final) ||
        (beammap_tod_output_iter >= 0 && completed_iter == beammap_tod_output_iter);
    if (write_beammap_ptc_this_iter) {
        write_beammap_ptc_products(completed_iter);
    }
    else {
        clear_beammap_ptc_diagnostics();
    }
}

template <class KidsProc, class RawObs>
void Beammap::run_loop(KidsProc &kidsproc, RawObs &rawobs) {
    // variable to control iteration
    bool keep_going = true;

    // declare random number generator
    boost::random::mt19937 eng;

    // boost random number generator (0,1)
    boost::random::uniform_int_distribution<> rands{0,1};
    const bool detector_grouping =
        typed_config.mapmaking.grouping ==
        citlali::config::MapGrouping::detector;

    log_beammap_masking_config();

    // iterative loop
    while (keep_going) {
        const bool locator_iter = is_beammap_locator_iter(current_iter);
        const bool measurement_iter = is_beammap_measurement_iter(current_iter);
        const bool first_measurement_iter = is_beammap_first_measurement_iter(current_iter);
        logger->info(
            "starting iter {} phase={} locator_iter={} measurement_start_iter={}",
            current_iter, beammap_iter_phase_name(current_iter),
            beammap_locator_iter, beammap_measurement_start_iter);

        const bool rerun_source_aware_rtc =
            maybe_run_beammap_source_aware_rtc(
                kidsproc, rawobs, first_measurement_iter, detector_grouping);

        prepare_beammap_iteration_state(
            rerun_source_aware_rtc, measurement_iter, first_measurement_iter,
            detector_grouping);

        // cleaning (separate from mapmaking loop due to jinc mapmaking parallelization)
        run_beammap_ptc_cleaning_pass(
            locator_iter, measurement_iter, detector_grouping);

        run_beammap_mapmaking_stage(
            locator_iter, measurement_iter, detector_grouping, rands, eng);

        const int completed_iter = current_iter;
        keep_going = advance_beammap_iteration_state();
        write_or_clear_beammap_ptc_products_for_iter(completed_iter, keep_going);
    }
}

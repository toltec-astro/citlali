#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_fit_qc_schema.h>
#include <citlali/core/engine/detail/beammap_detector_table_vectors.h>

Beammap::FitQCSignalVectors Beammap::make_beammap_fit_qc_signal_vectors() {
    FitQCSignalVectors vectors{
        Eigen::VectorXd::Zero(calib.n_dets),
        Eigen::VectorXd::Zero(calib.n_dets),
        Eigen::VectorXd::Zero(calib.n_dets),
        Eigen::VectorXd::Zero(calib.n_dets)};
    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        const double amp = params(i, 0);
        const double amp_err = perrors(i, 0);
        const double rms = calc_map_support_stddev(i, true);
        const double npos = static_cast<double>((omb.weight[i].array() > 0.0).count());
        vectors.n_weight_pos(i) = npos;
        if (std::isfinite(rms) && rms > 0.0) {
            vectors.map_rms(i) = rms;
            if (std::isfinite(amp)) {
                vectors.map_sig2noise(i) = amp / rms;
            }
        }
        if (std::isfinite(amp) && std::isfinite(amp_err) && amp_err > 0.0) {
            vectors.fit_sig2noise(i) = amp / amp_err;
        }
    }
    return vectors;
}

void Beammap::write_beammap_fit_qc_table(const std::string &apt_filename) {
    logger->info("writing beammap fit qc table");
    std::string fit_qc_filename = apt_filename + "_fit_qc";

    std::vector<std::string> fit_qc_header = beammap_fit_qc_schema::header_keys();

    const auto table_access = beammap_detector_table_vectors::make_accessors(
        calib.apt, calib.apt_header_units, calib.apt_header_description,
        prior_diag_values, calib.n_dets, n_prior_diag_cols);
    auto fit_signal = make_beammap_fit_qc_signal_vectors();

    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    const double sigma_to_fwhm_arcsec = pix_to_arcsec * STD_TO_FWHM;

    Eigen::VectorXd fit_bound_nhit = fit_diag_bound_nhit.cast<double>();
    Eigen::VectorXd fit_bound_code = fit_diag_bound_code.cast<double>();
    auto fit_bounds = beammap_detector_table_vectors::fit_bound_vectors(
        fit_diag_hit_upper, fit_diag_hit_lower);
    auto fit_init_limits =
        beammap_detector_table_vectors::fit_init_limit_vectors(
            fit_diag_init_params, fit_diag_lower_limits, fit_diag_upper_limits,
            pix_to_arcsec, sigma_to_fwhm_arcsec, omb.n_cols, omb.n_rows);

    const double fill_double = std::numeric_limits<double>::quiet_NaN();

    auto fruitloops = beammap_detector_table_vectors::fruitloops_qc_vectors(
        ptcproc, omb, calib.n_dets, pix_to_arcsec, fill_double);

    Eigen::MatrixXd fit_qc_table(calib.n_dets, fit_qc_header.size());
    Eigen::Index col = 0;
    fit_qc_table.col(col++) = table_access.apt_or_zero("uid");
    fit_qc_table.col(col++) = table_access.apt_or_zero("array");
    fit_qc_table.col(col++) = table_access.apt_or_zero("nw");
    fit_qc_table.col(col++) = table_access.apt_or_zero("kids_tone");
    fit_qc_table.col(col++) = good_fits.cast<double>();
    fit_qc_table.col(col++) = converged.cast<double>();
    fit_qc_table.col(col++) = converge_iter.cast<double>();
    fit_qc_table.col(col++) = table_access.apt_or_zero("flag");
    fit_qc_table.col(col++) = flag2.cast<double>();
    fit_qc_table.col(col++) = table_access.apt_or_zero("amp");
    fit_qc_table.col(col++) = table_access.apt_or_zero("amp_err");
    fit_qc_table.col(col++) = table_access.apt_or_zero("cal_amp");
    fit_qc_table.col(col++) = table_access.apt_or_zero("cal_amp_method");
    fit_qc_table.col(col++) = table_access.apt_or_zero("template_amp");
    fit_qc_table.col(col++) = table_access.apt_or_zero("template_offset");
    fit_qc_table.col(col++) = table_access.apt_or_zero("template_resid_rms");
    fit_qc_table.col(col++) = table_access.apt_or_zero("template_npix");
    fit_qc_table.col(col++) = table_access.apt_or_zero("template_amp_over_fit_amp");
    fit_qc_table.col(col++) = table_access.apt_or_zero("cal_amp_over_fit_amp");
    fit_qc_table.col(col++) = table_access.apt_or_zero("map_peak_amp");
    fit_qc_table.col(col++) = table_access.apt_or_zero("map_peak_amp_over_fit_amp");
    fit_qc_table.col(col++) = fit_signal.fit_sig2noise;
    fit_qc_table.col(col++) = fit_signal.map_rms;
    fit_qc_table.col(col++) = fit_signal.map_sig2noise;
    fit_qc_table.col(col++) = fit_signal.n_weight_pos;
    fit_qc_table.col(col++) = fruitloops.source_x_t;
    fit_qc_table.col(col++) = fruitloops.source_y_t;
    fit_qc_table.col(col++) = fruitloops.local_sigma;
    fit_qc_table.col(col++) = fruitloops.local_sigma_npix;
    fit_qc_table.col(col++) = fruitloops.amp_ref;
    fit_qc_table.col(col++) = fruitloops.peak_threshold;
    fit_qc_table.col(col++) = fruitloops.snr_threshold;
    fit_qc_table.col(col++) = fruitloops.adaptive_threshold;
    fit_qc_table.col(col++) = fruitloops.support_radius_arcsec;
    fit_qc_table.col(col++) = fruitloops.support.npix;
    fit_qc_table.col(col++) = fruitloops.support.signal_sum;
    fit_qc_table.col(col++) = fruitloops.support.x_span_arcsec;
    fit_qc_table.col(col++) = fruitloops.support.y_span_arcsec;
    fit_qc_table.col(col++) = table_access.apt_or_zero("rfi_masked_samples");
    fit_qc_table.col(col++) = table_access.apt_or_zero("rfi_masked_scans");
    fit_qc_table.col(col++) = table_access.apt_or_zero("scan_band_masked_samples");
    fit_qc_table.col(col++) = table_access.apt_or_zero("scan_band_masked_rows");
    fit_qc_table.col(col++) = table_access.apt_or_zero("scan_band_masked_edge");
    fit_qc_table.col(col++) = table_access.apt_or_zero("scan_band_mask_rejected");
    fit_qc_table.col(col++) = fit_bound_nhit;
    fit_qc_table.col(col++) = fit_bound_code;
    fit_qc_table.col(col++) = fit_bounds.amp;
    fit_qc_table.col(col++) = fit_bounds.x;
    fit_qc_table.col(col++) = fit_bounds.y;
    fit_qc_table.col(col++) = fit_bounds.a;
    fit_qc_table.col(col++) = fit_bounds.b;
    fit_qc_table.col(col++) = fit_bounds.angle;
    fit_qc_table.col(col++) = fit_init_limits.amp;
    fit_qc_table.col(col++) = fit_init_limits.x_t;
    fit_qc_table.col(col++) = fit_init_limits.y_t;
    fit_qc_table.col(col++) = fit_init_limits.a_fwhm;
    fit_qc_table.col(col++) = fit_init_limits.b_fwhm;
    fit_qc_table.col(col++) = fit_init_limits.low_a_fwhm;
    fit_qc_table.col(col++) = fit_init_limits.high_a_fwhm;
    fit_qc_table.col(col++) = fit_init_limits.low_b_fwhm;
    fit_qc_table.col(col++) = fit_init_limits.high_b_fwhm;
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_init_mode_col, -1.0);
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_used_col, 0.0);
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_fallback_blind_col, 0.0);
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_no_candidate_reason_col, 0.0);
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_slot_index_col, -1.0);
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_match_d2_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_match_score_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_candidate_snr_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_n_candidates_col, 0.0);
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_n_candidates_keep_col, 0.0);
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_n_candidates_gate_col, 0.0);
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_candidate_x_raw_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_candidate_y_raw_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_candidate_x_prior_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_candidate_y_prior_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_center_x_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_center_y_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_derot_elev_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_slot_x_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_slot_y_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_slot_sx_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.prior_diag_or(prior_slot_sy_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = table_access.apt_or_zero("final_prior_slot_index");
    fit_qc_table.col(col++) = table_access.apt_or_zero("final_prior_d2");
    fit_qc_table.col(col++) = table_access.apt_or_zero("x_t_raw");
    fit_qc_table.col(col++) = table_access.apt_or_zero("y_t_raw");
    fit_qc_table.col(col++) = table_access.apt_or_zero("x_t");
    fit_qc_table.col(col++) = table_access.apt_or_zero("y_t");
    fit_qc_table.col(col++) = table_access.apt_or_zero("x_t_derot");
    fit_qc_table.col(col++) = table_access.apt_or_zero("y_t_derot");
    fit_qc_table.col(col++) = table_access.apt_or_zero("a_fwhm");
    fit_qc_table.col(col++) = table_access.apt_or_zero("a_fwhm_err");
    fit_qc_table.col(col++) = table_access.apt_or_zero("b_fwhm");
    fit_qc_table.col(col++) = table_access.apt_or_zero("b_fwhm_err");
    fit_qc_table.col(col++) = table_access.apt_or_zero("angle");
    fit_qc_table.col(col++) = table_access.apt_or_zero("angle_err");
    fit_qc_table.col(col++) = table_access.apt_or_zero("flxscale");
    fit_qc_table.col(col++) = table_access.apt_or_zero("sens");

    YAML::Node fit_qc_meta =
        beammap_fit_qc_schema::make_metadata(*this, table_access,
                                             fit_qc_header);

    to_ecsv_from_matrix(fit_qc_filename, fit_qc_table, fit_qc_header, fit_qc_meta);
    logger->info("done writing beammap fit qc table {}.ecsv", fit_qc_filename);
}

void Beammap::write_detector_table_outputs() {
    if (typed_config.mapmaking.grouping !=
        citlali::config::MapGrouping::detector) {
        return;
    }

    const std::string apt_filename = write_beammap_apt_table();
    write_beammap_fit_qc_table(apt_filename);
}

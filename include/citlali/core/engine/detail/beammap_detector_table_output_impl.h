#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

void Beammap::write_detector_table_outputs() {
    if (map_grouping != "detector") {
        return;
    }

    logger->info("writing apt table");
    auto apt_filename = toltec_io.create_filename<engine_utils::toltecIO::apt, engine_utils::toltecIO::map,
                                                  engine_utils::toltecIO::raw>
                        (obsnum_dir_name + "raw/", redu_type, "", obsnum, telescope.sim_obs);

    Eigen::MatrixXd apt_table(calib.n_dets, calib.apt_header_keys.size());

    // copy to table
    Eigen::Index i = 0;
    for (auto const& x: calib.apt_header_keys) {
        if (x != "flag2") {
            apt_table.col(i) = calib.apt[x];
        }
        else {
            apt_table.col(i) = flag2.cast<double> ();
        }
        i++;
    }

    // write to ecsv
    to_ecsv_from_matrix(apt_filename, apt_table, calib.apt_header_keys, calib.apt_meta);

    logger->info("done writing apt table {}.ecsv",apt_filename);

    logger->info("writing beammap fit qc table");
    std::string fit_qc_filename = apt_filename + "_fit_qc";

    std::vector<std::string> fit_qc_header = {
        "uid",
        "array",
        "nw",
        "kids_tone",
        "good_fit",
        "converged",
        "converge_iter",
        "flag",
        "flag2",
        "amp",
        "amp_err",
        "cal_amp",
        "cal_amp_method",
        "template_amp",
        "template_offset",
        "template_resid_rms",
        "template_npix",
        "template_amp_over_fit_amp",
        "cal_amp_over_fit_amp",
        "map_peak_amp",
        "map_peak_amp_over_fit_amp",
        "fit_sig2noise",
        "map_rms",
        "map_sig2noise",
        "n_weight_pos",
        "fruitloops_source_x_t",
        "fruitloops_source_y_t",
        "fruitloops_local_sigma",
        "fruitloops_local_sigma_npix",
        "fruitloops_amp_ref",
        "fruitloops_peak_threshold",
        "fruitloops_snr_threshold",
        "fruitloops_adaptive_threshold",
        "fruitloops_support_radius_arcsec",
        "fruitloops_support_npix",
        "fruitloops_support_signal_sum",
        "fruitloops_support_x_span_arcsec",
        "fruitloops_support_y_span_arcsec",
        "rfi_masked_samples",
        "rfi_masked_scans",
        "scan_band_masked_samples",
        "scan_band_masked_rows",
        "scan_band_masked_edge",
        "scan_band_mask_rejected",
        "fit_bound_nhit",
        "fit_bound_code",
        "fit_bound_amp",
        "fit_bound_x",
        "fit_bound_y",
        "fit_bound_a",
        "fit_bound_b",
        "fit_bound_angle",
        "fit_init_amp",
        "fit_init_x_t",
        "fit_init_y_t",
        "fit_init_a_fwhm",
        "fit_init_b_fwhm",
        "fit_low_a_fwhm",
        "fit_high_a_fwhm",
        "fit_low_b_fwhm",
        "fit_high_b_fwhm",
        "prior_init_mode",
        "prior_used",
        "prior_fallback_blind",
        "prior_no_candidate_reason",
        "prior_slot_index",
        "prior_match_d2",
        "prior_match_score",
        "prior_candidate_snr",
        "prior_n_candidates",
        "prior_n_candidates_keep",
        "prior_n_candidates_gate",
        "prior_candidate_x_t_raw",
        "prior_candidate_y_t_raw",
        "prior_candidate_x_t_prior",
        "prior_candidate_y_t_prior",
        "prior_center_x_t",
        "prior_center_y_t",
        "prior_derot_elev",
        "prior_slot_x_t",
        "prior_slot_y_t",
        "prior_slot_sx",
        "prior_slot_sy",
        "final_prior_slot_index",
        "final_prior_d2",
        "x_t_raw",
        "y_t_raw",
        "x_t",
        "y_t",
        "x_t_derot",
        "y_t_derot",
        "a_fwhm",
        "a_fwhm_err",
        "b_fwhm",
        "b_fwhm_err",
        "angle",
        "angle_err",
        "flxscale",
        "sens"
    };

    auto apt_or_zero = [&](const std::string &key) -> Eigen::VectorXd {
        auto it = calib.apt.find(key);
        if (it != calib.apt.end() && it->second.size() == calib.n_dets) {
            return it->second;
        }
        return Eigen::VectorXd::Zero(calib.n_dets);
    };
    auto get_unit = [&](const std::string &key, const std::string &fallback) {
        auto it = calib.apt_header_units.find(key);
        if (it != calib.apt_header_units.end()) {
            return it->second;
        }
        return fallback;
    };
    auto get_description = [&](const std::string &key, const std::string &fallback) {
        auto it = calib.apt_header_description.find(key);
        if (it != calib.apt_header_description.end()) {
            return it->second;
        }
        return fallback;
    };
    auto prior_diag_or = [&](PriorDiagColumn diag_col, double fallback_value) -> Eigen::VectorXd {
        Eigen::VectorXd out(calib.n_dets);
        if (prior_diag_values.rows() == calib.n_dets && prior_diag_values.cols() == n_prior_diag_cols) {
            out = prior_diag_values.col(diag_col);
        }
        else {
            out.setConstant(fallback_value);
        }
        return out;
    };

    Eigen::VectorXd map_rms(calib.n_dets);
    Eigen::VectorXd fit_sig2noise(calib.n_dets);
    Eigen::VectorXd map_sig2noise(calib.n_dets);
    Eigen::VectorXd n_weight_pos(calib.n_dets);
    map_rms.setZero();
    fit_sig2noise.setZero();
    map_sig2noise.setZero();
    n_weight_pos.setZero();
    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        const double amp = params(i, 0);
        const double amp_err = perrors(i, 0);
        const double rms = calc_map_support_stddev(i, true);
        const double npos = static_cast<double>((omb.weight[i].array() > 0.0).count());
        n_weight_pos(i) = npos;
        if (std::isfinite(rms) && rms > 0.0) {
            map_rms(i) = rms;
            if (std::isfinite(amp)) {
                map_sig2noise(i) = amp / rms;
            }
        }
        if (std::isfinite(amp) && std::isfinite(amp_err) && amp_err > 0.0) {
            fit_sig2noise(i) = amp / amp_err;
        }
    }

    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    const double sigma_to_fwhm_arcsec = pix_to_arcsec * STD_TO_FWHM;

    Eigen::VectorXd fit_bound_nhit = fit_diag_bound_nhit.cast<double>();
    Eigen::VectorXd fit_bound_code = fit_diag_bound_code.cast<double>();
    auto bound_state = [&](Eigen::Index p) -> Eigen::VectorXd {
        return fit_diag_hit_upper.col(p).cast<double>() - fit_diag_hit_lower.col(p).cast<double>();
    };
    Eigen::VectorXd fit_bound_amp = bound_state(0);
    Eigen::VectorXd fit_bound_x = bound_state(1);
    Eigen::VectorXd fit_bound_y = bound_state(2);
    Eigen::VectorXd fit_bound_a = bound_state(3);
    Eigen::VectorXd fit_bound_b = bound_state(4);
    Eigen::VectorXd fit_bound_angle = bound_state(5);

    Eigen::VectorXd fit_init_amp = fit_diag_init_params.col(0);
    Eigen::VectorXd fit_init_x_t =
        (pix_to_arcsec * (fit_diag_init_params.col(1).array() - (omb.n_cols - 1) / 2.0)).matrix();
    Eigen::VectorXd fit_init_y_t =
        (pix_to_arcsec * (fit_diag_init_params.col(2).array() - (omb.n_rows - 1) / 2.0)).matrix();
    Eigen::VectorXd fit_init_a_fwhm = (sigma_to_fwhm_arcsec * fit_diag_init_params.col(3).array()).matrix();
    Eigen::VectorXd fit_init_b_fwhm = (sigma_to_fwhm_arcsec * fit_diag_init_params.col(4).array()).matrix();
    Eigen::VectorXd fit_low_a_fwhm = (sigma_to_fwhm_arcsec * fit_diag_lower_limits.col(3).array()).matrix();
    Eigen::VectorXd fit_high_a_fwhm = (sigma_to_fwhm_arcsec * fit_diag_upper_limits.col(3).array()).matrix();
    Eigen::VectorXd fit_low_b_fwhm = (sigma_to_fwhm_arcsec * fit_diag_lower_limits.col(4).array()).matrix();
    Eigen::VectorXd fit_high_b_fwhm = (sigma_to_fwhm_arcsec * fit_diag_upper_limits.col(4).array()).matrix();

    const double fill_double = std::numeric_limits<double>::quiet_NaN();
    auto fruitloops_or_nan = [&](const Eigen::VectorXd &values,
                                 double scale = 1.0) -> Eigen::VectorXd {
        Eigen::VectorXd out = Eigen::VectorXd::Constant(calib.n_dets, fill_double);
        if (values.size() == calib.n_dets) {
            out = (scale * values.array()).matrix();
        }
        return out;
    };
    auto fruitloops_int_or_nan = [&](const Eigen::VectorXi &values) -> Eigen::VectorXd {
        Eigen::VectorXd out = Eigen::VectorXd::Constant(calib.n_dets, fill_double);
        if (values.size() == calib.n_dets) {
            out = values.cast<double>();
        }
        return out;
    };

    Eigen::VectorXd fruitloops_source_x_t =
        fruitloops_or_nan(ptcproc.fruit_loops_source_lon, RAD_TO_ASEC);
    Eigen::VectorXd fruitloops_source_y_t =
        fruitloops_or_nan(ptcproc.fruit_loops_source_lat, RAD_TO_ASEC);
    Eigen::VectorXd fruitloops_local_sigma =
        fruitloops_or_nan(ptcproc.fruit_loops_local_sigma_map);
    Eigen::VectorXd fruitloops_local_sigma_npix =
        fruitloops_int_or_nan(ptcproc.fruit_loops_local_sigma_npix);
    Eigen::VectorXd fruitloops_amp_ref =
        fruitloops_or_nan(ptcproc.fruit_loops_amp_ref);
    Eigen::VectorXd fruitloops_adaptive_threshold =
        fruitloops_or_nan(ptcproc.fruit_loops_adaptive_threshold);
    Eigen::VectorXd fruitloops_support_radius_arcsec =
        fruitloops_or_nan(ptcproc.fruit_loops_adaptive_support_radius_rad, RAD_TO_ASEC);
    Eigen::VectorXd fruitloops_peak_threshold =
        Eigen::VectorXd::Constant(calib.n_dets, fill_double);
    Eigen::VectorXd fruitloops_snr_threshold =
        Eigen::VectorXd::Constant(calib.n_dets, fill_double);
    Eigen::VectorXd fruitloops_support_npix =
        Eigen::VectorXd::Constant(calib.n_dets, fill_double);
    Eigen::VectorXd fruitloops_support_signal_sum =
        Eigen::VectorXd::Constant(calib.n_dets, fill_double);
    Eigen::VectorXd fruitloops_support_x_span_arcsec =
        Eigen::VectorXd::Constant(calib.n_dets, fill_double);
    Eigen::VectorXd fruitloops_support_y_span_arcsec =
        Eigen::VectorXd::Constant(calib.n_dets, fill_double);

    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        const double amp_ref = fruitloops_amp_ref(i);
        if (ptcproc.fruit_loops_peak_fraction_limit > 0.0 &&
            std::isfinite(amp_ref) && amp_ref > 0.0) {
            fruitloops_peak_threshold(i) =
                ptcproc.fruit_loops_peak_fraction_limit * amp_ref;
        }
        const double sigma = fruitloops_local_sigma(i);
        if (ptcproc.fruit_loops_local_snr_floor > 0.0 &&
            std::isfinite(sigma) && sigma > 0.0) {
            fruitloops_snr_threshold(i) =
                ptcproc.fruit_loops_local_snr_floor * sigma;
        }

        if (i >= static_cast<Eigen::Index>(omb.signal.size()) ||
            i >= static_cast<Eigen::Index>(omb.weight.size()) ||
            omb.signal[i].rows() != omb.n_rows ||
            omb.signal[i].cols() != omb.n_cols ||
            omb.weight[i].rows() != omb.n_rows ||
            omb.weight[i].cols() != omb.n_cols) {
            continue;
        }
        const double threshold = fruitloops_adaptive_threshold(i);
        if (!std::isfinite(threshold) || threshold <= 0.0) {
            continue;
        }
        if (ptcproc.fruit_loops_source_valid.size() != calib.n_dets ||
            ptcproc.fruit_loops_source_valid(i) == 0 ||
            !std::isfinite(ptcproc.fruit_loops_source_lat(i)) ||
            !std::isfinite(ptcproc.fruit_loops_source_lon(i))) {
            continue;
        }

        const double center_row =
            ptcproc.fruit_loops_source_lat(i) / omb.pixel_size_rad +
            (omb.n_rows - 1) / 2.0;
        const double center_col =
            ptcproc.fruit_loops_source_lon(i) / omb.pixel_size_rad +
            (omb.n_cols - 1) / 2.0;
        if (!std::isfinite(center_row) || !std::isfinite(center_col)) {
            continue;
        }

        double support_radius_pix = std::numeric_limits<double>::infinity();
        const double support_radius_rad =
            (ptcproc.fruit_loops_adaptive_support_radius_rad.size() == calib.n_dets)
                ? ptcproc.fruit_loops_adaptive_support_radius_rad(i)
                : fill_double;
        if (std::isfinite(support_radius_rad) && support_radius_rad > 0.0) {
            support_radius_pix = support_radius_rad / omb.pixel_size_rad;
        }

        Eigen::Index npix = 0;
        double signal_sum = 0.0;
        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();
        for (Eigen::Index row = 0; row < omb.n_rows; ++row) {
            const double drow_pix = static_cast<double>(row) - center_row;
            for (Eigen::Index col = 0; col < omb.n_cols; ++col) {
                const double weight = omb.weight[i](row, col);
                const double signal = omb.signal[i](row, col);
                if (!std::isfinite(weight) || weight <= 0.0 ||
                    !std::isfinite(signal)) {
                    continue;
                }
                const double dcol_pix = static_cast<double>(col) - center_col;
                if (std::sqrt(drow_pix * drow_pix + dcol_pix * dcol_pix) >
                    support_radius_pix) {
                    continue;
                }
                bool include_pixel = false;
                if (ptcproc.fruit_mode == "upper") {
                    include_pixel = signal >= threshold;
                }
                else if (ptcproc.fruit_mode == "lower") {
                    include_pixel = signal <= -std::abs(threshold);
                }
                else {
                    include_pixel = std::abs(signal) >= threshold;
                }
                if (!include_pixel) {
                    continue;
                }
                const double x_arcsec = dcol_pix * pix_to_arcsec;
                const double y_arcsec = drow_pix * pix_to_arcsec;
                min_x = std::min(min_x, x_arcsec);
                max_x = std::max(max_x, x_arcsec);
                min_y = std::min(min_y, y_arcsec);
                max_y = std::max(max_y, y_arcsec);
                signal_sum += signal;
                ++npix;
            }
        }
        fruitloops_support_npix(i) = static_cast<double>(npix);
        fruitloops_support_signal_sum(i) = signal_sum;
        if (npix > 0) {
            fruitloops_support_x_span_arcsec(i) = max_x - min_x;
            fruitloops_support_y_span_arcsec(i) = max_y - min_y;
        }
    }

    Eigen::MatrixXd fit_qc_table(calib.n_dets, fit_qc_header.size());
    Eigen::Index col = 0;
    fit_qc_table.col(col++) = apt_or_zero("uid");
    fit_qc_table.col(col++) = apt_or_zero("array");
    fit_qc_table.col(col++) = apt_or_zero("nw");
    fit_qc_table.col(col++) = apt_or_zero("kids_tone");
    fit_qc_table.col(col++) = good_fits.cast<double>();
    fit_qc_table.col(col++) = converged.cast<double>();
    fit_qc_table.col(col++) = converge_iter.cast<double>();
    fit_qc_table.col(col++) = apt_or_zero("flag");
    fit_qc_table.col(col++) = flag2.cast<double>();
    fit_qc_table.col(col++) = apt_or_zero("amp");
    fit_qc_table.col(col++) = apt_or_zero("amp_err");
    fit_qc_table.col(col++) = apt_or_zero("cal_amp");
    fit_qc_table.col(col++) = apt_or_zero("cal_amp_method");
    fit_qc_table.col(col++) = apt_or_zero("template_amp");
    fit_qc_table.col(col++) = apt_or_zero("template_offset");
    fit_qc_table.col(col++) = apt_or_zero("template_resid_rms");
    fit_qc_table.col(col++) = apt_or_zero("template_npix");
    fit_qc_table.col(col++) = apt_or_zero("template_amp_over_fit_amp");
    fit_qc_table.col(col++) = apt_or_zero("cal_amp_over_fit_amp");
    fit_qc_table.col(col++) = apt_or_zero("map_peak_amp");
    fit_qc_table.col(col++) = apt_or_zero("map_peak_amp_over_fit_amp");
    fit_qc_table.col(col++) = fit_sig2noise;
    fit_qc_table.col(col++) = map_rms;
    fit_qc_table.col(col++) = map_sig2noise;
    fit_qc_table.col(col++) = n_weight_pos;
    fit_qc_table.col(col++) = fruitloops_source_x_t;
    fit_qc_table.col(col++) = fruitloops_source_y_t;
    fit_qc_table.col(col++) = fruitloops_local_sigma;
    fit_qc_table.col(col++) = fruitloops_local_sigma_npix;
    fit_qc_table.col(col++) = fruitloops_amp_ref;
    fit_qc_table.col(col++) = fruitloops_peak_threshold;
    fit_qc_table.col(col++) = fruitloops_snr_threshold;
    fit_qc_table.col(col++) = fruitloops_adaptive_threshold;
    fit_qc_table.col(col++) = fruitloops_support_radius_arcsec;
    fit_qc_table.col(col++) = fruitloops_support_npix;
    fit_qc_table.col(col++) = fruitloops_support_signal_sum;
    fit_qc_table.col(col++) = fruitloops_support_x_span_arcsec;
    fit_qc_table.col(col++) = fruitloops_support_y_span_arcsec;
    fit_qc_table.col(col++) = apt_or_zero("rfi_masked_samples");
    fit_qc_table.col(col++) = apt_or_zero("rfi_masked_scans");
    fit_qc_table.col(col++) = apt_or_zero("scan_band_masked_samples");
    fit_qc_table.col(col++) = apt_or_zero("scan_band_masked_rows");
    fit_qc_table.col(col++) = apt_or_zero("scan_band_masked_edge");
    fit_qc_table.col(col++) = apt_or_zero("scan_band_mask_rejected");
    fit_qc_table.col(col++) = fit_bound_nhit;
    fit_qc_table.col(col++) = fit_bound_code;
    fit_qc_table.col(col++) = fit_bound_amp;
    fit_qc_table.col(col++) = fit_bound_x;
    fit_qc_table.col(col++) = fit_bound_y;
    fit_qc_table.col(col++) = fit_bound_a;
    fit_qc_table.col(col++) = fit_bound_b;
    fit_qc_table.col(col++) = fit_bound_angle;
    fit_qc_table.col(col++) = fit_init_amp;
    fit_qc_table.col(col++) = fit_init_x_t;
    fit_qc_table.col(col++) = fit_init_y_t;
    fit_qc_table.col(col++) = fit_init_a_fwhm;
    fit_qc_table.col(col++) = fit_init_b_fwhm;
    fit_qc_table.col(col++) = fit_low_a_fwhm;
    fit_qc_table.col(col++) = fit_high_a_fwhm;
    fit_qc_table.col(col++) = fit_low_b_fwhm;
    fit_qc_table.col(col++) = fit_high_b_fwhm;
    fit_qc_table.col(col++) = prior_diag_or(prior_init_mode_col, -1.0);
    fit_qc_table.col(col++) = prior_diag_or(prior_used_col, 0.0);
    fit_qc_table.col(col++) = prior_diag_or(prior_fallback_blind_col, 0.0);
    fit_qc_table.col(col++) = prior_diag_or(prior_no_candidate_reason_col, 0.0);
    fit_qc_table.col(col++) = prior_diag_or(prior_slot_index_col, -1.0);
    fit_qc_table.col(col++) = prior_diag_or(prior_match_d2_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_match_score_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_candidate_snr_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_n_candidates_col, 0.0);
    fit_qc_table.col(col++) = prior_diag_or(prior_n_candidates_keep_col, 0.0);
    fit_qc_table.col(col++) = prior_diag_or(prior_n_candidates_gate_col, 0.0);
    fit_qc_table.col(col++) = prior_diag_or(prior_candidate_x_raw_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_candidate_y_raw_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_candidate_x_prior_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_candidate_y_prior_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_center_x_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_center_y_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_derot_elev_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_slot_x_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_slot_y_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_slot_sx_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = prior_diag_or(prior_slot_sy_col, std::numeric_limits<double>::quiet_NaN());
    fit_qc_table.col(col++) = apt_or_zero("final_prior_slot_index");
    fit_qc_table.col(col++) = apt_or_zero("final_prior_d2");
    fit_qc_table.col(col++) = apt_or_zero("x_t_raw");
    fit_qc_table.col(col++) = apt_or_zero("y_t_raw");
    fit_qc_table.col(col++) = apt_or_zero("x_t");
    fit_qc_table.col(col++) = apt_or_zero("y_t");
    fit_qc_table.col(col++) = apt_or_zero("x_t_derot");
    fit_qc_table.col(col++) = apt_or_zero("y_t_derot");
    fit_qc_table.col(col++) = apt_or_zero("a_fwhm");
    fit_qc_table.col(col++) = apt_or_zero("a_fwhm_err");
    fit_qc_table.col(col++) = apt_or_zero("b_fwhm");
    fit_qc_table.col(col++) = apt_or_zero("b_fwhm_err");
    fit_qc_table.col(col++) = apt_or_zero("angle");
    fit_qc_table.col(col++) = apt_or_zero("angle_err");
    fit_qc_table.col(col++) = apt_or_zero("flxscale");
    fit_qc_table.col(col++) = apt_or_zero("sens");

    YAML::Node fit_qc_meta;
    fit_qc_meta["obsnum"] = obsnum;
    fit_qc_meta["source"] = telescope.source_name;
    fit_qc_meta["creation_date"] = engine_utils::current_date_time();
    fit_qc_meta["date"] = date_obs.back();
    fit_qc_meta["map_grouping"] = map_grouping;
    fit_qc_meta["beammap_iter_max"] = beammap_iter_max;
    fit_qc_meta["beammap_iter_tolerance"] = beammap_iter_tolerance;
    fit_qc_meta["beammap_convergence_radius_arcsec"] = beammap_convergence_radius_arcsec;
    fit_qc_meta["beammap_phase_split_enabled"] = beammap_phase_split_enabled;
    fit_qc_meta["beammap_locator_iter"] = beammap_locator_iter;
    fit_qc_meta["beammap_measurement_start_iter"] = beammap_measurement_start_iter;
    fit_qc_meta["reference_detector_subtracted"] = beammap_subtract_reference;
    fit_qc_meta["reference_det"] = beammap_reference_det_found;
    fit_qc_meta["rfi_mask_enabled"] = beammap_rfi_mask_enabled;
    fit_qc_meta["rfi_mask_block_size_samples"] = beammap_rfi_mask_block_size_samples;
    fit_qc_meta["rfi_mask_min_good_samples"] = beammap_rfi_mask_min_good_samples;
    fit_qc_meta["rfi_mask_dilate_blocks"] = beammap_rfi_mask_dilate_blocks;
    fit_qc_meta["rfi_mask_sigma_threshold"] = beammap_rfi_mask_sigma_threshold;
    fit_qc_meta["rfi_mask_sigma_floor"] = beammap_rfi_mask_sigma_floor;
    fit_qc_meta["rfi_mask_max_flagged_fraction"] = beammap_rfi_mask_max_flagged_fraction;
    fit_qc_meta["detector_weighting_mode"] = beammap_detector_weighting_mode;
    fit_qc_meta["beammap_fit_radius_fwhm"] = beammap_fit_radius_fwhm;
    fit_qc_meta["rfi_mask_detectors_affected"] =
        static_cast<int>((apt_or_zero("rfi_masked_scans").array() > 0.0).count());
    fit_qc_meta["scan_band_mask_enabled"] = beammap_scan_band_mask_enabled;
    fit_qc_meta["scan_band_mask_edge_rows"] = beammap_scan_band_mask_edge_rows;
    fit_qc_meta["scan_band_mask_min_row_pixels"] = beammap_scan_band_mask_min_row_pixels;
    fit_qc_meta["scan_band_mask_min_contiguous_rows"] = beammap_scan_band_mask_min_contiguous_rows;
    fit_qc_meta["scan_band_mask_row_median_sigma_threshold"] =
        beammap_scan_band_mask_row_median_sigma_threshold;
    fit_qc_meta["scan_band_mask_row_sigma_ratio_threshold"] =
        beammap_scan_band_mask_row_sigma_ratio_threshold;
    fit_qc_meta["scan_band_mask_max_flagged_fraction"] =
        beammap_scan_band_mask_max_flagged_fraction;
    fit_qc_meta["scan_band_mask_detectors_affected"] =
        static_cast<int>((apt_or_zero("scan_band_masked_rows").array() > 0.0).count());
    fit_qc_meta["scan_band_mask_detectors_rejected"] =
        static_cast<int>((apt_or_zero("scan_band_mask_rejected").array() > 0.0).count());
    fit_qc_meta["fit_bound_any"] = static_cast<int>((fit_diag_bound_nhit.array() > 0).count());
    fit_qc_meta["beammap_priors_enabled"] = beammap_priors_enabled;
    fit_qc_meta["beammap_priors_filepath"] = beammap_priors_filepath;
    fit_qc_meta["beammap_priors_centered"] = beammap_soft_priors_are_centered;
    fit_qc_meta["beammap_priors_derotated"] = beammap_soft_priors_are_derotated;
    fit_qc_meta["beammap_priors_max_d2_iter0"] = beammap_priors_max_d2_iter0;
    fit_qc_meta["beammap_priors_max_d2_after_iter0"] = beammap_priors_max_d2_after_iter0;
    fit_qc_meta["beammap_priors_score_lambda_iter0"] = beammap_priors_score_lambda_iter0;
    fit_qc_meta["beammap_priors_score_lambda_after_iter0"] = beammap_priors_score_lambda_after_iter0;
    fit_qc_meta["beammap_priors_align_after_iter0"] = beammap_priors_align_after_iter0;
    fit_qc_meta["beammap_priors_alignment_scope"] = beammap_priors_alignment_scope;
    fit_qc_meta["beammap_priors_alignment_common_support"] =
        beammap_priors_alignment_common_support;
    fit_qc_meta["beammap_priors_alignment_common_support_quantile"] =
        beammap_priors_alignment_common_support_quantile;
    fit_qc_meta["beammap_priors_alignment_min_matches"] = beammap_priors_alignment_min_matches;
    fit_qc_meta["beammap_priors_alignment_max_d2"] = beammap_priors_alignment_max_d2;
    fit_qc_meta["beammap_priors_alignment_fit_rotation"] = beammap_priors_alignment_fit_rotation;
    fit_qc_meta["beammap_priors_alignment_max_rotation_deg"] = beammap_priors_alignment_max_rotation_deg;
    fit_qc_meta["beammap_priors_aligned_arrays"] = static_cast<int>(beammap_prior_array_alignment.size());

    std::map<std::string, std::string> fit_qc_units = {
        {"uid", "N/A"},
        {"array", "N/A"},
        {"nw", "N/A"},
        {"kids_tone", "N/A"},
        {"good_fit", "N/A"},
        {"converged", "N/A"},
        {"converge_iter", "N/A"},
        {"flag", "N/A"},
        {"flag2", "N/A"},
        {"amp", get_unit("amp", omb.sig_unit)},
        {"amp_err", get_unit("amp_err", omb.sig_unit)},
        {"cal_amp", get_unit("cal_amp", omb.sig_unit)},
        {"cal_amp_method", "N/A"},
        {"template_amp", get_unit("template_amp", omb.sig_unit)},
        {"template_offset", get_unit("template_offset", omb.sig_unit)},
        {"template_resid_rms", get_unit("template_resid_rms", omb.sig_unit)},
        {"template_npix", "pix"},
        {"template_amp_over_fit_amp", "N/A"},
        {"cal_amp_over_fit_amp", "N/A"},
        {"map_peak_amp", get_unit("map_peak_amp", omb.sig_unit)},
        {"map_peak_amp_over_fit_amp", "N/A"},
        {"fit_sig2noise", "N/A"},
        {"map_rms", omb.sig_unit},
        {"map_sig2noise", "N/A"},
        {"n_weight_pos", "pix"},
        {"fruitloops_source_x_t", "arcsec"},
        {"fruitloops_source_y_t", "arcsec"},
        {"fruitloops_local_sigma", omb.sig_unit},
        {"fruitloops_local_sigma_npix", "pix"},
        {"fruitloops_amp_ref", omb.sig_unit},
        {"fruitloops_peak_threshold", omb.sig_unit},
        {"fruitloops_snr_threshold", omb.sig_unit},
        {"fruitloops_adaptive_threshold", omb.sig_unit},
        {"fruitloops_support_radius_arcsec", "arcsec"},
        {"fruitloops_support_npix", "pix"},
        {"fruitloops_support_signal_sum", omb.sig_unit},
        {"fruitloops_support_x_span_arcsec", "arcsec"},
        {"fruitloops_support_y_span_arcsec", "arcsec"},
        {"rfi_masked_samples", "samples"},
        {"rfi_masked_scans", "scans"},
        {"scan_band_masked_samples", "samples"},
        {"scan_band_masked_rows", "rows"},
        {"scan_band_masked_edge", "N/A"},
        {"scan_band_mask_rejected", "N/A"},
        {"fit_bound_nhit", "N/A"},
        {"fit_bound_code", "N/A"},
        {"fit_bound_amp", "N/A"},
        {"fit_bound_x", "N/A"},
        {"fit_bound_y", "N/A"},
        {"fit_bound_a", "N/A"},
        {"fit_bound_b", "N/A"},
        {"fit_bound_angle", "N/A"},
        {"fit_init_amp", get_unit("amp", omb.sig_unit)},
        {"fit_init_x_t", "arcsec"},
        {"fit_init_y_t", "arcsec"},
        {"fit_init_a_fwhm", "arcsec"},
        {"fit_init_b_fwhm", "arcsec"},
        {"fit_low_a_fwhm", "arcsec"},
        {"fit_high_a_fwhm", "arcsec"},
        {"fit_low_b_fwhm", "arcsec"},
        {"fit_high_b_fwhm", "arcsec"},
        {"prior_init_mode", "N/A"},
        {"prior_used", "N/A"},
        {"prior_fallback_blind", "N/A"},
        {"prior_no_candidate_reason", "N/A"},
        {"prior_slot_index", "N/A"},
        {"prior_match_d2", "N/A"},
        {"prior_match_score", "N/A"},
        {"prior_candidate_snr", "N/A"},
        {"prior_n_candidates", "pix"},
        {"prior_n_candidates_keep", "pix"},
        {"prior_n_candidates_gate", "pix"},
        {"prior_candidate_x_t_raw", "arcsec"},
        {"prior_candidate_y_t_raw", "arcsec"},
        {"prior_candidate_x_t_prior", "arcsec"},
        {"prior_candidate_y_t_prior", "arcsec"},
        {"prior_center_x_t", "arcsec"},
        {"prior_center_y_t", "arcsec"},
        {"prior_derot_elev", "rad"},
        {"prior_slot_x_t", "arcsec"},
        {"prior_slot_y_t", "arcsec"},
        {"prior_slot_sx", "arcsec"},
        {"prior_slot_sy", "arcsec"},
        {"final_prior_slot_index", "N/A"},
        {"final_prior_d2", "N/A"},
        {"x_t_raw", get_unit("x_t", "arcsec")},
        {"y_t_raw", get_unit("y_t", "arcsec")},
        {"x_t", get_unit("x_t", "arcsec")},
        {"y_t", get_unit("y_t", "arcsec")},
        {"x_t_derot", get_unit("x_t", "arcsec")},
        {"y_t_derot", get_unit("y_t", "arcsec")},
        {"a_fwhm", get_unit("a_fwhm", "arcsec")},
        {"a_fwhm_err", get_unit("a_fwhm_err", "arcsec")},
        {"b_fwhm", get_unit("b_fwhm", "arcsec")},
        {"b_fwhm_err", get_unit("b_fwhm_err", "arcsec")},
        {"angle", get_unit("angle", "rad")},
        {"angle_err", get_unit("angle_err", "rad")},
        {"flxscale", get_unit("flxscale", "N/A")},
        {"sens", get_unit("sens", "N/A")}
    };
    std::map<std::string, std::string> fit_qc_desc = {
        {"uid", get_description("uid", "detector uid")},
        {"array", get_description("array", "array index")},
        {"nw", get_description("nw", "network index")},
        {"kids_tone", get_description("kids_tone", "index of tone in network")},
        {"good_fit", "fit returned a usable solution"},
        {"converged", "beammap iterative convergence flag"},
        {"converge_iter", get_description("converge_iter", "beammap convergence iteration")},
        {"flag", get_description("flag", "detector quality flag")},
        {"flag2", "bitwise detector quality flag"},
        {"amp", get_description("amp", "fitted beam amplitude")},
        {"amp_err", get_description("amp_err", "fitted beam amplitude uncertainty")},
        {"cal_amp", get_description("cal_amp", "amplitude used for beammap flux calibration")},
        {"cal_amp_method", get_description("cal_amp_method", "calibration amplitude method code")},
        {"template_amp", get_description("template_amp", "empirical-template matched amplitude")},
        {"template_offset", get_description("template_offset", "empirical-template fitted local offset")},
        {"template_resid_rms", get_description("template_resid_rms", "empirical-template residual RMS")},
        {"template_npix", get_description("template_npix", "number of pixels used by empirical-template amplitude fit")},
        {"template_amp_over_fit_amp", get_description("template_amp_over_fit_amp", "empirical-template amplitude divided by Gaussian fit amplitude")},
        {"cal_amp_over_fit_amp", get_description("cal_amp_over_fit_amp", "calibration amplitude divided by Gaussian fit amplitude")},
        {"map_peak_amp", get_description("map_peak_amp", "local map peak near Gaussian fit center")},
        {"map_peak_amp_over_fit_amp", get_description("map_peak_amp_over_fit_amp", "local map peak divided by Gaussian fit amplitude")},
        {"fit_sig2noise", "fitted amplitude divided by fitted amplitude uncertainty"},
        {"map_rms", "standard deviation of positive-weight detector map pixels, excluding fitted source core when possible"},
        {"map_sig2noise", "fitted amplitude divided by support-only detector map rms"},
        {"n_weight_pos", "number of detector-map pixels with positive weight"},
        {"fruitloops_source_x_t", "source-support x center used by adaptive fruit loops feedback"},
        {"fruitloops_source_y_t", "source-support y center used by adaptive fruit loops feedback"},
        {"fruitloops_local_sigma", "local annulus robust sigma used by adaptive fruit loops feedback"},
        {"fruitloops_local_sigma_npix", "number of local-annulus pixels used for fruit loops local sigma"},
        {"fruitloops_amp_ref", "amplitude reference used by adaptive fruit loops feedback"},
        {"fruitloops_peak_threshold", "peak-fraction threshold component for adaptive fruit loops feedback"},
        {"fruitloops_snr_threshold", "local-S/N threshold component for adaptive fruit loops feedback"},
        {"fruitloops_adaptive_threshold", "final adaptive threshold used for fruit loops map feedback"},
        {"fruitloops_support_radius_arcsec", "radial source-support limit used by adaptive fruit loops feedback"},
        {"fruitloops_support_npix", "number of final-map positive-weight pixels passing adaptive fruit loops threshold and support cuts"},
        {"fruitloops_support_signal_sum", "sum of final-map signal over pixels passing adaptive fruit loops threshold and support cuts"},
        {"fruitloops_support_x_span_arcsec", "x span of final-map pixels passing adaptive fruit loops threshold and support cuts"},
        {"fruitloops_support_y_span_arcsec", "y span of final-map pixels passing adaptive fruit loops threshold and support cuts"},
        {"rfi_masked_samples", "number of timestream samples masked by beammap rfi_mask"},
        {"rfi_masked_scans", "number of scans with at least one sample masked by beammap rfi_mask"},
        {"scan_band_masked_samples", "number of timestream samples masked by beammap scan_band_mask"},
        {"scan_band_masked_rows", "number of detector-map edge rows flagged by beammap scan_band_mask"},
        {"scan_band_masked_edge", "scan-band edge code (0 none, 1 top, 2 bottom, 3 both)"},
        {"scan_band_mask_rejected", "1 if scan_band_mask proposed a mask but rejected it due to max_flagged_fraction"},
        {"fit_bound_nhit", "number of fitted parameters at lower/upper bounds"},
        {"fit_bound_code", "bitmask of bound-hit parameters (see metadata legend)"},
        {"fit_bound_amp", "bound state for amplitude (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_x", "bound state for fitted x center (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_y", "bound state for fitted y center (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_a", "bound state for fitted a sigma/FWHM (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_b", "bound state for fitted b sigma/FWHM (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_angle", "bound state for fitted angle (-1 lower, 0 none, +1 upper)"},
        {"fit_init_amp", "initial amplitude used by Gaussian fitter"},
        {"fit_init_x_t", "initial x position converted to arcsec offset"},
        {"fit_init_y_t", "initial y position converted to arcsec offset"},
        {"fit_init_a_fwhm", "initial a FWHM implied by fitter initialization"},
        {"fit_init_b_fwhm", "initial b FWHM implied by fitter initialization"},
        {"fit_low_a_fwhm", "active lower bound for a FWHM"},
        {"fit_high_a_fwhm", "active upper bound for a FWHM"},
        {"fit_low_b_fwhm", "active lower bound for b FWHM"},
        {"fit_high_b_fwhm", "active upper bound for b FWHM"},
        {"prior_init_mode", "prior-init mode code (0 blind, 1 previous, 2 prior, -1 skipped/not fit)"},
        {"prior_used", "1 if the final initialization seed came from priors, else 0"},
        {"prior_fallback_blind", "1 if priors were attempted but blind fallback was used, else 0"},
        {"prior_no_candidate_reason", "reason code when priors produced no accepted candidate (see metadata legend)"},
        {"prior_slot_index", "matched prior slot index for the chosen prior-guided seed"},
        {"prior_match_d2", "Mahalanobis d^2 of chosen prior-guided seed in prior frame"},
        {"prior_match_score", "prior ranking score of chosen prior-guided seed"},
        {"prior_candidate_snr", "S/N metric of chosen prior-guided seed candidate"},
        {"prior_n_candidates", "number of weighted pixels above prior min_snr before top-N truncation"},
        {"prior_n_candidates_keep", "number of top-ranked candidates retained for prior scoring"},
        {"prior_n_candidates_gate", "number of retained candidates that passed the prior d^2 gate"},
        {"prior_candidate_x_t_raw", "chosen prior-guided candidate x offset before prior-frame transforms"},
        {"prior_candidate_y_t_raw", "chosen prior-guided candidate y offset before prior-frame transforms"},
        {"prior_candidate_x_t_prior", "chosen prior-guided candidate x offset in the prior frame"},
        {"prior_candidate_y_t_prior", "chosen prior-guided candidate y offset in the prior frame"},
        {"prior_center_x_t", "array-center x offset subtracted before prior matching"},
        {"prior_center_y_t", "array-center y offset subtracted before prior matching"},
        {"prior_derot_elev", "derotation elevation used for prior-frame matching"},
        {"prior_slot_x_t", "matched prior slot x center in the prior frame"},
        {"prior_slot_y_t", "matched prior slot y center in the prior frame"},
        {"prior_slot_sx", "matched prior slot soft x sigma"},
        {"prior_slot_sy", "matched prior slot soft y sigma"},
        {"final_prior_slot_index", "nearest prior slot index for final detector position in the prior frame"},
        {"final_prior_d2", "nearest-slot Mahalanobis d^2 for final detector position in the soft-prior frame"},
        {"x_t_raw", "raw x position before reference subtraction/derotation"},
        {"y_t_raw", "raw y position before reference subtraction/derotation"},
        {"x_t", get_description("x_t", "detector x position")},
        {"y_t", get_description("y_t", "detector y position")},
        {"x_t_derot", "detector x position after derotation transform"},
        {"y_t_derot", "detector y position after derotation transform"},
        {"a_fwhm", get_description("a_fwhm", "fitted major-axis FWHM")},
        {"a_fwhm_err", get_description("a_fwhm_err", "fitted major-axis FWHM uncertainty")},
        {"b_fwhm", get_description("b_fwhm", "fitted minor-axis FWHM")},
        {"b_fwhm_err", get_description("b_fwhm_err", "fitted minor-axis FWHM uncertainty")},
        {"angle", get_description("angle", "fitted beam angle")},
        {"angle_err", get_description("angle_err", "fitted beam angle uncertainty")},
        {"flxscale", get_description("flxscale", "flux conversion factor")},
        {"sens", get_description("sens", "detector sensitivity")}
    };

    for (const auto &key: fit_qc_header) {
        fit_qc_meta[key].push_back("units: " + fit_qc_units[key]);
        fit_qc_meta[key].push_back(fit_qc_desc[key]);
    }
    fit_qc_meta["flag2"].push_back("Good=0");
    fit_qc_meta["flag2"].push_back("BadFit=1");
    fit_qc_meta["flag2"].push_back("AzFWHM=2");
    fit_qc_meta["flag2"].push_back("ElFWHM=4");
    fit_qc_meta["flag2"].push_back("Sig2Noise=8");
    fit_qc_meta["flag2"].push_back("Sens=16");
    fit_qc_meta["flag2"].push_back("Position=32");
    fit_qc_meta["flag2"].push_back("PriorDist=64");
    fit_qc_meta["flag2"].push_back("NetworkPos=128");
    fit_qc_meta["cal_amp_method"].push_back("0: Gaussian fit amplitude fallback");
    fit_qc_meta["cal_amp_method"].push_back("1: empirical array-template matched amplitude");
    fit_qc_meta["fit_bound_code"].push_back("bit 0: amp lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 1: amp upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 2: x lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 3: x upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 4: y lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 5: y upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 6: a lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 7: a upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 8: b lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 9: b upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 10: angle lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 11: angle upper");
    fit_qc_meta["prior_init_mode"].push_back("-1: skipped before fitting on last attempted iteration");
    fit_qc_meta["prior_init_mode"].push_back("0: blind seed");
    fit_qc_meta["prior_init_mode"].push_back("1: previous-iteration seed");
    fit_qc_meta["prior_init_mode"].push_back("2: prior-guided seed");
    fit_qc_meta["prior_no_candidate_reason"].push_back("0: none");
    fit_qc_meta["prior_no_candidate_reason"].push_back("1: no slot group for (array,nw)");
    fit_qc_meta["prior_no_candidate_reason"].push_back("2: no valid weighted pixels");
    fit_qc_meta["prior_no_candidate_reason"].push_back("3: invalid robust sigma estimate");
    fit_qc_meta["prior_no_candidate_reason"].push_back("4: no candidates above min_snr");
    fit_qc_meta["prior_no_candidate_reason"].push_back("5: all retained candidates failed max_d2 gate");
    fit_qc_meta["scan_band_masked_edge"].push_back("0: none");
    fit_qc_meta["scan_band_masked_edge"].push_back("1: top");
    fit_qc_meta["scan_band_masked_edge"].push_back("2: bottom");
    fit_qc_meta["scan_band_masked_edge"].push_back("3: both");

    to_ecsv_from_matrix(fit_qc_filename, fit_qc_table, fit_qc_header, fit_qc_meta);
    logger->info("done writing beammap fit qc table {}.ecsv", fit_qc_filename);
}

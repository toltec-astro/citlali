#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

bool Beammap::is_beammap_locator_iter(Eigen::Index iter) const {
    if (!beammap_phase_split_enabled) {
        return iter <= 0;
    }
    return iter == static_cast<Eigen::Index>(beammap_locator_iter);
}

bool Beammap::is_beammap_measurement_iter(Eigen::Index iter) const {
    if (!beammap_phase_split_enabled) {
        return iter > 0;
    }
    return iter >= static_cast<Eigen::Index>(beammap_measurement_start_iter);
}

bool Beammap::is_beammap_first_measurement_iter(Eigen::Index iter) const {
    if (!beammap_phase_split_enabled) {
        return iter == 1;
    }
    return iter == static_cast<Eigen::Index>(beammap_measurement_start_iter);
}

bool Beammap::has_completed_beammap_measurement_iter(Eigen::Index iter) const {
    if (!beammap_phase_split_enabled) {
        return iter > 1;
    }
    return iter > static_cast<Eigen::Index>(beammap_measurement_start_iter);
}

std::string Beammap::beammap_iter_phase_name(Eigen::Index iter) const {
    if (!beammap_phase_split_enabled) {
        return "legacy";
    }
    if (is_beammap_locator_iter(iter)) {
        return "locator";
    }
    if (is_beammap_first_measurement_iter(iter)) {
        return "measurement_start";
    }
    if (is_beammap_measurement_iter(iter)) {
        return "measurement";
    }
    return "pre_measurement";
}

std::filesystem::path Beammap::resolve_soft_priors_filepath() const {
    namespace fs = std::filesystem;

    if (beammap_priors_filepath.empty() || beammap_priors_filepath == "null") {
        return {};
    }

    fs::path requested(beammap_priors_filepath);
    std::vector<fs::path> candidates;

    if (requested.is_absolute()) {
        candidates.push_back(requested);
    }
    else {
        candidates.push_back(requested);

        fs::path source_path(__FILE__);
        if (source_path.is_relative()) {
            source_path = fs::current_path() / source_path;
        }
        source_path = source_path.lexically_normal();
        fs::path repo_root = source_path;
        for (int i = 0; i < 5 && !repo_root.empty(); ++i) {
            repo_root = repo_root.parent_path();
        }
        if (!repo_root.empty()) {
            candidates.push_back(repo_root / requested);
        }
    }

    for (const auto &candidate : candidates) {
        try {
            if (fs::exists(candidate)) {
                return fs::absolute(candidate).lexically_normal();
            }
        }
        catch (const std::exception &) {
        }
    }

    return {};
}

bool Beammap::load_soft_priors() {
    beammap_soft_prior_slots.clear();
    beammap_soft_priors_loaded = false;
    beammap_soft_priors_are_centered = false;
    beammap_soft_priors_are_derotated = false;

    if (!beammap_priors_enabled) {
        return false;
    }

    if (beammap_priors_filepath.empty() || beammap_priors_filepath == "null") {
        logger->warn("beammap priors filepath is empty/null");
        return false;
    }
    const auto resolved_priors_filepath = resolve_soft_priors_filepath();
    if (resolved_priors_filepath.empty()) {
        logger->warn("beammap priors file does not exist: {}", beammap_priors_filepath);
        return false;
    }
    if (resolved_priors_filepath.string() != beammap_priors_filepath) {
        logger->info("beammap priors resolved {} -> {}", beammap_priors_filepath, resolved_priors_filepath.string());
        beammap_priors_filepath = resolved_priors_filepath.string();
    }

    auto [priors_table, priors_header, priors_meta] =
        to_map_from_ecsv_mixted_type(beammap_priors_filepath);
    static_cast<void>(priors_header);

    auto prior_frame_it = priors_meta.find("prior_frame");
    if (prior_frame_it != priors_meta.end()) {
        std::string prior_frame = prior_frame_it->second;
        std::transform(prior_frame.begin(), prior_frame.end(), prior_frame.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        beammap_soft_priors_are_centered = (prior_frame.find("center") != std::string::npos);
        beammap_soft_priors_are_derotated = (prior_frame.find("derot") != std::string::npos);
    }

    const std::vector<std::string> required_columns = {
        "array",
        "nw",
        "slot_index",
        "x_rel_med_arcsec",
        "y_rel_med_arcsec",
        "x_rel_sigma_soft_arcsec",
        "y_rel_sigma_soft_arcsec"
    };

    for (const auto &col : required_columns) {
        if (priors_table.find(col) == priors_table.end()) {
            logger->warn("beammap priors missing required column '{}': {}", col, beammap_priors_filepath);
            return false;
        }
    }

    const Eigen::Index n_rows = priors_table.at("array").size();
    for (const auto &col : required_columns) {
        if (priors_table.at(col).size() != n_rows) {
            logger->warn("beammap priors column '{}' has wrong size {} (expected {})",
                         col, priors_table.at(col).size(), n_rows);
            return false;
        }
    }
    if (n_rows <= 0) {
        logger->warn("beammap priors table has no rows: {}", beammap_priors_filepath);
        return false;
    }

    constexpr double sigma_floor_arcsec = 1e-3;
    Eigen::Index n_valid_rows = 0;
    Eigen::Index n_dropped_rows = 0;
    for (Eigen::Index i = 0; i < n_rows; ++i) {
        const double array_d = priors_table.at("array")(i);
        const double nw_d = priors_table.at("nw")(i);
        const double slot_d = priors_table.at("slot_index")(i);
        const double x_d = priors_table.at("x_rel_med_arcsec")(i);
        const double y_d = priors_table.at("y_rel_med_arcsec")(i);
        const double sx_d = priors_table.at("x_rel_sigma_soft_arcsec")(i);
        const double sy_d = priors_table.at("y_rel_sigma_soft_arcsec")(i);

        if (!(std::isfinite(array_d) && std::isfinite(nw_d) && std::isfinite(slot_d) &&
              std::isfinite(x_d) && std::isfinite(y_d) && std::isfinite(sx_d) && std::isfinite(sy_d))) {
            n_dropped_rows++;
            continue;
        }

        const int array = static_cast<int>(std::lround(array_d));
        const int nw = static_cast<int>(std::lround(nw_d));

        SoftPriorSlot slot;
        slot.slot_index = static_cast<int>(std::lround(slot_d));
        slot.x_arcsec = x_d;
        slot.y_arcsec = y_d;
        slot.sx_arcsec = std::max(sigma_floor_arcsec, std::abs(sx_d));
        slot.sy_arcsec = std::max(sigma_floor_arcsec, std::abs(sy_d));

        beammap_soft_prior_slots[{array, nw}].push_back(slot);
        n_valid_rows++;
    }

    for (auto &entry : beammap_soft_prior_slots) {
        auto &slots = entry.second;
        std::sort(slots.begin(), slots.end(),
                  [](const SoftPriorSlot &a, const SoftPriorSlot &b) {
                      if (a.slot_index == b.slot_index) {
                          return a.y_arcsec < b.y_arcsec;
                      }
                      return a.slot_index < b.slot_index;
                  });
    }

    if (beammap_soft_prior_slots.empty()) {
        logger->warn("beammap priors produced no valid slots: {}", beammap_priors_filepath);
        return false;
    }

    Eigen::Index n_slots = 0;
    for (const auto &entry : beammap_soft_prior_slots) {
        n_slots += static_cast<Eigen::Index>(entry.second.size());
    }
    beammap_soft_priors_loaded = true;
    logger->info("loaded beammap soft priors: {} slot rows across {} (array,nw) groups from {}",
                 n_slots, beammap_soft_prior_slots.size(), beammap_priors_filepath);
    if (n_dropped_rows > 0) {
        logger->warn("dropped {} non-finite prior rows (kept {})", n_dropped_rows, n_valid_rows);
    }

    return true;
}

bool Beammap::find_map_weighted_peak(Eigen::Index map_index, Eigen::Index &best_row,
                                     Eigen::Index &best_col, double &best_snr) const {
    best_row = -1;
    best_col = -1;
    best_snr = -std::numeric_limits<double>::infinity();

    if (map_index < 0 || map_index >= n_maps) {
        return false;
    }

    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    if (sig.rows() <= 0 || sig.cols() <= 0 || wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
        return false;
    }

    const double center_row = static_cast<double>(sig.rows() - 1) / 2.0;
    const double center_col = static_cast<double>(sig.cols() - 1) / 2.0;
    const double radius_pix = map_fitter.fitting_region_pix;
    const double radius2 = radius_pix * radius_pix;

    auto scan = [&](bool apply_radius) {
        bool found = false;
        for (Eigen::Index row = 0; row < sig.rows(); ++row) {
            for (Eigen::Index col = 0; col < sig.cols(); ++col) {
                const double s = sig(row, col);
                const double w = wt(row, col);
                if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                    continue;
                }
                if (apply_radius) {
                    const double dr = static_cast<double>(row) - center_row;
                    const double dc = static_cast<double>(col) - center_col;
                    if (dr * dr + dc * dc >= radius2) {
                        continue;
                    }
                }
                const double snr = s * std::sqrt(w);
                if (!std::isfinite(snr)) {
                    continue;
                }
                if (!found || snr > best_snr) {
                    best_row = row;
                    best_col = col;
                    best_snr = snr;
                    found = true;
                }
            }
        }
        return found;
    };

    if (radius_pix > 0.0 && scan(true)) {
        return true;
    }
    return scan(false);
}

void Beammap::configure_detector_source_centers_from_previous_fit() {
    if (map_grouping != "detector") {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        return;
    }

    if (!is_beammap_measurement_iter(current_iter)) {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        logger->info(
            "beammap detector source centers unavailable on iter {} phase={}: locator pass has no previous fits "
            "(ptc_mask_radius={:.3f} arcsec)",
            current_iter, beammap_iter_phase_name(current_iter), ptcproc.mask_radius_arcsec);
        return;
    }

    if (p0.rows() != n_maps || p0.cols() < 3 || good_fits.size() != n_maps) {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        logger->warn(
            "beammap detector source centers unavailable on iter {}: previous-fit state is incomplete "
            "(p0={}x{}, good_fits={})",
            current_iter, p0.rows(), p0.cols(), good_fits.size());
        return;
    }

    ptcproc.fruit_loops_source_lat = Eigen::VectorXd::Zero(n_maps);
    ptcproc.fruit_loops_source_lon = Eigen::VectorXd::Zero(n_maps);
    ptcproc.fruit_loops_source_valid = Eigen::VectorXi::Zero(n_maps);
    Eigen::VectorXd kernel_source_a_fwhm_rad = Eigen::VectorXd::Zero(n_maps);
    Eigen::VectorXd kernel_source_b_fwhm_rad = Eigen::VectorXd::Zero(n_maps);

    Eigen::Index n_valid = 0;
    Eigen::Index n_valid_fwhm = 0;
    std::vector<double> fwhm_arcsec_values;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        if (!good_fits(i) ||
            !std::isfinite(p0(i, 0)) || p0(i, 0) <= 0.0 ||
            !std::isfinite(p0(i, 1)) || !std::isfinite(p0(i, 2))) {
            continue;
        }
        ptcproc.fruit_loops_source_lat(i) =
            (p0(i, 2) - (omb.n_rows - 1) / 2.0) * omb.pixel_size_rad;
        ptcproc.fruit_loops_source_lon(i) =
            (p0(i, 1) - (omb.n_cols - 1) / 2.0) * omb.pixel_size_rad;
        ptcproc.fruit_loops_source_valid(i) = 1;
        n_valid++;

        if (p0.cols() > 4 &&
            std::isfinite(p0(i, 3)) && p0(i, 3) > 0.0 &&
            std::isfinite(p0(i, 4)) && p0(i, 4) > 0.0) {
            kernel_source_a_fwhm_rad(i) = STD_TO_FWHM * omb.pixel_size_rad * p0(i, 3);
            kernel_source_b_fwhm_rad(i) = STD_TO_FWHM * omb.pixel_size_rad * p0(i, 4);
            const double mean_fwhm_arcsec =
                RAD_TO_ASEC * (kernel_source_a_fwhm_rad(i) + kernel_source_b_fwhm_rad(i)) / 2.0;
            if (std::isfinite(mean_fwhm_arcsec) && mean_fwhm_arcsec > 0.0) {
                fwhm_arcsec_values.push_back(mean_fwhm_arcsec);
                n_valid_fwhm++;
            }
        }
    }

    logger->info(
        "beammap detector source centers using previous-fit centers for {}/{} detector maps "
        "on iter {} (ptc_mask_radius={:.3f} arcsec)",
        n_valid, n_maps, current_iter, ptcproc.mask_radius_arcsec);

    if (rtcproc.run_kernel) {
        double median_fwhm_arcsec = std::numeric_limits<double>::quiet_NaN();
        if (!fwhm_arcsec_values.empty()) {
            std::sort(fwhm_arcsec_values.begin(), fwhm_arcsec_values.end());
            median_fwhm_arcsec = fwhm_arcsec_values[fwhm_arcsec_values.size() / 2];
        }
        rtcproc.kernel.set_source_centers(ptcproc.fruit_loops_source_lat,
                                          ptcproc.fruit_loops_source_lon,
                                          ptcproc.fruit_loops_source_valid,
                                          kernel_source_a_fwhm_rad,
                                          kernel_source_b_fwhm_rad);
        logger->info(
            "beammap detector kernel placement using previous-fit centers for {}/{} detector maps on iter {}; fitted kernel FWHM available for {}/{} maps (median={:.3f} arcsec)",
            n_valid, n_maps, current_iter, n_valid_fwhm, n_maps, median_fwhm_arcsec);
    }
}

double Beammap::get_prior_derot_elev_rad() const {
    double derot_elev_rad = 0.0;
    auto tel_el_it = telescope.tel_data.find("TelElAct");
    if (tel_el_it != telescope.tel_data.end() && tel_el_it->second.size() > 0) {
        derot_elev_rad = tel_el_it->second.mean();
    }
    if (!std::isfinite(derot_elev_rad)) {
        derot_elev_rad = 0.0;
    }
    if (std::abs(derot_elev_rad) > pi) {
        derot_elev_rad *= DEG_TO_RAD;
    }
    return derot_elev_rad;
}

double Beammap::effective_prior_max_d2() const {
    return is_beammap_measurement_iter(current_iter)
               ? beammap_priors_max_d2_after_iter0
               : beammap_priors_max_d2_iter0;
}

double Beammap::effective_prior_score_lambda() const {
    return is_beammap_measurement_iter(current_iter)
               ? beammap_priors_score_lambda_after_iter0
               : beammap_priors_score_lambda_iter0;
}

bool Beammap::observed_to_prior_frame(int array, double x_raw_arcsec, double y_raw_arcsec,
                                      double derot_elev_rad, double &x_prior_arcsec,
                                      double &y_prior_arcsec, double *center_x_arcsec,
                                      double *center_y_arcsec,
                                      bool apply_empirical_alignment) const {
    if (!std::isfinite(x_raw_arcsec) || !std::isfinite(y_raw_arcsec)) {
        return false;
    }

    double x = x_raw_arcsec;
    double y = y_raw_arcsec;
    double center_x = std::numeric_limits<double>::quiet_NaN();
    double center_y = std::numeric_limits<double>::quiet_NaN();

    if (beammap_soft_priors_are_centered) {
        auto x_it = beammap_prior_array_center_x_arcsec.find(array);
        auto y_it = beammap_prior_array_center_y_arcsec.find(array);
        if (x_it == beammap_prior_array_center_x_arcsec.end() ||
            y_it == beammap_prior_array_center_y_arcsec.end() ||
            !std::isfinite(x_it->second) || !std::isfinite(y_it->second)) {
            return false;
        }
        center_x = x_it->second;
        center_y = y_it->second;
        x -= center_x;
        y -= center_y;
    }

    if (center_x_arcsec != nullptr) {
        *center_x_arcsec = center_x;
    }
    if (center_y_arcsec != nullptr) {
        *center_y_arcsec = center_y;
    }

    if (beammap_soft_priors_are_derotated && telescope.pixel_axes == "altaz") {
        if (!std::isfinite(derot_elev_rad)) {
            derot_elev_rad = 0.0;
        }
        if (std::abs(derot_elev_rad) > pi) {
            derot_elev_rad *= DEG_TO_RAD;
        }
        const double cos_rot = std::cos(-derot_elev_rad);
        const double sin_rot = std::sin(-derot_elev_rad);
        const double rot_az_off = cos_rot * x - sin_rot * y;
        const double rot_alt_off = sin_rot * x + cos_rot * y;
        x = -rot_az_off;
        y = -rot_alt_off;
    }

    if (apply_empirical_alignment) {
        auto align_it = beammap_prior_array_alignment.find(array);
        if (align_it != beammap_prior_array_alignment.end() && align_it->second.valid) {
            const auto &align = align_it->second;
            const double x_rot = align.cos_theta * x - align.sin_theta * y;
            const double y_rot = align.sin_theta * x + align.cos_theta * y;
            x = x_rot + align.dx_arcsec;
            y = y_rot + align.dy_arcsec;
        }
    }

    x_prior_arcsec = x;
    y_prior_arcsec = y;
    return std::isfinite(x_prior_arcsec) && std::isfinite(y_prior_arcsec);
}

bool Beammap::match_prior_slot(int array, int nw, double x_prior_arcsec, double y_prior_arcsec,
                               double &best_d2, int &best_slot, double *slot_x_arcsec,
                               double *slot_y_arcsec, double *slot_sx_arcsec,
                               double *slot_sy_arcsec) const {
    best_d2 = std::numeric_limits<double>::infinity();
    best_slot = -1;
    auto slots_it = beammap_soft_prior_slots.find({array, nw});
    if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty() ||
        !std::isfinite(x_prior_arcsec) || !std::isfinite(y_prior_arcsec)) {
        return false;
    }

    for (const auto &slot : slots_it->second) {
        if (!std::isfinite(slot.x_arcsec) || !std::isfinite(slot.y_arcsec) ||
            !std::isfinite(slot.sx_arcsec) || !std::isfinite(slot.sy_arcsec) ||
            slot.sx_arcsec <= 0.0 || slot.sy_arcsec <= 0.0) {
            continue;
        }
        const double dx = (x_prior_arcsec - slot.x_arcsec) / slot.sx_arcsec;
        const double dy = (y_prior_arcsec - slot.y_arcsec) / slot.sy_arcsec;
        const double d2 = dx * dx + dy * dy;
        if (std::isfinite(d2) && d2 < best_d2) {
            best_d2 = d2;
            best_slot = slot.slot_index;
            if (slot_x_arcsec != nullptr) {
                *slot_x_arcsec = slot.x_arcsec;
            }
            if (slot_y_arcsec != nullptr) {
                *slot_y_arcsec = slot.y_arcsec;
            }
            if (slot_sx_arcsec != nullptr) {
                *slot_sx_arcsec = slot.sx_arcsec;
            }
            if (slot_sy_arcsec != nullptr) {
                *slot_sy_arcsec = slot.sy_arcsec;
            }
        }
    }
    return std::isfinite(best_d2);
}

void Beammap::update_prior_frame_estimates() {
    beammap_prior_array_center_x_arcsec.clear();
    beammap_prior_array_center_y_arcsec.clear();
    beammap_prior_array_alignment.clear();

    std::map<int, std::vector<double>> x_by_array;
    std::map<int, std::vector<double>> y_by_array;
    std::set<int> arrays_missing;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        arrays_missing.insert(static_cast<int>(maps_to_arrays(i)));
    }

    Eigen::Index n_prev = 0;
    if (is_beammap_measurement_iter(current_iter) && p0.rows() == n_maps && p0.cols() > 2) {
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            if (i < good_fits.size() && !good_fits(i)) {
                continue;
            }
            if (fit_diag_bound_nhit.size() == n_maps && fit_diag_bound_nhit(i) > 0) {
                continue;
            }
            if (!(std::isfinite(p0(i, 0)) && p0(i, 0) > 0.0 &&
                  std::isfinite(p0(i, 1)) && std::isfinite(p0(i, 2)))) {
                continue;
            }
            const int array = static_cast<int>(maps_to_arrays(i));
            const double x_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 1) - (omb.n_cols - 1) / 2.0);
            const double y_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 2) - (omb.n_rows - 1) / 2.0);
            x_by_array[array].push_back(x_arcsec);
            y_by_array[array].push_back(y_arcsec);
            arrays_missing.erase(array);
            n_prev++;
        }
    }

    Eigen::Index n_blind = 0;
    if (!arrays_missing.empty()) {
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            const int array = static_cast<int>(maps_to_arrays(i));
            if (!arrays_missing.count(array)) {
                continue;
            }

            Eigen::Index peak_row = -1;
            Eigen::Index peak_col = -1;
            double peak_snr = -std::numeric_limits<double>::infinity();
            if (!find_map_weighted_peak(i, peak_row, peak_col, peak_snr)) {
                continue;
            }

            const double x_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (static_cast<double>(peak_col) - (omb.n_cols - 1) / 2.0);
            const double y_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (static_cast<double>(peak_row) - (omb.n_rows - 1) / 2.0);
            x_by_array[array].push_back(x_arcsec);
            y_by_array[array].push_back(y_arcsec);
            n_blind++;
        }
    }

    for (const auto &[array, xs] : x_by_array) {
        if (xs.empty()) {
            continue;
        }
        Eigen::Map<const Eigen::VectorXd> x_vec(xs.data(), static_cast<Eigen::Index>(xs.size()));
        auto y_it = y_by_array.find(array);
        if (y_it == y_by_array.end() || y_it->second.size() != xs.size()) {
            continue;
        }
        Eigen::Map<const Eigen::VectorXd> y_vec(y_it->second.data(), static_cast<Eigen::Index>(y_it->second.size()));
        beammap_prior_array_center_x_arcsec[array] = tula::alg::median(x_vec);
        beammap_prior_array_center_y_arcsec[array] = tula::alg::median(y_vec);
    }

    Eigen::Index n_alignment_matches = 0;
    if (beammap_priors_align_after_iter0 && is_beammap_measurement_iter(current_iter) &&
        p0.rows() == n_maps && p0.cols() > 2) {
        struct PriorPair {
            double obs_x = 0.0;
            double obs_y = 0.0;
            double slot_x = 0.0;
            double slot_y = 0.0;
        };
        std::map<int, std::vector<PriorPair>> pairs_by_array;
        std::vector<PriorPair> all_pairs;
        std::set<int> arrays_with_alignment_pairs;
        const double derot_elev_rad = get_prior_derot_elev_rad();

        for (Eigen::Index i = 0; i < n_maps; ++i) {
            if (i >= good_fits.size() || !good_fits(i)) {
                continue;
            }
            if (fit_diag_bound_nhit.size() == n_maps && fit_diag_bound_nhit(i) > 0) {
                continue;
            }
            if (!(std::isfinite(p0(i, 0)) && p0(i, 0) > 0.0 &&
                  std::isfinite(p0(i, 1)) && std::isfinite(p0(i, 2)))) {
                continue;
            }
            const int array = static_cast<int>(maps_to_arrays(i));
            const int nw = static_cast<int>(std::lround(calib.apt["nw"](i)));
            const double x_raw =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 1) - (omb.n_cols - 1) / 2.0);
            const double y_raw =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 2) - (omb.n_rows - 1) / 2.0);
            double x_prior = std::numeric_limits<double>::quiet_NaN();
            double y_prior = std::numeric_limits<double>::quiet_NaN();
            if (!observed_to_prior_frame(array, x_raw, y_raw, derot_elev_rad,
                                         x_prior, y_prior, nullptr, nullptr, false)) {
                continue;
            }
            double d2 = std::numeric_limits<double>::infinity();
            int slot_index = -1;
            double slot_x = std::numeric_limits<double>::quiet_NaN();
            double slot_y = std::numeric_limits<double>::quiet_NaN();
            if (!match_prior_slot(array, nw, x_prior, y_prior, d2, slot_index, &slot_x, &slot_y)) {
                continue;
            }
            static_cast<void>(slot_index);
            if (beammap_priors_alignment_max_d2 > 0.0 && d2 > beammap_priors_alignment_max_d2) {
                continue;
            }
            PriorPair pair{x_prior, y_prior, slot_x, slot_y};
            pairs_by_array[array].push_back(pair);
            all_pairs.push_back(pair);
            arrays_with_alignment_pairs.insert(array);
            n_alignment_matches++;
        }

        auto fit_prior_alignment = [&](const std::vector<PriorPair> &pairs,
                                       const std::string &label,
                                       PriorArrayAlignment &alignment) {
            if (pairs.size() < static_cast<std::size_t>(beammap_priors_alignment_min_matches)) {
                logger->debug("beammap prior alignment skipped {} matches={} min_matches={}",
                              label, pairs.size(), beammap_priors_alignment_min_matches);
                return false;
            }

            std::vector<double> dx_vals;
            std::vector<double> dy_vals;
            dx_vals.reserve(pairs.size());
            dy_vals.reserve(pairs.size());
            for (const auto &pair : pairs) {
                dx_vals.push_back(pair.slot_x - pair.obs_x);
                dy_vals.push_back(pair.slot_y - pair.obs_y);
            }
            Eigen::Map<Eigen::VectorXd> dx_vec(dx_vals.data(), static_cast<Eigen::Index>(dx_vals.size()));
            Eigen::Map<Eigen::VectorXd> dy_vec(dy_vals.data(), static_cast<Eigen::Index>(dy_vals.size()));
            double tx = tula::alg::median(dx_vec);
            double ty = tula::alg::median(dy_vec);

            double theta = 0.0;
            if (beammap_priors_alignment_fit_rotation) {
                double obs_mean_x = 0.0;
                double obs_mean_y = 0.0;
                double slot_mean_x = 0.0;
                double slot_mean_y = 0.0;
                for (const auto &pair : pairs) {
                    obs_mean_x += pair.obs_x + tx;
                    obs_mean_y += pair.obs_y + ty;
                    slot_mean_x += pair.slot_x;
                    slot_mean_y += pair.slot_y;
                }
                const double inv_n = 1.0 / static_cast<double>(pairs.size());
                obs_mean_x *= inv_n;
                obs_mean_y *= inv_n;
                slot_mean_x *= inv_n;
                slot_mean_y *= inv_n;

                double a = 0.0;
                double b = 0.0;
                for (const auto &pair : pairs) {
                    const double ox = pair.obs_x + tx - obs_mean_x;
                    const double oy = pair.obs_y + ty - obs_mean_y;
                    const double sx = pair.slot_x - slot_mean_x;
                    const double sy = pair.slot_y - slot_mean_y;
                    a += ox * sx + oy * sy;
                    b += ox * sy - oy * sx;
                }
                if (std::isfinite(a) && std::isfinite(b) &&
                    (std::abs(a) > 0.0 || std::abs(b) > 0.0)) {
                    theta = std::atan2(b, a);
                }
                const double max_theta = beammap_priors_alignment_max_rotation_deg * DEG_TO_RAD;
                if (!std::isfinite(theta) || std::abs(theta) > max_theta) {
                    logger->debug(
                        "beammap prior alignment {} rejected residual rotation {} deg (limit={} deg)",
                        label, theta * RAD_TO_DEG, beammap_priors_alignment_max_rotation_deg);
                    theta = 0.0;
                }
            }

            const double cos_theta = std::cos(theta);
            const double sin_theta = std::sin(theta);
            dx_vals.clear();
            dy_vals.clear();
            for (const auto &pair : pairs) {
                const double x_rot = cos_theta * pair.obs_x - sin_theta * pair.obs_y;
                const double y_rot = sin_theta * pair.obs_x + cos_theta * pair.obs_y;
                dx_vals.push_back(pair.slot_x - x_rot);
                dy_vals.push_back(pair.slot_y - y_rot);
            }
            Eigen::Map<Eigen::VectorXd> dx_vec_final(dx_vals.data(), static_cast<Eigen::Index>(dx_vals.size()));
            Eigen::Map<Eigen::VectorXd> dy_vec_final(dy_vals.data(), static_cast<Eigen::Index>(dy_vals.size()));
            tx = tula::alg::median(dx_vec_final);
            ty = tula::alg::median(dy_vec_final);

            double rss = 0.0;
            for (const auto &pair : pairs) {
                const double x_fit = cos_theta * pair.obs_x - sin_theta * pair.obs_y + tx;
                const double y_fit = sin_theta * pair.obs_x + cos_theta * pair.obs_y + ty;
                const double rx = x_fit - pair.slot_x;
                const double ry = y_fit - pair.slot_y;
                rss += rx * rx + ry * ry;
            }
            const double rms = std::sqrt(rss / static_cast<double>(pairs.size()));
            if (!(std::isfinite(tx) && std::isfinite(ty) && std::isfinite(rms))) {
                return false;
            }

            alignment.valid = true;
            alignment.cos_theta = cos_theta;
            alignment.sin_theta = sin_theta;
            alignment.theta_rad = theta;
            alignment.dx_arcsec = tx;
            alignment.dy_arcsec = ty;
            alignment.n_matches = static_cast<Eigen::Index>(pairs.size());
            alignment.rms_arcsec = rms;
            return true;
        };

        if (beammap_priors_alignment_scope == "common") {
            auto common_pairs = all_pairs;
            if (beammap_priors_alignment_common_support == "overlap_box" &&
                pairs_by_array.size() >= 2) {
                auto quantile = [](std::vector<double> values, double q) {
                    if (values.empty()) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    q = std::clamp(q, 0.0, 1.0);
                    std::sort(values.begin(), values.end());
                    const double pos = q * static_cast<double>(values.size() - 1);
                    const auto lo = static_cast<std::size_t>(std::floor(pos));
                    const auto hi = static_cast<std::size_t>(std::ceil(pos));
                    if (lo == hi) {
                        return values[lo];
                    }
                    const double frac = pos - static_cast<double>(lo);
                    return values[lo] * (1.0 - frac) + values[hi] * frac;
                };

                const double q_low = beammap_priors_alignment_common_support_quantile;
                const double q_high = 1.0 - beammap_priors_alignment_common_support_quantile;
                double overlap_x_low = -std::numeric_limits<double>::infinity();
                double overlap_x_high = std::numeric_limits<double>::infinity();
                double overlap_y_low = -std::numeric_limits<double>::infinity();
                double overlap_y_high = std::numeric_limits<double>::infinity();
                bool overlap_valid = true;

                for (const auto &[array, pairs] : pairs_by_array) {
                    static_cast<void>(array);
                    std::vector<double> xs;
                    std::vector<double> ys;
                    xs.reserve(pairs.size());
                    ys.reserve(pairs.size());
                    for (const auto &pair : pairs) {
                        if (std::isfinite(pair.slot_x) && std::isfinite(pair.slot_y)) {
                            xs.push_back(pair.slot_x);
                            ys.push_back(pair.slot_y);
                        }
                    }
                    const double x_low = quantile(xs, q_low);
                    const double x_high = quantile(xs, q_high);
                    const double y_low = quantile(ys, q_low);
                    const double y_high = quantile(ys, q_high);
                    if (!(std::isfinite(x_low) && std::isfinite(x_high) &&
                          std::isfinite(y_low) && std::isfinite(y_high))) {
                        overlap_valid = false;
                        break;
                    }
                    overlap_x_low = std::max(overlap_x_low, x_low);
                    overlap_x_high = std::min(overlap_x_high, x_high);
                    overlap_y_low = std::max(overlap_y_low, y_low);
                    overlap_y_high = std::min(overlap_y_high, y_high);
                }

                if (overlap_valid && overlap_x_low < overlap_x_high &&
                    overlap_y_low < overlap_y_high) {
                    std::vector<PriorPair> filtered_pairs;
                    filtered_pairs.reserve(all_pairs.size());
                    for (const auto &pair : all_pairs) {
                        if (pair.slot_x >= overlap_x_low && pair.slot_x <= overlap_x_high &&
                            pair.slot_y >= overlap_y_low && pair.slot_y <= overlap_y_high) {
                            filtered_pairs.push_back(pair);
                        }
                    }
                    if (filtered_pairs.size() >= static_cast<std::size_t>(beammap_priors_alignment_min_matches)) {
                        common_pairs.swap(filtered_pairs);
                    }
                    logger->info(
                        "beammap prior common alignment overlap_box (iter {}): q={} x=[{}, {}] y=[{}, {}] kept={}/{}",
                        current_iter, beammap_priors_alignment_common_support_quantile,
                        overlap_x_low, overlap_x_high, overlap_y_low, overlap_y_high,
                        common_pairs.size(), all_pairs.size());
                }
                else {
                    logger->debug(
                        "beammap prior common alignment overlap_box skipped: invalid overlap x=[{}, {}] y=[{}, {}]",
                        overlap_x_low, overlap_x_high, overlap_y_low, overlap_y_high);
                }
            }

            PriorArrayAlignment alignment;
            if (fit_prior_alignment(common_pairs, "scope=common", alignment)) {
                for (int array : arrays_with_alignment_pairs) {
                    beammap_prior_array_alignment[array] = alignment;
                }
                logger->info(
                    "beammap prior empirical alignment (iter {} scope=common): arrays={} matches={} dx={} dy={} rot_deg={} rms={}",
                    current_iter, arrays_with_alignment_pairs.size(), alignment.n_matches,
                    alignment.dx_arcsec, alignment.dy_arcsec,
                    alignment.theta_rad * RAD_TO_DEG, alignment.rms_arcsec);
            }
        }
        else {
            for (auto &[array, pairs] : pairs_by_array) {
                PriorArrayAlignment alignment;
                if (!fit_prior_alignment(pairs, fmt::format("array={}", array), alignment)) {
                    continue;
                }
                beammap_prior_array_alignment[array] = alignment;

                logger->info(
                    "beammap prior empirical alignment (iter {} array={}): matches={} dx={} dy={} rot_deg={} rms={}",
                    current_iter, array, alignment.n_matches, alignment.dx_arcsec,
                    alignment.dy_arcsec, alignment.theta_rad * RAD_TO_DEG, alignment.rms_arcsec);
            }
        }
    }

    logger->info(
        "beammap priors frame estimate (iter {}): previous={} blind={} arrays={} alignment_matches={} aligned_arrays={}",
        current_iter, n_prev, n_blind, beammap_prior_array_center_x_arcsec.size(),
        n_alignment_matches, beammap_prior_array_alignment.size());
}

bool Beammap::choose_prior_guided_init(Eigen::Index map_index, double &init_row, double &init_col) {
    init_row = -99.0;
    init_col = -99.0;

    auto set_prior_diag = [&](PriorDiagColumn col, double value) {
        if (map_index >= 0 && map_index < prior_diag_values.rows() &&
            col >= 0 && col < prior_diag_values.cols()) {
            prior_diag_values(map_index, col) = value;
        }
    };

    constexpr int prior_reason_none = 0;
    constexpr int prior_reason_no_slot_group = 1;
    constexpr int prior_reason_no_valid_weighted_pixels = 2;
    constexpr int prior_reason_invalid_sigma = 3;
    constexpr int prior_reason_below_min_snr = 4;
    constexpr int prior_reason_gate_rejected = 5;

    if (!beammap_soft_priors_loaded || map_grouping != "detector") {
        return false;
    }
    if (map_index < 0 || map_index >= n_maps || map_index >= calib.n_dets) {
        return false;
    }
    if (map_index >= maps_to_arrays.size() || map_index >= calib.apt["nw"].size()) {
        return false;
    }

    const int array = static_cast<int>(maps_to_arrays(map_index));
    const int nw = static_cast<int>(std::lround(calib.apt["nw"](map_index)));
    auto slots_it = beammap_soft_prior_slots.find({array, nw});
    if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_no_slot_group);
        return false;
    }

    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    if (sig.rows() <= 0 || sig.cols() <= 0 || wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_no_valid_weighted_pixels);
        return false;
    }

    struct Candidate {
        double snr = 0.0;
        Eigen::Index row = 0;
        Eigen::Index col = 0;
    };

    std::vector<double> valid_signal;
    std::vector<double> valid_weight;
    valid_signal.reserve(static_cast<std::size_t>(sig.size()));
    valid_weight.reserve(static_cast<std::size_t>(sig.size()));
    for (Eigen::Index row = 0; row < sig.rows(); ++row) {
        for (Eigen::Index col = 0; col < sig.cols(); ++col) {
            const double s = sig(row, col);
            const double w = wt(row, col);
            if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                continue;
            }
            valid_signal.push_back(s);
            valid_weight.push_back(w);
        }
    }
    if (valid_signal.empty()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_no_valid_weighted_pixels);
        return false;
    }

    Eigen::Map<Eigen::VectorXd> sig_vec(valid_signal.data(), static_cast<Eigen::Index>(valid_signal.size()));
    const double sig_med = tula::alg::median(sig_vec);
    Eigen::VectorXd sig_abs_dev = (sig_vec.array() - sig_med).abs().matrix();
    double sig_sigma = 1.4826 * tula::alg::median(sig_abs_dev);
    if (!std::isfinite(sig_sigma) || sig_sigma <= std::numeric_limits<double>::epsilon()) {
        sig_sigma = engine_utils::calc_std_dev(sig_vec);
    }
    if (!std::isfinite(sig_sigma) || sig_sigma <= std::numeric_limits<double>::epsilon()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_invalid_sigma);
        return false;
    }

    Eigen::Map<Eigen::VectorXd> wt_vec(valid_weight.data(), static_cast<Eigen::Index>(valid_weight.size()));
    double wt_med = tula::alg::median(wt_vec);
    if (!std::isfinite(wt_med) || wt_med <= std::numeric_limits<double>::epsilon()) {
        wt_med = 1.0;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(sig.size()));
    for (Eigen::Index row = 0; row < sig.rows(); ++row) {
        for (Eigen::Index col = 0; col < sig.cols(); ++col) {
            const double s = sig(row, col);
            const double w = wt(row, col);
            if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                continue;
            }
            const double snr = ((s - sig_med) / sig_sigma) * std::sqrt(w / wt_med);
            if (!std::isfinite(snr) || snr < beammap_priors_min_snr) {
                continue;
            }
            candidates.push_back({snr, row, col});
        }
    }
    if (candidates.empty()) {
        logger->debug("beammap priors init map={} no candidates above min_snr={:.4g} (med={:.4g} sigma={:.4g} wt_med={:.4g})",
                      map_index, beammap_priors_min_snr, sig_med, sig_sigma, wt_med);
        set_prior_diag(prior_n_candidates_col, 0.0);
        set_prior_diag(prior_n_candidates_keep_col, 0.0);
        set_prior_diag(prior_n_candidates_gate_col, 0.0);
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_below_min_snr);
        return false;
    }

    set_prior_diag(prior_n_candidates_col, static_cast<double>(candidates.size()));

    const std::size_t n_keep = std::min<std::size_t>(
        candidates.size(), static_cast<std::size_t>(std::max(1, beammap_priors_candidate_top_n)));
    set_prior_diag(prior_n_candidates_keep_col, static_cast<double>(n_keep));
    std::partial_sort(candidates.begin(), candidates.begin() + n_keep, candidates.end(),
                      [](const Candidate &a, const Candidate &b) { return a.snr > b.snr; });

    const double col0 = static_cast<double>(omb.n_cols - 1) / 2.0;
    const double row0 = static_cast<double>(omb.n_rows - 1) / 2.0;
    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    double derot_elev_rad = get_prior_derot_elev_rad();
    set_prior_diag(prior_derot_elev_col, derot_elev_rad);
    const double prior_max_d2 = effective_prior_max_d2();
    const double prior_score_lambda = effective_prior_score_lambda();

    bool found = false;
    double best_score = -std::numeric_limits<double>::infinity();
    double best_snr = -std::numeric_limits<double>::infinity();
    double best_d2 = std::numeric_limits<double>::infinity();
    Eigen::Index best_row = -1;
    Eigen::Index best_col = -1;
    int best_slot = -1;
    double best_x_raw = std::numeric_limits<double>::quiet_NaN();
    double best_y_raw = std::numeric_limits<double>::quiet_NaN();
    double best_x_prior = std::numeric_limits<double>::quiet_NaN();
    double best_y_prior = std::numeric_limits<double>::quiet_NaN();
    double best_slot_x = std::numeric_limits<double>::quiet_NaN();
    double best_slot_y = std::numeric_limits<double>::quiet_NaN();
    double best_slot_sx = std::numeric_limits<double>::quiet_NaN();
    double best_slot_sy = std::numeric_limits<double>::quiet_NaN();
    Eigen::Index n_gate = 0;

    for (std::size_t i = 0; i < n_keep; ++i) {
        const auto &cand = candidates[i];
        double x_arcsec_raw = pix_to_arcsec * (static_cast<double>(cand.col) - col0);
        double y_arcsec_raw = pix_to_arcsec * (static_cast<double>(cand.row) - row0);
        double center_x = std::numeric_limits<double>::quiet_NaN();
        double center_y = std::numeric_limits<double>::quiet_NaN();
        double x_arcsec = std::numeric_limits<double>::quiet_NaN();
        double y_arcsec = std::numeric_limits<double>::quiet_NaN();
        if (!observed_to_prior_frame(array, x_arcsec_raw, y_arcsec_raw, derot_elev_rad,
                                     x_arcsec, y_arcsec, &center_x, &center_y, true)) {
            continue;
        }

        double min_d2 = std::numeric_limits<double>::infinity();
        int min_slot = -1;
        double slot_x = std::numeric_limits<double>::quiet_NaN();
        double slot_y = std::numeric_limits<double>::quiet_NaN();
        double slot_sx = std::numeric_limits<double>::quiet_NaN();
        double slot_sy = std::numeric_limits<double>::quiet_NaN();
        if (!match_prior_slot(array, nw, x_arcsec, y_arcsec, min_d2, min_slot,
                              &slot_x, &slot_y, &slot_sx, &slot_sy)) {
            continue;
        }
        if (prior_max_d2 > 0.0 && min_d2 > prior_max_d2) {
            continue;
        }
        n_gate++;

        const double score = cand.snr - prior_score_lambda * min_d2;
        if (!found || score > best_score || (score == best_score && cand.snr > best_snr)) {
            found = true;
            best_score = score;
            best_snr = cand.snr;
            best_d2 = min_d2;
            best_row = cand.row;
            best_col = cand.col;
            best_slot = min_slot;
            best_x_raw = x_arcsec_raw;
            best_y_raw = y_arcsec_raw;
            best_x_prior = x_arcsec;
            best_y_prior = y_arcsec;
            best_slot_x = slot_x;
            best_slot_y = slot_y;
            best_slot_sx = slot_sx;
            best_slot_sy = slot_sy;
            if (std::isfinite(center_x) && std::isfinite(center_y)) {
                set_prior_diag(prior_center_x_col, center_x);
                set_prior_diag(prior_center_y_col, center_y);
            }
        }
    }

    set_prior_diag(prior_n_candidates_gate_col, static_cast<double>(n_gate));

    if (!found) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_gate_rejected);
        return false;
    }

    init_row = static_cast<double>(best_row);
    init_col = static_cast<double>(best_col);
    set_prior_diag(prior_used_col, 1.0);
    set_prior_diag(prior_no_candidate_reason_col, prior_reason_none);
    set_prior_diag(prior_slot_index_col, static_cast<double>(best_slot));
    set_prior_diag(prior_match_d2_col, best_d2);
    set_prior_diag(prior_match_score_col, best_score);
    set_prior_diag(prior_candidate_snr_col, best_snr);
    set_prior_diag(prior_candidate_x_raw_col, best_x_raw);
    set_prior_diag(prior_candidate_y_raw_col, best_y_raw);
    set_prior_diag(prior_candidate_x_prior_col, best_x_prior);
    set_prior_diag(prior_candidate_y_prior_col, best_y_prior);
    set_prior_diag(prior_slot_x_col, best_slot_x);
    set_prior_diag(prior_slot_y_col, best_slot_y);
    set_prior_diag(prior_slot_sx_col, best_slot_sx);
    set_prior_diag(prior_slot_sy_col, best_slot_sy);
    logger->debug(
        "beammap priors init map={} det={} array={} nw={} row={} col={} snr={} d2={} slot={} lambda={} max_d2={}",
        map_index, map_index, array, nw, init_row, init_col, best_snr, best_d2,
        best_slot, prior_score_lambda, prior_max_d2);
    return true;
}

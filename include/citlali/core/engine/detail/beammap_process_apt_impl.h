#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>

bool Beammap::gather_beammap_reference_candidates(
    const std::vector<Eigen::Index> &ref_nws,
    Eigen::VectorXd &x_t,
    Eigen::VectorXd &y_t,
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> &det_indices) {
    constexpr Eigen::Index min_reference_candidates = 25;
    auto nw_in_set = [](Eigen::Index nw,
                        const std::vector<Eigen::Index> &set) {
        return std::find(set.begin(), set.end(), nw) != set.end();
    };

    Eigen::Index n_match = 0;
    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        if (calib.apt["flag"](i) == 0) {
            auto nw = static_cast<Eigen::Index>(calib.apt["nw"](i));
            const double x = calib.apt["x_t"](i);
            const double y = calib.apt["y_t"](i);
            if (nw_in_set(nw, ref_nws) && std::isfinite(x) &&
                std::isfinite(y)) {
                n_match++;
            }
        }
    }
    if (n_match < min_reference_candidates) {
        return false;
    }

    x_t.resize(n_match);
    y_t.resize(n_match);
    det_indices.resize(n_match);
    Eigen::Index k = 0;
    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        if (calib.apt["flag"](i) == 0) {
            auto nw = static_cast<Eigen::Index>(calib.apt["nw"](i));
            const double x = calib.apt["x_t"](i);
            const double y = calib.apt["y_t"](i);
            if (nw_in_set(nw, ref_nws) && std::isfinite(x) &&
                std::isfinite(y)) {
                x_t(k) = x;
                y_t(k) = y;
                det_indices(k) = i;
                k++;
            }
        }
    }
    return true;
}

bool Beammap::gather_all_unflagged_beammap_reference_candidates(
    Eigen::VectorXd &x_t,
    Eigen::VectorXd &y_t,
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> &det_indices) {
    Eigen::Index n_unflagged = 0;
    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        if (calib.apt["flag"](i) == 0 &&
            std::isfinite(calib.apt["x_t"](i)) &&
            std::isfinite(calib.apt["y_t"](i))) {
            n_unflagged++;
        }
    }
    if (n_unflagged <= 0) {
        return false;
    }

    x_t.resize(n_unflagged);
    y_t.resize(n_unflagged);
    det_indices.resize(n_unflagged);
    Eigen::Index k = 0;
    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        const double x = calib.apt["x_t"](i);
        const double y = calib.apt["y_t"](i);
        if (calib.apt["flag"](i) == 0 && std::isfinite(x) &&
            std::isfinite(y)) {
            x_t(k) = x;
            y_t(k) = y;
            det_indices(k) = i;
            k++;
        }
    }
    return true;
}

void Beammap::resolve_automatic_beammap_reference_detector(
    double &ref_det_x_t, double &ref_det_y_t) {
    using IndexVector = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    logger->info("finding a reference detector");

    Eigen::VectorXd x_t, y_t, dist;
    IndexVector det_indices;
    double med_x_t = 0.0;
    double med_y_t = 0.0;

    const std::vector<Eigen::Index> primary_nws = {3};
    const std::vector<Eigen::Index> fallback_nws = {2, 3, 4};

    bool have_ref = false;
    if (gather_beammap_reference_candidates(
            primary_nws, x_t, y_t, det_indices)) {
        logger->info("using median of nw=3 for reference");
        have_ref = true;
    }
    else if (gather_beammap_reference_candidates(
                 fallback_nws, x_t, y_t, det_indices)) {
        logger->info("using median of nw=2,3,4 for reference");
        have_ref = true;
    }

    if (!have_ref) {
        logger->warn("no robust reference from nw=3 or nw=2,3,4; using all unflagged detectors");
        have_ref = gather_all_unflagged_beammap_reference_candidates(
            x_t, y_t, det_indices);
    }

    if (!have_ref) {
        logger->warn("all detectors are flagged; disabling reference subtraction");
        return;
    }

    logger->info("beammap reference candidate count: {}", x_t.size());
    med_x_t = tula::alg::median(x_t);
    med_y_t = tula::alg::median(y_t);

    if (!std::isfinite(med_x_t) || !std::isfinite(med_y_t)) {
        logger->warn("beammap reference median is non-finite ({},{}); disabling reference subtraction",
                     med_x_t, med_y_t);
        beammap_reference_det_found = -99;
        return;
    }

    dist = (x_t.array() - med_x_t).square().matrix() +
           (y_t.array() - med_y_t).square().matrix();
    Eigen::Index nearest_candidate = -1;
    dist.minCoeff(&nearest_candidate);
    if (nearest_candidate >= 0 && nearest_candidate < det_indices.size()) {
        beammap_reference_det_found = det_indices(nearest_candidate);

        // set reference x_t and y_t to the median location
        ref_det_x_t = med_x_t;
        ref_det_y_t = med_y_t;
        return;
    }

    logger->warn("beammap reference nearest candidate index {} is invalid; disabling reference subtraction",
                 nearest_candidate);
    beammap_reference_det_found = -99;
}

void Beammap::select_beammap_reference_detector(
    double &ref_det_x_t, double &ref_det_y_t) {
    const auto &reference_config =
        citlali::pipeline::beammap_config(*this).reference;
    const auto configured_reference_det =
        static_cast<Eigen::Index>(reference_config.reference_detector);

    // initial reference det
    beammap_reference_det_found = -99;

    // if particular reference detector is requested
    if (reference_config.subtract_reference_detector) {
        if (configured_reference_det >= 0 &&
            configured_reference_det < calib.n_dets) {
            beammap_reference_det_found = configured_reference_det;
            // set reference x_t and y_t
            ref_det_x_t = calib.apt["x_t"](beammap_reference_det_found);
            ref_det_y_t = calib.apt["y_t"](beammap_reference_det_found);
        }
        // else use detector closest to the median of selected networks
        else {
            if (configured_reference_det >= 0) {
                logger->warn("configured beammap_reference_det={} is out of range [0, {}); using automatic reference selection",
                             configured_reference_det, calib.n_dets);
            }
            resolve_automatic_beammap_reference_detector(
                ref_det_x_t, ref_det_y_t);
        }
        if (beammap_reference_det_found >= 0 &&
            beammap_reference_det_found < calib.n_dets) {
            double ref_det_actual_x_t =
                calib.apt["x_t"](beammap_reference_det_found);
            double ref_det_actual_y_t =
                calib.apt["y_t"](beammap_reference_det_found);
            logger->info("using reference median ({:.3f},{:.3f}) arcsec; nearest detector {} at ({:.3f},{:.3f}) arcsec",
                         ref_det_x_t, ref_det_y_t,
                         beammap_reference_det_found,
                         ref_det_actual_x_t, ref_det_actual_y_t);
            // record resolved reference detector for metadata; keep config value unchanged
            calib.apt_meta["reference_det"] = beammap_reference_det_found;
        }
        else {
            logger->warn("reference detector is invalid; leaving reference offsets at ({:.3f},{:.3f}) arcsec",
                         ref_det_x_t, ref_det_y_t);
        }
    }
    else {
        logger->info("no reference detector selected");
    }
}

void Beammap::record_beammap_reference_metadata(
    double ref_det_x_t, double ref_det_y_t) {
    // add reference detector to APT meta data
    calib.apt_meta["reference_x_t"] = ref_det_x_t;
    calib.apt_meta["reference_y_t"] = ref_det_y_t;
}

void Beammap::preserve_beammap_raw_detector_offsets() {
    // raw (not derotated or reference detector subtracted) detector x and y values
    calib.apt["x_t_raw"] = calib.apt["x_t"];
    calib.apt["y_t_raw"] = calib.apt["y_t"];
}

void Beammap::populate_beammap_derotation_elevation() {
    // per-detector derotation elevation for altaz beammaps
    calib.apt["derot_elev"].setConstant(telescope.tel_data["TelElAct"].mean());
    if (citlali::config::is_altaz_map_pixel_axes(telescope.pixel_axes) &&
        citlali::config::is_detector_map_grouping(
            citlali::pipeline::mapmaking_config(*this).grouping) &&
        !ptcs.empty()) {
        Eigen::MatrixXd elev_best(omb.n_rows, omb.n_cols);
        Eigen::MatrixXd dist2_best(omb.n_rows, omb.n_cols);
        elev_best.setConstant(std::numeric_limits<double>::quiet_NaN());
        dist2_best.setConstant(std::numeric_limits<double>::infinity());

        for (const auto &ptc : ptcs) {
            const auto &alt = ptc.tel_data.data.at("alt_phys");
            const auto &az = ptc.tel_data.data.at("az_phys");
            const auto &el = ptc.tel_data.data.at("TelElAct");
            for (Eigen::Index k = 0; k < alt.size(); ++k) {
                double row =
                    alt(k) / omb.pixel_size_rad + (omb.n_rows - 1) / 2.0;
                double col =
                    az(k) / omb.pixel_size_rad + (omb.n_cols - 1) / 2.0;
                Eigen::Index ir = static_cast<Eigen::Index>(std::llround(row));
                Eigen::Index ic = static_cast<Eigen::Index>(std::llround(col));
                if ((ir >= 0) && (ir < omb.n_rows) && (ic >= 0) &&
                    (ic < omb.n_cols)) {
                    double lat_center =
                        (static_cast<double>(ir) - (omb.n_rows - 1) / 2.0) *
                        omb.pixel_size_rad;
                    double lon_center =
                        (static_cast<double>(ic) - (omb.n_cols - 1) / 2.0) *
                        omb.pixel_size_rad;
                    double dlat = alt(k) - lat_center;
                    double dlon = az(k) - lon_center;
                    double dist2 = dlat * dlat + dlon * dlon;
                    if (dist2 < dist2_best(ir, ic)) {
                        dist2_best(ir, ic) = dist2;
                        elev_best(ir, ic) = el(k);
                    }
                }
            }
        }

        for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
            double row =
                (calib.apt["y_t_raw"](i) * ASEC_TO_RAD) /
                    omb.pixel_size_rad +
                (omb.n_rows - 1) / 2.0;
            double col =
                (calib.apt["x_t_raw"](i) * ASEC_TO_RAD) /
                    omb.pixel_size_rad +
                (omb.n_cols - 1) / 2.0;
            Eigen::Index ir = static_cast<Eigen::Index>(std::llround(row));
            Eigen::Index ic = static_cast<Eigen::Index>(std::llround(col));
            if ((ir >= 0) && (ir < omb.n_rows) && (ic >= 0) &&
                (ic < omb.n_cols)) {
                double elev = elev_best(ir, ic);
                if (std::isfinite(elev)) {
                    calib.apt["derot_elev"](i) = elev;
                }
            }
        }
    }
}

void Beammap::apply_beammap_reference_offsets(
    double ref_det_x_t, double ref_det_y_t) {
    // align to reference detector if specified and subtract its position from x and y
    calib.apt["x_t"] = calib.apt["x_t"].array() - ref_det_x_t;
    calib.apt["y_t"] = calib.apt["y_t"].array() - ref_det_y_t;
}

void Beammap::apply_beammap_derotation(bool derotate) {
    // derotated detector x and y values
    calib.apt["x_t_derot"] = calib.apt["x_t"];
    calib.apt["y_t_derot"] = calib.apt["y_t"];

    // tolerate telescope streams that provide elevation in degrees.
    Eigen::VectorXd derot_elev_rad = calib.apt["derot_elev"];
    const double max_abs_elev = derot_elev_rad.array().abs().maxCoeff();
    if (std::isfinite(max_abs_elev) && max_abs_elev > 2.0 * pi + 0.1) {
        logger->warn("derot_elev appears to be in degrees (max |elev|={:.4g}); converting to radians",
                     max_abs_elev);
        derot_elev_rad *= DEG_TO_RAD;
    }

    // calculate derotated positions
    Eigen::VectorXd rot_az_off =
        cos(-derot_elev_rad.array()) * calib.apt["x_t_derot"].array() -
        sin(-derot_elev_rad.array()) * calib.apt["y_t_derot"].array();
    Eigen::VectorXd rot_alt_off =
        sin(-derot_elev_rad.array()) * calib.apt["x_t_derot"].array() +
        cos(-derot_elev_rad.array()) * calib.apt["y_t_derot"].array();

    // overwrite x_t and y_t
    calib.apt["x_t_derot"] = -rot_az_off;
    calib.apt["y_t_derot"] = -rot_alt_off;

    if (derotate) {
        logger->info("derotating apt");
        // if derotation requested set default positions to derotated positions
        calib.apt["x_t"] = calib.apt["x_t_derot"];
        calib.apt["y_t"] = calib.apt["y_t_derot"];
    }
}

void Beammap::process_apt() {
    const auto &reference_config =
        citlali::pipeline::beammap_config(*this).reference;

    // reference detector x and y
    double ref_det_x_t = 0;
    double ref_det_y_t = 0;

    select_beammap_reference_detector(ref_det_x_t, ref_det_y_t);
    record_beammap_reference_metadata(ref_det_x_t, ref_det_y_t);
    preserve_beammap_raw_detector_offsets();
    populate_beammap_derotation_elevation();
    apply_beammap_reference_offsets(ref_det_x_t, ref_det_y_t);
    apply_beammap_derotation(reference_config.derotate);
}

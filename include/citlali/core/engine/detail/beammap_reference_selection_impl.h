#pragma once

// Beammap APT reference selection implementation detail.
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

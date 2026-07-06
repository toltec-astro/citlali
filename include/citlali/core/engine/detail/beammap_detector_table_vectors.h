#pragma once

// Beammap detector table vector helpers.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace beammap_detector_table_vectors {

inline Eigen::VectorXd double_or_nan(const Eigen::VectorXd &values,
                                     Eigen::Index n_dets,
                                     double scale = 1.0) {
    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_dets, std::numeric_limits<double>::quiet_NaN());
    if (values.size() == n_dets) {
        out = (scale * values.array()).matrix();
    }
    return out;
}

inline Eigen::VectorXd int_or_nan(const Eigen::VectorXi &values,
                                  Eigen::Index n_dets) {
    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_dets, std::numeric_limits<double>::quiet_NaN());
    if (values.size() == n_dets) {
        out = values.cast<double>();
    }
    return out;
}

inline Eigen::VectorXd positive_scaled_threshold(const Eigen::VectorXd &values,
                                                 Eigen::Index n_dets,
                                                 double scale) {
    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_dets, std::numeric_limits<double>::quiet_NaN());
    if (scale <= 0.0 || values.size() != n_dets) {
        return out;
    }
    for (Eigen::Index i = 0; i < n_dets; ++i) {
        if (std::isfinite(values(i)) && values(i) > 0.0) {
            out(i) = scale * values(i);
        }
    }
    return out;
}

struct FruitLoopsSupportVectors {
    Eigen::VectorXd npix;
    Eigen::VectorXd signal_sum;
    Eigen::VectorXd x_span_arcsec;
    Eigen::VectorXd y_span_arcsec;
};

template <class PtcProc, class MapBuffer>
FruitLoopsSupportVectors fruitloops_support_vectors(
    const PtcProc &ptcproc,
    const MapBuffer &omb,
    Eigen::Index n_dets,
    const Eigen::VectorXd &adaptive_threshold,
    double pix_to_arcsec,
    double fill_value) {
    FruitLoopsSupportVectors out{
        Eigen::VectorXd::Constant(n_dets, fill_value),
        Eigen::VectorXd::Constant(n_dets, fill_value),
        Eigen::VectorXd::Constant(n_dets, fill_value),
        Eigen::VectorXd::Constant(n_dets, fill_value)};

    for (Eigen::Index i = 0; i < n_dets; ++i) {
        if (i >= static_cast<Eigen::Index>(omb.signal.size()) ||
            i >= static_cast<Eigen::Index>(omb.weight.size()) ||
            omb.signal[i].rows() != omb.n_rows ||
            omb.signal[i].cols() != omb.n_cols ||
            omb.weight[i].rows() != omb.n_rows ||
            omb.weight[i].cols() != omb.n_cols) {
            continue;
        }
        const double threshold = adaptive_threshold(i);
        if (!std::isfinite(threshold) || threshold <= 0.0) {
            continue;
        }
        if (ptcproc.fruit_loops_source_valid.size() != n_dets ||
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
            (ptcproc.fruit_loops_adaptive_support_radius_rad.size() == n_dets)
                ? ptcproc.fruit_loops_adaptive_support_radius_rad(i)
                : fill_value;
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
        out.npix(i) = static_cast<double>(npix);
        out.signal_sum(i) = signal_sum;
        if (npix > 0) {
            out.x_span_arcsec(i) = max_x - min_x;
            out.y_span_arcsec(i) = max_y - min_y;
        }
    }

    return out;
}

template <class AptTable, class HeaderMap, class PriorDiagnostics>
struct DetectorTableAccessors {
    const AptTable &apt;
    const HeaderMap &units;
    const HeaderMap &descriptions;
    const PriorDiagnostics &prior_diag_values;
    Eigen::Index n_dets;
    Eigen::Index n_prior_diag_cols;

    Eigen::VectorXd apt_or_zero(const std::string &key) const {
        auto it = apt.find(key);
        if (it != apt.end() && it->second.size() == n_dets) {
            return it->second;
        }
        return Eigen::VectorXd::Zero(n_dets);
    }

    std::string unit(const std::string &key,
                     const std::string &fallback) const {
        auto it = units.find(key);
        if (it != units.end()) {
            return it->second;
        }
        return fallback;
    }

    std::string description(const std::string &key,
                            const std::string &fallback) const {
        auto it = descriptions.find(key);
        if (it != descriptions.end()) {
            return it->second;
        }
        return fallback;
    }

    template <class DiagColumn>
    Eigen::VectorXd prior_diag_or(DiagColumn diag_col,
                                  double fallback_value) const {
        Eigen::VectorXd out(n_dets);
        if (prior_diag_values.rows() == n_dets &&
            prior_diag_values.cols() == n_prior_diag_cols) {
            out = prior_diag_values.col(static_cast<Eigen::Index>(diag_col));
        }
        else {
            out.setConstant(fallback_value);
        }
        return out;
    }
};

template <class AptTable, class HeaderMap, class PriorDiagnostics>
DetectorTableAccessors<AptTable, HeaderMap, PriorDiagnostics> make_accessors(
    const AptTable &apt, const HeaderMap &units,
    const HeaderMap &descriptions, const PriorDiagnostics &prior_diag_values,
    Eigen::Index n_dets, Eigen::Index n_prior_diag_cols) {
    return {apt, units, descriptions, prior_diag_values, n_dets,
            n_prior_diag_cols};
}

inline Eigen::VectorXd bound_state(const Eigen::MatrixXi &hit_upper,
                                   const Eigen::MatrixXi &hit_lower,
                                   Eigen::Index param_index) {
    return hit_upper.col(param_index).cast<double>() -
           hit_lower.col(param_index).cast<double>();
}

struct FitBoundVectors {
    Eigen::VectorXd amp;
    Eigen::VectorXd x;
    Eigen::VectorXd y;
    Eigen::VectorXd a;
    Eigen::VectorXd b;
    Eigen::VectorXd angle;
};

inline FitBoundVectors fit_bound_vectors(const Eigen::MatrixXi &hit_upper,
                                         const Eigen::MatrixXi &hit_lower) {
    return {
        bound_state(hit_upper, hit_lower, 0),
        bound_state(hit_upper, hit_lower, 1),
        bound_state(hit_upper, hit_lower, 2),
        bound_state(hit_upper, hit_lower, 3),
        bound_state(hit_upper, hit_lower, 4),
        bound_state(hit_upper, hit_lower, 5)};
}

struct FitInitLimitVectors {
    Eigen::VectorXd amp;
    Eigen::VectorXd x_t;
    Eigen::VectorXd y_t;
    Eigen::VectorXd a_fwhm;
    Eigen::VectorXd b_fwhm;
    Eigen::VectorXd low_a_fwhm;
    Eigen::VectorXd high_a_fwhm;
    Eigen::VectorXd low_b_fwhm;
    Eigen::VectorXd high_b_fwhm;
};

inline FitInitLimitVectors fit_init_limit_vectors(
    const Eigen::MatrixXd &init_params,
    const Eigen::MatrixXd &lower_limits,
    const Eigen::MatrixXd &upper_limits,
    double pix_to_arcsec,
    double sigma_to_fwhm_arcsec,
    Eigen::Index n_cols,
    Eigen::Index n_rows) {
    return {
        init_params.col(0),
        (pix_to_arcsec *
         (init_params.col(1).array() - (n_cols - 1) / 2.0)).matrix(),
        (pix_to_arcsec *
         (init_params.col(2).array() - (n_rows - 1) / 2.0)).matrix(),
        (sigma_to_fwhm_arcsec * init_params.col(3).array()).matrix(),
        (sigma_to_fwhm_arcsec * init_params.col(4).array()).matrix(),
        (sigma_to_fwhm_arcsec * lower_limits.col(3).array()).matrix(),
        (sigma_to_fwhm_arcsec * upper_limits.col(3).array()).matrix(),
        (sigma_to_fwhm_arcsec * lower_limits.col(4).array()).matrix(),
        (sigma_to_fwhm_arcsec * upper_limits.col(4).array()).matrix()};
}

} // namespace beammap_detector_table_vectors

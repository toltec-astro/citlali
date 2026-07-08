#pragma once

// Beammap detector table accessor and fit-bound helpers.

#include <Eigen/Core>

#include <string>

#include <citlali/core/engine/detail/beammap_detector_table_common_vectors.h>

namespace beammap_detector_table_vectors {

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

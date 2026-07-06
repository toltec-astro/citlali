#pragma once

// Beammap detector table vector helpers.

#include <Eigen/Core>

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

} // namespace beammap_detector_table_vectors

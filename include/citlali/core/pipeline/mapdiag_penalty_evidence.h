#pragma once

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline constexpr const char *mapdiag_complete_map_evidence_view =
    "complete_map";
inline constexpr const char *mapdiag_feedback_excluded_evidence_view =
    "feedback_excluded_map";
inline constexpr const char *mapdiag_no_feedback_evidence_view =
    "complete_map_no_feedback";

template <class CompleteSignal, class FeedbackSignal>
std::string mapdiag_feedback_excluded_signal_error(
    const CompleteSignal &complete_signal,
    const FeedbackSignal &feedback_signal) {
    if (complete_signal.rows() != feedback_signal.rows() ||
        complete_signal.cols() != feedback_signal.cols()) {
        return "complete and accepted-feedback map dimensions differ";
    }
    for (Eigen::Index row = 0; row < complete_signal.rows(); ++row) {
        for (Eigen::Index col = 0; col < complete_signal.cols(); ++col) {
            const bool complete_finite =
                std::isfinite(static_cast<double>(complete_signal(row, col)));
            const bool feedback_finite =
                std::isfinite(static_cast<double>(feedback_signal(row, col)));
            if (complete_finite != feedback_finite) {
                return "complete and accepted-feedback map finite support differs";
            }
        }
    }
    return {};
}

template <class CompleteSignal, class FeedbackSignal>
Eigen::MatrixXd make_mapdiag_feedback_excluded_signal(
    const CompleteSignal &complete_signal,
    const FeedbackSignal &feedback_signal) {
    if (const auto error = mapdiag_feedback_excluded_signal_error(
            complete_signal, feedback_signal);
        !error.empty()) {
        throw std::runtime_error(
            "EL-F4 feedback-excluded mapdiag evidence is incompatible: " +
            error);
    }
    Eigen::MatrixXd evidence = complete_signal;
    for (Eigen::Index row = 0; row < evidence.rows(); ++row) {
        for (Eigen::Index col = 0; col < evidence.cols(); ++col) {
            if (std::isfinite(static_cast<double>(evidence(row, col)))) {
                evidence(row, col) -= feedback_signal(row, col);
            }
        }
    }
    return evidence;
}

}  // namespace citlali::pipeline

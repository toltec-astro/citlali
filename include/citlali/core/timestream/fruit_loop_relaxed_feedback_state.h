#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace citlali::fruit {

inline constexpr const char *fruit_loop_compact_relaxation_method_id =
    "SCI-FRUIT-EL-F1-COMPACT-RELAXATION-R0.1";

// This is the complete causal feedback state F_k for the bounded EL-F1
// experiment.  The measured complete map Q~_k remains in the ordinary map
// products.  Matrices are flattened in map, row, column order so the same
// object can be checkpointed without depending on a particular MapBuffer.
struct FruitLoopRelaxedFeedbackState {
    std::string method_id{fruit_loop_compact_relaxation_method_id};
    bool method_active = false;
    double alpha = 1.0;
    bool stored = false;
    std::string observation_id;
    int completed_iteration = -1;
    std::string map_grouping;
    Eigen::Index map_count = 0;
    Eigen::Index n_rows = 0;
    Eigen::Index n_cols = 0;
    double pixel_size_rad = 0.0;
    std::vector<float> wcs_cdelt;
    std::vector<int> wcs_naxis;
    std::vector<float> wcs_crpix;
    std::vector<float> wcs_crval;
    std::vector<std::string> wcs_cunit;
    std::vector<double> signal;
    std::vector<double> kernel;
    std::vector<double> weight;
    std::vector<double> median_rms;
};

inline bool fruit_loop_relaxation_alpha_is_approved(double alpha) {
    return alpha == 1.0 || alpha == 1.25 || alpha == 1.50;
}

inline bool fruit_loop_relaxed_feedback_enabled(double alpha) {
    return alpha != 1.0;
}

inline std::size_t fruit_loop_relaxed_feedback_plane_size(
    const FruitLoopRelaxedFeedbackState &state) {
    if (state.map_count <= 0 || state.n_rows <= 0 || state.n_cols <= 0) {
        return 0;
    }
    const auto maps = static_cast<std::size_t>(state.map_count);
    const auto rows = static_cast<std::size_t>(state.n_rows);
    const auto cols = static_cast<std::size_t>(state.n_cols);
    if (rows > std::numeric_limits<std::size_t>::max() / cols ||
        maps > std::numeric_limits<std::size_t>::max() / (rows * cols)) {
        throw std::overflow_error("EL-F1 feedback-state shape overflows size_t");
    }
    return maps * rows * cols;
}

inline std::string fruit_loop_relaxed_feedback_state_error(
    const FruitLoopRelaxedFeedbackState &state) {
    if (!state.method_active) {
        if (state.alpha != 1.0 || state.stored) {
            return "inactive EL-F1 method has non-default causal state";
        }
        return {};
    }
    if (state.method_id != fruit_loop_compact_relaxation_method_id) {
        return "method identity is not SCI-FRUIT-EL-F1-COMPACT-RELAXATION-R0.1";
    }
    if (!fruit_loop_relaxation_alpha_is_approved(state.alpha)) {
        return "alpha is outside the approved set {1.00, 1.25, 1.50}";
    }
    if (!state.stored) {
        if (fruit_loop_relaxed_feedback_enabled(state.alpha)) {
            return "relaxed alpha requires a stored feedback state";
        }
        return {};
    }
    if (!fruit_loop_relaxed_feedback_enabled(state.alpha)) {
        return "alpha=1 compatibility state must use the ordinary complete map product";
    }
    if (state.observation_id.empty() || state.completed_iteration < 0 ||
        state.map_grouping.empty() || state.map_count <= 0 ||
        state.n_rows <= 0 || state.n_cols <= 0 ||
        !std::isfinite(state.pixel_size_rad) || state.pixel_size_rad <= 0.0) {
        return "feedback identity or geometry is incomplete";
    }
    const auto plane_size = fruit_loop_relaxed_feedback_plane_size(state);
    if (state.signal.size() != plane_size ||
        state.kernel.size() != plane_size ||
        state.weight.size() != plane_size) {
        return "signal, kernel, or weight plane cardinality is inconsistent";
    }
    if (!state.median_rms.empty() &&
        state.median_rms.size() != static_cast<std::size_t>(state.map_count)) {
        return "median-RMS cardinality is inconsistent";
    }
    if (state.wcs_naxis.size() != 2 || state.wcs_cdelt.size() != 2 ||
        state.wcs_crpix.size() != 2 || state.wcs_crval.size() != 2 ||
        state.wcs_cunit.size() != 2) {
        return "spatial WCS identity must contain exactly two axes";
    }
    return {};
}

template <class Value>
bool fruit_loop_bitwise_equal(Value left, Value right) {
    static_assert(std::is_trivially_copyable_v<Value>);
    return std::memcmp(&left, &right, sizeof(Value)) == 0;
}

template <class Value>
bool fruit_loop_spatial_wcs_equal(const std::vector<Value> &left,
                                  const std::vector<Value> &right) {
    return left.size() == 2 && right.size() >= 2 && left[0] == right[0] &&
           left[1] == right[1];
}

template <class Value>
std::vector<Value> fruit_loop_spatial_wcs(const std::vector<Value> &values) {
    if (values.size() < 2) {
        throw std::invalid_argument(
            "EL-F1 feedback state requires two spatial WCS axes");
    }
    return {values[0], values[1]};
}

inline std::size_t fruit_loop_relaxed_feedback_offset(
    Eigen::Index map_index, Eigen::Index row, Eigen::Index col,
    Eigen::Index n_rows, Eigen::Index n_cols) {
    return (static_cast<std::size_t>(map_index) *
                static_cast<std::size_t>(n_rows) +
            static_cast<std::size_t>(row)) *
               static_cast<std::size_t>(n_cols) +
           static_cast<std::size_t>(col);
}

template <class MapBuffer>
std::string complete_map_for_relaxed_feedback_error(const MapBuffer &map) {
    const auto map_count = static_cast<Eigen::Index>(map.signal.size());
    if (map_count <= 0 || map.n_rows <= 0 || map.n_cols <= 0 ||
        static_cast<Eigen::Index>(map.kernel.size()) != map_count ||
        static_cast<Eigen::Index>(map.weight.size()) != map_count) {
        return "complete map lacks common signal, kernel, and weight planes";
    }
    if (map.map_grouping.empty() || !std::isfinite(map.pixel_size_rad) ||
        map.pixel_size_rad <= 0.0) {
        return "complete map grouping or pixel geometry is invalid";
    }
    for (Eigen::Index i = 0; i < map_count; ++i) {
        for (const auto *plane : {&map.signal[i], &map.kernel[i],
                                  &map.weight[i]}) {
            if (plane->rows() != map.n_rows || plane->cols() != map.n_cols) {
                return "complete map planes do not share one grid";
            }
        }
    }
    return {};
}

template <class MapBuffer>
void update_fruit_loop_relaxed_feedback_state(
    FruitLoopRelaxedFeedbackState &state, const MapBuffer &complete_map,
    const std::string &observation_id, int completed_iteration,
    double alpha) {
    if (!fruit_loop_relaxation_alpha_is_approved(alpha)) {
        throw std::invalid_argument(
            "EL-F1 alpha must be exactly 1.00, 1.25, or 1.50");
    }
    state.method_id = fruit_loop_compact_relaxation_method_id;
    state.method_active = true;
    state.alpha = alpha;
    if (!fruit_loop_relaxed_feedback_enabled(alpha)) {
        // The alpha=1 control deliberately retains the unmodified code path.
        // Its F_k is the ordinary complete Q~_k product named by the restart
        // checkpoint, so no duplicate matrix state is introduced.
        state.stored = false;
        state.observation_id.clear();
        state.completed_iteration = -1;
        state.signal.clear();
        state.kernel.clear();
        state.weight.clear();
        state.median_rms.clear();
        return;
    }
    if (observation_id.empty() || completed_iteration < 0) {
        throw std::invalid_argument(
            "EL-F1 feedback update requires observation and iteration identity");
    }
    if (const auto error =
            complete_map_for_relaxed_feedback_error(complete_map);
        !error.empty()) {
        throw std::invalid_argument("EL-F1 feedback update: " + error);
    }

    const auto map_count =
        static_cast<Eigen::Index>(complete_map.signal.size());
    const auto plane_size = static_cast<std::size_t>(map_count) *
                            static_cast<std::size_t>(complete_map.n_rows) *
                            static_cast<std::size_t>(complete_map.n_cols);
    const bool have_previous = state.stored;
    if (have_previous) {
        if (state.observation_id != observation_id ||
            state.completed_iteration + 1 != completed_iteration ||
            state.map_grouping != complete_map.map_grouping ||
            state.map_count != map_count ||
            state.n_rows != complete_map.n_rows ||
            state.n_cols != complete_map.n_cols ||
            !fruit_loop_bitwise_equal(state.pixel_size_rad,
                                      complete_map.pixel_size_rad) ||
            !fruit_loop_spatial_wcs_equal(state.wcs_cdelt,
                                          complete_map.wcs.cdelt) ||
            !fruit_loop_spatial_wcs_equal(state.wcs_naxis,
                                          complete_map.wcs.naxis) ||
            !fruit_loop_spatial_wcs_equal(state.wcs_crpix,
                                          complete_map.wcs.crpix) ||
            !fruit_loop_spatial_wcs_equal(state.wcs_crval,
                                          complete_map.wcs.crval) ||
            !fruit_loop_spatial_wcs_equal(state.wcs_cunit,
                                          complete_map.wcs.cunit)) {
            throw std::invalid_argument(
                "EL-F1 feedback update requires an unchanged observation route, grouping, WCS, and grid");
        }
        if (state.signal.size() != plane_size ||
            state.kernel.size() != plane_size) {
            throw std::invalid_argument(
                "EL-F1 prior feedback state has inconsistent plane cardinality");
        }
    }

    std::vector<double> next_signal(plane_size);
    std::vector<double> next_kernel(plane_size);
    std::vector<double> next_weight(plane_size);
    for (Eigen::Index i = 0; i < map_count; ++i) {
        for (Eigen::Index row = 0; row < complete_map.n_rows; ++row) {
            for (Eigen::Index col = 0; col < complete_map.n_cols; ++col) {
                const auto offset = fruit_loop_relaxed_feedback_offset(
                    i, row, col, complete_map.n_rows, complete_map.n_cols);
                const double q_signal = complete_map.signal[i](row, col);
                const double q_kernel = complete_map.kernel[i](row, col);
                if (have_previous) {
                    const bool signal_support = std::isfinite(q_signal);
                    const bool previous_signal_support =
                        std::isfinite(state.signal[offset]);
                    const bool kernel_support = std::isfinite(q_kernel);
                    const bool previous_kernel_support =
                        std::isfinite(state.kernel[offset]);
                    if (signal_support != previous_signal_support ||
                        kernel_support != previous_kernel_support ||
                        signal_support != kernel_support) {
                        throw std::invalid_argument(
                            "EL-F1 feedback update encountered a finite-support mismatch");
                    }
                    next_signal[offset] = signal_support
                        ? state.signal[offset] +
                              alpha * (q_signal - state.signal[offset])
                        : q_signal;
                    next_kernel[offset] = kernel_support
                        ? state.kernel[offset] +
                              alpha * (q_kernel - state.kernel[offset])
                        : q_kernel;
                }
                else {
                    if (std::isfinite(q_signal) != std::isfinite(q_kernel)) {
                        throw std::invalid_argument(
                            "EL-F1 initial signal and kernel finite support differ");
                    }
                    next_signal[offset] = q_signal;
                    next_kernel[offset] = q_kernel;
                }
                next_weight[offset] = complete_map.weight[i](row, col);
            }
        }
    }

    state.stored = true;
    state.observation_id = observation_id;
    state.completed_iteration = completed_iteration;
    state.map_grouping = complete_map.map_grouping;
    state.map_count = map_count;
    state.n_rows = complete_map.n_rows;
    state.n_cols = complete_map.n_cols;
    state.pixel_size_rad = complete_map.pixel_size_rad;
    // Feedback map sampling is two-dimensional.  The ordinary loader
    // reconstructs only the spatial WCS and deliberately omits/zeros the
    // spectral and Stokes entries, whose identity is represented by the map
    // grouping and ordered map planes instead.
    state.wcs_cdelt = fruit_loop_spatial_wcs(complete_map.wcs.cdelt);
    state.wcs_naxis = fruit_loop_spatial_wcs(complete_map.wcs.naxis);
    state.wcs_crpix = fruit_loop_spatial_wcs(complete_map.wcs.crpix);
    state.wcs_crval = fruit_loop_spatial_wcs(complete_map.wcs.crval);
    state.wcs_cunit = fruit_loop_spatial_wcs(complete_map.wcs.cunit);
    state.signal = std::move(next_signal);
    state.kernel = std::move(next_kernel);
    state.weight = std::move(next_weight);
    state.median_rms.clear();
    state.median_rms.reserve(
        static_cast<std::size_t>(complete_map.median_rms.size()));
    for (Eigen::Index i = 0; i < complete_map.median_rms.size(); ++i) {
        state.median_rms.push_back(complete_map.median_rms(i));
    }
}

template <class MapBuffer>
void apply_fruit_loop_relaxed_feedback_state(
    const FruitLoopRelaxedFeedbackState &state, MapBuffer &loaded_map,
    const std::string &expected_observation_id,
    int expected_completed_iteration) {
    if (!state.stored) {
        return;
    }
    if (const auto error = fruit_loop_relaxed_feedback_state_error(state);
        !error.empty()) {
        throw std::invalid_argument("EL-F1 feedback load: " + error);
    }
    if (state.observation_id != expected_observation_id ||
        state.completed_iteration != expected_completed_iteration) {
        throw std::invalid_argument(
            "EL-F1 feedback load observation or iteration identity mismatch");
    }
    if (const auto error = complete_map_for_relaxed_feedback_error(loaded_map);
        !error.empty()) {
        throw std::invalid_argument("EL-F1 feedback load: " + error);
    }
    std::string identity_mismatches;
    const auto record_mismatch = [&identity_mismatches](const char *name) {
        if (!identity_mismatches.empty()) {
            identity_mismatches += ",";
        }
        identity_mismatches += name;
    };
    if (state.map_grouping != loaded_map.map_grouping) {
        record_mismatch("map_grouping");
    }
    if (state.map_count !=
        static_cast<Eigen::Index>(loaded_map.signal.size())) {
        record_mismatch("map_count");
    }
    if (state.n_rows != loaded_map.n_rows) {
        record_mismatch("n_rows");
    }
    if (state.n_cols != loaded_map.n_cols) {
        record_mismatch("n_cols");
    }
    if (!fruit_loop_bitwise_equal(state.pixel_size_rad,
                                  loaded_map.pixel_size_rad)) {
        record_mismatch("pixel_size_rad");
    }
    if (!fruit_loop_spatial_wcs_equal(state.wcs_cdelt,
                                      loaded_map.wcs.cdelt)) {
        record_mismatch("wcs_cdelt");
    }
    if (!fruit_loop_spatial_wcs_equal(state.wcs_naxis,
                                      loaded_map.wcs.naxis)) {
        record_mismatch("wcs_naxis");
    }
    if (!fruit_loop_spatial_wcs_equal(state.wcs_crpix,
                                      loaded_map.wcs.crpix)) {
        record_mismatch("wcs_crpix");
    }
    if (!fruit_loop_spatial_wcs_equal(state.wcs_crval,
                                      loaded_map.wcs.crval)) {
        record_mismatch("wcs_crval");
    }
    if (!fruit_loop_spatial_wcs_equal(state.wcs_cunit,
                                      loaded_map.wcs.cunit)) {
        record_mismatch("wcs_cunit");
    }
    if (!identity_mismatches.empty()) {
        throw std::invalid_argument(
            "EL-F1 feedback load requires exact route, grouping, WCS, and "
            "grid identity; mismatches=" +
            identity_mismatches);
    }

    for (Eigen::Index i = 0; i < state.map_count; ++i) {
        for (Eigen::Index row = 0; row < state.n_rows; ++row) {
            for (Eigen::Index col = 0; col < state.n_cols; ++col) {
                const auto offset = fruit_loop_relaxed_feedback_offset(
                    i, row, col, state.n_rows, state.n_cols);
                if (!fruit_loop_bitwise_equal(
                        state.weight[offset], loaded_map.weight[i](row, col))) {
                    throw std::invalid_argument(
                        "EL-F1 feedback load newest-product weight identity mismatch");
                }
                loaded_map.signal[i](row, col) = state.signal[offset];
                loaded_map.kernel[i](row, col) = state.kernel[offset];
                loaded_map.weight[i](row, col) = state.weight[offset];
            }
        }
    }
    if (!state.median_rms.empty()) {
        if (loaded_map.median_rms.size() != state.map_count) {
            throw std::invalid_argument(
                "EL-F1 feedback load newest-product RMS cardinality mismatch");
        }
        for (Eigen::Index i = 0; i < state.map_count; ++i) {
            if (!fruit_loop_bitwise_equal(
                    state.median_rms[static_cast<std::size_t>(i)],
                    loaded_map.median_rms(i))) {
                throw std::invalid_argument(
                    "EL-F1 feedback load newest-product RMS identity mismatch");
            }
        }
    }
}

}  // namespace citlali::fruit

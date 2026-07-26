#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <mutex>
#include <vector>

namespace citlali::pipeline {

struct WeightValidationRestartState {
    int accumulated_iterations = 0;
    bool finalized = false;
    std::vector<double> ratio_penalty_sum;
    std::vector<double> ratio_value_sum;
    std::vector<int> ratio_value_count;
    std::vector<int> ratio_count;
    std::vector<double> atmospheric_penalty_sum;
    std::vector<double> atmospheric_correlation_sum;
    std::vector<int> atmospheric_count;
    std::vector<double> detector_penalty;
    std::vector<int> detector_validated;
};

template <class Vector>
auto weight_validation_std_vector(const Vector &values) {
    using Value = typename Vector::Scalar;
    if (values.size() == 0) {
        return std::vector<Value>{};
    }
    return std::vector<Value>(values.data(), values.data() + values.size());
}

template <class Processor>
WeightValidationRestartState snapshot_weight_validation_restart_state(
    const Processor &processor) {
    std::lock_guard<std::mutex> lock(*processor.weight_validation_mutex);
    return WeightValidationRestartState{
        processor.weight_validation_accumulated_iters,
        processor.weight_validation_finalized,
        weight_validation_std_vector(
            processor.weight_validation_ratio_penalty_sum),
        weight_validation_std_vector(
            processor.weight_validation_ratio_value_sum),
        weight_validation_std_vector(
            processor.weight_validation_ratio_value_count),
        weight_validation_std_vector(
            processor.weight_validation_ratio_count),
        weight_validation_std_vector(
            processor.weight_validation_atm_penalty_sum),
        weight_validation_std_vector(
            processor.weight_validation_atm_corr_sum),
        weight_validation_std_vector(
            processor.weight_validation_atm_count),
        weight_validation_std_vector(
            processor.weight_validation_detector_penalty),
        weight_validation_std_vector(
            processor.weight_validation_detector_validated),
    };
}

template <class EigenVector, class Values>
void restore_weight_validation_vector(EigenVector &target,
                                      const Values &values) {
    target.resize(static_cast<Eigen::Index>(values.size()));
    std::copy(values.begin(), values.end(), target.data());
}

template <class Processor>
void restore_weight_validation_restart_state(
    Processor &processor, const WeightValidationRestartState &state) {
    std::lock_guard<std::mutex> lock(*processor.weight_validation_mutex);
    processor.weight_validation_current_iter = 0;
    processor.weight_validation_current_iter_contribution_count = 0;
    processor.weight_validation_accumulated_iters =
        state.accumulated_iterations;
    processor.weight_validation_finalized = state.finalized;
    restore_weight_validation_vector(
        processor.weight_validation_ratio_penalty_sum,
        state.ratio_penalty_sum);
    restore_weight_validation_vector(
        processor.weight_validation_ratio_value_sum,
        state.ratio_value_sum);
    restore_weight_validation_vector(
        processor.weight_validation_ratio_value_count,
        state.ratio_value_count);
    restore_weight_validation_vector(
        processor.weight_validation_ratio_count, state.ratio_count);
    restore_weight_validation_vector(
        processor.weight_validation_atm_penalty_sum,
        state.atmospheric_penalty_sum);
    restore_weight_validation_vector(
        processor.weight_validation_atm_corr_sum,
        state.atmospheric_correlation_sum);
    restore_weight_validation_vector(
        processor.weight_validation_atm_count,
        state.atmospheric_count);
    restore_weight_validation_vector(
        processor.weight_validation_detector_penalty,
        state.detector_penalty);
    restore_weight_validation_vector(
        processor.weight_validation_detector_validated,
        state.detector_validated);
}

}  // namespace citlali::pipeline

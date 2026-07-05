#pragma once

// Included by mapdiag_observation_weight.h inside namespace citlali::pipeline.

inline double mapdiag_fraction_or_fill(double value, double total,
                                       double fill_value) {
    return (total > 0.0) ? value / total : fill_value;
}

template <class SourceValues, class DestValues>
void assign_mapdiag_obs_fraction_entry(
    std::size_t flat, const SourceValues &source_values, double total,
    double fill_value, DestValues &fraction_values) {
    fraction_values[flat] =
        mapdiag_fraction_or_fill(source_values[flat], total, fill_value);
}

template <class SourceValues, class DestValues>
void assign_mapdiag_obs_fraction_series(
    const SourceValues &source_values, double total, double fill_value,
    std::size_t n_obsnums, std::size_t map_index,
    DestValues &fraction_values) {
    for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
        const std::size_t flat = map_index * n_obsnums + obs_idx;
        assign_mapdiag_obs_fraction_entry(
            flat, source_values, total, fill_value, fraction_values);
    }
}

template <class SourceValues, class DestValues>
void assign_mapdiag_obs_fraction_pair(
    const SourceValues &weight_sum, double total_weight,
    const SourceValues &core_weight_sum, double total_core_weight,
    double fill_value, std::size_t n_obsnums, std::size_t map_index,
    DestValues &weight_frac, DestValues &core_weight_frac) {
    assign_mapdiag_obs_fraction_series(
        weight_sum, total_weight, fill_value, n_obsnums, map_index,
        weight_frac);
    assign_mapdiag_obs_fraction_series(
        core_weight_sum, total_core_weight, fill_value, n_obsnums,
        map_index, core_weight_frac);
}

template <class SourceValues, class DestValues, class Context>
void assign_mapdiag_obs_fraction_pair(
    const SourceValues &weight_sum, double total_weight,
    const SourceValues &core_weight_sum, double total_core_weight,
    double fill_value, const Context &context, std::size_t map_index,
    DestValues &weight_frac, DestValues &core_weight_frac) {
    assign_mapdiag_obs_fraction_pair(
        weight_sum, total_weight, core_weight_sum, total_core_weight,
        fill_value, context.n_obsnums, map_index, weight_frac,
        core_weight_frac);
}

template <class Values>
double sum_mapdiag_obs_values(const Values &values, std::size_t n_obsnums,
                              std::size_t map_index) {
    double total = 0.0;
    for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
        const std::size_t flat = map_index * n_obsnums + obs_idx;
        total += values[flat];
    }
    return total;
}

template <class Values>
MapdiagObsWeightTotals sum_mapdiag_obs_weight_totals(
    const Values &weight_sum, const Values &core_weight_sum,
    std::size_t n_obsnums, std::size_t map_index) {
    return {sum_mapdiag_obs_values(weight_sum, n_obsnums, map_index),
            sum_mapdiag_obs_values(core_weight_sum, n_obsnums, map_index)};
}

template <class Values, class Context>
MapdiagObsWeightTotals sum_mapdiag_obs_weight_totals(
    const Values &weight_sum, const Values &core_weight_sum,
    const Context &context, std::size_t map_index) {
    return sum_mapdiag_obs_weight_totals(
        weight_sum, core_weight_sum, context.n_obsnums, map_index);
}

template <class SourceValues, class DestValues, class Context>
void assign_mapdiag_obs_fractions_for_map(
    const SourceValues &weight_sum, const SourceValues &core_weight_sum,
    double fill_value, const Context &context, std::size_t map_index,
    DestValues &weight_frac, DestValues &core_weight_frac) {
    const auto obs_totals = sum_mapdiag_obs_weight_totals(
        weight_sum, core_weight_sum, context, map_index);
    assign_mapdiag_obs_fraction_pair(
        weight_sum, obs_totals.weight, core_weight_sum,
        obs_totals.core_weight, fill_value, context, map_index,
        weight_frac, core_weight_frac);
}


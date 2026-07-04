#pragma once

#include <cstddef>
#include <vector>

namespace citlali::pipeline {

struct MapdiagEdgeGuardIntRefs {
    std::vector<int> &applied;
    std::vector<int> &support_radius_pix;
    std::vector<int> &science_npix;
    std::vector<int> &support_npix;
    std::vector<int> &guardband_npix;
};

struct MapdiagEdgeGuardDoubleRefs {
    std::vector<double> &weight_thresholds;
    std::vector<double> &hits_thresholds;
    std::vector<double> &background_levels;
    std::vector<double> &science_frac;
    std::vector<double> &support_frac;
    std::vector<double> &guardband_rms_pre;
    std::vector<double> &guardband_rms_post;
    std::vector<double> &exterior_rms_pre;
    std::vector<double> &exterior_rms_post;
    std::vector<double> &exterior_max_abs_pre;
    std::vector<double> &exterior_max_abs_post;
};

template <class EdgeGuardState>
bool mapdiag_has_edge_guard_entry(std::size_t idx,
                                  const EdgeGuardState &state) {
    return idx < state.edge_guard_applied.size();
}

template <class EdgeGuardState>
void assign_mapdiag_edge_guard_int_entry(
    std::size_t idx, const EdgeGuardState &state,
    MapdiagEdgeGuardIntRefs refs) {
    refs.applied[idx] = state.edge_guard_applied[idx];
    refs.support_radius_pix[idx] = state.edge_guard_support_radius_pix[idx];
    refs.science_npix[idx] = state.edge_guard_science_npix[idx];
    refs.support_npix[idx] = state.edge_guard_support_npix[idx];
    refs.guardband_npix[idx] = state.edge_guard_guardband_npix[idx];
}

template <class EdgeGuardState>
void assign_mapdiag_edge_guard_double_entry(
    std::size_t idx, const EdgeGuardState &state,
    MapdiagEdgeGuardDoubleRefs refs) {
    refs.weight_thresholds[idx] = state.edge_guard_weight_threshold[idx];
    refs.hits_thresholds[idx] = state.edge_guard_hits_threshold[idx];
    refs.background_levels[idx] = state.edge_guard_background_level[idx];
    refs.science_frac[idx] = state.edge_guard_science_frac[idx];
    refs.support_frac[idx] = state.edge_guard_support_frac[idx];
    refs.guardband_rms_pre[idx] = state.edge_guard_guardband_rms_pre[idx];
    refs.guardband_rms_post[idx] = state.edge_guard_guardband_rms_post[idx];
    refs.exterior_rms_pre[idx] = state.edge_guard_exterior_rms_pre[idx];
    refs.exterior_rms_post[idx] = state.edge_guard_exterior_rms_post[idx];
    refs.exterior_max_abs_pre[idx] =
        state.edge_guard_exterior_max_abs_pre[idx];
    refs.exterior_max_abs_post[idx] =
        state.edge_guard_exterior_max_abs_post[idx];
}

}  // namespace citlali::pipeline

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace citlali::pipeline {

inline const char *beammap_normalize_support_cause_name(int cause) {
    switch (cause) {
    case 1:
        return "no_accum_weight";
    case 2:
        return "bad_grid_weight";
    case 3:
        return "support_threshold";
    default:
        return "unknown";
    }
}

template <class MapBuffer, class Calib, class Logger>
void log_beammap_normalize_support_summary(const MapBuffer &omb,
                                           Calib &calib,
                                           Eigen::Index current_iter,
                                           const Logger &logger) {
    if (omb.normalize_support_diag.empty()) {
        return;
    }

    Eigen::Index n_diag_maps = 0;
    Eigen::Index total_masked = 0;
    Eigen::Index total_no_accum_weight = 0;
    Eigen::Index total_bad_grid_weight = 0;
    Eigen::Index total_support_threshold = 0;
    Eigen::Index total_raw_signal_nonzero = 0;
    Eigen::Index total_adjacent_support = 0;
    std::vector<Eigen::Index> suspicious_maps;

    for (Eigen::Index map_index = 0;
         map_index < static_cast<Eigen::Index>(
                         omb.normalize_support_diag.size());
         ++map_index) {
        const auto &diag = omb.normalize_support_diag[map_index];
        if (diag.map_index < 0) {
            continue;
        }
        n_diag_maps++;
        total_masked += diag.n_masked;
        total_no_accum_weight += diag.n_masked_no_accum_weight;
        total_bad_grid_weight +=
            diag.n_masked_bad_grid_weight_with_accum_weight;
        total_support_threshold += diag.n_masked_by_support_threshold;
        total_raw_signal_nonzero += diag.n_masked_raw_signal_nonzero;
        total_adjacent_support += diag.n_masked_adjacent_support;
        if (diag.n_masked_bad_grid_weight_with_accum_weight > 0 ||
            diag.n_masked_by_support_threshold > 0 ||
            diag.n_masked_adjacent_support > 0 ||
            diag.n_masked_raw_signal_nonzero > 0) {
            suspicious_maps.push_back(map_index);
        }
    }

    logger->info(
        "beammap normalize support summary iter={} maps={} masked={} no_accum_weight={} bad_grid_weight_with_accum_weight={} support_threshold={} raw_signal_nonzero={} adjacent_support_holes={}",
        current_iter, n_diag_maps, total_masked, total_no_accum_weight,
        total_bad_grid_weight, total_support_threshold,
        total_raw_signal_nonzero, total_adjacent_support);

    auto support_diag_score = [&](Eigen::Index map_index) {
        const auto &diag = omb.normalize_support_diag[map_index];
        return diag.n_masked_adjacent_support +
               diag.n_masked_bad_grid_weight_with_accum_weight +
               diag.n_masked_by_support_threshold +
               diag.n_masked_raw_signal_nonzero;
    };
    std::sort(suspicious_maps.begin(), suspicious_maps.end(),
              [&](Eigen::Index lhs, Eigen::Index rhs) {
                  const auto lhs_score = support_diag_score(lhs);
                  const auto rhs_score = support_diag_score(rhs);
                  if (lhs_score != rhs_score) {
                      return lhs_score > rhs_score;
                  }
                  const double lhs_neighbor =
                      omb.normalize_support_diag[lhs]
                          .max_masked_neighbor_weight;
                  const double rhs_neighbor =
                      omb.normalize_support_diag[rhs]
                          .max_masked_neighbor_weight;
                  return std::isfinite(lhs_neighbor) &&
                                 std::isfinite(rhs_neighbor)
                             ? lhs_neighbor > rhs_neighbor
                             : std::isfinite(lhs_neighbor);
              });

    const Eigen::Index n_log = std::min<Eigen::Index>(
        10, static_cast<Eigen::Index>(suspicious_maps.size()));
    for (Eigen::Index rank = 0; rank < n_log; ++rank) {
        const Eigen::Index map_index = suspicious_maps[rank];
        const auto &diag = omb.normalize_support_diag[map_index];
        const int uid = (map_index < calib.apt["uid"].size())
                            ? static_cast<int>(
                                  std::lround(calib.apt["uid"](map_index)))
                            : -1;
        const int array =
            (map_index < calib.apt["array"].size())
                ? static_cast<int>(std::lround(calib.apt["array"](map_index)))
                : -1;
        const int nw = (map_index < calib.apt["nw"].size())
                           ? static_cast<int>(
                                 std::lround(calib.apt["nw"](map_index)))
                           : -1;
        const double x_t =
            (map_index < calib.apt["x_t"].size())
                ? calib.apt["x_t"](map_index)
                : std::numeric_limits<double>::quiet_NaN();
        const double y_t =
            (map_index < calib.apt["y_t"].size())
                ? calib.apt["y_t"](map_index)
                : std::numeric_limits<double>::quiet_NaN();
        logger->info(
            "beammap normalize support detail iter={} rank={} map={} uid={} array={} nw={} x_t={:.3f} y_t={:.3f} masked={} no_accum={} bad_grid_with_accum={} threshold={} raw_signal_nonzero={} adjacent_holes={} support_threshold={:.4g} max_raw_signal={:.4g} max_neighbor_weight={:.4g} max_neighbor_rc=({}, {}) max_neighbor_cause={}",
            current_iter, rank + 1, map_index, uid, array, nw, x_t, y_t,
            diag.n_masked, diag.n_masked_no_accum_weight,
            diag.n_masked_bad_grid_weight_with_accum_weight,
            diag.n_masked_by_support_threshold,
            diag.n_masked_raw_signal_nonzero, diag.n_masked_adjacent_support,
            diag.support_weight_threshold, diag.max_masked_abs_raw_signal,
            diag.max_masked_neighbor_weight, diag.max_neighbor_row,
            diag.max_neighbor_col,
            beammap_normalize_support_cause_name(diag.max_neighbor_cause));
    }
}

}  // namespace citlali::pipeline

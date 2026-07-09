#pragma once

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/config/mapmaking_config.h>

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <sstream>
#include <string>

namespace citlali::pipeline {

struct BeammapActiveMapSelection {
    Eigen::Matrix<bool, Eigen::Dynamic, 1> active_maps;
    Eigen::Index n_active_maps = 0;
    bool enabled = false;

    const Eigen::Matrix<bool, Eigen::Dynamic, 1> *ptr() const {
        return enabled ? &active_maps : nullptr;
    }
};

template <class Converged, class Logger>
BeammapActiveMapSelection select_unconverged_beammap_maps(
    citlali::config::MapGrouping grouping, const Converged &converged,
    Eigen::Index n_maps, const Logger &logger) {
    BeammapActiveMapSelection selection;
    selection.n_active_maps = n_maps;
    if (!citlali::config::is_detector_map_grouping(grouping) ||
        converged.size() != n_maps) {
        return selection;
    }

    const Eigen::Index n_converged =
        (converged.array() == true).count();
    if (n_converged <= 0 || n_converged >= n_maps) {
        return selection;
    }

    selection.active_maps.resize(n_maps);
    selection.n_active_maps = 0;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        selection.active_maps(i) = !converged(i);
        if (selection.active_maps(i)) {
            ++selection.n_active_maps;
        }
    }
    selection.enabled = true;
    logger->info(
        "beammap detector mapmaking: remaking {}/{} unconverged maps",
        selection.n_active_maps, n_maps);
    return selection;
}

template <class MapBuffer, class Logger>
void ensure_jinc_grid_weight_maps(citlali::config::MapMethod method,
                                  MapBuffer &omb, Eigen::Index n_maps,
                                  const Logger &logger) {
    if (!citlali::config::is_jinc_map_method(method) ||
        static_cast<Eigen::Index>(omb.grid_weight.size()) == n_maps) {
        return;
    }
    logger->info("allocating jinc grid_weight maps: current={} expected={}",
                 omb.grid_weight.size(), n_maps);
    omb.grid_weight.assign(static_cast<std::size_t>(n_maps),
                           Eigen::MatrixXd::Zero(omb.n_rows, omb.n_cols));
}

template <class MapBuffer, class PtcChunks, class RandomBits, class Generator>
void reset_beammap_mapmaking_buffers(
    MapBuffer &omb, PtcChunks &ptcs, Eigen::Index n_maps, bool run_kernel,
    bool run_noise, bool randomize_dets, Eigen::Index n_dets,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps,
    RandomBits &rands, Generator &eng) {
    omb.clear_contribution_diag();
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        if (active_maps != nullptr && !(*active_maps)(i)) {
            continue;
        }
        omb.signal[i].setZero();
        omb.weight[i].setZero();
        if (!omb.grid_weight.empty()) {
            omb.grid_weight[i].setZero();
        }

        if (!omb.coverage.empty()) {
            omb.coverage[i].setZero();
        }
        if (run_kernel) {
            omb.kernel[i].setZero();
        }
        if (!omb.noise.empty()) {
            omb.noise[i].setZero();
        }

        if (run_noise) {
            for (auto &ptcdata : ptcs) {
                if (randomize_dets) {
                    ptcdata.noise.data =
                        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::
                            Zero(omb.n_noise, n_dets)
                                .unaryExpr([&](int dummy) {
                                    return 2 * rands(eng) - 1;
                                });
                }
                else {
                    ptcdata.noise.data =
                        Eigen::Matrix<int, Eigen::Dynamic, 1>::
                            Zero(omb.n_noise)
                                .unaryExpr([&](int dummy) {
                                    return 2 * rands(eng) - 1;
                                });
                }
            }
        }
    }
}

template <class Ptc, class MapBuffer, class JincMapMaker, class Logger>
void log_beammap_jinc_preflight(
    const Ptc &ptc, const Eigen::VectorXd &detector_arrays,
    const MapBuffer &omb, const JincMapMaker &jinc_mm,
    const Logger &logger) {
    std::array<Eigen::Index, 3> array_counts = {0, 0, 0};
    for (Eigen::Index det = 0; det < ptc.scans.data.cols(); ++det) {
        auto array_index = static_cast<int>(detector_arrays(det));
        if (array_index >= 0 &&
            array_index < static_cast<int>(array_counts.size())) {
            array_counts[static_cast<std::size_t>(array_index)]++;
        }
    }

    Eigen::Index map_min = -1;
    Eigen::Index map_max = -1;
    if (ptc.map_indices.data.size() > 0) {
        map_min = ptc.map_indices.data.minCoeff();
        map_max = ptc.map_indices.data.maxCoeff();
    }

    std::ostringstream kernel_dims;
    for (int array_index = 0; array_index < 3; ++array_index) {
        auto it = jinc_mm.jinc_weights_mat.find(array_index);
        if (it == jinc_mm.jinc_weights_mat.end()) {
            continue;
        }
        if (kernel_dims.tellp() > 0) {
            kernel_dims << ", ";
        }
        kernel_dims << "a" << array_index << "="
                    << it->second.rows() << "x" << it->second.cols();
    }

    logger->info(
        "beammap jinc preflight: n_dets={} n_pts={} n_maps={} map_index_range=[{}, {}] "
        "subpixel_n={} kernel_dims=[{}] array_counts=[{},{},{}]",
        ptc.scans.data.cols(),
        ptc.scans.data.rows(),
        omb.signal.size(),
        map_min,
        map_max,
        jinc_mm.subpixel_n,
        kernel_dims.str(),
        array_counts[0],
        array_counts[1],
        array_counts[2]);
}

inline bool use_beammap_detector_ptc_weights(
    citlali::config::BeammapDetectorWeightingMode mode,
    bool measurement_iter) {
    using Mode = citlali::config::BeammapDetectorWeightingMode;
    return mode == Mode::ptc ||
           (mode == Mode::ptc_after_iter0 && measurement_iter);
}

}  // namespace citlali::pipeline

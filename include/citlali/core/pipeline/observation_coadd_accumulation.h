#pragma once

#include <citlali/core/pipeline/stage_profile.h>

#include <Eigen/Core>

#include <tuple>

namespace citlali::pipeline {

inline std::tuple<int, int> centered_coadd_offsets(Eigen::Index coadd_rows,
                                                   Eigen::Index coadd_cols,
                                                   Eigen::Index obs_rows,
                                                   Eigen::Index obs_cols) {
    return {static_cast<int>(0.5 * (coadd_rows - obs_rows)),
            static_cast<int>(0.5 * (coadd_cols - obs_cols))};
}

template <class CoaddMapBuffer, class ObservationMapBuffer>
void accumulate_weighted_map_into_coadd(CoaddMapBuffer &cmb,
                                        ObservationMapBuffer &omb,
                                        Eigen::Index map_index,
                                        int delta_row, int delta_col,
                                        bool run_kernel) {
    auto cmb_weight_block =
        cmb.weight.at(map_index).block(delta_row, delta_col, omb.n_rows,
                                       omb.n_cols);
    auto cmb_signal_block =
        cmb.signal.at(map_index).block(delta_row, delta_col, omb.n_rows,
                                       omb.n_cols);

    cmb_weight_block += omb.weight.at(map_index);
    cmb_signal_block += (omb.signal.at(map_index).array() *
                         omb.weight.at(map_index).array()).matrix();

    if (run_kernel) {
        auto cmb_kernel_block =
            cmb.kernel.at(map_index).block(delta_row, delta_col, omb.n_rows,
                                           omb.n_cols);
        cmb_kernel_block += (omb.kernel.at(map_index).array() *
                             omb.weight.at(map_index).array()).matrix();
    }

    if (!cmb.coverage.empty()) {
        auto cmb_coverage_block =
            cmb.coverage.at(map_index).block(delta_row, delta_col,
                                             omb.n_rows, omb.n_cols);
        cmb_coverage_block += omb.coverage.at(map_index);
    }
}

template <class CoaddMapBuffer, class ObservationMapBuffer>
void accumulate_noise_maps_into_coadd(CoaddMapBuffer &cmb,
                                      ObservationMapBuffer &omb,
                                      Eigen::Index map_index, int delta_row,
                                      int delta_col) {
    if (cmb.noise.empty() || omb.noise.empty()) {
        return;
    }

    for (Eigen::Index n = 0; n < cmb.n_noise; ++n) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
            cmb_noise_matrix(cmb.noise.at(map_index).data() +
                                 n * cmb.n_rows * cmb.n_cols,
                             cmb.n_rows, cmb.n_cols);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
            omb_noise_matrix(omb.noise.at(map_index).data() +
                                 n * omb.n_rows * omb.n_cols,
                             omb.n_rows, omb.n_cols);
        auto cmb_noise_block =
            cmb_noise_matrix.block(delta_row, delta_col, omb.n_rows,
                                   omb.n_cols);
        cmb_noise_block +=
            (omb_noise_matrix.array() * omb.weight.at(map_index).array())
                .matrix();
    }
}

template <class CoaddMapBuffer, class ObservationMapBuffer>
void accumulate_observation_into_coadd(CoaddMapBuffer &cmb,
                                       ObservationMapBuffer &omb,
                                       Eigen::Index n_maps,
                                       bool run_kernel) {
    const auto [delta_row, delta_col] =
        centered_coadd_offsets(cmb.n_rows, cmb.n_cols, omb.n_rows,
                               omb.n_cols);

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        accumulate_weighted_map_into_coadd(cmb, omb, i, delta_row, delta_col,
                                           run_kernel);
        accumulate_noise_maps_into_coadd(cmb, omb, i, delta_row, delta_col);
    }
}

template <class Engine>
bool should_run_observation_coadd(const Engine &engine) {
    return !engine.rtcproc.run_polarization;
}

template <class TodProc, class Logger>
void coadd_observation(TodProc &todproc,
                       StageProfileCollector &stage_profile,
                       const Logger &logger) {
    auto &engine = todproc.engine();
    (void)stage_profile;

    logger->info("coadding");
    const auto profile_scope = profile_stage(stage_profile, "observation.coadd", logger);
    if (should_run_observation_coadd(engine)) {
        todproc.coadd();
    }
}

}  // namespace citlali::pipeline

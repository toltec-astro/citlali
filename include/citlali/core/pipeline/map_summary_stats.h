#pragma once

#include <map>
#include <string>

#include <Eigen/Core>

namespace citlali::pipeline {

struct MapSummaryNonfiniteCounts {
    std::map<std::string, int> n_nans{
        {"signal", 0}, {"weight", 0}, {"kernel", 0},
        {"coverage", 0}, {"noise", 0}};
    std::map<std::string, int> n_infs{
        {"signal", 0}, {"weight", 0}, {"kernel", 0},
        {"coverage", 0}, {"noise", 0}};
};

template <class MapBuffer>
MapSummaryNonfiniteCounts count_map_summary_nonfinite(const MapBuffer &mb) {
    MapSummaryNonfiniteCounts counts;

    for (Eigen::Index i=0; i<mb.signal.size(); ++i) {
        counts.n_nans["signal"] += mb.signal[i].array().isNaN().count();
        counts.n_nans["weight"] += mb.weight[i].array().isNaN().count();

        if (!mb.kernel.empty()) {
            counts.n_nans["kernel"] += mb.kernel[i].array().isNaN().count();
        }
        if (!mb.coverage.empty()) {
            counts.n_nans["coverage"] += mb.coverage[i].array().isNaN().count();
        }

        counts.n_infs["signal"] += mb.signal[i].array().isInf().count();
        counts.n_infs["weight"] += mb.weight[i].array().isInf().count();

        if (!mb.kernel.empty()) {
            counts.n_infs["kernel"] += mb.kernel[i].array().isInf().count();
        }
        if (!mb.coverage.empty()) {
            counts.n_infs["coverage"] += mb.coverage[i].array().isInf().count();
        }

        if (!mb.noise.empty()) {
            const Eigen::Index n_noise_maps = mb.noise[i].dimension(2);
            for (Eigen::Index j=0; j<n_noise_maps; ++j) {
                Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
                    noise_matrix(mb.noise[i].data() + j * mb.n_rows * mb.n_cols,
                                 mb.n_rows, mb.n_cols);

                counts.n_nans["noise"] += noise_matrix.array().isNaN().count();
                counts.n_infs["noise"] += noise_matrix.array().isInf().count();
            }
        }
    }

    return counts;
}

}  // namespace citlali::pipeline

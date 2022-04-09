#pragma once

#include <string>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <citlali/core/utils/pointing.h>


namespace  mapmaking {

using map_dims_t = std::tuple<int, int, Eigen::VectorXd, Eigen::VectorXd>;
using map_extent_t = std::vector<double>;
using map_coord_t = std::vector<Eigen::VectorXd>;
using map_count_t = std::size_t;

class NoiseMapBuffer {
public:
    Eigen::Index nrows, ncols, nnoise;
    map_count_t map_count;
    double pixel_size;

    // Physical coordinates for rows and cols (radians)
    Eigen::VectorXd rcphys, ccphys;

    // tensor for noise map inclusion (nnoise,nobs,nmaps)
    Eigen::Tensor<double,3> noise_rand;

    // noise maps (nrows, ncols, nnoise) of length nmaps
    std::vector<Eigen::Tensor<double,3>> noise;

    // vector of psd classes
    std::vector<PSD> psd;

    // vector of histogram psds
    std::vector<Histogram> histogram;

    template <typename Derived>
    void normalize_maps(std::vector<Eigen::DenseBase<Derived>> &weight) {
        // normalize noise maps
        for (Eigen::Index k=0; k<nnoise; k++) {
            for (Eigen::Index mc=0; mc<map_count; mc++) {
                for (Eigen::Index i=0; i<nrows; i++) {
                    for (Eigen::Index j=0; j<ncols; j++) {
                        auto pixel_weight = weight.at(mc)(i,j);
                        if (pixel_weight != pixel_weight) {
                                SPDLOG_INFO("bad pixel weight {}", pixel_weight);
                        }
                        if (pixel_weight != 0. && pixel_weight == pixel_weight) {
                            noise.at(mc)(i,j,k) = (noise.at(mc)(i,j,k)) / pixel_weight;
                        }
                        else {
                            noise.at(mc)(i,j,k) = 0;
                        }
                    }
                }
            }
        }
    }
};

} // namespace mapmaking

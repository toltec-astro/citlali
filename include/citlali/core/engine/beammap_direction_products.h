#pragma once

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/engine/calib.h>
#include <citlali/core/engine/beammap_types.h>
#include <citlali/core/mapmaking/map.h>

#include <Eigen/Core>

#include <map>
#include <optional>
#include <string>

namespace citlali::engine_detail::beammap {

struct DirectionalProduct {
    citlali::config::BeammapDirectionMode mode =
        citlali::config::BeammapDirectionMode::standard;
    engine::Calib calib;
    Eigen::MatrixXd params;
    Eigen::MatrixXd perrors;
    Eigen::MatrixXd p0;
    Eigen::MatrixXd perror0;
    Eigen::Matrix<bool, Eigen::Dynamic, 1> converged;
    Eigen::Vector<int, Eigen::Dynamic> converge_iter;
    Eigen::Matrix<bool, Eigen::Dynamic, 1> good_fits;
    Eigen::Matrix<uint16_t, Eigen::Dynamic, 1> flag2;
    Eigen::MatrixXd fit_diag_init_params;
    Eigen::MatrixXd fit_diag_lower_limits;
    Eigen::MatrixXd fit_diag_upper_limits;
    Eigen::MatrixXi fit_diag_hit_lower;
    Eigen::MatrixXi fit_diag_hit_upper;
    Eigen::VectorXi fit_diag_bound_code;
    Eigen::VectorXi fit_diag_bound_nhit;
    Eigen::MatrixXd prior_diag_values;
    Eigen::VectorXd final_prior_d2_diag;
    Eigen::VectorXi final_prior_slot_index_diag;
    Eigen::Index reference_detector = -99;
    bool priors_centered = false;
    bool priors_derotated = false;
    std::map<int, double> prior_center_x_arcsec;
    std::map<int, double> prior_center_y_arcsec;
    std::map<int, PriorArrayAlignment> prior_alignment;
    std::map<std::string, double> source_flux_mjy_beam;
    std::map<std::string, double> source_flux_mjy_sr;
};

// Beammap-only owner for the two additional detector-map buffers requested by
// direction_mode=all. The ordinary Engine::omb remains the standard buffer.
// No RTC/PTC state is copied into this owner.
class DirectionProducts {
public:
    mapmaking::MapBuffer left{"omb_left"};
    mapmaking::MapBuffer right{"omb_right"};
    std::optional<DirectionalProduct> left_product;
    std::optional<DirectionalProduct> right_product;
    bool buffers_initialized = false;

    void reset() {
        left = mapmaking::MapBuffer{"omb_left"};
        right = mapmaking::MapBuffer{"omb_right"};
        left_product.reset();
        right_product.reset();
        buffers_initialized = false;
    }
};

}  // namespace citlali::engine_detail::beammap

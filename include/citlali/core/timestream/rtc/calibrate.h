#pragma once

#include <citlali/core/timestream/atmosphere_operator.h>
#include <citlali/core/timestream/timestream.h>
#include <citlali/core/utils/constants.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace timestream {

class Calibration {
public:
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    std::string extinction_model{"N/A"};
    std::string calibration_quality_regime{"not_applied"};
    std::string reduction_calibration_quality_regime{"not_applied"};
    std::string calibration_validity_reason{"extinction_not_applied"};
    bool calibration_valid = false;
    double realized_tau225 = std::numeric_limits<double>::quiet_NaN();
    double reduction_maximum_tau225 =
        std::numeric_limits<double>::quiet_NaN();

    void select_reference_spectral_index(
        std::optional<double> requested_alpha) {
        atmosphere_operator_.select_reference_spectral_index(requested_alpha);
    }

    void setup(double tau225) {
        calibration_quality_regime = std::string{
            FixedDjf25AtmosphereOperator::quality_regime(tau225)};
        if (!std::isfinite(reduction_maximum_tau225)
            || tau225 > reduction_maximum_tau225) {
            reduction_maximum_tau225 = tau225;
        }
        reduction_calibration_quality_regime = std::string{
            FixedDjf25AtmosphereOperator::quality_regime(
                reduction_maximum_tau225)};
        extinction_model = std::string{
            FixedDjf25AtmosphereOperator::operator_id()};
        calibration_valid = true;
        calibration_validity_reason = "valid";
        realized_tau225 = tau225;
    }

    void disable_extinction() {
        extinction_model = "N/A";
        calibration_quality_regime = "not_applied";
        calibration_valid = false;
        calibration_validity_reason = "extinction_not_applied";
        realized_tau225 = std::numeric_limits<double>::quiet_NaN();
    }

    std::optional<double> requested_reference_spectral_index_alpha() const {
        return atmosphere_operator_.requested_alpha();
    }

    double effective_reference_spectral_index_alpha() const {
        return atmosphere_operator_.effective_alpha();
    }

    bool reference_spectral_index_default_applied() const {
        return atmosphere_operator_.alpha_default_applied();
    }

    static constexpr std::string_view operator_id() {
        return FixedDjf25AtmosphereOperator::operator_id();
    }

    static constexpr std::string_view operator_contract_sha256() {
        return FixedDjf25AtmosphereOperator::contract_sha256();
    }

    static constexpr std::string_view operator_nodes_sha256() {
        return FixedDjf25AtmosphereOperator::nodes_sha256();
    }

    static constexpr std::string_view passband_set_id() {
        return FixedDjf25AtmosphereOperator::passband_set_id();
    }

    static constexpr std::string_view reference_profile_id() {
        return FixedDjf25AtmosphereOperator::reference_profile_id();
    }

    template <typename Derived>
    auto calc_tau(const Eigen::DenseBase<Derived> &elev, double tau225) const;

    template <TCDataKind tcdata_kind, class calib_t>
    void calibrate_tod(TCData<tcdata_kind, Eigen::MatrixXd> &, calib_t &);

    template <TCDataKind tcdata_kind, class calib_t, typename tau_t>
    void extinction_correction(
        TCData<tcdata_kind, Eigen::MatrixXd> &, calib_t &, const tau_t &);

private:
    FixedDjf25AtmosphereOperator atmosphere_operator_;
};

template <typename Derived>
auto Calibration::calc_tau(
    const Eigen::DenseBase<Derived> &elev, double tau225) const {
    std::map<int, Eigen::VectorXd> los_tau_by_array;
    for (int array_index = 0; array_index < 3; ++array_index) {
        auto &values = los_tau_by_array[array_index];
        values.resize(elev.size());
        for (Eigen::Index sample = 0; sample < elev.size(); ++sample) {
            const double elevation_rad = elev.derived()(sample);
            if (!std::isfinite(elevation_rad)) {
                throw std::domain_error("non-finite sample elevation");
            }
            const double elevation_deg = elevation_rad * 180.0 / pi;
            values(sample) = atmosphere_operator_.line_of_sight_optical_depth(
                array_index, tau225, elevation_deg);
        }
    }
    return los_tau_by_array;
}

template <TCDataKind tcdata_kind, class calib_t>
void Calibration::calibrate_tod(
    TCData<tcdata_kind, Eigen::MatrixXd> &in, calib_t &calib) {
    if (calib.flux_conversion_factor.size() < in.scans.data.cols()) {
        throw std::runtime_error(
            "calibrate_tod flux_conversion_factor is shorter than detector count");
    }
    if (calib.apt["flxscale"].size() < in.scans.data.cols()) {
        throw std::runtime_error(
            "calibrate_tod APT flxscale column is shorter than detector count");
    }
    for (Eigen::Index i = 0; i < in.scans.data.cols(); ++i) {
        in.fcf.data(i) = calib.flux_conversion_factor(i);
        in.scans.data.col(i) = in.scans.data.col(i).array()
            * in.fcf.data(i) * calib.apt["flxscale"](i);
    }
}

template <TCDataKind tcdata_kind, class calib_t, typename tau_t>
void Calibration::extinction_correction(
    TCData<tcdata_kind, Eigen::MatrixXd> &in, calib_t &calib,
    const tau_t &tau_freq) {
    const Eigen::Index detector_count = in.scans.data.cols();
    const Eigen::Index sample_count = in.scans.data.rows();
    if (calib.apt["array"].size() < detector_count) {
        throw std::runtime_error(
            "extinction correction APT array column is shorter than detector count");
    }
    if (in.fcf.data.size() < detector_count) {
        throw std::runtime_error(
            "extinction correction factor vector is shorter than detector count");
    }

    for (Eigen::Index detector = 0; detector < detector_count; ++detector) {
        const double raw_array_index = calib.apt["array"](detector);
        if (!std::isfinite(raw_array_index)
            || raw_array_index != std::floor(raw_array_index)
            || raw_array_index < 0.0 || raw_array_index > 2.0) {
            throw std::domain_error(
                "extinction correction has an unsupported detector array index");
        }
        const int array_index = static_cast<int>(raw_array_index);
        const auto found = tau_freq.find(array_index);
        if (found == tau_freq.end() || found->second.size() != sample_count) {
            throw std::domain_error(
                "extinction correction is missing sample-aligned array support");
        }
        for (Eigen::Index sample = 0; sample < sample_count; ++sample) {
            const double los_tau = found->second(sample);
            const double factor = std::exp(los_tau);
            if (!std::isfinite(los_tau) || los_tau < 0.0
                || !std::isfinite(factor) || factor < 1.0) {
                throw std::domain_error(
                    "extinction correction contains invalid LOS optical depth");
            }
        }
    }

    for (Eigen::Index detector = 0; detector < detector_count; ++detector) {
        const int array_index =
            static_cast<int>(calib.apt["array"](detector));
        const auto factor = tau_freq.at(array_index).array().exp();
        in.fcf.data(detector) =
            (in.fcf.data(detector) * factor).mean();
        in.scans.data.col(detector) =
            in.scans.data.col(detector).array() * factor;
    }
}

}  // namespace timestream

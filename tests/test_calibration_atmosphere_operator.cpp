#include <citlali/core/timestream/atmosphere_operator.h>
#include <citlali/core/timestream/rtc/calibrate.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace {

using Operator = timestream::FixedDjf25AtmosphereOperator;

const timestream::atmosphere_nodes::SeriesDescriptor &find_series(
    int array_index, int alpha, std::string_view anchor) {
    for (const auto &series : timestream::atmosphere_nodes::series) {
        if (series.array_index == array_index && series.alpha == alpha
            && series.anchor_id == anchor) {
            return series;
        }
    }
    throw std::logic_error("test series not found");
}

double node_value(
    int array_index, int alpha, std::string_view anchor,
    double elevation_deg) {
    const auto &series = find_series(array_index, alpha, anchor);
    for (std::size_t index = 0; index < series.count; ++index) {
        const auto offset = series.offset + index;
        if (timestream::atmosphere_nodes::elevation_deg[offset]
            == elevation_deg) {
            return timestream::atmosphere_nodes::los_optical_depth[offset];
        }
    }
    throw std::logic_error("test elevation node not found");
}

struct CalibrationFixture {
    Eigen::VectorXd flux_conversion_factor;
    std::map<std::string, Eigen::VectorXd> apt;
};

void admit_test_product(timestream::Calibration &calibration,
                        Eigen::Index detector_count,
                        bool extinction = true) {
    timestream::CalibrationProductAdmissionInputs inputs;
    inputs.target_unit = "mJy/beam";
    inputs.calibration_requested = true;
    inputs.extinction_requested = extinction;
    inputs.acquisition_identity_available = true;
    inputs.acquisition_identity_valid = true;
    inputs.apt_artifact_sha256 = "test-apt";
    inputs.acquisition_binding_sha256 = "test-binding-sha";
    inputs.raw_observation_identity = "test-raw-observation";
    inputs.acquisition_binding_mode = "test-binding";
    inputs.acquisition_key_schema = "test-key";
    inputs.response_identity = "test-response";
    inputs.target_unit_factor = Eigen::VectorXd::Ones(detector_count);
    inputs.detector_flxscale = Eigen::VectorXd::Ones(detector_count);
    inputs.detector_beam_major_fwhm_arcsec =
        Eigen::VectorXd::Ones(detector_count);
    inputs.detector_beam_minor_fwhm_arcsec =
        Eigen::VectorXd::Ones(detector_count);
    inputs.minimum_extinction_correction =
        Eigen::VectorXd::Ones(detector_count);
    inputs.maximum_extinction_correction =
        Eigen::VectorXd::Constant(detector_count, extinction ? 10.0 : 1.0);
    calibration.admit_product(inputs);
}

}  // namespace

TEST(calibration_atmosphere_operator, freezes_exact_artifact_identities) {
    EXPECT_EQ(
        Operator::contract_sha256(),
        "7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a");
    EXPECT_EQ(
        Operator::nodes_sha256(),
        "fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f");
    EXPECT_EQ(
        Operator::operator_id(),
        "am12_fixed_djf25_piecewise_linear_los_tau_v1");
}

TEST(calibration_atmosphere_operator, defaults_alpha_zero_and_rejects_others) {
    Operator model;
    EXPECT_FALSE(model.requested_alpha().has_value());
    EXPECT_DOUBLE_EQ(model.effective_alpha(), 0.0);
    EXPECT_TRUE(model.alpha_default_applied());

    for (const double alpha : {-1.0, 0.0, 2.0, 4.0}) {
        EXPECT_NO_THROW(model.select_reference_spectral_index(alpha));
        ASSERT_TRUE(model.requested_alpha().has_value());
        EXPECT_DOUBLE_EQ(*model.requested_alpha(), alpha);
        EXPECT_DOUBLE_EQ(model.effective_alpha(), alpha);
        EXPECT_FALSE(model.alpha_default_applied());
    }
    for (const double alpha : {
             -2.0, 1.0, 3.0,
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(
            model.select_reference_spectral_index(alpha),
            std::invalid_argument);
    }
}

TEST(calibration_atmosphere_operator, is_exact_at_all_supported_source_nodes) {
    Operator model;
    for (const int alpha : {-1, 0, 2, 4}) {
        model.select_reference_spectral_index(alpha);
        for (const auto &series : timestream::atmosphere_nodes::series) {
            if (series.alpha != alpha) {
                continue;
            }
            for (std::size_t index = 0; index < series.count; ++index) {
                const auto offset = series.offset + index;
                const double elevation =
                    timestream::atmosphere_nodes::elevation_deg[offset];
                if (elevation < Operator::minimum_elevation_deg) {
                    continue;
                }
                EXPECT_DOUBLE_EQ(
                    model.line_of_sight_optical_depth(
                        series.array_index, series.tau225, elevation),
                    timestream::atmosphere_nodes::los_optical_depth[offset]);
            }
        }
    }
}

TEST(calibration_atmosphere_operator, tau_zero_is_exact_unity) {
    Operator model;
    for (const int array : {0, 1, 2}) {
        for (const double elevation : {25.0, 42.5, 80.0}) {
            EXPECT_DOUBLE_EQ(
                model.line_of_sight_optical_depth(array, 0.0, elevation),
                0.0);
            EXPECT_DOUBLE_EQ(model.transmission(array, 0.0, elevation), 1.0);
            EXPECT_DOUBLE_EQ(
                model.extinction_correction(array, 0.0, elevation), 1.0);
        }
    }
}

TEST(calibration_atmosphere_operator, fails_closed_outside_support) {
    Operator model;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const double tau : {-0.001, 0.250001, nan}) {
        EXPECT_THROW(model.transmission(0, tau, 45.0), std::domain_error);
    }
    for (const double elevation : {24.999, 80.001, nan}) {
        EXPECT_THROW(
            model.transmission(0, 0.1, elevation), std::domain_error);
    }
    EXPECT_THROW(model.transmission(-1, 0.1, 45.0), std::out_of_range);
    EXPECT_THROW(model.transmission(3, 0.1, 45.0), std::out_of_range);
}

TEST(calibration_atmosphere_operator, is_positive_monotone_and_continuous) {
    Operator model;
    for (const int alpha : {-1, 0, 2, 4}) {
        model.select_reference_spectral_index(alpha);
        for (const int array : {0, 1, 2}) {
            for (const double elevation : {25.0, 31.5, 45.0, 67.5, 80.0}) {
                double previous = 1.0;
                for (int index = 0; index <= 100; ++index) {
                    const double tau = 0.25 * static_cast<double>(index) / 100.0;
                    const double correction =
                        model.extinction_correction(array, tau, elevation);
                    EXPECT_TRUE(std::isfinite(correction));
                    EXPECT_GE(correction, previous);
                    previous = correction;
                }
            }
            for (const double tau : {0.0504874104674104401, 0.15, 0.2, 0.25}) {
                double previous = model.extinction_correction(array, tau, 25.0);
                for (int elevation = 26; elevation <= 80; ++elevation) {
                    const double correction = model.extinction_correction(
                        array, tau, static_cast<double>(elevation));
                    EXPECT_LE(correction, previous);
                    previous = correction;
                }
            }
        }
    }
}

TEST(calibration_atmosphere_operator, has_no_operator_switch_at_tau015) {
    Operator model;
    EXPECT_EQ(
        Operator::quality_regime(0.15),
        "science_qualification_regime");
    EXPECT_EQ(
        Operator::quality_regime(std::nextafter(0.15, 1.0)),
        "engineering_availability_regime");
    for (const int array : {0, 1, 2}) {
        for (const double elevation : {25.0, 45.0, 80.0}) {
            const double at = model.line_of_sight_optical_depth(
                array, 0.15, elevation);
            const double left = model.line_of_sight_optical_depth(
                array, std::nextafter(0.15, 0.0), elevation);
            const double right = model.line_of_sight_optical_depth(
                array, std::nextafter(0.15, 1.0), elevation);
            EXPECT_NEAR(left, at, 4.0 * std::numeric_limits<double>::epsilon());
            EXPECT_NEAR(right, at, 4.0 * std::numeric_limits<double>::epsilon());
        }
    }
}

TEST(calibration_atmosphere_operator, uses_sample_elevation_los_nodes_once) {
    timestream::Calibration calibration;
    calibration.select_reference_spectral_index(0.0);
    calibration.setup(0.15);
    Eigen::Vector2d elevation_rad;
    elevation_rad << 25.0 * pi / 180.0, 80.0 * pi / 180.0;
    const auto los = calibration.calc_tau(elevation_rad, 0.15);

    EXPECT_DOUBLE_EQ(
        los.at(0)(1), node_value(0, 0, "tau015", 80.0));
    EXPECT_GT(los.at(0)(0), los.at(0)(1));

    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd> data;
    data.scans.data = Eigen::MatrixXd::Ones(2, 1);
    data.fcf.data = Eigen::VectorXd::Ones(1);
    CalibrationFixture fixture;
    fixture.apt["array"] = Eigen::VectorXd::Zero(1);
    admit_test_product(calibration, 1);
    calibration.extinction_correction(data, fixture, los);

    EXPECT_DOUBLE_EQ(data.scans.data(0, 0), std::exp(los.at(0)(0)));
    EXPECT_DOUBLE_EQ(data.scans.data(1, 0), std::exp(los.at(0)(1)));
    EXPECT_DOUBLE_EQ(
        data.fcf.data(0),
        0.5 * (std::exp(los.at(0)(0)) + std::exp(los.at(0)(1))));
}

TEST(calibration_atmosphere_operator, rejects_invalid_factor_before_mutation) {
    timestream::Calibration calibration;
    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd> data;
    data.scans.data = Eigen::MatrixXd::Ones(2, 2);
    data.fcf.data = Eigen::VectorXd::Ones(2);
    CalibrationFixture fixture;
    fixture.apt["array"].resize(2);
    fixture.apt["array"] << 0.0, 1.0;
    std::map<int, Eigen::VectorXd> los;
    los[0] = Eigen::Vector2d::Constant(0.1);
    los[1] = Eigen::Vector2d::Constant(0.2);
    los[1](1) = std::numeric_limits<double>::quiet_NaN();
    const auto original = data.scans.data;
    admit_test_product(calibration, 2);

    EXPECT_THROW(
        calibration.extinction_correction(data, fixture, los),
        std::domain_error);
    EXPECT_TRUE(data.scans.data.isApprox(original, 0.0));
    EXPECT_TRUE(data.fcf.data.isOnes());
}

TEST(calibration_atmosphere_operator,
     beammap_uses_shared_surface_once_at_top_of_atmosphere_boundary) {
    timestream::Calibration calibration;
    calibration.setup(0.12);
    Eigen::VectorXd elevation_rad(1);
    elevation_rad << 45.0 * pi / 180.0;
    const auto los = calibration.calc_tau(elevation_rad, 0.12);

    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd> data;
    data.scans.data = Eigen::MatrixXd::Ones(1, 1);
    data.fcf.data = Eigen::VectorXd::Ones(1);
    CalibrationFixture fixture;
    fixture.apt["array"] = Eigen::VectorXd::Zero(1);
    admit_test_product(calibration, 1);
    calibration.extinction_correction(data, fixture, los);

    const double top_of_atmosphere_source_flux_mjy = 1000.0;
    const double fitted_corrected_amplitude = data.scans.data(0, 0);
    const double beammap_flxscale =
        top_of_atmosphere_source_flux_mjy / fitted_corrected_amplitude;
    EXPECT_DOUBLE_EQ(
        beammap_flxscale * fitted_corrected_amplitude,
        top_of_atmosphere_source_flux_mjy);
    EXPECT_NE(
        beammap_flxscale * fitted_corrected_amplitude
            * std::exp(los.at(0)(0)),
        top_of_atmosphere_source_flux_mjy);
}

#include <citlali/core/timestream/rtc/calibrate.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <map>
#include <string>

namespace {

timestream::CalibrationProductAdmissionInputs valid_inputs(
    Eigen::Index detector_count = 2, bool extinction = true) {
    timestream::CalibrationProductAdmissionInputs inputs;
    inputs.target_unit = "mJy/beam";
    inputs.calibration_requested = true;
    inputs.extinction_requested = extinction;
    inputs.responsivity_required = true;
    inputs.sensitivity_required = true;
    inputs.beam_template_required = true;
    inputs.acquisition_identity_available = true;
    inputs.acquisition_identity_valid = true;
    inputs.acquisition_identity_detail = "test explicit-key binding";
    inputs.apt_lineage_available = true;
    inputs.apt_lineage_valid = true;
    inputs.apt_lineage_detail = "test selected-row lineage";
    inputs.apt_artifact_sha256 = "test-apt-sha256";
    inputs.apt_row_association_sha256 = "test-row-association-sha256";
    inputs.apt_observation_identity = "134723";
    inputs.apt_matched_observation_identity = "137389";
    inputs.apt_selected_source = "Neptune";
    inputs.tolapt_manifest_association_sha256 =
        "test-tolapt-manifest-association-sha256";
    inputs.acquisition_binding_sha256 = "test-binding-sha256";
    inputs.raw_observation_identity = "test-raw-observation";
    inputs.acquisition_binding_mode = "explicit_test_join";
    inputs.acquisition_key_schema = "artifact+network+local_tone";
    inputs.response_identity = "originating=test-beam;realized=identity";
    inputs.atmosphere_operator_id = "test-fixed-operator";
    inputs.atmosphere_operator_contract_sha256 = "test-operator-contract";
    inputs.atmosphere_node_table_sha256 = "test-node-table";
    inputs.passband_set_id = "test-passband";
    inputs.reference_profile_id = "test-reference-profile";
    inputs.reference_spectral_index_alpha = 0.0;
    inputs.reference_spectral_index_default_applied = true;
    inputs.tau225 = 0.1;
    inputs.applied_sample_extinction_state_sha256 =
        "test-complete-sample-extinction-state";
    inputs.package_lineage.selected_apt_source_path = "/test/apt.ecsv";
    inputs.package_lineage.selected_apt_sha256 = inputs.apt_artifact_sha256;
    inputs.package_lineage.apt_row_association_sha256 =
        inputs.apt_row_association_sha256;
    inputs.package_lineage.modern_tolapt_manifest_available = true;
    inputs.package_lineage.modern_tolapt_manifest_path =
        "/test/tolapt/manifest.yaml";
    inputs.package_lineage.modern_tolapt_manifest_sha256 =
        "test-tolapt-manifest-sha256";
    inputs.package_lineage.modern_tolapt_contract_version = "tolapt.run.v1";
    inputs.package_lineage.modern_tolapt_run_id = "test-run";
    inputs.package_lineage.modern_tolapt_output_key = "matched_design_apt";
    inputs.package_lineage.modern_tolapt_output_path = "matched.ecsv";
    inputs.package_lineage.tolapt_manifest_association_sha256 =
        inputs.tolapt_manifest_association_sha256;
    inputs.package_lineage.modern_tolapt_design_input =
        {"design.ecsv", "test-design-sha256", 1,
         "2026-08-09T00:00:00Z"};
    inputs.package_lineage.modern_tolapt_measured_input =
        {"measured.ecsv", "test-measured-sha256", 1,
         "2026-08-09T00:00:01Z"};
    inputs.package_lineage.raw_artifacts.push_back(
        {"test-raw.nc", "test-raw-sha256", "toltec0", 0,
         std::vector<double>(static_cast<std::size_t>(detector_count),
                             1.0e9)});
    for (Eigen::Index detector = 0; detector < detector_count; ++detector) {
        timestream::CalibrationLineageRow row;
        row.ordered_detector_index = detector;
        row.selected_source_row_index = detector;
        row.network = 0;
        row.network_local_tone = detector;
        row.absolute_tone_frequency_hz = 1.0e9;
        row.uid = std::to_string(detector);
        row.eligible = true;
        row.validity_basis = "test-valid-row";
        row.stable_association = "test-stable-association-" +
            std::to_string(detector);
        inputs.package_lineage.ordered_rows.push_back(std::move(row));
    }
    inputs.target_unit_factor = Eigen::VectorXd::Ones(detector_count);
    inputs.detector_flxscale = Eigen::VectorXd::Ones(detector_count);
    inputs.detector_responsivity = Eigen::VectorXd::Ones(detector_count);
    inputs.detector_sensitivity = Eigen::VectorXd::Ones(detector_count);
    inputs.detector_beam_major_fwhm_arcsec =
        Eigen::VectorXd::Constant(detector_count, 10.0);
    inputs.detector_beam_minor_fwhm_arcsec =
        Eigen::VectorXd::Constant(detector_count, 9.0);
    inputs.minimum_extinction_correction =
        Eigen::VectorXd::Ones(detector_count);
    inputs.maximum_extinction_correction =
        Eigen::VectorXd::Constant(detector_count, extinction ? 1.5 : 1.0);
    return inputs;
}

struct CalibrationFixture {
    std::map<std::string, Eigen::VectorXd> apt;
};

}  // namespace

TEST(calibration_product, admits_only_complete_atomic_product) {
    const auto product = timestream::admit_calibration_product(valid_inputs());
    ASSERT_TRUE(product.valid());
    EXPECT_EQ(product.validity_cause,
              timestream::CalibrationValidityCause::valid_complete_product);
    EXPECT_EQ(product.target_unit, "mJy/beam");
    EXPECT_FALSE(product.apt_artifact_sha256.empty());
    EXPECT_EQ(product.apt_row_association_sha256,
              "test-row-association-sha256");
    EXPECT_EQ(product.apt_observation_identity, "134723");
    EXPECT_EQ(product.apt_matched_observation_identity, "137389");
    EXPECT_EQ(product.apt_selected_source, "Neptune");
    EXPECT_EQ(product.tolapt_manifest_association_sha256,
              "test-tolapt-manifest-association-sha256");
    EXPECT_FALSE(product.response_identity.empty());
    EXPECT_FALSE(product.calibration_identity.empty());
    EXPECT_NE(product.calibration_identity,
              product.acquisition_binding_sha256);
    EXPECT_FALSE(product.factor_state_sha256.empty());
    ASSERT_FALSE(product.nuisances.empty());
    for (const auto &nuisance : product.nuisances) {
        EXPECT_FALSE(nuisance.correlation_scope.empty());
        EXPECT_NE(nuisance.uncertainty_source, "zero");
        EXPECT_FALSE(nuisance.validity.empty());
    }
    EXPECT_EQ(product.nuisances.front().value_availability,
              timestream::CalibrationNuisanceAvailability::available);
    EXPECT_EQ(product.nuisances.front().availability,
              timestream::CalibrationNuisanceAvailability::unavailable);
}

TEST(calibration_product,
     complete_sample_extinction_identity_prevents_equal_extrema_collision) {
    auto first_inputs = valid_inputs();
    auto second_inputs = first_inputs;
    first_inputs.applied_sample_extinction_state_sha256 =
        "sample-sequence-digest-a";
    second_inputs.applied_sample_extinction_state_sha256 =
        "sample-sequence-digest-b";
    const auto first =
        timestream::admit_calibration_product(first_inputs);
    const auto second =
        timestream::admit_calibration_product(second_inputs);
    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(second.valid());
    EXPECT_EQ(first.minimum_extinction_correction,
              second.minimum_extinction_correction);
    EXPECT_EQ(first.maximum_extinction_correction,
              second.maximum_extinction_correction);
    EXPECT_NE(first.factor_state_sha256, second.factor_state_sha256);
    EXPECT_NE(first.calibration_identity, second.calibration_identity);
}

TEST(calibration_product,
     canonical_identity_binds_complete_admitted_state_not_only_acquisition) {
    const auto baseline =
        timestream::admit_calibration_product(valid_inputs());
    ASSERT_TRUE(baseline.valid());

    auto changed_unit_factor = valid_inputs();
    changed_unit_factor.target_unit_factor(0) = 2.0;
    const auto factor_product =
        timestream::admit_calibration_product(changed_unit_factor);
    ASSERT_TRUE(factor_product.valid());
    EXPECT_EQ(factor_product.acquisition_binding_sha256,
              baseline.acquisition_binding_sha256);
    EXPECT_NE(factor_product.factor_state_sha256,
              baseline.factor_state_sha256);
    EXPECT_NE(factor_product.calibration_identity,
              baseline.calibration_identity);

    auto changed_response = valid_inputs();
    changed_response.response_identity += ";filtering=fir";
    const auto response_product =
        timestream::admit_calibration_product(changed_response);
    ASSERT_TRUE(response_product.valid());
    EXPECT_EQ(response_product.acquisition_binding_sha256,
              baseline.acquisition_binding_sha256);
    EXPECT_NE(response_product.calibration_identity,
              baseline.calibration_identity);

    auto changed_source_association = valid_inputs();
    changed_source_association.apt_selected_source = "Uranus";
    const auto source_product =
        timestream::admit_calibration_product(changed_source_association);
    ASSERT_TRUE(source_product.valid());
    EXPECT_NE(source_product.calibration_identity,
              baseline.calibration_identity);
}

TEST(calibration_product, rejects_unsupported_production_units) {
    for (const std::string unit : {"MJy/sr", "uK", "Jy/pixel", "Jy/beam"}) {
        auto inputs = valid_inputs();
        inputs.target_unit = unit;
        const auto product = timestream::admit_calibration_product(inputs);
        EXPECT_FALSE(product.valid());
        EXPECT_EQ(product.validity_cause,
                  timestream::CalibrationValidityCause::unsupported_target_unit);
    }
}

TEST(calibration_product, rejects_every_invalid_required_factor_class) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const int factor_class : {0, 1, 2, 3, 4, 5}) {
        auto inputs = valid_inputs();
        switch (factor_class) {
            case 0: inputs.target_unit_factor(0) = nan; break;
            case 1: inputs.detector_flxscale(0) = 0.0; break;
            case 2: inputs.detector_responsivity(0) = nan; break;
            case 3: inputs.detector_sensitivity(0) = -1.0; break;
            case 4: inputs.detector_beam_major_fwhm_arcsec(0) = 0.0; break;
            case 5: inputs.maximum_extinction_correction(0) = nan; break;
        }
        const auto product = timestream::admit_calibration_product(inputs);
        EXPECT_FALSE(product.valid());
        EXPECT_TRUE(
            product.validity_cause ==
                timestream::CalibrationValidityCause::invalid_required_factor ||
            product.validity_cause ==
                timestream::CalibrationValidityCause::invalid_atmosphere_support);
    }
}

TEST(calibration_product, distinguishes_missing_from_invalid_required_factors) {
    auto missing = valid_inputs();
    missing.detector_responsivity.resize(0);
    EXPECT_EQ(timestream::admit_calibration_product(missing).validity_cause,
              timestream::CalibrationValidityCause::missing_required_factor);

    auto invalid = valid_inputs();
    invalid.detector_responsivity(0) = 0.0;
    EXPECT_EQ(timestream::admit_calibration_product(invalid).validity_cause,
              timestream::CalibrationValidityCause::invalid_required_factor);
}

TEST(calibration_product, preserves_acquisition_identity_failure_causes) {
    auto unavailable = valid_inputs();
    unavailable.acquisition_identity_available = false;
    EXPECT_EQ(timestream::admit_calibration_product(unavailable).validity_cause,
              timestream::CalibrationValidityCause::acquisition_identity_unavailable);

    auto invalid = valid_inputs();
    invalid.acquisition_identity_valid = false;
    EXPECT_EQ(timestream::admit_calibration_product(invalid).validity_cause,
              timestream::CalibrationValidityCause::acquisition_identity_invalid);

    auto invalid_lineage = valid_inputs();
    invalid_lineage.apt_lineage_valid = false;
    invalid_lineage.apt_lineage_detail = "conflicting selected APT lineage";
    const auto lineage_product =
        timestream::admit_calibration_product(invalid_lineage);
    EXPECT_EQ(lineage_product.validity_cause,
              timestream::CalibrationValidityCause::acquisition_identity_invalid);
    EXPECT_EQ(lineage_product.validity_detail,
              "conflicting selected APT lineage");
}

TEST(calibration_product, transfers_only_conditional_variance_and_weight) {
    EXPECT_DOUBLE_EQ(timestream::transfer_conditional_variance(4.0, 3.0),
                     36.0);
    EXPECT_DOUBLE_EQ(
        timestream::transfer_conditional_inverse_variance(0.25, 3.0),
        0.25 / 9.0);
    EXPECT_TRUE(std::isnan(timestream::transfer_conditional_variance(-1.0, 2.0)));
    EXPECT_TRUE(std::isnan(
        timestream::transfer_conditional_inverse_variance(1.0, 0.0)));
}

TEST(calibration_product,
     inventories_every_existing_calibration_recipient_truthfully) {
    const std::string semantics{
        timestream::CalibrationProduct::weight_recipient_semantics};
    for (const std::string &recipient : {
             "approximate_weight:", "hybrid_weight:",
             "validated_weight:", "full_weight:", "constant_weight:",
             "naive_map_signal:", "naive_map_weight:",
             "noise_variance_I:"}) {
        const auto begin = semantics.find(recipient);
        ASSERT_NE(begin, std::string::npos)
            << "missing recipient=" << recipient;
        const auto end = semantics.find(';', begin);
        ASSERT_NE(end, std::string::npos)
            << "unterminated recipient=" << recipient;
        const auto record = semantics.substr(begin, end - begin);
        EXPECT_TRUE(record.find("coefficient=") != std::string::npos ||
                    record.find("recipient=") != std::string::npos)
            << "missing coefficient/recipient role=" << recipient;
        for (const std::string &required_field : {
                 "stage=", "units=", "normalization=", "support=",
                 "calibration="}) {
            EXPECT_NE(record.find(required_field), std::string::npos)
                << "recipient=" << recipient
                << " missing inventory field=" << required_field;
        }
    }
    EXPECT_NE(semantics.find(
                  "validated_weight:baseline=approximate_weight"),
              std::string::npos);
    EXPECT_NE(semantics.find(
                  "selected_APT_sens_contains_flxscale_once"),
              std::string::npos);
    EXPECT_NE(semantics.find(
                  "noise_variance_I:recipient=conditional_finite_stack_scatter_of_normalized_noise_realizations"),
              std::string::npos);
    EXPECT_NE(semantics.find(
                  "all_total_precision_and_significance_claims_fail_closed_without_nuisance_covariance"),
              std::string::npos);
}

TEST(calibration_product, rejection_precedes_tod_mutation) {
    timestream::Calibration calibration;
    auto inputs = valid_inputs();
    inputs.detector_flxscale(0) = 0.0;
    EXPECT_THROW(calibration.admit_product(inputs), std::domain_error);
    EXPECT_EQ(calibration.product.validity_cause,
              timestream::CalibrationValidityCause::invalid_required_factor);

    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd> data;
    data.scans.data = Eigen::MatrixXd::Ones(2, 2);
    data.fcf.data = Eigen::VectorXd::Ones(2);
    const auto original = data.scans.data;
    CalibrationFixture fixture;
    EXPECT_THROW(calibration.calibrate_tod(data, fixture), std::runtime_error);
    EXPECT_TRUE(data.scans.data.isApprox(original, 0.0));
    EXPECT_TRUE(data.fcf.data.isOnes());
}

TEST(calibration_product, production_path_applies_each_factor_once) {
    timestream::Calibration calibration;
    auto inputs = valid_inputs(2, true);
    inputs.target_unit_factor << 2.0, 3.0;
    inputs.detector_flxscale << 5.0, 7.0;
    inputs.minimum_extinction_correction << 2.0, 4.0;
    inputs.maximum_extinction_correction = inputs.minimum_extinction_correction;
    calibration.admit_product(inputs);

    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd> data;
    data.scans.data = Eigen::MatrixXd::Ones(1, 2);
    data.fcf.data = Eigen::VectorXd::Ones(2);
    CalibrationFixture fixture;
    fixture.apt["array"] = (Eigen::Vector2d() << 0.0, 1.0).finished();
    calibration.calibrate_tod(data, fixture);
    std::map<int, Eigen::VectorXd> los;
    los[0] = Eigen::VectorXd::Constant(1, std::log(2.0));
    los[1] = Eigen::VectorXd::Constant(1, std::log(4.0));
    calibration.extinction_correction(data, fixture, los);

    EXPECT_DOUBLE_EQ(data.scans.data(0, 0), 2.0 * 5.0 * 2.0);
    EXPECT_DOUBLE_EQ(data.scans.data(0, 1), 3.0 * 7.0 * 4.0);
    EXPECT_DOUBLE_EQ(data.fcf.data(0), 2.0 * 2.0);
    EXPECT_DOUBLE_EQ(data.fcf.data(1), 3.0 * 4.0);
}

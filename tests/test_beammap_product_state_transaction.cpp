#include <citlali/core/engine/beammap.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

template <typename Left, typename Right>
void expect_eigen_exact(const Left &left, const Right &right) {
    ASSERT_EQ(left.rows(), right.rows());
    ASSERT_EQ(left.cols(), right.cols());
    EXPECT_TRUE((left.array() == right.array()).all());
}

void expect_directional_product_state_exact(
    const citlali::engine_detail::beammap::DirectionalProduct &left,
    const citlali::engine_detail::beammap::DirectionalProduct &right) {
    EXPECT_EQ(left.mode, right.mode);
    EXPECT_EQ(left.calib.apt_filepath, right.calib.apt_filepath);
    ASSERT_EQ(left.calib.apt.size(), right.calib.apt.size());
    for (const auto &[key, values] : left.calib.apt) {
        ASSERT_TRUE(right.calib.apt.count(key));
        expect_eigen_exact(values, right.calib.apt.at(key));
    }
    EXPECT_EQ(YAML::Dump(left.calib.apt_meta),
              YAML::Dump(right.calib.apt_meta));
    EXPECT_EQ(left.calib.n_dets, right.calib.n_dets);
    EXPECT_EQ(left.calib.n_nws, right.calib.n_nws);
    EXPECT_EQ(left.calib.n_arrays, right.calib.n_arrays);
    expect_eigen_exact(left.calib.nws, right.calib.nws);
    expect_eigen_exact(left.calib.arrays, right.calib.arrays);
    expect_eigen_exact(left.calib.flux_conversion_factor,
                       right.calib.flux_conversion_factor);
    EXPECT_EQ(left.calib.mean_flux_conversion_factor,
              right.calib.mean_flux_conversion_factor);
    expect_eigen_exact(left.params, right.params);
    expect_eigen_exact(left.perrors, right.perrors);
    expect_eigen_exact(left.p0, right.p0);
    expect_eigen_exact(left.perror0, right.perror0);
    expect_eigen_exact(left.converged, right.converged);
    expect_eigen_exact(left.converge_iter, right.converge_iter);
    expect_eigen_exact(left.good_fits, right.good_fits);
    expect_eigen_exact(left.flag2, right.flag2);
    expect_eigen_exact(left.fit_diag_init_params,
                       right.fit_diag_init_params);
    expect_eigen_exact(left.fit_diag_lower_limits,
                       right.fit_diag_lower_limits);
    expect_eigen_exact(left.fit_diag_upper_limits,
                       right.fit_diag_upper_limits);
    expect_eigen_exact(left.fit_diag_hit_lower, right.fit_diag_hit_lower);
    expect_eigen_exact(left.fit_diag_hit_upper, right.fit_diag_hit_upper);
    expect_eigen_exact(left.fit_diag_bound_code, right.fit_diag_bound_code);
    expect_eigen_exact(left.fit_diag_bound_nhit, right.fit_diag_bound_nhit);
    expect_eigen_exact(left.prior_diag_values, right.prior_diag_values);
    expect_eigen_exact(left.final_prior_d2_diag,
                       right.final_prior_d2_diag);
    expect_eigen_exact(left.final_prior_slot_index_diag,
                       right.final_prior_slot_index_diag);
    EXPECT_EQ(left.reference_detector, right.reference_detector);
    EXPECT_EQ(left.priors_centered, right.priors_centered);
    EXPECT_EQ(left.priors_derotated, right.priors_derotated);
    EXPECT_EQ(left.prior_center_x_arcsec, right.prior_center_x_arcsec);
    EXPECT_EQ(left.prior_center_y_arcsec, right.prior_center_y_arcsec);
    ASSERT_EQ(left.prior_alignment.size(), right.prior_alignment.size());
    for (const auto &[array, alignment] : left.prior_alignment) {
        ASSERT_TRUE(right.prior_alignment.count(array));
        const auto &expected = right.prior_alignment.at(array);
        EXPECT_EQ(alignment.valid, expected.valid);
        EXPECT_EQ(alignment.cos_theta, expected.cos_theta);
        EXPECT_EQ(alignment.sin_theta, expected.sin_theta);
        EXPECT_EQ(alignment.theta_rad, expected.theta_rad);
        EXPECT_EQ(alignment.dx_arcsec, expected.dx_arcsec);
        EXPECT_EQ(alignment.dy_arcsec, expected.dy_arcsec);
        EXPECT_EQ(alignment.n_matches, expected.n_matches);
        EXPECT_EQ(alignment.rms_arcsec, expected.rms_arcsec);
    }
    EXPECT_EQ(left.source_flux_mjy_beam, right.source_flux_mjy_beam);
    EXPECT_EQ(left.source_flux_mjy_sr, right.source_flux_mjy_sr);
}

void seed_transaction_test_state(Beammap &beammap) {
    beammap.calib.apt_filepath = "standard-input.ecsv";
    beammap.calib.apt["uid"] = Eigen::Vector2d{101.0, 102.0};
    beammap.calib.apt_meta["beammap_direction_mode"] = "standard";
    beammap.calib.apt_meta["transaction_token"] = "preserve-me";
    beammap.calib.n_dets = 2;
    beammap.calib.n_nws = 1;
    beammap.calib.n_arrays = 1;
    beammap.calib.nws = Eigen::VectorXI::Constant(1, 7);
    beammap.calib.arrays = Eigen::VectorXI::Constant(1, 2);
    beammap.calib.flux_conversion_factor = Eigen::Vector2d{3.0, 4.0};
    beammap.calib.mean_flux_conversion_factor["a2000"] = 3.5;
    beammap.params = Eigen::MatrixXd::Constant(2, 7, 1.0);
    beammap.perrors = Eigen::MatrixXd::Constant(2, 7, 2.0);
    beammap.p0 = Eigen::MatrixXd::Constant(2, 7, 3.0);
    beammap.perror0 = Eigen::MatrixXd::Constant(2, 7, 4.0);
    beammap.converged =
        Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(2, true);
    beammap.converge_iter = Eigen::VectorXi::Constant(2, 5);
    beammap.good_fits =
        Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(2, true);
    beammap.flag2 =
        Eigen::Matrix<uint16_t, Eigen::Dynamic, 1>::Constant(2, 6);
    beammap.fit_diag_init_params = Eigen::MatrixXd::Constant(2, 7, 7.0);
    beammap.fit_diag_lower_limits = Eigen::MatrixXd::Constant(2, 7, 8.0);
    beammap.fit_diag_upper_limits = Eigen::MatrixXd::Constant(2, 7, 9.0);
    beammap.fit_diag_hit_lower = Eigen::MatrixXi::Constant(2, 7, 10);
    beammap.fit_diag_hit_upper = Eigen::MatrixXi::Constant(2, 7, 11);
    beammap.fit_diag_bound_code = Eigen::VectorXi::Constant(2, 12);
    beammap.fit_diag_bound_nhit = Eigen::VectorXi::Constant(2, 13);
    beammap.prior_diag_values = Eigen::MatrixXd::Constant(2, 3, 14.0);
    beammap.final_prior_d2_diag = Eigen::Vector2d{15.0, 16.0};
    beammap.final_prior_slot_index_diag = Eigen::Vector2i{17, 18};
    beammap.beammap_reference_det_found = 19;
    beammap.beammap_soft_priors_are_centered = true;
    beammap.beammap_soft_priors_are_derotated = true;
    beammap.beammap_prior_array_center_x_arcsec[2] = 20.0;
    beammap.beammap_prior_array_center_y_arcsec[2] = 21.0;
    auto &alignment = beammap.beammap_prior_array_alignment[2];
    alignment.valid = true;
    alignment.cos_theta = 0.5;
    alignment.sin_theta = 0.25;
    alignment.theta_rad = 0.75;
    alignment.dx_arcsec = 22.0;
    alignment.dy_arcsec = 23.0;
    alignment.n_matches = 24;
    alignment.rms_arcsec = 25.0;
    beammap.source_flux_mJy_beam["a2000"] = 26.0;
    beammap.source_flux_MJy_Sr["a2000"] = 27.0;
}

TEST(BeammapProductStateTransaction, RestoresExactStandardStateAfterFailure) {
    using citlali::config::BeammapDirectionMode;
    using citlali::engine_detail::beammap::DirectionalProduct;
    using citlali::engine_detail::beammap::ProductStateTransaction;
    using citlali::engine_detail::beammap::capture_product_state;
    using citlali::engine_detail::beammap::restore_product_state;

    Beammap beammap;
    seed_transaction_test_state(beammap);
    const auto expected = capture_product_state(
        beammap, BeammapDirectionMode::standard);

    EXPECT_THROW(
        [&]() {
            ProductStateTransaction transaction{beammap};
            restore_product_state(beammap, DirectionalProduct{});
            throw std::runtime_error("injected directional fit failure");
        }(),
        std::runtime_error);

    const auto actual = capture_product_state(
        beammap, BeammapDirectionMode::standard);
    expect_directional_product_state_exact(actual, expected);
}

TEST(BeammapProductStateTransaction, IsolatesNestedYamlMetadataMutation) {
    using citlali::config::BeammapDirectionMode;
    using citlali::engine_detail::beammap::ProductStateTransaction;
    using citlali::engine_detail::beammap::capture_product_state;

    Beammap beammap;
    seed_transaction_test_state(beammap);
    const auto expected = capture_product_state(
        beammap, BeammapDirectionMode::standard);

    {
        ProductStateTransaction transaction{beammap};
        beammap.calib.apt_meta["beammap_direction_mode"] = "right";
        beammap.calib.apt_meta["transaction_token"] = "mutated";
    }

    const auto actual = capture_product_state(
        beammap, BeammapDirectionMode::standard);
    expect_directional_product_state_exact(actual, expected);
    EXPECT_EQ(actual.calib.apt_meta["beammap_direction_mode"].as<std::string>(),
              "standard");
    EXPECT_EQ(actual.calib.apt_meta["transaction_token"].as<std::string>(),
              "preserve-me");
}

TEST(BeammapProductStateTransaction, ClonesIndependentDirectionalMetadata) {
    using citlali::engine_detail::beammap::clone_product_calib;

    Beammap beammap;
    seed_transaction_test_state(beammap);
    const engine::Calib common = clone_product_calib(beammap.calib);
    engine::Calib left = clone_product_calib(common);
    left.apt_meta["beammap_direction_mode"] = "left";
    left.apt_meta["transaction_token"] = "left-fit";
    engine::Calib right = clone_product_calib(common);
    right.apt_meta["beammap_direction_mode"] = "right";
    right.apt_meta["transaction_token"] = "right-fit";

    EXPECT_EQ(common.apt_meta["beammap_direction_mode"].as<std::string>(),
              "standard");
    EXPECT_EQ(common.apt_meta["transaction_token"].as<std::string>(),
              "preserve-me");
    EXPECT_EQ(left.apt_meta["beammap_direction_mode"].as<std::string>(),
              "left");
    EXPECT_EQ(right.apt_meta["beammap_direction_mode"].as<std::string>(),
              "right");
}

TEST(BeammapProductStateTransaction, RestoresMapBufferAfterFailure) {
    using citlali::engine_detail::beammap::ObservationMapBufferTransaction;
    mapmaking::MapBuffer standard{"standard"};
    mapmaking::MapBuffer directional{"left"};
    standard.n_rows = 11;
    directional.n_rows = 22;

    EXPECT_THROW(
        ([&]() {
            ObservationMapBufferTransaction transaction{
                standard, directional};
            EXPECT_EQ(standard.name, "left");
            EXPECT_EQ(standard.n_rows, 22);
            throw std::runtime_error("injected map output failure");
        }()),
        std::runtime_error);

    EXPECT_EQ(standard.name, "standard");
    EXPECT_EQ(standard.n_rows, 11);
    EXPECT_EQ(directional.name, "left");
    EXPECT_EQ(directional.n_rows, 22);
}

}  // namespace

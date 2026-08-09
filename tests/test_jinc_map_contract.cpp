#include <gtest/gtest.h>

#include <citlali/core/config/mapmaking_config_validation.h>
#include <citlali/core/mapmaking/jinc_contract.h>
#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/pipeline/beammap_mapmaking_policy.h>
#include <citlali/core/pipeline/mapmaking_provenance.h>
#include <citlali/core/pipeline/map_buffer_allocation.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

TEST(jinc_map_contract, direct_signed_lobe_equations) {
    const double q1 = 2.0;
    const double q2 = 3.0;
    const double c1 = 1.0;
    const double c2 = -0.25;
    const double d1 = 5.0;
    const double d2 = 8.0;
    const double n = q1 * c1 * d1 + q2 * c2 * d2;
    const double c = q1 * c1 + q2 * c2;
    const double q = q1 * c1 * c1 + q2 * c2 * c2;
    const double sum_abs = std::abs(q1 * c1) + std::abs(q2 * c2);

    const auto result = mapmaking::finalize_jinc_accumulators(
        n, c, q, sum_abs, 2);
    ASSERT_TRUE(result.formal_support);
    EXPECT_DOUBLE_EQ(n, 4.0);
    EXPECT_DOUBLE_EQ(c, 1.25);
    EXPECT_DOUBLE_EQ(q, 2.1875);
    EXPECT_DOUBLE_EQ(result.signal, n / c);
    EXPECT_DOUBLE_EQ(result.formal_weight, c * c / q);
}

TEST(jinc_map_contract, cancellation_rho_and_invalid_q_boundaries) {
    const auto exact = mapmaking::finalize_jinc_accumulators(
        1.0, 0.0, 2.0, 2.0, 100);
    EXPECT_TRUE(exact.exact_cancellation);
    EXPECT_FALSE(exact.formal_support);

    const auto unresolved = mapmaking::finalize_jinc_accumulators(
        1.0, 1e-15, 2.0, 2.0, 100);
    EXPECT_LT(unresolved.rho, unresolved.rho_resolution_bound);
    EXPECT_FALSE(unresolved.formal_support);

    const auto resolved = mapmaking::finalize_jinc_accumulators(
        1.0, 1e-10, 2.0, 2.0, 100);
    EXPECT_GE(resolved.rho, resolved.rho_resolution_bound);
    EXPECT_TRUE(resolved.formal_support);
    EXPECT_GT(resolved.formal_weight, 0.0);

    EXPECT_FALSE(mapmaking::finalize_jinc_accumulators(
        1.0, 1.0, 0.0, 1.0, 1).formal_support);
    EXPECT_FALSE(mapmaking::finalize_jinc_accumulators(
        1.0, 1.0, -1.0, 1.0, 1).formal_support);
    EXPECT_FALSE(mapmaking::finalize_jinc_accumulators(
        1.0, 1.0, std::numeric_limits<double>::infinity(), 1.0, 1)
                     .formal_support);
}

TEST(jinc_map_contract, unit_rescaling_and_extreme_finite_range) {
    const auto base = mapmaking::finalize_jinc_accumulators(
        12.0, 3.0, 5.0, 7.0, 4);
    constexpr double scale = 1e120;
    const auto rescaled = mapmaking::finalize_jinc_accumulators(
        12.0 / scale, 3.0 / (scale * scale),
        5.0 / (scale * scale), 7.0 / (scale * scale), 4);
    ASSERT_TRUE(base.formal_support);
    ASSERT_TRUE(rescaled.formal_support);
    EXPECT_DOUBLE_EQ(base.rho, rescaled.rho);
    EXPECT_NEAR(rescaled.signal / scale, base.signal, 1e-14);
    EXPECT_NEAR(rescaled.formal_weight * scale * scale,
                base.formal_weight, 1e-14);

    const auto extreme = mapmaking::finalize_jinc_accumulators(
        1e200, 1e150, 1e300, 1e150, 1);
    ASSERT_TRUE(extreme.formal_support);
    EXPECT_DOUBLE_EQ(extreme.signal, 1e50);
    EXPECT_DOUBLE_EQ(extreme.formal_weight, 1.0);
}

TEST(jinc_map_contract, phase_bins_are_point_quantized_and_bounded) {
    EXPECT_EQ(mapmaking::jinc_phase_bin(-0.5, 1), 0);
    EXPECT_EQ(mapmaking::jinc_phase_bin(0.5, 1), 0);
    EXPECT_EQ(mapmaking::jinc_phase_bin(-0.5, 4), 0);
    EXPECT_EQ(mapmaking::jinc_phase_bin(-0.25, 4), 1);
    EXPECT_EQ(mapmaking::jinc_phase_bin(0.0, 4), 2);
    EXPECT_EQ(mapmaking::jinc_phase_bin(0.5, 4), 3);
    EXPECT_THROW(mapmaking::jinc_phase_bin(0.0, 0), std::invalid_argument);
    EXPECT_THROW(mapmaking::jinc_phase_bin(
                     std::numeric_limits<double>::quiet_NaN(), 4),
                 std::invalid_argument);
}

TEST(jinc_map_contract, phase_refinement_samples_bin_centers_not_pixel_area) {
    mapmaking::JincMapmaker coarse;
    coarse.r_max = 3.0;
    coarse.subpixel_n = 1;
    coarse.array_names = {{0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    coarse.shape_params = {
        {0, Eigen::Vector3d(1.1, 1.67, 2.0)},
        {1, Eigen::Vector3d(1.1, 2.17, 2.0)},
        {2, Eigen::Vector3d(1.1, 3.17, 2.0)},
    };
    const double pixel_size = ((1.1 / 1000.0) / 45.0) / 2.0;
    coarse.allocate_jinc_matrix(pixel_size);
    EXPECT_TRUE(coarse.jinc_weights_mat_subpix.empty());
    EXPECT_DOUBLE_EQ(coarse.jinc_weights_mat.at(0)(6, 6), 1.0);

    auto refined = std::move(coarse);
    refined.subpixel_n = 4;
    refined.allocate_jinc_matrix(pixel_size);
    ASSERT_EQ(refined.jinc_weights_mat_subpix.at(0).size(), 16U);
    const double center_offset = -0.5 + 0.5 / 4.0;
    const double radius = pixel_size * std::sqrt(
        center_offset * center_offset + center_offset * center_offset);
    const double expected = refined.jinc_func(
        radius, 1.1, 1.67, 2.0, 3.0, (1.1 / 1000.0) / 45.0);
    EXPECT_DOUBLE_EQ(
        refined.jinc_weights_mat_subpix.at(0).front()(6, 6), expected);
    EXPECT_EQ(mapmaking::jinc_phase_bin(-0.5 + 1e-12, 4), 0);
    EXPECT_EQ(mapmaking::jinc_phase_bin(-0.25 - 1e-12, 4), 0);
    EXPECT_EQ(mapmaking::jinc_phase_bin(-0.25, 4), 1);
    EXPECT_LE(0.5 / 4.0, 0.5);
}

TEST(jinc_map_contract, square_cache_retains_corner_beyond_r_max) {
    mapmaking::JincMapmaker maker;
    maker.r_max = 3.0;
    maker.subpixel_n = 1;
    maker.array_names = {{0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    maker.shape_params = {
        {0, Eigen::Vector3d(1.1, 1.67, 2.0)},
        {1, Eigen::Vector3d(1.1, 2.17, 2.0)},
        {2, Eigen::Vector3d(1.1, 3.17, 2.0)},
    };
    const double pixel_size = ((1.1 / 1000.0) / 45.0) / 2.0;
    maker.allocate_jinc_matrix(pixel_size);

    const auto &resolved = maker.resolved_arrays.front();
    ASSERT_EQ(resolved.cache_half_width_pixels, 6);
    ASSERT_EQ(maker.jinc_weights_mat.at(0).rows(), 13);
    const double corner_radius = std::sqrt(2.0) * 6.0 * pixel_size;
    EXPECT_GT(corner_radius,
              resolved.r_max * resolved.array_scale_rad);
    EXPECT_TRUE(std::isfinite(maker.jinc_weights_mat.at(0)(0, 0)));
    EXPECT_NE(maker.jinc_weights_mat.at(0)(0, 0), 0.0);
}

TEST(jinc_map_contract, response_is_finite_below_at_and_above_r_max) {
    mapmaking::JincMapmaker maker;
    const double array_scale = (1.1 / 1000.0) / 45.0;
    const double r_max = 3.0;
    const auto response = [&](double normalized_radius) {
        return maker.jinc_func(normalized_radius * array_scale,
                               1.1, 1.67, 2.0, r_max, array_scale);
    };
    const double below = response(std::nextafter(r_max, 0.0));
    const double equal = response(r_max);
    const double above = response(
        std::nextafter(r_max, std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(std::isfinite(below));
    EXPECT_TRUE(std::isfinite(equal));
    EXPECT_TRUE(std::isfinite(above));
    EXPECT_NE(below, 0.0);
    EXPECT_NE(above, 0.0);
}

TEST(jinc_map_contract, square_cache_is_cropped_only_by_map_edges) {
    const auto corner = mapmaking::jinc_square_crop(5, 7, 0, 0, 5, 5);
    EXPECT_EQ(corner.map_row, 0);
    EXPECT_EQ(corner.map_col, 0);
    EXPECT_EQ(corner.cache_row, 2);
    EXPECT_EQ(corner.cache_col, 2);
    EXPECT_EQ(corner.rows, 3);
    EXPECT_EQ(corner.cols, 3);

    const auto interior = mapmaking::jinc_square_crop(9, 9, 4, 4, 5, 5);
    EXPECT_EQ(interior.map_row, 2);
    EXPECT_EQ(interior.map_col, 2);
    EXPECT_EQ(interior.cache_row, 0);
    EXPECT_EQ(interior.cache_col, 0);
    EXPECT_EQ(interior.rows, 5);
    EXPECT_EQ(interior.cols, 5);
    // A 5x5 square has 25 admitted cells; a radius-2 disk would have only 13.
    EXPECT_EQ(interior.rows * interior.cols, 25);
    EXPECT_THROW(mapmaking::jinc_square_crop(5, 5, 0, 0, 4, 5),
                 std::invalid_argument);
}

TEST(jinc_map_contract, selected_array_parameter_admission_is_fail_closed) {
    mapmaking::JincResolvedArrayParameters valid{
        0, "a1100", 1.1, 1.67, 2.0, 3.0, 1e-6, 2e-5, 60, 121,
        121};
    EXPECT_NO_THROW(mapmaking::validate_jinc_resolved_array(valid));
    for (int field = 0; field < 7; ++field) {
        auto invalid = valid;
        double *values[] = {&invalid.a, &invalid.b, &invalid.c,
                            &invalid.r_max, &invalid.pixel_size_rad,
                            &invalid.array_scale_rad};
        if (field < 6) {
            *values[field] = field % 2 == 0
                ? 0.0
                : std::numeric_limits<double>::quiet_NaN();
        }
        else {
            invalid.cache_rows = 0;
        }
        EXPECT_THROW(mapmaking::validate_jinc_resolved_array(invalid),
                     std::invalid_argument);
    }
    auto wrong_name = valid;
    wrong_name.array_name = "a1400";
    EXPECT_THROW(mapmaking::validate_jinc_resolved_array(wrong_name),
                 std::invalid_argument);

    citlali::config::MapmakingConfig config;
    config.jinc_filter.shape_params["a1100"][0] = 0.0;
    citlali::config::ValidationReport report;
    citlali::config::validate(config, report);
    EXPECT_TRUE(report.ok());
    config.method = citlali::config::MapMethod::jinc;
    report = {};
    citlali::config::validate(config, report);
    EXPECT_FALSE(report.ok());

    mapmaking::JincMapmaker maker;
    maker.r_max = 3.0;
    maker.subpixel_n = 1;
    maker.array_names = {{0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    maker.shape_params = {
        {0, Eigen::Vector3d(1.1, 1.67, 2.0)},
        {1, Eigen::Vector3d(1.1, 2.17, 2.0)},
        {2, Eigen::Vector3d(1.1, 3.17, 2.0)},
    };
    EXPECT_THROW(maker.allocate_jinc_matrix(0.0), std::invalid_argument);
    maker.r_max = std::numeric_limits<double>::infinity();
    EXPECT_THROW(maker.allocate_jinc_matrix(1e-6), std::invalid_argument);
    maker.r_max = 3.0;
    maker.subpixel_n = 0;
    EXPECT_THROW(maker.allocate_jinc_matrix(1e-6), std::invalid_argument);
    maker.subpixel_n = 1;
    maker.array_names.erase(2);
    EXPECT_THROW(maker.allocate_jinc_matrix(1e-6), std::invalid_argument);
}

TEST(jinc_map_contract,
     nonfinite_coefficient_generation_preserves_last_admitted_cache) {
    mapmaking::JincMapmaker maker;
    maker.r_max = 3.0;
    maker.subpixel_n = 1;
    maker.array_names = {{0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    maker.shape_params = {
        {0, Eigen::Vector3d(1.1, 1.67, 2.0)},
        {1, Eigen::Vector3d(1.1, 2.17, 2.0)},
        {2, Eigen::Vector3d(1.1, 3.17, 2.0)},
    };
    const double pixel_size = ((1.1 / 1000.0) / 45.0) / 2.0;
    maker.allocate_jinc_matrix(pixel_size);
    const auto admitted_digest =
        mapmaking::jinc_matrix_digest(maker.jinc_weights_mat.at(0));
    maker.subpixel_n = 2;
    maker.r_max = std::numeric_limits<double>::min();
    maker.shape_params.at(0)(0) = std::numeric_limits<double>::max();
    EXPECT_ANY_THROW(maker.allocate_jinc_matrix(
        std::numeric_limits<double>::min()));
    EXPECT_EQ(mapmaking::jinc_matrix_digest(maker.jinc_weights_mat.at(0)),
              admitted_digest);
}

TEST(jinc_map_contract, empirical_policy_can_downgrade_but_not_promote) {
    EXPECT_TRUE(mapmaking::jinc_empirical_support(true, true));
    EXPECT_FALSE(mapmaking::jinc_empirical_support(true, false));
    EXPECT_FALSE(mapmaking::jinc_empirical_support(false, true));
    EXPECT_FALSE(mapmaking::jinc_empirical_support(false, false));
}

TEST(jinc_map_contract, finalization_owns_formal_support_coverage_and_kernel) {
    mapmaking::MapBuffer buffer{"omb"};
    buffer.n_rows = 1;
    buffer.n_cols = 4;
    buffer.n_noise = 0;
    buffer.signal = {Eigen::MatrixXd(1, 4)};
    buffer.grid_weight = {Eigen::MatrixXd(1, 4)};
    buffer.weight = {Eigen::MatrixXd(1, 4)};
    buffer.coverage = {Eigen::MatrixXd(1, 4)};
    buffer.kernel = {Eigen::MatrixXd(1, 4)};
    buffer.signal[0] << 4.0, 1.0, 1.0, 1.0;
    buffer.grid_weight[0] << 2.0, 0.0, 1e-15, 1.0;
    buffer.weight[0] << 2.0, 2.0, 2.0, 0.0;
    buffer.coverage[0] << 7.5, 8.0, 9.0, 10.0;
    buffer.kernel[0] << -3.0, 4.0, 5.0, 6.0;
    buffer.jinc_products.allocate(1, 1, 4);
    buffer.jinc_products.denominator_sum_abs[0] << 2.0, 2.0, 2.0, 1.0;
    buffer.jinc_products.contributor_count[0] << 2, 2, 100, 1;

    buffer.normalize_maps();

    EXPECT_DOUBLE_EQ(buffer.signal[0](0, 0), 2.0);
    EXPECT_DOUBLE_EQ(buffer.weight[0](0, 0), 2.0);
    EXPECT_DOUBLE_EQ(buffer.coverage[0](0, 0), 7.5);
    EXPECT_DOUBLE_EQ(buffer.kernel[0](0, 0), -1.5);
    EXPECT_EQ(buffer.jinc_products.formal_support[0](0, 0), 1);
    for (Eigen::Index col = 1; col < 4; ++col) {
        EXPECT_DOUBLE_EQ(buffer.signal[0](0, col), 0.0);
        EXPECT_DOUBLE_EQ(buffer.weight[0](0, col), 0.0);
        EXPECT_DOUBLE_EQ(buffer.coverage[0](0, col), 0.0);
        EXPECT_DOUBLE_EQ(buffer.kernel[0](0, col), 0.0);
        EXPECT_EQ(buffer.jinc_products.formal_support[0](0, col), 0);
    }
    EXPECT_TRUE(buffer.grid_weight.empty());
    const auto &summary = buffer.jinc_products.provenance.realized;
    EXPECT_EQ(summary.formally_supported_pixel_count, 1U);
    EXPECT_EQ(summary.exact_cancellation_pixel_count, 1U);
    EXPECT_EQ(summary.unresolved_cancellation_pixel_count, 1U);
    EXPECT_EQ(summary.invalid_q_pixel_count, 1U);
}

TEST(jinc_map_contract,
     beammap_reset_clears_atomic_jinc_iteration_state_for_active_subset) {
    mapmaking::MapBuffer buffer{"omb"};
    buffer.n_rows = 1;
    buffer.n_cols = 2;
    citlali::pipeline::allocate_map_matrices(
        buffer, 2, true, false, true, false, "", true);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        buffer.signal[slot].setOnes();
        buffer.weight[slot].setOnes();
        buffer.grid_weight[slot].setOnes();
        buffer.coverage[slot].setOnes();
        buffer.jinc_products.denominator_sum_abs[slot].setOnes();
        buffer.jinc_products.contributor_count[slot].setOnes();
        buffer.jinc_products.formal_support[slot].setOnes();
    }
    Eigen::Matrix<bool, Eigen::Dynamic, 1> active(2);
    active << true, false;
    struct PtcNoiseStub {
        struct {
            Eigen::MatrixXi data;
        } noise;
    };
    std::vector<PtcNoiseStub> ptcs;
    std::uniform_int_distribution<int> bits(0, 1);
    std::mt19937 generator(42);
    citlali::pipeline::reset_beammap_mapmaking_buffers(
        buffer, ptcs, 2, false, false, false, 0, &active, bits,
        generator);

    EXPECT_TRUE(buffer.signal[0].isZero());
    EXPECT_TRUE(buffer.weight[0].isZero());
    EXPECT_TRUE(buffer.grid_weight[0].isZero());
    EXPECT_TRUE(buffer.coverage[0].isZero());
    EXPECT_TRUE(
        buffer.jinc_products.denominator_sum_abs[0].isZero());
    EXPECT_EQ(buffer.jinc_products.contributor_count[0].sum(), 0U);
    EXPECT_EQ(buffer.jinc_products.formal_support[0].sum(), 0U);
    EXPECT_DOUBLE_EQ(buffer.signal[1].sum(), 2.0);
    EXPECT_DOUBLE_EQ(
        buffer.jinc_products.denominator_sum_abs[1].sum(), 2.0);
    EXPECT_EQ(buffer.jinc_products.contributor_count[1].sum(), 2U);
}

TEST(jinc_map_contract,
     active_subset_finalization_retains_coherent_observation_summaries) {
    mapmaking::MapBuffer buffer{"omb"};
    buffer.n_rows = 1;
    buffer.n_cols = 1;
    citlali::pipeline::allocate_map_matrices(
        buffer, 2, true, false, true, false, "", true);
    const auto set_raw = [&](std::size_t slot, double numerator,
                             std::uint64_t count) {
        buffer.signal[slot](0, 0) = numerator;
        buffer.grid_weight[slot](0, 0) = 1.0;
        buffer.weight[slot](0, 0) = 1.0;
        buffer.coverage[slot](0, 0) = 0.25;
        buffer.jinc_products.denominator_sum_abs[slot](0, 0) = 1.0;
        buffer.jinc_products.contributor_count[slot](0, 0) = count;
    };
    set_raw(0, 2.0, 2);
    set_raw(1, 3.0, 3);
    Eigen::Matrix<bool, Eigen::Dynamic, 1> active(2);
    active << true, false;
    buffer.normalize_maps(&active);
    auto &summary = buffer.jinc_products.provenance.realized;
    ASSERT_EQ(summary.map_summaries.size(), 2U);
    EXPECT_EQ(summary.realization_pass_count, 1U);
    EXPECT_EQ(summary.realized_map_count, 1U);
    EXPECT_EQ(summary.last_pass_active_map_indices,
              std::vector<std::size_t>({0}));
    EXPECT_EQ(summary.contributor_count_max, 2U);

    buffer.grid_weight.assign(2, Eigen::MatrixXd::Zero(1, 1));
    active << false, true;
    set_raw(1, 3.0, 3);
    buffer.normalize_maps(&active);
    EXPECT_EQ(summary.realization_pass_count, 2U);
    EXPECT_EQ(summary.realized_map_count, 2U);
    EXPECT_EQ(summary.last_pass_active_map_indices,
              std::vector<std::size_t>({1}));
    EXPECT_EQ(summary.total_pixel_count, 2U);
    EXPECT_EQ(summary.formally_supported_pixel_count, 2U);
    EXPECT_EQ(summary.contributor_count_max, 3U);
    EXPECT_EQ(summary.map_summaries[0].realization_pass, 1U);
    EXPECT_EQ(summary.map_summaries[1].realization_pass, 2U);
}

TEST(jinc_map_contract, valid_two_level_sums_agree_under_declared_policy) {
    const std::array<double, 4> n_terms{2.0, -0.5, 4.0, 0.5};
    const std::array<double, 4> c_terms{1.0, -0.25, 2.0, 0.25};
    const std::array<double, 4> q_terms{1.0, 0.125, 2.0, 0.125};
    const auto sum = [](const auto &values) {
        return std::accumulate(values.begin(), values.end(), 0.0);
    };
    const double sequential_n = sum(n_terms);
    const double sequential_c = sum(c_terms);
    const double sequential_q = sum(q_terms);
    const double split_n = (n_terms[0] + n_terms[1]) +
                           (n_terms[2] + n_terms[3]);
    const double split_c = (c_terms[0] + c_terms[1]) +
                           (c_terms[2] + c_terms[3]);
    const double split_q = (q_terms[0] + q_terms[1]) +
                           (q_terms[2] + q_terms[3]);
    EXPECT_DOUBLE_EQ(sequential_n, split_n);
    EXPECT_DOUBLE_EQ(sequential_c, split_c);
    EXPECT_DOUBLE_EQ(sequential_q, split_q);
    const auto sequential = mapmaking::finalize_jinc_accumulators(
        sequential_n, sequential_c, sequential_q, 3.5, 4);
    const auto split = mapmaking::finalize_jinc_accumulators(
        split_n, split_c, split_q, 3.5, 4);
    EXPECT_TRUE(sequential.formal_support);
    EXPECT_DOUBLE_EQ(sequential.signal, split.signal);
    EXPECT_DOUBLE_EQ(sequential.formal_weight, split.formal_weight);
    EXPECT_EQ(sequential.rho_resolution_bound,
              mapmaking::jinc_rho_resolution_bound(4));
}

TEST(jinc_map_contract,
     production_population_paths_agree_for_maps_masks_coverage_and_kernel) {
    using PtcData =
        timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>;
    using Apt = std::map<std::string, Eigen::VectorXd>;
    constexpr Eigen::Index samples = 3;
    constexpr double sample_rate_hz = 4.0;
    const double pixel_size_rad = ((1.1 / 1000.0) / 45.0) / 2.0;

    auto make_data = [&]() {
        PtcData data;
        data.scans.data.resize(samples, 1);
        data.scans.data << 2.0, -1.0, 4.0;
        data.kernel.data.resize(samples, 1);
        data.kernel.data << 1.0, 0.5, -0.25;
        data.flags.data =
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
                samples, 1, false);
        data.weights.data = Eigen::VectorXd::Constant(1, 2.0);
        data.index.data = 9;
        data.tel_data.data["TelElAct"] = Eigen::VectorXd::Zero(samples);
        data.tel_data.data["alt_phys"] = Eigen::VectorXd::Zero(samples);
        data.tel_data.data["az_phys"] = Eigen::VectorXd::Zero(samples);
        data.pointing_offsets_arcsec.data["az"] =
            Eigen::VectorXd::Zero(samples);
        data.pointing_offsets_arcsec.data["alt"] =
            Eigen::VectorXd::Zero(samples);
        return data;
    };
    Apt apt;
    apt["array"] = Eigen::VectorXd::Zero(1);
    apt["flag"] = Eigen::VectorXd::Zero(1);
    apt["x_t"] = Eigen::VectorXd::Zero(1);
    apt["y_t"] = Eigen::VectorXd::Zero(1);
    apt["uid"] = Eigen::VectorXd::Constant(1, 101.0);
    Eigen::VectorXi map_indices = Eigen::VectorXi::Zero(1);
    std::string pixel_axes = "altaz";

    mapmaking::JincMapmaker maker;
    maker.r_max = 3.0;
    maker.subpixel_n = 1;
    maker.array_names = {{0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    maker.shape_params = {
        {0, Eigen::Vector3d(1.1, 1.67, 2.0)},
        {1, Eigen::Vector3d(1.1, 2.17, 2.0)},
        {2, Eigen::Vector3d(1.1, 3.17, 2.0)},
    };
    maker.allocate_jinc_matrix(pixel_size_rad);

    auto make_map = [&](const std::string &policy) {
        mapmaking::MapBuffer map{"omb"};
        map.n_rows = 15;
        map.n_cols = 15;
        map.pixel_size_rad = pixel_size_rad;
        map.map_grouping = "array";
        map.parallel_policy = policy;
        citlali::pipeline::allocate_map_matrices(
            map, 1, true, true, true, false, "", true);
        return map;
    };
    auto sequential = make_map("seq");
    auto concurrent = make_map("omp");
    mapmaking::MapBuffer no_coadd{"cmb"};
    auto sequential_data = make_data();
    auto concurrent_data = make_data();
    maker.populate_maps_jinc(
        sequential_data, sequential, no_coadd, map_indices, pixel_axes, apt,
        sample_rate_hz, true, false);
    maker.populate_maps_jinc_parallel(
        concurrent_data, concurrent, no_coadd, map_indices, pixel_axes, apt,
        sample_rate_hz, true, false);

    EXPECT_TRUE(sequential.signal[0].isApprox(concurrent.signal[0], 0.0));
    EXPECT_TRUE(sequential.grid_weight[0].isApprox(
        concurrent.grid_weight[0], 0.0));
    EXPECT_TRUE(sequential.weight[0].isApprox(concurrent.weight[0], 0.0));
    EXPECT_TRUE(sequential.coverage[0].isApprox(concurrent.coverage[0], 0.0));
    EXPECT_TRUE(sequential.kernel[0].isApprox(concurrent.kernel[0], 0.0));
    EXPECT_TRUE(sequential.jinc_products.denominator_sum_abs[0].isApprox(
        concurrent.jinc_products.denominator_sum_abs[0], 0.0));
    EXPECT_EQ(sequential.jinc_products.contributor_count[0],
              concurrent.jinc_products.contributor_count[0]);
    sequential.normalize_maps();
    concurrent.normalize_maps();
    EXPECT_TRUE(sequential.signal[0].isApprox(concurrent.signal[0], 0.0));
    EXPECT_TRUE(sequential.weight[0].isApprox(concurrent.weight[0], 0.0));
    EXPECT_TRUE(sequential.coverage[0].isApprox(concurrent.coverage[0], 0.0));
    EXPECT_TRUE(sequential.kernel[0].isApprox(concurrent.kernel[0], 0.0));
    EXPECT_EQ(sequential.jinc_products.formal_support[0],
              concurrent.jinc_products.formal_support[0]);
    EXPECT_DOUBLE_EQ(sequential.coverage[0](7, 7), 0.75);
    EXPECT_NEAR(sequential.signal[0](7, 7), 5.0 / 3.0, 1e-15);
    EXPECT_NEAR(sequential.kernel[0](7, 7), 2.5 / 6.0, 1e-15);
}

TEST(jinc_map_contract,
     actual_kernel_template_identity_tracks_loaded_and_source_center_state) {
    struct KernelStub {
        std::string type = "image";
        std::string filepath = "kernel.fits";
        double fwhm_rad = 1.0;
        double sigma_rad = 2.0;
        double sigma_limit = 3.0;
        std::string map_grouping = "detector";
        std::vector<std::string> img_ext_names{"KERNEL"};
        std::vector<Eigen::MatrixXd> images{
            Eigen::MatrixXd::Ones(2, 2)};
        Eigen::VectorXd source_lat = Eigen::VectorXd::Zero(1);
        Eigen::VectorXd source_lon = Eigen::VectorXd::Zero(1);
        Eigen::VectorXd source_a_fwhm_rad = Eigen::VectorXd::Ones(1);
        Eigen::VectorXd source_b_fwhm_rad = Eigen::VectorXd::Ones(1);
        Eigen::VectorXi source_valid = Eigen::VectorXi::Ones(1);
    } kernel;

    const auto initial =
        mapmaking::jinc_kernel_template_identity(kernel, true);
    kernel.source_lat(0) = 0.25;
    const auto moved =
        mapmaking::jinc_kernel_template_identity(kernel, true);
    EXPECT_NE(initial, moved);
    kernel.images[0](0, 0) = 2.0;
    EXPECT_NE(moved,
              mapmaking::jinc_kernel_template_identity(kernel, true));
    EXPECT_NE(initial,
              mapmaking::jinc_kernel_template_identity(kernel, false));
}

TEST(jinc_map_contract, compact_forward_only_provenance_serializes_joins) {
    citlali::config::MapmakingConfig request;
    request.method = citlali::config::MapMethod::jinc;
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(request, citlali::config::ReductionType::science);
    plan.begin_iteration();
    plan.begin_observation(0, "152390", 1, 1e-6, 1);

    mapmaking::JincObservationProvenance provenance;
    provenance.available = true;
    provenance.requested_digest = mapmaking::jinc_filter_config_digest(
        request.jinc_filter);
    provenance.effective_digest = provenance.requested_digest;
    provenance.requested_r_max = request.jinc_filter.r_max;
    provenance.effective_r_max = request.jinc_filter.r_max;
    provenance.requested_subpixel_n = request.jinc_filter.subpixel_n;
    provenance.effective_subpixel_n = request.jinc_filter.subpixel_n;
    provenance.kernel_template_identity =
        mapmaking::jinc_realization_identity_digest(
            "actual-upstream-kernel-template-v1",
            {{"type", "gaussian"}, {"enabled", "true"}});
    provenance.processing_configuration_identity =
        mapmaking::jinc_realization_identity_digest(
            "actual-enabled-processing-operators-v1",
            {{"temporal_fir_enabled", "true"},
             {"ptc_clean_enabled", "false"}});
    provenance.processing_realization_identity =
        mapmaking::jinc_processing_realization_identity(
            provenance.processing_configuration_identity, true, 2, 1);
    EXPECT_NE(
        provenance.processing_realization_identity,
        mapmaking::jinc_processing_realization_identity(
            provenance.processing_configuration_identity, true, 2, 2));
    provenance.coverage_sample_frequency_identity =
        "effective-processed-timestream-sample-rate-telescope-d_fsmp-v1";
    provenance.coverage_sample_frequency_hz = 4.0;
    provenance.realized.map_count = 1;
    provenance.realized.realized_map_count = 1;
    provenance.realized.realization_pass_count = 2;
    provenance.realized.last_pass_active_map_indices = {0};
    provenance.realized.total_pixel_count = 4;
    provenance.realized.formally_supported_pixel_count = 3;
    provenance.realized.contributor_count_max = 7;
    provenance.realized.rho_resolution_bound_max =
        mapmaking::jinc_rho_resolution_bound(7);
    provenance.realized.map_summaries = {
        mapmaking::JincMapRealizedSummary{
            true, 2, 4, 3, 0, 0, 1, 0, 7,
            mapmaking::jinc_rho_resolution_bound(7)}};
    mapmaking::record_jinc_product_join(
        provenance,
        {"jinc-finalized-signal-N-over-C", "raw_map_slot_0",
         "map.fits", "signal_I", "sha256:signal"});
    plan.record_observation_jinc_state(provenance);

    const auto node = citlali::pipeline::mapmaking_provenance_node(plan);
    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-mapmaking-provenance-v3");
    const auto state = node["observations"][0]["jinc_state"];
    EXPECT_TRUE(state["available"].as<bool>());
    EXPECT_EQ(
        state["realized"]["processing_configuration_identity"]
            .as<std::string>(),
        provenance.processing_configuration_identity);
    EXPECT_EQ(
        state["realized"]["processing_realization_identity"]
            .as<std::string>(),
        provenance.processing_realization_identity);
    EXPECT_EQ(state["realized"]["summation_method"].as<std::string>(),
              mapmaking::jinc_summation_identity);
    EXPECT_EQ(state["realized"]["coverage_sample_frequency_hz"].as<double>(),
              4.0);
    EXPECT_EQ(state["realized"]["realization_pass_count"].as<std::size_t>(),
              2U);
    EXPECT_EQ(state["realized"]["last_pass_active_map_indices"][0]
                  .as<std::size_t>(),
              0U);
    EXPECT_EQ(state["realized"]["map_summaries"][0]["realization_pass"]
                  .as<std::size_t>(),
              2U);
    EXPECT_EQ(state["realized"]["product_joins"].size(), 1U);
    EXPECT_EQ(state["realized"]["product_joins"][0]["hdu_name"].as<std::string>(),
              "signal_I");
    EXPECT_THROW(
        mapmaking::record_jinc_product_join(
            provenance,
            {"changed", "raw_map_slot_0", "map.fits", "signal_I",
             "sha256:changed"}),
        std::logic_error);

    citlali::pipeline::MapmakingExecutionPlan failed_plan;
    failed_plan.reset_from_request(
        request, citlali::config::ReductionType::science);
    failed_plan.begin_iteration();
    failed_plan.begin_observation(0, "152391", 1, 1e-6, 1);
    auto failed_provenance = provenance;
    failed_provenance.realized.product_joins.clear();
    EXPECT_THROW(
        failed_plan.record_observation_jinc_state(failed_provenance),
        std::logic_error);
    EXPECT_FALSE(failed_plan.observations.back().jinc_state.has_value());
}

}  // namespace

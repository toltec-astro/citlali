#include <gtest/gtest.h>

#include <citlali/core/engine/engine.h>
#include <citlali/core/mapmaking/map.h>
#include <citlali/core/pipeline/map_image_output_helpers.h>
#include <citlali/core/pipeline/science_map_provenance_serialization.h>
#include <citlali/core/utils/fits_io.h>

#include <fitsio.h>
#include <spdlog/sinks/null_sink.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct CapturedHdu {
    std::map<std::string, std::string> keys;

    template <class Value>
    void addKey(const std::string &name, const Value &value,
                const std::string &, bool = false) {
        std::ostringstream stream;
        stream << value;
        keys[name] = stream.str();
    }
};

struct CapturedImage {
    std::string name;
    std::string data_type;
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;
};

struct CapturedFitsEntry {
    std::string filepath = "captured-science-map-products";
    std::vector<std::shared_ptr<CapturedHdu>> hdus;
    std::vector<CapturedImage> images;

    template <class Derived>
    void add_hdu(const std::string &name,
                 const Eigen::DenseBase<Derived> &data) {
        using Scalar = std::remove_cv_t<typename Derived::Scalar>;
        std::string data_type;
        if constexpr (std::is_same_v<Scalar, double>) {
            data_type = "float64";
        }
        else if constexpr (std::is_same_v<Scalar, std::int64_t>) {
            data_type = "int64";
        }
        else if constexpr (std::is_same_v<Scalar, std::uint8_t>) {
            data_type = "uint8";
        }
        else {
            data_type = "unexpected";
        }
        images.push_back({name, data_type, data.rows(), data.cols()});
        hdus.push_back(std::make_shared<CapturedHdu>());
    }

    template <class Hdu, class Wcs>
    void add_wcs(const Hdu &, const Wcs &, double) {}
};

struct DummyWcs {};

using ScienceMapBufferFixture = mapmaking::MapBuffer;

std::shared_ptr<spdlog::logger> science_map_test_logger() {
    static const auto logger = [] {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        return std::make_shared<spdlog::logger>("science-map-fits-test", sink);
    }();
    return logger;
}

std::shared_ptr<ScienceMapBufferFixture> make_science_map_buffer(
    bool coadd = true) {
    auto map = std::make_shared<ScienceMapBufferFixture>(
        coadd ? "cmb" : "omb");
    map->n_rows = 2;
    map->n_cols = 2;
    map->n_noise = 0;
    map->sig_unit = "mJy/beam";
    map->map_grouping = "array";
    map->cov_cut = 1.0;
    map->science_products.allocate(1, 2, 2, coadd, true, true);
    auto &products = map->science_products;
    auto &realized = products.realized[0];
    mapmaking::ScienceMapBundleIdentity identity;
    identity.grouping = map->map_grouping;
    identity.signal_unit = map->sig_unit;
    identity.estimator_identity =
        mapmaking::science_map_coadd_estimator_version;
    identity.response_identity = "identity-response";
    identity.rows = map->n_rows;
    identity.cols = map->n_cols;
    identity.wcs.coordinate_frame = "equatorial-j2000";
    identity.wcs.projection = "TAN";
    identity.wcs.axis_types = {"RA---TAN", "DEC--TAN"};
    identity.wcs.axis_units = {"deg", "deg"};
    identity.wcs.pixel_scale = {-1.0 / 3600.0, 1.0 / 3600.0};
    identity.wcs.reference_world = {123.25, -45.5};
    identity.wcs.reference_pixel = {0.5, 0.5};
    identity.wcs.source_epoch = 2000.0;
    mapmaking::ScienceMapSlotIdentity slot;
    slot.grouping = "array";
    slot.group_identity = "array:0";
    slot.array_identity = 0;
    slot.frequency_hz = 150.0e9;
    identity.ordered_slots.push_back(slot);
    products.bundle_identity = identity;
    products.identity_admitted = true;
    realized.normalization.support_algorithm =
        mapmaking::science_map_normalization_support_version;
    realized.normalization.coefficient_stage =
        products.is_coadd
            ? mapmaking::science_map_coadd_normalization_coefficient_stage
            : mapmaking::science_map_observation_normalization_coefficient_stage;
    realized.science_policy.support_algorithm =
        mapmaking::science_map_policy_support_version;
    realized.science_policy.coefficient_stage = products.coefficient_stage;
    realized.normalization.requested_cut = 0.1;
    realized.normalization.realized_cut = 0.1;
    realized.normalization.realized_threshold = 0.3;
    realized.normalization.selected_positive_value = 3.0;
    realized.normalization.positive_value_count = 3;
    realized.normalization.selected_zero_based_index = 2;
    realized.normalization.selected_index_available = true;
    realized.science_policy.requested_cut = 1.0;
    realized.science_policy.realized_cut = 1.0;
    realized.science_policy.realized_threshold = 3.0;
    realized.science_policy.selected_positive_value = 3.0;
    realized.science_policy.positive_value_count = 3;
    realized.science_policy.selected_zero_based_index = 2;
    realized.science_policy.selected_index_available = true;

    products.geometric_hits[0] << 1, 2, 3, 4;
    products.contributing_hits[0] << 0, 1, 2, 3;
    products.coadd_observation_count[0] << 0, 0, 0, 0;
    products.upstream_eligible_exposure[0] << 0.5, 1.0, 1.5, 2.0;
    products.retained_exposure[0] << 0.0, 1.0, 1.5, 2.0;
    products.normalization_support[0] << 0, 1, 1, 1;
    products.science_policy_support[0] << 0, 0, 1, 1;
    products.science_valid[0] << 0, 0, 1, 1;
    map->signal = {Eigen::MatrixXd::Ones(2, 2)};
    Eigen::MatrixXd weight(2, 2);
    weight << 0.0, 1.0, 3.0, 3.0;
    map->weight = {weight};
    map->coverage = {products.retained_exposure[0]};
    map->median_err = Eigen::VectorXd::Constant(1, 3.0);
    map->median_rms = Eigen::VectorXd::Constant(1, 4.0);
    map->wcs.ctype = identity.wcs.axis_types;
    map->wcs.cunit = identity.wcs.axis_units;
    map->wcs.crval = {123.25F, -45.5F};
    map->wcs.cdelt = {-1.0F / 3600.0F, 1.0F / 3600.0F};
    map->wcs.crpix = {0.5F, 0.5F};
    map->wcs.naxis = {2, 2};
    mapmaking::science_map_finalize_realized_product_facts(*map, 0);
    return map;
}

const CapturedHdu &captured_hdu(const CapturedFitsEntry &entry,
                                const std::string &name) {
    for (std::size_t i = 0; i < entry.images.size(); ++i) {
        if (entry.images[i].name == name) {
            return *entry.hdus[i];
        }
    }
    throw std::runtime_error("missing captured HDU " + name);
}

const CapturedImage &captured_image(const CapturedFitsEntry &entry,
                                    const std::string &name) {
    for (const auto &image : entry.images) {
        if (image.name == name) {
            return image;
        }
    }
    throw std::runtime_error("missing captured image " + name);
}

bool captured_has_image(const CapturedFitsEntry &entry,
                        const std::string &name) {
    for (const auto &image : entry.images) {
        if (image.name == name) {
            return true;
        }
    }
    return false;
}

void set_noise_stack(
    mapmaking::MapBuffer &map,
    const std::vector<Eigen::MatrixXd> &realizations) {
    ASSERT_FALSE(realizations.empty());
    map.n_noise = static_cast<Eigen::Index>(realizations.size());
    map.noise.clear();
    map.noise.emplace_back(map.n_rows, map.n_cols, map.n_noise);
    for (Eigen::Index realization = 0; realization < map.n_noise;
         ++realization) {
        ASSERT_EQ(realizations[static_cast<std::size_t>(realization)].rows(),
                  map.n_rows);
        ASSERT_EQ(realizations[static_cast<std::size_t>(realization)].cols(),
                  map.n_cols);
        for (Eigen::Index row = 0; row < map.n_rows; ++row) {
            for (Eigen::Index col = 0; col < map.n_cols; ++col) {
                map.noise[0](row, col, realization) =
                    realizations[static_cast<std::size_t>(realization)](
                        row, col);
            }
        }
    }
}

std::shared_ptr<mapmaking::MapBuffer> make_noise_product_fixture(
    const Eigen::MatrixXd &signal, const Eigen::MatrixXd &weight,
    const std::vector<Eigen::MatrixXd> &realizations) {
    auto map = std::make_shared<mapmaking::MapBuffer>("noise-fixture");
    map->n_rows = signal.rows();
    map->n_cols = signal.cols();
    map->cov_cut = 0.0;
    map->signal = {signal};
    map->weight = {weight};
    set_noise_stack(*map, realizations);
    return map;
}

TEST(science_map_fits_products,
     conditional_stack_scatter_R1_is_descriptive_but_not_uncertainty) {
    const Eigen::MatrixXd signal = Eigen::MatrixXd::Constant(1, 1, 2.0);
    const Eigen::MatrixXd weight = Eigen::MatrixXd::Ones(1, 1);
    auto map = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 5.0)});

    map->calc_noise_products(Eigen::Index{0}, false, true);

    EXPECT_DOUBLE_EQ(map->noise_mean[0](0, 0), 5.0);
    EXPECT_DOUBLE_EQ(map->noise_variance[0](0, 0), 0.0);
    EXPECT_EQ(map->noise_stack_scatter_valid(0), 1);
    EXPECT_EQ(map->noise_uncertainty_use_valid(0), 0);
    EXPECT_EQ(map->noise_weight_scale_valid(0), 0);
    EXPECT_TRUE(std::isnan(map->noise_weight_scale(0)));
    EXPECT_TRUE(std::isnan(map->sig2noise_pixel[0](0, 0)));
    EXPECT_DOUBLE_EQ(map->point_source_uncertainty[0](0, 0), 0.0);
    EXPECT_TRUE(std::isnan(map->sig2noise_point_source[0](0, 0)));

    auto required_scale = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 5.0)});
    EXPECT_THROW(required_scale->calc_noise_products(
                     Eigen::Index{0}, true, true),
                 std::runtime_error);

    auto uncentered = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 5.0)});
    EXPECT_THROW(uncentered->calc_noise_products(
                     Eigen::Index{0}, false, false),
                 std::invalid_argument);
}

TEST(science_map_fits_products,
     conditional_stack_scatter_R2_uses_completed_R_normalization) {
    const Eigen::MatrixXd signal = Eigen::MatrixXd::Constant(1, 1, 2.0);
    const Eigen::MatrixXd weight = Eigen::MatrixXd::Constant(1, 1, 0.25);
    auto map = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, -2.0),
         Eigen::MatrixXd::Constant(1, 1, 2.0)});

    map->calc_noise_products(Eigen::Index{0}, false, true);

    EXPECT_DOUBLE_EQ(map->noise_mean[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(map->noise_variance[0](0, 0), 4.0);
    EXPECT_EQ(map->noise_uncertainty_use_valid(0), 1);
    EXPECT_EQ(map->noise_weight_scale_valid(0), 1);
    EXPECT_DOUBLE_EQ(map->noise_weight_median_ratio(0), 1.0);
    EXPECT_DOUBLE_EQ(map->noise_weight_scale(0), 1.0);
    EXPECT_DOUBLE_EQ(map->weight_empirical[0](0, 0), 0.25);
    EXPECT_DOUBLE_EQ(map->sig2noise_pixel[0](0, 0), 1.0);
    EXPECT_DOUBLE_EQ(map->point_source_uncertainty[0](0, 0), 2.0);
    EXPECT_DOUBLE_EQ(map->sig2noise_point_source[0](0, 0), 1.0);

    auto existing_use_only = make_noise_product_fixture(
        signal, Eigen::MatrixXd::Constant(1, 1, 0.5),
        {Eigen::MatrixXd::Constant(1, 1, -2.0),
         Eigen::MatrixXd::Constant(1, 1, 2.0)});
    existing_use_only->calc_noise_products(Eigen::Index{0}, true, true);
    EXPECT_DOUBLE_EQ(existing_use_only->weight_formal[0](0, 0), 0.5);
    EXPECT_DOUBLE_EQ(existing_use_only->noise_weight_scale(0), 0.5);
    EXPECT_DOUBLE_EQ(existing_use_only->weight[0](0, 0), 0.25);
}

TEST(science_map_fits_products,
     duplicate_complementary_and_simple_R2_designs_are_exact) {
    const Eigen::MatrixXd signal = Eigen::MatrixXd::Ones(1, 1);
    const Eigen::MatrixXd weight = Eigen::MatrixXd::Ones(1, 1);

    auto duplicate = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 3.0),
         Eigen::MatrixXd::Constant(1, 1, 3.0)});
    duplicate->calc_noise_products(Eigen::Index{0}, false, true);
    EXPECT_DOUBLE_EQ(duplicate->noise_variance[0](0, 0), 0.0);
    EXPECT_EQ(duplicate->noise_weight_scale_valid(0), 0);
    EXPECT_TRUE(std::isnan(duplicate->sig2noise_point_source[0](0, 0)));

    auto complementary = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, -2.0),
         Eigen::MatrixXd::Constant(1, 1, 2.0)});
    complementary->calc_noise_products(Eigen::Index{0}, false, true);
    EXPECT_DOUBLE_EQ(complementary->noise_variance[0](0, 0), 4.0);

    auto simple = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 0.0),
         Eigen::MatrixXd::Constant(1, 1, 2.0)});
    simple->calc_noise_products(Eigen::Index{0}, false, true);
    EXPECT_DOUBLE_EQ(simple->noise_mean[0](0, 0), 1.0);
    EXPECT_DOUBLE_EQ(simple->noise_variance[0](0, 0), 1.0);
}

TEST(science_map_fits_products,
     empty_scale_calibration_region_is_unavailable_and_fails_closed) {
    const Eigen::MatrixXd signal = Eigen::MatrixXd::Ones(1, 1);
    const Eigen::MatrixXd zero_weight = Eigen::MatrixXd::Zero(1, 1);
    const std::vector<Eigen::MatrixXd> realizations{
        Eigen::MatrixXd::Constant(1, 1, -1.0),
        Eigen::MatrixXd::Constant(1, 1, 1.0)};
    auto diagnostic = make_noise_product_fixture(
        signal, zero_weight, realizations);

    diagnostic->calc_noise_products(Eigen::Index{0}, false, true);
    EXPECT_EQ(diagnostic->noise_valid_pixels(0), 0.0);
    EXPECT_EQ(diagnostic->noise_weight_scale_valid(0), 0);
    EXPECT_TRUE(std::isnan(diagnostic->noise_weight_median_ratio(0)));
    EXPECT_TRUE(std::isnan(diagnostic->noise_weight_scale(0)));
    EXPECT_TRUE(std::isnan(diagnostic->sig2noise_pixel[0](0, 0)));

    auto required_scale = make_noise_product_fixture(
        signal, zero_weight, realizations);
    EXPECT_THROW(required_scale->calc_noise_products(
                     Eigen::Index{0}, true, true),
                 std::runtime_error);
}

TEST(science_map_fits_products,
     fixed_projection_preserves_two_pixel_covariance_without_dense_matrix) {
    Eigen::MatrixXd signal = Eigen::MatrixXd::Zero(1, 2);
    Eigen::MatrixXd weight = Eigen::MatrixXd::Ones(1, 2);
    Eigen::MatrixXd plus = Eigen::MatrixXd::Ones(1, 2);
    Eigen::MatrixXd minus = -Eigen::MatrixXd::Ones(1, 2);
    auto map = make_noise_product_fixture(signal, weight, {plus, minus});
    map->calc_noise_products(Eigen::Index{0}, false, true);

    Eigen::MatrixXd aperture = Eigen::MatrixXd::Ones(1, 2);
    const double projected_scatter =
        map->calc_fixed_projection_stack_scatter(0, aperture);
    const double diagonal_only = map->noise_variance[0].sum();
    EXPECT_DOUBLE_EQ(diagonal_only, 2.0);
    EXPECT_DOUBLE_EQ(projected_scatter, 4.0);
    EXPECT_DOUBLE_EQ(
        map->calc_fixed_projection_stack_scatter(0, aperture, 2.0),
        1.0);

    Eigen::MatrixXd first(1, 2);
    first << 2.0, 0.0;
    Eigen::MatrixXd second(1, 2);
    second << 0.0, 2.0;
    auto template_map = make_noise_product_fixture(
        signal, weight, {first, second});
    Eigen::MatrixXd fixed_template(1, 2);
    fixed_template << 0.5, -0.5;
    EXPECT_DOUBLE_EQ(
        template_map->calc_fixed_projection_stack_scatter(
            0, fixed_template),
        1.0);
    EXPECT_THROW(
        template_map->calc_fixed_projection_stack_scatter(
            0, fixed_template, 0.0),
        std::invalid_argument);
}

TEST(science_map_fits_products,
     publishes_canonical_noise_identities_with_legacy_alias_joins) {
    auto map = make_science_map_buffer(false);
    set_noise_stack(
        *map,
        {Eigen::MatrixXd::Constant(2, 2, -2.0),
         Eigen::MatrixXd::Constant(2, 2, 2.0)});
    map->science_products.bundle_identity->required_companions = {
        "noise_realization_0_I", "noise_realization_1_I"};
    mapmaking::science_map_finalize_realized_product_facts(*map, 0);
    map->calc_noise_products(Eigen::Index{0}, false, true);
    ASSERT_EQ(map->noise_weight_scale_valid(0), 1);
    CapturedFitsEntry primary;
    CapturedFitsEntry raw_support;
    DummyWcs wcs;

    citlali::pipeline::add_primary_map_image_hdus(
        primary, map, 0, "", "I", wcs, 2000.0, false, true, false,
        false, science_map_test_logger());
    citlali::pipeline::add_coverage_support_image_hdus(
        raw_support, map, 0, "", "I", wcs, 2000.0, false, true, false,
        science_map_test_logger());

    ASSERT_TRUE(captured_has_image(
        primary, "conditional_stack_scatter_I"));
    ASSERT_TRUE(captured_has_image(primary, "noise_variance_I"));
    const auto &scatter =
        captured_hdu(primary, "conditional_stack_scatter_I").keys;
    EXPECT_EQ(scatter.at("ESTTYPE"),
              "conditional_finite_stack_scatter");
    EXPECT_EQ(scatter.at("NOIPKG"), "citlali-noise-products");
    EXPECT_EQ(scatter.at("NOIPROV"),
              "noise_products_provenance.yaml");
    EXPECT_EQ(scatter.at("NOIPRID"),
              "conditional_finite_stack_scatter");
    EXPECT_EQ(scatter.at("NOIPVER"), "SCI-NOI-002-v1");
    EXPECT_EQ(scatter.at("NOIDGST"),
              citlali::pipeline::noise_product_semantic_digest(
                  citlali::pipeline::
                      noise_conditional_stack_scatter_product_id));
    EXPECT_EQ(scatter.find("NOIRCOMP"), scatter.end());
    const auto &legacy_scatter =
        captured_hdu(primary, "noise_variance_I").keys;
    EXPECT_EQ(legacy_scatter.at("ALIASOF"),
              "conditional_stack_scatter_I");
    EXPECT_EQ(legacy_scatter.at("DEPRCATD"), "true");
    EXPECT_EQ(legacy_scatter.at("NOIPRID"), scatter.at("NOIPRID"));

    ASSERT_TRUE(captured_has_image(
        raw_support, "coefficient_standardized_signal_I"));
    const auto &standardized = captured_hdu(
        raw_support, "coefficient_standardized_signal_I").keys;
    EXPECT_EQ(standardized.at("ESTTYPE"),
              "coefficient_standardized_signal");
    EXPECT_EQ(standardized.at("SIGSTAT"), "not_significance");
    EXPECT_EQ(standardized.at("NOIPRID"),
              "coefficient_standardized_signal");
    EXPECT_EQ(captured_hdu(raw_support, "sig2noise_I").keys.at("ALIASOF"),
              "coefficient_standardized_signal_I");
    EXPECT_EQ(
        captured_hdu(raw_support, "sig2noise_pixel_I").keys.at("ALIASOF"),
        "coefficient_standardized_signal_I");

    map->freeze_raw_science_parent();
    CapturedFitsEntry filtered_support;
    citlali::pipeline::add_coverage_support_image_hdus(
        filtered_support, map, 0, "", "I", wcs, 2000.0, true, true,
        false, science_map_test_logger());

    ASSERT_TRUE(captured_has_image(
        filtered_support, "filtered_pixel_stack_scatter_I"));
    const auto &filtered_scatter = captured_hdu(
        filtered_support, "filtered_pixel_stack_scatter_I").keys;
    EXPECT_EQ(filtered_scatter.at("ESTTYPE"),
              "filtered_pixel_stack_scatter");
    EXPECT_EQ(filtered_scatter.at("NOIPRID"),
              "filtered_pixel_stack_scatter");
    EXPECT_NE(filtered_scatter.at("NOIRESTR").find(
                  "strict_parity_pending_FLT"),
              std::string::npos);
    EXPECT_EQ(
        captured_hdu(filtered_support, "point_source_uncertainty_I")
            .keys.at("ALIASOF"),
        "filtered_pixel_stack_scatter_I");
    ASSERT_TRUE(captured_has_image(
        filtered_support, "conditional_stack_scatter_ratio_I"));
    const auto &ratio = captured_hdu(
        filtered_support, "conditional_stack_scatter_ratio_I").keys;
    EXPECT_EQ(ratio.at("ESTTYPE"),
              "conditional_stack_scatter_ratio");
    EXPECT_EQ(ratio.at("SIGSTAT"), "not_significance");
    EXPECT_EQ(
        captured_hdu(filtered_support, "sig2noise_point_source_I")
            .keys.at("ALIASOF"),
        "conditional_stack_scatter_ratio_I");
}

TEST(science_map_fits_products, writes_canonical_typed_planes_and_aliases) {
    auto map = make_science_map_buffer();
    CapturedFitsEntry output;
    DummyWcs wcs;

    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0,
        science_map_test_logger());

    ASSERT_EQ(output.images.size(), 10U);
    EXPECT_EQ(captured_image(output, "geometric_hits_I").data_type, "int64");
    EXPECT_EQ(captured_image(output, "contributing_hits_I").data_type,
              "int64");
    EXPECT_EQ(captured_image(output, "coadd_observation_count_I").data_type,
              "int64");
    EXPECT_EQ(captured_image(output, "upstream_eligible_exposure_I").data_type,
              "float64");
    EXPECT_EQ(captured_image(output, "retained_exposure_I").data_type,
              "float64");
    EXPECT_EQ(captured_image(output, "normalization_support_I").data_type,
              "uint8");
    EXPECT_EQ(captured_image(output, "science_policy_support_I").data_type,
              "uint8");
    EXPECT_EQ(captured_image(output, "science_valid_I").data_type, "uint8");
    EXPECT_EQ(captured_image(output, "coverage_I").data_type, "float64");
    EXPECT_EQ(captured_image(output, "coverage_bool_I").data_type, "uint8");

    const auto &valid = captured_hdu(output, "science_valid_I").keys;
    EXPECT_EQ(valid.at("DATTYP"), "uint8");
    EXPECT_EQ(valid.at("VALAUTH"), "true");
    EXPECT_EQ(valid.at("ESTTYPE"), valid.at("TYPE"));

    const auto &coverage = captured_hdu(output, "coverage_I").keys;
    EXPECT_EQ(coverage.at("BUNIT"), "detector s");
    EXPECT_EQ(coverage.at("ALIASOF"), "retained_exposure_I");
    EXPECT_EQ(coverage.at("DEPRCATD"), "false");
    EXPECT_EQ(coverage.at("VALAUTH"), "false");

    const auto &coverage_bool =
        captured_hdu(output, "coverage_bool_I").keys;
    EXPECT_EQ(coverage_bool.at("ALIASOF"), "science_policy_support_I");
    EXPECT_EQ(coverage_bool.at("DEPRCATD"), "true");
    EXPECT_EQ(coverage_bool.at("VALAUTH"), "false");
    EXPECT_EQ(coverage_bool.at("WTTHRESH"), "3");
}

TEST(science_map_fits_products, skips_products_declared_unavailable) {
    auto map = make_science_map_buffer(false);
    CapturedFitsEntry output;
    DummyWcs wcs;

    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0,
        science_map_test_logger());

    EXPECT_TRUE(captured_has_image(output, "contributing_hits_I"));
    EXPECT_FALSE(captured_has_image(output, "coadd_observation_count_I"));
    EXPECT_TRUE(captured_has_image(output, "science_valid_I"));
}

TEST(science_map_fits_products,
     supported_output_bundle_rejects_all_unavailable_or_missing_inventory) {
    auto map = make_science_map_buffer(false);
    EXPECT_TRUE(citlali::pipeline::science_map_supported_output_bundle_complete(
        map->science_products, map->signal.size(), map->n_rows,
        map->n_cols));

    map->science_products.realized[0].product_available.fill(false);
    EXPECT_FALSE(citlali::pipeline::science_map_supported_output_bundle_complete(
        map->science_products, map->signal.size(), map->n_rows,
        map->n_cols));

    map = make_science_map_buffer(false);
    map->science_products.science_valid.clear();
    EXPECT_FALSE(citlali::pipeline::science_map_supported_output_bundle_complete(
        map->science_products, map->signal.size(), map->n_rows,
        map->n_cols));
}

TEST(science_map_fits_products,
     coadd_writes_f010_hierarchy_without_significance_products) {
    auto map = make_science_map_buffer();
    map->science_products.is_coadd = true;
    map->freeze_raw_science_parent();
    CapturedFitsEntry output;
    DummyWcs wcs;

    EXPECT_NO_THROW(citlali::pipeline::add_coverage_support_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0, true, true, true,
        science_map_test_logger()));

    EXPECT_TRUE(captured_has_image(output, "science_valid_I"));
    EXPECT_FALSE(captured_has_image(output, "formal_standardized_signal_I"));
    EXPECT_FALSE(captured_has_image(
        output, "conditional_stack_scatter_I"));
    EXPECT_FALSE(captured_has_image(
        output, "coefficient_standardized_signal_I"));
    EXPECT_FALSE(captured_has_image(output, "sig2noise_I"));
    EXPECT_FALSE(captured_has_image(output, "sig2noise_pixel_I"));
    EXPECT_FALSE(captured_has_image(output, "point_source_flux_I"));
    EXPECT_FALSE(captured_has_image(output, "point_source_uncertainty_I"));
    EXPECT_FALSE(captured_has_image(
        output, "filtered_pixel_stack_scatter_I"));
    EXPECT_FALSE(captured_has_image(
        output, "conditional_stack_scatter_ratio_I"));
    EXPECT_FALSE(captured_has_image(output, "sig2noise_point_source_I"));
}

TEST(science_map_fits_products,
     selected_frozen_coadd_authority_controls_output_family) {
    auto map = make_science_map_buffer();
    map->freeze_raw_science_parent();
    map->science_products.is_coadd = false;
    CapturedFitsEntry output;
    DummyWcs wcs;

    EXPECT_NO_THROW(citlali::pipeline::add_coverage_support_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0, true, false, true,
        science_map_test_logger()));

    EXPECT_TRUE(captured_has_image(output, "science_valid_I"));
    EXPECT_FALSE(captured_has_image(output, "formal_standardized_signal_I"));
    EXPECT_FALSE(captured_has_image(output, "sig2noise_I"));
}

TEST(science_map_fits_products,
     coadd_primary_weight_omits_uncertainty_metadata) {
    auto map = make_science_map_buffer();
    map->science_products.is_coadd = true;
    CapturedFitsEntry output;
    DummyWcs wcs;

    citlali::pipeline::add_primary_map_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0, false, false, false,
        true, science_map_test_logger());

    const auto &weight = captured_hdu(output, "weight_I").keys;
    EXPECT_EQ(weight.at("PRECSTAT"), "not_established");
    EXPECT_EQ(weight.at("COVSTAT"), "unavailable");
    EXPECT_EQ(weight.find("MEDERR"), weight.end());
}

TEST(science_map_fits_products, rejects_nonidentical_coverage_alias) {
    auto map = make_science_map_buffer();
    map->coverage[0](0, 0) = -0.0;
    map->science_products.retained_exposure[0](0, 0) = 0.0;
    CapturedFitsEntry output;
    DummyWcs wcs;

    EXPECT_THROW(
        citlali::pipeline::add_science_map_product_image_hdus(
            output, map, 0, "", "I", wcs, 2000.0,
            science_map_test_logger()),
        citlali::error::Error);
    EXPECT_TRUE(output.images.empty());
}

TEST(science_map_fits_products,
     filtered_output_carries_immutable_raw_parent_after_live_mutation) {
    auto map = make_science_map_buffer(false);
    map->freeze_raw_science_parent();
    ASSERT_TRUE(map->raw_science_parent);
    const auto raw_digest =
        map->raw_science_parent->realized[0].raw_parent_digest;
    const auto raw_valid = map->raw_science_parent->science_valid[0];

    map->signal[0].setConstant(42.0);
    map->weight[0].setConstant(17.0);
    map->science_products.science_valid[0].setZero();
    map->refresh_science_products_after_coefficient_rescale(0);

    EXPECT_EQ(map->raw_science_parent->realized[0].raw_parent_digest,
              raw_digest);
    EXPECT_TRUE(citlali::pipeline::science_map_planes_bitwise_equal(
        map->raw_science_parent->science_valid[0], raw_valid));

    CapturedFitsEntry output;
    DummyWcs wcs;
    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0,
        science_map_test_logger(), true);
    const auto &valid = captured_hdu(output, "science_valid_I").keys;
    EXPECT_EQ(valid.at("RAWSTATE"), "immutable_input");
    EXPECT_EQ(valid.at("RAWPDGST"), raw_digest);
    const auto &coverage = captured_hdu(output, "coverage_I").keys;
    EXPECT_EQ(coverage.at("RAWPDGST"), raw_digest);
}

TEST(science_map_fits_products,
     filtered_output_without_frozen_raw_parent_fails_before_write) {
    auto map = make_science_map_buffer(false);
    CapturedFitsEntry output;
    DummyWcs wcs;
    EXPECT_THROW(
        citlali::pipeline::add_science_map_product_image_hdus(
            output, map, 0, "", "I", wcs, 2000.0,
            science_map_test_logger(), true),
        citlali::error::Error);
    EXPECT_TRUE(output.images.empty());
}

TEST(science_map_fits_products,
     missing_empirical_companions_fail_before_any_primary_or_f010_write) {
    auto map = make_science_map_buffer(false);
    DummyWcs wcs;

    CapturedFitsEntry primary_output;
    EXPECT_THROW(
        citlali::pipeline::add_primary_map_image_hdus(
            primary_output, map, 0, "", "I", wcs, 2000.0, true, true,
            false, false, science_map_test_logger()),
        citlali::error::Error);
    EXPECT_TRUE(primary_output.images.empty());

    CapturedFitsEntry support_output;
    EXPECT_THROW(
        citlali::pipeline::add_coverage_support_image_hdus(
            support_output, map, 0, "", "I", wcs, 2000.0, false, true,
            false, science_map_test_logger()),
        citlali::error::Error);
    EXPECT_TRUE(support_output.images.empty());
}

TEST(science_map_fits_products,
     missing_median_diagnostic_fails_before_primary_write) {
    auto map = make_science_map_buffer(false);
    map->median_err.resize(0);
    CapturedFitsEntry output;
    DummyWcs wcs;
    EXPECT_THROW(
        citlali::pipeline::add_primary_map_image_hdus(
            output, map, 0, "", "I", wcs, 2000.0, false, false,
            false, false, science_map_test_logger()),
        citlali::error::Error);
    EXPECT_TRUE(output.images.empty());
}

TEST(science_map_fits_products,
     products_off_observation_can_prepare_and_publish_median_diagnostic) {
    auto map = make_science_map_buffer(false);
    map->median_err.resize(0);
    map->calc_median_err();
    ASSERT_EQ(map->median_err.size(), 1);
    ASSERT_TRUE(std::isfinite(map->median_err(0)));

    CapturedFitsEntry output;
    DummyWcs wcs;
    EXPECT_NO_THROW(citlali::pipeline::add_primary_map_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0, false, false, false,
        false, science_map_test_logger()));
    ASSERT_EQ(output.images.size(), 2U);
    const auto &weight = captured_hdu(output, "weight_I").keys;
    EXPECT_NE(weight.find("MEDERR"), weight.end());
}

TEST(science_map_fits_products, labels_weights_as_nonprecision_coefficients) {
    CapturedHdu hdu;
    citlali::pipeline::add_weight_map_metadata(
        hdu, "1/(mJy/beam)^2", false);

    EXPECT_EQ(hdu.keys.at("ESTTYPE"),
              "nonprecision_normalization_coefficient");
    EXPECT_EQ(hdu.keys.at("TYPE"), hdu.keys.at("ESTTYPE"));
    EXPECT_EQ(hdu.keys.at("PRECSTAT"), "not_established");
    EXPECT_EQ(hdu.keys.at("COVSTAT"), "unavailable");
    EXPECT_EQ(hdu.keys.at("CALTYPE"), "formal");
    EXPECT_EQ(hdu.keys.at("DESCRIP").find("inverse variance"),
              std::string::npos);
}

struct FitsFileCleanup {
    std::string path;
    ~FitsFileCleanup() { std::remove(path.c_str()); }
};

struct FitsDirectoryCleanup {
    std::filesystem::path path;
    ~FitsDirectoryCleanup() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void configure_production_writer_engine(Engine &engine) {
    engine.logger = science_map_test_logger();
    engine.typed_config.runtime.reduction_type =
        citlali::config::ReductionType::science;
    engine.runtime_config_provenance =
        citlali::config::make_runtime_config_provenance(
            engine.typed_config.runtime, false);
    engine.typed_config.mapmaking.enabled = true;
    engine.typed_config.mapmaking.method = citlali::config::MapMethod::naive;
    engine.typed_config.mapmaking.grouping =
        citlali::config::MapGrouping::array;
    engine.typed_config.coadd.enabled = true;
    engine.typed_config.noise.enabled = true;
    engine.typed_config.noise.n_noise_maps = 2;
    engine.typed_config.noise.write_realizations = true;
    engine.typed_config.noise.products_enabled = true;
    engine.typed_config.post_processing.map_filtering.enabled = false;

    engine.map_indices.n_maps = 1;
    engine.map_indices.maps_to_arrays.resize(1);
    engine.map_indices.maps_to_arrays.setZero();
    engine.map_indices.arrays_to_maps.resize(1);
    engine.map_indices.arrays_to_maps.setZero();
    engine.map_indices.maps_to_stokes.resize(1);
    engine.map_indices.maps_to_stokes.setZero();
    engine.calib.n_arrays = 1;
    engine.calib.arrays.resize(1);
    engine.calib.arrays.setZero();
    engine.omb.map_grouping = "array";
    engine.telescope.pixel_axes = "radec";
    engine.telescope.sim_obs = false;
    engine.telescope.tel_header["Header.Source.Epoch"] =
        Eigen::VectorXd::Constant(1, 2000.0);
    engine.rtcproc.run_polarization = false;
    engine.rtcproc.polarization.stokes_params.clear();
    engine.rtcproc.polarization.stokes_params[0] = "I";
}

std::shared_ptr<ScienceMapBufferFixture> make_production_science_map_buffer(
    const Engine &engine, bool coadd, Eigen::Index rows, Eigen::Index cols,
    const std::array<double, 2> &reference_pixel) {
    auto map = std::make_shared<ScienceMapBufferFixture>(
        coadd ? "cmb" : "omb");
    map->n_rows = rows;
    map->n_cols = cols;
    map->n_noise = 2;
    map->pixel_size_rad = 2.0 * ASEC_TO_RAD;
    map->sig_unit = "mJy/beam";
    map->map_grouping = "array";
    map->cov_cut = 1.0;
    map->science_products.allocate(1, rows, cols, coadd, true, true);

    auto &products = map->science_products;
    auto &realized = products.realized[0];
    mapmaking::ScienceMapBundleIdentity identity;
    identity.grouping = map->map_grouping;
    identity.signal_unit = map->sig_unit;
    identity.estimator_identity =
        mapmaking::science_map_coadd_estimator_version;
    identity.response_identity =
        citlali::pipeline::science_map_response_identity(
            engine.rtcproc.kernel, false);
    identity.required_companions = {
        "noise_realization_0_I", "noise_realization_1_I"};
    identity.rows = rows;
    identity.cols = cols;
    identity.wcs.coordinate_frame = "radec";
    identity.wcs.projection = "TAN";
    identity.wcs.axis_types = {"RA---TAN", "DEC--TAN"};
    identity.wcs.axis_units = {"deg", "deg"};
    identity.wcs.pixel_scale = {
        -0.00055555555555555556, 0.00055555555555555556};
    identity.wcs.reference_world = {
        187.046325, 44.093558300000005};
    identity.wcs.reference_pixel = {
        reference_pixel[0], reference_pixel[1]};
    identity.wcs.source_epoch = 2000.0;
    identity.wcs.orientation_rad = 0.0;
    mapmaking::ScienceMapSlotIdentity slot;
    slot.ordered_slot = 0;
    slot.grouping = "array";
    slot.group_identity = "array:0";
    slot.array_identity = 0;
    slot.stokes_identity = 0;
    slot.frequency_hz = engine.toltec_io.array_freq_map.at(0);
    identity.ordered_slots.push_back(slot);
    products.bundle_identity = identity;
    products.identity_admitted = true;

    realized.normalization.support_algorithm =
        mapmaking::science_map_normalization_support_version;
    realized.normalization.coefficient_stage = coadd
        ? mapmaking::science_map_coadd_normalization_coefficient_stage
        : mapmaking::science_map_observation_normalization_coefficient_stage;
    realized.normalization.requested_cut = 0.1;
    realized.normalization.realized_cut = 0.1;
    realized.normalization.realized_threshold = 0.001453413509532904;
    realized.normalization.selected_positive_value = 0.01453413509532904;
    realized.normalization.positive_value_count =
        static_cast<std::size_t>(rows * cols - 1);
    realized.normalization.selected_zero_based_index =
        realized.normalization.positive_value_count / 2;
    realized.normalization.selected_index_available = true;
    realized.science_policy.support_algorithm =
        mapmaking::science_map_policy_support_version;
    realized.science_policy.coefficient_stage =
        realized.normalization.coefficient_stage;
    realized.science_policy.requested_cut = 1.0;
    realized.science_policy.realized_cut = 1.0;
    realized.science_policy.realized_threshold = 0.01453413509532904;
    realized.science_policy.selected_positive_value = 0.01453413509532904;
    realized.science_policy.positive_value_count =
        realized.normalization.positive_value_count;
    realized.science_policy.selected_zero_based_index =
        realized.normalization.selected_zero_based_index;
    realized.science_policy.selected_index_available = true;

    map->signal.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->weight.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->coverage.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->weight_formal.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->noise_variance.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->sig2noise_pixel.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->noise.emplace_back(rows, cols, map->n_noise);
    map->noise[0].setZero();
    for (Eigen::Index row = 0; row < rows; ++row) {
        for (Eigen::Index col = 0; col < cols; ++col) {
            const bool supported = row != 0 || col != 0;
            products.geometric_hits[0](row, col) = 1;
            products.contributing_hits[0](row, col) = supported ? 1 : 0;
            products.coadd_observation_count[0](row, col) =
                coadd && supported ? 1 : 0;
            products.upstream_eligible_exposure[0](row, col) = 1.0;
            products.retained_exposure[0](row, col) = supported ? 1.0 : 0.0;
            products.normalization_support[0](row, col) = supported ? 1 : 0;
            products.science_policy_support[0](row, col) = supported ? 1 : 0;
            products.science_valid[0](row, col) = supported ? 1 : 0;
            map->signal[0](row, col) =
                supported ? 10.0 + row + 0.01 * col : 0.0;
            map->weight[0](row, col) =
                supported ? realized.science_policy.realized_threshold : 0.0;
            map->coverage[0](row, col) =
                products.retained_exposure[0](row, col);
            map->weight_formal[0](row, col) =
                supported ? 2.0 * map->weight[0](row, col) : 0.0;
            map->noise_variance[0](row, col) = supported ? 4.0 : 0.0;
            map->sig2noise_pixel[0](row, col) =
                supported ? map->signal[0](row, col) / 2.0 : 0.0;
            for (Eigen::Index realization = 0;
                 realization < map->n_noise; ++realization) {
                map->noise[0](row, col, realization) = supported
                    ? 100.0 * (realization + 1) + 10.0 * row + col
                    : 0.0;
            }
        }
    }
    map->median_err = Eigen::VectorXd::Constant(1, 1.0);
    map->median_rms = Eigen::VectorXd::Constant(1, 2.0);
    map->wcs.ctype = {"RA---TAN", "DEC--TAN", "FREQ", "STOKES"};
    map->wcs.cunit = {"deg", "deg", "Hz", "1"};
    map->wcs.crval = {0.0F, 0.0F, 0.0F, 0.0F};
    map->wcs.cdelt = {0.0F, 0.0F, 1.0F, 1.0F};
    map->wcs.crpix = {0.0F, 0.0F, 0.0F, 0.0F};
    map->wcs.naxis = {
        static_cast<int>(cols), static_cast<int>(rows), 1, 1};
    mapmaking::science_map_finalize_realized_product_facts(*map, 0);
    return map;
}

struct FitsSpatialWcs {
    std::array<double, 2> cdelt{};
    std::array<double, 2> crpix{};
    std::array<double, 2> crval{};
    std::array<std::string, 2> ctype{};
    std::array<std::string, 2> cunit{};
    long cols = 0;
    long rows = 0;
};

double read_required_fits_double(fitsfile *file, const char *key) {
    double value = 0.0;
    int status = 0;
    if (fits_read_key(file, TDOUBLE, key, &value, nullptr, &status) != 0) {
        throw std::runtime_error(std::string{"missing FITS double key "} + key);
    }
    return value;
}

std::string read_required_fits_string(fitsfile *file, const char *key) {
    char value[FLEN_VALUE] = {};
    int status = 0;
    if (fits_read_key(file, TSTRING, key, value, nullptr, &status) != 0) {
        throw std::runtime_error(std::string{"missing FITS string key "} + key);
    }
    return value;
}

void move_to_required_image(fitsfile *file, const std::string &name) {
    int status = 0;
    if (fits_movnam_hdu(file, IMAGE_HDU, const_cast<char *>(name.c_str()), 0,
                       &status) != 0) {
        throw std::runtime_error("missing FITS image " + name);
    }
}

FitsSpatialWcs read_spatial_wcs(fitsfile *file, const std::string &name) {
    move_to_required_image(file, name);
    FitsSpatialWcs wcs;
    for (std::size_t axis = 0; axis < 2; ++axis) {
        const auto suffix = std::to_string(axis + 1);
        wcs.cdelt[axis] =
            read_required_fits_double(file, ("CDELT" + suffix).c_str());
        wcs.crpix[axis] =
            read_required_fits_double(file, ("CRPIX" + suffix).c_str());
        wcs.crval[axis] =
            read_required_fits_double(file, ("CRVAL" + suffix).c_str());
        wcs.ctype[axis] =
            read_required_fits_string(file, ("CTYPE" + suffix).c_str());
        wcs.cunit[axis] =
            read_required_fits_string(file, ("CUNIT" + suffix).c_str());
    }
    long axes[4] = {};
    int status = 0;
    if (fits_get_img_size(file, 4, axes, &status) != 0) {
        throw std::runtime_error("cannot read FITS image shape");
    }
    wcs.cols = axes[0];
    wcs.rows = axes[1];
    return wcs;
}

std::array<double, 2> inverse_tan_world(const FitsSpatialWcs &wcs,
                                        long row, long col) {
    const double deg_to_rad = M_PI / 180.0;
    const double xi =
        ((static_cast<double>(col) + 1.0) - wcs.crpix[0]) *
        wcs.cdelt[0] * deg_to_rad;
    const double eta =
        ((static_cast<double>(row) + 1.0) - wcs.crpix[1]) *
        wcs.cdelt[1] * deg_to_rad;
    const double ra0 = wcs.crval[0] * deg_to_rad;
    const double dec0 = wcs.crval[1] * deg_to_rad;
    const double denominator = std::cos(dec0) - eta * std::sin(dec0);
    const double ra = ra0 + std::atan2(xi, denominator);
    const double dec = std::atan2(
        std::sin(dec0) + eta * std::cos(dec0),
        std::hypot(denominator, xi));
    return {ra, dec};
}

double sky_separation_arcsec(const std::array<double, 2> &lhs,
                             const std::array<double, 2> &rhs) {
    const double half_delta_ra = (lhs[0] - rhs[0]) / 2.0;
    const double half_delta_dec = (lhs[1] - rhs[1]) / 2.0;
    const double haversine =
        std::sin(half_delta_dec) * std::sin(half_delta_dec) +
        std::cos(lhs[1]) * std::cos(rhs[1]) *
            std::sin(half_delta_ra) * std::sin(half_delta_ra);
    return 2.0 * std::asin(std::sqrt(std::clamp(haversine, 0.0, 1.0))) *
        (180.0 / M_PI) * 3600.0;
}

double maximum_wcs_separation_arcsec(
    const mapmaking::ScienceMapWcsIdentity &typed,
    const FitsSpatialWcs &physical) {
    FitsSpatialWcs typed_wcs;
    typed_wcs.cdelt = {typed.pixel_scale[0], typed.pixel_scale[1]};
    typed_wcs.crpix = {
        typed.reference_pixel[0] + 1.0,
        typed.reference_pixel[1] + 1.0};
    typed_wcs.crval = {typed.reference_world[0], typed.reference_world[1]};
    double maximum = 0.0;
    for (long row = 0; row < physical.rows; ++row) {
        for (long col = 0; col < physical.cols; ++col) {
            maximum = std::max(
                maximum,
                sky_separation_arcsec(
                    inverse_tan_world(typed_wcs, row, col),
                    inverse_tan_world(physical, row, col)));
        }
    }
    return maximum;
}

TEST(science_map_fits_products, preserves_native_fits_scalar_types) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    const std::string base =
        "/private/tmp/citlali-science-map-fits-types-" +
        std::to_string(nonce);
    FitsFileCleanup cleanup{base + ".fits"};

    using FitsOutput = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;
    FitsOutput output{base};
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> counts(1, 3);
    counts << -((std::int64_t{1} << 54) + 7), 17,
        (std::int64_t{1} << 54) + 3;
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask(1, 3);
    mask << 0, 1, 255;
    Eigen::MatrixXd values(1, 3);
    values << -2.5, 0.0, 7.25;
    output.add_hdu("counts", counts);
    output.add_hdu("mask", mask);
    output.add_hdu("values", values);
    output.pfits.reset();

    fitsfile *file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&file, cleanup.path.c_str(), READONLY, &status), 0);

    auto move_to_image = [&](const char *name, int expected_bitpix) {
        ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                                  const_cast<char *>(name), 0, &status),
                  0);
        int bitpix = 0;
        ASSERT_EQ(fits_get_img_type(file, &bitpix, &status), 0);
        EXPECT_EQ(bitpix, expected_bitpix);
    };

    move_to_image("counts", LONGLONG_IMG);
    long long count_values[3] = {};
    int any_null = 0;
    ASSERT_EQ(fits_read_img(file, TLONGLONG, 1, 3, nullptr, count_values,
                            &any_null, &status),
              0);
    EXPECT_EQ(count_values[0], static_cast<long long>(counts(0, 2)));
    EXPECT_EQ(count_values[1], static_cast<long long>(counts(0, 1)));
    EXPECT_EQ(count_values[2], static_cast<long long>(counts(0, 0)));

    move_to_image("mask", BYTE_IMG);
    unsigned char mask_values[3] = {};
    ASSERT_EQ(fits_read_img(file, TBYTE, 1, 3, nullptr, mask_values,
                            &any_null, &status),
              0);
    EXPECT_EQ(mask_values[0], 255);
    EXPECT_EQ(mask_values[1], 1);
    EXPECT_EQ(mask_values[2], 0);

    move_to_image("values", DOUBLE_IMG);
    double double_values[3] = {};
    ASSERT_EQ(fits_read_img(file, TDOUBLE, 1, 3, nullptr, double_values,
                            &any_null, &status),
              0);
    EXPECT_DOUBLE_EQ(double_values[0], values(0, 2));
    EXPECT_DOUBLE_EQ(double_values[1], values(0, 1));
    EXPECT_DOUBLE_EQ(double_values[2], values(0, 0));

    EXPECT_EQ(fits_close_file(file, &status), 0);
}

TEST(science_map_fits_products,
     round_trips_complete_f010_bundle_metadata_aliases_and_wcs) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    const std::string base =
        "/private/tmp/citlali-science-map-f010-bundle-" +
        std::to_string(nonce);
    FitsFileCleanup cleanup{base + ".fits"};
    using FitsOutput = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;
    FitsOutput output{base};
    auto map = make_science_map_buffer(true);

    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", map->wcs, 2000.0,
        science_map_test_logger());
    output.pfits.reset();

    fitsfile *file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&file, cleanup.path.c_str(), READONLY, &status), 0);
    const std::vector<std::pair<std::string, int>> products = {
        {"geometric_hits_I", LONGLONG_IMG},
        {"contributing_hits_I", LONGLONG_IMG},
        {"coadd_observation_count_I", LONGLONG_IMG},
        {"upstream_eligible_exposure_I", DOUBLE_IMG},
        {"retained_exposure_I", DOUBLE_IMG},
        {"normalization_support_I", BYTE_IMG},
        {"science_policy_support_I", BYTE_IMG},
        {"science_valid_I", BYTE_IMG},
        {"coverage_I", DOUBLE_IMG},
        {"coverage_bool_I", BYTE_IMG},
    };
    auto read_string_key = [&](const char *key) {
        char value[FLEN_VALUE] = {};
        char comment[FLEN_COMMENT] = {};
        EXPECT_EQ(fits_read_key(file, TSTRING, key, value, comment, &status),
                  0);
        return std::string(value);
    };
    for (const auto &[name, expected_bitpix] : products) {
        ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                                  const_cast<char *>(name.c_str()), 0,
                                  &status),
                  0)
            << name;
        int bitpix = 0;
        ASSERT_EQ(fits_get_img_type(file, &bitpix, &status), 0) << name;
        EXPECT_EQ(bitpix, expected_bitpix) << name;
        EXPECT_EQ(read_string_key("CTYPE1"), "RA---TAN") << name;
        EXPECT_EQ(read_string_key("CUNIT1"), "deg") << name;
        double value = 0.0;
        ASSERT_EQ(fits_read_key(file, TDOUBLE, "CRVAL1", &value, nullptr,
                                &status),
                  0)
            << name;
        EXPECT_DOUBLE_EQ(value, 123.25) << name;
        ASSERT_EQ(fits_read_key(file, TDOUBLE, "CRPIX1", &value, nullptr,
                                &status),
                  0)
            << name;
        EXPECT_DOUBLE_EQ(value, 1.5) << name;
        ASSERT_EQ(fits_read_key(file, TDOUBLE, "EQUINOX", &value, nullptr,
                                &status),
                  0)
            << name;
        EXPECT_DOUBLE_EQ(value, 2000.0) << name;
    }

    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("coverage_I"), 0, &status),
              0);
    EXPECT_EQ(read_string_key("ALIASOF"), "retained_exposure_I");
    EXPECT_EQ(read_string_key("BUNIT"), "detector s");
    double coverage[4] = {};
    int any_null = 0;
    ASSERT_EQ(fits_read_img(file, TDOUBLE, 1, 4, nullptr, coverage, &any_null,
                            &status),
              0);
    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("retained_exposure_I"), 0,
                              &status),
              0);
    double retained[4] = {};
    ASSERT_EQ(fits_read_img(file, TDOUBLE, 1, 4, nullptr, retained, &any_null,
                            &status),
              0);
    for (std::size_t index = 0; index < 4; ++index) {
        EXPECT_EQ(std::memcmp(&coverage[index], &retained[index],
                              sizeof(double)),
                  0);
    }

    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("coverage_bool_I"), 0,
                              &status),
              0);
    EXPECT_EQ(read_string_key("ALIASOF"), "science_policy_support_I");
    EXPECT_EQ(read_string_key("DEPRCATD"), "true");
    EXPECT_EQ(read_string_key("VALAUTH"), "false");
    unsigned char coverage_mask[4] = {};
    ASSERT_EQ(fits_read_img(file, TBYTE, 1, 4, nullptr, coverage_mask,
                            &any_null, &status),
              0);
    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("science_policy_support_I"),
                              0, &status),
              0);
    unsigned char policy_mask[4] = {};
    ASSERT_EQ(fits_read_img(file, TBYTE, 1, 4, nullptr, policy_mask,
                            &any_null, &status),
              0);
    for (std::size_t index = 0; index < 4; ++index) {
        EXPECT_EQ(coverage_mask[index], policy_mask[index]);
    }

    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("science_valid_I"), 0,
                              &status),
              0);
    EXPECT_EQ(read_string_key("VALAUTH"), "true");
    EXPECT_EQ(read_string_key("DATTYP"), "uint8");
    EXPECT_EQ(fits_close_file(file, &status), 0);
}

TEST(science_map_fits_products,
     filtered_fits_round_trips_complete_raw_parent_digest) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    const std::string base =
        "/private/tmp/citlali-science-map-filtered-parent-" +
        std::to_string(nonce);
    FitsFileCleanup cleanup{base + ".fits"};
    using FitsOutput = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;
    FitsOutput output{base};
    auto map = make_science_map_buffer(false);
    map->freeze_raw_science_parent();
    ASSERT_TRUE(map->raw_science_parent);
    const auto expected =
        map->raw_science_parent->realized[0].raw_parent_digest;
    ASSERT_GT(expected.size(), 68U);
    map->signal[0].setConstant(5.0);

    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", map->wcs, 2000.0,
        science_map_test_logger(), true);
    output.pfits.reset();

    fitsfile *file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&file, cleanup.path.c_str(), READONLY, &status),
              0);
    ASSERT_EQ(fits_movnam_hdu(
                  file, IMAGE_HDU,
                  const_cast<char *>("science_valid_I"), 0, &status),
              0);
    char *value = nullptr;
    char comment[FLEN_COMMENT] = {};
    ASSERT_EQ(fits_read_key_longstr(file, "RAWPDGST", &value, comment,
                                    &status),
              0);
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(std::string(value), expected);
    int free_status = 0;
    fits_free_memory(value, &free_status);
    EXPECT_EQ(free_status, 0);
    EXPECT_EQ(fits_close_file(file, &status), 0);
}

TEST(science_map_fits_products,
     coadd_enabled_observation_inventory_keeps_required_noise_file) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    FitsDirectoryCleanup cleanup{
        std::filesystem::path{"/private/tmp"} /
        ("citlali-science-map-observation-inventory-" +
         std::to_string(nonce))};
    Engine engine;
    configure_production_writer_engine(engine);
    engine.output_paths.obsnum_dir_name = cleanup.path.string() + "/";
    engine.observation_identity.obsnum = "152390";
    std::filesystem::create_directories(cleanup.path / "raw");

    ASSERT_NO_THROW(engine.create_obs_map_files());
    ASSERT_EQ(engine.map_fits_outputs.obs.size(), 1U);
    ASSERT_EQ(engine.map_fits_outputs.obs_noise.size(), 1U);
    EXPECT_TRUE(engine.map_fits_outputs.filtered_obs.empty());
    EXPECT_TRUE(engine.map_fits_outputs.filtered_obs_noise.empty());
    EXPECT_NE(engine.map_fits_outputs.obs[0].filepath.find("152390"),
              std::string::npos);
    EXPECT_NE(engine.map_fits_outputs.obs_noise[0].filepath.find("152390"),
              std::string::npos);
    EXPECT_NE(engine.map_fits_outputs.obs_noise[0].filepath.find("_noise"),
              std::string::npos);
}

TEST(science_map_fits_products,
     production_writer_preserves_wcs_threshold_and_realization_contracts) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    FitsDirectoryCleanup cleanup{
        std::filesystem::path{"/private/tmp"} /
        ("citlali-science-map-production-writer-" +
         std::to_string(nonce))};
    Engine engine;
    configure_production_writer_engine(engine);
    engine.output_paths.obsnum_dir_name = cleanup.path.string() + "/obs/";
    engine.observation_identity.obsnum = "152390";
    std::filesystem::create_directories(cleanup.path / "obs" / "raw");
    std::filesystem::create_directories(cleanup.path / "coadd" / "raw");
    ASSERT_NO_THROW(engine.create_obs_map_files());

    const std::string coadd_map_base =
        (cleanup.path / "coadd" / "raw" / "coadd_map").string();
    const std::string coadd_noise_base =
        (cleanup.path / "coadd" / "raw" / "coadd_noise").string();
    engine.map_fits_outputs.coadd.emplace_back(coadd_map_base);
    engine.map_fits_outputs.coadd_noise.emplace_back(coadd_noise_base);

    constexpr Eigen::Index obs_rows = 9;
    constexpr Eigen::Index obs_cols = 11;
    constexpr Eigen::Index coadd_rows = 13;
    constexpr Eigen::Index coadd_cols = 17;
    constexpr long delta_row = 2;
    constexpr long delta_col = 3;
    auto observation = make_production_science_map_buffer(
        engine, false, obs_rows, obs_cols, {5.0, 4.0});
    auto coadd = make_production_science_map_buffer(
        engine, true, coadd_rows, coadd_cols, {8.0, 6.0});
    auto *observation_map_files = &engine.map_fits_outputs.obs;
    auto *observation_noise_files = &engine.map_fits_outputs.obs_noise;
    auto *coadd_map_files = &engine.map_fits_outputs.coadd;
    auto *coadd_noise_files = &engine.map_fits_outputs.coadd_noise;
    ASSERT_NO_THROW(engine.write_maps(
        observation_map_files, observation_noise_files, observation, 0));
    ASSERT_NO_THROW(engine.write_maps(
        coadd_map_files, coadd_noise_files, coadd, 0));

    const auto observation_map_path =
        engine.map_fits_outputs.obs[0].filepath + ".fits";
    const auto observation_noise_path =
        engine.map_fits_outputs.obs_noise[0].filepath + ".fits";
    const auto coadd_map_path = coadd_map_base + ".fits";
    const auto coadd_noise_path = coadd_noise_base + ".fits";

    decltype(engine.map_fits_outputs.obs) failed_map_files;
    decltype(engine.map_fits_outputs.obs_noise) missing_noise_files;
    const std::string failed_map_base =
        (cleanup.path / "required_write_failure").string();
    failed_map_files.emplace_back(failed_map_base);
    auto failed_observation = make_production_science_map_buffer(
        engine, false, obs_rows, obs_cols, {5.0, 4.0});
    const auto failed_wcs = failed_observation->wcs;
    auto *failed_map_file_ptr = &failed_map_files;
    auto *missing_noise_file_ptr = &missing_noise_files;
    EXPECT_THROW(
        engine.write_maps(
            failed_map_file_ptr, missing_noise_file_ptr, failed_observation,
            0),
        std::runtime_error);
    EXPECT_TRUE(failed_map_files[0].hdus.empty());
    EXPECT_EQ(failed_observation->wcs.cdelt, failed_wcs.cdelt);
    EXPECT_EQ(failed_observation->wcs.crpix, failed_wcs.crpix);
    EXPECT_EQ(failed_observation->wcs.crval, failed_wcs.crval);

    engine.map_fits_outputs.obs[0].pfits.reset();
    engine.map_fits_outputs.obs_noise[0].pfits.reset();
    engine.map_fits_outputs.coadd[0].pfits.reset();
    engine.map_fits_outputs.coadd_noise[0].pfits.reset();
    failed_map_files[0].pfits.reset();

    fitsfile *observation_file = nullptr;
    fitsfile *observation_noise_file = nullptr;
    fitsfile *coadd_file = nullptr;
    fitsfile *coadd_noise_file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&observation_file, observation_map_path.c_str(),
                             READONLY, &status),
              0);
    status = 0;
    ASSERT_EQ(fits_open_file(&observation_noise_file,
                             observation_noise_path.c_str(), READONLY,
                             &status),
              0);
    status = 0;
    ASSERT_EQ(fits_open_file(&coadd_file, coadd_map_path.c_str(), READONLY,
                             &status),
              0);
    status = 0;
    ASSERT_EQ(fits_open_file(&coadd_noise_file, coadd_noise_path.c_str(),
                             READONLY, &status),
              0);

    const auto observation_wcs =
        read_spatial_wcs(observation_file, "signal_I");
    const auto coadd_wcs = read_spatial_wcs(coadd_file, "signal_I");
    const auto &observation_identity =
        observation->science_products.bundle_identity->wcs;
    const auto &coadd_identity =
        coadd->science_products.bundle_identity->wcs;
    const double observation_max_separation =
        maximum_wcs_separation_arcsec(
            observation_identity, observation_wcs);
    const double coadd_max_separation =
        maximum_wcs_separation_arcsec(coadd_identity, coadd_wcs);
    EXPECT_GT(observation_max_separation, 0.0);
    EXPECT_GT(coadd_max_separation, 0.0);
    EXPECT_LE(observation_max_separation, 0.1);
    EXPECT_LE(coadd_max_separation, 0.1);

    for (const auto *wcs : {&observation_wcs, &coadd_wcs}) {
        EXPECT_EQ(wcs->ctype[0], "RA---TAN");
        EXPECT_EQ(wcs->ctype[1], "DEC--TAN");
        EXPECT_EQ(wcs->cunit[0], "deg");
        EXPECT_EQ(wcs->cunit[1], "deg");
        EXPECT_TRUE(std::signbit(wcs->cdelt[0]));
        EXPECT_FALSE(std::signbit(wcs->cdelt[1]));
    }
    EXPECT_DOUBLE_EQ(observation_identity.orientation_rad, 0.0);
    EXPECT_DOUBLE_EQ(coadd_identity.orientation_rad, 0.0);
    EXPECT_EQ(observation_wcs.rows, obs_rows);
    EXPECT_EQ(observation_wcs.cols, obs_cols);
    EXPECT_EQ(coadd_wcs.rows, coadd_rows);
    EXPECT_EQ(coadd_wcs.cols, coadd_cols);
    EXPECT_DOUBLE_EQ(
        coadd_identity.reference_pixel[0],
        observation_identity.reference_pixel[0] + delta_col);
    EXPECT_DOUBLE_EQ(
        coadd_identity.reference_pixel[1],
        observation_identity.reference_pixel[1] + delta_row);
    EXPECT_DOUBLE_EQ(
        coadd_wcs.crpix[0], observation_wcs.crpix[0] + delta_col);
    EXPECT_DOUBLE_EQ(
        coadd_wcs.crpix[1], observation_wcs.crpix[1] + delta_row);

    const auto &coadd_realized = coadd->science_products.realized[0];
    const auto normalization_sidecar =
        citlali::pipeline::science_map_threshold_realization_node(
            coadd_realized.normalization);
    const auto policy_sidecar =
        citlali::pipeline::science_map_threshold_realization_node(
            coadd_realized.science_policy);
    const double normalization_authority =
        citlali::pipeline::science_map_exact_double_value(
            normalization_sidecar["realized_threshold"]);
    const double policy_authority =
        citlali::pipeline::science_map_exact_double_value(
            policy_sidecar["realized_threshold"]);
    auto verify_threshold_card = [&](const std::string &hdu_name,
                                     const std::string &estimator,
                                     double authority) {
        move_to_required_image(coadd_file, hdu_name);
        const double card = read_required_fits_double(coadd_file, "WTTHRESH");
        EXPECT_TRUE(std::isfinite(card)) << hdu_name;
        EXPECT_EQ(read_required_fits_string(coadd_file, "BUNIT"), "1")
            << hdu_name;
        EXPECT_EQ(read_required_fits_string(coadd_file, "ESTTYPE"), estimator)
            << hdu_name;
        EXPECT_LE(std::abs(card - authority),
                  1.0e-12 * std::abs(authority))
            << hdu_name;
        return card;
    };
    verify_threshold_card(
        "normalization_support_I", "normalization_support",
        normalization_authority);
    const double policy_card = verify_threshold_card(
        "science_policy_support_I", "science_policy_support",
        policy_authority);
    const double alias_card = verify_threshold_card(
        "coverage_bool_I", "science_policy_support", policy_authority);
    EXPECT_DOUBLE_EQ(policy_card, alias_card);
    move_to_required_image(coadd_file, "coverage_bool_I");
    EXPECT_EQ(read_required_fits_string(coadd_file, "ALIASOF"),
              "science_policy_support_I");

    ASSERT_TRUE(observation->science_products.bundle_identity);
    ASSERT_TRUE(coadd->science_products.bundle_identity);
    EXPECT_EQ(
        observation->science_products.bundle_identity->response_identity,
        coadd->science_products.bundle_identity->response_identity);
    EXPECT_EQ(
        observation->science_products.bundle_identity->required_companions,
        coadd->science_products.bundle_identity->required_companions);
    const std::vector<std::string> realization_names = {
        "signal_0_I", "signal_1_I"};
    auto verify_realization_file = [&](
        fitsfile *file, const ScienceMapBufferFixture &map) {
        int hdu_count = 0;
        int local_status = 0;
        ASSERT_EQ(fits_get_num_hdus(file, &hdu_count, &local_status), 0);
        EXPECT_EQ(hdu_count, 3);
        for (Eigen::Index realization = 0;
             realization < map.n_noise; ++realization) {
            const auto realization_index =
                static_cast<std::size_t>(realization);
            const auto realization_wcs =
                read_spatial_wcs(file, realization_names[realization_index]);
            EXPECT_EQ(realization_wcs.rows, map.n_rows);
            EXPECT_EQ(realization_wcs.cols, map.n_cols);
            EXPECT_EQ(read_required_fits_string(file, "UNIT"), map.sig_unit);
            std::vector<double> values(
                static_cast<std::size_t>(map.n_rows * map.n_cols));
            int any_null = 0;
            local_status = 0;
            ASSERT_EQ(
                fits_read_img(file, TDOUBLE, 1,
                              static_cast<long>(values.size()), nullptr,
                              values.data(), &any_null, &local_status),
                0);
            for (Eigen::Index row = 0; row < map.n_rows; ++row) {
                for (Eigen::Index output_col = 0;
                     output_col < map.n_cols; ++output_col) {
                    const Eigen::Index source_col =
                        map.n_cols - output_col - 1;
                    const auto flat = static_cast<std::size_t>(
                        row * map.n_cols + output_col);
                    EXPECT_DOUBLE_EQ(
                        values[flat],
                        map.noise[0](row, source_col, realization));
                    if (!map.science_products.normalization_support[0](
                            row, source_col)) {
                        EXPECT_DOUBLE_EQ(values[flat], 0.0);
                    }
                }
            }
        }
    };
    verify_realization_file(observation_noise_file, *observation);
    verify_realization_file(coadd_noise_file, *coadd);

    status = 0;
    EXPECT_EQ(fits_close_file(observation_file, &status), 0);
    status = 0;
    EXPECT_EQ(fits_close_file(observation_noise_file, &status), 0);
    status = 0;
    EXPECT_EQ(fits_close_file(coadd_file, &status), 0);
    status = 0;
    EXPECT_EQ(fits_close_file(coadd_noise_file, &status), 0);
}

TEST(science_map_fits_products,
     unavailable_profile_representation_has_explicit_absence) {
    mapmaking::ScienceMapProducts detector;
    detector.allocate(2, 3, 4, false, false, false);
    EXPECT_TRUE(detector.initialized);
    EXPECT_FALSE(detector.ordinary_contribution_predicate_available);
    EXPECT_TRUE(detector.geometric_hits.empty());
    ASSERT_EQ(detector.realized.size(), 2U);
    for (const auto &record : detector.realized) {
        for (const auto &reason : record.product_absence_reason) {
            EXPECT_EQ(reason,
                      "method-specific contribution predicate unavailable");
        }
    }
    EXPECT_TRUE(citlali::pipeline::science_map_unavailable_output_bundle_complete(
        detector, 2));
}

TEST(science_map_fits_products,
     detector_profile_preserves_empty_coverage_output_guard) {
    auto detector = std::make_shared<mapmaking::MapBuffer>("omb");
    detector->n_rows = 3;
    detector->n_cols = 4;
    detector->signal = {Eigen::MatrixXd::Ones(3, 4)};
    detector->weight = {Eigen::MatrixXd::Ones(3, 4)};
    detector->science_products.allocate(
        1, 3, 4, false, false, false,
        "detector-grouping science-map product profile is unavailable");
    ASSERT_TRUE(detector->coverage.empty());
    ASSERT_FALSE(citlali::pipeline::science_map_successor_coadd_product(
        detector->science_products));
    CapturedFitsEntry output;
    DummyWcs wcs;

    EXPECT_NO_THROW(citlali::pipeline::add_coverage_support_image_hdus(
        output, detector, 0, "detector_", "I", wcs, 2000.0, false,
        true, false, science_map_test_logger()));
    EXPECT_TRUE(output.images.empty());
}

TEST(science_map_fits_products,
     unavailable_legacy_coadd_is_not_promoted_to_successor_output_policy) {
    mapmaking::ScienceMapProducts legacy_coadd;
    legacy_coadd.allocate(
        1, 3, 4, true, false, false,
        "method-specific contribution predicate unavailable");

    EXPECT_FALSE(citlali::pipeline::science_map_successor_coadd_product(
        legacy_coadd));
}

TEST(science_map_fits_products,
     rejects_tampered_unavailable_inventory_before_first_hdu) {
    mapmaking::ScienceMapProducts unavailable;
    unavailable.allocate(
        1, 3, 4, false, false, false,
        "non-array map-grouping science-map product profile is unavailable");
    unavailable.realized[0].product_absence_reason[0].clear();
    CapturedFitsEntry output;

    EXPECT_THROW(
        citlali::pipeline::require_science_map_output_profile_authority(
            unavailable, 1, 3, 4, science_map_test_logger()),
        std::runtime_error);
    EXPECT_TRUE(output.images.empty());

    unavailable.realized[0].product_absence_reason[0] = "restored";
    unavailable.geometric_hits.emplace_back(
        mapmaking::ScienceMapCountPlane::Zero(3, 4));
    EXPECT_THROW(
        citlali::pipeline::require_science_map_output_profile_authority(
            unavailable, 1, 3, 4, science_map_test_logger()),
        std::runtime_error);
    EXPECT_TRUE(output.images.empty());
}

}  // namespace

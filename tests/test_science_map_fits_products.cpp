#include <gtest/gtest.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/pipeline/map_image_output_helpers.h>
#include <citlali/core/utils/fits_io.h>

#include <fitsio.h>
#include <spdlog/sinks/null_sink.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <map>
#include <memory>
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
    EXPECT_FALSE(captured_has_image(output, "sig2noise_I"));
    EXPECT_FALSE(captured_has_image(output, "sig2noise_pixel_I"));
    EXPECT_FALSE(captured_has_image(output, "point_source_flux_I"));
    EXPECT_FALSE(captured_has_image(output, "point_source_uncertainty_I"));
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

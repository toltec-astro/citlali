#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/pipeline/map_buffer_allocation.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using PtcData =
    timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>;
using Apt = std::map<std::string, Eigen::VectorXd>;

constexpr Eigen::Index kRows = 3;
constexpr Eigen::Index kCols = 3;
constexpr Eigen::Index kSamples = 5;
constexpr Eigen::Index kDetectors = 3;
constexpr Eigen::Index kNoiseRealizations = 2;
constexpr double kPixelSizeRad = 1.0e-5;
constexpr double kSampleRateHz = 4.0;

struct OrdinaryFixture {
    PtcData data;
    Apt apt;
};

OrdinaryFixture make_ordinary_fixture() {
    OrdinaryFixture fixture;
    auto &data = fixture.data;
    data.scans.data.resize(kSamples, kDetectors);
    data.scans.data <<
        1.0, 10.0, -1.0,
        2.0, 20.0, -2.0,
        3.0, 30.0, -3.0,
        4.0, 40.0, -4.0,
        5.0, 50.0, -5.0;
    data.kernel.data.resize(kSamples, kDetectors);
    data.kernel.data <<
        2.0, 3.0, 4.0,
        5.0, 6.0, 7.0,
        8.0, 9.0, 10.0,
        11.0, 12.0, 13.0,
        14.0, 15.0, 16.0;
    data.flags.data.resize(kSamples, kDetectors);
    data.flags.data.setConstant(false);
    data.flags.data(1, 1) = true;
    data.weights.data.resize(kDetectors);
    data.weights.data << 1.0, 2.0, 4.0;
    data.noise.data.resize(kNoiseRealizations, kDetectors);
    data.noise.data << 1, -1, 1,
                      -1, 1, 1;
    data.index.data = 17;

    Eigen::VectorXd lat(kSamples);
    Eigen::VectorXd lon(kSamples);
    lat << 0.0, -kPixelSizeRad, kPixelSizeRad,
           -kPixelSizeRad, kPixelSizeRad;
    lon << 0.0, -kPixelSizeRad, kPixelSizeRad,
           kPixelSizeRad, -kPixelSizeRad;
    data.tel_data.data["TelElAct"] = Eigen::VectorXd::Zero(kSamples);
    data.tel_data.data["alt_phys"] = lat;
    data.tel_data.data["az_phys"] = lon;
    data.pointing_offsets_arcsec.data["az"] =
        Eigen::VectorXd::Zero(kSamples);
    data.pointing_offsets_arcsec.data["alt"] =
        Eigen::VectorXd::Zero(kSamples);

    fixture.apt["array"] = Eigen::VectorXd::Zero(kDetectors);
    fixture.apt["flag"] = Eigen::VectorXd::Zero(kDetectors);
    fixture.apt["x_t"] = Eigen::VectorXd::Zero(kDetectors);
    fixture.apt["y_t"] = Eigen::VectorXd::Zero(kDetectors);
    fixture.apt["uid"].resize(kDetectors);
    fixture.apt["uid"] << 101.0, 102.0, 103.0;
    return fixture;
}

mapmaking::ScienceMapBundleIdentity make_bundle_identity(
    const std::string &grouping, Eigen::Index map_count) {
    mapmaking::ScienceMapBundleIdentity identity;
    identity.grouping = grouping;
    identity.signal_unit = "mJy/beam";
    identity.estimator_identity =
        "ordinary-naive-normalized-gridding-v1";
    identity.response_identity = "response:test-kernel";
    identity.required_companions = {
        "kernel_I", "noise_realization_0_I", "noise_realization_1_I"};
    identity.wcs.coordinate_frame = "altaz";
    identity.wcs.projection = "TAN";
    identity.wcs.axis_types = {"AZ---TAN", "ALT--TAN"};
    identity.wcs.axis_units = {"deg", "deg"};
    identity.wcs.pixel_scale = {-kPixelSizeRad, kPixelSizeRad};
    identity.wcs.reference_world = {0.0, 0.0};
    identity.wcs.reference_pixel = {1.0, 1.0};
    identity.wcs.source_epoch = 2000.0;
    identity.rows = kRows;
    identity.cols = kCols;
    for (Eigen::Index map_index = 0; map_index < map_count; ++map_index) {
        mapmaking::ScienceMapSlotIdentity slot;
        slot.ordered_slot = static_cast<std::size_t>(map_index);
        slot.grouping = grouping;
        slot.group_identity = grouping + ":" + std::to_string(map_index);
        slot.array_identity = grouping == "array" ? map_index : 0;
        slot.stokes_identity = 0;
        slot.frequency_hz = 2.0e11;
        identity.ordered_slots.push_back(slot);
    }
    return identity;
}

mapmaking::MapBuffer make_ordinary_map(const std::string &grouping,
                                       Eigen::Index map_count,
                                       bool with_noise = true) {
    mapmaking::MapBuffer map{"omb"};
    map.n_rows = kRows;
    map.n_cols = kCols;
    map.pixel_size_rad = kPixelSizeRad;
    map.map_grouping = grouping;
    map.parallel_policy = "seq";
    map.sig_unit = "mJy/beam";
    map.cov_cut = 0.0;
    map.n_noise = with_noise ? kNoiseRealizations : 0;
    map.randomize_dets = true;
    citlali::pipeline::allocate_map_matrices(
        map, map_count, false, true, true, true);
    if (with_noise) {
        for (Eigen::Index map_index = 0; map_index < map_count; ++map_index) {
            map.noise.emplace_back(kRows, kCols, kNoiseRealizations);
            map.noise.back().setZero();
        }
    }
    map.science_products.bundle_identity =
        make_bundle_identity(grouping, map_count);
    map.science_products.identity_admitted = true;
    return map;
}

mapmaking::MapBuffer make_noise_sentinel(Eigen::Index map_count) {
    mapmaking::MapBuffer coadd{"cmb"};
    coadd.n_rows = kRows;
    coadd.n_cols = kCols;
    coadd.pixel_size_rad = kPixelSizeRad;
    coadd.n_noise = kNoiseRealizations;
    coadd.randomize_dets = true;
    for (Eigen::Index map_index = 0; map_index < map_count; ++map_index) {
        coadd.noise.emplace_back(kRows, kCols, kNoiseRealizations);
        coadd.noise.back().setConstant(9876.5);
    }
    return coadd;
}

Eigen::VectorXi map_indices_for(const std::string &grouping) {
    Eigen::VectorXi map_indices(kDetectors);
    if (grouping == "array") {
        map_indices.setZero();
    }
    else {
        map_indices << 0, 1, 2;
    }
    return map_indices;
}

template <class Lhs, class Rhs>
void expect_matrix_exact(const Lhs &lhs, const Rhs &rhs) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (Eigen::Index col = 0; col < lhs.cols(); ++col) {
        for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
            if constexpr (std::is_floating_point_v<typename Lhs::Scalar>) {
                EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
                    static_cast<double>(lhs(row, col)),
                    static_cast<double>(rhs(row, col))))
                    << "row=" << row << " col=" << col;
            }
            else {
                EXPECT_EQ(lhs(row, col), rhs(row, col))
                    << "row=" << row << " col=" << col;
            }
        }
    }
}

void expect_noise_exact(const Eigen::Tensor<double, 3> &lhs,
                        const Eigen::Tensor<double, 3> &rhs) {
    ASSERT_EQ(lhs.dimension(0), rhs.dimension(0));
    ASSERT_EQ(lhs.dimension(1), rhs.dimension(1));
    ASSERT_EQ(lhs.dimension(2), rhs.dimension(2));
    for (Eigen::Index realization = 0; realization < lhs.dimension(2);
         ++realization) {
        for (Eigen::Index col = 0; col < lhs.dimension(1); ++col) {
            for (Eigen::Index row = 0; row < lhs.dimension(0); ++row) {
                EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
                    lhs(row, col, realization),
                    rhs(row, col, realization)))
                    << "row=" << row << " col=" << col
                    << " realization=" << realization;
            }
        }
    }
}

void expect_map_exact(const mapmaking::MapBuffer &lhs,
                      const mapmaking::MapBuffer &rhs) {
    ASSERT_EQ(lhs.signal.size(), rhs.signal.size());
    for (std::size_t map_index = 0; map_index < lhs.signal.size();
         ++map_index) {
        expect_matrix_exact(lhs.signal[map_index], rhs.signal[map_index]);
        expect_matrix_exact(lhs.weight[map_index], rhs.weight[map_index]);
        expect_matrix_exact(lhs.kernel[map_index], rhs.kernel[map_index]);
        expect_matrix_exact(lhs.coverage[map_index], rhs.coverage[map_index]);
        expect_matrix_exact(
            lhs.science_products.geometric_hits[map_index],
            rhs.science_products.geometric_hits[map_index]);
        expect_matrix_exact(
            lhs.science_products.contributing_hits[map_index],
            rhs.science_products.contributing_hits[map_index]);
        expect_matrix_exact(
            lhs.science_products.upstream_eligible_exposure[map_index],
            rhs.science_products.upstream_eligible_exposure[map_index]);
        expect_matrix_exact(
            lhs.science_products.retained_exposure[map_index],
            rhs.science_products.retained_exposure[map_index]);
        if (!lhs.noise.empty()) {
            expect_noise_exact(lhs.noise[map_index], rhs.noise[map_index]);
        }
    }
}

template <class PlaneSelector>
void expect_scan_merge_plane_within_registered_bound(
    const Eigen::MatrixXd &actual,
    const std::vector<mapmaking::MapBuffer> &per_scan,
    PlaneSelector select_plane, const char *plane_name) {
    const long double term_count =
        static_cast<long double>(per_scan.size());
    const long double epsilon =
        static_cast<long double>(std::numeric_limits<double>::epsilon());
    const long double gamma_n =
        (term_count * epsilon) / (1.0L - term_count * epsilon);
    for (Eigen::Index col = 0; col < actual.cols(); ++col) {
        for (Eigen::Index row = 0; row < actual.rows(); ++row) {
            long double reference = 0.0L;
            long double sum_abs = 0.0L;
            for (const auto &scan : per_scan) {
                const long double value = static_cast<long double>(
                    select_plane(scan)(row, col));
                reference += value;
                sum_abs += std::abs(value);
            }
            const long double error = std::abs(
                static_cast<long double>(actual(row, col)) - reference);
            const long double bound = 2.0L * gamma_n * sum_abs;
            EXPECT_LE(error, bound)
                << "policy="
                << mapmaking::science_map_parallel_equivalence_policy
                << " plane=" << plane_name << " row=" << row
                << " col=" << col << " reference="
                << static_cast<double>(reference) << " error="
                << static_cast<double>(error) << " bound="
                << static_cast<double>(bound);
        }
    }
}

template <class PlaneSelector>
void expect_scan_merge_count_exact(
    const mapmaking::ScienceMapCountPlane &actual,
    const std::vector<mapmaking::MapBuffer> &per_scan,
    PlaneSelector select_plane) {
    for (Eigen::Index col = 0; col < actual.cols(); ++col) {
        for (Eigen::Index row = 0; row < actual.rows(); ++row) {
            std::int64_t expected = 0;
            for (const auto &scan : per_scan) {
                expected += select_plane(scan)(row, col);
            }
            EXPECT_EQ(actual(row, col), expected);
        }
    }
}

void expect_scan_merge_realization_within_registered_bound(
    const Eigen::Tensor<double, 3> &actual,
    const std::vector<mapmaking::MapBuffer> &per_scan,
    Eigen::Index map_index) {
    const long double term_count =
        static_cast<long double>(per_scan.size());
    const long double epsilon =
        static_cast<long double>(std::numeric_limits<double>::epsilon());
    const long double gamma_n =
        (term_count * epsilon) / (1.0L - term_count * epsilon);
    for (Eigen::Index realization = 0;
         realization < actual.dimension(2); ++realization) {
        for (Eigen::Index col = 0; col < actual.dimension(1); ++col) {
            for (Eigen::Index row = 0; row < actual.dimension(0); ++row) {
                long double reference = 0.0L;
                long double sum_abs = 0.0L;
                for (const auto &scan : per_scan) {
                    const long double value = static_cast<long double>(
                        scan.noise[static_cast<std::size_t>(map_index)](
                            row, col, realization));
                    reference += value;
                    sum_abs += std::abs(value);
                }
                const long double error = std::abs(
                    static_cast<long double>(
                        actual(row, col, realization)) -
                    reference);
                const long double bound = 2.0L * gamma_n * sum_abs;
                EXPECT_LE(error, bound)
                    << "policy="
                    << mapmaking::science_map_parallel_equivalence_policy
                    << " plane=noise_realization map=" << map_index
                    << " realization=" << realization << " row=" << row
                    << " col=" << col << " reference="
                    << static_cast<double>(reference) << " error="
                    << static_cast<double>(error) << " bound="
                    << static_cast<double>(bound);
            }
        }
    }
}

mapmaking::MapBuffer run_ordinary_profile(const std::string &grouping,
                                          bool requested_parallel,
                                          mapmaking::MapBuffer *coadd_out = nullptr) {
    auto fixture = make_ordinary_fixture();
    const Eigen::Index map_count = grouping == "array" ? 1 : kDetectors;
    auto map = make_ordinary_map(grouping, map_count);
    auto coadd = make_noise_sentinel(map_count);
    auto map_indices = map_indices_for(grouping);
    std::string pixel_axes = "altaz";
    mapmaking::NaiveMapmaker mapmaker;
    mapmaker.run_polarization = false;
    if (requested_parallel) {
        mapmaker.populate_maps_naive_parallel(
            fixture.data, map, coadd, map_indices, pixel_axes, fixture.apt,
            kSampleRateHz, true, true);
    }
    else {
        mapmaker.populate_maps_naive(
            fixture.data, map, coadd, map_indices, pixel_axes, fixture.apt,
            kSampleRateHz, true, true);
    }
    if (coadd_out != nullptr) {
        *coadd_out = std::move(coadd);
    }
    return map;
}

mapmaking::MapBuffer independent_ordinary_reference(
    const std::string &grouping) {
    auto fixture = make_ordinary_fixture();
    const Eigen::Index map_count = grouping == "array" ? 1 : kDetectors;
    auto expected = make_ordinary_map(grouping, map_count);
    auto map_indices = map_indices_for(grouping);
    const double sample_seconds = 1.0 / kSampleRateHz;

    const auto &lat = fixture.data.tel_data.data.at("alt_phys");
    const auto &lon = fixture.data.tel_data.data.at("az_phys");
    for (Eigen::Index detector = 0; detector < kDetectors; ++detector) {
        const Eigen::Index map_index = map_indices(detector);
        for (Eigen::Index sample = 0; sample < kSamples; ++sample) {
            const Eigen::Index row = static_cast<Eigen::Index>(std::llround(
                lat(sample) / kPixelSizeRad + (kRows - 1) / 2.0));
            const Eigen::Index col = static_cast<Eigen::Index>(std::llround(
                lon(sample) / kPixelSizeRad + (kCols - 1) / 2.0));
            if (row < 0 || row >= kRows || col < 0 || col >= kCols) {
                continue;
            }
            expected.science_products.geometric_hits[map_index](row, col) +=
                1;
            if (fixture.apt.at("flag")(detector) != 0.0 ||
                fixture.data.flags.data(sample, detector)) {
                continue;
            }
            expected.science_products.upstream_eligible_exposure[map_index](
                row, col) += sample_seconds;
            const double coefficient = fixture.data.weights.data(detector);
            if (!(coefficient > 0.0)) {
                continue;
            }
            const double weighted_signal =
                coefficient * fixture.data.scans.data(sample, detector);
            expected.signal[map_index](row, col) += weighted_signal;
            expected.weight[map_index](row, col) += coefficient;
            expected.kernel[map_index](row, col) +=
                coefficient * fixture.data.kernel.data(sample, detector);
            expected.coverage[map_index](row, col) += sample_seconds;
            expected.science_products.contributing_hits[map_index](row, col) +=
                1;
            expected.science_products.retained_exposure[map_index](row, col) +=
                sample_seconds;
            for (Eigen::Index realization = 0;
                 realization < kNoiseRealizations; ++realization) {
                expected.noise[map_index](row, col, realization) +=
                    static_cast<double>(
                        fixture.data.noise.data(realization, detector)) *
                    weighted_signal;
            }
        }
    }
    return expected;
}

bool map_has_only_zero_accumulators(const mapmaking::MapBuffer &map) {
    for (std::size_t map_index = 0; map_index < map.signal.size();
         ++map_index) {
        if ((map.signal[map_index].array() != 0.0).any() ||
            (map.weight[map_index].array() != 0.0).any() ||
            (map.kernel[map_index].array() != 0.0).any() ||
            (map.coverage[map_index].array() != 0.0).any() ||
            (map.science_products.geometric_hits[map_index].array() != 0)
                .any() ||
            (map.science_products.contributing_hits[map_index].array() != 0)
                .any() ||
            (map.science_products.upstream_eligible_exposure[map_index]
                 .array() != 0.0)
                .any() ||
            (map.science_products.retained_exposure[map_index].array() != 0.0)
                .any()) {
            return false;
        }
        for (Eigen::Index realization = 0;
             realization < map.n_noise; ++realization) {
            for (Eigen::Index col = 0; col < map.n_cols; ++col) {
                for (Eigen::Index row = 0; row < map.n_rows; ++row) {
                    if (map.noise[map_index](row, col, realization) != 0.0) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool map_has_only_zero_numerical_contributions(
    const mapmaking::MapBuffer &map) {
    for (std::size_t map_index = 0; map_index < map.signal.size();
         ++map_index) {
        if ((map.signal[map_index].array() != 0.0).any() ||
            (map.weight[map_index].array() != 0.0).any() ||
            (map.kernel[map_index].array() != 0.0).any() ||
            (map.coverage[map_index].array() != 0.0).any() ||
            (map.science_products.contributing_hits[map_index].array() != 0)
                .any() ||
            (map.science_products.upstream_eligible_exposure[map_index]
                 .array() != 0.0)
                .any() ||
            (map.science_products.retained_exposure[map_index].array() != 0.0)
                .any()) {
            return false;
        }
        for (Eigen::Index realization = 0;
             realization < map.n_noise; ++realization) {
            for (Eigen::Index col = 0; col < map.n_cols; ++col) {
                for (Eigen::Index row = 0; row < map.n_rows; ++row) {
                    if (map.noise[map_index](row, col, realization) != 0.0) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

TEST(science_map_truth, small_matrix_mean_and_covariance_match_equations) {
    Eigen::Matrix<double, 2, 4> q;
    q << 1.0, 2.0, 0.0, 0.0,
         0.0, 0.0, 3.0, 1.0;
    const Eigen::Vector2d denominator = q.rowwise().sum();
    Eigen::Matrix<double, 2, 4> A = q;
    A.row(0) /= denominator(0);
    A.row(1) /= denominator(1);

    Eigen::Vector4d mean;
    mean << 2.0, 5.0, -1.0, 7.0;
    const Eigen::Vector2d mapped_mean = A * mean;
    EXPECT_NEAR(mapped_mean(0), 4.0, 1.0e-12);
    EXPECT_NEAR(mapped_mean(1), 1.0, 1.0e-12);

    Eigen::Matrix4d diagonal_covariance = Eigen::Matrix4d::Zero();
    diagonal_covariance.diagonal() << 4.0, 9.0, 16.0, 25.0;
    const Eigen::Matrix2d mapped_diagonal =
        A * diagonal_covariance * A.transpose();
    EXPECT_NEAR(mapped_diagonal(0, 0), 40.0 / 9.0, 1.0e-12);
    EXPECT_NEAR(mapped_diagonal(0, 1), 0.0, 1.0e-12);
    EXPECT_NEAR(mapped_diagonal(1, 0), 0.0, 1.0e-12);
    EXPECT_NEAR(mapped_diagonal(1, 1), 169.0 / 16.0, 1.0e-12);

    Eigen::Matrix4d correlated_covariance;
    correlated_covariance <<
        4.0, 1.0, 0.5, -0.25,
        1.0, 9.0, 0.0, 0.75,
        0.5, 0.0, 16.0, -2.0,
        -0.25, 0.75, -2.0, 25.0;
    const Eigen::Matrix2d mapped_correlated =
        A * correlated_covariance * A.transpose();
    EXPECT_NEAR(mapped_correlated(0, 0), 44.0 / 9.0, 1.0e-12);
    EXPECT_NEAR(mapped_correlated(0, 1), 11.0 / 48.0, 1.0e-12);
    EXPECT_NEAR(mapped_correlated(1, 0), 11.0 / 48.0, 1.0e-12);
    EXPECT_NEAR(mapped_correlated(1, 1), 157.0 / 16.0, 1.0e-12);

    // The covariance comparison is equation truth only. It does not promote
    // weight_I to precision or authorize a correlated GLS production path.
}

TEST(science_map_truth,
     ordinary_profiles_match_reference_and_requested_parallel_exactly) {
    for (const std::string grouping : {"array", "detector"}) {
        auto expected = independent_ordinary_reference(grouping);
        mapmaking::MapBuffer sequential_coadd;
        mapmaking::MapBuffer requested_parallel_coadd;
        auto sequential =
            run_ordinary_profile(grouping, false, &sequential_coadd);
        auto requested_parallel =
            run_ordinary_profile(grouping, true, &requested_parallel_coadd);

        expect_map_exact(sequential, expected);
        expect_map_exact(requested_parallel, expected);
        expect_map_exact(requested_parallel, sequential);

        ASSERT_EQ(sequential_coadd.noise.size(),
                  requested_parallel_coadd.noise.size());
        for (std::size_t map_index = 0;
             map_index < sequential_coadd.noise.size(); ++map_index) {
            Eigen::Tensor<double, 3> sentinel(
                kRows, kCols, kNoiseRealizations);
            sentinel.setConstant(9876.5);
            expect_noise_exact(sequential_coadd.noise[map_index], sentinel);
            expect_noise_exact(requested_parallel_coadd.noise[map_index],
                               sentinel);
        }
    }
}

TEST(science_map_truth,
     all_invalid_samples_have_geometric_hits_but_zero_valid_coverage) {
    auto fixture = make_ordinary_fixture();
    fixture.data.flags.data.setConstant(true);
    auto map = make_ordinary_map("array", 1);
    auto coadd = make_noise_sentinel(1);
    auto map_indices = map_indices_for("array");
    std::string pixel_axes = "altaz";
    mapmaking::NaiveMapmaker mapmaker;
    mapmaker.run_polarization = false;
    mapmaker.populate_maps_naive_parallel(
        fixture.data, map, coadd, map_indices, pixel_axes, fixture.apt,
        kSampleRateHz, true, true);

    EXPECT_EQ(map.science_products.geometric_hits[0].sum(),
              kSamples * kDetectors);
    EXPECT_EQ(map.science_products.contributing_hits[0].sum(), 0);
    EXPECT_DOUBLE_EQ(
        map.science_products.upstream_eligible_exposure[0].sum(), 0.0);
    EXPECT_DOUBLE_EQ(map.science_products.retained_exposure[0].sum(), 0.0);
    EXPECT_DOUBLE_EQ(map.coverage[0].sum(), 0.0);
    EXPECT_DOUBLE_EQ(map.weight[0].sum(), 0.0);

    map.normalize_maps();
    EXPECT_EQ(map.science_products.normalization_support[0].sum(), 0);
    EXPECT_EQ(map.science_products.science_policy_support[0].sum(), 0);
    EXPECT_EQ(map.science_products.science_valid[0].sum(), 0);
    EXPECT_DOUBLE_EQ(map.coverage[0].sum(), 0.0);
    EXPECT_EQ(map.science_products.realized[0]
                  .normalization.positive_value_count,
              0U);
    EXPECT_DOUBLE_EQ(map.science_products.realized[0]
                         .normalization.realized_threshold,
                     0.0);
}

TEST(science_map_truth,
     zero_coefficients_are_eligible_but_have_zero_numerical_support) {
    auto fixture = make_ordinary_fixture();
    fixture.data.flags.data.setConstant(false);
    fixture.data.weights.data.setZero();
    auto map = make_ordinary_map("array", 1);
    auto coadd = make_noise_sentinel(1);
    auto map_indices = map_indices_for("array");
    std::string pixel_axes = "altaz";
    mapmaking::NaiveMapmaker mapmaker;
    mapmaker.run_polarization = false;
    mapmaker.populate_maps_naive(
        fixture.data, map, coadd, map_indices, pixel_axes, fixture.apt,
        kSampleRateHz, true, true);

    EXPECT_EQ(map.science_products.geometric_hits[0].sum(),
              kSamples * kDetectors);
    EXPECT_DOUBLE_EQ(
        map.science_products.upstream_eligible_exposure[0].sum(),
        static_cast<double>(kSamples * kDetectors) / kSampleRateHz);
    EXPECT_EQ(map.science_products.contributing_hits[0].sum(), 0);
    EXPECT_DOUBLE_EQ(map.weight[0].sum(), 0.0);
    EXPECT_DOUBLE_EQ(map.coverage[0].sum(), 0.0);
    EXPECT_DOUBLE_EQ(map.science_products.retained_exposure[0].sum(), 0.0);

    map.normalize_maps();
    EXPECT_EQ(map.science_products.normalization_support[0].sum(), 0);
    EXPECT_EQ(map.science_products.science_policy_support[0].sum(), 0);
    EXPECT_EQ(map.science_products.science_valid[0].sum(), 0);
}

TEST(science_map_truth,
     valid_nonfinite_ordinary_inputs_fail_before_any_map_commit) {
    enum class Target { signal, coefficient, kernel, projection };
    const std::vector<double> nonfinite = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};

    for (const double bad : nonfinite) {
        for (const Target target : {Target::signal, Target::coefficient,
                                    Target::kernel, Target::projection}) {
            auto fixture = make_ordinary_fixture();
            fixture.data.flags.data.setConstant(true);
            fixture.data.flags.data(0, 0) = false;
            if (target == Target::signal) {
                fixture.data.scans.data(0, 0) = bad;
            }
            else if (target == Target::coefficient) {
                fixture.data.weights.data(0) = bad;
            }
            else if (target == Target::kernel) {
                fixture.data.kernel.data(0, 0) = bad;
            }
            else {
                fixture.data.tel_data.data["alt_phys"](0) = bad;
            }

            auto map = make_ordinary_map("array", 1);
            auto coadd = make_noise_sentinel(1);
            auto map_indices = map_indices_for("array");
            std::string pixel_axes = "altaz";
            mapmaking::NaiveMapmaker mapmaker;
            mapmaker.run_polarization = false;
            EXPECT_THROW(
                mapmaker.populate_maps_naive_parallel(
                    fixture.data, map, coadd, map_indices, pixel_axes,
                    fixture.apt, kSampleRateHz, true, true),
                std::runtime_error);
            EXPECT_TRUE(map_has_only_zero_accumulators(map));
        }
    }

    for (const double bad : nonfinite) {
        auto fixture = make_ordinary_fixture();
        fixture.data.flags.data.setConstant(true);
        fixture.data.scans.data(0, 0) = bad;
        fixture.data.weights.data(0) = bad;
        fixture.data.kernel.data(0, 0) = bad;
        fixture.data.tel_data.data["alt_phys"](0) = bad;
        auto map = make_ordinary_map("array", 1);
        auto coadd = make_noise_sentinel(1);
        auto map_indices = map_indices_for("array");
        std::string pixel_axes = "altaz";
        mapmaking::NaiveMapmaker mapmaker;
        mapmaker.run_polarization = false;
        EXPECT_NO_THROW(mapmaker.populate_maps_naive_parallel(
            fixture.data, map, coadd, map_indices, pixel_axes, fixture.apt,
            kSampleRateHz, true, true));
        EXPECT_TRUE(map_has_only_zero_numerical_contributions(map));
    }
}

TEST(science_map_truth,
     malformed_required_kernel_shape_fails_before_any_map_commit) {
    auto fixture = make_ordinary_fixture();
    fixture.data.kernel.data.resize(kSamples - 1, kDetectors);
    auto map = make_ordinary_map("array", 1);
    auto coadd = make_noise_sentinel(1);
    auto map_indices = map_indices_for("array");
    std::string pixel_axes = "altaz";
    mapmaking::NaiveMapmaker mapmaker;
    mapmaker.run_polarization = false;

    EXPECT_THROW(
        mapmaker.populate_maps_naive_parallel(
            fixture.data, map, coadd, map_indices, pixel_axes, fixture.apt,
            kSampleRateHz, true, true),
        std::runtime_error);
    EXPECT_TRUE(map_has_only_zero_accumulators(map));
}

TEST(science_map_truth,
     floating_and_count_aggregate_overflow_fail_before_bundle_commit) {
    enum class OverflowTarget {
        staged_signal,
        live_signal,
        live_realization,
        live_geometric_count,
    };

    for (const auto target : {
             OverflowTarget::staged_signal,
             OverflowTarget::live_signal,
             OverflowTarget::live_realization,
             OverflowTarget::live_geometric_count}) {
        auto fixture = make_ordinary_fixture();
        fixture.data.flags.data.setConstant(true);
        fixture.data.tel_data.data["alt_phys"].setZero();
        fixture.data.tel_data.data["az_phys"].setZero();
        fixture.data.weights.data.setOnes();
        fixture.data.scans.data.setZero();
        fixture.data.kernel.data.setZero();

        auto map = make_ordinary_map("array", 1);
        constexpr Eigen::Index center = 1;
        if (target == OverflowTarget::staged_signal) {
            fixture.data.flags.data(0, 0) = false;
            fixture.data.flags.data(1, 0) = false;
            fixture.data.scans.data(0, 0) =
                std::numeric_limits<double>::max();
            fixture.data.scans.data(1, 0) =
                std::numeric_limits<double>::max();
        }
        else if (target == OverflowTarget::live_signal) {
            fixture.data.flags.data(0, 0) = false;
            fixture.data.scans.data(0, 0) =
                std::numeric_limits<double>::max();
            map.signal[0](center, center) =
                std::numeric_limits<double>::max();
        }
        else if (target == OverflowTarget::live_realization) {
            fixture.data.flags.data(0, 0) = false;
            fixture.data.scans.data(0, 0) =
                std::numeric_limits<double>::max();
            map.noise[0](center, center, 0) =
                std::numeric_limits<double>::max();
        }
        else {
            map.science_products.geometric_hits[0](center, center) =
                std::numeric_limits<std::int64_t>::max();
        }

        const auto before = map;
        auto coadd = make_noise_sentinel(1);
        auto map_indices = map_indices_for("array");
        std::string pixel_axes = "altaz";
        mapmaking::NaiveMapmaker mapmaker;
        mapmaker.run_polarization = false;
        EXPECT_THROW(
            mapmaker.populate_maps_naive_parallel(
                fixture.data, map, coadd, map_indices, pixel_axes,
                fixture.apt, kSampleRateHz, true, true),
            std::runtime_error);
        expect_map_exact(map, before);
    }
}

TEST(science_map_truth,
     finite_projection_outside_index_domain_fails_before_bundle_commit) {
    auto fixture = make_ordinary_fixture();
    fixture.data.flags.data.setConstant(true);
    fixture.data.tel_data.data["alt_phys"].setConstant(1.0);
    fixture.data.tel_data.data["az_phys"].setZero();
    auto map = make_ordinary_map("array", 1);
    map.pixel_size_rad = 1.0e-300;
    const auto before = map;
    auto coadd = make_noise_sentinel(1);
    auto map_indices = map_indices_for("array");
    std::string pixel_axes = "altaz";
    mapmaking::NaiveMapmaker mapmaker;
    mapmaker.run_polarization = false;

    EXPECT_THROW(
        mapmaker.populate_maps_naive_parallel(
            fixture.data, map, coadd, map_indices, pixel_axes, fixture.apt,
            kSampleRateHz, true, true),
        std::runtime_error);
    expect_map_exact(map, before);
}

TEST(science_map_truth, repaired_primitive_is_race_free_under_concurrent_calls) {
    constexpr int kThreadCount = 8;
    constexpr std::array<double, kThreadCount> signal_scales{
        1.0e12, -1.0e12, 1.0e6, -1.0e6,
        3.0, -7.0, 1.0e-6, -2.0e-6};
    constexpr std::array<double, kThreadCount> kernel_scales{
        1.0e9, -1.0e9, 1.0e4, -1.0e4,
        0.25, -0.5, 1.0e-7, -3.0e-7};
    std::vector<OrdinaryFixture> fixtures;
    std::vector<Eigen::VectorXi> indices;
    std::vector<std::string> axes;
    fixtures.reserve(kThreadCount);
    indices.reserve(kThreadCount);
    axes.reserve(kThreadCount);
    for (int thread = 0; thread < kThreadCount; ++thread) {
        fixtures.push_back(make_ordinary_fixture());
        fixtures.back().data.scans.data *= signal_scales[thread];
        fixtures.back().data.kernel.data *= kernel_scales[thread];
        fixtures.back().data.weights.data *=
            1.0 + 0.125 * static_cast<double>(thread);
        fixtures.back().data.flags.data(
            (thread + 2) % kSamples, thread % kDetectors) =
            (thread % 2) == 0;
        fixtures.back().data.index.data = 100 + thread;
        indices.push_back(map_indices_for("array"));
        axes.emplace_back("altaz");
    }

    std::vector<mapmaking::MapBuffer> per_scan;
    per_scan.reserve(kThreadCount);
    for (int scan = 0; scan < kThreadCount; ++scan) {
        auto local_map = make_ordinary_map("array", 1, true);
        mapmaking::MapBuffer local_coadd{"unused"};
        mapmaking::NaiveMapmaker local_mapmaker;
        local_mapmaker.run_polarization = false;
        local_mapmaker.populate_maps_naive_parallel(
            fixtures[scan].data, local_map, local_coadd, indices[scan],
            axes[scan], fixtures[scan].apt, kSampleRateHz, true, true);
        per_scan.push_back(std::move(local_map));
    }
    ASSERT_FALSE(mapmaking::science_map_exact_double_equal(
        per_scan[0].signal[0].sum(), per_scan[1].signal[0].sum()));

    auto shared_map = make_ordinary_map("array", 1, true);
    mapmaking::MapBuffer shared_coadd{"unused"};
    mapmaking::NaiveMapmaker mapmaker;
    mapmaker.run_polarization = false;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::exception_ptr> errors(kThreadCount);
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int thread = 0; thread < kThreadCount; ++thread) {
        threads.emplace_back([&, thread] {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            try {
                mapmaker.populate_maps_naive_parallel(
                    fixtures[thread].data, shared_map, shared_coadd,
                    indices[thread], axes[thread], fixtures[thread].apt,
                    kSampleRateHz, true, true);
            }
            catch (...) {
                errors[thread] = std::current_exception();
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kThreadCount) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    for (auto &thread : threads) {
        thread.join();
    }
    for (const auto &error : errors) {
        EXPECT_EQ(error, nullptr);
    }

    expect_scan_merge_plane_within_registered_bound(
        shared_map.signal[0], per_scan,
        [](const auto &map) -> const auto & { return map.signal[0]; },
        "signal");
    expect_scan_merge_plane_within_registered_bound(
        shared_map.weight[0], per_scan,
        [](const auto &map) -> const auto & { return map.weight[0]; },
        "weight");
    expect_scan_merge_plane_within_registered_bound(
        shared_map.kernel[0], per_scan,
        [](const auto &map) -> const auto & { return map.kernel[0]; },
        "kernel");
    expect_scan_merge_plane_within_registered_bound(
        shared_map.coverage[0], per_scan,
        [](const auto &map) -> const auto & { return map.coverage[0]; },
        "coverage");
    expect_scan_merge_plane_within_registered_bound(
        shared_map.science_products.upstream_eligible_exposure[0], per_scan,
        [](const auto &map) -> const auto & {
            return map.science_products.upstream_eligible_exposure[0];
        },
        "upstream_eligible_exposure");
    expect_scan_merge_plane_within_registered_bound(
        shared_map.science_products.retained_exposure[0], per_scan,
        [](const auto &map) -> const auto & {
            return map.science_products.retained_exposure[0];
        },
        "retained_exposure");
    expect_scan_merge_count_exact(
        shared_map.science_products.geometric_hits[0], per_scan,
        [](const auto &map) -> const auto & {
            return map.science_products.geometric_hits[0];
        });
    expect_scan_merge_count_exact(
        shared_map.science_products.contributing_hits[0], per_scan,
        [](const auto &map) -> const auto & {
            return map.science_products.contributing_hits[0];
        });
    expect_scan_merge_realization_within_registered_bound(
        shared_map.noise[0], per_scan, 0);
}

}  // namespace

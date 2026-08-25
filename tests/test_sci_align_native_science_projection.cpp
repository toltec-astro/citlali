#include "sci_align_native_gap_fixture.h"

#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/pipeline/timestream_native_science_projection.h>

#include <gtest/gtest.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fixture = citlali::test_support::sci_align;
namespace pipeline = citlali::pipeline;
using PtcData =
    timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>;
using Apt = std::map<std::string, Eigen::VectorXd>;

constexpr double kPixelSizeRad = 1.0e-5;
constexpr double kSampleRateHz = 100.0;

fixture::NativeGapFixtureV1 complete_identical_time_fixture() {
    auto result = fixture::load_native_gap_fixture_v1();
    auto &network7 = result.network(7);
    network7.reconstructed_times_unix_sec =
        result.common_slot_reference_times_unix_sec;
    network7.packet_counters = {700, 701, 702, 703, 704};
    network7.legacy_presence_mask = Eigen::VectorXi::Ones(5);
    network7.expected_slot_native_rows = {700, 701, 702, 703, 704};
    network7.measured_values.resize(5, 2);
    network7.measured_values <<
        7.1, 8.3,
        7.8, 9.4,
        9.2, 10.1,
        10.4, 11.8,
        12.7, 13.2;
    network7.original_flag_bits =
        pipeline::NativeDetectorFlagBitsMatrix::Zero(5, 2);
    result.network(0).original_flag_bits =
        pipeline::NativeDetectorFlagBitsMatrix::Zero(5, 2);
    network7.expected_packet_contiguous_runs = {{700, 705}};
    result.expected_complete_cohort_slot_runs = {{0, 5}};
    return result;
}

pipeline::NativeRtcDispatchResult identity_rtc(
    const pipeline::NativeMeasuredDetectorScan &scan) {
    return pipeline::dispatch_native_rtc_runs(
        scan, {1, false}, [](const pipeline::NativeRtcRunInput &input) {
            return pipeline::NativeRtcProcessedRun{
                input.measured_values, input.input_flag_bits};
        });
}

pipeline::NativePtcCohortRequest ptc_request() {
    return {"all", pipeline::FinitePcaPlaceholder::checked(-77.0),
            {}, {}, false, false};
}

pipeline::NativeScienceProjectionRequest projection_request(
    const pipeline::NativeMeasuredDetectorScan &scan) {
    pipeline::NativeScienceProjectionRequest request;
    request.pixel_axes = "altaz";
    request.map_grouping = "detector";
    for (std::size_t detector = 0; detector < scan.detector_count();
         ++detector) {
        const auto column = static_cast<Eigen::Index>(detector);
        const auto &binding = scan.binding(column);
        request.detectors.push_back({
            column, binding.output_uid, binding.array,
            binding.network_id, binding.apt_flag, column, 0.0, 0.0});
    }
    return request;
}

struct CommittedProjection {
    std::shared_ptr<const pipeline::NativeMeasuredDetectorScan> scan;
    pipeline::NativeMeasuredDetectorLedger ledger;
    pipeline::NativePtcPreparedOperation prepared;
    pipeline::NativeScienceProjection projection;
};

CommittedProjection make_committed_projection(
    const fixture::NativeGapFixtureV1 &loaded) {
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    pipeline::NativeMeasuredDetectorLedger ledger{scan};
    const auto rtc = identity_rtc(*scan);
    auto prepared = pipeline::prepare_native_ptc_cohorts(
        ledger, rtc, ptc_request());
    const auto processed = pipeline::run_native_ptc_groups(
        prepared, [](const auto &group) { return group.values(); });
    pipeline::scatter_native_ptc_results_transactionally(
        ledger, prepared, processed);
    auto projection = pipeline::make_native_science_projection(
        ledger, prepared, projection_request(*scan));
    return {std::move(scan), std::move(ledger), std::move(prepared),
            std::move(projection)};
}

PtcData projection_input(
    const pipeline::NativeScienceProjection &projection,
    bool legacy_pointing = false) {
    PtcData result;
    result.scans.data = projection.values();
    result.flags.data = projection.flags();
    result.kernel.data.resize(projection.row_count(),
                              projection.detector_count());
    for (Eigen::Index row = 0; row < projection.row_count(); ++row) {
        for (Eigen::Index detector = 0;
             detector < projection.detector_count(); ++detector) {
            result.kernel.data(row, detector) =
                0.25 * static_cast<double>((row + 1) * (detector + 2));
        }
    }
    result.weights.data = Eigen::VectorXd::LinSpaced(
        projection.detector_count(), 1.0,
        static_cast<double>(projection.detector_count()));
    result.index.data = 5;
    if (legacy_pointing) {
        // At identical native times, these detector matrices collapse to the
        // established rectangular telescope representation exactly.
        result.tel_data.data["TelElAct"] =
            Eigen::VectorXd::Zero(projection.row_count());
        result.tel_data.data["alt_phys"] =
            projection.latitudes_rad().col(0);
        result.tel_data.data["az_phys"] =
            projection.longitudes_rad().col(0);
        result.pointing_offsets_arcsec.data["az"] =
            Eigen::VectorXd::Zero(projection.row_count());
        result.pointing_offsets_arcsec.data["alt"] =
            Eigen::VectorXd::Zero(projection.row_count());
    }
    return result;
}

Apt projection_apt(const pipeline::NativeScienceProjection &projection) {
    Apt apt;
    const auto count = projection.detector_count();
    apt["array"].resize(count);
    apt["flag"] = Eigen::VectorXd::Zero(count);
    apt["x_t"].resize(count);
    apt["y_t"].resize(count);
    apt["uid"].resize(count);
    for (Eigen::Index detector = 0; detector < count; ++detector) {
        const auto &binding = projection.detectors().at(
            static_cast<std::size_t>(detector));
        apt["array"](detector) = static_cast<double>(binding.array);
        apt["flag"](detector) = binding.apt_flag.has_value()
            ? static_cast<double>(*binding.apt_flag)
            : std::numeric_limits<double>::quiet_NaN();
        apt["x_t"](detector) = binding.az_offset_arcsec;
        apt["y_t"](detector) = binding.el_offset_arcsec;
        apt["uid"](detector) = static_cast<double>(binding.output_uid);
    }
    return apt;
}

Eigen::VectorXi projection_map_indices(
    const pipeline::NativeScienceProjection &projection) {
    Eigen::VectorXi result(projection.detector_count());
    for (Eigen::Index detector = 0;
         detector < projection.detector_count(); ++detector) {
        result(detector) = projection.detectors().at(
            static_cast<std::size_t>(detector)).map_index;
    }
    return result;
}

mapmaking::MapBuffer make_map(Eigen::Index map_count, bool jinc,
                              Eigen::Index n_noise = 0,
                              bool randomize_dets = true) {
    mapmaking::MapBuffer result{"omb"};
    result.n_rows = 11;
    result.n_cols = 11;
    result.pixel_size_rad = kPixelSizeRad;
    result.map_grouping = "detector";
    result.parallel_policy = "seq";
    citlali::pipeline::allocate_map_matrices(
        result, map_count, jinc, true, true, false, {}, jinc);
    result.n_noise = n_noise;
    result.randomize_dets = randomize_dets;
    for (Eigen::Index map = 0; map < map_count && n_noise > 0; ++map) {
        result.noise.emplace_back(result.n_rows, result.n_cols, n_noise);
        result.noise.back().setZero();
    }
    return result;
}

mapmaking::JincMapmaker make_jinc() {
    mapmaking::JincMapmaker result;
    result.run_polarization = false;
    result.subpixel_n = 1;
    Eigen::MatrixXd kernel(3, 3);
    kernel << 0.125, 0.25, 0.125,
              0.25, 1.0, 0.25,
              0.125, 0.25, 0.125;
    for (Eigen::Index array = 0; array < 3; ++array) {
        result.jinc_weights_mat[array] = kernel;
        result.jinc_weights_sq_mat[array] =
            kernel.array().square().matrix();
    }
    return result;
}

template <class MatrixLhs, class MatrixRhs>
void expect_matrix_exact(const MatrixLhs &lhs, const MatrixRhs &rhs) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
        for (Eigen::Index column = 0; column < lhs.cols(); ++column) {
            if constexpr (std::is_floating_point_v<typename MatrixLhs::Scalar>) {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(
                              static_cast<double>(lhs(row, column))),
                          std::bit_cast<std::uint64_t>(
                              static_cast<double>(rhs(row, column))));
            }
            else {
                EXPECT_EQ(lhs(row, column), rhs(row, column));
            }
        }
    }
}

void expect_map_exact(const mapmaking::MapBuffer &lhs,
                      const mapmaking::MapBuffer &rhs, bool jinc) {
    ASSERT_EQ(lhs.signal.size(), rhs.signal.size());
    for (std::size_t map = 0; map < lhs.signal.size(); ++map) {
        expect_matrix_exact(lhs.signal[map], rhs.signal[map]);
        expect_matrix_exact(lhs.weight[map], rhs.weight[map]);
        expect_matrix_exact(lhs.kernel[map], rhs.kernel[map]);
        expect_matrix_exact(lhs.coverage[map], rhs.coverage[map]);
        if (jinc) {
            expect_matrix_exact(lhs.grid_weight[map], rhs.grid_weight[map]);
            expect_matrix_exact(
                lhs.jinc_products.denominator_sum_abs[map],
                rhs.jinc_products.denominator_sum_abs[map]);
            expect_matrix_exact(
                lhs.jinc_products.contributor_count[map],
                rhs.jinc_products.contributor_count[map]);
        }
    }
    ASSERT_EQ(lhs.noise.size(), rhs.noise.size());
    for (std::size_t map = 0; map < lhs.noise.size(); ++map) {
        ASSERT_EQ(lhs.noise[map].dimensions(), rhs.noise[map].dimensions());
        for (Eigen::Index realization = 0;
             realization < lhs.noise[map].dimension(2); ++realization) {
            for (Eigen::Index column = 0;
                 column < lhs.noise[map].dimension(1); ++column) {
                for (Eigen::Index row = 0;
                     row < lhs.noise[map].dimension(0); ++row) {
                    EXPECT_EQ(
                        std::bit_cast<std::uint64_t>(
                            lhs.noise[map](row, column, realization)),
                        std::bit_cast<std::uint64_t>(
                            rhs.noise[map](row, column, realization)));
                }
            }
        }
    }
}

std::uint64_t map_checksum(const mapmaking::MapBuffer &map, bool jinc) {
    std::uint64_t result = UINT64_C(1469598103934665603);
    const auto consume = [&](const auto &matrix) {
        for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
            for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
                const auto bits = [&]() -> std::uint64_t {
                    if constexpr (std::is_floating_point_v<
                                      typename std::decay_t<decltype(matrix)>::Scalar>) {
                        return std::bit_cast<std::uint64_t>(
                            static_cast<double>(matrix(row, column)));
                    }
                    return static_cast<std::uint64_t>(
                        matrix(row, column));
                }();
                result ^= bits;
                result *= UINT64_C(1099511628211);
            }
        }
    };
    for (std::size_t index = 0; index < map.signal.size(); ++index) {
        consume(map.signal[index]);
        consume(map.weight[index]);
        consume(map.kernel[index]);
        consume(map.coverage[index]);
        if (jinc) {
            consume(map.grid_weight[index]);
            consume(map.jinc_products.denominator_sum_abs[index]);
            consume(map.jinc_products.contributor_count[index]);
        }
    }
    return result;
}

TEST(sci_align_native_science_projection,
     exact_cell_pointing_is_shared_by_mask_kernel_variance_and_map) {
    const auto committed = make_committed_projection(
        fixture::load_native_gap_fixture_v1());
    const auto &projection = committed.projection;
    ASSERT_EQ(projection.row_count(), 4);
    ASSERT_EQ(projection.detector_count(), 4);
    const auto all_measured = projection.map_center_source_mask(1.0e9);
    for (Eigen::Index row = 0; row < projection.row_count(); ++row) {
        for (Eigen::Index detector = 0;
             detector < projection.detector_count(); ++detector) {
            const auto &cell = projection.cell(row, detector);
            EXPECT_EQ(all_measured(row, detector), cell.projects());
            EXPECT_DOUBLE_EQ(
                projection.latitudes_rad()(row, detector),
                cell.latitude_rad);
            EXPECT_DOUBLE_EQ(
                projection.longitudes_rad()(row, detector),
                cell.longitude_rad);
            EXPECT_EQ(projection.flags()(row, detector), !cell.projects());
        }
    }
    // The same relational row retains distinct exact network-native times.
    EXPECT_NE(
        projection.cell(0, 0).identity.reconstructed_time_unix_sec(),
        projection.cell(0, 1).identity.reconstructed_time_unix_sec());
    EXPECT_NE(projection.latitudes_rad()(0, 0),
              projection.latitudes_rad()(0, 1));
}

TEST(sci_align_native_science_projection,
     identical_times_match_existing_naive_and_jinc_arithmetic_exactly) {
    const auto committed = make_committed_projection(
        complete_identical_time_fixture());
    const auto &projection = committed.projection;
    auto apt = projection_apt(projection);
    auto map_indices = projection_map_indices(projection);
    std::string pixel_axes = "altaz";

    auto native_input = projection_input(projection);
    auto legacy_input = projection_input(projection, true);
    auto native_naive = make_map(projection.detector_count(), false);
    auto legacy_naive = make_map(projection.detector_count(), false);
    mapmaking::MapBuffer empty_native{"cmb"};
    mapmaking::MapBuffer empty_legacy{"cmb"};
    mapmaking::NaiveMapmaker naive;
    naive.run_polarization = false;
    naive.populate_maps_naive_native(
        native_input, native_naive, empty_native, map_indices,
        pixel_axes, apt, kSampleRateHz, true, false, projection);
    naive.populate_maps_naive_science_contract(
        legacy_input, legacy_naive, empty_legacy, map_indices,
        pixel_axes, apt, kSampleRateHz, true, false);
    expect_map_exact(native_naive, legacy_naive, false);
    EXPECT_EQ(map_checksum(native_naive, false),
              map_checksum(legacy_naive, false));
    EXPECT_EQ(map_checksum(native_naive, false),
              UINT64_C(8052882556844240840));

    native_input = projection_input(projection);
    legacy_input = projection_input(projection, true);
    auto native_jinc = make_map(projection.detector_count(), true);
    auto legacy_jinc = make_map(projection.detector_count(), true);
    auto native_maker = make_jinc();
    auto legacy_maker = make_jinc();
    native_maker.populate_maps_jinc_parallel_native(
        native_input, native_jinc, empty_native, map_indices,
        pixel_axes, apt, kSampleRateHz, true, false, projection);
    legacy_maker.populate_maps_jinc_parallel(
        legacy_input, legacy_jinc, empty_legacy, map_indices,
        pixel_axes, apt, kSampleRateHz, true, false);
    expect_map_exact(native_jinc, legacy_jinc, true);
    EXPECT_EQ(map_checksum(native_jinc, true),
              map_checksum(legacy_jinc, true));
    EXPECT_EQ(map_checksum(native_jinc, true),
              UINT64_C(4269599267376700904));
}

TEST(sci_align_native_science_projection,
     invalid_cells_never_project_in_naive_or_jinc) {
    const auto committed = make_committed_projection(
        fixture::load_native_gap_fixture_v1());
    const auto &projection = committed.projection;
    auto apt = projection_apt(projection);
    auto map_indices = projection_map_indices(projection);
    std::string pixel_axes = "altaz";
    mapmaking::MapBuffer empty{"cmb"};

    auto naive_input = projection_input(projection);
    auto naive_map = make_map(projection.detector_count(), false);
    mapmaking::NaiveMapmaker naive;
    naive.run_polarization = false;
    naive.populate_maps_naive_native(
        naive_input, naive_map, empty, map_indices, pixel_axes, apt,
        kSampleRateHz, true, false, projection);

    auto jinc_input = projection_input(projection);
    auto jinc_map = make_map(projection.detector_count(), true);
    auto jinc = make_jinc();
    jinc.populate_maps_jinc_parallel_native(
        jinc_input, jinc_map, empty, map_indices, pixel_axes, apt,
        kSampleRateHz, true, false, projection);
    for (Eigen::Index detector = 0;
         detector < projection.detector_count(); ++detector) {
        const auto valid = static_cast<Eigen::Index>(
            (projection.flags().col(detector).array() == false).count());
        const double naive_weight = naive_map.weight.at(
            static_cast<std::size_t>(detector)).sum();
        const auto jinc_contributors = jinc_map.jinc_products
            .contributor_count.at(static_cast<std::size_t>(detector)).sum();
        EXPECT_DOUBLE_EQ(
            naive_weight,
            valid * projection_input(projection).weights.data(detector));
        EXPECT_EQ(jinc_contributors, valid * 9);
    }
}

TEST(sci_align_native_science_projection,
     identical_times_match_existing_jinc_noise_arithmetic_exactly) {
    const auto committed = make_committed_projection(
        complete_identical_time_fixture());
    const auto &projection = committed.projection;
    auto apt = projection_apt(projection);
    auto map_indices = projection_map_indices(projection);
    std::string pixel_axes = "altaz";
    auto native_input = projection_input(projection);
    native_input.noise.data.resize(2, projection.detector_count());
    native_input.noise.data << 1, -1, 1, -1,
                              -1, -1, 1, 1;
    auto legacy_input = projection_input(projection, true);
    legacy_input.noise.data = native_input.noise.data;
    auto native_map = make_map(projection.detector_count(), true, 2, true);
    auto legacy_map = make_map(projection.detector_count(), true, 2, true);
    mapmaking::MapBuffer empty_native{"cmb"};
    mapmaking::MapBuffer empty_legacy{"cmb"};
    auto native_maker = make_jinc();
    auto legacy_maker = make_jinc();

    native_maker.populate_maps_jinc_parallel_native(
        native_input, native_map, empty_native, map_indices,
        pixel_axes, apt, kSampleRateHz, true, true, projection);
    legacy_maker.populate_maps_jinc_parallel(
        legacy_input, legacy_map, empty_legacy, map_indices,
        pixel_axes, apt, kSampleRateHz, true, true);
    expect_map_exact(native_map, legacy_map, true);
}

TEST(sci_align_native_science_projection,
     stale_foreign_incomplete_duplicate_and_nonfinite_candidates_fail_closed) {
    const auto loaded = complete_identical_time_fixture();
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto rtc = identity_rtc(*scan);
    pipeline::NativeMeasuredDetectorLedger uncommitted{scan};
    const auto uncommitted_prepared =
        pipeline::prepare_native_ptc_cohorts(
            uncommitted, rtc, ptc_request());
    EXPECT_THROW(
        pipeline::make_native_science_projection(
            uncommitted, uncommitted_prepared,
            projection_request(*scan)),
        std::logic_error);

    auto committed = make_committed_projection(loaded);
    auto foreign = make_committed_projection(loaded);
    EXPECT_THROW(
        pipeline::make_native_science_projection(
            committed.ledger, foreign.prepared,
            projection_request(*committed.scan)),
        std::logic_error);

    auto incomplete = projection_request(*committed.scan);
    incomplete.detectors.pop_back();
    EXPECT_THROW(
        pipeline::make_native_science_projection(
            committed.ledger, committed.prepared, incomplete),
        std::invalid_argument);
    auto duplicate = projection_request(*committed.scan);
    duplicate.detectors[1].detector_column = 0;
    EXPECT_THROW(
        pipeline::make_native_science_projection(
            committed.ledger, committed.prepared, duplicate),
        std::invalid_argument);
    auto nonfinite = projection_request(*committed.scan);
    nonfinite.detectors[0].az_offset_arcsec =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        pipeline::make_native_science_projection(
            committed.ledger, committed.prepared, nonfinite),
        std::invalid_argument);
    auto unknown_grouping = projection_request(*committed.scan);
    unknown_grouping.map_grouping = "unknown";
    EXPECT_THROW(
        pipeline::make_native_science_projection(
            committed.ledger, committed.prepared, unknown_grouping),
        std::invalid_argument);
    auto unresolved_grouping = projection_request(*committed.scan);
    unresolved_grouping.map_grouping = "auto";
    EXPECT_THROW(
        pipeline::make_native_science_projection(
            committed.ledger, committed.prepared, unresolved_grouping),
        std::invalid_argument);

    auto input = projection_input(committed.projection);
    auto apt = projection_apt(committed.projection);
    auto map_indices = projection_map_indices(committed.projection);
    std::string pixel_axes = "altaz";
    auto map = make_map(committed.projection.detector_count(), false);
    const auto before = map_checksum(map, false);
    input.scans.data(0, 0) += 1.0;
    mapmaking::MapBuffer empty{"cmb"};
    mapmaking::NaiveMapmaker naive;
    naive.run_polarization = false;
    EXPECT_THROW(
        naive.populate_maps_naive_native(
            input, map, empty, map_indices, pixel_axes, apt,
            kSampleRateHz, true, false, committed.projection),
        std::logic_error);
    EXPECT_EQ(map_checksum(map, false), before);

    input = projection_input(committed.projection);
    input.scans.data(0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        naive.populate_maps_naive_native(
            input, map, empty, map_indices, pixel_axes, apt,
            kSampleRateHz, true, false, committed.projection),
        std::logic_error);
    EXPECT_EQ(map_checksum(map, false), before);

    input = projection_input(committed.projection);
    input.flags.data(0, 0) = !input.flags.data(0, 0);
    EXPECT_THROW(
        naive.populate_maps_naive_native(
            input, map, empty, map_indices, pixel_axes, apt,
            kSampleRateHz, true, false, committed.projection),
        std::logic_error);
    EXPECT_EQ(map_checksum(map, false), before);

    // The native bridge does not replace JINC's unique map-owner authority.
    auto duplicate_owner_request = projection_request(*committed.scan);
    duplicate_owner_request.detectors[1].map_index =
        duplicate_owner_request.detectors[0].map_index;
    const auto duplicate_owner =
        pipeline::make_native_science_projection(
            committed.ledger, committed.prepared,
            std::move(duplicate_owner_request));
    auto jinc_input = projection_input(duplicate_owner);
    auto jinc_apt = projection_apt(duplicate_owner);
    auto duplicate_indices = projection_map_indices(duplicate_owner);
    auto jinc_map = make_map(duplicate_owner.detector_count(), true);
    const auto jinc_before = map_checksum(jinc_map, true);
    auto jinc = make_jinc();
    EXPECT_THROW(
        jinc.populate_maps_jinc_parallel_native(
            jinc_input, jinc_map, empty, duplicate_indices, pixel_axes,
            jinc_apt, kSampleRateHz, true, false, duplicate_owner),
        std::runtime_error);
    EXPECT_EQ(map_checksum(jinc_map, true), jinc_before);

    // A later issued operation makes the committed Stage 5 snapshot stale.
    (void)committed.ledger.issue_operation();
    EXPECT_THROW(
        pipeline::make_native_science_projection(
            committed.ledger, committed.prepared,
            projection_request(*committed.scan)),
        std::logic_error);
}

TEST(sci_align_native_science_projection,
     typed_null_offsets_are_resolved_only_for_excluded_detectors) {
    const auto missing = std::numeric_limits<double>::quiet_NaN();
    EXPECT_DOUBLE_EQ(
        pipeline::native_science_projection_detail::
            resolve_detector_offset_arcsec(12.5, std::int64_t{0}),
        12.5);
    EXPECT_DOUBLE_EQ(
        pipeline::native_science_projection_detail::
            resolve_detector_offset_arcsec(12.5, std::nullopt),
        12.5);
    EXPECT_DOUBLE_EQ(
        pipeline::native_science_projection_detail::
            resolve_detector_offset_arcsec(missing, std::int64_t{1}),
        0.0);
    EXPECT_DOUBLE_EQ(
        pipeline::native_science_projection_detail::
            resolve_detector_offset_arcsec(missing, std::nullopt),
        0.0);
    EXPECT_THROW(
        pipeline::native_science_projection_detail::
            resolve_detector_offset_arcsec(missing, std::int64_t{0}),
        std::logic_error);
}

TEST(sci_align_native_science_projection,
     typed_detector_request_permutation_is_exactly_invariant) {
    auto committed = make_committed_projection(
        complete_identical_time_fixture());
    auto permuted = projection_request(*committed.scan);
    std::reverse(permuted.detectors.begin(), permuted.detectors.end());
    const auto second = pipeline::make_native_science_projection(
        committed.ledger, committed.prepared, std::move(permuted));
    expect_matrix_exact(committed.projection.values(), second.values());
    expect_matrix_exact(committed.projection.flags(), second.flags());
    expect_matrix_exact(committed.projection.latitudes_rad(),
                        second.latitudes_rad());
    expect_matrix_exact(committed.projection.longitudes_rad(),
                        second.longitudes_rad());
    EXPECT_EQ(committed.projection.detectors(), second.detectors());
}

TEST(sci_align_native_science_projection,
     parallel_jinc_result_is_exact_at_openmp_thread_counts_1_2_4_8) {
    const auto committed = make_committed_projection(
        complete_identical_time_fixture());
    auto apt = projection_apt(committed.projection);
    auto map_indices = projection_map_indices(committed.projection);
    std::string pixel_axes = "altaz";
    std::optional<std::uint64_t> reference;
    for (const int threads : std::array<int, 4>{1, 2, 4, 8}) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        auto input = projection_input(committed.projection);
        auto map = make_map(committed.projection.detector_count(), true);
        mapmaking::MapBuffer empty{"cmb"};
        auto jinc = make_jinc();
        jinc.populate_maps_jinc_parallel_native(
            input, map, empty, map_indices, pixel_axes, apt,
            kSampleRateHz, true, false, committed.projection);
        const auto checksum = map_checksum(map, true);
        if (!reference.has_value()) reference = checksum;
        EXPECT_EQ(checksum, *reference);
    }
}

}  // namespace

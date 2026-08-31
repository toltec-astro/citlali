#include <citlali/core/pipeline/rtc_filter_d2_measurement.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;

Eigen::VectorXd vector(std::initializer_list<double> values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    Eigen::Index index = 0;
    for (const auto value : values) result(index++) = value;
    return result;
}

std::shared_ptr<const pipeline::PairedReadoutOccurrenceAxis> axis(
    pipeline::TimestreamNetworkId network,
    pipeline::TimestreamNativeRow first_row,
    std::initializer_list<double> times,
    std::vector<pipeline::TimestreamPacketCounter> counters) {
    auto timing = std::make_shared<const pipeline::NativeNetworkAlignment>(
        network, first_row, vector(times), std::move(counters));
    std::vector<pipeline::NativeOccurrenceInterval> intervals;
    for (const auto time : times) {
        intervals.push_back({time - 0.004, time + 0.004});
    }
    return std::make_shared<const pipeline::PairedReadoutOccurrenceAxis>(
        std::move(timing), first_row, std::move(intervals));
}

pipeline::PairedReadoutNetwork network(
    pipeline::TimestreamNetworkId network_id,
    pipeline::TimestreamNativeRow first_row,
    std::initializer_list<double> times,
    std::vector<pipeline::TimestreamPacketCounter> counters,
    std::int64_t uid_base, double base,
    std::optional<std::size_t> r_invalid = {}) {
    const auto rows = static_cast<Eigen::Index>(times.size());
    constexpr Eigen::Index columns = 2;
    pipeline::PairedReadoutMatrix x(rows, columns);
    pipeline::PairedReadoutMatrix r(rows, columns);
    for (Eigen::Index row = 0; row < rows; ++row) {
        for (Eigen::Index column = 0; column < columns; ++column) {
            x(row, column) = base + 10.0 * row + column;
            r(row, column) = base + 100.0 + 10.0 * row + column;
        }
    }
    std::vector<pipeline::ReadoutMemberState> x_state(
        static_cast<std::size_t>(rows * columns),
        pipeline::ReadoutMemberState::measured(true, true, true, true));
    auto r_state = x_state;
    if (r_invalid) {
        r_state.at(*r_invalid) =
            pipeline::ReadoutMemberState::measured(
                true, false, true, true);
    }
    std::vector<pipeline::PairedReadoutDetectorIdentity> detectors;
    for (Eigen::Index column = 0; column < columns; ++column) {
        detectors.push_back({uid_base + column,
                             network_id < 7 ? 0 : 1,
                             network_id, 1000 + network_id, column});
    }
    auto mapping =
        std::make_shared<const pipeline::NativeReadoutMappingIdentity>(
            pipeline::NativeReadoutMappingIdentity{
                "producer", "instance-" + std::to_string(network_id),
                "tune", "revision", "transform", "raw-x", "raw-r"});
    return pipeline::PairedReadoutNetwork::admit(
        axis(network_id, first_row, times, std::move(counters)),
        std::move(detectors), std::move(mapping), std::move(x),
        std::move(r), std::move(x_state), std::move(r_state));
}

std::shared_ptr<const pipeline::PairedReadout> fixture() {
    std::vector<pipeline::PairedReadoutNetwork> networks;
    networks.push_back(network(
        0, 10, {1000.0000, 1000.0100, 1000.0200, 1000.0300},
        {100, 101, 102, 103}, 100, 1.0, 3));
    networks.push_back(network(
        7, 70, {1000.0025, 1000.0125, 1000.0325},
        {700, 701, 703}, 700, 11.0));
    return pipeline::PairedReadout::admit(
        {152390, 0, 2}, {0, 7}, std::move(networks));
}

std::shared_ptr<const pipeline::RtcFilterD2SourceMask> source_mask(
    const std::shared_ptr<const pipeline::PairedReadout> &paired,
    pipeline::TimestreamNetworkId network_id,
    pipeline::RtcFilterD2SourceMaskDisposition disposition =
        pipeline::RtcFilterD2SourceMaskDisposition::applied) {
    const auto &network = paired->network(network_id);
    std::vector<std::uint8_t> values(network.cell_count(), 0U);
    if (disposition ==
        pipeline::RtcFilterD2SourceMaskDisposition::applied) {
        values.front() = 1U;
    }
    return pipeline::RtcFilterD2SourceMask::admit(
        network.occurrence_axis_handle(), network.detector_count(),
        "route-source-mask-v1", disposition, std::move(values));
}

std::shared_ptr<const pipeline::RtcFilterD2LineMask> pending_lines() {
    return pipeline::RtcFilterD2LineMask::admit(
        "established-line-mask-v1",
        pipeline::RtcFilterD2LineMaskDisposition::pending, {});
}

TEST(rtc_filter_d2_measurement,
     prefilter_is_zero_copy_and_preserves_independent_network_axes) {
    const auto paired = fixture();
    const auto nw0 = pipeline::RtcFilterD2NetworkPlane::observe_prefilter(
        paired, 0, pipeline::ReadoutMember::x, source_mask(paired, 0),
        pending_lines());
    const auto nw7 = pipeline::RtcFilterD2NetworkPlane::observe_prefilter(
        paired, 7, pipeline::ReadoutMember::x, source_mask(paired, 7),
        pending_lines());

    EXPECT_EQ(nw0->contiguous_values().data(),
              paired->network(0)
                  .contiguous_values(pipeline::ReadoutMember::x)
                  .data());
    EXPECT_EQ(nw7->contiguous_values().data(),
              paired->network(7)
                  .contiguous_values(pipeline::ReadoutMember::x)
                  .data());
    EXPECT_DOUBLE_EQ(
        nw0->occurrence_axis_handle()->identity(10)
            .reconstructed_time_unix_sec(),
        1000.0000);
    EXPECT_DOUBLE_EQ(
        nw7->occurrence_axis_handle()->identity(70)
            .reconstructed_time_unix_sec(),
        1000.0025);
    EXPECT_NE(nw0->occurrence_axis_handle().get(),
              nw7->occurrence_axis_handle().get());
    EXPECT_EQ(nw0->memory_evidence().owned_numeric_bytes, 0U);
    EXPECT_EQ(nw0->memory_evidence().referenced_native_axis_count, 1U);
    EXPECT_EQ(nw0->signal_unit_id(), "raw-x");
}

TEST(rtc_filter_d2_measurement,
     pair_wide_validity_preserves_member_local_input_causes) {
    const auto paired = fixture();
    const auto observed =
        pipeline::RtcFilterD2NetworkPlane::observe_prefilter(
            paired, 0, pipeline::ReadoutMember::x,
            source_mask(paired, 0), pending_lines());

    EXPECT_TRUE(paired->network(0)
                    .state(pipeline::ReadoutMember::x, 11, 1)
                    .valid());
    EXPECT_FALSE(paired->network(0)
                     .state(pipeline::ReadoutMember::r, 11, 1)
                     .valid());
    EXPECT_FALSE(observed->valid(11, 1));
    EXPECT_EQ(paired->network(0)
                  .state(pipeline::ReadoutMember::x, 11, 1)
                  .causes(),
              pipeline::ReadoutMemberCause::none);
    EXPECT_TRUE(pipeline::has_cause(
        paired->network(0).pair_causes(11, 1),
        pipeline::PairedReadoutCause::r_original_invalid));
}

TEST(rtc_filter_d2_measurement,
     residual_owns_only_derived_plane_and_shares_exact_native_facts) {
    const auto paired = fixture();
    const auto prefilter =
        pipeline::RtcFilterD2NetworkPlane::observe_prefilter(
            paired, 7, pipeline::ReadoutMember::x,
            source_mask(paired, 7), pending_lines());
    auto residual = prefilter->values();
    residual.array() -= 4.0;
    std::vector<std::uint8_t> valid(
        static_cast<std::size_t>(residual.size()), 1U);
    valid[2] = 0U;
    const auto observed =
        pipeline::RtcFilterD2NetworkPlane::observe_post_cleaning_residual(
            prefilter, std::move(residual), std::move(valid),
            {"established-ptc-standard-pca-v1", "config:sha256:test", "nw",
             false, false});

    EXPECT_EQ(observed->prefilter_handle(), prefilter);
    EXPECT_EQ(observed->paired_handle(), paired);
    EXPECT_EQ(observed->occurrence_axis_handle(),
              prefilter->occurrence_axis_handle());
    EXPECT_EQ(observed->source_mask_handle(),
              prefilter->source_mask_handle());
    EXPECT_EQ(observed->line_mask_handle(), prefilter->line_mask_handle());
    EXPECT_EQ(observed->cleaning_realization().grouping, "nw");
    EXPECT_DOUBLE_EQ(observed->values()(0, 0), 7.0);
    EXPECT_FALSE(observed->valid(71, 0));
    EXPECT_TRUE(observed->valid(70, 1));
    EXPECT_EQ(observed->memory_evidence().owned_numeric_bytes,
              static_cast<std::size_t>(observed->values().size()) *
                  sizeof(double));
    EXPECT_EQ(observed->memory_evidence().owned_residual_validity_bytes,
              static_cast<std::size_t>(observed->values().size()));
}

TEST(rtc_filter_d2_measurement,
     a_gap_remains_network_local_and_does_not_manufacture_another_axis_row) {
    const auto paired = fixture();
    const auto nw0 = pipeline::RtcFilterD2NetworkPlane::observe_prefilter(
        paired, 0, pipeline::ReadoutMember::x, source_mask(paired, 0),
        pending_lines());
    const auto nw7 = pipeline::RtcFilterD2NetworkPlane::observe_prefilter(
        paired, 7, pipeline::ReadoutMember::x, source_mask(paired, 7),
        pending_lines());

    ASSERT_EQ(nw7->physical_runs().size(), 2U);
    ASSERT_EQ(nw0->physical_runs().size(), 1U);
    EXPECT_EQ(nw7->occurrence_count(), 3);
    EXPECT_EQ(nw0->occurrence_count(), 4);
    EXPECT_EQ(nw7->occurrence_axis_handle()->identity(72).packet_counter(),
              703U);
    EXPECT_EQ(nw0->occurrence_axis_handle()->past_last_native_row(), 14);
}

TEST(rtc_filter_d2_measurement,
     rejects_foreign_axes_sampling_changes_and_false_line_operator_claims) {
    const auto paired = fixture();
    EXPECT_THROW(
        pipeline::RtcFilterD2NetworkPlane::observe_prefilter(
            paired, 0, pipeline::ReadoutMember::x,
            source_mask(paired, 7), pending_lines()),
        std::invalid_argument);

    const auto prefilter =
        pipeline::RtcFilterD2NetworkPlane::observe_prefilter(
            paired, 0, pipeline::ReadoutMember::x,
            source_mask(paired, 0), pending_lines());
    pipeline::PairedReadoutMatrix wrong_rows(
        prefilter->occurrence_count() - 1, prefilter->detector_count());
    wrong_rows.setZero();
    EXPECT_THROW(
        pipeline::RtcFilterD2NetworkPlane::observe_post_cleaning_residual(
            prefilter, std::move(wrong_rows),
            std::vector<std::uint8_t>(6, 1U),
            {"clean", "config", "nw", false, false}),
        std::invalid_argument);

    auto values = prefilter->values();
    EXPECT_THROW(
        pipeline::RtcFilterD2NetworkPlane::observe_post_cleaning_residual(
            prefilter, std::move(values),
            std::vector<std::uint8_t>(prefilter->network().cell_count(), 1U),
            {"clean", "config", "nw", true, false}),
        std::invalid_argument);

    EXPECT_THROW(
        pipeline::RtcFilterD2LineMask::admit(
            "lines", pipeline::RtcFilterD2LineMaskDisposition::applied,
            {{"line", 10.0, 11.0, true, ""}}),
        std::invalid_argument);
}

}  // namespace

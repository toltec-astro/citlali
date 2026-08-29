#include <citlali/core/pipeline/common_analysis_grid_paired_readout.h>
#include <citlali/core/pipeline/network_timed_rtc_pass_through.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;

Eigen::VectorXd vector(std::initializer_list<double> values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    Eigen::Index index = 0;
    for (const auto value : values)
        result(index++) = value;
    return result;
}

std::shared_ptr<const pipeline::NativeReadoutMappingIdentity>
mapping(pipeline::TimestreamNetworkId network_id) {
    const auto suffix = std::to_string(network_id);
    return std::make_shared<const pipeline::NativeReadoutMappingIdentity>(
        pipeline::NativeReadoutMappingIdentity{
            "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1",
            "producer:network-timed:" + suffix, "tune:network-timed:" + suffix,
            "mapping:network-timed:" + suffix,
            "iq-to-xr:network-timed:" + suffix, "raw-x:network-timed:" + suffix,
            "raw-r:network-timed:" + suffix});
}

pipeline::PairedReadoutNetwork
network(pipeline::TimestreamNetworkId network_id,
        pipeline::TimestreamNativeRow first_native_row,
        std::initializer_list<double> times,
        std::vector<pipeline::TimestreamPacketCounter> counters,
        std::int64_t detector_uid, double x_base, double r_base,
        std::optional<std::size_t> x_invalid_offset = std::nullopt,
        std::optional<std::size_t> r_invalid_offset = std::nullopt) {
    auto timing = std::make_shared<const pipeline::NativeNetworkAlignment>(
        network_id, first_native_row, vector(times), std::move(counters));
    std::vector<pipeline::NativeOccurrenceInterval> intervals;
    intervals.reserve(times.size());
    for (const auto time : times) {
        intervals.push_back({time - 0.004, time + 0.004});
    }
    auto axis = std::make_shared<const pipeline::PairedReadoutOccurrenceAxis>(
        std::move(timing), first_native_row, std::move(intervals));
    std::vector<pipeline::PairedReadoutDetectorIdentity> detectors{
        {detector_uid, network_id == 0 ? 0 : 1, network_id, 1000 + network_id,
         0}};
    pipeline::PairedReadoutMatrix x(static_cast<Eigen::Index>(times.size()), 1);
    pipeline::PairedReadoutMatrix r(static_cast<Eigen::Index>(times.size()), 1);
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        x(row, 0) = x_base + static_cast<double>(row);
        r(row, 0) = r_base + static_cast<double>(row);
    }
    const auto valid =
        pipeline::ReadoutMemberState::measured(true, true, true, true);
    const auto invalid =
        pipeline::ReadoutMemberState::measured(true, false, true, true);
    std::vector<pipeline::ReadoutMemberState> x_states(times.size(), valid);
    std::vector<pipeline::ReadoutMemberState> r_states(times.size(), valid);
    if (x_invalid_offset)
        x_states.at(*x_invalid_offset) = invalid;
    if (r_invalid_offset)
        r_states.at(*r_invalid_offset) = invalid;
    return pipeline::PairedReadoutNetwork::admit(
        std::move(axis), std::move(detectors), mapping(network_id),
        std::move(x), std::move(r), std::move(x_states), std::move(r_states));
}

struct NetworkTimedRtcFixture {
    std::shared_ptr<const pipeline::PairedReadout> native;
    std::shared_ptr<const pipeline::NativePairedReadoutView> logical;
    std::vector<std::shared_ptr<const pipeline::NativePairedReadoutView>>
        split_partitions;
};

NetworkTimedRtcFixture fixture() {
    const pipeline::NativeObservationScope scope{152390, 0, 4};
    std::vector<pipeline::PairedReadoutNetwork> networks;
    // r-only evidence at nw0 row 12.
    networks.push_back(
        network(0, 10, {1000.0000, 1000.0100, 1000.0200, 1000.0300},
                {100, 101, 102, 103}, 500, 1.0, 101.0, std::nullopt, 2));
    // nw7 is independently timed and has a delivered counter discontinuity;
    // x-only evidence is at row 71.
    networks.push_back(network(7, 70, {1000.0025, 1000.0125, 1000.0325},
                               {700, 701, 703}, 700, 11.0, 111.0, 1,
                               std::nullopt));
    auto native =
        pipeline::PairedReadout::admit(scope, {0, 7}, std::move(networks));
    auto logical = pipeline::NativePairedReadoutView::full(native);
    std::vector<std::shared_ptr<const pipeline::NativePairedReadoutView>>
        split_partitions;
    split_partitions.push_back(pipeline::NativePairedReadoutView::admit(
        native, {{0, 10, 12}, {7, 70, 71}}));
    split_partitions.push_back(pipeline::NativePairedReadoutView::admit(
        native, {{0, 12, 14}, {7, 71, 73}}));
    return {std::move(native), std::move(logical), std::move(split_partitions)};
}

std::shared_ptr<const pipeline::RtcApplicationContext>
context(const NetworkTimedRtcFixture &data, bool conditioned_r_requested = true,
        std::uint64_t request = 7) {
    return pipeline::RtcApplicationContext::admit(
        {request}, data.logical, pipeline::RtcApplicationUse::rtc_terminal,
        "native-network-occurrences:nw0+n7",
        {"positive-away-from-reference", "readout-reference:x",
         "raw-baseline:x"},
        {"positive-orthogonal-to-x", "readout-reference:r", "raw-baseline:r"},
        "finite-raw-paired-domain", {"rtc-terminal"},
        {"SCI-RTC-network-timing-owner-decision", "identity-no-filter-M1"},
        conditioned_r_requested);
}

std::shared_ptr<const pipeline::NetworkTimedRtcPlan>
plan(const std::shared_ptr<const pipeline::RtcApplicationContext> &request,
     std::span<const std::shared_ptr<const pipeline::NativePairedReadoutView>>
         partitions,
     std::uint64_t attempt = 11, std::uint64_t resolution = 13) {
    return pipeline::resolve_network_timed_rtc_pass_through(
        pipeline::learn_network_timed_rtc_pass_through(request, partitions,
                                                       attempt),
        resolution);
}

TEST(network_timed_rtc_pass_through,
     context_and_evidence_reference_native_facts_without_duplication) {
    const auto data = fixture();
    const auto request = context(data);
    const auto evidence = pipeline::learn_network_timed_rtc_pass_through(
        request, data.split_partitions, 11);

    EXPECT_EQ(request->input_handle(), data.logical);
    EXPECT_EQ(request->input_handle()->parent_handle(), data.native);
    EXPECT_EQ(request->identity().request, 7U);
    EXPECT_EQ(request->interval_id(), "native-network-occurrences:nw0+n7");
    EXPECT_EQ(evidence->context_handle(), request);
    EXPECT_EQ(evidence->identity_evidence_handle()->input_handle(),
              data.logical);
    EXPECT_EQ(evidence->summary().examined_cell_count, 7U);
    EXPECT_EQ(evidence->summary().accepted_event_count, 2U);
    EXPECT_EQ(evidence->summary().direct_x_event_count, 1U);
    EXPECT_EQ(evidence->summary().direct_r_event_count, 1U);
    EXPECT_EQ(evidence->summary().x_and_r_event_count, 0U);
    EXPECT_EQ(evidence->memory_evidence().derived_event_bytes,
              2U * sizeof(pipeline::RtcEvidenceEvent));

    const auto *from_r = evidence->find(0, 12, 0);
    ASSERT_NE(from_r, nullptr);
    EXPECT_FALSE(from_r->direct_x());
    EXPECT_TRUE(from_r->direct_r());
    EXPECT_EQ(evidence->scientific_identity(*from_r),
              (pipeline::RtcNativeCellIdentity{0, 12, 500}));
    EXPECT_TRUE(
        pipeline::has_cause(evidence->member_local_causes(*from_r),
                            pipeline::PairedReadoutCause::r_original_invalid));

    const auto *from_x = evidence->find(7, 71, 0);
    ASSERT_NE(from_x, nullptr);
    EXPECT_TRUE(from_x->direct_x());
    EXPECT_FALSE(from_x->direct_r());
    EXPECT_EQ(evidence->scientific_identity(*from_x),
              (pipeline::RtcNativeCellIdentity{7, 71, 700}));
}

TEST(network_timed_rtc_pass_through,
     m1_preserves_distinct_network_times_identities_and_values_exactly) {
    const auto data = fixture();
    const auto request = context(data);
    const auto resolved = plan(request, data.split_partitions);
    const auto applied = pipeline::apply_network_timed_rtc_pass_through(
        resolved, data.split_partitions);

    EXPECT_EQ(resolved->policy().despiking,
              pipeline::NetworkTimedRtcOperationDisposition::not_selected);
    EXPECT_EQ(resolved->policy().level_shift_correction,
              pipeline::NetworkTimedRtcOperationDisposition::not_selected);
    EXPECT_EQ(resolved->policy().donor_replacement,
              pipeline::NetworkTimedRtcOperationDisposition::not_selected);
    EXPECT_EQ(resolved->policy().temporal_filter,
              pipeline::NetworkTimedRtcOperationDisposition::identity);
    EXPECT_EQ(resolved->policy().phase_zero_sampling,
              pipeline::NetworkTimedRtcOperationDisposition::identity);
    EXPECT_FALSE(resolved->policy().coordinate_dependent_operation);

    ASSERT_NE(applied.product, nullptr);
    EXPECT_EQ(applied.product->input_handle(), data.logical);
    EXPECT_EQ(applied.product->native_parent_handle(), data.native);
    EXPECT_EQ(applied.product->network_spans().size(), 2U);
    EXPECT_EQ(applied.product->network_spans()[0],
              pipeline::NativeOccurrenceSpan({0, 10, 14}));
    EXPECT_EQ(applied.product->network_spans()[1],
              pipeline::NativeOccurrenceSpan({7, 70, 73}));
    EXPECT_EQ(applied.product->output_native_occurrence_count(), 7U);
    EXPECT_EQ(applied.product->output_cell_count(), 7U);
    EXPECT_DOUBLE_EQ(applied.product->output_time_unix_sec(0, 10), 1000.0000);
    EXPECT_DOUBLE_EQ(applied.product->output_time_unix_sec(7, 70), 1000.0025);
    EXPECT_NE(applied.product->output_time_unix_sec(0, 10),
              applied.product->output_time_unix_sec(7, 70));
    EXPECT_EQ(applied.product->identity(0, 10, 0),
              (pipeline::RtcNativeCellIdentity{0, 10, 500}));
    EXPECT_EQ(applied.product->identity(7, 70, 0),
              (pipeline::RtcNativeCellIdentity{7, 70, 700}));

    for (const auto &span : applied.product->network_spans()) {
        const auto &source = data.native->network(span.network_id);
        for (auto row = span.first_native_row; row < span.past_last_native_row;
             ++row) {
            EXPECT_EQ(applied.product->representative_native_identity(
                          span.network_id, row),
                      source.occurrence_axis_handle()->identity(row));
            EXPECT_EQ(std::bit_cast<std::uint64_t>(
                          applied.product->output_time_unix_sec(span.network_id,
                                                                row)),
                      std::bit_cast<std::uint64_t>(
                          source.occurrence_axis_handle()
                              ->identity(row)
                              .reconstructed_time_unix_sec()));
            for (const auto member :
                 {pipeline::ReadoutMember::x, pipeline::ReadoutMember::r}) {
                EXPECT_EQ(
                    std::bit_cast<std::uint64_t>(
                        applied.product->raw_parent_value(
                            member, span.network_id, row, 0)),
                    std::bit_cast<std::uint64_t>(source.value(member, row, 0)));
            }
        }
    }

    EXPECT_EQ(applied.realization.completion,
              pipeline::NetworkTimedRtcCompletionState::complete);
    EXPECT_EQ(applied.realization.engineering_partition_count, 2U);
    EXPECT_EQ(applied.realization.output_native_occurrence_count, 7U);
    EXPECT_EQ(applied.realization.output_cell_count, 7U);
    EXPECT_EQ(applied.realization.pair_ineligible_cell_count, 2U);
    EXPECT_EQ(applied.realization.x_available_cell_count, 6U);
    EXPECT_EQ(applied.realization.r_available_cell_count, 6U);
    EXPECT_EQ(applied.realization.realized_sampling_factor, 1U);
    EXPECT_TRUE(applied.realization.conditioned_r_requested);
    EXPECT_EQ(applied.product->memory_evidence().logical_owned_bytes(), 0U);
    EXPECT_LT(sizeof(pipeline::NetworkTimedRtcRealization), 160U);
}

TEST(network_timed_rtc_pass_through,
     pair_consequence_is_bidirectional_while_local_causes_remain_local) {
    const auto data = fixture();
    const auto applied = pipeline::apply_network_timed_rtc_pass_through(
        plan(context(data), data.split_partitions), data.split_partitions);

    EXPECT_EQ(applied.product->pair_disposition(0, 12, 0),
              pipeline::NetworkTimedRtcPairDisposition::ineligible);
    EXPECT_EQ(applied.product->member_availability(pipeline::ReadoutMember::x,
                                                   0, 12, 0),
              pipeline::NetworkTimedRtcMemberAvailability::available);
    EXPECT_EQ(applied.product->member_availability(pipeline::ReadoutMember::r,
                                                   0, 12, 0),
              pipeline::NetworkTimedRtcMemberAvailability::unavailable);
    EXPECT_EQ(
        applied.product->pair_cause_role(pipeline::ReadoutMember::x, 0, 12, 0),
        pipeline::NetworkTimedRtcPairCauseRole::inferred_from_r);
    EXPECT_EQ(
        applied.product->pair_cause_role(pipeline::ReadoutMember::r, 0, 12, 0),
        pipeline::NetworkTimedRtcPairCauseRole::direct);
    EXPECT_EQ(applied.product->raw_member_local_causes(
                  pipeline::ReadoutMember::x, 0, 12, 0),
              pipeline::ReadoutMemberCause::none);
    EXPECT_TRUE(
        pipeline::has_cause(applied.product->raw_member_local_causes(
                                pipeline::ReadoutMember::r, 0, 12, 0),
                            pipeline::ReadoutMemberCause::producer_invalid));
    EXPECT_TRUE(applied.product->conditioned_value(pipeline::ReadoutMember::x,
                                                   0, 12, 0));
    EXPECT_FALSE(applied.product->conditioned_value(pipeline::ReadoutMember::r,
                                                    0, 12, 0));

    EXPECT_EQ(applied.product->pair_disposition(7, 71, 0),
              pipeline::NetworkTimedRtcPairDisposition::ineligible);
    EXPECT_EQ(
        applied.product->pair_cause_role(pipeline::ReadoutMember::r, 7, 71, 0),
        pipeline::NetworkTimedRtcPairCauseRole::inferred_from_x);
    EXPECT_TRUE(applied.product->conditioned_value(pipeline::ReadoutMember::r,
                                                   7, 71, 0));
}

TEST(network_timed_rtc_pass_through,
     conditioned_r_request_changes_only_the_optional_projection) {
    const auto data = fixture();
    const auto with_r = pipeline::apply_network_timed_rtc_pass_through(
        pipeline::resolve_network_timed_rtc_pass_through(
            pipeline::learn_network_timed_rtc_pass_through(
                context(data, true, 20), 21),
            22));
    const auto without_r = pipeline::apply_network_timed_rtc_pass_through(
        pipeline::resolve_network_timed_rtc_pass_through(
            pipeline::learn_network_timed_rtc_pass_through(
                context(data, false, 30), 31),
            32));

    EXPECT_TRUE(with_r.product->conditioned_r_requested());
    EXPECT_FALSE(without_r.product->conditioned_r_requested());
    EXPECT_EQ(without_r.product->member_availability(pipeline::ReadoutMember::r,
                                                     0, 10, 0),
              pipeline::NetworkTimedRtcMemberAvailability::not_requested);
    EXPECT_FALSE(without_r.product->conditioned_value(
        pipeline::ReadoutMember::r, 0, 10, 0));
    EXPECT_THROW(without_r.product->member_availability(
                     pipeline::ReadoutMember::r, 0, 14, 0),
                 std::out_of_range);
    EXPECT_DOUBLE_EQ(without_r.product->raw_parent_value(
                         pipeline::ReadoutMember::r, 0, 10, 0),
                     101.0);
    for (const auto &span : data.logical->spans()) {
        for (auto row = span.first_native_row; row < span.past_last_native_row;
             ++row) {
            EXPECT_EQ(with_r.product->identity(span.network_id, row, 0),
                      without_r.product->identity(span.network_id, row, 0));
            EXPECT_DOUBLE_EQ(
                with_r.product->output_time_unix_sec(span.network_id, row),
                without_r.product->output_time_unix_sec(span.network_id, row));
            EXPECT_EQ(
                with_r.product->pair_disposition(span.network_id, row, 0),
                without_r.product->pair_disposition(span.network_id, row, 0));
            EXPECT_EQ(with_r.product->conditioned_value(
                          pipeline::ReadoutMember::x, span.network_id, row, 0),
                      without_r.product->conditioned_value(
                          pipeline::ReadoutMember::x, span.network_id, row, 0));
        }
    }
}

TEST(network_timed_rtc_pass_through,
     scientific_output_is_invariant_to_engineering_partitioning) {
    const auto data = fixture();
    const std::vector<std::shared_ptr<const pipeline::NativePairedReadoutView>>
        one{data.logical};
    const auto one_result = pipeline::apply_network_timed_rtc_pass_through(
        plan(context(data, true, 40), one, 41, 42), one);
    const auto split_result = pipeline::apply_network_timed_rtc_pass_through(
        plan(context(data, true, 50), data.split_partitions, 51, 52),
        data.split_partitions);

    EXPECT_EQ(one_result.realization.output_native_occurrence_count,
              split_result.realization.output_native_occurrence_count);
    EXPECT_EQ(one_result.realization.output_cell_count,
              split_result.realization.output_cell_count);
    EXPECT_EQ(one_result.realization.pair_ineligible_cell_count,
              split_result.realization.pair_ineligible_cell_count);
    EXPECT_EQ(one_result.realization.x_available_cell_count,
              split_result.realization.x_available_cell_count);
    EXPECT_EQ(one_result.realization.r_available_cell_count,
              split_result.realization.r_available_cell_count);
    for (const auto &span : data.logical->spans()) {
        for (auto row = span.first_native_row; row < span.past_last_native_row;
             ++row) {
            EXPECT_EQ(one_result.product->identity(span.network_id, row, 0),
                      split_result.product->identity(span.network_id, row, 0));
            EXPECT_DOUBLE_EQ(
                one_result.product->output_time_unix_sec(span.network_id, row),
                split_result.product->output_time_unix_sec(span.network_id,
                                                           row));
            EXPECT_EQ(
                one_result.product->pair_disposition(span.network_id, row, 0),
                split_result.product->pair_disposition(span.network_id, row,
                                                       0));
            for (const auto member :
                 {pipeline::ReadoutMember::x, pipeline::ReadoutMember::r}) {
                EXPECT_EQ(one_result.product->conditioned_value(
                              member, span.network_id, row, 0),
                          split_result.product->conditioned_value(
                              member, span.network_id, row, 0));
                EXPECT_EQ(one_result.product->pair_cause_role(
                              member, span.network_id, row, 0),
                          split_result.product->pair_cause_role(
                              member, span.network_id, row, 0));
            }
        }
    }
}

TEST(network_timed_rtc_pass_through,
     a_gap_in_one_network_does_not_manufacture_support_in_another) {
    const auto data = fixture();
    const auto applied = pipeline::apply_network_timed_rtc_pass_through(
        plan(context(data), data.split_partitions), data.split_partitions);

    EXPECT_EQ(applied.product->network_spans()[0].occurrence_count(), 4U);
    EXPECT_EQ(applied.product->network_spans()[1].occurrence_count(), 3U);
    EXPECT_DOUBLE_EQ(applied.product->output_time_unix_sec(0, 12), 1000.0200);
    EXPECT_DOUBLE_EQ(applied.product->output_time_unix_sec(7, 72), 1000.0325);
    EXPECT_THROW(applied.product->output_time_unix_sec(7, 73),
                 std::out_of_range);
    const auto discontinuity = data.native->network(7)
                                   .occurrence_axis_handle()
                                   ->native_timing_handle()
                                   ->discontinuity_between(71, 72);
    ASSERT_TRUE(discontinuity);
    EXPECT_EQ(discontinuity->before_counter, 701);
    EXPECT_EQ(discontinuity->after_counter, 703);
    EXPECT_EQ(applied.realization.output_native_occurrence_count, 7U);
}

TEST(common_analysis_grid_paired_readout,
     explicit_view_preserves_grid_time_source_time_and_native_absence) {
    const auto data = fixture();
    std::map<pipeline::TimestreamNetworkId,
             std::vector<pipeline::NativeSlotAssociation>>
        associations;
    std::vector<pipeline::NativeSlotAssociation> nw0(4);
    for (std::size_t slot = 0; slot < nw0.size(); ++slot) {
        nw0[slot].native_row =
            10 + static_cast<pipeline::TimestreamNativeRow>(slot);
    }
    associations.emplace(0, std::move(nw0));
    std::vector<pipeline::NativeSlotAssociation> nw7(4);
    nw7[0].native_row = 70;
    nw7[1].native_row = 71;
    nw7[2].absence_reason = pipeline::CoincidenceAbsenceReason::no_candidate;
    nw7[3].native_row = 72;
    associations.emplace(7, std::move(nw7));
    const auto nw0_timing = data.native->network(0)
                                .occurrence_axis_handle()
                                ->native_timing_handle();
    const auto nw7_timing = data.native->network(7)
                                .occurrence_axis_handle()
                                ->native_timing_handle();
    const auto relation = pipeline::CommonAnalysisGridRelation::admit(
        data.native->scope(),
        std::vector<pipeline::NativeNetworkAlignment>{*nw0_timing, *nw7_timing},
        vector({1000.0000, 1000.0100, 1000.0200, 1000.0300}),
        std::move(associations));
    const auto view = pipeline::CommonAnalysisGridPairedReadoutView::admit(
        data.native, relation, 0, 4);

    EXPECT_EQ(view->analysis_slot_count(), 4U);
    EXPECT_EQ(view->view_cell_count(), 8U);
    EXPECT_EQ(view->mapped_cell_count(), 7U);
    EXPECT_DOUBLE_EQ(view->grid_time_unix_sec(0), 1000.0000);
    ASSERT_TRUE(view->source_network_time_unix_sec(7, 0));
    EXPECT_DOUBLE_EQ(*view->source_network_time_unix_sec(7, 0), 1000.0025);
    EXPECT_NE(view->grid_time_unix_sec(0),
              *view->source_network_time_unix_sec(7, 0));
    ASSERT_TRUE(view->source_time_residual_sec(7, 0));
    EXPECT_NEAR(*view->source_time_residual_sec(7, 0), 0.0025, 1e-12);
    ASSERT_TRUE(view->representative_native_identity(7, 0));
    EXPECT_EQ(view->representative_native_identity(7, 0)->native_row(), 70);
    ASSERT_NE(view->source_occurrence_interval(7, 0), nullptr);
    EXPECT_EQ(*view->source_occurrence_interval(7, 0),
              data.native->network(7).occurrence_axis_handle()->interval(70));
    ASSERT_TRUE(view->state(pipeline::ReadoutMember::x, 7, 0, 0));
    EXPECT_EQ(view->state(pipeline::ReadoutMember::x, 7, 0, 0)->origin(),
              pipeline::ReadoutMemberOrigin::measured);
    EXPECT_TRUE(view->mapped(0, 2));
    EXPECT_FALSE(view->mapped(7, 2));
    EXPECT_EQ(view->absence_reason(7, 2),
              pipeline::CoincidenceAbsenceReason::no_candidate);
    EXPECT_FALSE(view->value(pipeline::ReadoutMember::x, 7, 2, 0));
    EXPECT_EQ(view->source_occurrence_interval(7, 2), nullptr);
    ASSERT_TRUE(view->native_pair_causes(0, 2, 0));
    EXPECT_TRUE(
        pipeline::has_cause(*view->native_pair_causes(0, 2, 0),
                            pipeline::PairedReadoutCause::r_original_invalid));
    ASSERT_TRUE(view->value(pipeline::ReadoutMember::x, 7, 3, 0));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(
                  *view->value(pipeline::ReadoutMember::x, 7, 3, 0)),
              std::bit_cast<std::uint64_t>(data.native->network(7).value(
                  pipeline::ReadoutMember::x, 72, 0)));
}

TEST(network_timed_rtc_pass_through,
     ordinary_rtc_headers_exclude_common_analysis_grid_dependencies) {
    namespace fs = std::filesystem;
    const auto repository = fs::path{__FILE__}.parent_path().parent_path();
    const std::vector<fs::path> headers{
        repository / "include/citlali/core/pipeline/identity_rtc.h",
        repository /
            "include/citlali/core/pipeline/network_timed_rtc_pass_through.h"};
    const std::vector<std::string> forbidden{
        "CommonAnalysisGridPairedReadoutView",
        "common_analysis_grid_paired_readout.h", "NativeAlignmentPlan",
        "common_slot", "aligned_paired_readout.h"};

    for (const auto &header : headers) {
        std::ifstream stream(header);
        ASSERT_TRUE(stream) << header;
        std::ostringstream content;
        content << stream.rdbuf();
        for (const auto &token : forbidden) {
            EXPECT_EQ(content.str().find(token), std::string::npos)
                << header << " contains forbidden ordinary RTC dependency "
                << token;
        }
    }
}

TEST(network_timed_rtc_pass_through,
     incomplete_context_and_partition_schedule_fail_closed) {
    const auto data = fixture();
    EXPECT_THROW(pipeline::RtcApplicationContext::admit(
                     {0}, data.logical,
                     pipeline::RtcApplicationUse::rtc_terminal, "interval",
                     {"sign", "reference", "baseline"},
                     {"sign", "reference", "baseline"}, "domain",
                     {"rtc-terminal"}, {}, true),
                 std::invalid_argument);

    const auto request = context(data);
    const std::vector<std::shared_ptr<const pipeline::NativePairedReadoutView>>
        incomplete{pipeline::NativePairedReadoutView::admit(
            data.native, {{0, 10, 12}, {7, 70, 71}})};
    EXPECT_THROW(
        pipeline::learn_network_timed_rtc_pass_through(request, incomplete, 1),
        pipeline::IncompleteNativePartitionSchedule);
}

} // namespace

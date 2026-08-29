#include <citlali/core/pipeline/network_timed_rtc_only_route.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;

Eigen::VectorXd route_vector(std::initializer_list<double> values) {
  Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (const auto value : values)
    result(index++) = value;
  return result;
}

std::shared_ptr<const pipeline::NativeReadoutMappingIdentity>
route_mapping(pipeline::TimestreamNetworkId network_id) {
  const auto suffix = std::to_string(network_id);
  return std::make_shared<const pipeline::NativeReadoutMappingIdentity>(
      pipeline::NativeReadoutMappingIdentity{
          "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1",
          "producer:terminal:" + suffix, "tune:terminal:" + suffix,
          "mapping:terminal:" + suffix, "iq-to-xr:terminal:" + suffix,
          "raw-x:terminal:" + suffix, "raw-r:terminal:" + suffix});
}

pipeline::PairedReadoutNetwork
route_network(pipeline::TimestreamNetworkId network_id,
              pipeline::TimestreamNativeRow first_native_row,
              std::initializer_list<double> times,
              std::vector<pipeline::TimestreamPacketCounter> counters,
              std::int64_t detector_uid, double x_base, double r_base,
              std::size_t invalid_offset,
              pipeline::ReadoutMember invalid_member) {
  auto timing = std::make_shared<const pipeline::NativeNetworkAlignment>(
      network_id, first_native_row, route_vector(times), std::move(counters));
  std::vector<pipeline::NativeOccurrenceInterval> intervals;
  intervals.reserve(times.size());
  for (const auto time : times)
    intervals.push_back({time - 0.004, time + 0.004});
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
  if (invalid_member == pipeline::ReadoutMember::x)
    x_states.at(invalid_offset) = invalid;
  else
    r_states.at(invalid_offset) = invalid;
  return pipeline::PairedReadoutNetwork::admit(
      std::move(axis), std::move(detectors), route_mapping(network_id),
      std::move(x), std::move(r), std::move(x_states), std::move(r_states));
}

struct NetworkTimedRtcOnlyFixture {
  std::shared_ptr<const pipeline::PairedReadout> native;
  std::shared_ptr<const pipeline::NativePairedReadoutView> logical;
  std::vector<std::shared_ptr<const pipeline::NativePairedReadoutView>>
      split_partitions;
};

NetworkTimedRtcOnlyFixture route_fixture(double time_offset = 0.0) {
  std::vector<pipeline::PairedReadoutNetwork> networks;
  networks.push_back(route_network(
      0, 10,
      {1000.0000 + time_offset, 1000.0100 + time_offset,
       1000.0200 + time_offset, 1000.0300 + time_offset},
      {100, 101, 102, 103}, 500, 1.0, 101.0, 2, pipeline::ReadoutMember::r));
  networks.push_back(route_network(
      7, 70,
      {1000.0025 + time_offset, 1000.0125 + time_offset,
       1000.0325 + time_offset},
      {700, 701, 703}, 700, 11.0, 111.0, 1, pipeline::ReadoutMember::x));
  auto native = pipeline::PairedReadout::admit(
      pipeline::NativeObservationScope{152390, 0, 4}, {0, 7},
      std::move(networks));
  auto logical = pipeline::NativePairedReadoutView::full(native);
  std::vector<std::shared_ptr<const pipeline::NativePairedReadoutView>>
      split_partitions{pipeline::NativePairedReadoutView::admit(
                           native, {{0, 10, 12}, {7, 70, 71}}),
                       pipeline::NativePairedReadoutView::admit(
                           native, {{0, 12, 14}, {7, 71, 73}})};
  return {std::move(native), std::move(logical), std::move(split_partitions)};
}

std::shared_ptr<const pipeline::RtcApplicationContext> route_context(
    const NetworkTimedRtcOnlyFixture &data,
    pipeline::RtcApplicationUse use = pipeline::RtcApplicationUse::rtc_terminal,
    std::uint64_t request = 7) {
  return pipeline::RtcApplicationContext::admit(
      {request}, data.logical, use, "native-network-occurrences:nw0+nw7",
      {"positive-away-from-reference", "readout-reference:x", "raw-baseline:x"},
      {"positive-orthogonal-to-x", "readout-reference:r", "raw-baseline:r"},
      "finite-raw-paired-domain", {"rtc-terminal"},
      {"SCI-RTC-network-timing-owner-decision", "identity-no-filter-M1"}, true);
}

pipeline::NetworkTimedRtcLogicalFinalization route_finalization(
    const std::shared_ptr<const pipeline::RtcApplicationContext> &context,
    std::uint64_t run, std::uint64_t identity = 11) {
  return {identity,
          {run},
          context,
          context->input_handle()->native_occurrence_count(),
          context->input_handle()->detector_occurrence_count(),
          true};
}

pipeline::NetworkTimedRtcOnlyRouteRequest
route_request(const NetworkTimedRtcOnlyFixture &data, std::uint64_t run = 13) {
  auto context = route_context(data);
  return {
      {run}, context, data.split_partitions, route_finalization(context, run)};
}

TEST(network_timed_rtc_only_route,
     publishes_one_complete_inspectable_native_axis_product) {
  const auto data = route_fixture();
  pipeline::NetworkTimedRtcOnlyProductSlot publication;
  const auto outcome =
      pipeline::run_network_timed_rtc_only(route_request(data), publication);

  ASSERT_TRUE(outcome.complete());
  ASSERT_NE(outcome.published_product, nullptr);
  EXPECT_EQ(publication.snapshot(), outcome.published_product);
  const auto &bundle = *outcome.published_product;
  EXPECT_EQ(bundle.terminal_result().identity.run, 13U);
  EXPECT_EQ(bundle.terminal_result().state,
            pipeline::NetworkTimedRtcOnlyTerminalState::complete);
  EXPECT_EQ(bundle.terminal_result().failure_cause,
            pipeline::NetworkTimedRtcOnlyFailureCause::none);
  EXPECT_TRUE(bundle.terminal_result().failure_detail.empty());
  EXPECT_EQ(bundle.context_handle()->identity().request, 7U);
  EXPECT_EQ(bundle.finalization().run_identity,
            bundle.terminal_result().identity);
  EXPECT_EQ(bundle.finalization().context_handle, bundle.context_handle());
  EXPECT_EQ(bundle.plan_handle()->identity(),
            bundle.realization().plan_identity);
  EXPECT_EQ(bundle.evidence_handle(), bundle.plan_handle()->evidence_handle());

  const auto &product = *bundle.timestream_handle();
  EXPECT_EQ(product.native_parent_handle(), data.native);
  EXPECT_EQ(product.network_spans().size(), 2U);
  EXPECT_EQ(product.output_native_occurrence_count(), 7U);
  EXPECT_EQ(product.output_cell_count(), 7U);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(product.output_time_unix_sec(0, 10)),
            std::bit_cast<std::uint64_t>(1000.0000));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(product.output_time_unix_sec(7, 70)),
            std::bit_cast<std::uint64_t>(1000.0025));
  EXPECT_NE(product.output_time_unix_sec(0, 10),
            product.output_time_unix_sec(7, 70));
  EXPECT_EQ(product.identity(0, 10, 0),
            (pipeline::RtcNativeCellIdentity{0, 10, 500}));
  EXPECT_EQ(product.identity(7, 70, 0),
            (pipeline::RtcNativeCellIdentity{7, 70, 700}));
  EXPECT_EQ(product.memory_evidence().owned_numeric_bytes, 0U);

  const auto &diagnostics = outcome.terminal.diagnostics;
  EXPECT_EQ(diagnostics.network_count, 2U);
  EXPECT_EQ(diagnostics.engineering_partition_count, 2U);
  EXPECT_EQ(diagnostics.detector_count, 2U);
  EXPECT_EQ(diagnostics.native_occurrence_count, 7U);
  EXPECT_EQ(diagnostics.detector_occurrence_count, 7U);
  EXPECT_EQ(diagnostics.evidence_event_count, 2U);
  EXPECT_EQ(diagnostics.direct_x_event_count, 1U);
  EXPECT_EQ(diagnostics.direct_r_event_count, 1U);
  EXPECT_EQ(diagnostics.x_and_r_event_count, 0U);
  EXPECT_EQ(diagnostics.pair_ineligible_cell_count, 2U);
  EXPECT_EQ(diagnostics.x_payload_available_cell_count, 7U);
  EXPECT_EQ(diagnostics.r_payload_available_cell_count, 7U);
  EXPECT_EQ(diagnostics.x_numerically_valid_cell_count, 6U);
  EXPECT_EQ(diagnostics.r_numerically_valid_cell_count, 6U);
  EXPECT_EQ(diagnostics.rtc_owned_numeric_bytes, 0U);
  EXPECT_EQ(diagnostics.context_admission_entry_count, 1U);
  EXPECT_EQ(diagnostics.learn_entry_count, 1U);
  EXPECT_EQ(diagnostics.resolve_entry_count, 1U);
  EXPECT_EQ(diagnostics.apply_entry_count, 1U);
  EXPECT_EQ(diagnostics.finalization_entry_count, 1U);
  EXPECT_EQ(diagnostics.publication_entry_count, 1U);
  EXPECT_STREQ(pipeline::network_timed_rtc_only_terminal_state_name(
                   outcome.terminal.state),
               "complete");
  EXPECT_STREQ(pipeline::network_timed_rtc_only_failure_cause_name(
                   outcome.terminal.failure_cause),
               "none");
}

TEST(network_timed_rtc_only_route,
     an_independent_network_gap_creates_no_terminal_slot_or_absence) {
  const auto data = route_fixture();
  pipeline::NetworkTimedRtcOnlyProductSlot publication;
  const auto outcome =
      pipeline::run_network_timed_rtc_only(route_request(data), publication);
  ASSERT_TRUE(outcome.complete());
  const auto &product = *outcome.published_product->timestream_handle();

  EXPECT_EQ(product.network_spans()[0].occurrence_count(), 4U);
  EXPECT_EQ(product.network_spans()[1].occurrence_count(), 3U);
  EXPECT_DOUBLE_EQ(product.output_time_unix_sec(0, 12), 1000.0200);
  EXPECT_DOUBLE_EQ(product.output_time_unix_sec(7, 72), 1000.0325);
  EXPECT_THROW(product.output_time_unix_sec(7, 73), std::out_of_range);
  EXPECT_EQ(outcome.terminal.diagnostics.native_occurrence_count, 7U);
}

TEST(network_timed_rtc_only_route,
     finalization_is_context_bound_and_required_content_is_fail_closed) {
  const auto data = route_fixture();
  const auto foreign_data = route_fixture(10.0);
  auto replayed = route_request(foreign_data);
  const auto original = route_request(data);
  ASSERT_EQ(original.context->identity(), replayed.context->identity());
  ASSERT_EQ(original.context->input_handle()->native_occurrence_count(),
            replayed.context->input_handle()->native_occurrence_count());
  ASSERT_EQ(original.context->input_handle()->detector_occurrence_count(),
            replayed.context->input_handle()->detector_occurrence_count());
  ASSERT_NE(original.context, replayed.context);
  ASSERT_NE(original.context->input_handle()
                ->network(0)
                .occurrence_axis_handle()
                ->identity(10)
                .reconstructed_time_unix_sec(),
            replayed.context->input_handle()
                ->network(0)
                .occurrence_axis_handle()
                ->identity(10)
                .reconstructed_time_unix_sec());
  replayed.finalization = original.finalization;
  pipeline::NetworkTimedRtcOnlyProductSlot replayed_publication;
  const auto identity_outcome =
      pipeline::run_network_timed_rtc_only(replayed, replayed_publication);
  EXPECT_FALSE(identity_outcome.complete());
  EXPECT_EQ(identity_outcome.terminal.state,
            pipeline::NetworkTimedRtcOnlyTerminalState::finalization_failed);
  EXPECT_EQ(identity_outcome.terminal.failure_cause,
            pipeline::NetworkTimedRtcOnlyFailureCause::
                finalization_identity_mismatch);
  EXPECT_EQ(identity_outcome.terminal.failure_detail,
            "network-timed RTC-only finalization does not bind the exact "
            "admitted run and context");
  EXPECT_EQ(replayed_publication.snapshot(), nullptr);

  auto stale_run = route_request(data);
  stale_run.finalization.run_identity.run = 999;
  pipeline::NetworkTimedRtcOnlyProductSlot stale_run_publication;
  const auto stale_run_outcome =
      pipeline::run_network_timed_rtc_only(stale_run, stale_run_publication);
  EXPECT_FALSE(stale_run_outcome.complete());
  EXPECT_EQ(stale_run_outcome.terminal.failure_cause,
            pipeline::NetworkTimedRtcOnlyFailureCause::
                finalization_identity_mismatch);
  EXPECT_EQ(stale_run_publication.snapshot(), nullptr);

  auto incomplete = route_request(data);
  --incomplete.finalization.completed_native_occurrence_count;
  pipeline::NetworkTimedRtcOnlyProductSlot incomplete_publication;
  const auto incomplete_outcome =
      pipeline::run_network_timed_rtc_only(incomplete, incomplete_publication);
  EXPECT_FALSE(incomplete_outcome.complete());
  EXPECT_EQ(incomplete_outcome.terminal.failure_cause,
            pipeline::NetworkTimedRtcOnlyFailureCause::
                required_logical_content_incomplete);
  EXPECT_EQ(incomplete_outcome.terminal.failure_detail,
            "network-timed RTC-only completed counts do not match the "
            "logical product");
  EXPECT_EQ(incomplete_publication.snapshot(), nullptr);

  auto unfinished = route_request(data);
  unfinished.finalization.observation_facts_finalized = false;
  pipeline::NetworkTimedRtcOnlyProductSlot unfinished_publication;
  const auto unfinished_outcome =
      pipeline::run_network_timed_rtc_only(unfinished, unfinished_publication);
  EXPECT_FALSE(unfinished_outcome.complete());
  EXPECT_EQ(
      unfinished_outcome.terminal.failure_cause,
      pipeline::NetworkTimedRtcOnlyFailureCause::observation_facts_incomplete);
  EXPECT_EQ(unfinished_publication.snapshot(), nullptr);
}

TEST(network_timed_rtc_only_route,
     context_and_partition_failures_do_not_enter_scientific_stages) {
  const auto data = route_fixture();
  auto nonterminal = route_request(data);
  nonterminal.context =
      route_context(data, pipeline::RtcApplicationUse::sci_cal_handoff, 21);
  nonterminal.finalization = route_finalization(nonterminal.context, 13);
  pipeline::NetworkTimedRtcOnlyProductSlot nonterminal_publication;
  const auto nonterminal_outcome = pipeline::run_network_timed_rtc_only(
      nonterminal, nonterminal_publication);
  EXPECT_EQ(nonterminal_outcome.terminal.failure_cause,
            pipeline::NetworkTimedRtcOnlyFailureCause::
                nonterminal_application_context);
  EXPECT_EQ(nonterminal_outcome.terminal.diagnostics.learn_entry_count, 0U);
  EXPECT_EQ(nonterminal_publication.snapshot(), nullptr);

  auto incomplete = route_request(data);
  incomplete.engineering_partitions.pop_back();
  pipeline::NetworkTimedRtcOnlyProductSlot incomplete_publication;
  const auto incomplete_outcome =
      pipeline::run_network_timed_rtc_only(incomplete, incomplete_publication);
  EXPECT_EQ(
      incomplete_outcome.terminal.failure_cause,
      pipeline::NetworkTimedRtcOnlyFailureCause::incomplete_logical_support);
  EXPECT_EQ(incomplete_outcome.terminal.diagnostics.learn_entry_count, 0U);
  EXPECT_EQ(incomplete_publication.snapshot(), nullptr);
}

TEST(network_timed_rtc_only_route,
     publication_is_no_replace_and_preserves_the_committed_product) {
  const auto data = route_fixture();
  pipeline::NetworkTimedRtcOnlyProductSlot publication;
  const auto first =
      pipeline::run_network_timed_rtc_only(route_request(data, 1), publication);
  ASSERT_TRUE(first.complete());
  const auto committed = publication.snapshot();

  const auto second =
      pipeline::run_network_timed_rtc_only(route_request(data, 2), publication);
  EXPECT_FALSE(second.complete());
  EXPECT_EQ(second.terminal.state,
            pipeline::NetworkTimedRtcOnlyTerminalState::publication_failed);
  EXPECT_EQ(
      second.terminal.failure_cause,
      pipeline::NetworkTimedRtcOnlyFailureCause::publication_slot_occupied);
  EXPECT_EQ(second.terminal.failure_detail,
            "network-timed RTC-only product slot already contains a "
            "completion");
  EXPECT_EQ(publication.snapshot(), committed);
  EXPECT_EQ(publication.snapshot()->terminal_result().identity.run, 1U);
}

TEST(network_timed_rtc_only_route,
     terminal_dependency_closure_excludes_cross_network_and_later_stages) {
  namespace fs = std::filesystem;
  const auto header =
      fs::path{__FILE__}.parent_path().parent_path() /
      "include/citlali/core/pipeline/network_timed_rtc_only_route.h";
  std::ifstream stream(header);
  ASSERT_TRUE(stream) << header;
  std::ostringstream content;
  content << stream.rdbuf();
  const std::vector<std::string> forbidden{
      "common_analysis_grid_paired_readout.h",
      "CommonAnalysisGridPairedReadoutView",
      "NativeAlignmentPlan",
      "aligned_paired_readout.h",
      "Astrometry",
      "CalibrationPlan",
      "Ptc",
      "Mapmaking"};
  for (const auto &token : forbidden) {
    EXPECT_EQ(content.str().find(token), std::string::npos)
        << "ordinary RTC terminal route contains forbidden dependency "
        << token;
  }
}

} // namespace

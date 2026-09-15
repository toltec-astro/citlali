#pragma once
// Private offline caller adapter. YAML stops here; scientific decisions and
// their validation remain in the existing concrete RTC components.
#include <citlali/core/pipeline/timestream_rtc_pipeline.h>
#include <yaml-cpp/yaml.h>

namespace citlali::rtc_multidetector_tool {
using namespace pipeline;
inline constexpr std::string_view no_mask_authority =
    "owner-152390-initial-no-spatial-mask-2026-09-15";

inline void require(bool condition, const char *message) {
  if (!condition) throw std::invalid_argument(message);
}
inline void require_no_mask_scope(NativeObservationScope scope,
                                  const std::string &authority) {
  require(scope.observation == 152390 && scope.subobservation == 0 && scope.scan == 2 &&
              authority == no_mask_authority,
          "initial no-mask choice requires exact 152390 scope and owner authority");
}
inline RtcEventRange read_range(const YAML::Node &n) {
  require(n && n.IsSequence() && n.size() == 2, "native support requires two half-open row bounds");
  RtcEventRange r{n[0].as<TimestreamNativeRow>(), n[1].as<TimestreamNativeRow>()};
  require(r.present(), "empty or negative native support");
  return r;
}
inline std::vector<RtcEventRange> read_ranges(const YAML::Node &n) {
  require(n && n.IsSequence(), "support must be explicitly supplied, including an empty set");
  std::vector<RtcEventRange> out;
  for (const auto &r : n) out.push_back(read_range(r));
  return out;
}

struct ReviewedInputs {
  std::shared_ptr<const RtcDonorFillFacts> facts;
  std::vector<RtcDonorSelectedEvent> events;
  std::shared_ptr<const RtcExistingScanBinding> scans;
};

// Exact binding includes source revision, the original input configuration and
// its verified content identities. Selections never transfer to a new Learn
// generation by ordinal alone. No candidate or stable segment is inferred.
inline ReviewedInputs read_review(
    const YAML::Node &review, const std::string &binding,
    std::shared_ptr<const RtcEventAssessmentEvidence> evidence,
    std::span<const int> channels, std::span<const double> flxscale,
    TimestreamNetworkId network, const std::string &apt_identity) {
  require(evidence && review["schema"].as<std::string>() == "rtc-reviewed-selection-v1" &&
              review["learning_binding"].as<std::string>() == binding &&
              review["VAL_generation"].as<std::uint64_t>() ==
                  evidence->spike_handle()->val_snapshot_handle()->generation().value &&
              review["approved"].as<bool>(),
          "review is unapproved or bound to different original Learn/VAL");
  const auto authority = review["authority"].as<std::string>();
  const auto stable_authority = review["stable_support_authority"].as<std::string>();
  const auto contamination_authority = review["contamination_authority"].as<std::string>();
  require(!authority.empty() && !stable_authority.empty() && !contamination_authority.empty(),
          "review needs explicit event, stable-support and contamination authorities");
  const auto &net = evidence->spike_handle()->input_handle()->network(network);
  require(channels.size() == net.detectors().size() && flxscale.size() == channels.size(),
          "review adapter detector/factor shape differs from parent");
  const auto &axis = net.occurrence_axis();
  const auto records = review["detectors"];
  require(records.IsSequence() && records.size() == channels.size(),
          "review must explicitly account for every projected detector");
  std::vector<RtcDonorDetectorFacts> facts;
  const std::string convention = "canonical-apt-v2:flxscale:mJy/beam/xs:prior-static";
  for (std::size_t d = 0; d < channels.size(); ++d) {
    const auto record = records[d];
    require(record["channel"].as<int>() == channels[d] &&
                record["occurrence"].as<std::string>() == net.detector(d).detector_occurrence_id,
            "review detector occurrence/order differs from native projection");
    RtcDonorDetectorFacts fact{network, static_cast<std::uint32_t>(d),
        net.detector(d).detector_occurrence_id,
        apt_identity + ":" + net.detector(d).detector_occurrence_id + ":field=flxscale",
        convention, std::isfinite(flxscale[d]) ? std::optional<double>(flxscale[d]) : std::nullopt,
        {axis.first_native_row(), axis.past_last_native_row()},
        read_ranges(record["stable_segments"]), read_ranges(record["contaminated"])};
    facts.push_back(std::move(fact));
  }
  ReviewedInputs result;
  result.facts = RtcDonorFillFacts::bind(evidence, apt_identity, convention,
      stable_authority, contamination_authority, std::move(facts));
  require(review["events"].IsSequence(), "accepted-event selection must be explicit");
  std::set<std::size_t> seen;
  for (const auto &entry : review["events"]) {
    const auto index = entry["event"].as<std::size_t>();
    require(index < evidence->events().size() && seen.insert(index).second &&
                entry["disposition"].as<std::string>() == "accepted_isolated_event",
            "invalid, repeated or unaccepted event selection");
    const auto &event = evidence->events()[index];
    const auto &seed = evidence->spike_handle()->candidates()[event.seed];
    require(event.network == network && event.detector < channels.size() &&
                entry["channel"].as<int>() == channels[event.detector] &&
                entry["seed_earlier_row"].as<TimestreamNativeRow>() == seed.earlier_row,
            "review event differs from exact original candidate");
    result.events.push_back({evidence, authority, RtcDonorSelectionState::accepted_isolated_event,
                             index, read_range(entry["affected"])});
  }
  if (review["existing_scans"] && !review["existing_scans"].IsNull()) {
    const auto scans = review["existing_scans"];
    require(scans["state"].as<std::string>() == "conservative_native_support_bound",
            "existing scan native support is unavailable");
    std::vector<RtcExistingScanNativeSupport> supports;
    require(scans["support"].IsSequence(), "existing scans require explicit native rows");
    for (const auto &s : scans["support"]) {
      const auto r = read_range(s["rows"]);
      supports.push_back({s["scan"].as<std::uint64_t>(), {network, r.first, r.past_last}});
    }
    result.scans = RtcExistingScanBinding::admit(
        evidence->spike_handle()->input_handle()->parent_handle(),
        scans["processing_generation"].as<std::string>(),
        scans["native_relation_authority"].as<std::string>(),
        scans["timing_uncertainty_authority"].as<std::string>(),
        RtcExistingScanSupportState::conservative_native_support_bound, std::move(supports));
  }
  return result;
}
} // namespace citlali::rtc_multidetector_tool

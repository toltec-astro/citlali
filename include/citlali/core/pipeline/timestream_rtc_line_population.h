#pragma once

#include <citlali/core/pipeline/timestream_rtc_notch_recovery.h>
#include <map>

namespace citlali::pipeline {

// Caller supplies an exact occurrence-scoped APT association. A static
// inverse-sensitivity weight is an optional accounting reference, never CAL
// or a filter-dependent inverse variance. Units/convention are explicit.
struct RtcLinePopulationMember {
  TimestreamNetworkId network = -1;
  std::uint32_t detector = 0;
  std::string occurrence, array_association, science_product;
  RtcOpticalArray array = RtcOpticalArray::a2000;
  std::optional<double> reference_weight;
  std::string weight_authority, weight_convention;
};

struct RtcLinePopulationSupport {
  RtcEventRange rows;
  // Overlapping facts, not precedence-selected causes: x invalid, r invalid,
  // x noise-screening, r noise-screening, accepted jump respectively.
  std::uint8_t cause_bits = 0;
};

struct RtcLinePopulationDetector {
  RtcLinePopulationMember member;
  std::vector<RtcLinePopulationSupport> support;
  std::size_t paired_original_cells = 0, after_existing_exclusions_cells = 0;
};

// Runtime Learn product for the observation actually in memory. It retains
// original spectral/VAL/stage/time/window/uncertainty and transient handles;
// it does not import corpus ranks, choose a filter or claim interference.
class RtcLinePopulationEvidence {
public:
  static std::shared_ptr<const RtcLinePopulationEvidence>
  learn(std::shared_ptr<const RtcLinePowerEvidence> lines,
        std::shared_ptr<const RtcTransientExclusionPlan> exclusions,
        std::vector<RtcLinePopulationMember> members, std::uint64_t attempt) {
    if (!lines || !exclusions || !attempt ||
        lines->snapshot_handle().get() !=
            exclusions->val_snapshot_handle().get() ||
        lines->spectral_handle()->original_spike_handle().get() !=
            exclusions->screening_handle()->evidence_handle().get())
      throw std::invalid_argument("RTC population requires exact original "
                                  "Learn/VAL/exclusion bindings");
    const auto &input = *exclusions->input_handle();
    std::size_t expected = 0;
    for (const auto &span : input.spans())
      expected += input.network(span.network_id).detector_count();
    if (members.size() != expected)
      throw std::invalid_argument(
          "RTC population must declare every detector in its exact input");
    auto out = std::shared_ptr<RtcLinePopulationEvidence>(
        new RtcLinePopulationEvidence);
    out->lines_ = std::move(lines);
    out->exclusions_ = std::move(exclusions);
    out->attempt_ = attempt;
    std::map<std::pair<TimestreamNetworkId, std::uint32_t>, bool> seen;
    for (auto member : members) {
      const auto &net = input.network(member.network);
      const auto &binding = net.detector(member.detector);
      if (!seen.emplace(std::pair{member.network, member.detector}, true)
               .second ||
          member.occurrence != binding.detector_occurrence_id ||
          member.array_association != binding.detector_association_record_id ||
          member.science_product.empty() ||
          (member.reference_weight &&
           (!std::isfinite(*member.reference_weight) ||
            *member.reference_weight <= 0 || member.weight_authority.empty() ||
            member.weight_convention.empty())))
        throw std::invalid_argument(
            "RTC population requires unique exact members and explicit "
            "positive reference weights");
      // Validate even an unweighted array enum; no accidental fallback band.
      (void)rtc_optical_scale(member.array, 1.);
      RtcLinePopulationDetector record;
      record.member = std::move(member);
      const auto &m = record.member;
      const auto span = input.span(m.network);
      for (const auto &run : net.occurrence_axis().contiguous_runs()) {
        const auto first =
            std::max(run.first_native_row, span.first_native_row);
        const auto last =
            std::min(run.past_last_native_row, span.past_last_native_row);
        for (auto row = first; row < last; ++row) {
          std::uint8_t bits = 0;
          for (int c = 0; c < 2; ++c) {
            const auto coordinate = static_cast<NativeReadoutCoordinate>(c);
            if (!net.state(coordinate, row, m.detector).valid())
              bits |= 1 << c;
            else if (!std::isfinite(net.value(coordinate, row, m.detector)))
              throw std::invalid_argument(
                  "RTC population unexpected nonfinite in admitted support");
          }
          const auto causes =
              out->exclusions_->causes(m.network, row, m.detector);
          for (int c = 0; c < 2; ++c)
            if (causes.screening[c] != RtcSpikeNoiseCause::none)
              bits |= 1 << (c + 2);
          if (causes.accepted_jump)
            bits |= 16;
          record.paired_original_cells += !(bits & 3);
          record.after_existing_exclusions_cells += !bits;
          if (row != first && !record.support.empty() &&
              record.support.back().cause_bits == bits)
            record.support.back().rows.past_last = row + 1;
          else
            record.support.push_back({{row, row + 1}, bits});
        }
      }
      out->detectors_.push_back(std::move(record));
    }
    return out;
  }
  const auto &line_handle() const noexcept { return lines_; }
  const auto &exclusion_handle() const noexcept { return exclusions_; }
  const auto &detectors() const noexcept { return detectors_; }
  std::size_t logical_owned_bytes() const noexcept {
    std::size_t bytes = detectors_.size() * sizeof(RtcLinePopulationDetector);
    for (const auto &d : detectors_) {
      bytes += d.support.size() * sizeof(RtcLinePopulationSupport);
      bytes += d.member.occurrence.size() + d.member.array_association.size() +
               d.member.science_product.size() + d.member.weight_authority.size() +
               d.member.weight_convention.size();
    }
    return bytes;
  }
  auto attempt() const noexcept { return attempt_; }
  static constexpr bool interference_admitted = false,
                        treatment_selected = false;
  // Missing cross-detector phase/coherence, source qualification and downstream
  // relation stay absent. Shared frequency alone supplies none of these facts.
private:
  std::shared_ptr<const RtcLinePowerEvidence> lines_;
  std::shared_ptr<const RtcTransientExclusionPlan> exclusions_;
  std::vector<RtcLinePopulationDetector> detectors_;
  std::uint64_t attempt_ = 0;
};

struct RtcLinePopulationContribution {
  RtcLinePopulationMember member;
  std::shared_ptr<const RtcNotchRecoveryPlan> baseline, recovery;
  std::size_t baseline_cells = 0;
  std::optional<std::size_t> recovery_cells;
  double native_cell_seconds = NAN;
  std::optional<double> baseline_weight_seconds, recovery_weight_seconds;
};

// Consider performs exact support/cost comparison only. Every population
// member needs a baseline, so a small replay subset cannot silently become an
// ensemble denominator. Missing recovery is unavailable, not a zero result.
class RtcLinePopulationComparison {
public:
  static std::shared_ptr<const RtcLinePopulationComparison>
  consider(std::shared_ptr<const RtcLinePopulationEvidence> population,
           std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> baselines,
           std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> recoveries,
           std::uint64_t attempt) {
    if (!population || !attempt ||
        baselines.size() != population->detectors().size())
      throw std::invalid_argument(
          "RTC comparison requires complete population baseline");
    using Key = std::pair<TimestreamNetworkId, std::uint32_t>;
    auto index = [&](const auto &plans, bool baseline) {
      std::map<Key, std::shared_ptr<const RtcNotchRecoveryPlan>> result;
      for (const auto &p : plans) {
        if (!p ||
            p->transient_handle().get() !=
                population->exclusion_handle().get() ||
            p->assessment_handle()->candidate_handle()->line_handle().get() !=
                population->line_handle().get())
          throw std::invalid_argument(
              "RTC comparison cannot import foreign plan/evidence");
        const auto &c = *p->assessment_handle()->candidate_handle();
        if (!result.emplace(Key{c.network(), c.detector()}, p).second ||
            !p->finite_five_second_footprint() ||
            (baseline &&
             (p->domain().reject || !c.specification().centered_notch.empty())))
          throw std::invalid_argument("RTC comparison needs unique finite "
                                      "plans and notch-free baseline");
      }
      return result;
    };
    auto base = index(baselines, true), recovery = index(recoveries, false);
    auto out = std::shared_ptr<RtcLinePopulationComparison>(
        new RtcLinePopulationComparison);
    out->population_ = std::move(population);
    out->attempt_ = attempt;
    auto cells = [](const auto &p) {
      std::size_t count = 0;
      for (const auto &r : p->finite_retained_runs())
        count += r.past_last - r.first;
      return count;
    };
    for (const auto &d : out->population_->detectors()) {
      const Key key{d.member.network, d.member.detector};
      const auto b = base.find(key), n = recovery.find(key);
      if (b == base.end() || b->second->domain().array != d.member.array)
        throw std::invalid_argument(
            "RTC population baseline array/coverage mismatch");
      RtcLinePopulationContribution r;
      r.member = d.member;
      r.baseline = b->second;
      r.baseline_cells = cells(r.baseline);
      r.native_cell_seconds = r.baseline->domain().nominal_interval_seconds;
      if (d.member.reference_weight)
        r.baseline_weight_seconds = r.baseline_cells * r.native_cell_seconds *
                                    *d.member.reference_weight;
      if (n != recovery.end()) {
        r.recovery = n->second;
        const auto &bs = r.baseline->assessment_handle()
                             ->candidate_handle()
                             ->specification();
        const auto &ns = r.recovery->assessment_handle()
                             ->candidate_handle()
                             ->specification();
        if (ns.centered_lowpass != bs.centered_lowpass ||
            ns.factor != bs.factor ||
            ns.input_interval_seconds != bs.input_interval_seconds ||
            r.recovery->domain().motion.get() !=
                r.baseline->domain().motion.get() ||
            r.recovery->domain().array != d.member.array ||
            r.recovery->domain().speed_ceiling_arcsec_per_sec !=
                r.baseline->domain().speed_ceiling_arcsec_per_sec ||
            r.recovery->domain().speed_margin_fraction !=
                r.baseline->domain().speed_margin_fraction ||
            r.recovery->domain().cadence_margin_fraction !=
                r.baseline->domain().cadence_margin_fraction ||
            r.recovery->domain().nominal_interval_seconds !=
                r.native_cell_seconds ||
            r.recovery->input_causes() != r.baseline->input_causes())
          throw std::invalid_argument(
              "RTC population alternatives require identical baseline chain "
              "and admitted support");
        r.recovery_cells = cells(r.recovery);
        if (d.member.reference_weight)
          r.recovery_weight_seconds = *r.recovery_cells *
                                      r.native_cell_seconds *
                                      *d.member.reference_weight;
        recovery.erase(n);
      }
      out->contributions_.push_back(std::move(r));
    }
    if (!recovery.empty())
      throw std::invalid_argument("RTC recovery outside the named population");
    return out;
  }
  const auto &population_handle() const noexcept { return population_; }
  const auto &contributions() const noexcept { return contributions_; }
  auto attempt() const noexcept { return attempt_; }
  static constexpr bool scientific_recovery_admitted = false,
                        apply_authorized = false;

private:
  std::shared_ptr<const RtcLinePopulationEvidence> population_;
  std::vector<RtcLinePopulationContribution> contributions_;
  std::uint64_t attempt_ = 0;
};

} // namespace citlali::pipeline

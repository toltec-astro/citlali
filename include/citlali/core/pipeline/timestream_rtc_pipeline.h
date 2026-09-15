#pragma once

#include <citlali/core/pipeline/timestream_rtc_notch_recovery.h>

namespace citlali::pipeline {

class RtcPipelineReassessment;

// One explicit complete RTC attempt. Scientific selections are supplied by
// their existing owners; this boundary adds no classifier, selector or stop
// threshold. Every detector in the admitted view has an explicit finite plan.
class RtcPipelinePlan {
public:
  static std::shared_ptr<const RtcPipelinePlan> consider(
      std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> plans,
      std::shared_ptr<const RtcSpectralTransientConsideration> original,
      std::uint64_t attempt) {
    if (!original || original->spectral_handle()->conditioned_handle() ||
        !attempt || plans.empty())
      throw std::invalid_argument("RTC complete plan requires original evidence and explicit detector plans");
    const auto input = original->spectral_handle()->original_spike_handle()->input_handle();
    const auto snapshot = original->spectral_handle()->original_spike_handle()->val_snapshot_handle();
    std::sort(plans.begin(), plans.end(), [](const auto &a, const auto &b) {
      if (!a || !b) throw std::invalid_argument("RTC detector plan absent");
      const auto &x = *a->assessment_handle()->candidate_handle();
      const auto &y = *b->assessment_handle()->candidate_handle();
      return std::pair{x.network(),x.detector()} < std::pair{y.network(),y.detector()};
    });
    std::size_t index = 0;
    for (const auto &span : input->spans())
      for (std::uint32_t detector = 0; detector < input->network(span.network_id).detectors().size(); ++detector) {
        if (index == plans.size() || !plans[index])
          throw std::invalid_argument("RTC complete plan omits a detector");
        const auto &p = plans[index++];
        const auto &candidate = *p->assessment_handle()->candidate_handle();
        if (p->input_handle().get() != input.get() || p->snapshot_handle().get() != snapshot.get() ||
            p->transient_handle().get() != plans.front()->transient_handle().get() ||
            candidate.network() != span.network_id || candidate.detector() != detector ||
            candidate.line_handle()->spectral_handle().get() != original->spectral_handle().get() ||
            !candidate.specification().notches.empty() || !p->finite_five_second_footprint() ||
            p->transient_handle()->screening_handle()->evidence_handle().get() !=
                original->transient_handle()->evidence_handle()->spike_handle().get())
          throw std::invalid_argument("RTC complete plan has foreign evidence, support, detector or nonfinite stage");
      }
    if (index != plans.size()) throw std::invalid_argument("RTC complete plan repeats a detector");
    std::map<std::pair<TimestreamNetworkId,std::uint32_t>,
        const RtcNotchRecoveryPlan *> by_detector;
    for (const auto &p : plans) {
      const auto &candidate = *p->assessment_handle()->candidate_handle();
      by_detector.emplace(std::pair{candidate.network(),candidate.detector()},p.get());
    }
    // A newly selected replacement is known contamination for another donor's
    // original fit/selection. Do not combine individually valid plans while
    // silently reusing support that the complete plan has since invalidated.
    std::map<std::pair<TimestreamNetworkId,std::uint32_t>,std::vector<RtcEventRange>> selected;
    for (const auto &p : plans) for (const auto &d : p->donor_plans())
      selected[{d->event().network,d->event().detector}].push_back(d->selection().affected);
    for (auto &[key,ranges] : selected) ranges=rtc_event_assessment_detail::merge(std::move(ranges));
    for (const auto &p : plans) for (const auto &d : p->donor_plans()) {
      for (const auto &sample : d->medians()) for (auto detector : sample.eligible) {
        const auto &source = *by_detector.at({d->event().network,detector});
        if (source.domain().reject || source.input_causes().at(sample.row-source.first_native_row()) !=
            RtcNotchRecoveryCause::retained)
          throw std::invalid_argument("RTC complete plan invalidates a donor's original median support");
        const auto it=selected.find({d->event().network,detector});
        if (it!=selected.end() && rtc_event_assessment_detail::contains(it->second,sample.row))
          throw std::invalid_argument("RTC complete plan invalidates a donor's original selection support");
      }
      const auto &e=d->event();const auto &net=input->network(e.network);
      const auto &ranges=selected.at({e.network,e.detector});
      for (const auto &side : e.background[0].support) if(side.usable)
        for(auto row=side.first_used;row<=side.last_used;++row)
          if(net.state(NativeReadoutCoordinate::x,row,e.detector).valid() &&
              !rtc_event_assessment_detail::contains(e.neighbor_exclusions,row) &&
              rtc_event_assessment_detail::contains(ranges,row))
            throw std::invalid_argument("RTC complete plan invalidates a donor's original background support");
    }
    return std::shared_ptr<const RtcPipelinePlan>(new RtcPipelinePlan{
        std::move(plans), std::move(original), attempt});
  }
  static std::shared_ptr<const RtcPipelinePlan> reconsider(
      std::shared_ptr<const RtcPipelineReassessment> evidence,
      std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> complete_plans,
      std::uint64_t attempt);
  const auto &detector_plans() const noexcept { return plans_; }
  const auto &original_consideration() const noexcept { return original_; }
  const auto &reassessment_handle() const noexcept { return reassessment_; }
  const auto &input_handle() const noexcept { return plans_.front()->input_handle(); }
  const auto &snapshot_handle() const noexcept { return plans_.front()->snapshot_handle(); }
  auto attempt() const noexcept { return attempt_; }
  static constexpr bool automatic_selection = false, production_authorized = false;
private:
  RtcPipelinePlan(std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> p,
                  std::shared_ptr<const RtcSpectralTransientConsideration> o, std::uint64_t a)
      : plans_{std::move(p)}, original_{std::move(o)}, attempt_{a} {}
  std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> plans_;
  std::shared_ptr<const RtcSpectralTransientConsideration> original_;
  std::shared_ptr<const RtcPipelineReassessment> reassessment_;
  std::uint64_t attempt_;
};

class RtcPipelineResult {
public:
  static std::shared_ptr<const RtcPipelineResult> apply(
      std::shared_ptr<const RtcPipelinePlan> plan,
      std::shared_ptr<const NativePairedReadoutView> original,
      std::shared_ptr<const ValSnapshot> snapshot,
      std::span<const std::shared_ptr<const NativePairedReadoutView>> partitions,
      std::span<const RtcRecoveryInjection> injections = {}) {
    if (!plan || !original || original.get() != plan->input_handle().get() ||
        !snapshot || snapshot.get() != plan->snapshot_handle().get())
      throw std::invalid_argument("RTC complete Apply requires exact original x/r and frozen VAL; no cumulative input");
    require_exact_native_partition_schedule(*original, partitions);
    if (!injections.empty() && injections.size() != plan->detector_plans().size())
      throw std::invalid_argument("RTC paired injection requires a complete frozen detector-plan binding");
    auto out = std::shared_ptr<RtcPipelineResult>(new RtcPipelineResult);
    out->plan_ = std::move(plan);
    for (std::size_t i = 0; i < out->plan_->detector_plans().size(); ++i)
      out->results_.push_back(RtcNotchRecoveryResult::apply(out->plan_->detector_plans()[i],
          original, snapshot, partitions, injections.empty() ? nullptr : &injections[i]));
    return out;
  }
  const auto &plan_handle() const noexcept { return plan_; }
  const auto &detector_results() const noexcept { return results_; }

  // Explicit native intermediate stages preserve fast-event evidence before
  // decimation. Snapshot rebinding creates a new product, never mutates one.
  // This review profile uses producer/frozen-RTC availability; VAL is an exact
  // context binding, not an inferred new scientific-use/admission policy.
  std::shared_ptr<const RtcConditionedNativeProduct> native_product(
      bool after_lowpass, std::shared_ptr<const ValSnapshot> snapshot) const {
    auto ancestor = snapshot;
    while (ancestor && ancestor.get() != plan_->snapshot_handle().get())
      ancestor = ancestor->parent_snapshot_handle();
    if (!ancestor) throw std::invalid_argument("RTC native evidence snapshot is not a descendant of its frozen plan");
    auto out = std::shared_ptr<RtcConditionedNativeProduct>(new RtcConditionedNativeProduct);
    out->original_ = plan_->original_consideration()->spectral_handle()->original_spike_handle();
    out->snapshot_ = std::move(snapshot);
    out->attempt_ = plan_->attempt(); out->after_lowpass_ = after_lowpass;
    for (const auto &span : plan_->input_handle()->spans()) {
      auto subject = ValNativeRealization::create(plan_->input_handle()->parent_handle(),
          {ValProducer::rtc, plan_->attempt()}, after_lowpass ? 2 : 1,
          ValNativeProductRole::derived_residual, span.network_id);
      out->identities_.push_back(RtcSpectralInputIdentity::bind(subject, out->snapshot_, span,
          RtcSpectralInputStage::native_conditioned,
          after_lowpass ? "rtc-post-lowpass-native-before-decimation" : "rtc-post-notch-native",
          plan_->attempt()));
    }
    for (const auto &r : results_) {
      const auto &candidate = *r->plan_handle()->assessment_handle()->candidate_handle();
      const auto &values = after_lowpass ? r->filtered_native_pair() : r->conditioned_native_pair();
      RtcConditionedNativeProduct::Column column{candidate.network(), candidate.detector(),
          r->plan_handle()->first_native_row(),
          {r, &values}, {}, r};
      column.state.reserve(values.rows());
      for (Eigen::Index i = 0; i < values.rows(); ++i) {
        const auto row = column.first + i;
        std::uint8_t bits = 0;
        for (auto c : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r})
          if (r->coordinate_stage_available(c, row, after_lowpass)) bits |= 1U << static_cast<unsigned>(c);
        if (r->representative_replaced(row)) bits |= 4U;
        if (r->replacement_influence(row, after_lowpass)) bits |= 8U;
        if (r->unrepaired_influence(row, after_lowpass)) bits |= 16U;
        if (r->requires_representative_exclusion(row)) bits |= 32U;
        column.state.push_back(bits);
      }
      out->columns_.push_back(std::move(column));
    }
    return out;
  }
  static constexpr bool map_input_authorized = false;
private:
  RtcPipelineResult() = default;
  std::shared_ptr<const RtcPipelinePlan> plan_;
  std::vector<std::shared_ptr<const RtcNotchRecoveryResult>> results_;
};

// Treatment outcomes are considered alongside unchanged original transient
// evidence. Disappearance after filtering cannot retrospectively classify an
// event or admit its treatment. No automatic convergence/fallback is supplied.
class RtcPipelineReassessment {
public:
  static std::shared_ptr<const RtcPipelineReassessment> consider(
      std::shared_ptr<const RtcPipelineResult> previous,
      std::shared_ptr<const RtcSpectralTransientConsideration> conditioned,
      std::uint64_t attempt) {
    if (!previous || !conditioned || !attempt ||
        conditioned->transient_handle().get() !=
            previous->plan_handle()->original_consideration()->transient_handle().get())
      throw std::invalid_argument("RTC reassessment requires original transient evidence and exact prior Apply");
    const auto product = conditioned->spectral_handle()->conditioned_handle();
    if (!product || product->producer_attempt() != previous->plan_handle()->attempt() ||
        product->columns().size() != previous->detector_results().size())
      throw std::invalid_argument("RTC reassessment requires numerical conditioned evidence from the prior attempt");
    for (std::size_t i = 0; i < product->columns().size(); ++i)
      if (product->columns()[i].source.get() != previous->detector_results()[i].get())
        throw std::invalid_argument("RTC reassessment cannot substitute another realization of Apply");
    return std::shared_ptr<const RtcPipelineReassessment>(new RtcPipelineReassessment{
        std::move(previous), std::move(conditioned), attempt});
  }
  const auto &previous_handle() const noexcept { return previous_; }
  const auto &conditioned_consideration() const noexcept { return conditioned_; }
  const auto &original_consideration() const noexcept { return previous_->plan_handle()->original_consideration(); }
  auto attempt() const noexcept { return attempt_; }
  static constexpr bool classification_authorized = false, stopping_rule_selected = false;
private:
  RtcPipelineReassessment(std::shared_ptr<const RtcPipelineResult> p,
      std::shared_ptr<const RtcSpectralTransientConsideration> c, std::uint64_t a)
      : previous_{std::move(p)}, conditioned_{std::move(c)}, attempt_{a} {}
  std::shared_ptr<const RtcPipelineResult> previous_;
  std::shared_ptr<const RtcSpectralTransientConsideration> conditioned_;
  std::uint64_t attempt_;
};

inline std::shared_ptr<const RtcPipelinePlan> RtcPipelinePlan::reconsider(
    std::shared_ptr<const RtcPipelineReassessment> evidence,
    std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> complete_plans,
    std::uint64_t attempt) {
  if (!evidence || attempt <= evidence->previous_handle()->plan_handle()->attempt())
    throw std::invalid_argument("RTC successor requires a distinct later complete-plan attempt");
  // The accepted numerical stage owners currently resolve initial-VAL plans.
  // Later-VAL Learn is valid evidence, but cannot authorize those old plans
  // under a different snapshot. A later-generation planning use must be bound
  // explicitly before that extension, rather than silently ignoring new facts.
  if (evidence->conditioned_consideration()->spectral_handle()->conditioned_handle()->snapshot_handle().get() !=
      evidence->previous_handle()->plan_handle()->snapshot_handle().get())
    throw StaleRtcValGeneration("RTC successor numerical plans require a matching bound VAL generation");
  auto checked = consider(std::move(complete_plans), evidence->original_consideration(), attempt);
  auto out = std::shared_ptr<RtcPipelinePlan>(new RtcPipelinePlan{
      checked->detector_plans(), checked->original_consideration(), attempt});
  out->reassessment_ = std::move(evidence);
  return out;
}

} // namespace citlali::pipeline

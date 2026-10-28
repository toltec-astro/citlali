#pragma once
#include <citlali/core/pipeline/timestream_rtc_pipeline.h>

namespace citlali::pipeline {

// Observation-local paired scientific-use decision. Indices are exact parent
// detector columns, never persistent identities. Original values/VAL are retained.
struct RtcDisturbanceFacts {
  std::set<std::pair<std::size_t,std::uint64_t>> event_scans;
  bool numerical_output=false, original_available=false, fragmented_pair=false;
  bool input_failure=false;
};
inline bool rtc_repeated_disturbance_rejection(const RtcDisturbanceFacts &f) {
  if(!f.numerical_output || !f.original_available || !f.fragmented_pair || f.input_failure)return false;
  // Two distinct events must be assignable to two distinct science segments.
  for(auto a:f.event_scans)for(auto b:f.event_scans)
    if(a.first!=b.first && a.second!=b.second)return true;
  return false;
}

class RtcDisturbanceExclusionPlan {
public:
  static constexpr const char *identity="repeated-level-shifts-unassessable-remainder-v1";
  struct Record {
    TimestreamNetworkId network;
    std::uint32_t detector;
    bool previously_unavailable=false, dropped=false;
    std::set<std::pair<std::size_t,std::uint64_t>> event_scans;
    std::array<std::size_t,2> assessment_samples{};
  };
  static std::shared_ptr<const RtcDisturbanceExclusionPlan> consider(
      std::shared_ptr<const RtcPipelineResult> result,
      std::shared_ptr<const RtcTreatmentOutcomeEvidence> outcome,
      std::shared_ptr<const RtcJumpExclusionPlan> jumps) {
    if(!result || !outcome || !jumps ||
       jumps->input_handle().get()!=result->plan_handle()->input_handle().get() ||
       outcome->original_handle().get()!=result->plan_handle()->original_consideration()->spectral_handle().get() ||
       !outcome->conditioned_handle()->conditioned_handle()->after_lowpass() ||
       outcome->conditioned_handle()->conditioned_handle()->producer_attempt()!=result->plan_handle()->attempt() ||
       outcome->conditioned_handle()->grid_policy()!=RtcConditionedSpectralGrid::require_full_stretch)
      throw std::invalid_argument("disturbance exclusion requires exact original and final strict-grid evidence");
    auto out=std::shared_ptr<RtcDisturbanceExclusionPlan>(new RtcDisturbanceExclusionPlan);
    out->initial_=std::move(result);out->evidence_=std::move(outcome);out->jumps_=std::move(jumps);
    for(const auto &r:out->initial_->detector_results()) {
      const auto &p=*r->plan_handle();const auto &c=*p.assessment_handle()->candidate_handle();
      if(out->evidence_->conditioned_handle()->conditioned_handle()->column(c.network(),c.detector()).source.get()!=r.get())
        throw std::invalid_argument("disturbance evidence belongs to another numerical realization");
      Record record{c.network(),c.detector()};RtcDisturbanceFacts facts;
      for(const auto &support:out->jumps_->processing_support()) {
        const auto &g=out->jumps_->admission_handle()->groups().at(support.original_group);
        if(g.network==c.network() && g.detector==c.detector() && g.admitted())
          for(auto scan:support.science_scans)record.event_scans.emplace(support.original_group,scan);
      }
      facts.event_scans=record.event_scans;facts.original_available=true;facts.fragmented_pair=true;
      for(Eigen::Index i=0;i<r->filtered_native_pair().rows();++i)
        for(auto coordinate:{NativeReadoutCoordinate::x,NativeReadoutCoordinate::r})
          facts.numerical_output |= r->coordinate_stage_available(coordinate,p.first_native_row()+i,true);
      for(auto coordinate:{NativeReadoutCoordinate::x,NativeReadoutCoordinate::r}) {
        const auto &original=out->evidence_->original_handle()->spectrum(c.network(),c.detector(),coordinate);
        const auto &after=out->evidence_->conditioned_handle()->spectrum(c.network(),c.detector(),coordinate);
        facts.original_available &= original.available();
        facts.fragmented_pair &= after.cause==RtcSpectralCause::fixed_grid_unavailable && !after.centering_support.empty();
        for(const auto *s:{&original,&after})for(const auto &run:s->runs)
          facts.input_failure |= run.unexpected_nonfinite_samples || run.cause==RtcSpectralRunCause::input_consistency_failure || run.cause==RtcSpectralRunCause::arithmetic_nonfinite;
        for(auto span:after.centering_support)record.assessment_samples[static_cast<unsigned>(coordinate)]+=span.past_last-span.first;
      }
      record.previously_unavailable=!facts.numerical_output;
      record.dropped=rtc_repeated_disturbance_rejection(facts);
      out->records_.push_back(std::move(record));
    }
    return out;
  }
  bool has_drops() const {return std::any_of(records_.begin(),records_.end(),[](const auto &r){return r.dropped;});}
  const auto &records() const {return records_;}
  const auto &initial_handle() const {return initial_;}
  const auto &evidence_handle() const {return evidence_;}
  std::shared_ptr<const RtcPipelinePlan> replacement_plan(std::uint64_t attempt) const {
    if(!attempt || attempt==initial_->plan_handle()->attempt())throw std::invalid_argument("disturbance application needs a fresh attempt");
    // Current configured natural-event path has no donor repair admission.
    // Never silently reuse even a median that happens to remain finite.
    for(const auto &p:initial_->plan_handle()->detector_plans())
      if(!p->donor_plans().empty())throw std::invalid_argument("disturbance exclusion requires donor recomputation; donor-bearing route unsupported");
    std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> plans;
    for(std::size_t i=0;i<records_.size();++i) {
      const auto &p=initial_->plan_handle()->detector_plans().at(i);
      auto domain=p->domain();domain.reject |= records_[i].dropped;
      plans.push_back(RtcNotchRecoveryPlan::consider(p->assessment_handle(),p->transient_handle(),p->snapshot_handle(),
          domain,attempt+1+i,{},p->event_decisions(),p->donor_continuity()));
    }
    return RtcPipelinePlan::consider(std::move(plans),initial_->plan_handle()->original_consideration(),attempt);
  }
private:
  RtcDisturbanceExclusionPlan()=default;
  std::shared_ptr<const RtcPipelineResult> initial_;
  std::shared_ptr<const RtcTreatmentOutcomeEvidence> evidence_;
  std::shared_ptr<const RtcJumpExclusionPlan> jumps_;
  std::vector<Record> records_;
};
}

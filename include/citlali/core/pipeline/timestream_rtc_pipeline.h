#pragma once
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/timestream_rtc_treatment_outcome.h>

#include <citlali/core/pipeline/timestream_rtc_notch_recovery.h>

namespace citlali::pipeline {

class RtcPipelineReassessment;
class RtcPipelineDecision;
struct RtcPipelineAdvanceResult;

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
  // Execute a considered disposition. Retain/unavailable preserve this exact
  // candidate; one prescribed revision reuses apply on the original pair.
  static RtcPipelineAdvanceResult advance(
      std::shared_ptr<const RtcPipelineResult> current,
      std::shared_ptr<const RtcPipelineDecision> decision,
      std::shared_ptr<const NativePairedReadoutView> original,
      std::shared_ptr<const ValSnapshot> snapshot,
      std::span<const std::shared_ptr<const NativePairedReadoutView>> partitions);
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
      column.review_use_admitted.reserve(values.rows());
      column.speed_restrictions.reserve(values.rows());
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
        column.review_use_admitted.push_back(r->spectral_review_admitted(row,after_lowpass));
        column.speed_restrictions.push_back(static_cast<std::uint8_t>(r->plan_handle()->speed_restrictions().at(i)));
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

// Diagnostic consequences belong to RTC Learn, not a downstream qualification
// or calibration owner. Domains describe tests; purposes do not select policy.
struct RtcConsequenceDomain {
  std::string identity, source_model, regime, geometry_identity, units;
  std::vector<citlali::config::ReductionType> purposes;
  std::string injected_identity, line_free_identity;
  std::vector<RtcEventRange> windows, ringing_windows;
  std::vector<std::string> required_unavailable, window_unavailable;
};
struct RtcConsequenceMetrics {
  double projection = NAN, peak_ratio = NAN, waveform_error = NAN;
  double centroid_x_arcsec = NAN, centroid_y_arcsec = NAN;
  double negative_ringing_fraction = NAN;
};
struct RtcConsequenceRecord {
  TimestreamNetworkId network;
  std::uint32_t detector;
  std::size_t own_total = 0, peer_total = 0, common_total = 0, expected = 0;
  std::vector<std::int64_t> rows, ringing_rows;
  bool available = false;
  double source_energy = 0, added_line_rms = NAN;
  std::string unavailable;
  RtcConsequenceMetrics measured;
  std::optional<RtcConsequenceMetrics> line_free;
};
class RtcConsequenceEvidence {
public:
  using Positions = Eigen::Matrix<double,Eigen::Dynamic,2>;
  static std::shared_ptr<const RtcConsequenceEvidence> learn(
      std::shared_ptr<const RtcPipelineResult> baseline,
      std::shared_ptr<const RtcPipelineResult> peer,
      const std::shared_ptr<const RtcPipelineResult> &injected,
      const std::shared_ptr<const RtcPipelineResult> &line_free,
      std::span<const RtcRecoveryInjection> source,
      std::span<const Positions> positions,
      RtcConsequenceDomain domain,
      std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t attempt) {
    if(!baseline || !peer || !injected || !snapshot || !attempt || domain.identity.empty() ||
       domain.source_model.empty() || domain.regime.empty() || domain.geometry_identity.empty() ||
       domain.units.empty() || domain.purposes.empty() || domain.injected_identity.empty())
      throw std::invalid_argument("RTC consequence requires explicit purpose/model/domain and exact Apply");
    for(auto purpose:domain.purposes)
      if(purpose!=citlali::config::ReductionType::science && purpose!=citlali::config::ReductionType::pointing &&
         purpose!=citlali::config::ReductionType::beammap && purpose!=citlali::config::ReductionType::oof)
        throw std::invalid_argument("unknown consequence purpose");
    const auto &p=baseline->plan_handle();const auto &q=peer->plan_handle();
    if(p->input_handle().get()!=q->input_handle().get() || p->snapshot_handle().get()!=snapshot.get() ||
       q->snapshot_handle().get()!=snapshot.get() || injected->plan_handle().get()!=p.get() ||
       (line_free && line_free->plan_handle().get()!=p.get()) ||
       bool(line_free)!=!domain.line_free_identity.empty())
      throw std::invalid_argument("RTC consequence foreign input/plan/VAL/overlay");
    const auto count=baseline->detector_results().size();
    if(source.size()!=count || positions.size()!=count || peer->detector_results().size()!=count ||
       domain.windows.size()!=count || domain.ringing_windows.size()!=count ||
       (!domain.window_unavailable.empty() && domain.window_unavailable.size()!=count))
      throw std::invalid_argument("RTC consequence requires complete paired cohort");
    auto out=std::shared_ptr<RtcConsequenceEvidence>(new RtcConsequenceEvidence);
    out->baseline_=std::move(baseline);out->peer_=std::move(peer);out->domain_=std::move(domain);out->attempt_=attempt;
    for(std::size_t d=0;d<count;++d) {
      const auto &b=*out->baseline_->detector_results()[d], &other=*out->peer_->detector_results()[d];
      const auto &z=*injected->detector_results()[d];const auto &plan=*b.plan_handle();
      const auto &candidate=*plan.assessment_handle()->candidate_handle();
      const auto &peer_candidate=*other.plan_handle()->assessment_handle()->candidate_handle();
      const auto first=plan.first_native_row();const auto n=b.filtered_native_pair().rows();
      if(source[d].plan.get()!=b.plan_handle().get() || source[d].delta.rows()!=n ||
         positions[d].rows()!=n || !source[d].delta.allFinite() || !positions[d].allFinite() ||
         candidate.network()!=peer_candidate.network() || candidate.detector()!=peer_candidate.detector() ||
         candidate.specification().factor!=peer_candidate.specification().factor ||
         candidate.specification().input_interval_seconds!=peer_candidate.specification().input_interval_seconds ||
         plan.transient_handle().get()!=other.plan_handle()->transient_handle().get() ||
         plan.event_decisions().get()!=other.plan_handle()->event_decisions().get() ||
         plan.donor_plans()!=other.plan_handle()->donor_plans() ||
         plan.input_causes()!=other.plan_handle()->input_causes() ||
         plan.support_causes()!=other.plan_handle()->support_causes() ||
         plan.speed_restrictions()!=other.plan_handle()->speed_restrictions() ||
         b.injection_identity()!="none" || other.injection_identity()!="none" ||
         z.injection_identity()!=out->domain_.injected_identity ||
         z.output_native_rows()!=b.output_native_rows() || z.causes()!=b.causes())
        throw std::invalid_argument("RTC consequence changed cadence, support, template or injection identity");
      if(line_free && (line_free->detector_results()[d]->injection_identity()!=out->domain_.line_free_identity ||
          line_free->detector_results()[d]->output_native_rows()!=b.output_native_rows() ||
          line_free->detector_results()[d]->causes()!=b.causes()))
        throw std::invalid_argument("RTC consequence line-free pair differs from frozen plan");
      RtcConsequenceRecord record{candidate.network(),candidate.detector()};
      const auto window=out->domain_.windows[d], ring=out->domain_.ringing_windows[d];
      if(window.first<first || window.past_last>first+n || window.first>=window.past_last ||
         ring.first<first || ring.past_last>first+n || ring.first>window.first || ring.past_last<window.past_last)
        throw std::invalid_argument("RTC consequence invalid estimator domain");
      const auto factor=candidate.specification().factor;
      // This current diagnostic uses existing phase-zero scheduled output and
      // necessary independent-center restrictions; it authorizes no map input.
      double added_power=0;
      for(auto row:b.output_native_rows()) if(b.map_center_admitted(row)) {
        ++record.own_total;
        if(std::binary_search(other.output_native_rows().begin(),other.output_native_rows().end(),row) && other.map_center_admitted(row)) {
          ++record.common_total;
          if(line_free){const auto i=row-first;const double delta=z.filtered_native_pair()(i,0)-line_free->detector_results()[d]->filtered_native_pair()(i,0);
            if(!std::isfinite(delta))throw std::invalid_argument("nonfinite admitted line consequence");added_power+=delta*delta;}

          if(window.first<=row && row<window.past_last)record.rows.push_back(row);
          if(ring.first<=row && row<ring.past_last)record.ringing_rows.push_back(row);
        }
      }
      if(line_free && record.common_total)record.added_line_rms=std::sqrt(added_power/record.common_total);
      for(auto row:record.rows){const double value=source[d].delta(row-first,0);record.source_energy+=value*value;}
      for(auto row:other.output_native_rows())record.peer_total+=other.map_center_admitted(row);
      for(auto row=window.first;row<window.past_last;++row)record.expected+=((row-first)%factor)==0;
      if(!out->domain_.window_unavailable.empty() && !out->domain_.window_unavailable[d].empty())
        record.unavailable=out->domain_.window_unavailable[d];
      else if(record.rows.size()!=record.expected || record.expected<4)record.unavailable="incomplete declared crossing on common scheduled support";
      else {
        auto evaluate=[&](const RtcPipelineResult &value) {
          RtcConsequenceMetrics m;double energy=0,dot=0,error=0,peak=0,got_peak=-INFINITY;
          double area=0,refarea=0,xx=0,yy=0,rx=0,ry=0;
          for(auto row:record.rows) {
            const auto i=row-first;
            const double s=source[d].delta(i,0), y=value.detector_results()[d]->filtered_native_pair()(i,0)-b.filtered_native_pair()(i,0);
            if(!std::isfinite(y))throw std::invalid_argument("nonfinite admitted consequence sample");
            energy+=s*s;dot+=s*y;error+=(y-s)*(y-s);peak=std::max(peak,s);got_peak=std::max(got_peak,y);
            area+=y;refarea+=s;xx+=positions[d](i,0)*y;yy+=positions[d](i,1)*y;
            rx+=positions[d](i,0)*s;ry+=positions[d](i,1)*s;
          }
          if(!(energy>0) || !(peak>0) || !(refarea>0))return m;
          m.projection=dot/energy;m.peak_ratio=got_peak/peak;m.waveform_error=std::sqrt(error/energy);
          if(area>0){m.centroid_x_arcsec=xx/area-rx/refarea;m.centroid_y_arcsec=yy/area-ry/refarea;}
          double minimum=0;
          for(auto row:record.ringing_rows) {
            const auto i=row-first;
            const double y=value.detector_results()[d]->filtered_native_pair()(i,0)-b.filtered_native_pair()(i,0);
            if(!std::isfinite(y))throw std::invalid_argument("nonfinite admitted ringing sample");
            minimum=std::min(minimum,y);
          }
          m.negative_ringing_fraction=-minimum/peak;return m;
        };
        record.measured=evaluate(*injected);
        if(line_free)record.line_free=evaluate(*line_free);
        record.available=std::isfinite(record.measured.projection) && (!record.line_free || std::isfinite(record.line_free->projection));
        if(!record.available)record.unavailable="source energy or arithmetic unavailable";
      }
      out->records_.push_back(std::move(record));
    }
    return out;
  }
  const auto &baseline_handle() const noexcept{return baseline_;}
  const auto &peer_handle() const noexcept{return peer_;}
  const auto &domain() const noexcept{return domain_;}
  const auto &records() const noexcept{return records_;}
  auto attempt() const noexcept{return attempt_;}
  static constexpr bool acceptance_requirement_selected=false, science_qualified=false;
private:
  std::shared_ptr<const RtcPipelineResult> baseline_,peer_;
  RtcConsequenceDomain domain_;
  std::vector<RtcConsequenceRecord> records_;
  std::uint64_t attempt_=0;
};

// Treatment outcomes are considered alongside unchanged original transient
// evidence. Disappearance after filtering cannot retrospectively classify an
// event or admit its treatment. No automatic convergence/fallback is supplied.
class RtcPipelineReassessment {
public:
  static std::shared_ptr<const RtcPipelineReassessment> consider(
      std::shared_ptr<const RtcPipelineResult> previous,
      std::shared_ptr<const RtcSpectralTransientConsideration> conditioned,
      std::uint64_t attempt,
      std::shared_ptr<const RtcTreatmentOutcomeEvidence> outcome = nullptr,
      std::vector<std::shared_ptr<const RtcConsequenceEvidence>> consequences = {}) {
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
    if (outcome && (outcome->original_handle().get() != previous->plan_handle()->original_consideration()->spectral_handle().get() ||
                    outcome->conditioned_handle().get() != conditioned->spectral_handle().get()))
      throw std::invalid_argument("RTC outcome must bind exact original reference and conditioned stage/VAL/Apply evidence");
    for(const auto &e:consequences)
      if(!e || (e->baseline_handle().get()!=previous.get() && e->peer_handle().get()!=previous.get()) ||
         product->snapshot_handle().get()!=previous->plan_handle()->snapshot_handle().get())
        throw std::invalid_argument("RTC consequence must bind exact previous Apply and VAL");
    return std::shared_ptr<const RtcPipelineReassessment>(new RtcPipelineReassessment{
        std::move(previous), std::move(conditioned), attempt, std::move(outcome), std::move(consequences)});
  }
  const auto &previous_handle() const noexcept { return previous_; }
  const auto &conditioned_consideration() const noexcept { return conditioned_; }
  const auto &outcome_handle() const noexcept { return outcome_; }
  const auto &consequence_handles() const noexcept { return consequences_; }
  const auto &original_consideration() const noexcept { return previous_->plan_handle()->original_consideration(); }
  auto attempt() const noexcept { return attempt_; }
  static constexpr bool classification_authorized = false, stopping_rule_selected = false;
private:
  RtcPipelineReassessment(std::shared_ptr<const RtcPipelineResult> p,
      std::shared_ptr<const RtcSpectralTransientConsideration> c, std::uint64_t a,
      std::shared_ptr<const RtcTreatmentOutcomeEvidence> o,
      std::vector<std::shared_ptr<const RtcConsequenceEvidence>> e)
      : previous_{std::move(p)}, conditioned_{std::move(c)}, outcome_{std::move(o)}, consequences_{std::move(e)}, attempt_{a} {}
  std::shared_ptr<const RtcPipelineResult> previous_;
  std::shared_ptr<const RtcSpectralTransientConsideration> conditioned_;
  std::shared_ptr<const RtcTreatmentOutcomeEvidence> outcome_;
  std::vector<std::shared_ptr<const RtcConsequenceEvidence>> consequences_;
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


// Execution selection is distinct from scientific qualification. This bounded
// owner-approved development use has no residual-line acceptance policy.
enum class RtcPipelineDisposition { retain, revise, unavailable };
enum class RtcPipelineQualification { unresolved };
enum class RtcPipelineSelectionIntent {
  retain_development_candidate, prescribed_finite_revision,
  require_scientific_qualification
};
enum class RtcPipelineDecisionCause {
  authorized_candidate, prescribed_revision, missing_authority,
  missing_outcome, unavailable_outcome, qualification_unavailable,
  no_op_revision, repeated_plan, revision_budget_exhausted,
  diagnostic_overlay_unbound
};
inline const char *rtc_pipeline_disposition_name(RtcPipelineDisposition d) {
  switch(d) {
  case RtcPipelineDisposition::retain:return "retain";
  case RtcPipelineDisposition::revise:return "revise";
  case RtcPipelineDisposition::unavailable:return "unavailable";
  }
  throw std::invalid_argument("unknown RTC disposition");
}
inline const char *rtc_pipeline_decision_cause_name(RtcPipelineDecisionCause c) {
  switch(c) {
  case RtcPipelineDecisionCause::authorized_candidate:return "explicitly-authorized-development-candidate";
  case RtcPipelineDecisionCause::prescribed_revision:return "explicitly-prescribed-complete-revision";
  case RtcPipelineDecisionCause::missing_authority:return "decision-authority-purpose-or-positive-rationale-missing";
  case RtcPipelineDecisionCause::missing_outcome:return "matched-support-outcome-required";
  case RtcPipelineDecisionCause::unavailable_outcome:return "required-coordinate-outcome-unavailable";
  case RtcPipelineDecisionCause::qualification_unavailable:return "residual-line-acceptance-policy-unselected";
  case RtcPipelineDecisionCause::no_op_revision:return "prescribed-plan-does-not-change-execution";
  case RtcPipelineDecisionCause::repeated_plan:return "prescribed-plan-repeats-an-earlier-attempt";
  case RtcPipelineDecisionCause::revision_budget_exhausted:return "one-revision-operational-budget-exhausted";
  case RtcPipelineDecisionCause::diagnostic_overlay_unbound:return "diagnostic-overlay-replay-authority-unavailable";
  }
  throw std::invalid_argument("unknown RTC decision cause");
}

// A supplied selection, not an inferred policy. The caller supplies its owner
// authority and exact evidence subject; Consider freezes a copy. The only
// revision scope in this increment is explicitly supplied finite coefficients,
// with existing event/support/motion/sampling/donor controls unchanged.
struct RtcPipelineSelection {
  std::shared_ptr<const RtcPipelineReassessment> subject;
  RtcPipelineSelectionIntent intent = RtcPipelineSelectionIntent::retain_development_candidate;
  std::string authority, purpose, positive_rationale;
  std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> complete_revision;
  std::uint64_t next_attempt = 0;
};
struct RtcPipelineDecisionScope {
  TimestreamNetworkId network;
  std::uint32_t detector;
  NativeReadoutCoordinate coordinate;
};
struct RtcPipelineDecisionIssue {
  RtcPipelineDecisionCause cause;
  // Absent means the complete original paired view, not detector rejection.
  std::optional<RtcPipelineDecisionScope> scope;
};

class RtcPipelineDecision {
public:
  static std::shared_ptr<const RtcPipelineDecision> consider(
      std::shared_ptr<const RtcPipelineReassessment> evidence,
      std::shared_ptr<const ValSnapshot> current_snapshot,
      std::optional<RtcPipelineSelection> selection, std::uint64_t attempt) {
    if(!evidence || attempt<=evidence->attempt() ||
        evidence->attempt()<=evidence->previous_handle()->plan_handle()->attempt())
      throw std::invalid_argument("RTC decision requires a distinct later reassessment/decision attempt");
    const auto previous=evidence->previous_handle();const auto plan=previous->plan_handle();
    const auto spectral=evidence->conditioned_consideration()->spectral_handle();
    if(!current_snapshot || current_snapshot.get()!=plan->snapshot_handle().get() ||
        current_snapshot.get()!=spectral->conditioned_handle()->snapshot_handle().get())
      throw StaleRtcValGeneration("RTC decision must reassess changed VAL; it cannot reuse the old plan binding");
    if(selection && selection->subject.get()!=evidence.get())
      throw std::invalid_argument("RTC selection cannot authorize a foreign or stale reassessment/stage/Apply");
    auto out=std::shared_ptr<RtcPipelineDecision>(new RtcPipelineDecision);
    out->evidence_=std::move(evidence);out->snapshot_=std::move(current_snapshot);
    out->selection_=std::move(selection);out->attempt_=attempt;
    auto unavailable=[&](RtcPipelineDecisionCause cause){
      out->cause_=cause;out->issues_.push_back({cause,std::nullopt});return out;
    };
    if(!out->selection_ || out->selection_->authority.empty() || out->selection_->purpose.empty() ||
        out->selection_->positive_rationale.empty())
      return unavailable(RtcPipelineDecisionCause::missing_authority);
    const auto &s=*out->selection_;
    if(s.intent!=RtcPipelineSelectionIntent::retain_development_candidate &&
        s.intent!=RtcPipelineSelectionIntent::prescribed_finite_revision &&
        s.intent!=RtcPipelineSelectionIntent::require_scientific_qualification)
      throw std::invalid_argument("unknown RTC selection intent");
    if(s.intent!=RtcPipelineSelectionIntent::prescribed_finite_revision &&
        (!s.complete_revision.empty() || s.next_attempt))
      throw std::invalid_argument("RTC retain/qualification request cannot hide a revised plan");
    const auto &outcome=out->evidence_->outcome_handle();
    if(!outcome)return unavailable(RtcPipelineDecisionCause::missing_outcome);
    // Completeness and identity are guaranteed by existing Learn/Consider.
    // Inspect every coordinate for this complete paired-cohort review. Never
    // turn absent r or insufficient support into a zero or a passing ratio.
    for(const auto &r:outcome->records())if(!r.available())
      out->issues_.push_back({RtcPipelineDecisionCause::unavailable_outcome,
                             RtcPipelineDecisionScope{r.network,r.detector,r.coordinate}});
    if(!out->issues_.empty()){out->cause_=RtcPipelineDecisionCause::unavailable_outcome;return out;}
    for(const auto &r:previous->detector_results())if(r->injection_identity()!="none")
      return unavailable(RtcPipelineDecisionCause::diagnostic_overlay_unbound);
    if(s.intent==RtcPipelineSelectionIntent::require_scientific_qualification)
      return unavailable(RtcPipelineDecisionCause::qualification_unavailable);
    if(s.intent==RtcPipelineSelectionIntent::retain_development_candidate) {
      out->disposition_=RtcPipelineDisposition::retain;
      out->cause_=RtcPipelineDecisionCause::authorized_candidate;
      out->selected_=plan;return out;
    }
    if(s.next_attempt<=attempt)
      throw std::invalid_argument("RTC prescribed revision needs a distinct later plan attempt");
    auto next=RtcPipelinePlan::reconsider(out->evidence_,s.complete_revision,s.next_attempt);
    if(!same_controls(*plan,*next))
      throw std::invalid_argument("RTC bounded revision cannot change event/donor/validity/motion/sampling controls");
    if(same_execution(*plan,*next))return unavailable(RtcPipelineDecisionCause::no_op_revision);
    // Inspect lineage for a repeated plan before reporting the operational
    // bound. Neither outcome is scientific convergence or permission to reject.
    auto ancestor=plan->reassessment_handle();
    while(ancestor) {
      const auto prior=ancestor->previous_handle()->plan_handle();
      if(same_execution(*prior,*next))return unavailable(RtcPipelineDecisionCause::repeated_plan);
      ancestor=prior->reassessment_handle();
    }
    if(plan->reassessment_handle())return unavailable(RtcPipelineDecisionCause::revision_budget_exhausted);
    out->selected_=std::move(next);out->disposition_=RtcPipelineDisposition::revise;
    out->cause_=RtcPipelineDecisionCause::prescribed_revision;return out;
  }
  const auto &reassessment_handle() const noexcept{return evidence_;}
  const auto &snapshot_handle() const noexcept{return snapshot_;}
  const auto &selection() const noexcept{return selection_;}
  const auto &selected_plan() const noexcept{return selected_;}
  const auto &issues() const noexcept{return issues_;}
  auto attempt() const noexcept{return attempt_;}
  auto disposition() const noexcept{return disposition_;}
  auto cause() const noexcept{return cause_;}
  static constexpr auto qualification=RtcPipelineQualification::unresolved;
  static constexpr const char *missing_qualification="residual-line acceptance policy unselected; existing qualification limits remain";
  static constexpr bool scientifically_qualified=false, downstream_admission_authorized=false,
                        production_authorized=false, stopping_rule_selected=false;
private:
  RtcPipelineDecision()=default;
  static bool same_ranges(const std::vector<RtcEventRange> &a,const std::vector<RtcEventRange> &b) {
    return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin(),[](auto x,auto y){return x.first==y.first&&x.past_last==y.past_last;});
  }
  static bool same_controls(const RtcPipelinePlan &a,const RtcPipelinePlan &b) {
    if(a.input_handle().get()!=b.input_handle().get() || a.snapshot_handle().get()!=b.snapshot_handle().get() ||
       a.original_consideration().get()!=b.original_consideration().get() || a.detector_plans().size()!=b.detector_plans().size())return false;
    for(std::size_t i=0;i<a.detector_plans().size();++i) {
      const auto &p=*a.detector_plans()[i],&q=*b.detector_plans()[i];const auto &x=p.domain(),&y=q.domain();
      const auto &s=p.assessment_handle()->candidate_handle()->specification(),&t=q.assessment_handle()->candidate_handle()->specification();
      if(s.science_domain.has_value()!=t.science_domain.has_value())return false;
      if(s.science_domain && (s.science_domain->identity!=t.science_domain->identity ||
          s.science_domain->array!=t.science_domain->array ||
          s.science_domain->trial_speed_arcsec_per_sec!=t.science_domain->trial_speed_arcsec_per_sec))return false;
      if(p.transient_handle()!=q.transient_handle() || p.event_decisions()!=q.event_decisions() ||
         p.assessment_handle()->joint_handle()!=q.assessment_handle()->joint_handle() ||
         p.assessment_handle()->candidate_handle()->line_handle()!=q.assessment_handle()->candidate_handle()->line_handle() ||
         p.donor_plans()!=q.donor_plans() || p.donor_continuity()!=q.donor_continuity() ||
         p.input_causes()!=q.input_causes() || p.support_causes()!=q.support_causes() ||
         p.speed_restrictions()!=q.speed_restrictions() || !same_ranges(p.runs(),q.runs()) ||
         !same_ranges(p.learning_runs(),q.learning_runs()) || p.first_native_row()!=q.first_native_row() ||
         x.identity!=y.identity || x.motion!=y.motion || x.detector_array_association!=y.detector_array_association || x.array!=y.array ||
         x.speed_ceiling_arcsec_per_sec!=y.speed_ceiling_arcsec_per_sec || x.speed_margin_fraction!=y.speed_margin_fraction ||
         x.cadence_margin_fraction!=y.cadence_margin_fraction || x.nominal_interval_seconds!=y.nominal_interval_seconds ||
         x.notch_guard_samples!=y.notch_guard_samples || x.reject!=y.reject || x.speed_support!=y.speed_support ||
         s.input_interval_seconds!=t.input_interval_seconds || s.factor!=t.factor ||
         s.state_support_identity!=t.state_support_identity || !s.notches.empty() || !t.notches.empty())return false;
    }
    return true;
  }
  static bool same_execution(const RtcPipelinePlan &a,const RtcPipelinePlan &b) {
    if(!same_controls(a,b))return false;
    for(std::size_t i=0;i<a.detector_plans().size();++i) {
      const auto &s=a.detector_plans()[i]->assessment_handle()->candidate_handle()->specification();
      const auto &t=b.detector_plans()[i]->assessment_handle()->candidate_handle()->specification();
      const auto identity_notch=[](const auto &v){return v.empty() || (v.size()==1 && v.front()==1.);};
      if(s.centered_lowpass!=t.centered_lowpass ||
         (s.centered_notch!=t.centered_notch && !(identity_notch(s.centered_notch)&&identity_notch(t.centered_notch))))return false;
    }
    // Reallocated plans/new labels or attempt IDs are not numerical revisions.
    return true;
  }
  std::shared_ptr<const RtcPipelineReassessment> evidence_;
  std::shared_ptr<const ValSnapshot> snapshot_;
  std::optional<RtcPipelineSelection> selection_;
  std::shared_ptr<const RtcPipelinePlan> selected_;
  std::vector<RtcPipelineDecisionIssue> issues_;
  RtcPipelineDisposition disposition_=RtcPipelineDisposition::unavailable;
  RtcPipelineDecisionCause cause_=RtcPipelineDecisionCause::missing_authority;
  std::uint64_t attempt_=0;
};

struct RtcPipelineAdvanceResult {
  std::shared_ptr<const RtcPipelineDecision> decision;
  // Available for authorized inspection even when the reassessment is unresolved.
  std::shared_ptr<const RtcPipelineResult> candidate;
  bool revision_executed=false;
};
inline RtcPipelineAdvanceResult RtcPipelineResult::advance(
    std::shared_ptr<const RtcPipelineResult> current,
    std::shared_ptr<const RtcPipelineDecision> decision,
    std::shared_ptr<const NativePairedReadoutView> original,
    std::shared_ptr<const ValSnapshot> snapshot,
    std::span<const std::shared_ptr<const NativePairedReadoutView>> partitions) {
  if(!current || !decision || decision->reassessment_handle()->previous_handle().get()!=current.get() ||
     !original || original.get()!=current->plan_handle()->input_handle().get() ||
     !snapshot || snapshot.get()!=decision->snapshot_handle().get())
    throw std::invalid_argument("RTC advance requires exact current attempt, decision, original pair and VAL");
  require_exact_native_partition_schedule(*original,partitions);
  if(decision->disposition()!=RtcPipelineDisposition::revise)
    return {std::move(decision),std::move(current),false};
  if(current->plan_handle()->reassessment_handle() || !decision->selected_plan() ||
     decision->selected_plan()->reassessment_handle().get()!=decision->reassessment_handle().get())
    throw std::invalid_argument("RTC advance cannot exceed one revision or substitute its complete plan");
  auto next=apply(decision->selected_plan(),original,snapshot,partitions);
  return {std::move(decision),std::move(next),true};
}

} // namespace citlali::pipeline

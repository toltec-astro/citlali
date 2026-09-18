#pragma once
#include <citlali/core/pipeline/timestream_rtc_donor_fill.h>

namespace citlali::pipeline {

enum class RtcEventTreatmentClass : std::uint8_t {
    isolated_admission_unavailable, admitted_level_shift,
    no_resolved_excursion, unresolved_extent, accepted_declared_contaminant
};

// Test-only truth attached to an explicitly modified parent. This is neither a
// natural-event admission rule nor a manual list of natural event identifiers.
struct RtcDeclaredContaminant {
    std::shared_ptr<const NativePairedReadoutObservation> modified_parent;
    TimestreamNetworkId network=-1;
    std::uint32_t detector=0;
    RtcEventRange rows;
    std::string reference_identity, model_identity;
};

struct RtcEventTreatmentRecord {
    std::size_t event=0;
    TimestreamNetworkId network=-1;
    std::uint32_t detector=0;
    RtcEventTreatmentClass disposition=RtcEventTreatmentClass::isolated_admission_unavailable;
    RtcEventRange affected, operation_unavailable;
    std::vector<std::size_t> cohort_coincident_events;
    bool recovered=false, source_outside=false, background_available=false, seeded_excursion=false;
    bool target_domain_available=false;
    bool proposed_isolation_prerequisites=false; // report-only, not admission
    std::shared_ptr<const RtcDonorFillPlan> donor;
};

// RTC Consider connects existing evidence and policies. Natural isolated-event
// policy is explicitly absent; the result remains executable on resolved
// support without upgrading undecided candidates into clean input.
class RtcEventTreatmentDecision {
public:
    static std::shared_ptr<const RtcEventTreatmentDecision> consider(
        std::shared_ptr<const RtcEventAssessmentDecision> events,
        std::shared_ptr<const RtcTransientExclusionPlan> transients,
        std::vector<RtcDonorDetectorFacts> factors,
        std::string factor_authority, std::string convention,
        std::map<TimestreamNetworkId,std::vector<RtcEventRange>> timing_unavailable,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t id,
        std::optional<RtcDeclaredContaminant> truth=std::nullopt) {
        if(!events || !transients || !snapshot || !id ||
            events->original_screening_handle()->evidence_handle().get()!=transients->screening_handle()->evidence_handle().get() ||
            events->evidence_handle().get()!=&transients->jump_plan_handle()->admission_handle()->assessment() ||
            transients->val_snapshot_handle().get()!=snapshot.get())
            throw std::invalid_argument("RTC event treatment requires exact original evidence, exclusions and VAL");
        const auto evidence=events->evidence_handle();const auto &spikes=*evidence->spike_handle();
        if(truth && (truth->modified_parent.get()!=spikes.input_handle()->parent_handle().get() ||
            !truth->rows.present() || truth->reference_identity.empty() || truth->model_identity.empty() ||
            spikes.input_handle()->network(truth->network).mapping_authority().paired_xr_record_id.find("declared-contaminant:")==std::string::npos))
            throw std::invalid_argument("declared contaminant requires exact modified parent and retained reference/model");
        auto out=std::shared_ptr<RtcEventTreatmentDecision>(new RtcEventTreatmentDecision);
        out->events_=std::move(events);out->transients_=std::move(transients);out->id_=id;
        out->timing_unavailable_=std::move(timing_unavailable);
        // Physical runs and resolved boundaries own segmentation. Candidate
        // masks are donor contamination, not claims of a constant plateau.
        std::map<std::pair<TimestreamNetworkId,std::uint32_t>,std::vector<RtcEventRange>> boundaries;
        for(std::size_t i=0;i<evidence->events().size();++i) {
            const auto &e=evidence->events()[i];const auto &review=out->events_->event_reviews()[i];
            const auto &axis=spikes.input_handle()->network(e.network).occurrence_axis();
            RtcEventTreatmentRecord r;r.event=i;r.network=e.network;r.detector=e.detector;
            r.recovered=!e.refinement_limited;r.background_available=true;
            for(std::size_t c=0;c<2;++c) {
                r.background_available &= e.background[c].available();
                r.recovered &= e.recovery[c].recovered();
                if(e.recovery[c].affected.present()) {
                    r.affected=hull(r.affected,e.recovery[c].affected);
                    r.seeded_excursion |= e.seeded[c];
                }
            }
            r.source_outside=!review.source_protection_unavailable && !review.protected_optical_assessment_required;
            for(std::size_t j=0;j<evidence->events().size();++j) {
                const auto &peer=evidence->events()[j];
                if(peer.network==e.network && peer.detector!=e.detector && overlaps(e.trial_exclusion,peer.trial_exclusion))
                    r.cohort_coincident_events.push_back(j);
            }
            r.proposed_isolation_prerequisites=r.recovered && r.background_available && r.seeded_excursion && r.affected.present() &&
                r.source_outside && !review.health_concern && r.cohort_coincident_events.empty();
            const auto &jump=out->transients_->jump_plan_handle()->admission_handle()->groups().at(i);
            if(jump.admitted()) {
                r.disposition=RtcEventTreatmentClass::admitted_level_shift;r.affected={};
                for(const auto &c:jump.coordinates)if(c.admitted())r.affected=hull(r.affected,c.affected);
                boundaries[{e.network,e.detector}].push_back(r.affected);
            } else {
                // A confirmed finite recovery supplies an operation domain,
                // not hard-event truth. Guards remain distinct from extent.
                r.operation_unavailable=hull(e.trial_exclusion,r.affected);
                if(!r.recovered) {
                    r.disposition=RtcEventTreatmentClass::unresolved_extent;
                    // No finite extent was established. Do not manufacture an
                    // endpoint at a search deadline or storage/processing edge.
                    r.operation_unavailable=rtc_event_assessment_detail::run_for(axis,e.trial_exclusion.first);
                    boundaries[{e.network,e.detector}].push_back(r.operation_unavailable);
                } else if(!r.seeded_excursion) r.disposition=RtcEventTreatmentClass::no_resolved_excursion;
                if(truth && e.network==truth->network && e.detector==truth->detector &&
                    r.proposed_isolation_prerequisites && overlaps(r.affected,truth->rows) &&
                    r.affected.first<=truth->rows.first && r.affected.past_last>=truth->rows.past_last) {
                    r.disposition=RtcEventTreatmentClass::accepted_declared_contaminant;
                    r.operation_unavailable={};
                }
            }
            out->records_.push_back(std::move(r));
        }
        for(auto &f:factors) {
            const auto &net=spikes.input_handle()->network(f.network);const auto &axis=net.occurrence_axis();
            if(!f.stable_segments.empty() || !f.contaminated.empty())
                throw std::invalid_argument("RTC resolver owns support; hand-drawn support is not an input");
            std::vector<RtcEventRange> excluded=out->timing_unavailable_[f.network];
            for(const auto &d:out->transients_->detectors())if(d.network==f.network && d.detector==f.detector)
                    excluded.insert(excluded.end(),d.rows.begin(),d.rows.end());
            for(auto b:boundaries[{f.network,f.detector}])excluded.push_back(b);
            // Pair validity and actual screening/exclusion are required for
            // the stable domain. Nonfinite admitted payload is a failure.
            for(auto row=axis.first_native_row();row<axis.past_last_native_row();++row) {
                if(!net.state(NativeReadoutCoordinate::x,row,f.detector).valid() ||
                    !net.state(NativeReadoutCoordinate::r,row,f.detector).valid()) excluded.push_back({row,row+1});
                else for(auto c:{NativeReadoutCoordinate::x,NativeReadoutCoordinate::r})
                    if(!std::isfinite(net.value(c,row,f.detector)))throw std::invalid_argument("unexpected nonfinite in resolved stable support");
            }
            excluded=rtc_event_assessment_detail::merge(std::move(excluded));
            for(const auto &run:axis.contiguous_runs()) {
                auto begin=run.first_native_row;
                for(auto x:excluded) {
                    if(x.past_last<=begin || x.first>=run.past_last_native_row)continue;
                    if(begin<x.first)f.stable_segments.push_back({begin,x.first});
                    begin=std::max(begin,std::min(x.past_last,run.past_last_native_row));
                }
                if(begin<run.past_last_native_row)f.stable_segments.push_back({begin,run.past_last_native_row});
            }
            f.contaminated=excluded;
            for(const auto &r:out->records_)if(r.network==f.network && r.detector==f.detector) {
                if(r.disposition!=RtcEventTreatmentClass::accepted_declared_contaminant)
                    f.contaminated.push_back(evidence->events()[r.event].trial_exclusion);
                if(r.affected.present())f.contaminated.push_back(r.affected);
                if(r.operation_unavailable.present())f.contaminated.push_back(r.operation_unavailable);
            }
            f.contaminated=rtc_event_assessment_detail::merge(std::move(f.contaminated));
        }
        out->facts_=RtcDonorFillFacts::bind(evidence,std::move(factor_authority),std::move(convention),
            "rtc-original-evidence-boundary-screening-support-v1:signal-plus-background;per-operation-fit-and-boundary-check-required",
            "rtc-all-candidate-guards+pair-extent+unresolved-domain+selected-exclusions-v1",std::move(factors));
        for(auto &r:out->records_) {
            const auto &e=evidence->events()[r.event];const auto *f=out->facts_->find(e.network,e.detector);
            auto context=hull(e.trial_exclusion,r.affected);
            for(const auto &side:e.background[0].support)if(side.usable)context=hull(context,{side.first_used,side.last_used+1});
            r.target_domain_available=f && std::any_of(f->stable_segments.begin(),f->stable_segments.end(),[&](auto s){return covers(s,context);});
            if(r.disposition==RtcEventTreatmentClass::accepted_declared_contaminant)
                r.donor=RtcDonorFillPlan::consider({evidence,truth->model_identity,RtcDonorSelectionState::accepted_isolated_event,r.event,r.affected},
                    out->facts_,out->transients_,snapshot,id+r.event+1);
        }
        // Freeze sparse per-detector intervals once. Sample loops query the
        // established detector axis, never scan other detectors' event records.
        for(const auto &r:out->records_) {
            auto &s=out->support_[{r.network,r.detector}];
            if(r.operation_unavailable.present())s.pending.push_back(r.operation_unavailable);
            if(r.disposition==RtcEventTreatmentClass::accepted_declared_contaminant && r.affected.present())
                s.accepted.push_back(r.affected);
            if(r.donor && r.donor->cause()==RtcDonorFillCause::ready && r.affected.present())
                s.ready.push_back(r.affected);
        }
        for(auto &[key,s]:out->support_) {
            s.pending=rtc_event_assessment_detail::merge(std::move(s.pending));
            s.accepted=rtc_event_assessment_detail::merge(std::move(s.accepted));
            s.ready=rtc_event_assessment_detail::merge(std::move(s.ready));
        }
        return out;
    }
    const auto &records()const noexcept{return records_;}
    const auto &facts_handle()const noexcept{return facts_;}
    const auto &transient_handle()const noexcept{return transients_;}
    const auto &events_handle()const noexcept{return events_;}
    const auto &timing_unavailable()const noexcept{return timing_unavailable_;}
    bool pending(TimestreamNetworkId n,std::uint32_t d,TimestreamNativeRow row)const {
        const auto it=support_.find({n,d});
        return it!=support_.end() && rtc_event_assessment_detail::contains(it->second.pending,row);
    }
    bool accepted_support(TimestreamNetworkId n,std::uint32_t d,TimestreamNativeRow row)const {
        const auto it=support_.find({n,d});
        return it!=support_.end() && rtc_event_assessment_detail::contains(it->second.accepted,row);
    }
    bool donor_ready(TimestreamNetworkId n,std::uint32_t d,TimestreamNativeRow row)const {
        const auto it=support_.find({n,d});
        return it!=support_.end() && rtc_event_assessment_detail::contains(it->second.ready,row);
    }
    static bool overlaps(RtcEventRange a,RtcEventRange b){return a.present()&&b.present()&&a.first<b.past_last&&b.first<a.past_last;}
    static bool covers(RtcEventRange a,RtcEventRange b){return a.present()&&b.present()&&a.first<=b.first&&a.past_last>=b.past_last;}
    static RtcEventRange hull(RtcEventRange a,RtcEventRange b){if(!a.present())return b;if(!b.present())return a;return {std::min(a.first,b.first),std::max(a.past_last,b.past_last)};}
private:
    static bool inside(RtcEventRange a,TimestreamNativeRow row){return a.present()&&row>=a.first&&row<a.past_last;}
    std::shared_ptr<const RtcEventAssessmentDecision> events_;
    std::shared_ptr<const RtcTransientExclusionPlan> transients_;
    std::shared_ptr<const RtcDonorFillFacts> facts_;
    std::vector<RtcEventTreatmentRecord> records_;
    struct DetectorSupport { std::vector<RtcEventRange> pending,accepted,ready; };
    std::map<std::pair<TimestreamNetworkId,std::uint32_t>,DetectorSupport> support_;
    std::map<TimestreamNetworkId,std::vector<RtcEventRange>> timing_unavailable_;
    std::uint64_t id_=0;
};
} // namespace citlali::pipeline

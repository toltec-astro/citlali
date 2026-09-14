#pragma once

#include <citlali/core/pipeline/timestream_rtc_jump_transition.h>

namespace citlali::pipeline {

// Owner-approved single diagnostic pass, RTC-JUMP-REASSESSMENT-001. All rows
// are half-open original native cells; values remain in their coordinate's
// original units. No modified observation, physical-event union or Apply plan.
struct RtcJumpReassessmentPolicy {
    static constexpr std::string_view identity = "rtc-jump-reassessment-2026-09-11-v1";
    static constexpr std::size_t maximum_additional_passes = 1;
};
using RtcJumpFitRows = std::array<std::vector<RtcEventRange>, 2>;

namespace rtc_jump_reassessment_detail {
inline const RtcEventAssessmentEvidence &assessment(const RtcJumpTransitionEvidence &e) {
    return *e.request_handle()->consistency_handle()->evidence_handle()->amplitude_handle()->review_handle()->evidence_handle();
}
inline std::size_t size(const std::vector<RtcEventRange> &ranges) {
    std::size_t n = 0;
    for (auto r : ranges) n += r.past_last - r.first;
    return n;
}
inline void append(std::vector<RtcEventRange> &ranges, TimestreamNativeRow row) {
    if (!ranges.empty() && ranges.back().past_last == row) ++ranges.back().past_last;
    else ranges.push_back({row, row + 1});
}
inline std::size_t overlap(const RtcJumpFitRows &rows, const std::vector<RtcEventRange> &mask) {
    std::size_t n = 0;
    for (const auto &side : rows) for (auto r : side) for (auto m : mask)
        n += std::max<TimestreamNativeRow>(0, std::min(r.past_last,m.past_last)-std::max(r.first,m.first));
    return n;
}
// Recover exact admitted rows, not a bounding interval. The count assertion
// detects drift from the accepted learner's selection before any new fit runs.
inline RtcJumpFitRows admitted(const RtcSpikeEvidence &spikes, const RtcAssessedEvent &event,
                              std::size_t c, const std::array<RtcEventFitSupport,2> &support,
                              double flank) {
    using namespace rtc_event_assessment_detail;
    RtcJumpFitRows out;
    const auto &net = spikes.input_handle()->network(event.network);
    const auto &axis = net.occurrence_axis();
    double begin=INFINITY,end=-INFINITY;
    for(auto row=event.trial_exclusion.first;row<event.trial_exclusion.past_last;++row) {
        const auto &s=axis.occurrence(row).integration_support;
        begin=std::min(begin,s.begin_unix_sec);end=std::max(end,s.end_unix_sec);
    }
    for(std::size_t side=0;side<2;++side) {
        if(support[side].usable==0) continue;
        for(auto row=support[side].first_used;row<=support[side].last_used;++row) {
            const auto &s=axis.occurrence(row).integration_support;
            const bool in=side==0 ? row<event.trial_exclusion.first && s.begin_unix_sec>=begin-flank && s.end_unix_sec<=begin :
                row>=event.trial_exclusion.past_last && s.begin_unix_sec>=end && s.end_unix_sec<=end+flank;
            if(in && !contains(event.neighbor_exclusions,row) && net.state(coord(c),row,event.detector).valid() &&
               std::isfinite(net.value(coord(c),row,event.detector))) append(out[side],row);
        }
        if(size(out[side])!=support[side].usable)
            throw std::invalid_argument("RTC reassessment does not reproduce original admitted fit rows");
    }
    return out;
}
inline RtcJumpFitRows subtract(const RtcJumpFitRows &rows, const std::vector<RtcEventRange> &mask) {
    RtcJumpFitRows out;
    for(std::size_t side=0;side<2;++side) for(auto r:rows[side])
        for(auto row=r.first;row<r.past_last;++row)
            if(!rtc_event_assessment_detail::contains(mask,row)) append(out[side],row);
    return out;
}
} // namespace rtc_jump_reassessment_detail

struct RtcJumpSupportAudit {
    std::size_t event = 0;
    std::array<RtcJumpFitRows,2> primary, shorter;
    // Same-coordinate overlap answers the original 4,464-bracket question;
    // paired overlap also records the partner-coordinate effect of the mask.
    std::array<std::size_t,2> own_primary_overlap{}, own_short_overlap{}, paired_primary_overlap{}, paired_short_overlap{};
    std::vector<RtcEventRange> measured_union;
};
class RtcJumpSupportEvidence {
public:
    static std::shared_ptr<const RtcJumpSupportEvidence> learn(
        std::shared_ptr<const RtcJumpTransitionEvidence> parent, std::uint64_t id) {
        if(!parent || !id) throw std::invalid_argument("RTC support audit requires transition evidence and identity");
        auto out=std::shared_ptr<RtcJumpSupportEvidence>(new RtcJumpSupportEvidence);
        out->parent_=std::move(parent);out->id_=id;
        const auto &a=rtc_jump_reassessment_detail::assessment(*out->parent_);
        const auto &spikes=*a.spike_handle();
        const auto &shorts=out->parent_->request_handle()->consistency_handle()->evidence_handle()->coordinates();
        for(std::size_t i=0;i<a.events().size();++i) {
            const auto &pair=out->parent_->coordinates()[i];
            if(pair[0].cause==RtcJumpTransitionCause::not_requested && pair[1].cause==RtcJumpTransitionCause::not_requested) continue;
            RtcJumpSupportAudit audit;audit.event=i;const auto &event=a.events()[i];
            for(const auto &t:pair) if(t.available()) audit.measured_union.push_back(t.affected);
            audit.measured_union=rtc_event_assessment_detail::merge(std::move(audit.measured_union));
            for(std::size_t c=0;c<2;++c) {
                if(event.background[c].available()) audit.primary[c]=rtc_jump_reassessment_detail::admitted(spikes,event,c,event.background[c].support,RtcEventBackgroundPolicy::flank_seconds);
                if(shorts[i][c].available()) audit.shorter[c]=rtc_jump_reassessment_detail::admitted(spikes,event,c,shorts[i][c].support,RtcJumpConsistencyPolicy::short_flank_seconds);
                if(pair[c].available()) {
                    audit.own_primary_overlap[c]=rtc_jump_reassessment_detail::overlap(audit.primary[c],{pair[c].affected});
                    audit.own_short_overlap[c]=rtc_jump_reassessment_detail::overlap(audit.shorter[c],{pair[c].affected});
                }
                audit.paired_primary_overlap[c]=rtc_jump_reassessment_detail::overlap(audit.primary[c],audit.measured_union);
                audit.paired_short_overlap[c]=rtc_jump_reassessment_detail::overlap(audit.shorter[c],audit.measured_union);
            }
            out->audits_.push_back(std::move(audit));
        }
        return out;
    }
    const auto &parent_handle() const noexcept {return parent_;}
    const auto &audits() const noexcept {return audits_;}
    std::uint64_t attempt() const noexcept {return id_;}
private:
    RtcJumpSupportEvidence()=default;
    std::shared_ptr<const RtcJumpTransitionEvidence> parent_;
    std::vector<RtcJumpSupportAudit> audits_;
    std::uint64_t id_=0;
};
struct RtcJumpRefitSelection {bool requested=false;std::vector<RtcEventRange> paired_mask;};
class RtcJumpRefitRequest {
public:
    static std::shared_ptr<const RtcJumpRefitRequest> consider(
        std::shared_ptr<const RtcJumpSupportEvidence> audit, std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t id) {
        if(!audit) throw std::invalid_argument("RTC refit request requires support audit");
        const auto &a=rtc_jump_reassessment_detail::assessment(*audit->parent_handle());
        rtc_jump_detail::require_snapshot(a,snapshot,id);
        auto out=std::shared_ptr<RtcJumpRefitRequest>(new RtcJumpRefitRequest);
        out->audit_=std::move(audit);out->id_=id;
        for(const auto &s:out->audit_->audits()) {
            RtcJumpRefitSelection selection;
            for(std::size_t c=0;c<2;++c) selection.requested|=s.paired_primary_overlap[c]>0 || s.paired_short_overlap[c]>0;
            if(selection.requested) {
                selection.paired_mask=a.events()[s.event].neighbor_exclusions;
                selection.paired_mask.push_back(a.events()[s.event].trial_exclusion);
                selection.paired_mask.insert(selection.paired_mask.end(),s.measured_union.begin(),s.measured_union.end());
                selection.paired_mask=rtc_event_assessment_detail::merge(std::move(selection.paired_mask));
            }
            out->selections_.push_back(std::move(selection));
        }
        return out;
    }
    const auto &audit_handle() const noexcept {return audit_;}
    const auto &selections() const noexcept {return selections_;}
    std::uint64_t consideration() const noexcept {return id_;}
private:
    RtcJumpRefitRequest()=default;
    std::shared_ptr<const RtcJumpSupportEvidence> audit_;
    std::vector<RtcJumpRefitSelection> selections_;
    std::uint64_t id_=0;
};
struct RtcJumpRefitCoordinate {
    RtcJumpFitRows primary_rows, short_rows;
    RtcEventCoordinateBackground primary;
    RtcJumpShortFit shorter;
    RtcEventRecovery recovery;
};
struct RtcJumpRefitCounts {std::size_t requested_groups=0,fit_calls=0,iterations=0,peak_scratch_rows=0;};
namespace rtc_jump_reassessment_detail {
inline RtcEventCoordinateBackground refit(const RtcSpikeEvidence &spikes,const RtcAssessedEvent &event,
        std::size_t c,const RtcJumpFitRows &rows,const RtcEventCubicFit &frozen,bool fit_cubic,RtcJumpRefitCounts &counts) {
    RtcEventCoordinateBackground out;out.pre_scale_fit=frozen;
    const auto &net=spikes.input_handle()->network(event.network);const auto &axis=net.occurrence_axis();
    for(std::size_t side=0;side<2;++side) for(auto r:rows[side]) for(auto row=r.first;row<r.past_last;++row) {
        auto &s=out.support[side];const auto &cell=axis.occurrence(row).integration_support;
        if(!s.usable){s.first_used=row;s.begin_unix_sec=cell.begin_unix_sec;}
        ++s.usable;s.last_used=row;s.end_unix_sec=cell.end_unix_sec;
    }
    out.support_cause=RtcEventFitCause::none;
    if(out.support[0].usable<RtcEventBackgroundPolicy::minimum_samples || out.support[1].usable<RtcEventBackgroundPolicy::minimum_samples) {
        out.support_cause=RtcEventFitCause::insufficient_samples;return out;
    }
    if(!frozen.available() || !std::isfinite(frozen.scale) || frozen.scale<=0) {
        out.support_cause=RtcEventFitCause::zero_scale;return out;
    }
    const auto n=out.support[0].usable+out.support[1].usable;
    counts.peak_scratch_rows=std::max(counts.peak_scratch_rows,n);
    Eigen::MatrixXd design(n,5);Eigen::VectorXd values(n);std::size_t at=0;
    for(std::size_t side=0;side<2;++side) for(auto r:rows[side]) for(auto row=r.first;row<r.past_last;++row) {
        if(!net.state(rtc_event_assessment_detail::coord(c),row,event.detector).valid())
            throw std::invalid_argument("RTC refit original admitted state changed");
        const double t=(rtc_event_assessment_detail::time(axis,row)-event.origin)/event.time_scale;
        design.row(at)<<1,t,t*t,t*t*t,static_cast<double>(side);
        values[at++]=net.value(rtc_event_assessment_detail::coord(c),row,event.detector);
    }
    if(fit_cubic) {++counts.fit_calls;out.cubic=rtc_event_background_detail::fit(design.leftCols(4),values,frozen.scale);counts.iterations+=out.cubic.iterations;}
    ++counts.fit_calls;out.cubic_with_offset=rtc_event_background_detail::fit(design,values,frozen.scale);counts.iterations+=out.cubic_with_offset.iterations;
    return out;
}
inline auto physical_runs(const RtcSpikeEvidence &spikes) {
    std::map<TimestreamNetworkId,std::vector<NativeContiguousRun>> runs;
    for(const auto &span:spikes.input_handle()->spans()) runs[span.network_id]=spikes.input_handle()->network(span.network_id).occurrence_axis().contiguous_runs();
    return runs;
}
inline NativeContiguousRun run_for(const RtcSpikeEvidence &spikes,const RtcAssessedEvent &event,
        const std::map<TimestreamNetworkId,std::vector<NativeContiguousRun>> &runs) {
    const auto row=spikes.candidates()[event.seed].earlier_row;
    for(const auto &run:runs.at(event.network))
        if(row>=run.first_native_row && row<run.past_last_native_row) return run;
    throw std::invalid_argument("RTC reassessment event has no physical run");
}
} // namespace rtc_jump_reassessment_detail
class RtcJumpRefitEvidence {
public:
    static std::shared_ptr<const RtcJumpRefitEvidence> learn(std::shared_ptr<const RtcJumpRefitRequest> request,std::uint64_t id) {
        if(!request || !id) throw std::invalid_argument("RTC refit Learn requires request and identity");
        auto out=std::shared_ptr<RtcJumpRefitEvidence>(new RtcJumpRefitEvidence);out->request_=std::move(request);out->id_=id;
        const auto &audit=*out->request_->audit_handle();const auto &a=rtc_jump_reassessment_detail::assessment(*audit.parent_handle());
        const auto &spikes=*a.spike_handle();const auto runs=rtc_jump_reassessment_detail::physical_runs(spikes);
        const auto &old_short=audit.parent_handle()->request_handle()->consistency_handle()->evidence_handle()->coordinates();
        out->coordinates_.resize(audit.audits().size());
        for(std::size_t i=0;i<audit.audits().size();++i) {
            if(!out->request_->selections()[i].requested) continue;
            ++out->counts_.requested_groups;
            const auto &s=audit.audits()[i];const auto &old=a.events()[s.event];
            const auto &mask=out->request_->selections()[i].paired_mask;
            const auto run=rtc_jump_reassessment_detail::run_for(spikes,old,runs);
            for(std::size_t c=0;c<2;++c) {
                auto &r=out->coordinates_[i][c];
                r.primary_rows=rtc_jump_reassessment_detail::subtract(s.primary[c],mask);
                r.short_rows=rtc_jump_reassessment_detail::subtract(s.shorter[c],mask);
                if(old.background[c].available()) r.primary=rtc_jump_reassessment_detail::refit(spikes,old,c,r.primary_rows,old.background[c].pre_scale_fit,true,out->counts_);
                if(old_short[s.event][c].available()) {
                    const auto fit=rtc_jump_reassessment_detail::refit(spikes,old,c,r.short_rows,old_short[s.event][c].pre_scale_fit,false,out->counts_);
                    r.shorter.support=fit.support;r.shorter.pre_scale_fit=fit.pre_scale_fit;r.shorter.cubic_with_offset=fit.cubic_with_offset;
                    r.shorter.cause=fit.support_cause==RtcEventFitCause::none ? RtcJumpShortFitCause::none : RtcJumpShortFitCause::insufficient_samples;
                    r.shorter.scratch_rows=fit.support[0].usable+fit.support[1].usable;
                }
                // Reuse the established recovery test, with the refitted model
                // and original scale. No regrouping or support expansion.
                auto updated=old;updated.background[c]=r.primary;
                r.recovery=rtc_event_assessment_detail::recover(spikes,updated,{run.first_native_row,run.past_last_native_row},c);
            }
        }
        return out;
    }
    const auto &request_handle() const noexcept {return request_;}
    const auto &coordinates() const noexcept {return coordinates_;}
    const auto &counts() const noexcept {return counts_;}
    std::uint64_t attempt() const noexcept {return id_;}
private:
    RtcJumpRefitEvidence()=default;
    std::shared_ptr<const RtcJumpRefitRequest> request_;
    std::vector<std::array<RtcJumpRefitCoordinate,2>> coordinates_;
    RtcJumpRefitCounts counts_;
    std::uint64_t id_=0;
};
struct RtcJumpRemeasureSelection {
    RtcJumpConsistencyCause consistency=RtcJumpConsistencyCause::primary_not_passed;
    bool requested=false;
};
class RtcJumpRemeasureRequest {
public:
    static std::shared_ptr<const RtcJumpRemeasureRequest> consider(std::shared_ptr<const RtcJumpRefitEvidence> refit,
            std::shared_ptr<const ValSnapshot> snapshot,std::uint64_t id) {
        if(!refit) throw std::invalid_argument("RTC remeasurement requires refit evidence");
        const auto &audit=*refit->request_handle()->audit_handle();
        rtc_jump_detail::require_snapshot(rtc_jump_reassessment_detail::assessment(*audit.parent_handle()),snapshot,id);
        auto out=std::shared_ptr<RtcJumpRemeasureRequest>(new RtcJumpRemeasureRequest);out->refit_=std::move(refit);out->id_=id;
        const auto &amplitude=audit.parent_handle()->request_handle()->consistency_handle()->evidence_handle()->amplitude_handle()->coordinates();
        out->selections_.resize(audit.audits().size());
        for(std::size_t i=0;i<audit.audits().size();++i) for(std::size_t c=0;c<2;++c) {
            if(!out->refit_->request_handle()->selections()[i].requested) continue;
            const auto &r=out->refit_->coordinates()[i][c];auto &s=out->selections_[i][c];
            if(!r.primary.available() || !r.shorter.available()) {s.consistency=RtcJumpConsistencyCause::short_fit_unavailable;continue;}
            s.consistency=rtc_jump_detail::consistency(r.primary.cubic_with_offset.offset,r.shorter.cubic_with_offset.offset,amplitude[audit.audits()[i].event][c].sigma_delta);
            s.requested=s.consistency==RtcJumpConsistencyCause::passes && !r.recovery.recovered();
        }
        return out;
    }
    const auto &refit_handle() const noexcept {return refit_;}
    const auto &selections() const noexcept {return selections_;}
    std::uint64_t consideration() const noexcept {return id_;}
private:
    RtcJumpRemeasureRequest()=default;
    std::shared_ptr<const RtcJumpRefitEvidence> refit_;
    std::vector<std::array<RtcJumpRemeasureSelection,2>> selections_;
    std::uint64_t id_=0;
};
struct RtcJumpRemeasuredCoordinate {
    RtcJumpTransition transition;
    std::size_t primary_overlap=0,short_overlap=0;
};
class RtcJumpReassessmentEvidence {
public:
    static std::shared_ptr<const RtcJumpReassessmentEvidence> learn(std::shared_ptr<const RtcJumpRemeasureRequest> request,std::uint64_t id) {
        if(!request || !id) throw std::invalid_argument("RTC remeasurement Learn requires request and identity");
        auto out=std::shared_ptr<RtcJumpReassessmentEvidence>(new RtcJumpReassessmentEvidence);out->request_=std::move(request);out->id_=id;
        const auto &refit=*out->request_->refit_handle();const auto &audit=*refit.request_handle()->audit_handle();
        const auto &a=rtc_jump_reassessment_detail::assessment(*audit.parent_handle());const auto &spikes=*a.spike_handle();
        const auto runs=rtc_jump_reassessment_detail::physical_runs(spikes);
        std::map<rtc_event_assessment_detail::Key,std::vector<std::size_t>> indices;
        for(std::size_t i=0;i<audit.audits().size();++i) if(out->request_->selections()[i][0].requested || out->request_->selections()[i][1].requested) {
            const auto &event=a.events()[audit.audits()[i].event];indices.try_emplace({event.network,event.detector});
        }
        for(std::size_t i=0;i<spikes.candidates().size();++i) {
            const auto &b=spikes.blocks()[spikes.candidates()[i].noise_block_index];auto it=indices.find({b.network_id,b.detector_index});
            if(it!=indices.end()) it->second.push_back(i);
        }
        for(auto &[key,list]:indices) std::sort(list.begin(),list.end(),[&](auto x,auto y){return std::tie(spikes.candidates()[x].earlier_row,x)<std::tie(spikes.candidates()[y].earlier_row,y);});
        out->coordinates_.resize(audit.audits().size());
        for(std::size_t i=0;i<audit.audits().size();++i) {
            if(!out->request_->selections()[i][0].requested && !out->request_->selections()[i][1].requested) continue;
            auto event=a.events()[audit.audits()[i].event];const auto run=rtc_jump_reassessment_detail::run_for(spikes,event,runs);
            const auto neighbors=rtc_jump_transition_detail::neighbor_masks(spikes,event,run,indices.at({event.network,event.detector}));
            std::vector<RtcEventRange> measured;
            for(std::size_t c=0;c<2;++c) if(out->request_->selections()[i][c].requested) {
                event.background[c]=refit.coordinates()[i][c].primary;
                auto &r=out->coordinates_[i][c];r.transition=rtc_jump_transition_detail::measure(spikes,event,c,run,neighbors);
                if(r.transition.available()) measured.push_back(r.transition.affected);
            }
            measured=rtc_event_assessment_detail::merge(std::move(measured));
            // Check the new paired support against both coordinates' actual
            // successor fits. A second refit is deliberately not an interface.
            for(std::size_t c=0;c<2;++c) {
                auto &r=out->coordinates_[i][c];
                r.primary_overlap=rtc_jump_reassessment_detail::overlap(refit.coordinates()[i][c].primary_rows,measured);
                r.short_overlap=rtc_jump_reassessment_detail::overlap(refit.coordinates()[i][c].short_rows,measured);
            }
        }
        return out;
    }
    const auto &request_handle() const noexcept {return request_;}
    const auto &coordinates() const noexcept {return coordinates_;}
    std::uint64_t attempt() const noexcept {return id_;}
    static constexpr bool hard_event_accepted=false,apply_authorized=false;
private:
    RtcJumpReassessmentEvidence()=default;
    std::shared_ptr<const RtcJumpRemeasureRequest> request_;
    std::vector<std::array<RtcJumpRemeasuredCoordinate,2>> coordinates_;
    std::uint64_t id_=0;
};
// Final diagnostic Consider: a surviving numerical measurement still carries
// all missing scientific admission/protection/timing conditions from its parent.
enum class RtcJumpReassessmentCause : std::uint8_t {
    unchanged_measured, original_unavailable, refit_unavailable,
    consistency_failed, confirmed_recovery, transition_unavailable,
    support_still_overlaps, reassessed_measured
};
class RtcJumpReassessmentDecision {
public:
    static std::shared_ptr<const RtcJumpReassessmentDecision> consider(
            std::shared_ptr<const RtcJumpReassessmentEvidence> evidence,
            std::shared_ptr<const ValSnapshot> snapshot,std::uint64_t id) {
        if(!evidence) throw std::invalid_argument("RTC reassessment decision requires evidence");
        const auto &request=*evidence->request_handle();const auto &refit=*request.refit_handle();
        const auto &audit=*refit.request_handle()->audit_handle();
        rtc_jump_detail::require_snapshot(rtc_jump_reassessment_detail::assessment(*audit.parent_handle()),snapshot,id);
        auto out=std::shared_ptr<RtcJumpReassessmentDecision>(new RtcJumpReassessmentDecision);out->evidence_=std::move(evidence);out->id_=id;
        for(std::size_t i=0;i<audit.audits().size();++i) {
            std::array<RtcJumpReassessmentCause,2> pair;
            bool overlap=false;
            for(const auto &r:out->evidence_->coordinates()[i]) overlap|=r.primary_overlap>0 || r.short_overlap>0;
            for(std::size_t c=0;c<2;++c) {
                const auto &r=refit.coordinates()[i][c];
                if(!refit.request_handle()->selections()[i].requested)
                    pair[c]=audit.parent_handle()->coordinates()[audit.audits()[i].event][c].available() ?
                        RtcJumpReassessmentCause::unchanged_measured : RtcJumpReassessmentCause::original_unavailable;
                else if(!r.primary.available() || !r.shorter.available()) pair[c]=RtcJumpReassessmentCause::refit_unavailable;
                else if(request.selections()[i][c].consistency!=RtcJumpConsistencyCause::passes) pair[c]=RtcJumpReassessmentCause::consistency_failed;
                else if(r.recovery.recovered()) pair[c]=RtcJumpReassessmentCause::confirmed_recovery;
                else if(!out->evidence_->coordinates()[i][c].transition.available()) pair[c]=RtcJumpReassessmentCause::transition_unavailable;
                else pair[c]=overlap ? RtcJumpReassessmentCause::support_still_overlaps : RtcJumpReassessmentCause::reassessed_measured;
            }
            out->coordinates_.push_back(pair);
        }
        return out;
    }
    const auto &evidence_handle() const noexcept {return evidence_;}
    const auto &coordinates() const noexcept {return coordinates_;}
    std::uint64_t consideration() const noexcept {return id_;}
    static constexpr bool hard_event_accepted=false,apply_authorized=false;
private:
    RtcJumpReassessmentDecision()=default;
    std::shared_ptr<const RtcJumpReassessmentEvidence> evidence_;
    std::vector<std::array<RtcJumpReassessmentCause,2>> coordinates_;
    std::uint64_t id_=0;
};
} // namespace citlali::pipeline

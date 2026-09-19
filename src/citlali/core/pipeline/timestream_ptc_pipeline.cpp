#include <citlali/core/pipeline/timestream_ptc_pipeline.h>
#include <set>
#include <chrono>
namespace citlali::pipeline {
namespace {
PtcApplied publish_requested_domain(const PtcGroupEvidence &g,PtcApplied compact) {
    PtcApplied out=std::move(compact);
    auto values=std::move(out.values);auto causes=std::move(out.causes);
    out.values=PtcMatrix::Zero(g.slots.size(),g.detectors.size());
    out.causes=PtcMask::Constant(g.slots.size(),g.detectors.size(),17); // excluded + no eligible segment input
    for(std::size_t d=0;d<g.fit_columns.size();++d) {
        out.values.col(g.fit_columns[d])=values.col(d);
        out.causes.col(g.fit_columns[d])=causes.col(d);
    }
    return out;
}
} // namespace
std::shared_ptr<const PtcEvidence> PtcEvidence::learn(PtcCalSource source,
    const ProcessingScanNativeProjection &projection,PtcSolverRequest request) {
    const auto grid=source.grid_handle();
    if(!projection.binding || projection.binding->parent_handle().get()!=grid->align_handle()->paired_handle().get())
        throw std::invalid_argument("PTC processing segmentation requires exact CAL/RTC native parent");
    auto out=std::shared_ptr<PtcEvidence>(new PtcEvidence{std::move(source)});out->processing_=projection.binding;
    const auto &first=grid->detectors().front();std::vector<std::size_t> detectors;
    const int array=out->source_.detector_factors().front().array;
    for(std::size_t d=0;d<grid->detectors().size();++d) {
        const auto &g=grid->detectors()[d];
        if(g.network!=first.network || g.first!=first.first || g.factor!=first.factor || g.scheduled_count!=first.scheduled_count ||
           out->source_.detector_factors()[d].array!=array)
            throw std::invalid_argument("PTC connected network grouping requires one within-array exact scheduled grid; array grouping not yet connected");
        detectors.push_back(d);
    }
    // The immutable binding is the authority, not a mutable projection mirror.
    std::vector<ProcessingScanNativeRecord> segments;
    const auto physical_runs=grid->align_handle()->paired_handle()->network(first.network).occurrence_axis().contiguous_runs();
    for(const auto &support:projection.binding->supports()) {
        if(support.native.network_id!=first.network)continue;
        for(const auto &run:physical_runs) {
            RtcEventRange native{std::max(support.native.first_native_row,run.first_native_row),
                std::min(support.native.past_last_native_row,run.past_last_native_row)};
            if(native.present()){ProcessingScanNativeRecord segment;segment.scan=support.scan;segment.science_native.push_back(native);segments.push_back(segment);}
        }
    }
    std::set<std::size_t> used;
    for(const auto &scan:segments)for(const auto native:scan.science_native) {
        PtcGroupEvidence group;group.scan=scan.scan;group.network=first.network;group.native=native;group.detectors=detectors;
        for(std::size_t slot=0;slot<first.scheduled_count;++slot) {
            const auto row=first.first+static_cast<TimestreamNativeRow>(slot)*first.factor;
            if(row>=native.first && row<native.past_last) {
                if(!used.insert(slot).second)throw std::invalid_argument("PTC processing segments overlap");
                group.slots.push_back(slot);
            }
        }
        if(group.slots.empty())continue;
        const auto preparing=std::chrono::steady_clock::now();
        PtcMatrix values=PtcMatrix::Zero(group.slots.size(),detectors.size());PtcMask mask=PtcMask::Zero(values.rows(),values.cols());
        for(std::size_t t=0;t<group.slots.size();++t)for(std::size_t d=0;d<detectors.size();++d) {
            const auto slot=group.slots[t];const auto occurrence=out->source_.occurrence(d,slot);
            // basis_fit_admission and application use the same binary finite
            // CAL eligibility for this request. Engineering influence alone
            // is not a veto; direct replacements/exclusions never enter a fit.
            const auto value=out->source_.value(d,slot);
            if(value && !out->source_.causes(d,slot) && !occurrence.representative_replaced && !occurrence.representative_excluded) {
                values(t,d)=*value;mask(t,d)=1;
            }
        }
        // Admission is upstream of either estimator, fixed for the whole fit.
        // Never remove a partially observed column because a fit dislikes it.
        for(std::size_t d=0;d<detectors.size();++d) {
            const auto count=static_cast<std::size_t>((mask.col(d)!=0).count());
            group.eligible_per_detector.push_back(count);
            if(count)group.fit_columns.push_back(d);
        }
        PtcMatrix admitted(values.rows(),group.fit_columns.size());
        PtcMask admitted_mask(values.rows(),group.fit_columns.size());
        for(std::size_t d=0;d<group.fit_columns.size();++d) {
            admitted.col(d)=values.col(group.fit_columns[d]);admitted_mask.col(d)=mask.col(group.fit_columns[d]);
        }
        group.input=PtcPrepared::prepare(admitted,admitted_mask);
        group.input.preparation_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-preparing).count();
        group.fit=ptc_learn(group.input,request);out->groups_.push_back(std::move(group));
    }
    if(out->groups_.empty())throw std::invalid_argument("PTC no processing science support intersects CAL schedule");
    return out;
}
std::shared_ptr<const PtcPlan> PtcPlan::consider(std::shared_ptr<const PtcEvidence> evidence,
    std::shared_ptr<const ValSnapshot> snapshot,std::uint64_t instance) {
    if(!evidence || !instance || evidence->source().val_snapshot_handle().get()!=snapshot.get())
        throw std::invalid_argument("PTC Consider requires exact CAL generation and nonzero plan identity");
    auto out=std::shared_ptr<PtcPlan>(new PtcPlan);out->evidence_=std::move(evidence);out->instance_=instance;return out;
}
std::shared_ptr<const PtcAppliedSignal> PtcAppliedSignal::apply(std::shared_ptr<const PtcPlan> plan,
    const PtcCalSource &source,std::shared_ptr<const ValSnapshot> snapshot) {
    if(!plan || plan->evidence_handle()->source().signal_handle().get()!=source.signal_handle().get() ||
        plan->snapshot_handle().get()!=snapshot.get() || source.val_snapshot_handle().get()!=snapshot.get())
        throw std::invalid_argument("PTC Apply requires frozen plan and exact CAL/VAL realization");
    auto out=std::shared_ptr<PtcAppliedSignal>(new PtcAppliedSignal);out->plan_=std::move(plan);
    for(const auto &g:out->plan_->evidence_handle()->groups()) {
        out->groups_.push_back(publish_requested_domain(g,ptc_apply(g.input,g.fit)));out->available_+=out->groups_.back().retained;
    }
    return out;
}
PtcApplied PtcAppliedSignal::response(std::size_t group,const PtcCalSource &source,const PtcMatrix &response) const {
    if(source.signal_handle().get()!=plan_->evidence_handle()->source().signal_handle().get() ||
       source.val_snapshot_handle().get()!=plan_->snapshot_handle().get())throw std::invalid_argument("PTC response requires exact CAL input binding");
    const auto &g=plan_->evidence_handle()->groups().at(group);
    if(response.rows()!=static_cast<Eigen::Index>(g.slots.size()) || response.cols()!=static_cast<Eigen::Index>(g.detectors.size()))
        throw std::invalid_argument("PTC response differs from full requested CAL grid");
    PtcMatrix compact(response.rows(),g.fit_columns.size());
    for(std::size_t d=0;d<g.fit_columns.size();++d)compact.col(d)=response.col(g.fit_columns[d]);
    auto result=publish_requested_domain(g,ptc_response(g.input,g.fit,compact));
    const auto &data=groups_.at(group);
    for(Eigen::Index t=0;t<result.values.rows();++t)for(Eigen::Index d=0;d<result.values.cols();++d)
        if(data.causes(t,d)) {
            if(!result.causes(t,d))--result.retained;
            result.causes(t,d)|=data.causes(t,d);result.values(t,d)=0;
        }
    return result;
}
std::shared_ptr<const ValPtcOutputFacts> ValPtcOutputFacts::preserve(std::shared_ptr<const PtcAppliedSignal> signal) {
    if(!signal)throw std::invalid_argument("PTC facts require their realized output");
    return std::shared_ptr<const ValPtcOutputFacts>(new ValPtcOutputFacts{std::move(signal)});
}
std::uint8_t ValPtcOutputFacts::at(const std::shared_ptr<const PtcAppliedSignal> &exact,std::size_t group,std::size_t time,std::size_t detector) const {
    if(exact.get()!=signal_.get())throw std::invalid_argument("PTC VAL query belongs to another realization");
    const auto &causes=signal_->groups().at(group).causes;
    if(time>=static_cast<std::size_t>(causes.rows()) || detector>=static_cast<std::size_t>(causes.cols()))throw std::out_of_range("PTC output occurrence outside segment");
    return causes(time,detector);
}
std::shared_ptr<const ValSnapshot> ValSnapshot::commit_ptc_output(std::shared_ptr<const ValSnapshot> base,std::shared_ptr<const ValPtcOutputFacts> facts) {
    if(!base || !facts || facts->signal_handle()->plan_handle()->snapshot_handle().get()!=base.get())
        throw std::invalid_argument("PTC VAL commit requires exact CAL generation");
    if(base->generation_.value==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("PTC VAL generation overflow");
    return std::shared_ptr<const ValSnapshot>(new ValSnapshot{std::move(base),{},{},{},std::move(facts)});
}
} // namespace citlali::pipeline

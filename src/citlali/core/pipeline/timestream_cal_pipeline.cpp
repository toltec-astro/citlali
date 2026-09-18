#include <citlali/core/pipeline/timestream_cal_pipeline.h>
#include <cfenv>

namespace citlali::pipeline {
std::shared_ptr<const CalEvidence> CalEvidence::learn(CalRtcSource source,
    std::shared_ptr<const AstRtcCoordinates> ast,std::shared_ptr<const CalWvrEvidence> wvr,
    std::shared_ptr<const CalAtmosphereSurface> atmosphere,std::string apt,
    std::vector<CalDetectorFactor> factors) {
    const auto grid=source.rtc_terminal_handle()->grid_handle();
    if(!ast || ast->grid_handle().get()!=grid.get() || !wvr || !atmosphere || apt.empty() ||
       wvr->scope()!=grid->align_handle()->scope() || wvr->source_identity()!=ast->telescope_identity() ||
       factors.size()!=grid->detectors().size())
        throw std::invalid_argument("CAL Learn requires exact RTC, AST, same-observation WVR and selected APT bindings");
    for(std::size_t d=0;d<factors.size();++d) {
        const auto &g=grid->detectors()[d];const auto &f=factors[d];
        const auto &expected=grid->align_handle()->paired_handle()->network(g.network).detector(g.detector);
        if(f.detector.network_id!=expected.network_id || f.detector.storage_column!=expected.storage_column ||
           f.detector.detector_occurrence_id!=expected.detector_occurrence_id ||
           f.detector.detector_association_record_id!=expected.detector_association_record_id ||
           f.detector.tone_or_channel_id!=expected.tone_or_channel_id || f.array<0 || f.array>2 ||
           f.array!=ast->geometry()[d].array ||
           !f.selected_row_identity.starts_with(apt+":row=") ||
           f.selected_row_identity.empty() || f.selected_row_identity!=ast->geometry()[d].selected_apt_row)
            throw std::invalid_argument("CAL detector factor differs from its acquisition/geometry binding");
    }
    auto out=std::shared_ptr<CalEvidence>(new CalEvidence{std::move(source)});
    out->ast_=std::move(ast);out->wvr_=std::move(wvr);out->atmosphere_=std::move(atmosphere);
    out->apt_identity_=std::move(apt);out->factors_=std::move(factors);
    double first=std::numeric_limits<double>::infinity(),last=-first;
    // Full original observation endpoints, before RTC or CAL masking and
    // including a last native occurrence not on the decimated schedule.
    for(auto network:grid->align_handle()->participant_network_ids()) {
        const auto &axis=grid->align_handle()->paired_handle()->network(network).occurrence_axis();
        first=std::min(first,grid->align_handle()->occurrence_assignment(network,axis.first_native_row()).assigned_time_unix_sec);
        last=std::max(last,grid->align_handle()->occurrence_assignment(network,axis.past_last_native_row()-1).assigned_time_unix_sec);
    }
    if(out->wvr_->single_reading()) {
        const auto &interval=out->wvr_->observation_interval();
        if(!interval || interval->first_unix_sec!=first || interval->last_unix_sec!=last)
            throw std::invalid_argument("single WVR reading requires the exact full original RTC observation interval");
    }
    out->quality_=out->wvr_->quality(first,last);return out;
}
std::shared_ptr<const CalPlan> CalPlan::consider(std::shared_ptr<const CalEvidence> evidence,
    std::shared_ptr<const ValSnapshot> val,std::uint64_t instance) {
    if(!evidence || !instance || val.get()!=evidence->source().val_snapshot_handle().get() || std::fegetround()!=FE_TONEAREST)
        throw std::invalid_argument("CAL Consider requires exact frozen RTC output VAL and nonzero plan identity");
    auto out=std::shared_ptr<CalPlan>(new CalPlan);out->evidence_=std::move(evidence);out->instance_=instance;
    const auto &e=*out->evidence_;const auto &grid=e.source().rtc_terminal_handle()->grid_handle();
    for(std::size_t d=0;d<grid->detectors().size();++d) {
        out->entries_.emplace_back();auto &entries=out->entries_.back();entries.reserve(grid->detectors()[d].scheduled_count);
        const auto &factor=e.factors()[d];
        const bool good_factor=factor.uniquely_matched && factor.flxscale_mJy_beam_per_x &&
            std::isfinite(*factor.flxscale_mJy_beam_per_x) && *factor.flxscale_mJy_beam_per_x!=0;
        for(std::size_t s=0;s<grid->detectors()[d].scheduled_count;++s) {
            const auto fact=val->committed_rtc_output_facts_handle()->at(RtcOutputGrid::val_target(grid,d,s,NativeReadoutCoordinate::x));
            const auto &occ=fact.occurrence;Entry entry;
            if(!fact.numerical_available)entry.causes|=cal_rtc_unavailable;
            if(occ.representative_replaced || occ.representative_excluded)entry.causes|=cal_direct_replacement_or_exclusion;
            if(!good_factor)entry.causes|=cal_invalid_factor;
            const auto direction=e.ast_handle()->at(d,s);
            if(!direction)entry.causes|=cal_pointing_unavailable;
            const auto wvr=e.wvr_handle()->at(occ.representative.assigned_time_unix_sec);entry.wvr_cause=wvr.cause;
            if(!wvr.tau225)entry.causes|=(wvr.cause==CalWvrCause::negative || wvr.cause==CalWvrCause::nonfinite)?
                cal_invalid_atmosphere:cal_outside_supported_calibration;
            std::optional<double> correction;
            if(wvr.tau225 && direction) {
                correction=e.atmosphere_handle()->correction(factor.array,*wvr.tau225,direction->telescope_elevation_deg);
                if(!correction)entry.causes|=cal_outside_supported_calibration;
            }
            if(!entry.causes) {
                const double multiplier=*factor.flxscale_mJy_beam_per_x**correction;
                if(std::isfinite(multiplier) && multiplier!=0)entry.multiplier=multiplier;
                else entry.causes|=cal_numeric_failure;
            }
            entries.push_back(entry);
        }
    }
    return out;
}
std::shared_ptr<const CalAppliedSignal> CalAppliedSignal::apply(std::shared_ptr<const CalPlan> plan,
    const CalRtcSource &source,std::shared_ptr<const ValSnapshot> val) {
    if(!plan || source.rtc_terminal_handle().get()!=plan->evidence_handle()->source().rtc_terminal_handle().get() ||
       val.get()!=plan->snapshot_handle().get() || std::fegetround()!=FE_TONEAREST)
        throw std::invalid_argument("CAL Apply requires its frozen plan, original conditioned RTC source and exact VAL");
    auto out=std::shared_ptr<CalAppliedSignal>(new CalAppliedSignal);out->plan_=std::move(plan);
    for(std::size_t d=0;d<out->plan_->entries().size();++d) {
        out->cells_.emplace_back();auto &cells=out->cells_.back();cells.reserve(out->plan_->entries()[d].size());
        for(std::size_t s=0;s<out->plan_->entries()[d].size();++s) {
            const auto &entry=out->plan_->entries()[d][s];Cell cell;cell.causes=entry.causes;
            if(entry.multiplier) {
                const auto x=source.conditioned_x(d,s);
                if(!x)throw std::logic_error("CAL frozen admission contradicts exact RTC source");
                cell.value=*x**entry.multiplier;
                if(!std::isfinite(cell.value))cell.causes|=cal_numeric_failure;
                else ++out->available_;
            }
            cells.push_back(cell);
        }
    }
    return out;
}
std::optional<double> CalAppliedSignal::value(std::size_t d,std::size_t s) const {
    const auto &cell=cells_.at(d).at(s);return cell.causes?std::nullopt:std::optional<double>{cell.value};
}
std::uint16_t CalAppliedSignal::causes(std::size_t d,std::size_t s) const {return cells_.at(d).at(s).causes;}
std::shared_ptr<const ValCalOutputFacts> ValCalOutputFacts::preserve(std::shared_ptr<const CalAppliedSignal> signal) {
    if(!signal)throw std::invalid_argument("CAL facts require their applied realization");
    return std::shared_ptr<const ValCalOutputFacts>(new ValCalOutputFacts{std::move(signal)});
}
std::uint16_t ValCalOutputFacts::at(const std::shared_ptr<const CalAppliedSignal> &exact,std::size_t d,std::size_t s) const {
    if(exact.get()!=signal_.get())throw std::invalid_argument("CAL VAL query belongs to another calibrated realization");
    return signal_->causes(d,s);
}
std::shared_ptr<const ValSnapshot> ValSnapshot::commit_cal_output(std::shared_ptr<const ValSnapshot> base,
    std::shared_ptr<const ValCalOutputFacts> facts) {
    if(!base || !facts || facts->signal_handle()->plan_handle()->snapshot_handle().get()!=base.get())
        throw std::invalid_argument("CAL VAL commit requires the exact admitted RTC generation");
    if(base->generation_.value==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("CAL VAL generation overflow");
    return std::shared_ptr<const ValSnapshot>(new ValSnapshot{std::move(base),{}, {},std::move(facts)});
}
} // namespace citlali::pipeline

#pragma once
// Private, inert known-truth comparison. The production evidence chain is read
// without mutation. Explicit anchor probes never become producer candidates.
namespace {
using Node = YAML::Node;
Node num(double x) {return std::isfinite(x) ? Node(x) : Node(YAML::NodeType::Null);}
Node range_node(RtcEventRange r) {Node n(YAML::NodeType::Sequence);n.push_back(r.first);n.push_back(r.past_last);return n;}
Node ranges_node(const std::vector<RtcEventRange> &rs) {Node n(YAML::NodeType::Sequence);for(auto r:rs)n.push_back(range_node(r));return n;}
Node rows_node(const RtcJumpFitRows &rs) {Node n(YAML::NodeType::Sequence);for(const auto &side:rs)n.push_back(ranges_node(side));return n;}
Node fit_node(const RtcEventCubicFit &f) {
    Node n;n["available"]=f.available();n["cause"]=int(f.cause);n["iterations"]=f.iterations;
    n["offset"]=num(f.offset);n["scale"]=num(f.scale);n["huber_loss"]=num(f.huber_loss);
    for(double x:f.coefficients)n["coefficients"].push_back(num(x));return n;
}
Node recovery_node(const RtcEventRecovery &r) {
    Node n;n["cause"]=int(r.cause);n["affected"]=range_node(r.affected);
    n["confirmation"]=range_node(r.confirmation);n["examined"]=range_node(r.examined);return n;
}
Node transition_node(const RtcJumpTransition &t) {
    Node n;n["available"]=t.available();n["cause"]=int(t.cause);n["rows"]=range_node(t.affected);
    n["begin"]=t.affected.present()?num(t.physical_bound.begin_unix_sec):num(NAN);
    n["end"]=t.affected.present()?num(t.physical_bound.end_unix_sec):num(NAN);
    n["frozen_scale"]=num(t.frozen_residual_scale);n["examined"]=range_node(t.examined);
    n["examined_rows"]=t.examined_rows;n["invalid_rows"]=t.invalid_rows;n["excluded_rows"]=t.excluded_rows;
    n["observation_truncated"]=t.observation_truncated;n["acquisition_truncated"]=t.acquisition_truncated;
    n["exceeds_fitting_exclusion"]=t.exceeds_fitting_exclusion;n["multiple_candidate_edges"]=t.multiple_candidate_edges;
    for(const auto &c:t.confirmations){Node q;q["rows"]=range_node(c.rows);q["begin"]=c.rows.present()?num(c.physical.begin_unix_sec):num(NAN);q["end"]=c.rows.present()?num(c.physical.end_unix_sec):num(NAN);q["also_matches_other_reference"]=c.also_matches_other_reference;n["confirmations"].push_back(q);}
    return n;
}
std::string render(const Node &n) {
    YAML::Emitter e;e.SetDoublePrecision(17);e.SetSeqFormat(YAML::Flow);e.SetMapFormat(YAML::Flow);e<<n;
    if(!e.good())throw std::runtime_error("diagnostic YAML serialization failed");return e.c_str();
}
void require_same(const Node &a,const Node &b,const char *what) {
    if(render(a)!=render(b))throw std::runtime_error(std::string("explicit-anchor parity failed: ")+what);
}
struct DiagnosticVersion {
    RtcEventCoordinateBackground primary;
    RtcJumpShortFit shorter;
    RtcJumpFitRows primary_rows,short_rows;
    RtcEventRecovery recovery;
    RtcJumpTransition actual_transition,ungated_transition;
    RtcJumpConsistencyCause consistency=RtcJumpConsistencyCause::primary_not_passed;
    bool retained=false,transition_requested=false;
    int saved_terminal=-1;
    std::size_t primary_overlap=0,short_overlap=0;
};
Node version_node(const DiagnosticVersion &v,double sigma) {
    Node n;n["primary_available"]=v.primary.available();n["primary_support_cause"]=int(v.primary.support_cause);
    n["primary_pre_scale_fit"]=fit_node(v.primary.pre_scale_fit);n["primary_cubic"]=fit_node(v.primary.cubic);
    n["primary_offset"]=fit_node(v.primary.cubic_with_offset);n["short_available"]=v.shorter.available();
    n["short_cause"]=int(v.shorter.cause);n["short_pre_scale_fit"]=fit_node(v.shorter.pre_scale_fit);n["short_offset"]=fit_node(v.shorter.cubic_with_offset);
    n["primary_rows"]=rows_node(v.primary_rows);n["short_rows"]=rows_node(v.short_rows);
    n["consistency_cause"]=int(v.consistency);n["consistency_evaluated"]=v.primary.available() && v.shorter.available();
    n["sigma_delta_comparison_tolerance"]=num(sigma);n["offset_uncertainty_estimated"]=false;
    n["recovery"]=recovery_node(v.recovery);n["transition_requested"]=v.transition_requested;
    n["actual_transition"]=transition_node(v.actual_transition);n["ungated_transition_probe"]=transition_node(v.ungated_transition);
    n["retained"]=v.retained;n["saved_terminal"]=v.saved_terminal;n["primary_overlap"]=v.primary_overlap;n["short_overlap"]=v.short_overlap;
    if(!v.primary.available() || !v.shorter.available())n["first_decisive_stage"]="fit_availability";
    else if(v.consistency!=RtcJumpConsistencyCause::passes)n["first_decisive_stage"]="consistency";
    else if(v.recovery.recovered())n["first_decisive_stage"]="confirmed_recovery";
    else if(!v.actual_transition.available())n["first_decisive_stage"]="transition_measurement";
    else if(!v.retained)n["first_decisive_stage"]="remaining_paired_overlap";
    else n["first_decisive_stage"]="retained";
    return n;
}
// The known-truth exclusion replaces the target's inferred exclusion. Preserve
// original outer limits, basis, producer validity and every other candidate's
// existing guard, including a neighboring injected spike in the same group.
RtcJumpFitRows truth_rows(const RtcSpikeEvidence &spikes,const RtcAssessedEvent &e,
        std::size_t c,double flank,RtcEventRange truth,const std::vector<RtcEventRange> &mask) {
    const auto &net=spikes.input_handle()->network(e.network);const auto &axis=net.occurrence_axis();
    const double low=axis.occurrence(e.trial_exclusion.first).integration_support.begin_unix_sec-flank;
    const double high=axis.occurrence(e.trial_exclusion.past_last-1).integration_support.end_unix_sec+flank;
    RtcJumpFitRows rows;const auto run=rtc_event_assessment_detail::run_for(axis,truth.first);
    for(auto row=run.first;row<run.past_last;++row) {
        const auto &cell=axis.occurrence(row).integration_support;
        if(cell.begin_unix_sec<low || cell.end_unix_sec>high || rtc_event_assessment_detail::contains(mask,row))continue;
        if(!net.state(rtc_event_assessment_detail::coord(c),row,e.detector).valid() || !std::isfinite(net.value(rtc_event_assessment_detail::coord(c),row,e.detector)))continue;
        if(row<truth.first)rtc_jump_reassessment_detail::append(rows[0],row);
        else if(row>=truth.past_last)rtc_jump_reassessment_detail::append(rows[1],row);
    }return rows;
}
RtcJumpShortFit as_short(const RtcEventCoordinateBackground &fit) {
    RtcJumpShortFit s;s.support=fit.support;s.pre_scale_fit=fit.pre_scale_fit;s.cubic_with_offset=fit.cubic_with_offset;
    s.cause=fit.support_cause==RtcEventFitCause::none ? RtcJumpShortFitCause::none : RtcJumpShortFitCause::insufficient_samples;
    return s;
}
struct DiagnosticTrial {const char *name;int ramp_cells=0;bool jump=false,spike=false;int pulse_cells=0;};
int diagnose_injections(const Background &original,const std::array<double,2> &sigma,bool synthetic) {
    const Attempt unmodified(original);
    const auto center=static_cast<TimestreamNativeRow>(original.times.size()/2);
    std::vector<DiagnosticTrial> trials{{"unmodified"},{"sharp_step",0,true},{"finite_3_cells",3,true},
        {"finite_12_cells",12,true},{"sharp_plus_neighbor_spike",0,true,true},{"spike_only_control",0,false,true}};
    if(synthetic){trials.push_back({"finite_pulse_3_cells",0,false,false,3});trials.push_back({"finite_pulse_12_cells",0,false,false,12});}
    for(const auto &trial:trials) {
        auto input=original;input.identity=original.identity+":"+trial.name;
        if(trial.jump)for(std::size_t i=center;i<input.times.size();++i){const double f=trial.ramp_cells?std::min(1.,(double(i-center)+.5)/trial.ramp_cells):i==std::size_t(center)?.5:1.;input.x[i]+=20*sigma[0]*f;input.r[i]-=20*sigma[1]*f;}
        if(trial.spike){const auto i=trial.jump?center+25:center;input.x[i]+=12*sigma[0];input.r[i]+=12*sigma[1];}
        if(trial.pulse_cells)for(auto i=center;i<=center+trial.pulse_cells;++i){const double f=(i==center || i==center+trial.pulse_cells)?.5:1.;input.x[i]+=20*sigma[0]*f;input.r[i]-=20*sigma[1]*f;}
        const RtcEventRange truth{center,center+(trial.pulse_cells?trial.pulse_cells+1:std::max(1,trial.ramp_cells))};
        Attempt a(input);const auto &spikes=*a.spikes;const auto &axis=spikes.input_handle()->network(0).occurrence_axis();
        const auto run=axis.contiguous_runs().front();
        std::vector<std::size_t> associated,indices;std::array<std::size_t,2> detected{};
        for(std::size_t i=0;i<spikes.candidates().size();++i){indices.push_back(i);const auto &s=spikes.candidates()[i];if(s.earlier_row<truth.past_last && s.later_row>=truth.first){associated.push_back(i);++detected[s.coordinate==NativeReadoutCoordinate::x?0:1];}}
        std::sort(indices.begin(),indices.end(),[&](auto i,auto j){return std::tie(spikes.candidates()[i].earlier_row,i)<std::tie(spikes.candidates()[j].earlier_row,j);});
        std::vector<RtcEventRange> truth_mask{truth};
        for(auto i:indices)if(std::find(associated.begin(),associated.end(),i)==associated.end())truth_mask.push_back(rtc_event_assessment_detail::trial(axis,{run.first_native_row,run.past_last_native_row},spikes.candidates()[i]));
        truth_mask=rtc_event_assessment_detail::merge(std::move(truth_mask));
        Node result;result["schema"]="rtc-jump-loss-diagnostic-v1";result["background_identity"]=original.identity;result["trial"]=trial.name;
        result["jump_injected"]=trial.jump;result["pulse_injected"]=trial.pulse_cells>0;result["truth_cells"]=range_node(truth);
        const double tb=trial.ramp_cells?input.cells[center].begin_unix_sec:input.times[center];
        const double te=trial.ramp_cells?input.cells[truth.past_last-1].end_unix_sec:trial.pulse_cells?input.times[center+trial.pulse_cells]:tb;
        result["truth_begin"]=tb;result["truth_end"]=te;result["truth_duration_seconds"]=te-tb;
        for(std::size_t c=0;c<2;++c){result["injected_offset"].push_back(trial.jump?(c?-20:20)*sigma[c]:0.);result["injected_excursion"].push_back((trial.jump || trial.pulse_cells)?(c?-20:20)*sigma[c]:trial.spike?12*sigma[c]:0.);result["sigma_delta_unmodified"].push_back(sigma[c]);result["candidate_edges_at_truth"].push_back(detected[c]);}
        result["all_candidate_edges"]=spikes.candidates().size();result["groups"]=Node(YAML::NodeType::Sequence);
        result["all_candidates"]=Node(YAML::NodeType::Sequence);
        for(std::size_t i=0;i<spikes.candidates().size();++i){const auto &s=spikes.candidates()[i];Node q;q["index"]=i;q["earlier"]=s.earlier_row;q["later"]=s.later_row;q["coordinate"]=s.coordinate==NativeReadoutCoordinate::x?"x":"r";q["noise_block"]=s.noise_block_index;result["all_candidates"].push_back(q);}
        const auto &re=*a.final->evidence_handle();const auto &rr=*re.request_handle();const auto &refit=*rr.refit_handle();
        const auto &request=*refit.request_handle();const auto &audit=*request.audit_handle();
        const auto &consistency=*a.transition->request_handle()->consistency_handle();
        const auto &shorts=*consistency.evidence_handle();const auto &amplitude=*shorts.amplitude_handle();
        std::vector<std::size_t> events;
        for(std::size_t i=0;i<a.assessment->events().size();++i)for(auto c:a.assessment->events()[i].candidates)
            if(std::find(associated.begin(),associated.end(),c)!=associated.end()){events.push_back(i);break;}
        // Missing candidates stay missing in end-to-end metrics. One explicitly
        // supplied location is a separate downstream diagnostic, never a repair.
        const bool forced=events.empty();if(forced)events.push_back(std::numeric_limits<std::size_t>::max());
        std::size_t parity_checks=0;
        for(auto id:events) {
            RtcAssessedEvent event;std::array<RtcJumpShortFit,2> old_short;std::array<double,2> tolerance{NAN,NAN};
            RtcSpikeCandidate seed;std::vector<RtcSpikeCandidate> members;
            if(!forced){event=a.assessment->events()[id];seed=spikes.candidates()[event.seed];for(auto i:event.candidates)members.push_back(spikes.candidates()[i]);for(std::size_t c=0;c<2;++c){old_short[c]=shorts.coordinates()[id][c];tolerance[c]=amplitude.coordinates()[id][c].sigma_delta;}}
            else {
                seed.earlier_row=center-1;seed.later_row=center;event.network=0;event.detector=0;event.seeded={true,true};
                for(std::size_t c=0;c<2;++c){auto s=seed;s.coordinate=rtc_event_assessment_detail::coord(c);members.push_back(s);}
                event.trial_exclusion=rtc_event_assessment_detail::trial(axis,{run.first_native_row,run.past_last_native_row},seed);
                rtc_event_assessment_detail::background(spikes,indices,event,{run.first_native_row,run.past_last_native_row});
                for(std::size_t c=0;c<2;++c){for(const auto &block:spikes.blocks())if(block.first<=seed.earlier_row && block.past_last>seed.earlier_row && block.coordinates[c].available())tolerance[c]=block.coordinates[c].scale;
                    if(event.background[c].available() && rtc_jump_detail::amplitude(event.background[c].cubic_with_offset.offset,tolerance[c])==RtcJumpAmplitudeCause::passes){RtcJumpFitCounts count;old_short[c]=rtc_jump_detail::short_fit(spikes,event,c,run,count);}}
            }
            std::vector<RtcEventRange> neighbors;
            if(!forced)neighbors=rtc_jump_transition_detail::neighbor_masks(spikes,event,run,indices);
            else for(auto i:indices)neighbors.push_back(rtc_event_assessment_detail::trial(axis,{run.first_native_row,run.past_last_native_row},spikes.candidates()[i]));
            neighbors=rtc_event_assessment_detail::merge(std::move(neighbors));
            std::array<DiagnosticVersion,2> before,current,oracle;
            auto found=std::find_if(audit.audits().begin(),audit.audits().end(),[&](const auto &x){return x.event==id;});
            const auto ai=static_cast<std::size_t>(std::distance(audit.audits().begin(),found));
            const bool has_audit=found!=audit.audits().end();
            bool requested=has_audit && request.selections()[ai].requested;
            std::vector<RtcEventRange> current_mask=requested?request.selections()[ai].paired_mask:event.neighbor_exclusions;
            current_mask.push_back(event.trial_exclusion);current_mask=rtc_event_assessment_detail::merge(std::move(current_mask));
            for(std::size_t c=0;c<2;++c) {
                auto &v=before[c];v.primary=event.background[c];v.shorter=old_short[c];
                if(v.primary.available())v.primary_rows=rtc_jump_reassessment_detail::admitted(spikes,event,c,v.primary.support,2);
                if(v.shorter.available())v.short_rows=rtc_jump_reassessment_detail::admitted(spikes,event,c,v.shorter.support,1);
                v.recovery=diagnostic_recover_at(spikes,event,{run.first_native_row,run.past_last_native_row},c,seed,members);
                v.ungated_transition=diagnostic_measure_at(spikes,event,c,run,neighbors,seed,members);
                if(v.primary.available() && v.shorter.available())v.consistency=rtc_jump_detail::consistency(v.primary.cubic_with_offset.offset,v.shorter.cubic_with_offset.offset,tolerance[c]);
                if(!forced){require_same(recovery_node(v.recovery),recovery_node(event.recovery[c]),"original recovery");require_same(transition_node(v.ungated_transition),transition_node(rtc_jump_transition_detail::measure(spikes,event,c,run,neighbors)),"original transition");parity_checks+=2;
                    v.actual_transition=a.transition->coordinates()[id][c];v.transition_requested=a.transition->request_handle()->coordinates()[id][c]==RtcJumpTransitionRequestCause::requested;v.consistency=consistency.coordinates()[id][c].cause;}
                else {v.transition_requested=v.consistency==RtcJumpConsistencyCause::passes && !v.recovery.recovered();if(v.transition_requested)v.actual_transition=v.ungated_transition;}
                v.retained=v.actual_transition.available();current[c]=v;
                if(requested){auto &q=current[c];const auto &r=refit.coordinates()[ai][c];q.primary=r.primary;q.shorter=r.shorter;q.primary_rows=r.primary_rows;q.short_rows=r.short_rows;q.recovery=r.recovery;q.consistency=rr.selections()[ai][c].consistency;q.transition_requested=rr.selections()[ai][c].requested;q.actual_transition=re.coordinates()[ai][c].transition;q.saved_terminal=int(a.final->coordinates()[ai][c]);q.retained=q.saved_terminal==0 || q.saved_terminal==7;q.primary_overlap=re.coordinates()[ai][c].primary_overlap;q.short_overlap=re.coordinates()[ai][c].short_overlap;
                    auto revised=event;revised.background[c]=q.primary;q.ungated_transition=diagnostic_measure_at(spikes,revised,c,run,neighbors,seed,members);
                    require_same(recovery_node(diagnostic_recover_at(spikes,revised,{run.first_native_row,run.past_last_native_row},c,seed,members)),recovery_node(q.recovery),"refit recovery");require_same(transition_node(q.ungated_transition),transition_node(rtc_jump_transition_detail::measure(spikes,revised,c,run,neighbors)),"refit transition");parity_checks+=2;}
                auto &o=oracle[c];o.primary_rows=truth_rows(spikes,event,c,2,truth,truth_mask);o.short_rows=truth_rows(spikes,event,c,1,truth,truth_mask);RtcJumpRefitCounts count;
                o.primary=rtc_jump_reassessment_detail::refit(spikes,event,c,o.primary_rows,v.primary.pre_scale_fit,true,count);
                if(v.shorter.pre_scale_fit.available())o.shorter=as_short(rtc_jump_reassessment_detail::refit(spikes,event,c,o.short_rows,v.shorter.pre_scale_fit,false,count));
                auto revised=event;revised.background[c]=o.primary;
                o.recovery=diagnostic_recover_at(spikes,revised,{run.first_native_row,run.past_last_native_row},c,seed,members);
                o.ungated_transition=diagnostic_measure_at(spikes,revised,c,run,neighbors,seed,members);
                if(!forced){require_same(recovery_node(o.recovery),recovery_node(rtc_event_assessment_detail::recover(spikes,revised,{run.first_native_row,run.past_last_native_row},c)),"truth recovery");require_same(transition_node(o.ungated_transition),transition_node(rtc_jump_transition_detail::measure(spikes,revised,c,run,neighbors)),"truth transition");parity_checks+=2;}
                if(o.primary.available() && o.shorter.available())o.consistency=rtc_jump_detail::consistency(o.primary.cubic_with_offset.offset,o.shorter.cubic_with_offset.offset,tolerance[c]);
                o.transition_requested=o.consistency==RtcJumpConsistencyCause::passes && !o.recovery.recovered();if(o.transition_requested)o.actual_transition=o.ungated_transition;
            }
            if(forced) {
                std::vector<RtcEventRange> old_bounds;for(const auto &v:before)if(v.actual_transition.available())old_bounds.push_back(v.actual_transition.affected);
                old_bounds=rtc_event_assessment_detail::merge(std::move(old_bounds));
                for(const auto &v:before)requested|=rtc_jump_reassessment_detail::overlap(v.primary_rows,old_bounds)>0 || rtc_jump_reassessment_detail::overlap(v.short_rows,old_bounds)>0;
                if(requested) {
                    current_mask.push_back(event.trial_exclusion);current_mask.insert(current_mask.end(),old_bounds.begin(),old_bounds.end());current_mask=rtc_event_assessment_detail::merge(std::move(current_mask));
                    std::vector<RtcEventRange> new_bounds;
                    for(std::size_t c=0;c<2;++c) {
                        auto &v=current[c];v=DiagnosticVersion{};RtcJumpRefitCounts count;
                        v.primary_rows=rtc_jump_reassessment_detail::subtract(before[c].primary_rows,current_mask);v.short_rows=rtc_jump_reassessment_detail::subtract(before[c].short_rows,current_mask);
                        if(before[c].primary.available())v.primary=rtc_jump_reassessment_detail::refit(spikes,event,c,v.primary_rows,before[c].primary.pre_scale_fit,true,count);
                        if(before[c].shorter.available())v.shorter=as_short(rtc_jump_reassessment_detail::refit(spikes,event,c,v.short_rows,before[c].shorter.pre_scale_fit,false,count));
                        auto revised=event;revised.background[c]=v.primary;
                        v.recovery=diagnostic_recover_at(spikes,revised,{run.first_native_row,run.past_last_native_row},c,seed,members);v.ungated_transition=diagnostic_measure_at(spikes,revised,c,run,neighbors,seed,members);
                        if(v.primary.available() && v.shorter.available())v.consistency=rtc_jump_detail::consistency(v.primary.cubic_with_offset.offset,v.shorter.cubic_with_offset.offset,tolerance[c]);
                        v.transition_requested=v.primary.available() && v.shorter.available() && v.consistency==RtcJumpConsistencyCause::passes && !v.recovery.recovered();
                        if(v.transition_requested)v.actual_transition=v.ungated_transition;
                        if(v.actual_transition.available())new_bounds.push_back(v.actual_transition.affected);
                    }
                    new_bounds=rtc_event_assessment_detail::merge(std::move(new_bounds));bool conflict=false;
                    for(auto &v:current){v.primary_overlap=rtc_jump_reassessment_detail::overlap(v.primary_rows,new_bounds);v.short_overlap=rtc_jump_reassessment_detail::overlap(v.short_rows,new_bounds);conflict|=v.primary_overlap || v.short_overlap;}
                    for(auto &v:current)v.retained=v.actual_transition.available() && !conflict;
                }
            }
            std::vector<RtcEventRange> measured;for(const auto &o:oracle)if(o.actual_transition.available())measured.push_back(o.actual_transition.affected);
            measured=rtc_event_assessment_detail::merge(std::move(measured));bool overlap=false;
            for(auto &o:oracle){o.primary_overlap=rtc_jump_reassessment_detail::overlap(o.primary_rows,measured);o.short_overlap=rtc_jump_reassessment_detail::overlap(o.short_rows,measured);overlap|=o.primary_overlap || o.short_overlap;}
            for(auto &o:oracle)o.retained=o.actual_transition.available() && !overlap;
            Node group;group["event"]=forced?Node(YAML::NodeType::Null):Node(id);group["forced_location"]=forced;
            group["origin"]=event.origin;group["time_scale"]=event.time_scale;group["trial_exclusion"]=range_node(event.trial_exclusion);
            auto original_mask=event.neighbor_exclusions;original_mask.push_back(event.trial_exclusion);original_mask=rtc_event_assessment_detail::merge(std::move(original_mask));
            group["original_exclusions"]=ranges_node(original_mask);group["inferred_reassessment_exclusions"]=ranges_node(current_mask);
            group["truth_exclusions"]=ranges_node(truth_mask);group["refit_requested"]=requested;
            group["candidate_members"]=Node(YAML::NodeType::Sequence);for(const auto &m:members){Node z;z["coordinate"]=m.coordinate==NativeReadoutCoordinate::x?"x":"r";z["earlier"]=m.earlier_row;z["later"]=m.later_row;group["candidate_members"].push_back(z);}
            for(std::size_t c=0;c<2;++c){Node coord;coord["coordinate"]=c?"r":"x";coord["original"]=version_node(before[c],tolerance[c]);coord["current"]=version_node(current[c],tolerance[c]);coord["truth_fit"]=version_node(oracle[c],tolerance[c]);
                for(const auto &[name,version]:std::array<std::pair<const char *,const DiagnosticVersion *>,3>{{{"original",&before[c]},{"current",&current[c]},{"truth_fit",&oracle[c]}}}) {
                    RtcJumpRefitCounts count;
                    const auto control=rtc_jump_reassessment_detail::refit(*unmodified.spikes,event,c,version->primary_rows,version->primary.pre_scale_fit,true,count);
                    coord[name]["matched_uninjected_primary"]=fit_node(control.cubic_with_offset);
                    const auto short_control=rtc_jump_reassessment_detail::refit(*unmodified.spikes,event,c,version->short_rows,version->shorter.pre_scale_fit,false,count);
                    coord[name]["matched_uninjected_short"]=fit_node(short_control.cubic_with_offset);
                }
                const auto amplitude_cause=forced?rtc_jump_detail::amplitude(before[c].primary.cubic_with_offset.offset,tolerance[c]):amplitude.coordinates()[id][c].cause;
                coord["original_amplitude_cause"]=int(amplitude_cause);
                if(amplitude_cause!=RtcJumpAmplitudeCause::passes){coord["original"]["first_decisive_stage"]="original_amplitude";if(!requested)coord["current"]["first_decisive_stage"]="original_amplitude";}
                group["coordinates"].push_back(coord);}
            result["groups"].push_back(group);
        }
        result["explicit_anchor_parity_checks"]=parity_checks;result["end_to_end_candidate_detected"]=!associated.empty();
        result["matched_uninjected_fit_scope"]="same original background, exact per-version rows/basis/frozen scales; descriptive injection increment, no assumption of event-free real background";
        result["known_location_diagnostic_only"]=forced;result["hard_event_accepted"]=false;result["apply_authorized"]=false;
        for(std::size_t i=0;i<input.times.size();++i){Node s;s.push_back(input.source_rows[i]);s.push_back(input.times[i]);s.push_back(input.cells[i].begin_unix_sec);s.push_back(input.cells[i].end_unix_sec);s.push_back(input.x[i]);s.push_back(input.xs[i].valid());s.push_back(input.r[i]);s.push_back(input.rs[i].valid());s.push_back(original.x[i]);s.push_back(original.r[i]);result["samples"].push_back(s);}
        std::cout<<"---\n"<<render(result)<<"\n...\n";
        if(!std::cout)throw std::runtime_error("diagnostic output write failed");
    }
    std::cout.flush();if(!std::cout)throw std::runtime_error("diagnostic output flush failed");return 0;
}
} // namespace

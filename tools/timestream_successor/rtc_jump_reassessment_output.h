#pragma once
// Private diagnostic serialization; no application route or policy ownership.
namespace {
void reassessment_ranges(std::ostream &out,const std::vector<pipeline::RtcEventRange> &ranges) {
    out<<'[';bool comma=false;for(auto r:ranges){if(comma)out<<',';comma=true;out<<'['<<r.first<<','<<r.past_last<<']';}out<<']';
}
void reassessment_rows(std::ostream &out,const pipeline::RtcJumpFitRows &rows) {
    out<<'[';reassessment_ranges(out,rows[0]);out<<',';reassessment_ranges(out,rows[1]);out<<']';
}
void reassessment_bound(std::ostream &out,const pipeline::RtcJumpTransition &b) {
    out<<"{\"available\":"<<(b.available()?"true":"false")<<",\"cause\":"<<int(b.cause)<<",\"rows\":["<<b.affected.first<<','<<b.affected.past_last<<"],\"begin\":";
    number(out,b.affected.present()?b.physical_bound.begin_unix_sec:NAN);out<<",\"end\":";number(out,b.affected.present()?b.physical_bound.end_unix_sec:NAN);
    out<<",\"confirmations\":[";
    for(std::size_t side=0;side<2;++side){if(side)out<<',';const auto &q=b.confirmations[side];out<<"{\"rows\":["<<q.rows.first<<','<<q.rows.past_last<<"],\"begin\":";number(out,q.rows.present()?q.physical.begin_unix_sec:NAN);out<<",\"end\":";number(out,q.rows.present()?q.physical.end_unix_sec:NAN);out<<'}';}
    out<<"]}";
}
void write_reassessment(const fs::path &output,const pipeline::RtcJumpReassessmentDecision &decision) {
    const auto &e=*decision.evidence_handle();const auto &rq=*e.request_handle();const auto &refit=*rq.refit_handle();
    const auto &request=*refit.request_handle();const auto &audit=*request.audit_handle();
    const auto &a=pipeline::rtc_jump_reassessment_detail::assessment(*audit.parent_handle());
    const auto &old_short=audit.parent_handle()->request_handle()->consistency_handle()->evidence_handle()->coordinates();
    const auto &amplitude=audit.parent_handle()->request_handle()->consistency_handle()->evidence_handle()->amplitude_handle()->coordinates();
    std::ofstream out(output/"jump-reassessment.jsonl");require(bool(out),"cannot open reassessment output");
    for(std::size_t i=0;i<audit.audits().size();++i) {
        const auto &s=audit.audits()[i];const auto &event=a.events()[s.event];
        out<<"{\"event\":"<<s.event<<",\"network\":"<<event.network<<",\"detector\":"<<event.detector<<",\"refit_requested\":"<<(request.selections()[i].requested?"true":"false")<<",\"paired_mask\":";
        reassessment_ranges(out,request.selections()[i].paired_mask);out<<",\"measured_union\":";reassessment_ranges(out,s.measured_union);
        out<<",\"coordinates\":[";
        for(std::size_t c=0;c<2;++c) {
            if(c)out<<',';const auto &r=refit.coordinates()[i][c];const auto &m=e.coordinates()[i][c];
            out<<"{\"own_primary_overlap\":"<<s.own_primary_overlap[c]<<",\"own_short_overlap\":"<<s.own_short_overlap[c]<<",\"paired_primary_overlap\":"<<s.paired_primary_overlap[c]<<",\"paired_short_overlap\":"<<s.paired_short_overlap[c]
               <<",\"original_primary_rows\":";reassessment_rows(out,s.primary[c]);out<<",\"original_short_rows\":";reassessment_rows(out,s.shorter[c]);
            out<<",\"new_primary_rows\":";reassessment_rows(out,r.primary_rows);out<<",\"new_short_rows\":";reassessment_rows(out,r.short_rows);
            out<<",\"sigma_delta\":";number(out,amplitude[s.event][c].sigma_delta);out<<",\"frozen_primary_scale\":";number(out,event.background[c].pre_scale_fit.scale);out<<",\"frozen_short_scale\":";number(out,old_short[s.event][c].pre_scale_fit.scale);
            out<<",\"primary_available\":"<<(r.primary.available()?"true":"false")<<",\"primary_support_cause\":"<<int(r.primary.support_cause)<<",\"primary_cubic\":";fit_json(out,r.primary.cubic);
            out<<",\"primary_offset\":";fit_json(out,r.primary.cubic_with_offset);out<<",\"short_available\":"<<(r.shorter.available()?"true":"false")<<",\"short_cause\":"<<int(r.shorter.cause)<<",\"short_offset\":";fit_json(out,r.shorter.cubic_with_offset);
            out<<",\"recovery_cause\":"<<int(r.recovery.cause)<<",\"consistency_cause\":"<<int(rq.selections()[i][c].consistency)<<",\"remeasure_requested\":"<<(rq.selections()[i][c].requested?"true":"false")<<",\"transition\":";reassessment_bound(out,m.transition);
            out<<",\"new_primary_overlap\":"<<m.primary_overlap<<",\"new_short_overlap\":"<<m.short_overlap<<",\"diagnostic_cause\":"<<int(decision.coordinates()[i][c])<<'}';
        }
        out<<"],\"hard_event_accepted\":false,\"apply_authorized\":false,\"scan_assignment\":\"unavailable: native-to-existing-scan timing relation absent\"}\n";
    }
    out.close();require(bool(out),"reassessment output write/close failed");
}
void write_reassessment_examples(const fs::path &output,const fs::path &selection,int obs,int nw,
        const pipeline::RtcEventAssessmentEvidence &a) {
    const auto &spikes=*a.spike_handle();const auto &net=spikes.input_handle()->network(nw);const auto &axis=net.occurrence_axis();
    std::ifstream input(selection);require(bool(input),"cannot open exact example selection");
    std::ofstream out(output/"reassessment-examples.jsonl");require(bool(out),"cannot open example output");
    int o,n,d;std::int64_t earlier;std::size_t event_id;std::string label;
    while(input>>o>>n>>d>>earlier>>label>>event_id) if(o==obs && n==nw) {
        require(event_id<a.events().size(),"selected event absent");const auto &event=a.events()[event_id];
        const bool member=std::any_of(event.candidates.begin(),event.candidates.end(),[&](auto i){return spikes.candidates()[i].earlier_row==earlier;});
        require(event.detector==static_cast<std::uint32_t>(d) && member,"selected event identity changed");
        out<<"{\"case\":"<<std::quoted(label)<<",\"event\":"<<event_id<<",\"detector\":"<<d<<",\"original_primary_rows\":[";
        for(std::size_t c=0;c<2;++c) {
            if(c)out<<',';pipeline::RtcJumpFitRows fit_rows;
            if(event.background[c].available()) fit_rows=pipeline::rtc_jump_reassessment_detail::admitted(spikes,event,c,event.background[c].support,2);
            reassessment_rows(out,fit_rows);
        }
        out<<"],\"samples\":[";bool comma=false;
        for(auto row=std::max(axis.first_native_row(),earlier-520);row<std::min(axis.past_last_native_row(),earlier+522);++row) {
            if(comma)out<<',';comma=true;const auto &cell=axis.occurrence(row).integration_support;
            out<<'['<<row<<',';number(out,axis.native_identity(row).reconstructed_time_unix_sec());out<<',';number(out,cell.begin_unix_sec);out<<',';number(out,cell.end_unix_sec);
            for(auto c:{pipeline::NativeReadoutCoordinate::x,pipeline::NativeReadoutCoordinate::r}){out<<',';number(out,net.value(c,row,d));out<<','<<(net.state(c,row,d).valid()?"true":"false");}out<<']';
        }
        out<<"],\"scan_boundaries\":null,\"scan_boundary_cause\":\"native-to-existing-scan timing relation absent\"}\n";
    }
    require(input.eof(),"invalid example selection record");out.close();require(bool(out),"example output close failed");
    // Fixed matched injection backgrounds: first 20 seconds, selected before
    // inspecting reassessment. Original finite/valid state is exported verbatim.
    const int detector=obs==152385 && nw==4 ? 61 : obs==152430 && nw==8 ? 253 : -1;
    if(detector<0) return;
    std::ofstream background(output/"injection-background.txt");require(bool(background),"cannot open injection background");
    const auto first=axis.first_native_row();const double end=axis.native_identity(first).reconstructed_time_unix_sec()+20;
    for(auto row=first;row<axis.past_last_native_row() && axis.native_identity(row).reconstructed_time_unix_sec()<end;++row) {
        const auto &cell=axis.occurrence(row).integration_support;
        background<<std::setprecision(17)<<row<<' '<<axis.native_identity(row).reconstructed_time_unix_sec()<<' '<<cell.begin_unix_sec<<' '<<cell.end_unix_sec;
        for(auto c:{pipeline::NativeReadoutCoordinate::x,pipeline::NativeReadoutCoordinate::r}) background<<' '<<net.value(c,row,detector)<<' '<<net.state(c,row,detector).valid();
        background<<'\n';
    }
    background.close();require(bool(background),"injection background close failed");
}
} // namespace

// Bounded real-input test driver. Reuses the exact earlier acceptance runner's
// verified raw/Tune/APT adapters; never invokes that runner or an Apply route.
// Keeping its private helpers in this translation unit avoids a second solver
// implementation. This executable is excluded from application builds.
#define main unused_identity_acceptance_main
#include "identity_route_acceptance.cpp"
#undef main
#include <citlali/core/pipeline/timestream_rtc_event_assessment.h>
#include <citlali/core/pipeline/timestream_rtc_jump_consistency.h>

namespace {
void number(std::ostream &out, double x) {
    if (std::isfinite(x)) out << std::setprecision(17) << x;
    else out << "null";
}
void fit_json(std::ostream &out, const pipeline::RtcEventCubicFit &f) {
    out << "{\"cause\":" << static_cast<int>(f.cause) << ",\"iterations\":" << f.iterations
        << ",\"scale\":"; number(out,f.scale);
    out << ",\"offset\":"; number(out,f.offset);
    out << ",\"huber_loss\":"; number(out,f.huber_loss);
    out << ",\"coefficients\":[";
    for (int i=0;i<4;++i) { if(i) out << ','; number(out,f.coefficients[i]); }
    out << "]}";
}
}

// argv: raw file, Tune fit report, compact-v2 manifest, output directory,
// trial exclusion half-width seconds. No automatic event extent is selected.
int main(int argc, char **argv) {
    try {
        const auto execution_started=std::chrono::steady_clock::now();
        require(argc==6,"expected raw, Tune, APT manifest, new output directory, trial half-width seconds");
        const fs::path raw_path=argv[1], tune_path=argv[2], manifest=argv[3], output=argv[4];
        const double guard=std::stod(argv[5]);
        require(guard==0.05,"assessment policy requires explicit 50 ms starting half-width");
        require(!fs::exists(output),"output exists; preserve earlier attempts");
        auto [logger, logs]=configure_logging();
        const auto verified=apt::verify_bundle_filesystem(manifest,true);
        const auto relation=pipeline::admit_canonical_apt_detector_relation_v2(verified);
        netCDF::NcFile raw_file(raw_path.string(),netCDF::NcFile::read);
        const auto nw=read_netcdf_scalar<int>(raw_file,"Header.Toltec.RoachIndex");
        const auto obs=read_netcdf_scalar<int>(raw_file,"Header.Toltec.ObsNum");
        const auto sub=read_netcdf_scalar<int>(raw_file,"Header.Toltec.SubObsNum");
        const auto scan=read_netcdf_scalar<int>(raw_file,"Header.Toltec.ScanNum");
        require(obs==relation.observation().observation && sub==relation.observation().subobservation &&
                scan==relation.observation().scan,"raw observation differs from exact APT relation");
        const auto found=std::find_if(relation.raw_sources().begin(),relation.raw_sources().end(),
                                     [&](const auto &s){return s.network==nw;});
        require(found!=relation.raw_sources().end(),"APT does not bind input network");
        const auto source=*found;
        verify_raw_file(raw_path,source);
        const auto rows=static_cast<std::int64_t>(raw_file.getVar("Data.Toltec.Is").getDim(0).getSize());
        const auto tune_digest="sha256:"+citlali::utils::sha256_file(tune_path);
        const auto kmp=std::find_if(verified.sources.begin(),verified.sources.end(),[&](const auto &s){return s.role==apt::SourceRole::kmp && s.network==nw;});
        require(kmp!=verified.sources.end() && kmp->content_sha256==tune_digest &&
                kmp->byte_count==fs::file_size(tune_path),"Tune report differs from exact APT KMP source binding");
        const auto tune=read_tune_facts(tune_path,source.channel_count);
        require(tune.observation==read_netcdf_scalar<int>(raw_file,"Header.Toltec.TargSweepObsNum") &&
                tune.subobservation==read_netcdf_scalar<int>(raw_file,"Header.Toltec.TargSweepSubObsNum") &&
                tune.scan==read_netcdf_scalar<int>(raw_file,"Header.Toltec.TargSweepScanNum") &&
                tune.network==nw,"Tune report differs from raw header's calibration relation");
        double fpga=0,hz=0; std::int64_t accum=0;
        auto ts=read_timestamp_slice(raw_path,0,rows,fpga,hz,accum);
        require(fpga>0 && hz>0 && accum>0 && tune.accumulation_length>0,"invalid producer cadence");
        const double duration=static_cast<double>(accum)/fpga;
        require(std::abs(duration-1/hz)<=8*std::numeric_limits<double>::epsilon()*duration,
                "producer cadence fields disagree");
        // Native producer clock only; no telescope synchronization or cross-network timing claim.
        auto timing=std::make_shared<const pipeline::NativeNetworkAlignment>(
            pipeline::make_native_network_alignment(nw,0,ts,fpga,0.0));
        NetworkInput input{source,raw_path,tune_path,citlali::utils::sha256_file(tune_path),
                           fpga,hz,accum,tune.accumulation_length,tune.valid,timing};
        TemporaryDirectory tunes;
        const auto normalized=normalized_tune_report(input,tunes.path());
        const auto started=std::chrono::steady_clock::now();
        auto raw=citlali::compat::kidscpp::read_raw_timestream_slice(raw_path,
            tula::container_utils::IndexSlice{0,static_cast<Eigen::Index>(rows),std::nullopt});
        kids::TimeStreamSolver solver(kids::TimeStreamSolver::Config{
            {"fitreportfile",normalized.string()},{"exmode",std::string{"seq"}},{"extra_output",false}});
        auto solved=solver(raw);
        require(solved.data_out.xs.data.rows()==rows && solved.data_out.xs.data.cols()==source.channel_count &&
                solved.data_out.rs.data.rows()==rows && solved.data_out.rs.data.cols()==source.channel_count,
                "original paired solver shape mismatch");
        RuntimeConfig config; config.kids_model="gainlintrend";
        auto mapping=std::make_shared<pipeline::NativeReadoutMappingAuthority>(*mapping_identity(input,config));
        mapping->applicability_domain_id="observation="+std::to_string(obs)+":network="+std::to_string(nw);
        mapping->event_time_epoch_meaning_id="producer-native-clock:no-telescope-sync:census-only";
        mapping->timing_uncertainty_state_id="unquantified:uniform-average-center-trial:rtc-native-readout-uniform-average-assumption-v1";
        // The owner-approved uniform-average assumption remains provisional.
        // Midpoint convention is a translation for these within-network fits;
        // no source membership, optical timing or calibrated epoch is claimed.
        auto axis=occurrence_axis(input,0,rows,NativeEventTimeRole::integration_center);
        const auto runs=axis->contiguous_runs();
        auto detectors=detector_axis(relation,input);
        pipeline::NativePairedReadoutNetworkIngress ingress{axis,detectors,mapping,
            member_states(solved.data_out.xs.data,tune.valid),member_states(solved.data_out.rs.data,tune.valid)};
        std::vector<pipeline::NativePairedReadoutNetwork> networks;
        networks.push_back(pipeline::take_native_paired_kids_solver_result(std::move(ingress),std::move(solved)));
        auto parent=std::make_shared<const pipeline::NativePairedReadoutObservation>(
            pipeline::NativePairedReadoutObservation::admit(pipeline::NativeObservationScope{obs,sub,scan},
                {nw},std::move(networks)));
        auto val=pipeline::ValSnapshot::initial(parent);
        auto view=pipeline::NativePairedReadoutView::full(parent);
        auto protection=pipeline::RtcSpikeSourceProtection::admit(parent,
            "rtc-census-source-membership-unavailable",pipeline::RtcSpikeProtection::unavailable);
        const auto ingress_finished=std::chrono::steady_clock::now();
        auto spikes=pipeline::learn_rtc_spike_candidates(view,val,protection,1);
        const auto spikes_finished=std::chrono::steady_clock::now();
        std::vector<pipeline::RtcEventPeerEligibility> eligible;
        std::vector<bool> clear(source.channel_count,false);
        for(const auto &row:verified.apt.rows) if(row.network==nw) {
            auto zero=[&](std::string name) {
                const auto it=row.fields.find(name);
                if(it==row.fields.end()) return false;
                const auto *v=std::get_if<std::int64_t>(&it->second);
                return v && *v==0;
            };
            require(row.channel>=0 && row.channel<source.channel_count,"APT quality row channel mismatch");
            clear[row.channel]=tune.valid[row.channel] && zero("flag") && zero("flag2");
        }
        for(std::uint32_t d=0;d<source.channel_count;++d)
            eligible.push_back({nw,d,detectors[d].detector_occurrence_id,clear[d]});
        const auto population=pipeline::RtcEventPeerPopulation::admit(spikes,
            "verified-apt-sha256:"+citlali::utils::sha256_file(manifest)+":Tune-valid:flag=0:flag2=0",std::move(eligible));
        const auto population_finished=std::chrono::steady_clock::now();
        const auto assessment=pipeline::learn_rtc_event_assessment(spikes,population,2);
        const auto assessment_finished=std::chrono::steady_clock::now();
        const auto decision=pipeline::RtcEventAssessmentDecision::consider(assessment,val,3);
        const auto review_finished=std::chrono::steady_clock::now();
        const auto jump_amplitude=pipeline::RtcJumpAmplitudeDecision::consider(decision,val,4);
        const auto amplitude_finished=std::chrono::steady_clock::now();
        const auto jump_evidence=pipeline::RtcJumpConsistencyEvidence::learn(jump_amplitude,5);
        const auto short_fit_finished=std::chrono::steady_clock::now();
        const auto jump_decision=pipeline::RtcJumpConsistencyDecision::consider(jump_evidence,val,6);
        const auto jump_consider_finished=std::chrono::steady_clock::now();
        fs::create_directories(output);
        std::ofstream candidates(output/"candidates.jsonl"),events(output/"events.jsonl"),blocks(output/"health-blocks.jsonl"),summaries(output/"detectors.jsonl"),displays(output/"examples.jsonl");
        require(candidates && events && blocks && summaries && displays,"cannot open assessment outputs");
        std::vector<std::array<std::size_t,2>> counts(source.channel_count);
        for(std::size_t i=0;i<spikes->candidates().size();++i) {
            const auto &s=spikes->candidates()[i];const auto &b=spikes->blocks()[s.noise_block_index];
            const auto c=s.coordinate==pipeline::NativeReadoutCoordinate::x?0:1;++counts[b.detector_index][c];
            candidates<<"{\"candidate\":"<<i<<",\"detector\":"<<b.detector_index<<",\"seed_coordinate\":"<<c<<",\"earlier_row\":"<<s.earlier_row<<",\"later_row\":"<<s.later_row<<",\"score\":";number(candidates,s.absolute_score);
            candidates<<",\"peer_context\":[";
            for(std::size_t k=0;k<2;++k) {if(k)candidates<<',';const auto &p=assessment->candidate_peer_context()[i][k];candidates<<"{\"eligible_peers\":"<<p.eligible_peers<<",\"shared_samples\":"<<p.strongest_shared_samples<<",\"usable_peers\":"<<p.usable_peers<<",\"strongest_peer\":"<<p.strongest_peer<<",\"level_correlation\":";number(candidates,p.strongest_level_correlation);candidates<<",\"difference_correlation\":";number(candidates,p.strongest_difference_correlation);candidates<<",\"edge_delay_seconds\":";number(candidates,p.strongest_edge_delay_seconds);candidates<<'}';}candidates<<"]}\n";
        }
        for(const auto &h:assessment->health_blocks()) {
            const auto &b=spikes->blocks()[h.noise_block_index];
            blocks<<"{\"noise_block\":"<<h.noise_block_index<<",\"detector\":"<<b.detector_index<<",\"first\":"<<b.first<<",\"end\":"<<b.past_last<<",\"complete\":"<<(h.complete?"true":"false")<<",\"coordinates\":[";
            for(std::size_t c=0;c<2;++c) {if(c)blocks<<',';blocks<<"{\"scale\":";number(blocks,b.coordinates[c].scale);blocks<<",\"scale_cause\":"<<static_cast<int>(b.coordinates[c].cause)<<",\"admitted_differences\":"<<b.coordinates[c].admitted_differences<<",\"edges\":"<<h.edges[c]<<",\"peers\":"<<h.peer_count[c]<<",\"peer_median_scale\":";number(blocks,h.peer_median_scale[c]);blocks<<",\"scale_ratio\":";number(blocks,h.scale_ratio[c]);blocks<<",\"edge_fraction\":";number(blocks,h.edge_fraction[c]);blocks<<'}';}blocks<<"]}\n";
        }
        auto range=[](std::ostream &o,const pipeline::RtcEventRange &r){o<<'['<<r.first<<','<<r.past_last<<']';};
        for(std::size_t index=0;index<assessment->events().size();++index) {
            const auto &a=assessment->events()[index];const auto &review=decision->event_reviews()[index];
            events<<"{\"event\":"<<index<<",\"detector\":"<<a.detector<<",\"seed\":"<<a.seed<<",\"candidates\":[";
            for(std::size_t i=0;i<a.candidates.size();++i){if(i)events<<',';events<<a.candidates[i];}
            events<<"],\"trial_exclusion\":";range(events,a.trial_exclusion);events<<",\"origin\":";number(events,a.origin);events<<",\"time_scale\":";number(events,a.time_scale);
            events<<",\"neighbor_exclusions\":[";for(std::size_t i=0;i<a.neighbor_exclusions.size();++i){if(i)events<<',';range(events,a.neighbor_exclusions[i]);}events<<"],\"coordinates\":[";
            for(std::size_t c=0;c<2;++c) {
                if(c)events<<',';const auto &b=a.background[c];const auto &r=a.recovery[c];const auto &p=a.peers[c];
                events<<"{\"seeded\":"<<(a.seeded[c]?"true":"false")<<",\"available\":"<<(b.available()?"true":"false")<<",\"support_cause\":"<<static_cast<int>(b.support_cause)<<",\"pre_count\":"<<b.support[0].usable<<",\"post_count\":"<<b.support[1].usable<<",\"neighbor_excluded\":"<<a.excluded_neighbor_samples[c]<<",\"pre_invalid\":"<<b.support[0].invalid<<",\"post_invalid\":"<<b.support[1].invalid<<",\"pre\":";fit_json(events,b.pre_scale_fit);events<<",\"cubic\":";fit_json(events,b.cubic);events<<",\"with_offset\":";fit_json(events,b.cubic_with_offset);
                events<<",\"recovery_cause\":"<<static_cast<int>(r.cause)<<",\"affected\":";range(events,r.affected);events<<",\"confirmation\":";range(events,r.confirmation);events<<",\"examined\":";range(events,r.examined);
                events<<",\"peer_context\":{\"eligible_peers\":"<<p.eligible_peers<<",\"usable_peers\":"<<p.usable_peers<<",\"strongest_peer\":"<<p.strongest_peer<<",\"shared_samples\":"<<p.strongest_shared_samples<<",\"level_correlation\":";number(events,p.strongest_level_correlation);events<<",\"difference_correlation\":";number(events,p.strongest_difference_correlation);events<<",\"edge_delay_seconds\":";number(events,p.strongest_edge_delay_seconds);events<<"}}";
            }
            events<<"],\"refinement_limited\":"<<(a.refinement_limited?"true":"false")<<",\"review_disposition\":"<<static_cast<int>(review.disposition)<<",\"health_concern\":"<<(review.health_concern?"true":"false")<<",\"source_protection_unavailable\":"<<(review.source_protection_unavailable?"true":"false")<<",\"protected_optical_required\":"<<(review.protected_optical_assessment_required?"true":"false")<<",\"spectral_context_unavailable\":true,\"hard_event_accepted\":false,\"apply_authorized\":false,\"observation_truncated\":"<<(a.observation_truncated?"true":"false")<<",\"gap_truncated\":"<<(a.gap_truncated?"true":"false")<<",\"peak_scratch_rows\":"<<a.peak_scratch_rows<<"}\n";
        }
        for(const auto &h:decision->health_reviews()) {
            std::size_t excluded=0;
            for(std::size_t i=0;i<spikes->blocks().size();++i) {const auto &b=spikes->blocks()[i];if(b.detector_index==h.detector && decision->original_screening_handle()->requires_pair_exclusion_from_mapmaking(i)) excluded+=b.past_last-b.first;}
            summaries<<"{\"detector\":"<<h.detector<<",\"occurrence\":"<<std::quoted(detectors[h.detector].detector_occurrence_id)<<",\"tune_valid\":"<<(tune.valid[h.detector]?"true":"false")<<",\"peer_eligible\":"<<(clear[h.detector]?"true":"false")<<",\"rows\":"<<rows<<",\"candidate_counts\":["<<counts[h.detector][0]<<','<<counts[h.detector][1]<<"],\"pair_screening_excluded_rows\":"<<excluded<<",\"complete_blocks\":"<<h.complete_blocks<<",\"available_blocks\":["<<h.available_blocks[0]<<','<<h.available_blocks[1]<<"],\"concerning_blocks\":["<<h.concerning_blocks[0]<<','<<h.concerning_blocks[1]<<"],\"coordinate_concern\":["<<(h.coordinate_concern[0]?"true":"false")<<','<<(h.coordinate_concern[1]?"true":"false")<<"],\"health_assessment_available\":["<<(h.assessment_available[0]?"true":"false")<<','<<(h.assessment_available[1]?"true":"false")<<"],\"health_review_concern\":"<<(h.concern()?"true":"false")<<"}\n";
        }
        // Fixed display selections only. These rows never enter scientific policy.
        struct Example {int obs,nw,channel;std::int64_t earlier;const char *label;};
        const std::array selected{
            Example{152385,4,61,2835,"A"},Example{152430,8,253,1971,"B"},Example{152390,12,114,84653,"C"},
            Example{152418,2,342,6253,"D"},Example{152418,5,60,936,"E"},Example{152390,8,6,61725,"F"}};
        for(const auto &item:selected) if(item.obs==obs && item.nw==nw) {
            const double center=std::midpoint(axis->native_identity(item.earlier).reconstructed_time_unix_sec(),axis->native_identity(item.earlier+1).reconstructed_time_unix_sec());
            displays<<"{\"case\":"<<std::quoted(std::string(item.label))<<",\"detector\":"<<item.channel<<",\"reference_earlier_row\":"<<item.earlier<<",\"origin\":";number(displays,center);displays<<",\"samples\":[";
            bool comma=false;
            for(auto row=std::max<std::int64_t>(0,item.earlier-520);row<std::min<std::int64_t>(rows,item.earlier+522);++row) {
                if(comma)displays<<',';comma=true;displays<<'['<<row<<',';number(displays,axis->native_identity(row).reconstructed_time_unix_sec()-center);
                for(auto c:{pipeline::NativeReadoutCoordinate::x,pipeline::NativeReadoutCoordinate::r}) {displays<<',';number(displays,parent->network(nw).value(c,row,item.channel));displays<<','<<(parent->network(nw).state(c,row,item.channel).valid()?"true":"false");}displays<<']';
            }displays<<"]}\n";
        }
        // Explicit close exposes trailing buffered writes and close failures.
        for(auto *stream:{&candidates,&events,&blocks,&summaries,&displays}) {
            stream->close();
            require(static_cast<bool>(*stream),"assessment output write/close failed");
        }
        const auto original_output_finished=std::chrono::steady_clock::now();
        std::ofstream jumps(output/"jump-consistency.jsonl");
        require(static_cast<bool>(jumps),"cannot open jump consistency output");
        std::array<std::size_t,6> amplitude_counts{};
        std::array<std::size_t,7> consistency_counts{};
        std::size_t consistent_without_recovery=0;
        for(std::size_t i=0;i<assessment->events().size();++i) {
            jumps<<"{\"event\":"<<i<<",\"detector\":"<<assessment->events()[i].detector<<",\"coordinates\":[";
            for(std::size_t c=0;c<2;++c) {
                const auto &a=jump_amplitude->coordinates()[i][c];
                const auto &s=jump_evidence->coordinates()[i][c];
                const auto &d=jump_decision->coordinates()[i][c];
                ++amplitude_counts[static_cast<std::size_t>(a.cause)];
                ++consistency_counts[static_cast<std::size_t>(d.cause)];
                consistent_without_recovery+=d.passes() && !d.confirmed_recovery_excludes_persistent_shift;
                if(c) jumps<<',';
                jumps<<"{\"candidate\":"; if(a.candidate) jumps<<*a.candidate;else jumps<<"null";
                jumps<<",\"noise_block\":"; if(a.noise_block) jumps<<*a.noise_block;else jumps<<"null";
                jumps<<",\"sigma_delta\":";number(jumps,a.sigma_delta);
                jumps<<",\"primary_offset_sigma\":";number(jumps,a.offset_sigma);
                jumps<<",\"amplitude_cause\":"<<static_cast<int>(a.cause)
                    <<",\"short_fit_cause\":"<<static_cast<int>(s.cause)<<",\"short_fit\":";
                if(s.cause==pipeline::RtcJumpShortFitCause::not_requested) jumps<<"null";
                else {
                    jumps<<"{\"available\":"<<(s.available()?"true":"false")<<",\"support\":[";
                    for(std::size_t side=0;side<2;++side) {
                        if(side)jumps<<',';const auto &p=s.support[side];
                        jumps<<"{\"usable\":"<<p.usable<<",\"invalid\":"<<p.invalid<<",\"neighbor_excluded\":"<<s.neighbor_excluded[side]
                            <<",\"first_used\":"<<p.first_used<<",\"last_used\":"<<p.last_used<<",\"begin\":";number(jumps,p.begin_unix_sec);
                        jumps<<",\"end\":";number(jumps,p.end_unix_sec);jumps<<'}';
                    }
                    jumps<<"],\"pre_scale_fit\":";fit_json(jumps,s.pre_scale_fit);
                    jumps<<",\"with_offset\":";fit_json(jumps,s.cubic_with_offset);
                    jumps<<",\"scratch_rows\":"<<s.scratch_rows<<'}';
                }
                jumps<<",\"consistency_cause\":"<<static_cast<int>(d.cause)<<",\"short_offset_sigma\":";number(jumps,d.short_offset_sigma);
                jumps<<",\"offset_difference_sigma\":";number(jumps,d.offset_difference_sigma);
                jumps<<",\"confirmed_recovery_excludes_persistent_shift\":"<<(d.confirmed_recovery_excludes_persistent_shift?"true":"false")<<'}';
            }
            jumps<<"],\"hard_event_accepted\":false,\"apply_authorized\":false}\n";
        }
        jumps.close();require(static_cast<bool>(jumps),"jump consistency output write/close failed");
        const auto jump_output_finished=std::chrono::steady_clock::now();
        const auto seconds=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
        const auto &jc=jump_evidence->counts();
        std::ofstream timings(output/"timing.json");
        timings<<"{\"policy\":"<<std::quoted(std::string(pipeline::RtcJumpConsistencyPolicy::identity))<<",\"stages_seconds\":{";
        const std::array stages{
            std::pair{"input_verification",seconds(execution_started,started)},
            std::pair{"producer_and_native_ingress",seconds(started,ingress_finished)},
            std::pair{"spike_learn",seconds(ingress_finished,spikes_finished)},
            std::pair{"peer_population",seconds(spikes_finished,population_finished)},
            std::pair{"original_event_assessment",seconds(population_finished,assessment_finished)},
            std::pair{"original_consider",seconds(assessment_finished,review_finished)},
            std::pair{"jump_amplitude_consider",seconds(review_finished,amplitude_finished)},
            std::pair{"jump_short_fit_learn",seconds(amplitude_finished,short_fit_finished)},
            std::pair{"jump_consistency_consider",seconds(short_fit_finished,jump_consider_finished)},
            std::pair{"original_output",seconds(jump_consider_finished,original_output_finished)},
            std::pair{"jump_output",seconds(original_output_finished,jump_output_finished)}};
        for(std::size_t i=0;i<stages.size();++i){if(i)timings<<',';timings<<std::quoted(stages[i].first)<<':';number(timings,stages[i].second);}
        timings<<"},\"measured_total_seconds\":";number(timings,seconds(execution_started,jump_output_finished));
        timings<<",\"requested_coordinates\":"<<jc.requested_coordinates<<",\"pre_fit_calls\":"<<jc.pre_fit_calls
            <<",\"joint_fit_calls\":"<<jc.joint_fit_calls<<",\"available_coordinates\":"<<jc.available_coordinates
            <<",\"pre_reported_iterations\":"<<jc.pre_iterations<<",\"joint_reported_iterations\":"<<jc.joint_iterations
            <<",\"peak_short_scratch_rows\":"<<jc.peak_scratch_rows<<",\"amplitude_cause_counts\":[";
        for(std::size_t i=0;i<amplitude_counts.size();++i){if(i)timings<<',';timings<<amplitude_counts[i];}
        timings<<"],\"consistency_cause_counts\":[";
        for(std::size_t i=0;i<consistency_counts.size();++i){if(i)timings<<',';timings<<consistency_counts[i];}
        timings<<"],\"consistent_without_confirmed_recovery\":"<<consistent_without_recovery
            <<",\"timing_scope\":\"local inert driver through diagnostic output close; excludes final receipt/hash and process shutdown; not production RTC/PTC\",\"iteration_count_definition\":\"IRLS loop entries on successful and failed fits; zero before loop\"}\n";
        timings.close();require(static_cast<bool>(timings),"jump timing output write/close failed");
        require(logs->errors==0 && logs->criticals==0,"unexpected producer error-level messages");
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::ofstream receipt(output/"receipt.json");
        receipt<<"{\"status\":\"PASS-event-assessment-census\",\"policy\":"<<std::quoted(std::string(pipeline::RtcEventAssessmentPolicy::identity))<<",\"observation\":"<<obs<<",\"network\":"<<nw<<",\"rows\":"<<rows<<",\"channels\":"<<source.channel_count<<",\"physical_runs\":"<<runs.size()<<",\"candidate_edges\":"<<spikes->candidates().size()<<",\"assessed_events\":"<<assessment->events().size()<<",\"raw_sha256\":"<<std::quoted(citlali::utils::sha256_file(raw_path))<<",\"tune_sha256\":"<<std::quoted(input.tune_sha256)<<",\"manifest_sha256\":"<<std::quoted(citlali::utils::sha256_file(manifest))<<",\"peer_population\":"<<std::quoted(population->source_identity())<<",\"elapsed_seconds\":";number(receipt,elapsed);receipt<<",\"source_protection\":\"unavailable\",\"spectral_context\":\"owner-deferred\",\"hard_classification\":\"not_performed\",\"Apply\":\"not_performed\"}\n";
        receipt.close();
        require(static_cast<bool>(receipt),"assessment receipt write/close failed");
        std::cout<<"ASSESSMENT obs="<<obs<<" network="<<nw<<" candidates="<<spikes->candidates().size()<<" events="<<assessment->events().size()<<" seconds="<<elapsed<<'\n';return 0;
    } catch(const std::exception &e) {std::cerr<<"ASSESSMENT INPUT OR EXECUTION FAILURE: "<<e.what()<<'\n';return 2;}
}

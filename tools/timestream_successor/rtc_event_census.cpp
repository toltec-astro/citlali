// Bounded real-input test driver. Reuses the exact earlier acceptance runner's
// verified raw/Tune/APT adapters; never invokes that runner or an Apply route.
// Keeping its private helpers in this translation unit avoids a second solver
// implementation. This executable is excluded from application builds.
#define main unused_identity_acceptance_main
#include "identity_route_acceptance.cpp"
#undef main
#include <citlali/core/pipeline/timestream_rtc_event_background.h>

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
        require(argc==6,"expected raw, Tune, APT manifest, new output directory, trial half-width seconds");
        const fs::path raw_path=argv[1], tune_path=argv[2], manifest=argv[3], output=argv[4];
        const double guard=std::stod(argv[5]);
        require(std::isfinite(guard) && guard>0 && guard<=2,"invalid explicit trial half-width");
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
        auto spikes=pipeline::learn_rtc_spike_candidates(view,val,protection,1);
        auto screening=pipeline::RtcSpikeLearningDecision::consider(spikes,val,1);
        fs::create_directories(output);
        std::ofstream events(output/"fits.jsonl"), blocks(output/"noise-blocks.jsonl"), summaries(output/"detectors.jsonl");
        require(events && blocks && summaries,"cannot open census outputs");
        struct Counts { std::array<std::size_t,2> candidates{}, fit_available{}, noise_available{}, noise_unavailable{}; std::size_t paired_excluded_rows=0; };
        std::vector<Counts> counts(source.channel_count);
        for(std::size_t i=0;i<spikes->blocks().size();++i) {
            const auto &b=spikes->blocks()[i]; auto &s=counts[b.detector_index];
            if(screening->requires_pair_exclusion_from_mapmaking(i)) s.paired_excluded_rows+=b.past_last-b.first;
            blocks << "{\"detector\":" << b.detector_index << ",\"first\":" << b.first << ",\"end\":" << b.past_last << ",\"coordinates\":[";
            for(int c=0;c<2;++c) {
                const auto &n=b.coordinates[c];
                if(n.available()) ++s.noise_available[c]; else ++s.noise_unavailable[c];
                if(c) blocks << ',';
                blocks << "{\"cause\":" << static_cast<int>(n.cause) << ",\"admitted_differences\":" << n.admitted_differences << ",\"excluded_differences\":" << n.excluded_differences << ",\"scale\":";
                number(blocks,n.scale); blocks << '}';
            }
            blocks << "]}\n";
        }
        // Six bounded display examples; these choices have no scientific effect.
        std::array<std::shared_ptr<const pipeline::RtcEventBackgroundEvidence>,6> examples{};
        double max_score=-1, max_offset=-1, min_offset=std::numeric_limits<double>::infinity();
        for(std::size_t i=0;i<spikes->candidates().size();++i) {
            const auto &candidate=spikes->candidates()[i]; const auto &b=spikes->blocks()[candidate.noise_block_index];
            auto &s=counts[b.detector_index]; const int seed_coordinate=candidate.coordinate==pipeline::NativeReadoutCoordinate::x?0:1;
            ++s.candidates[seed_coordinate];
            const auto run=std::find_if(runs.begin(),runs.end(),[&](const auto &r){return r.first_native_row<=candidate.earlier_row && r.past_last_native_row>candidate.later_row;});
            require(run!=runs.end(),"candidate crosses physical run");
            const double center=std::midpoint(axis->native_identity(candidate.earlier_row).reconstructed_time_unix_sec(),axis->native_identity(candidate.later_row).reconstructed_time_unix_sec());
            auto first=candidate.earlier_row, end=candidate.later_row+1;
            while(first>run->first_native_row && axis->occurrence(first-1).integration_support.end_unix_sec>center-guard) --first;
            while(end<run->past_last_native_row && axis->occurrence(end).integration_support.begin_unix_sec<center+guard) ++end;
            const auto evidence=pipeline::learn_rtc_event_background(spikes,{i,first,end},i+1);
            const auto decision=pipeline::RtcEventBackgroundDecision::consider(evidence,val,i+1);
            if(candidate.absolute_score>max_score) { max_score=candidate.absolute_score; examples[0]=evidence; }
            if(evidence->coordinates()[0].available()) {
                const auto &x=evidence->coordinates()[0];
                const double ratio=std::abs(x.cubic_with_offset.offset)/x.pre_scale_fit.scale;
                if(ratio>max_offset) { max_offset=ratio; examples[1]=evidence; }
                if(ratio<min_offset) { min_offset=ratio; examples[2]=evidence; }
            }
            if(!examples[3] && (!evidence->coordinates()[0].available() || !evidence->coordinates()[1].available())) examples[3]=evidence;
            if(i==spikes->candidates().size()/2) examples[4]=evidence;
            if(!examples[5] && seed_coordinate==1) examples[5]=evidence;
            events << "{\"candidate\":" << i << ",\"detector\":" << b.detector_index << ",\"seed_coordinate\":" << seed_coordinate << ",\"earlier_row\":" << candidate.earlier_row << ",\"later_row\":" << candidate.later_row << ",\"score\":";
            number(events,candidate.absolute_score);
            events << ",\"excluded_first\":" << first << ",\"excluded_end\":" << end << ",\"origin\":"; number(events,evidence->origin_unix_sec());
            events << ",\"time_scale\":"; number(events,evidence->time_scale_seconds());
            events << ",\"coordinates\":[";
            for(int c=0;c<2;++c) {
                const auto &e=evidence->coordinates()[c]; const auto requirement=decision->requirements(c==0?pipeline::NativeReadoutCoordinate::x:pipeline::NativeReadoutCoordinate::r);
                require(requirement.source_protection_unavailable && requirement.offset_uncertainty_and_acceptance_required && requirement.extent_containment_and_recovery_required,"census lost required unresolved predicates");
                if(e.available()) ++s.fit_available[c];
                if(c) events << ',';
                events << "{\"available\":" << (e.available()?"true":"false") << ",\"support_cause\":" << static_cast<int>(e.support_cause) << ",\"pre_count\":" << e.support[0].usable << ",\"post_count\":" << e.support[1].usable << ",\"pre_invalid\":" << e.support[0].invalid << ",\"post_invalid\":" << e.support[1].invalid << ",\"observation_truncated\":" << (requirement.incomplete_observation_context?"true":"false") << ",\"gap_truncated\":" << (requirement.physical_gap_context?"true":"false") << ",\"pre\":";
                fit_json(events,e.pre_scale_fit); events << ",\"cubic\":"; fit_json(events,e.cubic); events << ",\"with_offset\":"; fit_json(events,e.cubic_with_offset); events << '}';
            }
            events << "]}\n";
        }
        std::ofstream displays(output/"examples.jsonl");
        const std::array<std::string_view,6> reasons{"largest_candidate_score","largest_absolute_x_offset_over_pre_scale","smallest_absolute_x_offset_over_pre_scale","first_unavailable_fit","middle_candidate_index","first_r_seed"};
        for(std::size_t slot=0;slot<examples.size();++slot) {
            if(!examples[slot]) continue;
            const auto &e=*examples[slot]; const auto &seed=spikes->candidates()[e.request().candidate_index];
            const auto &block=spikes->blocks()[seed.noise_block_index];
            displays << "{\"selection\":" << std::quoted(std::string(reasons[slot])) << ",\"candidate\":" << e.request().candidate_index << ",\"detector\":" << block.detector_index << ",\"samples\":[";
            bool comma=false;
            const auto run=std::find_if(runs.begin(),runs.end(),[&](const auto &r){return r.first_native_row<=seed.earlier_row && r.past_last_native_row>seed.later_row;});
            // Hard bounded at 2048 original rows per display, no resampling.
            auto first=std::max(run->first_native_row,seed.earlier_row-1024);
            auto end=std::min(run->past_last_native_row,first+2048);
            for(auto row=first;row<end;++row) {
                const double t=axis->native_identity(row).reconstructed_time_unix_sec();
                if(std::abs(t-e.origin_unix_sec())>2.2+guard) continue;
                if(comma) displays << ','; comma=true;
                displays << '[' << row << ','; number(displays,t-e.origin_unix_sec());
                for(auto coordinate:{pipeline::NativeReadoutCoordinate::x,pipeline::NativeReadoutCoordinate::r}) {
                    displays << ','; number(displays,parent->network(nw).value(coordinate,row,block.detector_index));
                    displays << ',' << (parent->network(nw).state(coordinate,row,block.detector_index).valid()?"true":"false");
                }
                displays << ']';
            }
            displays << "]}\n";
        }
        require(static_cast<bool>(displays),"example output write failed");
        for(std::size_t d=0;d<counts.size();++d) {
            const auto &s=counts[d];
            std::ofstream binding(output/"detector-bindings.jsonl",std::ios::app);
            binding << "{\"detector\":" << d << ",\"occurrence\":" << std::quoted(detectors[d].detector_occurrence_id) << ",\"relation\":" << std::quoted(detectors[d].detector_association_record_id) << ",\"channel\":" << std::quoted(detectors[d].tone_or_channel_id) << "}\n";
            require(static_cast<bool>(binding),"detector binding output write failed");
            summaries << "{\"detector\":" << d << ",\"tune_valid\":" << (tune.valid[d]?"true":"false") << ",\"rows\":" << rows << ",\"pair_screening_excluded_rows\":" << s.paired_excluded_rows << ",\"candidate_counts\":[" << s.candidates[0] << ',' << s.candidates[1] << "],\"available_coordinate_fits\":[" << s.fit_available[0] << ',' << s.fit_available[1] << "],\"available_noise_blocks\":[" << s.noise_available[0] << ',' << s.noise_available[1] << "],\"unavailable_noise_blocks\":[" << s.noise_unavailable[0] << ',' << s.noise_unavailable[1] << "]}\n";
        }
        require(events && blocks && summaries,"census output write failed");
        require(logs->errors==0 && logs->criticals==0,"unexpected producer error-level messages");
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::ofstream receipt(output/"receipt.json");
        receipt << "{\"status\":\"PASS-descriptive-census\",\"observation\":" << obs << ",\"network\":" << nw << ",\"rows\":" << rows << ",\"channels\":" << source.channel_count << ",\"physical_runs\":" << runs.size() << ",\"raw_sha256\":" << std::quoted(citlali::utils::sha256_file(raw_path)) << ",\"tune_sha256\":" << std::quoted(input.tune_sha256) << ",\"manifest_sha256\":" << std::quoted(citlali::utils::sha256_file(manifest)) << ",\"candidate_edges\":" << spikes->candidates().size() << ",\"trial_half_width_seconds\":" << guard << ",\"elapsed_seconds\":" << elapsed << ",\"source_protection\":\"unavailable\",\"classification\":\"not_performed\",\"Apply\":\"not_performed\"}\n";
        require(static_cast<bool>(receipt),"receipt write failed");
        std::cout << "CENSUS obs=" << obs << " network=" << nw << " channels=" << source.channel_count << " rows=" << rows << " candidates=" << spikes->candidates().size() << " seconds=" << elapsed << '\n';
        return 0;
    } catch(const std::exception &e) { std::cerr << "CENSUS INPUT OR EXECUTION FAILURE: " << e.what() << '\n'; return 2; }
}

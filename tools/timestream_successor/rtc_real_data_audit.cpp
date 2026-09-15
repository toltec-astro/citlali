// Inert real-data audit; exact existing verified ingress and runtime Learn.
#define main unused_identity_acceptance_main
#include "identity_route_acceptance.cpp"
#undef main
#include <citlali/core/pipeline/timestream_rtc_line_power.h>
#include <citlali/core/pipeline/timestream_rtc_line_population.h>
#include <bit>
namespace {
void number(std::ostream &o,double v) {if(std::isfinite(v))o<<std::setprecision(17)<<v;else o<<"null";}
void doubles(std::ostream &o,const std::vector<double> &v){o.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(double));}
std::uint64_t pair_fingerprint(const pipeline::NativePairedReadoutNetwork &net){
    std::uint64_t h=1469598103934665603ULL;
    const auto &axis=net.occurrence_axis();
    for(std::uint32_t d=0;d<net.detector_count();++d)
        for(auto c:{pipeline::NativeReadoutCoordinate::x,pipeline::NativeReadoutCoordinate::r})
            for(auto r=axis.first_native_row();r<axis.past_last_native_row();++r){
                h^=std::bit_cast<std::uint64_t>(net.value(c,r,d));h*=1099511628211ULL;
                h^=net.state(c,r,d).valid();h*=1099511628211ULL;
            }
    return h;
}
void regions(std::ostream &o,const pipeline::RtcLinePowerCoordinate &c){
 o<<'[';bool comma=false;
 for(auto &r:c.regions){if(comma)o<<',';comma=true;
 o<<'['<<r.first_bin<<','<<r.past_last_bin<<','<<r.peak_bin<<',';number(o,r.positive_excess_power);o<<',';number(o,r.stored_psd_power_fraction);o<<',';number(o,r.peak_contrast);o<<','<<r.extent_incomplete<<','<<r.background_neighborhood_truncated<<']';}o<<']';
}
}
// argv: raw, exact Tune, compact-v2 manifest, NEW output directory, independent selection TSV.
int main(int argc,char **argv){try{
 const auto execution_started=std::chrono::steady_clock::now();
 require(argc==6 || (argc==7 && std::string(argv[6])=="--population-context"),"expected raw, Tune, APT manifest, NEW output, selection TSV and optional --population-context");
 const fs::path raw_path=argv[1],tune_path=argv[2],manifest=argv[3],output=argv[4];
 require(!fs::exists(output),"output exists; preserve prior attempts");
 require(std::endian::native==std::endian::little,"audit binary format requires little endian");
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
        const auto original_before=pair_fingerprint(parent->network(nw));
        const auto ingress_finished=std::chrono::steady_clock::now();
        auto spikes=pipeline::learn_rtc_spike_candidates(view,val,protection,1);
        const auto spikes_finished=std::chrono::steady_clock::now();

        using namespace pipeline;
        // Numerical audit binding only: four epoch-double ULPs bound arithmetic
        // roundoff, not physical timing uncertainty or an operational jitter policy.
        double max_ulp=0;
        for(std::int64_t r=0;r<rows;++r){const double t=axis->native_identity(r).reconstructed_time_unix_sec();max_ulp=std::max(max_ulp,std::nextafter(t,INFINITY)-t);}
        const double bound=4*max_ulp/duration;
        auto native=ValNativeRealization::create(parent,{ValProducer::align,1},1,ValNativeProductRole::original_input,nw);
        auto identity=RtcSpectralInputIdentity::bind(native,val,view->span(nw),RtcSpectralInputStage::original_reference,"rtc-audit-original-native-producer-clock",1);
        auto spectral=RtcNativeSpectralEvidence::learn_initial(spikes,{identity},{{nw,"audit-only:4-ULP-epoch-roundoff-envelope:not-operational-cadence-qualification",duration,bound}},2);
        const auto spectral_finished=std::chrono::steady_clock::now();
        auto line1=RtcLinePowerEvidence::learn(spectral,val,RtcLinePowerProfile::sensitivity_1_hz,3);
        auto line2=RtcLinePowerEvidence::learn(spectral,val,RtcLinePowerProfile::initial_2_hz,3);
        auto line4=RtcLinePowerEvidence::learn(spectral,val,RtcLinePowerProfile::sensitivity_4_hz,3);
        const auto lines_finished=std::chrono::steady_clock::now();
        fs::create_directories(output);
        std::ofstream meta(output/"spectra.jsonl"),binary(output/"spectra.f64",std::ios::binary),windows(output/"windows.f64",std::ios::binary),native_time(output/"native-time.f64",std::ios::binary);
        require(meta&&binary&&windows&&native_time,"cannot open audit outputs");
        std::vector<double> relative_time;relative_time.reserve(rows);
        const auto origin=axis->native_identity(0).reconstructed_time_unix_sec();
        for(std::int64_t r=0;r<rows;++r)relative_time.push_back(axis->native_identity(r).reconstructed_time_unix_sec()-origin);
        doubles(native_time,relative_time);
        const auto &sn=spectral->network(nw);const auto &net=parent->network(nw);
        std::vector<double> hann(sn.fft_samples);
        for(std::size_t i=0;i<hann.size();++i)hann[i]=.5-.5*std::cos(2*std::numbers::pi*i/(hann.size()-1));
        std::vector<double> sens(source.channel_count,NAN),flags(source.channel_count,NAN),flags2(source.channel_count,NAN);
        std::vector<int> arrays(source.channel_count,-1);
        for(const auto &a:verified.apt.rows)if(a.network==nw){
            require(a.channel>=0&&a.channel<source.channel_count,"APT channel out of range");arrays[a.channel]=a.array;
            for(auto field:{"sens","flag","flag2"}){const auto it=a.fields.find(field);double value=NAN;if(it!=a.fields.end()){if(auto v=std::get_if<double>(&it->second))value=*v;else if(auto v=std::get_if<std::int64_t>(&it->second))value=*v;}
                (std::string(field)=="sens"?sens:std::string(field)=="flag"?flags:flags2)[a.channel]=value;
            }
        }
        std::size_t window_offset=0,coordinate_index=0;double max_replay_error=0;
        if (argc == 7) {
            // Runtime observation context is constructed from this invocation's
            // actual Learn handles, never from offline corpus rankings.
            const auto population_started = std::chrono::steady_clock::now();
            std::vector<RtcEventPeerEligibility> peers;
            std::vector<RtcLinePopulationMember> members;
            const auto population_weight_authority = "sha256:" + citlali::utils::sha256_file(manifest);
            for (std::uint32_t d = 0; d < net.detector_count(); ++d) {
                const auto &binding = net.detector(d);
                const bool good = flags[d] == 0 && flags2[d] == 0;
                peers.push_back({nw, d, binding.detector_occurrence_id, good});
                members.push_back({nw, d, binding.detector_occurrence_id,
                    binding.detector_association_record_id, "original-native-x:observation-local-APT-proxy",
                    static_cast<RtcOpticalArray>(arrays[d]),
                    good && std::isfinite(sens[d]) && sens[d] > 0 ? std::optional<double>(1 / (sens[d]*sens[d])) : std::nullopt,
                    population_weight_authority,
                    "static-APT-sens-inverse-square;flag=flag2=0;conditional-independent-noise-proxy"});
            }
            auto peer = RtcEventPeerPopulation::admit(spikes, "exact-APT-good-observation-population", std::move(peers));
            auto events = RtcEventAssessmentDecision::consider(learn_rtc_event_assessment(spikes, peer, 10), val, 11);
            auto amplitude = RtcJumpAmplitudeDecision::consider(events, val, 12);
            auto consistency = RtcJumpConsistencyDecision::consider(RtcJumpConsistencyEvidence::learn(amplitude, 13), val, 14);
            auto transition = RtcJumpTransitionEvidence::learn(RtcJumpTransitionRequest::consider(consistency, val, 15), 16);
            auto support = RtcJumpSupportEvidence::learn(transition, 17);
            auto refit = RtcJumpRefitEvidence::learn(RtcJumpRefitRequest::consider(support, val, 18), 19);
            auto remeasurement = RtcJumpReassessmentEvidence::learn(RtcJumpRemeasureRequest::consider(refit, val, 20), 21);
            auto admitted = RtcJumpAdmissionDecision::consider(RtcJumpReassessmentDecision::consider(remeasurement, val, 22), val, 23);
            auto jumps = RtcJumpExclusionPlan::consider(admitted, nullptr, val, 24);
            auto transients = RtcTransientExclusionPlan::consider(events->original_screening_handle(), jumps, val, 25);
            const auto population_ready = std::chrono::steady_clock::now();
            auto population = RtcLinePopulationEvidence::learn(line2, transients, std::move(members), 26);
            const auto population_finished = std::chrono::steady_clock::now();
            std::ofstream pop(output / "population.jsonl");
            for (const auto &d : population->detectors()) {
                pop << "{\"detector\":" << d.member.detector << ",\"occurrence\":" << std::quoted(d.member.occurrence)
                    << ",\"array_association\":" << std::quoted(d.member.array_association)
                    << ",\"array\":" << static_cast<int>(d.member.array)
                    << ",\"weight\":"; number(pop, d.member.reference_weight.value_or(NAN));
                pop << ",\"weight_authority\":" << std::quoted(d.member.weight_authority)
                    << ",\"paired_original_cells\":" << d.paired_original_cells
                    << ",\"after_existing_exclusions_cells\":" << d.after_existing_exclusions_cells
                    << ",\"support\":[";
                bool comma = false;
                for (const auto &s : d.support) { if (comma) pop << ','; comma = true;
                    pop << '[' << s.rows.first << ',' << s.rows.past_last << ',' << static_cast<int>(s.cause_bits) << ']'; }
                pop << "]}\n";
            }
            pop.close(); require(bool(pop), "runtime population export failed");
            std::ofstream pr(output / "population-receipt.json");
            pr << "{\"source_revision\":" << std::quoted(std::string(CITLALI_GIT_REVISION))
               << ",\"runtime_Learn\":true,\"offline_corpus_rank_consumed\":false,\"attempt\":26,\"VAL_generation\":0,\"motion_context\":null,\"scan_binding\":null,\"treatment_selected\":false,\"population_logical_owned_bytes\":" << population->logical_owned_bytes()
               << ",\"population_seconds\":";
            number(pr, std::chrono::duration<double>(population_finished-population_ready).count());
            pr << ",\"existing_transient_context_seconds\":";
            number(pr, std::chrono::duration<double>(population_ready-population_started).count());
            pr << ",\"seconds\":";
            number(pr, std::chrono::duration<double>(std::chrono::steady_clock::now()-population_started).count());
            pr << "}\n"; pr.close(); require(bool(pr), "population receipt failed");
        }
        for(const auto &s:spectral->spectra()){
            const auto &l1=line1->coordinates()[coordinate_index],&l2=line2->coordinates()[coordinate_index],&l4=line4->coordinates()[coordinate_index];
            std::vector<double> empty(sn.frequency_hz.size(),NAN);
            for(const auto *v:{&s.psd,&l1.background,&l2.background,&l4.background})doubles(binary,v->empty()?empty:*v);
            meta<<"{\"detector\":"<<s.detector<<",\"coordinate\":"<<static_cast<int>(s.coordinate)<<",\"occurrence\":"<<std::quoted(detectors[s.detector].detector_occurrence_id)<<",\"tune_valid\":"<<(tune.valid[s.detector]?"true":"false")<<",\"cause\":"<<static_cast<int>(s.cause)<<",\"window_offset\":"<<window_offset<<",\"window_count\":"<<s.windows.size()<<",\"population_median\":";number(meta,s.population_median);
            meta<<",\"array\":"<<arrays[s.detector]<<",\"apt_sens\":";number(meta,sens[s.detector]);meta<<",\"apt_flag\":";number(meta,flags[s.detector]);meta<<",\"apt_flag2\":";number(meta,flags2[s.detector]);
            meta<<",\"centering_support\":[";for(std::size_t i=0;i<s.centering_support.size();++i){if(i)meta<<',';meta<<'['<<s.centering_support[i].first<<','<<s.centering_support[i].past_last<<']';}meta<<"],\"runs\":[";
            for(std::size_t i=0;i<s.runs.size();++i){if(i)meta<<',';const auto &r=s.runs[i];meta<<'['<<r.rows.first<<','<<r.rows.past_last<<','<<static_cast<int>(r.cause)<<','<<r.admitted_samples<<','<<r.declared_invalid_samples<<','<<r.unexpected_nonfinite_samples<<']';}
            meta<<"],\"regions_1\":";regions(meta,l1);meta<<",\"regions_2\":";regions(meta,l2);meta<<",\"regions_4\":";regions(meta,l4);meta<<"}\n";
            if(s.available()){
                std::size_t target=0;double best=-1;
                for(const auto &r:l2.regions)if(sn.frequency_hz[r.peak_bin]>=2 && r.positive_excess_power>best){best=r.positive_excess_power;target=r.peak_bin;}
                Eigen::FFT<double> fft;fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
                std::vector<double> sum(s.psd.size(),0.);
                for(const auto &win:s.windows){
                    // Re-express each accepted stored window for time-local audit;
                    // reproduce its accepted pooled PSD before trusting this export.
                    std::vector<double> samples;
                    for(auto r=win.rows.first;r<win.rows.past_last;++r)samples.push_back((net.value(s.coordinate,r,s.detector)-s.population_median)-win.centered_chunk_median);
                    samples.resize(sn.fft_samples,0.);for(std::size_t i=0;i<samples.size();++i)samples[i]*=hann[i];
                    std::vector<std::complex<double>> ft;fft.fwd(ft,samples);
                    std::vector<double> p(s.psd.size());double total=0,low=0;std::array<double,4> bands{};
                    for(std::size_t k=0;k<p.size();++k){p[k]=std::norm(ft[k])/sn.window_norm;if(p.size()>2&&k>0&&k+1<p.size())p[k]*=2;sum[k]+=p[k];total+=p[k];const auto f=sn.frequency_hz[k];if(f<2)low+=p[k];bands[f<2?0:f<10?1:f<30?2:3]+=p[k];}
                    auto triplet=[&](std::size_t k){double v=0;for(auto j=k?k-1:0;j<std::min(p.size(),k+2);++j)v+=p[j];return v;};
                    double peak3=-1;std::size_t peak=0;
                    for(std::size_t k=1;k+1<p.size();++k)if(sn.frequency_hz[k]>=2&&triplet(k)>peak3){peak3=triplet(k);peak=k;}
                    std::vector<double> record{static_cast<double>(s.detector),static_cast<double>(s.coordinate),static_cast<double>(win.rows.first),static_cast<double>(win.rows.past_last),total*l2.bin_increment_hz,total>0?low/total:NAN,static_cast<double>(peak),total>0?peak3/total:NAN,static_cast<double>(target),total>0?triplet(target)/total:NAN};
                    for(auto b:bands)record.push_back(b*l2.bin_increment_hz);
                    doubles(windows,record);++window_offset;
                }
                const double scale=*std::max_element(s.psd.begin(),s.psd.end());
                for(std::size_t k=0;k<s.psd.size();++k){double error=std::abs(sum[k]/s.windows.size()-s.psd[k]);max_replay_error=std::max(max_replay_error,scale>0?error/scale:error);require(error<=2e-12*scale,"accepted-window replay differs from pooled PSD");}
            }else require(s.windows.empty(),"unavailable spectrum has unexported windows");
            ++coordinate_index;
        }
        std::ifstream selection(argv[5]);require(static_cast<bool>(selection),"independent selection absent");
        std::set<int> selected;int so,snwid,sd;
        while(selection>>so>>snwid>>sd)if(so==obs&&snwid==nw){require(sd>=0&&sd<source.channel_count,"selection detector out of range");selected.insert(sd);}
        // Four supplementary raw inspections ranked only after independent draws.
        std::vector<std::pair<double,int>> scores;
        for(const auto &s:line2->coordinates())if(s.available()&&s.coordinate==NativeReadoutCoordinate::x){double best=0;for(const auto &r:s.regions)if(r.peak_bin<sn.frequency_hz.size()&&sn.frequency_hz[r.peak_bin]>=2&&std::isfinite(r.stored_psd_power_fraction))best=std::max(best,r.stored_psd_power_fraction);scores.emplace_back(best,s.detector);}
        std::sort(scores.rbegin(),scores.rend());for(std::size_t i=0;i<std::min(std::size_t{4},scores.size());++i)selected.insert(scores[i].second);
        for(auto d:selected){std::ofstream rawout(output/("samples-"+std::to_string(d)+".f64"),std::ios::binary);require(static_cast<bool>(rawout),"sample output unavailable");for(std::int64_t r=0;r<rows;++r)doubles(rawout,{net.value(NativeReadoutCoordinate::x,r,d),net.value(NativeReadoutCoordinate::r,r,d),static_cast<double>(net.state(NativeReadoutCoordinate::x,r,d).valid()),static_cast<double>(net.state(NativeReadoutCoordinate::r,r,d).valid())});rawout.close();require(static_cast<bool>(rawout),"sample close failed");}
        for(auto *o:{&meta,&binary,&windows,&native_time}){o->close();require(static_cast<bool>(*o),"audit output close failed");}
        const auto original_after=pair_fingerprint(net);require(original_before==original_after,"original paired value/state fingerprint changed");
        const auto finished=std::chrono::steady_clock::now();
        auto seconds=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
        std::ofstream receipt(output/"receipt.json");
        receipt<<"{\"status\":\"PASS-inert-audit\",\"compiled_revision\":"<<std::quoted(std::string(CITLALI_GIT_REVISION))<<",\"observation\":"<<obs<<",\"network\":"<<nw<<",\"rows\":"<<rows<<",\"channels\":"<<source.channel_count<<",\"raw_sha256\":"<<std::quoted(citlali::utils::sha256_file(raw_path))<<",\"tune_sha256\":"<<std::quoted(citlali::utils::sha256_file(tune_path))<<",\"manifest_sha256\":"<<std::quoted(citlali::utils::sha256_file(manifest))<<",\"native_origin_unix_sec\":";number(receipt,origin);
        receipt<<",\"nominal_interval\":";number(receipt,duration);receipt<<",\"roundoff_bound_fraction\":";number(receipt,bound);receipt<<",\"measured_interval\":";number(receipt,sn.interval_seconds);receipt<<",\"minimum_interval\":";number(receipt,sn.minimum_interval_seconds);receipt<<",\"maximum_interval\":";number(receipt,sn.maximum_interval_seconds);
        receipt<<",\"cadence_available\":"<<(sn.cadence_available?"true":"false")<<",\"physical_runs\":"<<runs.size()<<",\"fft_samples\":"<<sn.fft_samples<<",\"bins\":"<<sn.frequency_hz.size()<<",\"window_records\":"<<window_offset<<",\"window_record_columns\":14,\"window_replay_max_scaled_error\":";number(receipt,max_replay_error);
        receipt<<",\"original_pair_fingerprint_before\":"<<std::quoted(std::to_string(original_before))<<",\"original_pair_fingerprint_after\":"<<std::quoted(std::to_string(original_after))<<",\"source_protection\":\"unknown-retained\",\"use_policy\":"<<std::quoted(std::string(RtcInitialSpectralPolicy::identity))<<",\"VAL_generation\":0,\"producer_attempt\":1,\"spectral_attempt\":2,\"line_attempt\":3,\"Apply\":false,\"stages_seconds\":{\"ingress\":";number(receipt,seconds(execution_started,ingress_finished));receipt<<",\"spikes\":";number(receipt,seconds(ingress_finished,spikes_finished));receipt<<",\"spectral\":";number(receipt,seconds(spikes_finished,spectral_finished));receipt<<",\"lines\":";number(receipt,seconds(spectral_finished,lines_finished));receipt<<",\"audit_export\":";number(receipt,seconds(lines_finished,finished));receipt<<"}}\n";
        receipt.close();require(static_cast<bool>(receipt),"receipt close failed");
        std::cout<<"RTC real-data audit PASS obs="<<obs<<" nw="<<nw<<" seconds="<<seconds(execution_started,finished)<<"\n";return 0;
    }catch(const std::exception &e){std::cerr<<"RTC real-data audit FAIL: "<<e.what()<<'\n';return 1;}}

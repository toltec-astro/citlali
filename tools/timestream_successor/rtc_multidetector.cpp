// Bounded representative RTC caller. Scientific owners remain the existing
// typed components; YAML and file checks live only at this offline boundary.
#define main unused_identity_acceptance_main
#include "identity_route_acceptance.cpp"
#undef main
#include "rtc_multidetector_bindings.h"
#include <citlali/core/pipeline/timestream_rtc_output_grid.h>
#include <bit>

namespace {
using namespace citlali::pipeline;
namespace caller = citlali::rtc_multidetector_tool;
fs::path checked_file(const YAML::Node &n) {
  const fs::path p=n["path"].as<std::string>();
  require(citlali::utils::sha256_file(p)==n["sha256"].as<std::string>(),"input digest mismatch: "+p.string());
  return p;
}
double field(const auto &row,const std::string &name) {
  auto it=row.fields.find(name);if(it==row.fields.end()) return NAN;
  if(auto value=std::get_if<double>(&it->second)) return *value;
  if(auto value=std::get_if<std::int64_t>(&it->second)) return *value;
  return NAN;
}
void write_yaml(const fs::path &p,const YAML::Node &node) {
  YAML::Emitter serialized;serialized.SetDoublePrecision(17);serialized<<node;
  std::ofstream out(p);out<<serialized.c_str()<<'\n';out.close();require(bool(out),"required output failed: "+p.string());
}
YAML::Node range(RtcEventRange r) {YAML::Node n;n.push_back(r.first);n.push_back(r.past_last);return n;}
void write_matrix(const fs::path &p,const auto &m) {
  std::ofstream out(p,std::ios::binary);
  for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j){double v=m(i,j);out.write(reinterpret_cast<const char*>(&v),8);}
  out.close();require(bool(out),"required matrix output failed");
}
}
#include "rtc_processing_scan_input.h"
#include "rtc_common_mode_output.h"
#include "rtc_treatment_outcome_output.h"
#include "rtc_consequence_study.h"
int main(int argc,char **argv) {
  try {
    const auto began=std::chrono::steady_clock::now();
    require(argc==3 || argc==4,"expected input JSON, NEW output directory, optional explicit reviewed-selection YAML");
    require(std::endian::native==std::endian::little,"native binary audit requires little endian");
    const auto cfg=YAML::LoadFile(argv[1]);
    require(cfg["schema"].as<std::string>()=="rtc-multidetector-experiment-v1","unexpected caller profile");
    const fs::path output=argv[2];require(!fs::exists(output),"preserve previous output");
    auto checked=[&](const std::string &key){return checked_file(cfg[key]);};
    const auto raw_path = checked("raw"), tune_path = checked("tune"),
               manifest = checked("manifest");
    const auto prior_receipt_path = checked("audit_receipt");
    const auto prior = YAML::LoadFile(prior_receipt_path.string());
    require(prior["raw_sha256"].as<std::string>() ==
                    citlali::utils::sha256_file(raw_path) &&
                prior["tune_sha256"].as<std::string>() ==
                    citlali::utils::sha256_file(tune_path) &&
                prior["manifest_sha256"].as<std::string>() ==
                    citlali::utils::sha256_file(manifest),
            "audit raw/Tune/APT binding mismatch");
    auto [logger, logs] = configure_logging();
    const auto verified = apt::verify_bundle_filesystem(manifest, true);
    const auto relation =
        pipeline::admit_canonical_apt_detector_relation_v2(verified);
    netCDF::NcFile raw_file(raw_path.string(), netCDF::NcFile::read);
    const auto nw =
        read_netcdf_scalar<int>(raw_file, "Header.Toltec.RoachIndex");
    const auto obs = read_netcdf_scalar<int>(raw_file, "Header.Toltec.ObsNum");
    const auto sub =
        read_netcdf_scalar<int>(raw_file, "Header.Toltec.SubObsNum");
    const auto scan =
        read_netcdf_scalar<int>(raw_file, "Header.Toltec.ScanNum");
    require(obs == relation.observation().observation &&
                sub == relation.observation().subobservation &&
                scan == relation.observation().scan,
            "raw observation differs from exact APT relation");
    const auto found = std::find_if(
        relation.raw_sources().begin(), relation.raw_sources().end(),
        [&](const auto &s) { return s.network == nw; });
    require(found != relation.raw_sources().end(),
            "APT does not bind input network");
    const auto source = *found;
    verify_raw_file(raw_path, source);
    const auto rows = static_cast<std::int64_t>(
        raw_file.getVar("Data.Toltec.Is").getDim(0).getSize());
    const auto tune_digest = "sha256:" + citlali::utils::sha256_file(tune_path);
    const auto kmp = std::find_if(
        verified.sources.begin(), verified.sources.end(), [&](const auto &s) {
          return s.role == apt::SourceRole::kmp && s.network == nw;
        });
    require(kmp != verified.sources.end() &&
                kmp->content_sha256 == tune_digest &&
                kmp->byte_count == fs::file_size(tune_path),
            "Tune report differs from exact APT KMP source binding");
    const auto tune = read_tune_facts(tune_path, source.channel_count);
    require(tune.observation ==
                    read_netcdf_scalar<int>(raw_file,
                                            "Header.Toltec.TargSweepObsNum") &&
                tune.subobservation ==
                    read_netcdf_scalar<int>(
                        raw_file, "Header.Toltec.TargSweepSubObsNum") &&
                tune.scan == read_netcdf_scalar<int>(
                                 raw_file, "Header.Toltec.TargSweepScanNum") &&
                tune.network == nw,
            "Tune report differs from raw header's calibration relation");
    double fpga = 0, hz = 0;
    std::int64_t accum = 0;
    auto ts = read_timestamp_slice(raw_path, 0, rows, fpga, hz, accum);
    require(fpga > 0 && hz > 0 && accum > 0 && tune.accumulation_length > 0,
            "invalid producer cadence");
    const double duration = static_cast<double>(accum) / fpga;
    require(std::abs(duration - 1 / hz) <=
                8 * std::numeric_limits<double>::epsilon() * duration,
            "producer cadence fields disagree");
    // Native producer clock only; no telescope synchronization or cross-network
    // timing claim.
    auto timing = std::make_shared<const pipeline::NativeNetworkAlignment>(
        pipeline::make_native_network_alignment(nw, 0, ts, fpga, 0.0));
    NetworkInput input{
        source,     raw_path, tune_path, citlali::utils::sha256_file(tune_path),
        fpga,       hz,       accum,     tune.accumulation_length,
        tune.valid, timing};
    const bool full_health=cfg["common_mode_health"] && cfg["common_mode_health"].as<std::string>()=="full-network-learn-only";
    const bool census = bool(cfg["common_mode_census"]);
    if (census) {
      require(full_health && nw == 0 && (obs == 152390 || obs == 152392) && argc == 3 &&
                  !cfg["declared_contaminant"], "census is bounded original network0 Learn-only");
      const auto selection = YAML::LoadFile(checked_file(cfg["common_mode_census"]["selection"]).string());
      require(selection["schema"].as<std::string>() == "rtc-common-mode-repeatability-selection-v1" &&
                  selection["selected_before_new_health_results"].as<bool>(), "census selection was not frozen");
      std::size_t matches=0;
      for (const auto &entry : selection["observations"])
        if (entry["observation"].as<int>() == obs) {
          ++matches;
          for (const auto &key : {"raw", "tune", "manifest", "telescope"})
            require(entry[key]["sha256"].as<std::string>() == cfg[key]["sha256"].as<std::string>(),
                    "census input differs from frozen selection");
        }
      require(matches==1,"census observation absent or duplicated in selection");
    }
    require((obs == 152390 || (census && obs == 152392)) && sub == 0 && scan == 2 && (nw == 12 || (full_health && nw == 0)),
            "bounded caller requires NGC4449/152390/0/2: network12 Apply or network0 diagnostic only");
    const int health_array=nw==0 ? 0 : 2;
    require(rows == prior["rows"].as<std::int64_t>() &&
                nw == prior["network"].as<int>() &&
                source.channel_count == prior["channels"].as<int>(),
            "audit shape/scope mismatch");
    std::vector<int> channels;
    std::vector<std::string> sample_hashes;
    require(cfg["detectors"].IsSequence() && cfg["detectors"].size() >= 2,
            "multi-detector caller requires explicit simultaneous columns");
    NativePairedReadoutMatrix x(rows, cfg["detectors"].size()), r(rows, cfg["detectors"].size());
    std::vector<NativeReadoutCoordinateState> xs(rows*x.cols(),
        NativeReadoutCoordinateState::measured(true,false,true,false)), rs(xs);
    for (std::size_t d=0; d<cfg["detectors"].size(); ++d) {
      const auto entry=cfg["detectors"][d];
      const int channel=entry["channel"].as<int>();
      require(channel>=0 && channel<source.channel_count &&
                  (channels.empty() || channel>channels.back()),
              "projected channels must be unique and sorted native channels");
      const auto path=checked_file(entry["samples"]);
      require(path.filename()=="samples-"+std::to_string(channel)+".f64" &&
                  fs::file_size(path)==static_cast<std::uintmax_t>(rows)*32,
              "audit column filename or size mismatch");
      channels.push_back(channel);
      sample_hashes.push_back(citlali::utils::sha256_file(path));
      std::ifstream stream(path,std::ios::binary);
      for(std::int64_t row=0; row<rows; ++row) {
        std::array<double,4> a;
        stream.read(reinterpret_cast<char*>(a.data()),32);
        require(bool(stream),"audit sample read failed");
        x(row,d)=a[0];r(row,d)=a[1];
        require((a[2]==0 || a[2]==1) && (a[3]==0 || a[3]==1),"audit state malformed");
        const auto cell=static_cast<std::size_t>(row*x.cols()+d);
        xs[cell]=NativeReadoutCoordinateState::measured(true,tune.valid[channel] && std::isfinite(a[0]),true,std::isfinite(a[0]));
        rs[cell]=NativeReadoutCoordinateState::measured(true,tune.valid[channel] && std::isfinite(a[1]),true,std::isfinite(a[1]));
        require(xs[cell].valid()==(a[2]!=0) && rs[cell].valid()==(a[3]!=0),
                "export differs from exact producer state convention");
      }
    }
    YAML::Node contaminant_record;
    if(cfg["declared_contaminant"]){
      const auto model=cfg["declared_contaminant"];
      const int channel=model["channel"].as<int>();
      const auto it=std::find(channels.begin(),channels.end(),channel);
      require(it!=channels.end(),"contaminant target outside cohort");
      const auto d=it-channels.begin();const auto row=model["native_row"].as<std::int64_t>();
      require(row>0 && row+1<rows && xs[row*x.cols()+d].valid() && rs[row*x.cols()+d].valid(),
        "contaminant requires originally usable paired support");
      const double dx=model["x_delta"].as<double>(),dr=model["r_delta"].as<double>();
      require(std::isfinite(dx) && std::isfinite(dr) && dx!=0 && dr!=0,"invalid declared paired contaminant");
      contaminant_record=YAML::Clone(model);contaminant_record["original_x"]=x(row,d);contaminant_record["original_r"]=r(row,d);
      x(row,d)+=dx;r(row,d)+=dr;
    }
    const auto original_x=x, original_r=r;
    auto config = load_runtime_config(checked("effective_config"));
    require(config.interface_offset_present[nw] &&
                config.interface_offsets_sec[nw] == 0,
            "audit and AST timing require the accepted zero network offset");
    auto mapping = std::make_shared<pipeline::NativeReadoutMappingAuthority>(
        *mapping_identity(input, config));
    mapping->applicability_domain_id =
        "observation=" + std::to_string(obs) + ":network=" + std::to_string(nw);
    mapping->event_time_epoch_meaning_id =
        "producer-native-clock+exact-effective-zero-interface-offset:accepted-"
        "AST-mapping";
    std::string projection;
    for(std::size_t d=0;d<channels.size();++d)
      projection+=std::to_string(channels[d])+":"+sample_hashes[d]+"\n";
    mapping->paired_xr_record_id += ":exact-simultaneous-column-projection:sha256:"+
        citlali::utils::sha256(projection);
    if(cfg["declared_contaminant"])mapping->paired_xr_record_id+=":declared-contaminant:"+
        citlali::utils::sha256_file(argv[1]);
    mapping->timing_uncertainty_state_id =
        "unquantified:uniform-average-center-trial:rtc-native-readout-uniform-"
        "average-assumption-v1";
    // The owner-approved uniform-average assumption remains provisional.
    // AST uses the recovered effective zero interface offset; the readout
    // midpoint/boxcar remains the explicit provisional owner assumption.
    auto axis = occurrence_axis(input, 0, rows,
                                NativeEventTimeRole::integration_center);
    const auto runs = axis->contiguous_runs();
    const auto all_detectors=detector_axis(relation,input);
    std::vector<NativeReadoutDetectorBinding> detectors;
    std::vector<double> factors;
    std::vector<bool> peer_good;
    for(std::size_t d=0;d<channels.size();++d) {
      auto binding=all_detectors.at(channels[d]);binding.storage_column=d;
      detectors.push_back(std::move(binding));
      const auto aptrow=std::find_if(verified.apt.rows.begin(),verified.apt.rows.end(),
          [&](const auto &a){return a.network==nw && a.channel==channels[d];});
      require(aptrow!=verified.apt.rows.end() && aptrow->array==health_array,"projection must bind exact within-network APT array rows");
      factors.push_back(field(*aptrow,"flxscale"));
      peer_good.push_back(field(*aptrow,"flag")==0 && field(*aptrow,"flag2")==0);
    }
    std::vector<NativePairedReadoutNetwork> networks;
    networks.push_back(NativePairedReadoutNetwork::admit(axis,detectors,mapping,
        std::move(x),std::move(r),std::move(xs),std::move(rs)));
    auto parent = std::make_shared<const NativePairedReadoutObservation>(
        NativePairedReadoutObservation::admit(
            NativeObservationScope{obs, sub, scan}, {nw}, std::move(networks)));
    std::optional<RtcDeclaredContaminant> declared_contaminant;
    if(cfg["declared_contaminant"]){
      const auto m=cfg["declared_contaminant"];const auto row=m["native_row"].as<std::int64_t>();
      const auto d=std::find(channels.begin(),channels.end(),m["channel"].as<int>())-channels.begin();
      const auto reference=checked_file(m["reference_receipt"]);
      const auto ref=YAML::LoadFile(reference.string());
      const auto reference_configuration=checked_file(m["reference_configuration"]);
      auto unmodified_config=YAML::Clone(cfg);unmodified_config.remove("declared_contaminant");
      require(ref["source_revision"].as<std::string>()==std::string(CITLALI_GIT_REVISION) &&
        ref["configuration_sha256"].as<std::string>()==citlali::utils::sha256_file(reference_configuration) &&
        YAML::Dump(unmodified_config)==YAML::Dump(YAML::LoadFile(reference_configuration.string())),
        "declared contaminant may not alter the untouched reference configuration");
      require(ref["Apply_performed"].as<bool>() && ref["original_pair_unchanged"].as<bool>() &&
        ref["original_parent"].as<std::string>()+":declared-contaminant:"+citlali::utils::sha256_file(argv[1])==mapping->paired_xr_record_id,
        "contaminant does not identify its untouched applied reference");
      declared_contaminant=RtcDeclaredContaminant{parent,nw,static_cast<std::uint32_t>(d),{row,row+1},
        "sha256:"+citlali::utils::sha256_file(reference),"one-occurrence-additive-x-and-r-test-contaminant;not-a-sky-injection"};
    }
    auto val=ValSnapshot::initial(parent);
    auto view=NativePairedReadoutView::full(parent);
    std::optional<RecoveredProcessingScans> recovered_scans;
    if(cfg["decision_apply"]) recovered_scans=recover_processing_scans(cfg,parent,verified,nw);
    const auto protection_authority=cfg["source_protection_authority"].as<std::string>();
    if (obs == 152390) caller::require_no_mask_scope(parent->scope(),protection_authority);
    else require(census && protection_authority == "rtc-census-source-membership-unavailable",
                 "repeat source membership must remain unavailable");
    auto protection=RtcSpikeSourceProtection::admit(parent,protection_authority,
                        obs==152390 ? RtcSpikeProtection::outside_source : RtcSpikeProtection::unavailable);
    std::shared_ptr<const AstScanMotionNetworkView> motion;
    if(argc==4 || recovered_scans){
      const auto telescope=load_telescope(checked("telescope"),parent->scope(),census && obs==152392);
      const auto motion_identity = census && obs==152392
          ? AstScanMotionIdentityBinding{1523920001,1523920002,1523920003,1523920004}
          : ast_identity_binding;
      auto ast=build_ast_scan_motion_product(telescope.source,motion_identity);
      const auto accepted_ast=YAML::LoadFile(checked("ast_acceptance").string());
      if (census && obs == 152392) {
        require(accepted_ast["schema"].as<std::string>() == "citlali-wp7-rtc-filter-fixture-census-v3" &&
                    accepted_ast["source_revision"].as<std::string>() == "adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7" &&
                    accepted_ast["observation"].as<int>() == obs &&
                    accepted_ast["telescope_ast"]["policy_id"].as<std::string>() == std::string(ast_scan_motion_policy_id),
                "repeat AST evidence differs from accepted scope/policy");
        std::size_t matches=0;
        for (const auto &entry : accepted_ast["inputs"])
          if (entry["role"].as<std::string>() == "telescope") {
            ++matches;
            require(entry["sha256"].as<std::string>() == telescope.sha256,
                    "repeat telescope differs from accepted motion census");
          }
        require(matches==1,"repeat AST requires one exact telescope input");
      } else require(accepted_ast["source_revision"].as<std::string>()=="adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7" &&
          accepted_ast["authority_policy_id"].as<std::string>()==std::string(ast_scan_motion_policy_id) &&
          accepted_ast["observation"].as<int>()==obs &&
          accepted_ast["telescope"]["sha256"].as<std::string>()==telescope.sha256,
          "AST acceptance differs from exact telescope/policy/scope");
      motion=AstScanMotionNetworkView::admit(ast,timing);
    }
    const auto ingress_finished=std::chrono::steady_clock::now();
    double learn_seconds=0,consider_seconds=0;
    auto measure=[](double &seconds,auto operation){const auto at=std::chrono::steady_clock::now();
      auto result=operation();seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-at).count();return result;};
    auto spikes=measure(learn_seconds,[&]{return learn_rtc_spike_candidates(view,val,protection,1);});
    std::shared_ptr<const RtcCommonModeEvidence> health;
    double health_seconds=0;
    if(cfg["common_mode_health"]) {
      require(recovered_scans.has_value() && motion,"health needs the existing processing and motion binding");
      RtcCommonModeDomain domain;domain.network=nw;domain.scans=recovered_scans->projection.binding;
      domain.motion=motion;domain.population_authority="exact-APT-flag=flag2=0-and-Tune-valid:sha256:"+citlali::utils::sha256_file(manifest);
      domain.array=nw==0 ? RtcOpticalArray::a1100 : RtcOpticalArray::a2000;domain.nominal_interval_seconds=duration;
      domain.speed_ceiling_arcsec_per_sec=235;domain.output_factor=2;
      for(std::size_t d=0;d<channels.size();++d)domain.members.push_back({detectors[d].detector_occurrence_id,peer_good[d] && tune.valid[channels[d]],factors[d]});
      health=measure(health_seconds,[&]{return RtcCommonModeEvidence::learn(spikes,std::move(domain),30);});
      if(cfg["common_mode_health"].as<std::string>()=="full-network-learn-only") {
        require(channels.size()==static_cast<std::size_t>(source.channel_count),"health replay requires full network");
        auto native=ValNativeRealization::create(parent,{ValProducer::align,1},1,ValNativeProductRole::original_input,nw);
        auto identity=RtcSpectralInputIdentity::bind(native,val,view->span(nw),RtcSpectralInputStage::original_reference,"exact-simultaneous-audit-original-projection",1);
        auto spectra=measure(learn_seconds,[&]{return RtcNativeSpectralEvidence::learn_initial(spikes,{identity},{{nw,"audit-four-epoch-ULP-arithmetic-envelope",duration,prior["roundoff_bound_fraction"].as<double>()}},18);});
        auto lines=RtcLinePowerEvidence::learn(spectra,val,RtcLinePowerProfile::initial_2_hz,19);
        std::vector<std::uint32_t> fixed_targets;
        if (census) {
          const auto selection=YAML::LoadFile(checked_file(cfg["common_mode_census"]["selection"]).string());
          for (const auto &target:selection["follow_targets"]) {
            const auto channel=target[obs==152390 ? "baseline_channel" : "repeat_channel"];
            if (channel.IsNull()) continue;
            const auto it=std::find(channels.begin(),channels.end(),channel.as<int>());
            require(it!=channels.end(),"follow target absent from exact inventory");
            fixed_targets.push_back(it-channels.begin());
          }
        }
        export_health(output/"health",health,lines,channels,cfg,argv[1],health_seconds,
                      census ? &fixed_targets : nullptr);
        if (census) export_census_reference_checks(output,health,lines,channels,cfg,argv[1],fixed_targets);
        write_yaml(output/"processing-scans.yaml",recovered_scans->receipt);
        const auto &net=parent->network(nw);
        for(std::uint32_t d=0;d<channels.size();++d)for(std::int64_t row=0;row<rows;++row)
          require(std::bit_cast<std::uint64_t>(original_x(row,d))==std::bit_cast<std::uint64_t>(net.value(NativeReadoutCoordinate::x,row,d)) &&
            std::bit_cast<std::uint64_t>(original_r(row,d))==std::bit_cast<std::uint64_t>(net.value(NativeReadoutCoordinate::r,row,d)),"health changed original pair");
        YAML::Node receipt;receipt["source_revision"]=std::string(CITLALI_GIT_REVISION);receipt["configuration_sha256"]=citlali::utils::sha256_file(argv[1]);
        receipt["original_pair_unchanged"]=true;receipt["Apply_performed"]=false;receipt["production_filtering_active"]=false;
        receipt["status"]="PASS-full-network-diagnostic-only";receipt["rows"]=rows;receipt["detectors"]=channels.size();receipt["network"]=nw;receipt["native_integration_seconds"]=duration;
        receipt["ingress_seconds"]=std::chrono::duration<double>(ingress_finished-began).count();receipt["existing_learn_seconds"]=learn_seconds;
        receipt["total_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();write_yaml(output/"receipt.yaml",receipt);
        std::cout<<"PASS-full-network-diagnostic-only"<<std::endl;return 0;
      }
      require(cfg["common_mode_health"].as<std::string>()=="report-and-continue","unknown bounded health mode");
    }
    std::vector<RtcEventPeerEligibility> peers;
    for(std::uint32_t d=0;d<channels.size();++d)
      peers.push_back({nw,d,detectors[d].detector_occurrence_id,
                       peer_good[d] && tune.valid[channels[d]]});
    auto peer=RtcEventPeerPopulation::admit(spikes,
        "exact-projected-APT-population:flag=flag2=0:Tune-valid:sha256:"+
        citlali::utils::sha256_file(manifest),std::move(peers));
    auto event_evidence=measure(learn_seconds,[&]{return learn_rtc_event_assessment(spikes,peer,2);});
    auto events=measure(consider_seconds,[&]{return RtcEventAssessmentDecision::consider(event_evidence,val,3);});
    auto amplitude=measure(consider_seconds,[&]{return RtcJumpAmplitudeDecision::consider(events,val,4);});
    auto short_evidence=measure(learn_seconds,[&]{return RtcJumpConsistencyEvidence::learn(amplitude,5);});
    auto consistency=measure(consider_seconds,[&]{return RtcJumpConsistencyDecision::consider(short_evidence,val,6);});
    auto transition_request=measure(consider_seconds,[&]{return RtcJumpTransitionRequest::consider(consistency,val,7);});
    auto transition=measure(learn_seconds,[&]{return RtcJumpTransitionEvidence::learn(transition_request,8);});
    auto support=measure(learn_seconds,[&]{return RtcJumpSupportEvidence::learn(transition,9);});
    auto refit_request=measure(consider_seconds,[&]{return RtcJumpRefitRequest::consider(support,val,10);});
    auto refit=measure(learn_seconds,[&]{return RtcJumpRefitEvidence::learn(refit_request,11);});
    auto remeasure_request=measure(consider_seconds,[&]{return RtcJumpRemeasureRequest::consider(refit,val,12);});
    auto remeasurement=measure(learn_seconds,[&]{return RtcJumpReassessmentEvidence::learn(remeasure_request,13);});
    auto reassessment=measure(consider_seconds,[&]{return RtcJumpReassessmentDecision::consider(remeasurement,val,14);});
    auto admitted=measure(consider_seconds,[&]{return RtcJumpAdmissionDecision::consider(reassessment,val,15);});
    const auto transients_finished=std::chrono::steady_clock::now();
    auto native=ValNativeRealization::create(parent,{ValProducer::align,1},1,ValNativeProductRole::original_input,nw);
    auto identity=RtcSpectralInputIdentity::bind(native,val,view->span(nw),
        RtcSpectralInputStage::original_reference,"exact-simultaneous-audit-original-projection",1);
    const std::vector<RtcSpectralCadenceDomain> cadence{{nw,"audit-four-epoch-ULP-arithmetic-envelope",
        duration,prior["roundoff_bound_fraction"].as<double>()}};
    auto spectral=measure(learn_seconds,[&]{return RtcNativeSpectralEvidence::learn_initial(spikes,{identity},cadence,18);});
    auto lines=measure(learn_seconds,[&]{return RtcLinePowerEvidence::learn(spectral,val,RtcLinePowerProfile::initial_2_hz,19);});
    auto joint=measure(consider_seconds,[&]{return RtcLinePowerConsideration::rank(lines,
        RtcSpectralTransientConsideration::consider(spectral,val,events,val,20),21);});
    const auto learn_finished=std::chrono::steady_clock::now();
    if(health)export_health(output/"health",health,lines,channels,cfg,argv[1],health_seconds);
    // The immutable configuration includes exact content hashes for every
    // scientific input. A changed code/config/VAL requires a new selection.
    const auto binding=citlali::utils::sha256(std::string(CITLALI_GIT_REVISION)+"\n"+
        citlali::utils::sha256_file(argv[1])+"\nVAL-generation=0\n"+mapping->paired_xr_record_id);
    fs::create_directories(output);
    if(recovered_scans) write_yaml(output/"processing-scans.yaml",recovered_scans->receipt);
    std::ofstream native_times(output/"native-time.f64",std::ios::binary);
    for(auto row=axis->first_native_row();row<axis->past_last_native_row();++row){
      double t=axis->native_identity(row).reconstructed_time_unix_sec();
      native_times.write(reinterpret_cast<const char*>(&t),8);
    }
    native_times.close();require(bool(native_times),"native time export failed");
    YAML::Node receipt;
    if(declared_contaminant)receipt["declared_contaminant"]=contaminant_record;
    receipt["schema"]="rtc-multidetector-learning-v1";
    receipt["source_revision"]=std::string(CITLALI_GIT_REVISION);
    receipt["configuration_sha256"]=citlali::utils::sha256_file(argv[1]);
    receipt["learning_binding"]=binding;receipt["VAL_generation"]=0;
    receipt["observation"]=obs;receipt["network"]=nw;receipt["rows"]=rows;
    receipt["source_protection_authority"]=protection_authority;
    receipt["source_protection"]="explicit-empty-protected-region";
    receipt["astronomical_signal_retained"]=true;
    receipt["original_parent"]=mapping->paired_xr_record_id;
    receipt["automatic_spike_admission"]=false;receipt["production_filtering_active"]=false;
    receipt["Apply_performed"]=false;
    receipt["candidate_count"]=spikes->candidates().size();
    receipt["event_count"]=events->evidence_handle()->events().size();
    std::size_t jump_count=0;
    for(const auto &g:admitted->groups())jump_count+=g.admitted();
    receipt["admitted_jump_groups"]=jump_count;
    receipt["existing_scan_binding_required"]=jump_count!=0;
    receipt["event_selection_state"]="unavailable-until-explicit-review";
    receipt["stable_support_state"]="unavailable-until-explicit-review";
    receipt["spectral_estimator"]=std::string(RtcInitialSpectralPolicy::estimator);
    receipt["spectral_conventions"]=std::string(RtcInitialSpectralPolicy::conventions);
    receipt["native_integration_seconds"]=duration;
    receipt["cadence_interval_seconds"]=spectral->network(nw).interval_seconds;
    receipt["physical_runs"]=runs.size();
    receipt["native_time_sha256"]=citlali::utils::sha256_file(output/"native-time.f64");
    for(const auto &run:runs)receipt["native_runs"].push_back(range({run.first_native_row,run.past_last_native_row}));
    for(std::size_t d=0;d<channels.size();++d){
      YAML::Node entry;entry["detector"]=d;entry["channel"]=channels[d];
      entry["occurrence"]=detectors[d].detector_occurrence_id;
      entry["array_association"]=detectors[d].detector_association_record_id;
      entry["samples_sha256"]=sample_hashes[d];entry["peer_eligible"]=peer->eligible(nw,d);
      entry["prior_flxscale"]=factors[d];entry["factor_authority"]="sha256:"+citlali::utils::sha256_file(manifest);
      entry["factor_unit"]="mJy/beam/xs";
      receipt["detectors"].push_back(entry);
    }
    YAML::Node event_table(YAML::NodeType::Sequence);
    for(std::size_t i=0;i<events->evidence_handle()->events().size();++i){
      const auto &e=events->evidence_handle()->events()[i];const auto &review=events->event_reviews()[i];
      const auto &seed=spikes->candidates()[e.seed];YAML::Node entry;
      entry["event"]=i;entry["channel"]=channels[e.detector];entry["detector"]=e.detector;
      entry["seed_earlier_row"]=seed.earlier_row;entry["trial_exclusion"]=range(e.trial_exclusion);
      entry["origin_unix_seconds"]=e.origin;entry["time_scale_seconds"]=e.time_scale;
      entry["review_disposition"]=static_cast<int>(review.disposition);
      entry["health_concern"]=review.health_concern;entry["refinement_limited"]=e.refinement_limited;
      entry["hard_spike_accepted"]=false;entry["jump_admitted"]=admitted->groups().at(i).admitted();
      for(const auto &r:e.neighbor_exclusions)entry["neighbor_exclusions"].push_back(range(r));
      for(std::size_t c=0;c<2;++c){
        const auto &b=e.background[c];const auto &recovery=e.recovery[c];const auto &context=e.peers[c];
        YAML::Node coordinate;coordinate["seeded"]=e.seeded[c];coordinate["background_available"]=b.available();
        coordinate["background_cause"]=static_cast<int>(b.support_cause);
        coordinate["recovery_cause"]=static_cast<int>(recovery.cause);
        coordinate["affected"]=range(recovery.affected);coordinate["confirmation"]=range(recovery.confirmation);
        coordinate["examined"]=range(recovery.examined);coordinate["scale"]=b.cubic.scale;
        for(auto v:b.cubic.coefficients)coordinate["cubic"].push_back(v);
        for(auto v:b.cubic_with_offset.coefficients)coordinate["cubic_with_offset"].push_back(v);
        coordinate["offset"]=b.cubic_with_offset.offset;
        for(const auto &side:b.support){YAML::Node s;s["usable"]=side.usable;
          s["rows"]=range({side.first_used,side.last_used+1});coordinate["fit_support"].push_back(s);}
        coordinate["usable_peers"]=context.usable_peers;
        coordinate["strongest_peer_channel"]=channels.at(context.strongest_peer);
        coordinate["strongest_level_correlation"]=context.strongest_level_correlation;
        coordinate["strongest_difference_correlation"]=context.strongest_difference_correlation;
        entry["coordinates"].push_back(coordinate);
      }
      event_table.push_back(entry);
    }
    write_yaml(output/"events.yaml",event_table);
    receipt["events_sha256"]=citlali::utils::sha256_file(output/"events.yaml");
    std::ofstream psd(output/"original-psd.f64",std::ios::binary);
    for(const auto &s:spectral->spectra()){
      psd.write(reinterpret_cast<const char*>(s.psd.data()),s.psd.size()*8);
      YAML::Node entry;entry["channel"]=channels[s.detector];entry["coordinate"]=static_cast<int>(s.coordinate);
      entry["available"]=s.available();entry["cause"]=static_cast<int>(s.cause);entry["bins"]=s.psd.size();
      for(const auto &window:s.windows)entry["windows"].push_back(range(window.rows));
      receipt["spectra"].push_back(entry);
    }
    psd.close();require(bool(psd),"original spectrum output failed");
    // Preserve the learning product even if subsequent explicit Consider
    // refuses missing required scan/support bindings. This is not Apply success.
    write_yaml(output/"learning-receipt.yaml",receipt);
    if(argc==4 || recovered_scans){
      const auto considered_at=std::chrono::steady_clock::now();
      require(!(argc==4 && recovered_scans),"automatic decisions cannot consume manual selections");
      caller::ReviewedInputs selected;
      if(argc==4){
      const auto selections=YAML::LoadFile(argv[3]);
      selected=caller::read_review(selections,binding,events->evidence_handle(),channels,factors,nw,
                                             "sha256:"+citlali::utils::sha256_file(manifest));
      } else {
        require(cfg["decision_apply"]["policy"].as<std::string>()==
            "existing-authorities-with-unavailable-isolated-admission-v1","unknown decision policy");
        selected.scans=recovered_scans->projection.binding;
      }
      // This call intentionally refuses missing scan support when any jump is
      // admitted. Acquisition ScanNum is never used as the processing scan.
      auto jumps=RtcJumpExclusionPlan::consider(admitted,selected.scans,val,22);
      auto transient=RtcTransientExclusionPlan::consider(events->original_screening_handle(),jumps,val,23);
      std::vector<std::vector<std::shared_ptr<const RtcDonorFillPlan>>> donors(channels.size());
      for(const auto &event:selected.events){
        auto donor=RtcDonorFillPlan::consider(event,selected.facts,transient,val,24+event.event);
        donors[donor->event().detector].push_back(donor);
      }
      std::shared_ptr<const RtcEventTreatmentDecision> decision;
      if(recovered_scans){
        std::vector<RtcDonorDetectorFacts> factor_facts;
        const auto authority="sha256:"+citlali::utils::sha256_file(manifest);
        for(std::uint32_t d=0;d<channels.size();++d){
          RtcDonorDetectorFacts f;f.network=nw;f.detector=d;
          f.detector_occurrence_id=detectors[d].detector_occurrence_id;
          f.factor_identity=authority+":channel="+std::to_string(channels[d]);
          f.factor_convention="mJy/beam/xs";
          if(std::isfinite(factors[d]) && factors[d]!=0)f.prior_flxscale=factors[d];
          f.factor_support={axis->first_native_row(),axis->past_last_native_row()};
          factor_facts.push_back(std::move(f));
        }
        decision=RtcEventTreatmentDecision::consider(events,transient,std::move(factor_facts),authority,
          "mJy/beam/xs",{{nw,recovered_scans->projection.native_outside_processing}},val,2000,declared_contaminant);
        selected.facts=decision->facts_handle();
        YAML::Node decisions(YAML::NodeType::Sequence);
        for(const auto &r:decision->records()){
          YAML::Node n;n["event"]=r.event;n["channel"]=channels[r.detector];n["network"]=r.network;
          n["seed_earlier_row"]=spikes->candidates()[events->evidence_handle()->events()[r.event].seed].earlier_row;
          n["origin_unix_seconds"]=events->evidence_handle()->events()[r.event].origin;
          constexpr std::array names{"isolated-admission-unavailable","admitted-level-shift",
            "no-resolved-excursion","unresolved-extent","accepted-declared-contaminant"};
          n["class"]=names.at(static_cast<std::size_t>(r.disposition));
          n["affected"]=range(r.affected);n["operation_unavailable"]=range(r.operation_unavailable);
          n["seeded_excursion"]=r.seeded_excursion;n["recovered"]=r.recovered;n["background_available"]=r.background_available;
          n["source_outside"]=r.source_outside;n["target_domain_available"]=r.target_domain_available;
          n["cohort_coincident_events"]=r.cohort_coincident_events;
          n["proposed_isolation_prerequisites"]=r.proposed_isolation_prerequisites;
          n["donor_admission_available"]=bool(r.donor);
          if(r.donor){n["donor_cause"]=static_cast<int>(r.donor->cause());
            if(r.donor->cause()==RtcDonorFillCause::ready)donors[r.detector].push_back(r.donor);}
          decisions.push_back(n);
        }
        write_yaml(output/"event-decisions.yaml",decisions);
        YAML::Node domains;
        for(const auto &f:selected.facts->detectors()){
          YAML::Node d;d["channel"]=channels[f.detector];d["factor_identity"]=f.factor_identity;
          for(auto r:f.stable_segments)d["stable_domains"].push_back(range(r));
          for(auto r:f.contaminated)d["donor_contaminated"].push_back(range(r));
          domains["detectors"].push_back(d);
        }
        domains["authority"]=selected.facts->segmentation_authority();
        for(const auto &d:transient->detectors()){
          YAML::Node n;n["channel"]=channels[d.detector];
          for(auto r:d.rows)n["excluded"].push_back(range(r));
          domains["transient_exclusions"].push_back(n);
        }
        write_yaml(output/"support-decisions.yaml",domains);
      }
      receipt["event_selection_state"]=decision ? "runtime-all-event-dispositions;natural-isolated-policy-unavailable" : "explicit-reviewed-selection";
      receipt["stable_support_state"]=decision ? "runtime-original-evidence-boundary-and-support-resolution" : "explicit-reviewed-support";
      receipt["transient_excluded_pair_cells"]=transient->counts().union_pair_cells;
      receipt["Consider_connection_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-considered_at).count();
      const auto root_output=output;
      for(bool continuity : (decision ? std::vector<bool>{false,true} : std::vector<bool>{true})){
      const auto output=decision ? root_output/(continuity ? "donor-continuity" : "exclusion-control") : root_output;
      fs::create_directories(output);
      YAML::Node arm;
      const auto planning_at=std::chrono::steady_clock::now();
      std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> plans;
      for(std::uint32_t d=0;d<channels.size();++d){
        RtcLineTransferSpecification s;s.identity="multidetector-explicit-finite-plan";
        s.lowpass_identity=cfg["lowpass_identity"].as<std::string>();
        s.state_support_identity="two-centered-finite-FIRs;complete-native-footprints;no-padding;ordered-binary64-FMA;fixed-native-phase0";
        s.input_interval_seconds=spectral->network(nw).interval_seconds;s.factor=2;
        s.centered_lowpass=cfg["fir"].as<std::vector<double>>();
        const auto operation=cfg["detectors"][d]["filter"].as<std::string>();
        require(operation=="lowpass" || operation=="w1-t3","unapproved experiment design");
        if(operation=="w1-t3"){
          s.finite_notch_identity=cfg["finite_notch_identity"].as<std::string>();
          s.centered_notch=cfg["finite_notch"].as<std::vector<double>>();
        }
        RtcNotchRecoveryDomain domain;domain.identity="152390-a2000-235arcsec-s-experiment-only";
        domain.detector_array_association=detectors[d].detector_association_record_id;
        domain.motion=motion;domain.array=RtcOpticalArray::a2000;
        domain.speed_ceiling_arcsec_per_sec=235;domain.nominal_interval_seconds=duration;
        const auto speed_comparison=cfg["speed_support_comparison"] ? cfg["speed_support_comparison"].as<std::string>() : "both";
        require(speed_comparison=="control" || speed_comparison=="low-only" || speed_comparison=="high-only" || speed_comparison=="both","unknown bounded speed-support comparison");
        domain.speed_support = speed_comparison=="control" ? RtcSpeedSupportTreatment::comparison_reject_both :
            speed_comparison=="low-only" ? RtcSpeedSupportTreatment::comparison_low_only :
            speed_comparison=="high-only" ? RtcSpeedSupportTreatment::comparison_high_only :
            RtcSpeedSupportTreatment::original_paired_measurements;
        arm["speed_support_comparison"]=speed_comparison;
        arm["speed_support_authority"]="rtc-fixed-filter-original-speed-support-2026-09-16";
        auto candidate=RtcLineTransferCandidate::bind(lines,nw,d,s);
        auto assessment=RtcLineTransferAssessment::consider(candidate,joint,val,100+d);
        plans.push_back(RtcNotchRecoveryPlan::consider(assessment,transient,val,domain,200+d,continuity ? donors[d] : std::vector<std::shared_ptr<const RtcDonorFillPlan>>{},decision,continuity));
      }
      const auto complete=RtcPipelinePlan::consider(plans,joint->joint_handle(),1000+continuity);
      arm["Consider_plan_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-planning_at).count();
      const std::array partitions{view};
      const auto applied_at=std::chrono::steady_clock::now();
      auto result=RtcPipelineResult::apply(complete,view,val,partitions);
      arm["Apply_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-applied_at).count();
      receipt["Apply_performed"]=true;
      if(cfg["prepare_output_grid"] && cfg["prepare_output_grid"].as<bool>()) {
        const auto start=std::chrono::steady_clock::now();
        auto ast_views=AstScanMotionNetworkViews::admit(parent->scope(),
            motion->raw_product_handle(),{axis->native_timing_handle()});
        auto align=IdentityRouteAlignContext::admit(parent,ast_views,val);
        auto grid=RtcOutputGrid::prepare(result,align);
        YAML::Node record;
        record["state"]="prepared-RTC-occurrences-only";
        record["terminal_publication_performed"]=false;
        record["CAL_admission_performed"]=false;
        record["AST_detector_coordinate_parent"]="unavailable-not-bound-by-this-diagnostic";
        record["output_VAL_target_binding"]="exact-RTC-grid-slot-and-coordinate";
        record["output_VAL_publication"]="not-performed-before-complete-handoff";
        record["descriptor_bytes"]=grid->owned_descriptor_bytes();
        for(std::size_t d=0;d<grid->detectors().size();++d){
          YAML::Node column;const auto &g=grid->detectors()[d];
          column["channel"]=channels[d];column["scheduled_slots"]=g.scheduled_count;
          std::size_t x_available=0,r_available=0,replaced=0,influence=0;
          for(std::size_t slot=0;slot<g.scheduled_count;++slot){
            auto fact=grid->occurrence(d,slot);
            require(val->contains(RtcOutputGrid::val_target(grid,d,slot,NativeReadoutCoordinate::x)) &&
                val->contains(RtcOutputGrid::val_target(grid,d,slot,NativeReadoutCoordinate::r)),
                "prepared output does not bind the exact VAL parent");
            x_available+=fact.x_available;r_available+=fact.r_available;
            replaced+=fact.representative_replaced;influence+=fact.replacement_influence;
            require(fact.representative.network_occurrence.native_row()==g.first+static_cast<TimestreamNativeRow>(slot*g.factor),
                "prepared output occurrence changed native phase");
          }
          column["x_available"]=x_available;column["r_available"]=r_available;
          column["representative_replaced"]=replaced;column["replacement_influence"]=influence;
          record["detectors"].push_back(column);
        }
        record["prepare_and_inspect_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        arm["output_grid"]=record;
      }
      // Diagnostic overlays bind to this already frozen complete plan. They
      // never enter Learn/Consider or modify its masks/coefficients/decisions.
      for(const auto &probe:cfg["fixed_plan_injections"]){
        const auto id=probe["identity"].as<std::string>();
        require(!id.empty() && id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")==std::string::npos,"invalid fixed-plan injection identity");
        require(probe["detectors"].size()==channels.size(),"injection must bind complete paired cohort");
        std::vector<RtcRecoveryInjection> injections;
        for(std::size_t d=0;d<channels.size();++d){
          const auto bound=probe["detectors"][d];
          require(bound["channel"].as<int>()==channels[d],"injection detector order differs");
          const auto path=checked_file(bound);
          require(fs::file_size(path)==std::uintmax_t(rows)*16,"injection paired shape differs");
          RtcRecoveryInjection injection{plans[d],id,Eigen::Matrix<double,Eigen::Dynamic,2>(rows,2)};
          std::ifstream stream(path,std::ios::binary);
          for(Eigen::Index i=0;i<rows;++i)for(int c=0;c<2;++c)stream.read(reinterpret_cast<char*>(&injection.delta(i,c)),8);
          require(bool(stream)&&injection.delta.allFinite(),"invalid paired injection");
          injections.push_back(std::move(injection));
        }
        const auto injected=RtcPipelineResult::apply(complete,view,val,partitions,injections);
        const auto directory=output/("injection-"+id);fs::create_directories(directory);
        for(std::size_t d=0;d<channels.size();++d){
          const auto &a=*injected->detector_results()[d];const auto &b=*result->detector_results()[d];
          require(a.output_native_rows()==b.output_native_rows()&&a.causes()==b.causes(),"paired injection changed frozen support");
          const auto difference=(a.filtered_native_pair()-b.filtered_native_pair()).eval();
          write_matrix(directory/(std::to_string(channels[d])+"-delta.f64"),difference);
        }
        arm["fixed_plan_injections"].push_back(probe);
      }
      if(argc==4)receipt["review_sha256"]=citlali::utils::sha256_file(argv[3]);
      for(std::size_t d=0;d<channels.size();++d){
        const auto &r=*result->detector_results()[d];const auto stem=std::to_string(channels[d]);
        write_matrix(output/(stem+"-conditioned.f64"),r.conditioned_native_pair());
        write_matrix(output/(stem+"-filtered.f64"),r.filtered_native_pair());
        std::ofstream states(output/(stem+"-state.u8"),std::ios::binary);
        std::size_t replaced=0,excluded=0;
        for(auto row=axis->first_native_row();row<axis->past_last_native_row();++row){
          std::uint8_t b=0;
          if(r.coordinate_stage_available(NativeReadoutCoordinate::x,row,true))b|=1;
          if(r.coordinate_stage_available(NativeReadoutCoordinate::r,row,true))b|=2;
          if(r.representative_replaced(row)){b|=4;++replaced;}
          if(r.requires_representative_exclusion(row)){b|=8;++excluded;}
          if(r.replacement_influence(row,true))b|=16;
          if(r.unrepaired_influence(row,true))b|=32;
          states.write(reinterpret_cast<const char*>(&b),1);
        }
        states.close();require(bool(states),"state output failed");
        std::ofstream center(output/(stem+"-map-center.u8"),std::ios::binary);
        for(auto row=axis->first_native_row();row<axis->past_last_native_row();++row){
          const std::uint8_t admitted=r.map_center_admitted(row);center.write(reinterpret_cast<const char*>(&admitted),1);
        }
        center.close();require(bool(center),"center disposition output failed");
        for(const auto &name:{std::string("support-causes"),std::string("speed-evidence")}){
          std::ofstream stream(output/(stem+"-"+name+".u8"),std::ios::binary);
          for(std::size_t i=0;i<plans[d]->input_causes().size();++i){
            const std::uint8_t value=name=="support-causes" ? static_cast<std::uint8_t>(plans[d]->support_causes()[i]) : static_cast<std::uint8_t>(plans[d]->speed_restrictions()[i]);
            stream.write(reinterpret_cast<const char*>(&value),1);
          }
          stream.close();require(bool(stream),"support/speed evidence output failed");
        }
        std::ofstream causes(output/(stem+"-causes.u8"),std::ios::binary);
        for(auto cause:r.causes()){const auto c=static_cast<std::uint8_t>(cause);causes.write(reinterpret_cast<const char*>(&c),1);}
        causes.close();require(bool(causes),"cause output failed");
        std::ofstream selected_rows(output/(stem+"-rows.i64"),std::ios::binary);
        for(auto row:r.output_native_rows())selected_rows.write(reinterpret_cast<const char*>(&row),8);
        selected_rows.close();require(bool(selected_rows),"selected native row output failed");
        YAML::Node record;record["channel"]=channels[d];record["replaced_rows"]=replaced;
        record["representative_excluded_rows"]=excluded;
        record["output_rows"]=r.output_native_rows().size();record["factor"]=2;record["phase_native_rows"]=0;
        std::size_t center_count=0;for(auto row:r.output_native_rows())center_count+=r.map_center_admitted(row);
        record["direct_map_center_admitted_rows"]=center_count;
        record["map_route_authorized"]=false;
        const auto &spec=plans[d]->assessment_handle()->candidate_handle()->specification();
        record["lowpass_FIR"]=spec.centered_lowpass;record["finite_notch_FIR"]=spec.centered_notch;
        record["sampling_speed_limit_arcsec_per_sec"]=plans[d]->sampling_speed_limit_arcsec_per_sec();
        record["AST_source"]=motion->raw_product_handle()->source_handle()->metadata().source_artifact_identity;
        for(const auto &run:plans[d]->runs())record["admitted_runs"].push_back(range(run));
        for(const auto &donor:donors[d]){YAML::Node dr;dr["event"]=donor->selection().event;
          dr["included_in_arm"]=continuity;dr["cause"]=static_cast<int>(donor->cause());dr["affected"]=range(donor->selection().affected);
          for(const auto &m:donor->medians()){YAML::Node cell;cell["native_row"]=m.row;cell["value"]=m.value;
            for(auto id:m.eligible)cell["eligible_channels"].push_back(channels[id]);
            for(std::size_t i=0;i<m.central_count;++i)cell["central_channels"].push_back(channels[m.central[i]]);
            dr["median_population"].push_back(cell);}
          record["donors"].push_back(dr);}
        std::ofstream inputs(output/(stem+"-input-causes.u8"),std::ios::binary);
        for(auto cause:plans[d]->input_causes()){const auto c=static_cast<std::uint8_t>(cause);inputs.write(reinterpret_cast<const char*>(&c),1);}
        inputs.close();require(bool(inputs),"input cause output failed");
        arm["realized_detectors"].push_back(record);
      }
      std::vector<std::shared_ptr<const RtcConsequenceEvidence>> consequences;
      if(cfg["consequence_study"]) {
        consequences=run_consequence_study(cfg["consequence_study"],output/"purpose-study",channels,result);
        arm["purpose_consequence_evidence_products"]=consequences.size();
      }
      const auto relearn_at=std::chrono::steady_clock::now();
      for(bool lowpass:{false,true}){
        auto conditioned=RtcNativeSpectralEvidence::learn_conditioned(result->native_product(lowpass,val),val,cadence,1100+lowpass);
        auto considered=RtcSpectralTransientConsideration::consider(conditioned,val,events,val,1200+lowpass);
        const auto outcome_started=std::chrono::steady_clock::now();
        const auto outcome=RtcTreatmentOutcomeEvidence::learn(spectral,conditioned,1250+lowpass);
        const auto outcome_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-outcome_started).count();
        const auto reassessment=RtcPipelineReassessment::consider(result,considered,(consequences.empty()?1300:9000)+lowpass,outcome,
            lowpass?consequences:std::vector<std::shared_ptr<const RtcConsequenceEvidence>>{});
        YAML::Node stage;stage["after_lowpass"]=lowpass;stage["producer_attempt"]=complete->attempt();
        const auto stage_path=output/(lowpass ? "post-lowpass-psd.f64" : "post-notch-psd.f64");
        std::ofstream stage_psd(stage_path,std::ios::binary);
        stage["classification_authorized"]=reassessment->classification_authorized;
        const auto outcome_name=lowpass ? "post-lowpass-outcome" : "post-notch-outcome";
        write_treatment_outcome(output,outcome_name,*reassessment->outcome_handle(),channels);
        stage["matched_outcome_file"]=std::string(outcome_name)+".yaml";
        stage["matched_outcome_seconds"]=outcome_seconds;
        for(const auto &s:conditioned->spectra()){YAML::Node c;c["channel"]=channels[s.detector];
          c["coordinate"]=static_cast<int>(s.coordinate);c["available"]=s.available();c["cause"]=static_cast<int>(s.cause);
          c["bins"]=s.psd.size();stage_psd.write(reinterpret_cast<const char*>(s.psd.data()),s.psd.size()*8);
          for(const auto &w:s.windows){YAML::Node win;win["rows"]=range(w.rows);
            win["representative_replacements"]=w.representative_replacements;
            win["replacement_influenced_samples"]=w.replacement_influenced_samples;
            win["unrepaired_influenced_samples"]=w.unrepaired_influenced_samples;
            win["representative_exclusions"]=w.representative_exclusions;
            c["windows"].push_back(win);}
          stage["spectra"].push_back(c);}
        stage_psd.close();require(bool(stage_psd),"conditioned spectrum output failed");
        stage["numerical_psd_sha256"]=citlali::utils::sha256_file(stage_path);
        arm["relearned"].push_back(stage);
        if(lowpass) {
          // Explicit offline owner selection. No config entry means unavailable,
          // never an implicit retain based on a scalar residual or empty list.
          const auto decision_started=std::chrono::steady_clock::now();
          std::optional<RtcPipelineSelection> selection;
          if(const auto supplied=cfg["reassessment_selection"]) {
            const auto intent=supplied["intent"].as<std::string>();
            require(intent=="retain-development-candidate" || intent=="require-scientific-qualification",
                    "real-data replay does not authorize an automatic revision");
            selection=RtcPipelineSelection{reassessment,
                intent=="retain-development-candidate" ? RtcPipelineSelectionIntent::retain_development_candidate :
                                                        RtcPipelineSelectionIntent::require_scientific_qualification,
                supplied["authority"].as<std::string>(),supplied["purpose"].as<std::string>(),
                supplied["positive_rationale"].as<std::string>(),{},0};
          }
          const auto disposition=RtcPipelineDecision::consider(reassessment,val,selection,consequences.empty()?1400:9100);
          const auto decision_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-decision_started).count();
          const auto advance_started=std::chrono::steady_clock::now();
          const auto advanced=RtcPipelineResult::advance(result,disposition,view,val,partitions);
          const auto advance_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-advance_started).count();
          result=advanced.candidate;
          receipt["RTC_scientific_qualification"]="unresolved";
          std::cout<<"RTC reassessment plan="<<complete->attempt()<<" disposition="
                   <<rtc_pipeline_disposition_name(disposition->disposition())
                   <<" qualification=unresolved revision_executed="<<advanced.revision_executed<<'\n';
          arm["reassessment"]=write_reassessment_decision(output,advanced);
          arm["reassessment"]["Consider_decision_seconds"]=decision_seconds;
          arm["reassessment"]["advance_seconds"]=advance_seconds;
          require(!advanced.revision_executed && result->plan_handle().get()==complete.get(),
                  "bounded real-data disposition must preserve the frozen baseline");
        }
      }
      arm["conditioned_relearning_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-relearn_at).count();
      arm["attempt"]=result->plan_handle()->attempt();arm["original_parent"]=mapping->paired_xr_record_id;
      write_yaml(output/"apply-receipt.yaml",arm);
      receipt[continuity ? "donor_continuity" : "exclusion_control"]=arm;
      }
    }
    const auto &net=parent->network(nw);
    for(std::uint32_t d=0;d<channels.size();++d)for(std::int64_t row=0;row<rows;++row){
      require(std::bit_cast<std::uint64_t>(original_x(row,d))==std::bit_cast<std::uint64_t>(net.value(NativeReadoutCoordinate::x,row,d)) &&
          std::bit_cast<std::uint64_t>(original_r(row,d))==std::bit_cast<std::uint64_t>(net.value(NativeReadoutCoordinate::r,row,d)),"original pair changed");
    }
    receipt["original_pair_unchanged"]=true;receipt["admitted_parent_unchanged"]=true;
    receipt["ingress_seconds"]=std::chrono::duration<double>(ingress_finished-began).count();
    receipt["Learn_seconds"]=learn_seconds;receipt["Consider_evidence_seconds"]=consider_seconds;
    receipt["transient_seconds"]=std::chrono::duration<double>(transients_finished-ingress_finished).count();
    receipt["spectral_seconds"]=std::chrono::duration<double>(learn_finished-transients_finished).count();
    receipt["total_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    receipt["status"]=recovered_scans ? "PASS-runtime-decision-Apply;natural-isolated-policy-unavailable" : argc==4 ? "PASS-explicit-reviewed-Apply" : "PASS-Learn-awaiting-explicit-selections";
    write_yaml(output/"receipt.yaml",receipt);
    std::cout<<receipt["status"].as<std::string>()<<" detectors="<<channels.size()<<" events="<<event_table.size()<<" jumps="<<jump_count<<'\n';
    return 0;
  } catch(const std::exception &e){std::cerr<<"RTC multi-detector caller FAIL: "<<e.what()<<'\n';return 1;}
}

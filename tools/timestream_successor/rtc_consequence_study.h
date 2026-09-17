#pragma once
// Offline file/experiment adapter. Frozen Apply and consequence Learn retain
// numerical ownership; this adapter selects no scientific acceptance policy.
namespace {
Eigen::Matrix<double,Eigen::Dynamic,2> consequence_matrix(const YAML::Node &binding,std::int64_t rows) {
  const auto path=checked_file(binding);require(fs::file_size(path)==std::uintmax_t(rows)*16,"consequence paired shape");
  Eigen::Matrix<double,Eigen::Dynamic,2> m(rows,2);std::ifstream in(path,std::ios::binary);
  for(Eigen::Index i=0;i<rows;++i)for(int c=0;c<2;++c)in.read(reinterpret_cast<char*>(&m(i,c)),8);
  require(bool(in)&&m.allFinite(),"consequence input nonfinite or truncated");return m;
}
YAML::Node consequence_metrics(const RtcConsequenceMetrics &m) {
  YAML::Node n;
  auto put=[&](const char *name,double v){if(std::isfinite(v))n[name]=v;else n[name]=YAML::Null;};
  put("projection",m.projection);put("peak_ratio",m.peak_ratio);put("waveform_error",m.waveform_error);
  put("centroid_x_arcsec",m.centroid_x_arcsec);put("centroid_y_arcsec",m.centroid_y_arcsec);
  n["centroid_available"]=std::isfinite(m.centroid_x_arcsec)&&std::isfinite(m.centroid_y_arcsec);
  if(!n["centroid_available"].as<bool>())n["centroid_unavailable"]="crossing/source unsupported or nonpositive response sum";
  put("negative_ringing_fraction",m.negative_ringing_fraction);return n;
}
YAML::Node consequence_record(const RtcConsequenceEvidence &e,const std::vector<int> &channels) {
  YAML::Node n;n["domain"]=e.domain().identity;n["source_model"]=e.domain().source_model;
  n["regime"]=e.domain().regime;n["geometry_identity"]=e.domain().geometry_identity;n["units"]=e.domain().units;
  n["injected_identity"]=e.domain().injected_identity;n["line_free_identity"]=e.domain().line_free_identity;
  n["stage"]="scheduled-final-output-F2-phase0";n["estimator"]="unrenormalized-template-projection-and-sampled-centroid;diagnostic-surrogate";
  n["plan_attempt"]=e.baseline_handle()->plan_handle()->attempt();n["comparison_plan_attempt"]=e.peer_handle()->plan_handle()->attempt();
  n["VAL_generation"]=e.baseline_handle()->plan_handle()->snapshot_handle()->generation().value;
  n["attempt"]=e.attempt();n["acceptance_requirement_selected"]=false;n["scientific_qualification"]="unresolved";
  for(auto p:e.domain().purposes)n["purposes"].push_back(std::string(citlali::config::to_string(p)));
  for(const auto &q:e.domain().required_unavailable)n["required_unavailable"].push_back(q);
  for(const auto &r:e.records()){
    YAML::Node d;d["channel"]=channels[r.detector];d["network"]=r.network;d["available"]=r.available;d["unavailable"]=r.unavailable;
    d["source_energy"]=r.source_energy;if(std::isfinite(r.added_line_rms))d["added_line_rms_native_x"]=r.added_line_rms;
    d["own_total"]=r.own_total;d["peer_total"]=r.peer_total;d["common_total"]=r.common_total;d["expected"]=r.expected;
    d["rows"]=r.rows;d["ringing_rows"]=r.ringing_rows;d["measured"]=consequence_metrics(r.measured);
    if(r.line_free){d["line_free"]=consequence_metrics(*r.line_free);
      if(r.available)d["additional_projection_error"]=r.measured.projection-r.line_free->projection;}
    n["records"].push_back(d);
  }
  return n;
}
std::vector<std::shared_ptr<const RtcConsequenceEvidence>> run_consequence_study(
    const YAML::Node &binding,const fs::path &output,const std::vector<int> &channels,
    const std::shared_ptr<const RtcPipelineResult> &current) {
  const auto started=std::chrono::steady_clock::now();const auto path=checked_file(binding);const auto cfg=YAML::LoadFile(path.string());
  require(cfg["schema"].as<std::string>()=="rtc-purpose-consequence-study-v1","unknown consequence study");
  const auto original=current->plan_handle()->input_handle();const auto val=current->plan_handle()->snapshot_handle();
  const std::array parts{original};const auto &plans=current->plan_handle()->detector_plans();
  std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> lowplans;
  for(std::size_t d=0;d<plans.size();++d){
    const auto &p=*plans[d];const auto &old=*p.assessment_handle()->candidate_handle();auto spec=old.specification();
    spec.identity+=";purpose-study-lowpass-only";spec.centered_notch.clear();spec.finite_notch_identity.clear();
    auto c=RtcLineTransferCandidate::bind(old.line_handle(),old.network(),old.detector(),spec);
    auto a=RtcLineTransferAssessment::consider(c,p.assessment_handle()->joint_handle(),val,3000+d);
    lowplans.push_back(RtcNotchRecoveryPlan::consider(a,p.transient_handle(),val,p.domain(),3100+d,p.donor_plans(),p.event_decisions(),p.donor_continuity()));
  }
  auto lowplan=RtcPipelinePlan::consider(lowplans,current->plan_handle()->original_consideration(),3200+current->plan_handle()->attempt());
  auto low=RtcPipelineResult::apply(lowplan,original,val,parts);
  fs::create_directories(output);YAML::Node report;
  report["study_sha256"]=citlali::utils::sha256_file(path);report["original_unchanged"]=true;
  report["lowpass_plan_attempt"]=lowplan->attempt();report["baseline_plan_attempt"]=current->plan_handle()->attempt();
  report["scientific_qualification"]="unresolved";report["missing_acceptance_requirement"]="purpose-specific residual-contamination and source-error limits unselected";
  const auto n=current->detector_results()[0]->filtered_native_pair().rows();
  std::vector<RtcConsequenceEvidence::Positions> xy;
  for(std::size_t d=0;d<channels.size();++d){
    require(cfg["positions"][d]["channel"].as<int>()==channels[d],"consequence geometry detector order");
    xy.push_back(consequence_matrix(cfg["positions"][d],n));
    const auto &r=*low->detector_results()[d];const auto stem=std::to_string(channels[d]);
    write_matrix(output/(stem+"-lowpass.f64"),r.filtered_native_pair());
    std::ofstream mask(output/(stem+"-lowpass-centers.u8"),std::ios::binary);
    for(auto row=r.plan_handle()->first_native_row();row<r.plan_handle()->first_native_row()+n;++row){
      const std::uint8_t admitted=r.map_center_admitted(row);mask.write(reinterpret_cast<const char*>(&admitted),1);}
    mask.close();require(bool(mask),"consequence support export failed");
    for(bool selected:{false,true}){
      const auto &v=*(selected?current:low)->detector_results()[d];
      std::ofstream eligibility(output/(stem+(selected?"-selected-review.u8":"-lowpass-review.u8")),std::ios::binary);
      for(auto row=v.plan_handle()->first_native_row();row<v.plan_handle()->first_native_row()+n;++row){
        const std::uint8_t good=v.spectral_review_admitted(row,true);eligibility.write(reinterpret_cast<const char*>(&good),1);}
      eligibility.close();require(bool(eligibility),"consequence spectral support output failed");
    }
  }
  std::vector<std::shared_ptr<const RtcConsequenceEvidence>> evidence;
  std::uint64_t attempt=5000;
  for(const auto &case_node:cfg["cases"]){
    const auto id=case_node["identity"].as<std::string>();
    require(id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")==std::string::npos,"unsafe consequence identity");
    const bool controlled=case_node["controlled"].as<bool>();
    for(bool candidate:{false,true}) {
      const auto baseline=candidate?current:low;const auto peer=candidate?low:current;
      RtcConsequenceDomain domain;domain.identity=id;domain.source_model=case_node["source_model"].as<std::string>();
      domain.regime=case_node["regime"].as<std::string>();domain.geometry_identity=cfg["geometry_identity"].as<std::string>();
      domain.units="original-native-x fixture units;fractional projection;sky tangent arcsec;no calibration qualification";
      domain.injected_identity=id+":study-sha256:"+citlali::utils::sha256_file(path);
      domain.line_free_identity=controlled?domain.injected_identity+":line-free":"";
      for(const auto &p:case_node["purposes"]){const auto parsed=citlali::config::parse_reduction_type(p.as<std::string>());require(bool(parsed),"unknown purpose");domain.purposes.push_back(*parsed);}
      domain.required_unavailable=case_node["required_unavailable"].as<std::vector<std::string>>();
      std::vector<RtcRecoveryInjection> sources,present,absent;
      for(std::size_t d=0;d<channels.size();++d) {
        const auto entry=case_node["detectors"][d];require(entry["channel"].as<int>()==channels[d],"consequence detector order");
        auto s=consequence_matrix(entry["source"],n);auto a=s;
        if(controlled)a+=consequence_matrix(entry["noise"],n);
        auto y=a;if(controlled)y+=consequence_matrix(entry["line"],n);
        const auto plan=baseline->plan_handle()->detector_plans()[d];
        sources.push_back({plan,"source-sha256:"+entry["source"]["sha256"].as<std::string>(),std::move(s)});
        present.push_back({plan,domain.injected_identity,std::move(y)});
        if(controlled)absent.push_back({plan,domain.line_free_identity,std::move(a)});
        domain.window_unavailable.push_back(entry["window_unavailable"].as<std::string>());
        const auto win=entry["window"],ring=entry["ringing_window"];
        domain.windows.push_back({win[0].as<std::int64_t>(),win[1].as<std::int64_t>()});
        domain.ringing_windows.push_back({ring[0].as<std::int64_t>(),ring[1].as<std::int64_t>()});
      }
      const auto at=std::chrono::steady_clock::now();
      const auto y=RtcPipelineResult::apply(baseline->plan_handle(),original,val,parts,present);
      const auto absent_result=controlled?RtcPipelineResult::apply(baseline->plan_handle(),original,val,parts,absent):nullptr;
      const auto applied=std::chrono::steady_clock::now();
      auto measured=RtcConsequenceEvidence::learn(baseline,peer,y,absent_result,sources,xy,domain,val,attempt++);
      auto node=consequence_record(*measured,channels);node["treatment"]=candidate?"selected-notch-lowpass":"lowpass-only";
      node["Apply_seconds"]=std::chrono::duration<double>(applied-at).count();
      node["Learn_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-applied).count();
      report["cases"].push_back(node);evidence.push_back(std::move(measured));
      // Few diagnostic traces, retaining final schedule, both coordinates and
      // immutable overlay identities. They never become new original inputs.
      if(!controlled)for(std::size_t d=0;d<channels.size();++d){
        const auto delta=(y->detector_results()[d]->filtered_native_pair()-baseline->detector_results()[d]->filtered_native_pair()).eval();
        write_matrix(output/(id+"-"+(candidate?"selected-":"lowpass-")+std::to_string(channels[d])+".f64"),delta);
      }
    }
    std::cout<<"RTC purpose consequence case="<<id<<" measured; qualification=unresolved\n";
  }
  report["total_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  write_yaml(output/"consequences.yaml",report);return evidence;
}
}

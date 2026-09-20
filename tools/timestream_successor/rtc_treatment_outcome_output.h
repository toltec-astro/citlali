#pragma once
// Offline serialization only. Numerical comparison and support belong to RTC.
void write_treatment_outcome(const fs::path &output, const std::string &name,
    const citlali::pipeline::RtcTreatmentOutcomeEvidence &e, const std::vector<int> &channels) {
  YAML::Node out;out["schema"]="rtc-treatment-outcome-v1";out["use_policy"]=e.use_policy;
  out["attempt"]=e.attempt();out["original_evidence_attempt"]=e.original_handle()->attempt();
  out["conditioned_evidence_attempt"]=e.conditioned_handle()->attempt();
  out["producer_attempt"]=e.conditioned_handle()->conditioned_handle()->producer_attempt();
  out["classification_authorized"]=e.classification_authorized;
  out["stopping_rule_selected"]=e.stopping_rule_selected;
  out["independent_noise_estimate"]=e.independent_noise_estimate;
  out["uncertainty"]= "unavailable;shared-data treatment outcome";
  out["power_convention"]="sum(stored_D2_PSD*df);includes_DC;no_background_subtraction";
  out["logical_owned_bytes"]=e.logical_owned_bytes();out["peak_scratch_samples"]=e.peak_scratch_samples();
  const auto before_path=output/(name+"-original-psd.f64"),after_path=output/(name+"-conditioned-psd.f64");
  std::ofstream before(before_path,std::ios::binary),after(after_path,std::ios::binary);
  std::ofstream document(output/(name+".yaml"));
  rtc_yaml_node(document,out);document<<"records:\n"<<std::setprecision(17);
  for(const auto &r:e.records()) {
    const auto &n=e.original_handle()->network(r.network),&m=e.conditioned_handle()->network(r.network);
    YAML::Node row;row["channel"]=channels.at(r.detector);row["network"]=r.network;row["coordinate"]=static_cast<int>(r.coordinate);
    row["available"]=r.available();row["cause"]=static_cast<int>(r.cause);
    row["original_VAL_generation"]=n.input->snapshot_handle()->generation().value;
    row["conditioned_VAL_generation"]=m.input->snapshot_handle()->generation().value;
    row["original_processing_stage"]=n.input->processing_stage();row["conditioned_processing_stage"]=m.input->processing_stage();
    row["original_eligible_samples"]=r.original_eligible_samples;
    row["conditioned_eligible_samples"]=r.conditioned_eligible_samples;
    row["common_eligible_samples"]=r.common_eligible_samples;row["window_union_samples"]=r.window_union_samples;
    row["window_union_seconds"]=r.window_union_seconds;
    row["original_cause"]=static_cast<int>(e.original_handle()->spectrum(r.network,r.detector,r.coordinate).cause);
    row["conditioned_cause"]=static_cast<int>(e.conditioned_handle()->spectrum(r.network,r.detector,r.coordinate).cause);
    row["matched_original_cause"]=static_cast<int>(r.original_matched.cause);
    row["matched_conditioned_cause"]=static_cast<int>(r.conditioned_matched.cause);
    for(auto s:r.common_support)row["common_support"].push_back(range(s));
    for(auto s:r.window_union)row["window_union"].push_back(range(s));
    row["original_population_median"]=r.original_matched.population_median;
    row["conditioned_population_median"]=r.conditioned_matched.population_median;
    row["frequency_hz"]=n.frequency_hz;
    row["original_bins"]=r.original_matched.psd.size();row["conditioned_bins"]=r.conditioned_matched.psd.size();
    before.write(reinterpret_cast<const char*>(r.original_matched.psd.data()),r.original_matched.psd.size()*8);
    after.write(reinterpret_cast<const char*>(r.conditioned_matched.psd.data()),r.conditioned_matched.psd.size()*8);
    if(r.available()) {
      row["original_stored_power"]=r.power.original;row["conditioned_stored_power"]=r.power.conditioned;
      if(r.power.conditioned_over_original)row["conditioned_over_original"]=*r.power.conditioned_over_original;
    }
    document<<"-\n";rtc_yaml_node(document,row,2);
    if(r.available() && !r.original_matched.windows.empty()) {
      document<<"  windows:\n";
      for(std::size_t i=0;i<r.original_matched.windows.size();++i) {
        const auto &a=r.original_matched.windows[i],&b=r.conditioned_matched.windows[i];
        document<<"    - {rows: ["<<a.rows.first<<", "<<a.rows.past_last<<"], run_index: "<<a.run_index
          <<", begin_unix_sec: "<<a.support_begin_unix_sec<<", end_unix_sec: "<<a.support_end_unix_sec
          <<", padded_samples: "<<a.padded_samples<<", original_chunk_median: "<<a.centered_chunk_median
          <<", conditioned_chunk_median: "<<b.centered_chunk_median
          <<", source_counts_outside_protected_unknown: ["<<a.source_counts[0]<<", "<<a.source_counts[1]<<", "<<a.source_counts[2]
          <<"], representative_replacements: "<<b.representative_replacements
          <<", replacement_influenced_samples: "<<b.replacement_influenced_samples
          <<", unrepaired_influenced_samples: "<<b.unrepaired_influenced_samples
          <<", representative_exclusions: "<<b.representative_exclusions<<"}\n";
      }
    }
  }
  before.close();after.close();require(bool(before)&&bool(after),"outcome spectrum output failed");
  YAML::Node hashes;hashes["original_psd_sha256"]=citlali::utils::sha256_file(before_path);
  hashes["conditioned_psd_sha256"]=citlali::utils::sha256_file(after_path);
  rtc_yaml_node(document,hashes);document.close();require(bool(document),"outcome metadata output failed");
}

YAML::Node write_reassessment_decision(const fs::path &output,
    const citlali::pipeline::RtcPipelineAdvanceResult &step) {
  using namespace citlali::pipeline;
  const auto &d=*step.decision;const auto &e=*d.reassessment_handle();
  YAML::Node out;out["schema"]="rtc-reassessment-execution-decision-v1";
  out["execution_disposition"]=rtc_pipeline_disposition_name(d.disposition());
  out["reason"]=rtc_pipeline_decision_cause_name(d.cause());
  out["decision_attempt"]=d.attempt();out["reassessment_attempt"]=e.attempt();
  out["evaluated_plan_attempt"]=e.previous_handle()->plan_handle()->attempt();
  out["selected_candidate_attempt"]=step.candidate->plan_handle()->attempt();
  out["scientific_qualification"]="unresolved";
  out["missing_qualification"]=d.missing_qualification;
  out["scientifically_qualified"]=d.scientifically_qualified;
  out["downstream_admission_authorized"]=d.downstream_admission_authorized;
  out["production_authorized"]=d.production_authorized;out["stopping_rule_selected"]=d.stopping_rule_selected;
  out["revision_executed"]=step.revision_executed;out["maximum_revisions"]=1;
  out["VAL_generation"]=d.snapshot_handle()->generation().value;
  out["original_parent_preserved"]=step.candidate->plan_handle()->input_handle().get()==e.previous_handle()->plan_handle()->input_handle().get();
  if(d.selection()) {
    out["authority"]=d.selection()->authority;out["purpose"]=d.selection()->purpose;
    out["positive_rationale"]=d.selection()->positive_rationale;
    out["outcome_requirement"]=d.selection()->outcome_requirement==RtcOutcomeRequirement::every_coordinate ?
        "every-coordinate" : "rtc-available-detector-completion-2026-09-19-v2";
    out["completion_policy_authority"]="doc/SCI_RTC_PTC_PARTIAL_COMPLETION_OWNER_BINDING_2026-09-18.md";
  }
  if(e.outcome_handle()) {
    out["outcome_attempt"]=e.outcome_handle()->attempt();
    out["coordinate_outcomes"]=e.outcome_handle()->records().size();
    std::size_t available=0;for(const auto &r:e.outcome_handle()->records())available+=r.available();
    out["available_coordinate_outcomes"]=available;
    out["outcome_stage"]=e.outcome_handle()->conditioned_handle()->conditioned_handle()->after_lowpass() ?
        "post-lowpass-native-before-decimation" : "post-notch-native";
  }
  for(const auto &c:e.consequence_handles()) {
    YAML::Node n;n["domain"]=c->domain().identity;n["source_model"]=c->domain().source_model;
    n["measured_plan_attempt"]=c->baseline_handle()->plan_handle()->attempt();
    n["stage"]="scheduled-final-output-F2-phase0";
    std::size_t available=0,centroids=0;for(const auto &r:c->records()){available+=r.available;centroids+=std::isfinite(r.measured.centroid_x_arcsec)&&std::isfinite(r.measured.centroid_y_arcsec);}
    n["projection_available_detectors"]=available;n["projection_unavailable_detectors"]=c->records().size()-available;
    n["centroid_available_detectors"]=centroids;
    n["required_unavailable"]=c->domain().required_unavailable;
    for(auto p:c->domain().purposes)n["purposes"].push_back(std::string(citlali::config::to_string(p)));
    n["missing_acceptance_requirement"]="purpose-specific limits unselected";
    out["purpose_consequences"].push_back(n);
  }
  for(const auto &issue:d.issues()) {
    YAML::Node n;n["missing_requirement"]=rtc_pipeline_decision_cause_name(issue.cause);
    if(issue.scope){n["network"]=issue.scope->network;n["detector_local_column"]=issue.scope->detector;
      n["coordinate"]=static_cast<int>(issue.scope->coordinate);}
    else n["scope"]="complete-original-paired-view";
    out["unavailable_requirements"].push_back(n);
  }
  write_yaml(output/"reassessment-decision.yaml",out);return out;
}

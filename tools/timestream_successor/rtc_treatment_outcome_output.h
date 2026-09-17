#pragma once
// Offline serialization only. Numerical comparison and support belong to RTC.
YAML::Node write_treatment_outcome(const fs::path &output, const std::string &name,
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
      for(std::size_t i=0;i<r.original_matched.windows.size();++i) {
        const auto &a=r.original_matched.windows[i],&b=r.conditioned_matched.windows[i];
        YAML::Node win;win["rows"]=range(a.rows);win["run_index"]=a.run_index;
        win["begin_unix_sec"]=a.support_begin_unix_sec;win["end_unix_sec"]=a.support_end_unix_sec;
        win["padded_samples"]=a.padded_samples;win["original_chunk_median"]=a.centered_chunk_median;
        win["conditioned_chunk_median"]=b.centered_chunk_median;
        for(auto count:a.source_counts)win["source_counts_outside_protected_unknown"].push_back(count);
        win["representative_replacements"]=b.representative_replacements;win["replacement_influenced_samples"]=b.replacement_influenced_samples;
        win["unrepaired_influenced_samples"]=b.unrepaired_influenced_samples;win["representative_exclusions"]=b.representative_exclusions;
        row["windows"].push_back(win);
      }
    }
    out["records"].push_back(row);
  }
  before.close();after.close();require(bool(before)&&bool(after),"outcome spectrum output failed");
  out["original_psd_sha256"]=citlali::utils::sha256_file(before_path);
  out["conditioned_psd_sha256"]=citlali::utils::sha256_file(after_path);
  write_yaml(output/(name+".yaml"),out);return out;
}

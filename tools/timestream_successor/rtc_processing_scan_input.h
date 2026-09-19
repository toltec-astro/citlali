#pragma once
// Private offline adapter, included after the existing identity ingress helpers.
#include <citlali/core/engine/telescope.h>
#include <citlali/core/timestream/rtc/rtcproc.h>
#include <citlali/core/pipeline/timestream_processing_scan_native.h>
#include <citlali/core/pipeline/timestream_alignment_helpers.h>
#include <citlali/core/pipeline/observation_setup_validation.h>

namespace {
struct RecoveredProcessingScans {
  citlali::pipeline::ProcessingScanNativeProjection projection;
  YAML::Node receipt;
};

struct PreparedProcessingScans {
  Eigen::VectorXd grid;
  Eigen::MatrixXI indices;
  double rate;
  int inner_context, outer_context;
  std::string generation, configuration;
  // Native clocks are kept by network; the common grid is only the accepted
  // processing-slot relation, never a replacement for native physical runs.
  std::map<int,std::shared_ptr<const citlali::pipeline::NativeNetworkAlignment>> native_timing;
  std::map<int,std::tuple<double,double,std::int64_t>> cadence;
};
PreparedProcessingScans prepare_processing_scans(const YAML::Node &cfg,
    const citlali::pipeline::NativeObservationScope &scope,
    const auto &verified, RtcPerformanceTrace &performance) {
  using namespace citlali::pipeline;
  const auto request=cfg["decision_apply"];
  const auto effective=YAML::LoadFile(checked_file(cfg["effective_config"]).string());
  const auto provenance_path=checked_file(request["processing_provenance"]);
  const auto provenance=YAML::LoadFile(provenance_path.string());
  require(provenance["canonical_run_identity"]["config_sources"][0]["sha256"].as<std::string>()==
      "sha256:"+cfg["effective_config"]["sha256"].as<std::string>(),"processing generation uses another configuration");
  const auto actual_scope=provenance["realized"]["native_cohort_provenance"]["value"]["observation_binding"]["observation"];
  require(actual_scope["observation"].as<int>()==scope.observation &&
      actual_scope["subobservation"].as<int>()==scope.subobservation &&
      actual_scope["scan"].as<int>()==scope.scan,"processing generation has another observation");
  require(effective["runtime"]["interp_over_gaps"].as<bool>() &&
      !effective["timestream"]["polarimetry"]["enabled"].as<bool>(),
      "bounded timing adapter requires the recorded gap-grid and no HWPR time participation");
  std::map<int,std::string> expected;
  std::size_t input_index = 0;
  if (cfg["common_mode_census"]) {
    const auto name = std::to_string(scope.observation) + "_" +
                      std::to_string(scope.subobservation) + "_" +
                      std::to_string(scope.scan);
    std::size_t matches = 0;
    for (std::size_t i = 0; i < effective["inputs"].size(); ++i)
      if (effective["inputs"][i]["meta"]["name"].as<std::string>() == name) {
        input_index = i;
        ++matches;
      }
    require(matches == 1, "census needs one exact effective-config observation");
  }
  for (const auto &entry:effective["inputs"][input_index]["data_items"]) {
    auto interface=entry["meta"]["interface"].as<std::string>();
    if(interface.starts_with("toltec")) expected.emplace(std::stoi(interface.substr(6)),
        fs::path(entry["filepath"].as<std::string>()).filename().string());
  }
  require(request["timing_inputs"].size()==expected.size(),"incomplete processing timing population");
  performance.mark("processing_generation_inputs_bound");
  PreparedProcessingScans prepared;
  const auto runtime=load_runtime_config(checked_file(cfg["effective_config"]));
  std::vector<Eigen::VectorXd> times;std::set<int> seen;double rate=-1;
  for (const auto &entry:request["timing_inputs"]) {
    const int network=entry["network"].as<int>();const auto path=checked_file(entry);
    require(expected.contains(network) && seen.insert(network).second && expected.at(network)==path.filename(),
        "processing timing input differs from existing generation");
    const auto source=std::find_if(verified.sources.begin(),verified.sources.end(),[&](const auto &s){
        return s.role==apt::SourceRole::raw && s.network==network;});
    require(source!=verified.sources.end() && source->content_sha256=="sha256:"+entry["sha256"].as<std::string>() &&
        source->byte_count==fs::file_size(path),"processing timing source differs from exact APT raw identity");
    netCDF::NcFile f(path.string(),netCDF::NcFile::read);
    require(read_netcdf_scalar<int>(f,"Header.Toltec.ObsNum")==scope.observation &&
        read_netcdf_scalar<int>(f,"Header.Toltec.RoachIndex")==network &&
        read_netcdf_scalar<int>(f,"Header.Toltec.SubObsNum")==scope.subobservation &&
        read_netcdf_scalar<int>(f,"Header.Toltec.ScanNum")==scope.scan,"processing timestamp scope mismatch");
    const auto count=f.getVar("Data.Toltec.Ts").getDim(0).getSize();
    double fpga=0,hz=0;std::int64_t accum=0;
    auto ts=read_timestamp_slice(path,0,count,fpga,hz,accum);
    rate=reconcile_sample_rate_hz(rate,hz,network);
    require(runtime.interface_offset_present[network],"processing offset absent");
    times.push_back(network_time_from_timestream_matrix(ts.cast<double>(),fpga,runtime.interface_offsets_sec[network]));
    prepared.native_timing.emplace(network,std::make_shared<const NativeNetworkAlignment>(
        make_native_network_alignment(network,0,ts,fpga,0.0)));
    prepared.cadence.emplace(network,std::tuple{fpga,hz,accum});
  }
  const auto overlap=find_common_timestream_overlap(times,"RTC existing processing generation");
  performance.mark("processing_network_times_recovered");
  const auto grid=build_common_gap_time_grid(overlap.max_start,overlap.min_end,1/rate,"RTC existing processing generation");
  citlali::config::TimestreamChunkingConfig chunking;
  const auto node=effective["timestream"]["chunking"];
  chunking.mode=node["chunk_mode"].as<std::string>();chunking.value=node["value"].as<double>();
  chunking.force=node["force_chunking"].as<bool>();
  require(chunking.force && chunking.mode=="duration" && chunking.value==10,
      "unexpected existing processing interval definition");
  engine::Telescope telescope;telescope.logger=spdlog::default_logger();
  telescope.obs_pgm="Lissajous";telescope.exec_mode=false;telescope.fsmp=rate;
  // In the selected forced-duration branch Hold contributes cardinality only.
  // Its physical values are not used to define a new telescope/physical scan.
  telescope.tel_data["Hold"]=Eigen::VectorXd::Zero(grid.size());
  const auto observation=provenance["observation"]["value"];
  const auto inner=observation["filter_edge_guard_samples"],outer=observation["filter_outer_context_samples"];
  require(inner["available"].as<bool>() && outer["available"].as<bool>(),"processing context unavailable");
  // Serialized guard_samples is NOT context_samples. Recover the latter
  // through its existing owner, from the exact enabled FIR configuration.
  const auto raw=effective["timestream"]["raw_time_chunk"];
  require(!raw["IIR_filter"]["enabled"].as<bool>() && inner["value"].as<int>()==0,
      "unsupported historical edge-context policy");
  timestream::RTCProc rtc;
  rtc.run_tod_filter=raw["filter"]["enabled"].as<bool>();rtc.run_tod_iir_highpass=false;
  rtc.filter.n_terms=raw["filter"]["n_terms"].as<int>();rtc.filter_edge_guard.enabled=false;
  rtc.configure_filter_edge_guard(rate);
  telescope.inner_scans_chunk=rtc.filter_edge_guard.context_samples;
  telescope.outer_scans_chunk=outer["value"].as<int>();
  telescope.calc_scan_indices(chunking);
  const auto recorded_scans=provenance["realized"]["native_cohort_provenance"]["value"]["scans"];
  require(recorded_scans.size()==static_cast<std::size_t>(telescope.scan_indices.cols()),"processing scan population differs from recorded generation");
  for(std::size_t i=0;i<recorded_scans.size();++i){
    const auto r=recorded_scans[i];
    require(r["scan_index"].as<std::size_t>()==i &&
      r["rtc"]["selected_input_row_count"].as<int>()==telescope.scan_indices(1,i)-telescope.scan_indices(0,i)+1 &&
      r["rtc"]["loaded_input_row_count"].as<int>()==telescope.scan_indices(3,i)-telescope.scan_indices(2,i)+1,
      "reconstructed processing intervals disagree with recorded inner/outer support");
  }
  const auto generation="sha256:"+request["processing_provenance"]["sha256"].as<std::string>();
  performance.mark("processing_scan_relation_prepared");
  prepared.grid=grid;prepared.indices=telescope.scan_indices;prepared.rate=rate;
  prepared.inner_context=telescope.inner_scans_chunk;prepared.outer_context=telescope.outer_scans_chunk;
  prepared.generation=generation;prepared.configuration=cfg["effective_config"]["sha256"].as<std::string>();
  return prepared;
}
RecoveredProcessingScans recover_processing_scans(const YAML::Node &cfg,
    std::shared_ptr<const citlali::pipeline::NativePairedReadoutObservation> parent,
    const auto &verified, int target_network, RtcPerformanceTrace &performance,
    const PreparedProcessingScans *common=nullptr) {
  using namespace citlali::pipeline;
  std::optional<PreparedProcessingScans> local;
  if(!common)local=prepare_processing_scans(cfg,parent->scope(),verified,performance);
  const auto &prepared=common?*common:*local;
  const auto &grid=prepared.grid;const auto rate=prepared.rate;
  const auto &generation=prepared.generation;
  const auto &axis=parent->network(target_network).occurrence_axis();
  std::vector<Eigen::VectorXd> target_times{axis.native_timing_handle()->reconstructed_times_unix_sec()};
  const auto masks=build_common_time_grid_masks(target_times,grid,grid[0],1/rate,0.5/rate,spdlog::default_logger());
  auto associations=make_gap_native_slot_associations(*axis.native_timing_handle(),grid,masks[0],1/rate);
  RecoveredProcessingScans out;
  out.projection=project_processing_scans_to_native(parent,target_network,grid,associations,prepared.indices,
      .5/rate,generation,"existing-gap-grid-native-slot-associations+Telescope::calc_scan_indices;sha256:"+
      cfg["effective_config"]["sha256"].as<std::string>());
  performance.mark("processing_scan_native_projection_complete");
  out.receipt["generation"]=generation;out.receipt["grid_rows"]=grid.size();
  out.receipt["grid_first_unix_seconds"]=grid[0];out.receipt["grid_last_unix_seconds"]=grid[grid.size()-1];
  out.receipt["association_tolerance_seconds"]=.5/rate;
  out.receipt["maximum_association_residual_seconds"]=out.projection.maximum_association_residual_seconds;
  out.receipt["absolute_epoch_uncertainty"]="unquantified;not-required-for-existing-relative-slot-membership";
  out.receipt["readout_integration_assumption"]="preserved-provisional-uniform-average";
  out.receipt["inner_context_samples"]=prepared.inner_context;
  out.receipt["outer_context_samples"]=prepared.outer_context;
  out.receipt["timing_inputs"]=cfg["decision_apply"]["timing_inputs"];
  for (const auto &scan:out.projection.scans) {
    YAML::Node s;s["scan"]=scan.scan;s["science_slots"]=range(scan.science_slots);s["context_slots"]=range(scan.context_slots);
    s["unmapped_science_slots"]=scan.unmapped_science_slots;
    for(auto r:scan.science_native)s["science_native"].push_back(range(r));
    for(auto r:scan.context_native)s["context_native"].push_back(range(r));
    out.receipt["scans"].push_back(s);
  }
  for(auto r:out.projection.native_outside_processing)out.receipt["native_outside_processing"].push_back(range(r));
  return out;
}
}

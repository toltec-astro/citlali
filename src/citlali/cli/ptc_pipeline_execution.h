// Private application publication: numerical policy and products live in PTC.
#include <citlali/core/pipeline/timestream_ptc_pipeline.h>
#include <sys/resource.h>
namespace {
YAML::Node execute_connected_ptc(const PtcCalSource &source,const ProcessingScanNativeProjection &scans,
    const YAML::Node &cfg,const citlali::cli::DevelopmentPtcRequest &requested,const fs::path &output) {
    const auto began=std::chrono::steady_clock::now();
    const auto effective=YAML::LoadFile(checked_file(cfg["effective_config"]).string());
    const auto cleaning=effective["timestream"]["processed_time_chunk"]["clean"];
    require(cleaning && cleaning["enabled"].as<bool>() && cleaning["grouping"].as<std::vector<std::string>>()==std::vector<std::string>{"nw"},
        "successor.PTC_grouping_unavailable: explicit network cleaning required; no array/common-grid substitution");
    const auto pca=cleaning["standard_pca"];
    require(pca["enabled"].as<bool>() && pca["stddev_limit"].as<double>()==0 && cleaning["mask_radius_arcsec"].as<double>()==0,
        "successor.PTC_selection_unavailable: adaptive rank or source mask not connected");
    for(const auto *name:{"adaptive_selector","corr_grouping","marchenko_pastur","null_model"})
        require(!cleaning[name] || !cleaning[name]["enabled"] || !cleaning[name]["enabled"].as<bool>(),
            "successor.PTC_selection_unavailable: alternate cleaning selector requested");
    require(!cleaning["tau"] || cleaning["tau"].as<double>()==0,"successor.PTC_tau_selection_unavailable");
    PtcSolverRequest request;
    request.method=requested.method=="observed-als"?PtcMethod::observed_als:PtcMethod::pairwise_covariance;
    const std::array<std::string,3> arrays{"a1100","a1400","a2000"};
    const auto ranks=pca["n_eig_to_cut"][arrays.at(source.detector_factors().front().array)].as<std::vector<int>>();
    require(ranks.size()==1 && ranks.front()>0,"successor.PTC_explicit_rank_unavailable");
    request.rank=requested.rank?requested.rank:ranks.front();
    const auto evidence=PtcEvidence::learn(source,scans,request);
    const auto plan=PtcPlan::consider(evidence,source.val_snapshot_handle(),1);
    const auto signal=PtcAppliedSignal::apply(plan,source,source.val_snapshot_handle());
    const auto val=ValSnapshot::commit_ptc_output(source.val_snapshot_handle(),ValPtcOutputFacts::preserve(signal));
    const auto destination=output/"ptc";fs::create_directory(destination);
    YAML::Node record;record["schema"]="citlali-ptc-output-v1";record["unit"]=std::string(PtcCalSource::unit);
    record["method"]=requested.method;record["rank"]=request.rank;
    record["use_policy"]=std::string(PtcEvidence::use_policy);record["source_CAL_receipt_sha256"]=citlali::utils::sha256_file(output/"cal"/"receipt.yaml");
    record["input_VAL_generation"]=source.val_snapshot_handle()->generation().value;record["output_VAL_generation"]=val->generation().value;
    record["processing_generation"]=scans.binding->processing_generation();record["CAL_classification"]=(output/"cal"/"receipt.yaml").string();
    record["centering"]="per-detector-eligible-arithmetic-mean-not-restored";record["scaling"]="identity";
    record["metric"]="binary-eligible-Euclidean";record["relative_rank_tolerance"]=request.relative_rank_tolerance;
    record["relative_objective_tolerance"]=request.relative_objective_tolerance;record["iteration_limit"]=request.iteration_limit;
    record["requested_response"]="unavailable-upstream-complete-response;exact-local-PTC-operator-implemented";
    record["response_seconds"]=0.;record["response_executed"]=false;record["covariance"]="unavailable-not-zero";
    record["science_qualified"]=false;record["MAP_executed"]=false;record["FRUIT_executed"]=false;
    record["source_CAL_unchanged"]=true;record["r_role"]="retained-diagnostic-inert-through-parent";
    record["excluded_storage"]="zero-sentinel-never-a-measurement;eligibility-and-causes-authoritative";
    record["threads"]=Eigen::nbThreads();record["build_identity"]=std::string(CITLALI_GIT_VERSION);
    record["cache"]="prepared-once-per-group;basis-dependent-factors-reused-only-within-fixed-basis-and-mask";
    std::size_t failed=0,scheduled=0,eligible=0;
    for(std::size_t g=0;g<evidence->groups().size();++g) {
        const auto &group=evidence->groups()[g];const auto &fit=group.fit;const auto &applied=signal->groups()[g];const auto &p=group.input;
        YAML::Node r;r["scan"]=group.scan;r["network"]=group.network;r["native_interval"]=range(group.native);
        r["detector_indices"]=group.detectors;r["scheduled_times"]=p.centered.rows();r["detectors"]=p.centered.cols();
        r["eligible"]=p.eligible_count;r["complete_time_fraction"]=double(p.complete_times)/p.centered.rows();
        r["time_mask_patterns"]=p.time_patterns.size();r["detector_mask_patterns"]=p.detector_patterns.size();
        r["converged"]=fit.converged;r["stopping_reason"]=fit.stopping_reason;r["iterations"]=fit.iterations;
        r["basis_rows"]=fit.basis.rows();r["basis_columns"]=fit.basis.cols();r["objective"]=fit.objective;r["initialization"]=fit.initialization;r["preparation_seconds"]=p.preparation_seconds;
        r["fit_seconds"]=fit.fit_seconds;r["initialization_seconds"]=fit.initialization_seconds;r["covariance_seconds"]=fit.covariance_seconds;
        r["decomposition_seconds"]=fit.decomposition_seconds;r["coefficient_seconds"]=fit.coefficient_seconds;r["basis_seconds"]=fit.basis_seconds;r["check_seconds"]=fit.check_seconds;
        r["coefficient_factorizations"]=fit.coefficient_factorizations;r["coefficient_factor_reuses"]=fit.coefficient_factor_reuses;
        r["apply_seconds"]=applied.seconds;r["application_failed_times"]=applied.failed_times;r["retained"]=applied.retained;
        r["apply_factorizations"]=applied.factorizations;r["apply_factor_reuses"]=applied.factor_reuses;
        r["working_matrix_bytes"]=p.centered.size()*sizeof(double)+p.eligible.size()+p.mean.size()*sizeof(double)+fit.basis.size()*sizeof(double)+applied.values.size()*sizeof(double)+applied.causes.size();
        const auto writing=std::chrono::steady_clock::now();const auto prefix="segment-"+std::to_string(g);
        write_matrix(destination/(prefix+"-centered.f64"),p.centered);write_matrix(destination/(prefix+"-mean.f64"),p.mean);
        write_matrix(destination/(prefix+"-basis.f64"),fit.basis);write_matrix(destination/(prefix+"-cleaned.f64"),applied.values);
        auto bytes=[&](const std::string &suffix,const auto &matrix){std::ofstream stream(destination/(prefix+suffix),std::ios::binary);
            stream.write(reinterpret_cast<const char*>(matrix.data()),matrix.size());stream.close();require(bool(stream),"required PTC masks output failed");};
        bytes("-eligible.u8",p.eligible);bytes("-causes.u8",applied.causes);
        std::ofstream slots(destination/(prefix+"-slots.i64"),std::ios::binary);
        for(auto s:group.slots){const auto value=static_cast<std::int64_t>(s);slots.write(reinterpret_cast<const char*>(&value),8);}slots.close();require(bool(slots),"required PTC occurrence output failed");
        for(const auto *suffix:{"-centered.f64","-mean.f64","-basis.f64","-cleaned.f64","-eligible.u8","-causes.u8","-slots.i64"})
            r["files"][prefix+suffix]=citlali::utils::sha256_file(destination/(prefix+suffix));
        r["output_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-writing).count();
        record["segments"].push_back(r);failed+=!fit.converged;scheduled+=p.centered.size();eligible+=p.eligible_count;
    }
    record["failed_fits"]=failed;record["available"]=signal->available_count();record["scheduled_in_processing_segments"]=scheduled;record["eligible"]=eligible;
    std::size_t all_scheduled=0;for(const auto &d:source.grid_handle()->detectors())all_scheduled+=d.scheduled_count;
    record["scheduled_outside_processing_science_support"]=all_scheduled-scheduled;
    record["state"]=failed?"partial-PTC-fit-unavailable":signal->available_count()?"PTC-development-output":"no-PTC-output";
    record["wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    struct rusage usage{};if(getrusage(RUSAGE_SELF,&usage)==0) {
#ifdef __APPLE__
        record["process_peak_rss_bytes"]=usage.ru_maxrss;
#else
        record["process_peak_rss_bytes"]=usage.ru_maxrss*1024;
#endif
    }
    record["peak_memory_scope"]="whole-process-including-RTC-CAL;working_matrix_bytes-is-group-storage-not-peak";
    write_yaml(destination/"receipt.yaml",record);return record;
}
} // namespace

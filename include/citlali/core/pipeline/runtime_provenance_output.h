#pragma once

#include <citlali/core/config/runtime_execution_plan.h>
#include <citlali/core/pipeline/atomic_yaml_output.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>

namespace citlali::pipeline {

inline constexpr const char *runtime_provenance_schema_version =
    "citlali-runtime-provenance-v2";
inline constexpr const char *runtime_provenance_filename =
    "runtime_provenance.yaml";

inline YAML::Node runtime_config_node(
    const citlali::config::RuntimeConfig &config) {
    YAML::Node node;
    node["verbose"] = config.verbose;
    node["interp_over_gaps"] = config.interp_over_gaps;
    node["crop_detector_to_telescope_support"] =
        config.crop_detector_to_telescope_support;
    node["n_threads"] = config.n_threads;
    node["output_dir"] = config.output_dir;
    node["parallel_policy"] = std::string(
        citlali::config::to_string(config.parallel_policy));
    node["reduction_type"] = std::string(
        citlali::config::to_string(config.reduction_type));
    node["use_subdir"] = config.use_subdir;
    return node;
}

inline YAML::Node runtime_provenance_node(
    const citlali::config::RuntimeConfigProvenance &provenance) {
    YAML::Node root;
    root["schema_version"] = runtime_provenance_schema_version;
    root["initialized"] = provenance.initialized;
    root["requested"] = runtime_config_node(provenance.requested);
    root["effective"]["values"] =
        runtime_config_node(provenance.effective.values);
    root["effective"]["threads"]["requested"] =
        provenance.effective.threads.requested_threads;
    root["effective"]["threads"]["effective"] =
        provenance.effective.threads.effective_threads;
    root["effective"]["threads"]["omp"] =
        provenance.effective.threads.omp_threads;
    root["effective"]["threads"]["eigen"] =
        provenance.effective.threads.eigen_threads;
    root["effective"]["threads"]["fftw_plan"] =
        provenance.effective.threads.fftw_plan_threads;
    root["effective"]["threads"]["wiener_filter_omp"] =
        provenance.effective.threads.wiener_filter_omp;
    root["effective"]["threads"]["adjusted"] =
        provenance.effective.threads.adjusted;
    root["effective"]["threads"]["adjustment_reason"] =
        provenance.effective.threads.adjustment_reason;
    root["effective"]["threads"]["availability"]["slurm_cpus_per_task"] =
        provenance.effective.threads.availability.slurm_cpus_per_task;
    root["effective"]["threads"]["availability"]["affinity_cpus"] =
        provenance.effective.threads.availability.affinity_cpus;
    root["effective"]["threads"]["availability"]["hardware_cpus"] =
        provenance.effective.threads.availability.hardware_cpus;
    root["effective"]["threads"]["availability"]["available_threads"] =
        provenance.effective.threads.availability.available_threads;
    root["effective"]["threads"]["availability"]["source"] =
        provenance.effective.threads.availability.source;
    root["realized"]["threads"]["omp"] =
        provenance.realized.omp_threads;
    root["realized"]["threads"]["eigen"] =
        provenance.realized.eigen_threads;
    root["realized"]["threads"]["fftw_plan"] =
        provenance.realized.fftw_plan_threads;
    root["realized"]["threads"]["fftw_initialized"] =
        provenance.realized.fftw_threads_initialized;
    root["realized"]["parallel_policy"] = std::string(
        citlali::config::to_string(provenance.realized.parallel_policy));
    root["realized"]["reduction_type"] = std::string(
        citlali::config::to_string(provenance.realized.reduction_type));
    return root;
}

inline std::filesystem::path runtime_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / runtime_provenance_filename;
}

inline void write_runtime_provenance_file(
    const std::filesystem::path &reduction_dir,
    const citlali::config::RuntimeConfigProvenance &provenance) {
    const auto output_path = runtime_provenance_path(reduction_dir);
    write_yaml_file_atomic(output_path, runtime_provenance_node(provenance));
}

}  // namespace citlali::pipeline

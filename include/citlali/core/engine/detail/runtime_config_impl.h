#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
citlali::config::RuntimeConfig Engine::get_runtime_config(CT &config) {
    citlali::config::RuntimeConfig runtime_config;
    auto &diagnostics = citlali::pipeline::config_diagnostics(*this);

    // verbose mode?
    bool verbose = runtime_config.verbose;
    citlali::pipeline::read_mirrored_config_value(
        config, std::tuple{"runtime","verbose"}, verbose,
        runtime_config.verbose, diagnostics);

    // output directory
    std::string output_dir = runtime_config.output_dir;
    citlali::pipeline::read_mirrored_config_value(
        config, std::tuple{"runtime","output_dir"}, output_dir,
        runtime_config.output_dir, diagnostics);

    // number of threads to use
    int n_threads = runtime_config.n_threads;
    citlali::pipeline::read_mirrored_config_value(
        config, std::tuple{"runtime","n_threads"}, n_threads,
        runtime_config.n_threads, diagnostics);

    // overall parallel policy
    std::string parallel_policy{
        std::string(citlali::config::to_string(
            runtime_config.parallel_policy))};
    citlali::pipeline::read_parsed_mirrored_config_value(
        config, std::tuple{"runtime","parallel_policy"}, parallel_policy,
        runtime_config.parallel_policy, citlali::config::parse_parallel_policy,
        diagnostics, {"seq","omp"});

    // reduction type (science, pointing, beammap)
    std::string reduction_type{
        std::string(citlali::config::to_string(runtime_config.reduction_type))};
    citlali::pipeline::read_parsed_mirrored_config_value(
        config, std::tuple{"runtime","reduction_type"}, reduction_type,
        runtime_config.reduction_type, citlali::config::parse_reduction_type,
        diagnostics, {"science","pointing","beammap"});

    // create redu00, redu01... subdirectories
    bool use_subdir = runtime_config.use_subdir;
    citlali::pipeline::read_mirrored_config_value(
        config, std::tuple{"runtime","use_subdir"}, use_subdir,
        runtime_config.use_subdir, diagnostics);

    // interp over gaps in align_timestream
    bool interp_over_gaps = runtime_config.interp_over_gaps;
    citlali::pipeline::read_mirrored_config_value(
        config, std::tuple{"runtime","interp_over_gaps"}, interp_over_gaps,
        runtime_config.interp_over_gaps, diagnostics);

    return runtime_config;
}

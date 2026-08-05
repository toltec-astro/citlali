#pragma once

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <string>
#include <tuple>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
citlali::config::RuntimeConfig read_runtime_config(
    Config &config, Diagnostics &diagnostics) {
    citlali::config::RuntimeConfig runtime_config;

    bool verbose = runtime_config.verbose;
    read_mirrored_config_value(
        config, std::tuple{"runtime", "verbose"}, verbose,
        runtime_config.verbose, diagnostics);

    std::string output_dir = runtime_config.output_dir;
    read_mirrored_config_value(
        config, std::tuple{"runtime", "output_dir"}, output_dir,
        runtime_config.output_dir, diagnostics);

    int n_threads = runtime_config.n_threads;
    read_mirrored_config_value(
        config, std::tuple{"runtime", "n_threads"}, n_threads,
        runtime_config.n_threads, diagnostics);

    std::string parallel_policy{
        std::string(citlali::config::to_string(
            runtime_config.parallel_policy))};
    read_parsed_mirrored_config_value(
        config, std::tuple{"runtime", "parallel_policy"}, parallel_policy,
        runtime_config.parallel_policy, citlali::config::parse_parallel_policy,
        diagnostics, {"seq", "omp"});

    std::string reduction_type{
        std::string(citlali::config::to_string(runtime_config.reduction_type))};
    read_parsed_mirrored_config_value(
        config, std::tuple{"runtime", "reduction_type"}, reduction_type,
        runtime_config.reduction_type, citlali::config::parse_reduction_type,
        diagnostics, {"science", "pointing", "beammap"});

    bool use_subdir = runtime_config.use_subdir;
    read_mirrored_config_value(
        config, std::tuple{"runtime", "use_subdir"}, use_subdir,
        runtime_config.use_subdir, diagnostics);

    bool interp_over_gaps = runtime_config.interp_over_gaps;
    read_mirrored_config_value(
        config, std::tuple{"runtime", "interp_over_gaps"}, interp_over_gaps,
        runtime_config.interp_over_gaps, diagnostics);

    bool crop_detector_to_telescope_support =
        runtime_config.crop_detector_to_telescope_support;
    read_mirrored_config_value(
        config, std::tuple{"runtime", "crop_detector_to_telescope_support"},
        crop_detector_to_telescope_support,
        runtime_config.crop_detector_to_telescope_support, diagnostics);

    return runtime_config;
}

}  // namespace citlali::pipeline

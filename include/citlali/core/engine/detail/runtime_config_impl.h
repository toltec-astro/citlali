#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/config_parse_tracking.h>

template<typename CT>
citlali::config::RuntimeConfig Engine::get_runtime_config(CT &config) {
    citlali::config::RuntimeConfig runtime_config;

    // verbose mode?
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"runtime","verbose"}, verbose_mode,
        runtime_config.verbose, missing_keys, invalid_keys);

    // output directory
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"runtime","output_dir"}, output_dir,
        runtime_config.output_dir, missing_keys, invalid_keys);

    // number of threads to use
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"runtime","n_threads"}, n_threads,
        runtime_config.n_threads, missing_keys, invalid_keys);

    // overall parallel policy
    citlali::engine_detail::read_parsed_mirrored_config_value(
        config, std::tuple{"runtime","parallel_policy"}, parallel_policy,
        runtime_config.parallel_policy, citlali::config::parse_parallel_policy,
        missing_keys, invalid_keys, {"seq","omp"});

    // reduction type (science, pointing, beammap)
    std::string reduction_type{
        std::string(citlali::config::to_string(runtime_config.reduction_type))};
    citlali::engine_detail::read_parsed_mirrored_config_value(
        config, std::tuple{"runtime","reduction_type"}, reduction_type,
        runtime_config.reduction_type, citlali::config::parse_reduction_type,
        missing_keys, invalid_keys, {"science","pointing","beammap"});

    // create redu00, redu01... subdirectories
    bool use_subdir = runtime_config.use_subdir;
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"runtime","use_subdir"}, use_subdir,
        runtime_config.use_subdir, missing_keys, invalid_keys);

    // interp over gaps in align_timestream
    bool interp_over_gaps = runtime_config.interp_over_gaps;
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"runtime","interp_over_gaps"}, interp_over_gaps,
        runtime_config.interp_over_gaps, missing_keys, invalid_keys);

    return runtime_config;
}

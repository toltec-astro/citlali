#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/config_parse_tracking.h>

template<typename CT>
citlali::config::RuntimeConfig Engine::get_runtime_config(CT &config) {
    citlali::config::RuntimeConfig runtime_config;

    auto parsed_cleanly = [&](std::size_t missing_before,
                              std::size_t invalid_before) {
        return citlali::engine_detail::config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before);
    };

    // verbose mode?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, verbose_mode, missing_keys, invalid_keys,
                         std::tuple{"runtime","verbose"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            runtime_config.verbose = verbose_mode;
        }
    }
    // output directory
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, output_dir, missing_keys, invalid_keys,
                         std::tuple{"runtime","output_dir"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            runtime_config.output_dir = output_dir;
        }
    }
    // number of threads to use
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, n_threads, missing_keys, invalid_keys,
                         std::tuple{"runtime","n_threads"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            runtime_config.n_threads = n_threads;
        }
    }
    // overall parallel policy
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, parallel_policy, missing_keys, invalid_keys,
                         std::tuple{"runtime","parallel_policy"},{"seq","omp"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_parallel_policy(parallel_policy)) {
                runtime_config.parallel_policy = *parsed;
            }
        }
    }
    // reduction type (science, pointing, beammap)
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, redu_type, missing_keys, invalid_keys,
                         std::tuple{"runtime","reduction_type"},{"science","pointing","beammap"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_reduction_type(redu_type)) {
                runtime_config.reduction_type = *parsed;
            }
        }
    }
    // create redu00, redu01... subdirectories
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, use_subdir, missing_keys, invalid_keys,
                         std::tuple{"runtime","use_subdir"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            runtime_config.use_subdir = use_subdir;
        }
    }
    // interp over gaps in align_timestream
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, interp_over_gaps, missing_keys, invalid_keys,
                         std::tuple{"runtime","interp_over_gaps"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            runtime_config.interp_over_gaps = interp_over_gaps;
        }
    }
    return runtime_config;
}

#pragma once

#include <citlali/core/pipeline/fruit_loop_restart_lifecycle.h>
#include <citlali/core/pipeline/output_config_copy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <string>

namespace citlali::pipeline {

template <class Engine>
bool should_prepare_iteration_output_layout(const Engine &engine) {
    return fruit_loops_config(engine).save_all_iters ||
           engine.iteration.fruit_iter == 0 ||
           first_restarted_iteration(engine);
}

template <class TodProc>
void create_iteration_output_directory(
    TodProc &todproc, StageProfileCollector &stage_profile) {
    todproc.create_output_dir(stage_profile);
}

template <class ConfigFilepaths, class Logger>
void copy_iteration_config_files(const ConfigFilepaths &config_filepaths,
                                 const std::string &reduction_dir,
                                 const Logger &logger) {
    copy_config_files_to_reduction_dir(config_filepaths, reduction_dir,
                                       logger);
}

template <class TodProc, class ConfigFilepaths, class Logger>
void prepare_iteration_output_layout_if_needed(
    TodProc &todproc, const ConfigFilepaths &config_filepaths,
    StageProfileCollector &stage_profile,
    const Logger &logger) {
    auto &engine = todproc.engine();

    if (should_prepare_iteration_output_layout(engine)) {
        create_iteration_output_directory(todproc, stage_profile);
        copy_iteration_config_files(
            config_filepaths, engine.output_paths.redu_dir_name, logger);
    }
}

}  // namespace citlali::pipeline

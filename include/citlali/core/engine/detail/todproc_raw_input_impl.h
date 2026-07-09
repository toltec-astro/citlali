#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/rawobs_detector_inventory.h>
#include <citlali/core/pipeline/rawobs_tone_frequency_inventory.h>
#include <citlali/core/pipeline/reduction_output_dirs.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::get_apt_from_files(const RawObs &rawobs) {
    const auto inventory =
        citlali::pipeline::read_rawobs_detector_inventory(
            rawobs, engine().toltec_io.nw_to_array_map, logger);
    citlali::pipeline::populate_internal_apt_from_detector_inventory(
        engine().calib, inventory);
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::get_tone_freqs_from_files(const RawObs &rawobs) {
    const auto tone_freqs =
        citlali::pipeline::read_rawobs_tone_frequencies(rawobs, logger);
    citlali::pipeline::assign_tone_frequencies_by_network(
        engine().calib, tone_freqs, logger);

    if (!engine().telescope.sim_obs) {
        citlali::pipeline::flag_duplicate_tones(
            engine().calib, engine().rtcproc.delta_f_min_Hz, logger);
    }
}

// create output directories
template <class EngineType>
void TimeOrderedDataProc<EngineType>::create_output_dir() {
    // redu subdir
    engine().redu_dir_name = "";

    // create reduction subdir
    if (citlali::config::reduction_subdirs_active(
            engine().typed_config.runtime)) {
        engine().redu_dir_name =
            citlali::pipeline::next_reduction_subdir_path(
                engine().output_dir, engine().redu_dir_num);
        fs::create_directories(engine().redu_dir_name);
        citlali::pipeline::configure_reduction_logging_and_profile(
            engine().redu_dir_name, logger);
    }
    else {
        engine().redu_dir_name = engine().output_dir + "/";
        citlali::pipeline::configure_reduction_logging_and_profile(
            engine().redu_dir_name, logger);
    }

    // coadded subdir
    if (citlali::pipeline::coadd_outputs_enabled(engine())) {
        engine().coadd_dir_name = engine().redu_dir_name + "/coadded/";
        citlali::pipeline::create_coadd_output_dirs(
            engine().coadd_dir_name,
            citlali::pipeline::map_filter_outputs_enabled(engine()), logger);
    }
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::check_inputs(const RawObs &rawobs) {
    const Eigen::Index n_dets =
        citlali::pipeline::read_rawobs_detector_count(rawobs, logger);

    // check if number of detectors in apt file is equal to those in files
    if (n_dets != engine().calib.n_dets) {
        logger->error("number of detectors in data files and apt file do not match");
        std::exit(EXIT_FAILURE);
    }
}

// align tod with telescope

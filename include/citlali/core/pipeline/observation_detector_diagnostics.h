#pragma once

namespace citlali::pipeline {

template <class Engine>
bool should_load_adc_snap_diagnostics(const Engine &engine) {
    return !engine.telescope.sim_obs;
}

template <class TodProc, class RawObs, class Logger>
void load_tone_frequency_diagnostics(TodProc &todproc, const RawObs &rawobs,
                                     const Logger &logger) {
    logger->debug("getting tone frequencies");
    todproc.get_tone_freqs_from_files(rawobs);
}

template <class TodProc, class RawObs, class Logger>
void load_raw_detector_diagnostics(TodProc &todproc, const RawObs &rawobs,
                                   const Logger &logger) {
    load_tone_frequency_diagnostics(todproc, rawobs, logger);

    if (should_load_adc_snap_diagnostics(todproc.engine())) {
        logger->debug("getting adc snap data");
        todproc.get_adc_snap_from_files(rawobs);
    }
}

template <class TodProc, class RawObs, class Logger>
void load_reduction_observation_detector_diagnostics(
    TodProc &todproc, const RawObs &rawobs, const Logger &logger) {
    load_raw_detector_diagnostics(todproc, rawobs, logger);
}

}  // namespace citlali::pipeline

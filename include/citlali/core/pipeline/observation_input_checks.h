#pragma once

namespace citlali::pipeline {

template <class Logger>
void log_observation_input_check(const Logger &logger) {
    logger->debug("checking inputs");
}

template <class TodProc, class RawObs>
void run_observation_input_check(TodProc &todproc, const RawObs &rawobs) {
    todproc.check_inputs(rawobs);
}

template <class TodProc, class RawObs, class Logger>
void check_observation_inputs(TodProc &todproc, const RawObs &rawobs,
                              const Logger &logger) {
    log_observation_input_check(logger);
    todproc.check_inputs(rawobs);
}

}  // namespace citlali::pipeline

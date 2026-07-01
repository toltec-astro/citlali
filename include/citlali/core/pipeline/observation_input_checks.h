#pragma once

namespace citlali::pipeline {

template <class TodProc, class RawObs, class Logger>
void check_observation_inputs(TodProc &todproc, const RawObs &rawobs,
                              const Logger &logger) {
    logger->debug("checking inputs");
    todproc.check_inputs(rawobs);
}

}  // namespace citlali::pipeline

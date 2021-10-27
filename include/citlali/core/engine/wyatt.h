#pragma once

#include <vector>
#include <Eigen/Core>

#include <tula/grppi.h>

#include <citlali/core/engine/engine.h>
#include <citlali/core/utils/fitting.h>


using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Wyatt: public EngineBase {

    void setup();
    auto run_timestream();
    auto run_loop();

    template <class KidsProc, class RawObs>
    auto timestream_pipeline(KidsProc &, RawObs &);

    template <class KidsProc, class RawObs>
    auto loop_pipeline(KidsProc &, RawObs &);

    template <class KidsProc, class RawObs>
    auto pipeline(KidsProc &, RawObs &);

    void output();

};

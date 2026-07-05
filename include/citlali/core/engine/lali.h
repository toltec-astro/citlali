#pragma once

#include <functional>
#include <mutex>
#include <condition_variable>

#include <citlali/core/engine/engine.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Lali: public Engine {
public:
    using input_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;
    using run_stage_t = decltype(grppi::farm(
        std::declval<int>(), std::declval<std::function<void(input_t &)>>() ));

    // initial setup for each obs
    void setup();

    // main grppi pipeline
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // run the reduction for the obs
    auto run() -> run_stage_t;

    // output files
    template <mapmaking::MapType map_type>
    void output();
};


#include <citlali/core/engine/detail/lali_setup_pipeline_impl.h>
#include <citlali/core/engine/detail/lali_run_impl.h>
#include <citlali/core/engine/detail/lali_output_impl.h>

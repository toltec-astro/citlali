#pragma once

#include "config.h"

#include "timestream/timestream.h"

#include "map/map_utils.h"
#include "map/map.h"

#include "result.h"


// TCData is the data structure of which RTCData and PTCData are a part
using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// Selects the type of TCData
using timestream::LaliDataKind;

namespace beammap {

class Beammap : public Result {
public:

    void setup();
    auto run();
};

// Sets up the map dimensions and lowpass+highpass filter
void Beammap::setup() {

  // Check if lowpass+highpass filter requested.
  // If so, make the filter once here and reuse
  // it later for each RTCData.

  if (config.get_typed<bool>(std::tuple{"tod", "filter", "enabled"})) {
    auto fLow = config.get_typed<double>(std::tuple{"tod", "filter", "flow"});
    auto fHigh = config.get_typed<double>(std::tuple{"tod", "filter", "fhigh"});
    auto aGibbs = config.get_typed<double>(std::tuple{"tod", "filter", "agibbs"});
    auto nTerms = config.get_typed<int>(std::tuple{"tod", "filter", "nterms"});

    filter.makefilter(fLow, fHigh, aGibbs, nTerms, samplerate);
  }

}

// Runs the timestream -> map analysis pipeline.
auto Beammap::run(){

}

} // namespace

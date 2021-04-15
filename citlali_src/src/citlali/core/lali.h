#pragma once

#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/math/constants/constants.hpp>

#include "config.h"

#include "timestream/timestream.h"

#include "map/map_utils.h"
#include "map/map.h"
#include "map/coadd.h"

#include "result.h"

/*
This header file holds the main class and its methods for Lali.
*/

//TCData is the data structure of which RTCData and PTCData are a part
using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

//Selects the type of TCData
using timestream::LaliDataKind;

namespace lali {

/*
class PipelineInput {
    std::vector<std::string> config_files;
    std::vector<std::string> input_files;
    std::shared_ptr<YamlConfig> config_;
};

class AzTECInput : public PipelineInput {

    enum DataType { UseAzTEC = 0, UseTolTEC = 1 };

    aztec::BeammapData Data;
    template<DataType datatype>
    static AzTECInput getAztecData(int argc, char *argv[])
    {
        Tester tester;
        tester.getInputs(argc, argv);
        tester.getConfig();

        if constexpr (datatype == UseAzTEC) {
            tester.getData();
        }

        return tester;
    }
};

class TolTECInput : public PipelineInput {
    toltec::Data Data;
};
*/

enum DataType {
    AzTEC_Testing = 0,
    AzTEC_as_TolTEC_Testing = 1,
    AzTEC = 2,
    TolTEC = 3,
    MUSCAT = 4
};

class DataStruct {
    int scans;
    int scan_indices;
    int meta;
};

// template<typename InputType>
class Lali : public Result {
public:
    //Lali(std::shared_ptr<YamlConfig> c_): config(std::move(c_)) {}
    //Lali(int ac, char *av[]): argc(ac), argv(std::move(av)) {}

    // Config file
    YamlConfig config;

    lali::TelData telMD;

    // Total number of detectors and scans
    int ndetectors, nscans;

    // Sample rate
    double samplerate;

    // Lowpass+Highpass filter class
    timestream::Filter filter;

    // Class to hold and populate maps
    mapmaking::MapStruct Maps;
    // Class to hold and populate coadded maps
    mapmaking::CoaddedMapStruct CoaddedMaps;

    // Number of threads to use for the grppi::farm
    Eigen::Index nThreads = Eigen::nbThreads();

    // std::map for the detector Az/El offsets
    std::unordered_map<std::string, Eigen::VectorXd> offsets;

    // Eigen Vector for nws
    Eigen::VectorXd nw;

    // Eigen Vector for array names
    Eigen::VectorXd array_name;

    // Array indices
    std::vector<std::tuple<int,int>> array_index;

    //Random generator for noise maps
    // boost::random_device rd;
    //  boost::random::mt19937 rng{rd};
    // boost::random::uniform_int_distribution<> rands{0, 1};

    void setup();
    auto run();
};

// Sets up the map dimensions and lowpass+highpass filter
void Lali::setup()
{

  // Check if lowpass+highpass filter requested.
  // If so, make the filter once here and reuse
  // it later for each RTCData.

  if (config.get_typed<bool>(std::tuple{"tod", "filter", "enabled"})) {
    auto fLow = config.get_typed<double>(std::tuple{"tod", "filter", "flow"});
    auto fHigh = config.get_typed<double>(std::tuple{"tod", "filter", "fhigh"});
    auto aGibbs = config.get_typed<double>(std::tuple{"tod", "filter", "agibbs"});
    auto nTerms = config.get_typed<int>(std::tuple{"tod", "filter", "nterms"});
    // auto samplerate = config.get_typed<double>("std::tuple{"tod", "filter", "samplerate"});

    filter.makefilter(fLow, fHigh, aGibbs, nTerms, samplerate);
  }

  // Resize the maps to nrows x ncols and set rcphys and ccphys
  // Maps.allocateMaps(Data.telMetaData, offsets, config);
  // CoaddedMaps.allocateMaps(Maps, Data.telMetaData, offsets, config);

}

// Runs the timestream -> map analysis pipeline.
auto Lali::run(){

    auto farm =  grppi::farm(nThreads,[&](auto in) -> TCData<LaliDataKind::PTC,Eigen::MatrixXd> {

        SPDLOG_INFO("scans before rtcproc {}", in.scans.data);

        /*Stage 1: RTCProc*/
        RTCProc rtcproc(config);
        TCData<LaliDataKind::PTC,Eigen::MatrixXd> out;
        rtcproc.run(in, out, this);

        SPDLOG_INFO("scans after rtcproc {}", out.scans.data);

        /*Stage 2: PTCProc*/
        PTCProc ptcproc(config);
        ptcproc.run(out, out, this);

        SPDLOG_INFO("scans after ptcproc {}", out.scans.data);

        /*Stage 3 Populate Map*/
        out.mnum.data = in.mnum.data;
        Maps.mapPopulate(out, offsets, config);

        SPDLOG_INFO("----------------------------------------------------");
        SPDLOG_INFO("*Done with scan {}...*",out.index.data);
        SPDLOG_INFO("----------------------------------------------------");

        // Return farm object to pipeline
        return out;
    });

    return farm;
}
} //namespace

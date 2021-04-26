#pragma once

#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/math/constants/constants.hpp>
#include "config.h"
#include "timestream/timestream.h"
#include "map/map_utils.h"
#include "map/map.h"
// #include "map/coadd.h"
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
    int n_detectors, nscans;

    // Sample rate
    double samplerate;

    // Lowpass+Highpass filter class
    timestream::Filter filter;

    // Class to hold and populate maps
    mapmaking::MapStruct Maps;
    // Class to hold and populate coadded maps
    // mapmaking::CoaddedMapStruct CoaddedMaps;

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

    template <typename Derived, class C, class RawObs>
    auto pipeline(Eigen::DenseBase<Derived> &, C &, RawObs &);
};

// Sets up the lowpass+highpass filter
void Lali::setup() {

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
        Maps.mapPopulate(out, offsets, config, array_index);

        SPDLOG_INFO("----------------------------------------------------");
        SPDLOG_INFO("*Done with scan {}...*",out.index.data);
        SPDLOG_INFO("----------------------------------------------------");

        // Return farm object to pipeline
        return out;
    });

    return farm;
}

template <typename Derived, class C, class RawObs>
auto Lali::pipeline(Eigen::DenseBase<Derived> &scanindicies, C &kidsproc, RawObs &rawobs){
    // do grppi reduction
    auto ex_name = config.get_str(std::tuple{"runtime","policy"});
    auto ncores = config.get_str(std::tuple{"runtime","ncores"});

    grppi::pipeline(grppiex::dyn_ex(ex_name),
        [&]() -> std::optional<TCData<LaliDataKind::RTC, Eigen::MatrixXd>> {
        // Variable for current scan
        static auto scan = 0;
        // Current scanlength
        Eigen::Index scanlength;
        // Index of the start of the current scan
        Eigen::Index si = 0;

        while (scan < scanindicies.cols()) {

            // First scan index for current scan
            si = scanindicies(2, scan);
            SPDLOG_INFO("si {}", si);
            // Get length of current scan (do we need the + 1?)
            scanlength = scanindicies(3, scan) - scanindicies(2, scan) + 1;
            SPDLOG_INFO("scanlength {}", scanlength);

            // Declare a TCData to hold data
            // predefs::TCData<predefs::LaliDataKind::RTC, Eigen::MatrixXd> rtc;
            TCData<LaliDataKind::RTC, Eigen::MatrixXd> rtc;

            // Get scan indices and push into current RTC
            rtc.scanindex.data = scanindicies.col(scan);

            // This index keeps track of which scan the RTC actually belongs to.
            rtc.index.data = scan + 1;

            // Get telescope pointings for scan (move to Eigen::Maps to save
            // memory and time)

            // Get the requested map type
            auto maptype = config.get_str(std::tuple{"map","type"});
            SPDLOG_INFO("mapy_type {}", maptype);

            // Put that scan's telescope pointing into RTC
            if (std::strcmp("RaDec", maptype.c_str()) == 0) {
                rtc.telLat.data = telMD.telMetaData["TelRaPhys"].segment(si, scanlength);
                rtc.telLon.data = telMD.telMetaData["TelDecPhys"].segment(si, scanlength);
            }

            else if (std::strcmp("AzEl", maptype.c_str()) == 0) {
                rtc.telLat.data = telMD.telMetaData["TelAzPhys"].segment(si, scanlength);
                rtc.telLon.data = telMD.telMetaData["TelElPhys"].segment(si, scanlength);
            }

            rtc.telElDes.data = telMD.telMetaData["TelElDes"].segment(si, scanlength);
            rtc.ParAng.data = telMD.telMetaData["ParAng"].segment(si, scanlength);

            rtc.scans.data.resize(scanlength, n_detectors);
            rtc.scans.data = kidsproc.populate_rtc(rawobs, rtc.scanindex.data, scanlength, n_detectors);

            rtc.flags.data.resize(scanlength, n_detectors);
            rtc.flags.data.setOnes();

            // Eigen::MatrixXd scans;
            // rtc.scans.data.setRandom(scanlength, n_detectors);
            //addsource(rtc, offsets, config);

            // Increment scan
            scan++;

            return rtc;
        }
        return {};

    },
        run());

    SPDLOG_INFO("Normalizing Maps by Weight Map");
    {
        logging::scoped_timeit timer("mapNormalize()");
        Maps.mapNormalize(config);
    }
}

} //namespace

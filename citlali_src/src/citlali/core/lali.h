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

/*enum DataType {
    AzTEC_Testing = 0,
    AzTEC_as_TolTEC_Testing = 1,
    AzTEC = 2,
    TolTEC = 3,
    MUSCAT = 4
};*/


// template<typename InputType>
class Lali : public Result {
public:
    //Lali(std::shared_ptr<YamlConfig> c_): config(std::move(c_)) {}
    //Lali(int ac, char *av[]): argc(ac), argv(std::move(av)) {}

    // Config file
    YamlConfig config;

    lali::TelData telMD;

    // Total number of detectors and scans
    int n_detectors, nscans, obsid;

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
    std::unordered_map<std::string, Eigen::VectorXd> fwhms;

    // Eigen Vector for nws
    Eigen::VectorXd nw;

    // Eigen Vector for array names
    Eigen::VectorXd array_name;

    // Eigen Vector for flux calibration
    Eigen::VectorXd fluxscale;

    // Array indices
    std::vector<std::tuple<int,int>> array_index;

    // Detector indices
    std::vector<std::tuple<int,int>> det_index;

    Eigen::MatrixXd fittedParams, fittedParams_0;

    //Random generator for noise maps
    // boost::random_device rd;
    // boost::random::mt19937 rng{rd};
    // boost::random::uniform_int_distribution<> rands{0, 1};

    void setup();
    auto run();

    template <typename Derived, class C, class RawObs>
    auto pipeline(Eigen::DenseBase<Derived> &, C &, RawObs &);

    void output();

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

    auto farm =  grppi::farm(nThreads,[&](auto input_tuple) -> TCData<LaliDataKind::PTC,Eigen::MatrixXd> {

        // SPDLOG_INFO("scans before rtcproc {}", in.scans.data);

        auto in = std::get<0>(input_tuple);
        auto si = std::get<1>(input_tuple);

        Eigen::Index scanlength = in.scans.data.rows();

        in.flags.data.resize(scanlength, n_detectors);
        in.flags.data.setOnes();

        // Get telescope pointings for scan (move to Eigen::Maps to save
        // memory and time)

        // Get the requested map type
        auto maptype = config.get_str(std::tuple{"map","type"});
        // SPDLOG_INFO("map_type {}", maptype);

        // Put that scan's telescope pointing into RTC
        if (std::strcmp("RaDec", maptype.c_str()) == 0) {
            in.telLat.data = telMD.telMetaData["TelDecPhys"].segment(si, scanlength);
            in.telLon.data = telMD.telMetaData["TelRaPhys"].segment(si, scanlength);
        }

        else if (std::strcmp("AzEl", maptype.c_str()) == 0) {
            in.telLat.data = telMD.telMetaData["TelElPhys"].segment(si, scanlength);
            in.telLon.data = telMD.telMetaData["TelAzPhys"].segment(si, scanlength);
        }

        in.telElDes.data = telMD.telMetaData["TelElDes"].segment(si, scanlength);
        in.ParAng.data = telMD.telMetaData["ParAng"].segment(si, scanlength);

        /*Stage 1: RTCProc*/
        RTCProc rtcproc(config);
        TCData<LaliDataKind::PTC,Eigen::MatrixXd> out;
        rtcproc.run(in, out, this);

        SPDLOG_INFO("scans after rtcproc {}", out.scans.data);

        /*Stage 2: PTCProc*/
        PTCProc ptcproc(config);
        ptcproc.run(out, out, this);

        /*Stage 3 Populate Map*/
        Maps.mapPopulate(out, offsets, config, det_index);

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
    nThreads = config.get_typed<int>(std::tuple{"runtime","ncores"});

    omp_set_num_threads(nThreads);

    grppi::pipeline(grppiex::dyn_ex(ex_name),
        [&]() -> std::optional<std::tuple<TCData<LaliDataKind::RTC, Eigen::MatrixXd>, Eigen::Index>> {
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

            // rtc.scans.data.resize(scanlength, n_detectors);
            rtc.scans.data = kidsproc.populate_rtc(rawobs, rtc.scanindex.data, scanlength, n_detectors);

            // Increment scan
            scan++;

            // Return tuple of RTCData and scanindex
            return std::tuple<TCData<LaliDataKind::RTC, Eigen::MatrixXd>, Eigen::Index> (rtc, si);
        }
        return {};

    },
        run());

    SPDLOG_INFO("Normalizing Maps by Weight Map");
    Maps.mapNormalize(config);
}

void Lali::output() {
    std::string filepath = config.get_str(std::tuple{"runtime","output_filepath"});
    for (int i = 0; i < array_index.size(); i++) {
        auto filename = composeFilename<lali::TolTEC, lali::Simu, lali::Science>(this, i);
        writeMapsToFITS(this, filepath, filename, i, det_index);
        //std::string out = filepath + filename + std::to_string(i) + ".nc";
        // writeMapsToNetCDF(this, filepath, filename);
    }
}

} //namespace

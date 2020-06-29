#pragma once
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/math/constants/constants.hpp>
static double pi = boost::math::constants::pi<double>();

// Arcseconds in 360 degrees
#define ASEC_CIRC 1296000.0
// rad per arcsecond
#define RAD_ASEC (2.0*pi / ASEC_CIRC)

#include "config.h"
#include "read.h"

#include "tester.h"

#include "timestream/timestream.h"

#include "map/map_utils.h"
#include "map/map.h"
#include "map/coadd.h"

#include "../core/result.h"

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

class Lali : public Result, public Tester
{
public:
    //Lali(std::shared_ptr<YamlConfig> c_): config(std::move(c_)) {}
    //Lali(int ac, char *av[]): argc(ac), argv(std::move(av)) {}

    std::shared_ptr<YamlConfig> config;

    Tester tester;

    // Total number of detectors and scans
    int ndetectors, nscans;

    // Lowpass+Highpass filter class
    timestream::Filter filter;

    // Class to hold and populate maps
    mapmaking::MapStruct Maps;
    // Class to hold and populate coadded maps
    mapmaking::CoaddedMapStruct CoaddedMaps;

    // Number of threads to use for the grppi::farm
    Eigen::Index nThreads = Eigen::nbThreads();

    // std::map for the detector Az/El offsets
    std::map<std::string, Eigen::VectorXd> offsets;

    //Random generator for noise maps
    boost::random_device rd;
    boost::random::mt19937 rng{rd};
    boost::random::uniform_int_distribution<> rands{0,1};

    void setup();

    template<DataType datatype>
    void makeTestData(int argc, char *argv[]) {

        if constexpr (datatype == UseAzTEC) {
            tester = tester.getAztecData<datatype>(argc, argv);
            config = std::move(tester.config_);
            Data = std::move(tester.Data);

            // Set ndetectors and nscans from the data
            ndetectors = Data.meta.template get_typed<int>("ndetectors");
            nscans = Data.meta.template get_typed<int>("nscans");
        }
    }

    auto run();

};

// Sets up the map dimensions and lowpass+highpass filter
void Lali::setup()
{

  // Check if lowpass+highpass filter requested.
  // If so, make the filter once here and reuse
  // it later for each RTCData.

  if (this->config->get_typed<int>("proc.rtc.filter")) {
    auto fLow = this->config->get_typed<double>("proc.rtc.filter.flow");
    auto fHigh = this->config->get_typed<double>("proc.rtc.filter.fhigh");
    auto aGibbs = this->config->get_typed<double>("proc.rtc.filter.agibbs");
    auto nTerms = this->config->get_typed<int>("proc.rtc.filter.nterms");
    auto samplerate = this->config->get_typed<double>("proc.rtc.samplerate");

    filter.makefilter(fLow, fHigh, aGibbs, nTerms, samplerate);
  }

  // Get the detector Az and El offsets and place them in a
  // std::map

  offsets["azOffset"] = Eigen::Map<Eigen::VectorXd>(
      &(this->config->get_typed<std::vector<double>>("az_offset"))[0],
      ndetectors);
  offsets["elOffset"] = Eigen::Map<Eigen::VectorXd>(
      &(this->config->get_typed<std::vector<double>>("el_offset"))[0],
      ndetectors);

  // Resize the maps to nrows x ncols and set rcphys and ccphys
  Maps.allocateMaps(Data.telMetaData, offsets, config);
  CoaddedMaps.allocateMaps(Maps, Data.telMetaData, offsets, config);


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
        ptcproc.run(out, out);

        SPDLOG_INFO("scans after ptcproc {}", out.scans.data);

        /*Stage 3 Populate Map*/
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

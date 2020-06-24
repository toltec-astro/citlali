#pragma once

#include <boost/math/constants/constants.hpp>
static double pi = boost::math::constants::pi<double>();

// Arcseconds in 360 degrees
#define ASEC_CIRC 1296000.0
// rad per arcsecond
#define RAD_ASEC (2.0*pi / ASEC_CIRC)

#include "config.h"
#include "read.h"

#include "timestream/timestream.h"

#include "map/map.h"
#include "map/map_utils.h"
#include "map/coadd.h"

#include "../core/result.h"

/*
This file holds the main class and its methods for Lali.
*/

//TCData is the data structure of which RTCData and PTCData are a part
using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

//Selects the type of TCData
using timestream::LaliDataKind;

// Namespaces for getting cmd line values
namespace po = boost::program_options;
namespace pt = boost::property_tree;

namespace lali {

class Lali : public Result {
public:
  std::shared_ptr<YamlConfig> config;

  // Total number of detectors and scans
  int ndetectors, nscans;

  // Lowpass+Highpass filter class
  timestream::Filter filter;

  // Class to hold and populate maps
  mapmaking::MapStruct Maps;
  // Class to hold and populate coadded maps
  mapmaking::CoaddedMapStruct CoaddedMaps;
  // Temporary class for AzTEC data reading & storage
  aztec::BeammapData Data;

  // Number of threads to use for the grppi::farm
  Eigen::Index nThreads = Eigen::nbThreads();

  // std::map for the detector Az/El offsets
  std::map<std::string, Eigen::VectorXd> offsets;

  // std::vectors for cmd line inputs
  std::vector<std::string> config_files;
  std::vector<std::string> input_files;

  auto getInputs(int argc, char *argv[]);
  auto getConfig();
  void getData();
  void setup();
  auto run();
};

// Gets the data and config files specified at the cmd line and
// puts them into std vectors.
auto Lali::getInputs(int argc, char *argv[]){
    using RC = pt::ptree;
    RC rc;
    try {
        po::options_description opts_desc{"Options"};
        opts_desc.add_options()("help,h", "Help screen")(
            "config_file,c", po::value<std::vector<std::string>>()
                                 ->multitoken()
                                 ->zero_tokens()
                                 ->composing())(
            "input_file,i", po::value<std::vector<std::string>>()
                                ->multitoken()
                                ->zero_tokens()
                                ->composing());
        po::positional_options_description inputs_desc{};
        inputs_desc.add("input_file", -1);
        po::command_line_parser parser{argc, argv};
        parser.options(opts_desc).positional(inputs_desc).allow_unregistered();
        po::variables_map vm;
        auto parsed = parser.run();
        po::store(parsed, vm);
        po::notify(vm);

        if (vm.empty() || vm.count("help")) {
            std::cout << opts_desc << '\n';
            return 0;
        }

        if (vm.count("config_file")) {
            config_files = vm["config_file"].as<std::vector<std::string>>();
        }
        SPDLOG_INFO("number of config files: {}", config_files.size());
        for (const auto &f : config_files) {
            SPDLOG_INFO("   {}", f);
        }

        if (vm.count("input_file")) {
            input_files = vm["input_file"].as<std::vector<std::string>>();
        }
        SPDLOG_INFO("number of input files: {}", input_files.size());
        for (const auto &f : input_files) {
            SPDLOG_INFO("   {}", f);
        }

        auto unparsed_opts =
            collect_unrecognized(parsed.options, po::exclude_positional);
        for (const auto &opt : unparsed_opts) {
            SPDLOG_INFO("unparsed options: {}", opt);
        }


    } catch (const po::error &e) {
        SPDLOG_ERROR("{}", e.what());
        return 1;
    } catch (std::runtime_error &e) {
        SPDLOG_ERROR("{}", e.what());
        return 1;
    } catch (...) {
        auto what = boost::current_exception_diagnostic_information();
        SPDLOG_ERROR("unhandled exception: {}, abort", what);
        throw;
    }

    return 0;
 }


 // Gets the config file from getInputs() and creates a config object
auto Lali::getConfig(){

    YAML::Node root;
    try {
        // Generalize to read multiple config files
        root = YAML::LoadFile(config_files.front());
        config = std::move(std::make_shared<lali::YamlConfig>(root));

    } catch (const std::exception& e){
        SPDLOG_ERROR("{}", e.what());
        return 0;
    }

    return 0;
}

// Loads the data from the input data files
void Lali::getData(){
    static auto it = input_files.begin();

    try{
        Data = aztec::BeammapData::fromNcFile(*(it));

        // Set ndetectors and nscans from the data
        ndetectors = Data.meta.template get_typed<int>("ndetectors");
        nscans = Data.meta.template get_typed<int>("nscans");
    }
    catch (const aztec::DataIOError &e) {
        SPDLOG_WARN("failed to read input {}: {}", *it, e.what());
    }
}

// Sets up the map dimensions and lowpass+highpass filter
void Lali::setup() {
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

  // Using the offsets, find the map max and min values and calculate
  // nrows and ncols
  Maps.setRowsCols<mapmaking::Individual>(Maps, Data.telMetaData, offsets, config);
  // Resize the maps to nrows x ncols and set rcphys and ccphys
  Maps.allocateMaps();
}

// Runs the timestream -> map analysis pipeline.
auto Lali::run(){

    auto farm =  grppi::farm(nThreads,[&](auto in) -> TCData<LaliDataKind::PTC,Eigen::MatrixXd> {

        //SPDLOG_INFO("scans before rtcproc {}", in.scans.data);

        /*Stage 1: RTCProc*/
        RTCProc rtcproc(config);
        TCData<LaliDataKind::PTC,Eigen::MatrixXd> out;
        rtcproc.run(in, out, this);

        //SPDLOG_INFO("scans after rtcproc {}", out.scans.data);

        /*Stage 2: PTCProc*/
        PTCProc ptcproc(config);
        ptcproc.run(out, out);

        //SPDLOG_INFO("scans after ptcproc {}", out.scans.data);

        /*Stage 3 Populate Map*/
        Maps.mapPopulate(out, offsets, Data.telMetaData, config);

        SPDLOG_INFO("----------------------------------------------------");
        SPDLOG_INFO("*Done with scan {}...*",out.index.data);
        SPDLOG_INFO("----------------------------------------------------");

        // Return farm object to pipeline
        return out;
    });

    return farm;
}
} //namespace

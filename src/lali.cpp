#include <boost/exception/diagnostic_information.hpp>
#include <boost/program_options.hpp> // po::options_description, po::variables_map, ...
#include <boost/property_tree/ptree.hpp> // pt::ptree
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <Eigen/Dense>
#include <grppi/grppi.h>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <omp.h>

#include "../common_utils/src/utils/enum.h"
#include "../common_utils/src/utils/formatter/enum.h"
#include "../common_utils/src/utils/formatter/matrix.h"
#include "../common_utils/src/utils/formatter/utils.h"
#include "../common_utils/src/utils/logging.h"
#include "../common_utils/src/utils/grppiex.h"
#include "../common_utils/src/utils/mpi.h"

#include "../core/timestream/TCData.h"
#include "../core/timestream/read.h"
#include "../core/timestream/mapresult.h"
#include "../core/map/mapmaking.h"
#include "../core/map/calcpsd.h"
#include "../core/map/maphist.h"

#include "../core/lali.h"
#include "../core/config.h"

#include "/Users/mmccrackan/matplotlib-cpp/matplotlibcpp.h"

#include "../core/map/wiener2.h"

// namespaces
namespace po = boost::program_options;
namespace pt = boost::property_tree;
namespace plt = matplotlibcpp;

using aztec::BeammapData;
using aztec::mapResult;
using aztec::DataIOError;

//TcData is the data structure of which RTCData and PTCData are a part
using timestream::TCData;

//Selects the type of TCData
using timestream::LaliDataKind;

//Command line parser
void po2pt(po::variables_map &in, pt::ptree &out) {

    for (auto it = in.begin(); it != in.end(); ++it) {
        const auto &t = it->second.value().type();
        if ((t == typeid(int)) || (t == typeid(size_t))) {
            out.put<int>(it->first, in[it->first].as<int>());
        } else if ((t == typeid(float)) || (t == typeid(double))) {
            out.put<double>(it->first, in[it->first].as<double>());
        } else if (t == typeid(std::string)) {
            out.put<std::string>(it->first, in[it->first].as<std::string>());
        } else {
            throw std::runtime_error(
                fmt::format("unknown type in config: {}", t.name()));
        }
    }
}


int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
{
    //These scoped_timit expressions time the code within the enclosing brackets
   logging::scoped_timeit timer("lali");

    using RC = pt::ptree;
    RC rc; // stores config
    SPDLOG_INFO("start lali process");
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
        // handle command lines
        if (vm.empty() || vm.count("help")) {
            std::cout << opts_desc << '\n';
            return 0;
        }
        std::vector<std::string> config_files;
        if (vm.count("config_file")) {
            config_files = vm["config_file"].as<std::vector<std::string>>();
        }
        SPDLOG_INFO("number of config files: {}", config_files.size());
        for (const auto &f : config_files) {
            SPDLOG_INFO("   {}", f);
        }
        std::vector<std::string> input_files;
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
        // setup config
        YAML::Node root;
        try {
            root = YAML::LoadFile(config_files.front());
        } catch (const std::exception& e){
            SPDLOG_ERROR("{}", e.what());
            return 0;
        }

        //create a pointer to the config class so it can be shared around
        auto yamlconfig = std::make_shared<lali::YamlConfig>(root);

        //mutexes to prevent race conditions in timestream and mapmaking analysis
        std::mutex farm_mutex;
        std::mutex random_mutex;

        //This is the class that performs all science map mapmaking processes
        lali::laliclass lal(yamlconfig);

        //File iterator.  Out of date.
        static auto it = input_files.begin();

        //Try to get BeammapData
        try{
            lal.bd = BeammapData::fromNcFile(*(it));
        }
        catch (const DataIOError &e) {
            SPDLOG_WARN("failed to read input {}: {}", *it, e.what());
        }

        //Sets up the maps and various variables
        lal.setup();

        //Tells the pipeline how it should be parallelized
        auto ex_name = yamlconfig->get_typed<std::string>("pipeline_ex_policy");

        SPDLOG_INFO("use grppi execution mode: {}", ex_name);

        //Per scan weights for after analysis
        Eigen::MatrixXd scan_wt(lal.nscans,lal.ndet);

        //Start of the pipeline.  This is to be removed.
        grppi::pipeline(grppiex::dyn_ex(ex_name), [&]() -> std::optional<TCData<LaliDataKind::RTC>> {
            //scan index variable
            static auto x = 0;
            //scanlength
            Eigen::Index scanlength = 0;

            //current scan index
            Eigen::Index si = 0;

            //testing
            //lal.nscans = 180;

            while (x != lal.nscans){
                SPDLOG_INFO("On scan {}/{}",x+1,lal.nscans);
                TCData<LaliDataKind::RTC> rtc;

                //first scan index for current scan
                si = lal.bd.scanindex(2,x);
                //get length of current scan
                scanlength = lal.bd.scanindex(3,x) - lal.bd.scanindex(2,x);
                //get reference to BeammapData and put into the current RTC
                rtc.scans.data = lal.bd.scans.block(si,0,scanlength,lal.ndet);
                //Get scan indices and push into current RTC
                rtc.scanindex.data = lal.bd.scanindex.col(x);


                //For testing
                /*si = lal.bd.scanindex(2,0);
                scanlength = 4882;
                rtc.scans.data = lal.bd.scans.block(si,0,scanlength,lal.ndet);
                rtc.scanindex.data = lal.bd.scanindex.col(0);
                rtc.scanindex.data.row(1) = rtc.scanindex.data.row(0).array() + scanlength;
                rtc.scanindex.data.row(3) = rtc.scanindex.data.row(0).array() + scanlength + 32;
                */
                SPDLOG_INFO("RTC {} ", rtc.scans.data);

                x++;
                //return RTC
                return rtc;
                }
                return {};
         },

            //this includes all the processes
             lal.process()
         );

        //It is required to normalize the map after the pipeline is completed since we are streaming it per scan
        mapmaking::mapnormalize(lal);

        //Plotting fun
        /*auto b = cin.get();

        do{
            if (b =='s'){
                Eigen::MatrixXd signalmatrix = Eigen::Map<Eigen::MatrixXd> (lal.br.mapstruct.signal.data(),lal.br.mapstruct.signal.dimension(0),lal.br.mapstruct.signal.dimension(1));
                const int colors = 1;
                Eigen::MatrixXf fff = signalmatrix.cast <float> ();
                float* zptr = &(fff)(0);
                plt::imshow(zptr,lal.br.mapstruct.ncols, lal.br.mapstruct.nrows, colors);
                plt::show();
            }

            if (b=='w'){
                Eigen::MatrixXd signalmatrix = Eigen::Map<Eigen::MatrixXd> (lal.br.mapstruct.wtt.data(),lal.br.mapstruct.wtt.dimension(0),lal.br.mapstruct.wtt.dimension(1));
                const int colors = 1;
                Eigen::MatrixXf fff = signalmatrix.cast <float> ();
                float* zptr = &(fff)(0);
                plt::imshow(zptr,lal.br.mapstruct.ncols, lal.br.mapstruct.nrows, colors);
                plt::show();
            }


            if (b=='n'){
                Eigen::Tensor<double, 2> signaltensor = lal.br.mapstruct.noisemaps.chip(0, 0);
                Eigen::MatrixXd signalmatrix = Eigen::Map<Eigen::MatrixXd> (signaltensor.data(),signaltensor.dimension(0),signaltensor.dimension(1));
                const int colors = 1;
                Eigen::MatrixXf fff = signalmatrix.cast <float> ();
                float* zptr = &(fff)(0);
                plt::imshow(zptr,lal.br.mapstruct.ncols, lal.br.mapstruct.nrows, colors);
                plt::show();
            }

        } while(cin.get()!='\n');
        */

        //Get output file name
        auto output_filepath = yamlconfig->get_typed<std::string>("output_filepath");

        //Calculate Map PSD

        //Define psd class
        psdclass psdc;
        {
            logging::scoped_timeit timer("calcMapPsd");
            auto [psd,psdFreq,psd2d,psd2dFreq] = mapmaking::calcMapPsd(lal.br.mapstruct,0.);

            //need to clean this up
            psdc.psd = std::move(psd);
            psdc.psdFreq = std::move(psdFreq);
            psdc.psd2d = std::move(psd2d);
            psdc.psd2dFreq = std::move(psd2dFreq);
        }

        //Calcluate Map Histogram

        //Define maphistogram class
        maphiststruct mhs;
        {
            logging::scoped_timeit timer("calcMapHistogram");
            auto [histBins,histVals] = mapmaking::calcMapHistogram(lal.br.mapstruct, 200, 0);

            //need to clean this up
            mhs.histBins = std::move(histBins);
            mhs.histVals = std::move(histVals);
        }

        {
            logging::scoped_timeit timer("wiener");

        //Create Wiener filter
        mapmaking::wiener Wiener(lal.br.mapstruct,yamlconfig);
        //Run Wiener filter on coadded maps
        //Wiener.filterCoaddition(lal.br.mapstruct, psdc);

        SPDLOG_INFO("rr {}",Wiener.rr);
        SPDLOG_INFO("vvq {}",Wiener.vvq);
        }

        //Save everything
        SPDLOG_INFO("Saving Maps");
        lal.br.toNcFile(output_filepath + "citlali_maps.nc", lal.bd);
        SPDLOG_INFO("Saving PSD");
        psdc.toNcFile(output_filepath + "psd.nc");
        SPDLOG_INFO("Saving Hists");
        mhs.toNcFile(output_filepath + "hist.nc");


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
    }
    return 0;
}

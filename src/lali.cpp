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
#include "../common_utils/src/utils/eigen.h"

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

#include <string>

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

        SPDLOG_INFO("telraphys {}", lal.bd.telescope_data["TelRaPhys"]);

        //Sets up the maps and various variables
        lal.setup();

        //Tells the pipeline how it should be parallelized
        auto ex_name = yamlconfig->get_typed<std::string>("pipeline_ex_policy");

        SPDLOG_INFO("use grppi execution mode: {}", ex_name);

        //Per scan weights for after analysis
        Eigen::MatrixXd scan_wt(lal.nscans,lal.ndet);

        auto proc = lal.process();

        //Start of the pipeline.  This is to be removed.
        //grppi::pipeline(grppiex::dyn_ex(ex_name), [&]() -> std::optional<TCData<LaliDataKind::RTC,Eigen::Map<Eigen::MatrixXd>>> {
        grppi::pipeline(grppiex::dyn_ex(ex_name), [&]() -> std::optional<TCData<LaliDataKind::RTC,Eigen::MatrixXd>> {
        //scan index variable
            static auto x = 0;
            //scanlength
            Eigen::Index scanlength;

            //current scan index
            Eigen::Index si = 0;

            //testing
            lal.nscans = 180;

            lal.rtc_times.resize(lal.nscans);
            lal.ptc_times.resize(lal.nscans);
            lal.map_times.resize(lal.nscans);

            while (x < lal.nscans){
                SPDLOG_INFO("On scan {}/{}",x+1,lal.nscans);

                //first scan index for current scan
                //si = lal.bd.scanindex(2,x);
                //get length of current scan
                //scanlength = lal.bd.scanindex(3,x) - lal.bd.scanindex(2,x);
                //get reference to BeammapData and put into the current RTC
                //rtc.scans.data = lal.bd.scans.block(si,0,scanlength,lal.ndet);
                //Get scan indices and push into current RTC
                //rtc.scanindex.data = lal.bd.scanindex.col(x);

                //For testing
                si = lal.bd.scanindex(2,0);
                scanlength = 4882;
                //TCData<LaliDataKind::RTC,Eigen::Map<Eigen::MatrixXd>> rtc;
                TCData<LaliDataKind::RTC,Eigen::MatrixXd> rtc;

                rtc.scans.data = lal.bd.scans.block(si,0,scanlength,lal.ndet);
                auto scan_data = lal.bd.scans.block(si,0,scanlength,lal.ndet);
                //new (&rtc.scans.data) Eigen::Map<Eigen::MatrixXd> (scan_data.data(),scanlength,lal.ndet);
                rtc.scanindex.data = lal.bd.scanindex.col(0);
                rtc.scanindex.data.row(1) = rtc.scanindex.data.row(0).array() + scanlength;
                rtc.scanindex.data.row(3) = rtc.scanindex.data.row(0).array() + scanlength + 32;

                rtc.index.data = x;

                x++;
                //return RTC
                return rtc;
                }
                return {};
         },

            //this includes all the processes
             //lal.process()
            proc
         );

        SPDLOG_INFO("rtc_times {}", lal.rtc_times);
        SPDLOG_INFO("ptc_times {}", lal.ptc_times);
        SPDLOG_INFO("map_times {}", lal.map_times);

        SPDLOG_INFO("Timestream rate: {} RTC/s", lal.nscans/(lal.rtc_times.sum()/1000));
        SPDLOG_INFO("PCA rate: {} PTC/s", lal.nscans/(lal.ptc_times.sum()/1000));
        SPDLOG_INFO("Map rate: {} PTC/s", lal.nscans/(lal.map_times.sum()/1000));

        //It is required to normalize the map after the pipeline is completed since we are streaming it per scan
        mapmaking::mapnormalize(lal);

        //Get output file name
        /*auto output_filepath = yamlconfig->get_typed<std::string>("output_filepath");

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



        netCDF::NcFile fo("/Users/mmccrackan/MacanaDevel/aztec_c++/test/coadded_maps/coadded_test.nc", netCDF::NcFile::read);

        auto vars = fo.getVars();

        Eigen::Index nrows = 620;
        Eigen::Index ncols = 552;

        Eigen::MatrixXd signal(nrows,ncols);
        signal.setZero();

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> signal2(nrows,ncols);
        signal2.setZero();

        const auto& sigvar =  vars.find("signal")->second;
        sigvar.getVar(signal.data());

        sigvar.getVar(signal2.data());

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weight(nrows,ncols);
        weight.setZero();

        const auto& wtvar =  vars.find("weight")->second;
        wtvar.getVar(weight.data());

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kernel(nrows,ncols);
        kernel.setZero();

        const auto& kvar =  vars.find("kernel")->second;
        kvar.getVar(kernel.data());

        Eigen::VectorXd rowCoordsPhys(nrows);
        rowCoordsPhys.setZero();

        const auto& rcvar =  vars.find("rowCoordsPhys")->second;
        rcvar.getVar(rowCoordsPhys.data());

        Eigen::VectorXd colCoordsPhys(ncols);
        colCoordsPhys.setZero();

        const auto& ccvar =  vars.find("colCoordsPhys")->second;
        ccvar.getVar(colCoordsPhys.data());

        fo.close();


        std::string psd_path = "/Users/mmccrackan/MacanaDevel/aztec_c++/test/noise_maps/";
        netCDF::NcFile pfo(psd_path + "average_noise_psd.nc", netCDF::NcFile::read);

        auto pvars = pfo.getVars();

        Eigen::Index npsd = 295;

        Eigen::VectorXd psd(npsd);
        psd.setZero();

        const auto& psdvar =  pvars.find("psd")->second;
        psdvar.getVar(psd.data());


        Eigen::VectorXd psdFreq(npsd);
        psdFreq.setZero();

        const auto& psdfvar =  pvars.find("psdFreq")->second;
        psdfvar.getVar(psdFreq.data());

        pfo.close();

        psdc.psd = psd;
        psdc.psdFreq = psdFreq;

        //Eigen::TensorMap<Eigen::Tensor<double, 2>> sig(signal2.data(),nrows,ncols);
        //Eigen::TensorMap<Eigen::Tensor<double, 2>> wt(weight.data(),nrows,ncols);
        //Eigen::TensorMap<Eigen::Tensor<double, 2>> ker(kernel.data(),nrows,ncols);

        lal.br.mapstruct.kernel.resize(nrows,ncols);
        lal.br.mapstruct.signal.resize(nrows,ncols);
        lal.br.mapstruct.wtt.resize(nrows,ncols);
        lal.br.mapstruct.noisemaps.resize(nrows,ncols,5);

        for(int i=0;i<nrows;i++){
            for(int j=0;j<ncols;j++){

                lal.br.mapstruct.kernel(i,j) = kernel(i,j);
                lal.br.mapstruct.wtt(i,j) = weight(i,j);
                lal.br.mapstruct.signal(i,j) = signal2(i,j);

            }
        }

        //lal.br.mapstruct.signal = signal2;
        //lal.br.mapstruct.wtt = weight;
        //lal.br.mapstruct.kernel = kernel;

        lal.br.mapstruct.rowcoordphys = rowCoordsPhys;
        lal.br.mapstruct.colcoordphys = colCoordsPhys;

        lal.br.mapstruct.nrows = nrows;
        lal.br.mapstruct.ncols = ncols;


        std::string npath = "/Users/mmccrackan/MacanaDevel/aztec_c++/test/noise_maps/";
        for(int k =0;k<5;k++){
            netCDF::NcFile nfo(npath + "noise" + std::to_string(k) + ".nc", netCDF::NcFile::read);
            auto nvars = nfo.getVars();
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> nn(nrows,ncols);
            const auto& nvar =  nvars.find("noise")->second;
            nvar.getVar(nn.data());

            for(int i=0;i<nrows;i++){
                for(int j=0;j<ncols;j++){
                    lal.br.mapstruct.noisemaps(i,j,k) = nn(i,j);
                }
             }
        }


        const int colors = 1;
        Eigen::MatrixXf fff = psd.cast <float> ();
        Eigen::MatrixXf freq = psdFreq.cast <float> ();
        float* xptr = &(freq)(0);

        std::vector<double> x(npsd);
        std::vector<double> y(npsd);

        for(int i=0;i<npsd;i++){
            x[i] = psdFreq[i];
            y[i] = psd[i];
        }

        float* yptr = &(fff)(0);
        plt::plot(x,y);
        //plt::imshow(zptr,lal.br.mapstruct.nrows, lal.br.mapstruct.ncols, colors);
        plt::show();



        {
            logging::scoped_timeit timer("wiener");

        //Create Wiener filter
        mapmaking::wiener Wiener(lal.br.mapstruct,yamlconfig);
        //Run Wiener filter on coadded maps
        Wiener.filterCoaddition(lal.br.mapstruct, psdc);
        Wiener.filterNoiseMaps(lal.br.mapstruct);
        }

        //Save everything
        SPDLOG_INFO("Saving Maps");
        lal.br.toNcFile(output_filepath + "citlali_maps.nc", lal.bd);
        SPDLOG_INFO("Saving PSD");
        psdc.toNcFile(output_filepath + "psd.nc");
        SPDLOG_INFO("Saving Hists");
        mhs.toNcFile(output_filepath + "hist.nc");

*/
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

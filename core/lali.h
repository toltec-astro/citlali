#pragma once
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include "timestream/read.h"
#include "timestream/timestream.h"
#include "timestream/mapresult.h"
#include "map/map.h"

using config::Config;
using aztec::mapResult;
using aztec::BeammapData;
using aztec::mapResult;
using aztec::DataIOError;

//declare timestream classes before pipeline
using timestream::TCData;

using timestream::RTCProc;
using timestream::PTCProc;

using timestream::LaliDataKind;
using namespace mapmaking;

namespace lali{

/**
 * @brief This class holds and carries out most of science map making steps. It takes a
 * yaml config class as input. It includes a setup function to create the map buffers
 * and other variables and a process function to run the actual pipeline.
 */
class laliclass {
public:
    laliclass(std::shared_ptr<YamlConfig> config_): config(std::move(config_)) {}
    std::shared_ptr<YamlConfig> config;

    //Random generator for noise maps
    boost::random_device rd;
    boost::random::mt19937 rng{rd};
    boost::random::uniform_int_distribution<> rands{0,1};

    //data class to hold raw data (will be removed)
    BeammapData bd;

    Eigen::VectorXd rtc_times;
    Eigen::VectorXd ptc_times;
    Eigen::VectorXd map_times;

    //Mutexes to prevent race conditions
    std::mutex farm_mutex;
    std::mutex random_mutex;

    //Holds the map result
    mapResult br;

    double samplerate;
    int dsf;

    //Matricies for offsets
    Eigen::MatrixXd offsets;

    //Number of nodes for each stage
    int n, n2, n3;

    //number of detectors and number of scans for later stages
    int ndet, nscans;

    auto getfiles();
    auto setup();
    auto process();

};

//This function sets up variables from the config file and gets things like the offsets
auto laliclass::setup() {
    double pixelsize = config->get_typed<double>("pixel_size");
    double mgrid_0 = config->get_typed<double>("mgrid0");
    double mgrid_1 = config->get_typed<double>("mgrid1");
    int NNoiseMapsPerObs = config->get_typed<int>("noisemaps");

    samplerate = config->get_typed<double>("proc.rtc.samplerate");

    dsf = config->get_typed<int>("proc.rtc.downsample.downsamplefactor");

    //n = config->get_typed<int>("proc.rtc.cores");
    n = Eigen::nbThreads();
    n2 = config->get_typed<int>("proc.ptc.cores");
    n3 = config->get_typed<int>("proc.map.cores");

    ndet = bd.meta.template get_typed<int>("ndetectors");
    nscans = bd.meta.template get_typed<int>("nscans");

    //Setting up map buffer
    br.mapstruct.NNoiseMapsPerObs = NNoiseMapsPerObs;
    br.mapstruct.pixelsize = pixelsize;
    br.mapstruct.mgrid_0 = mgrid_0;
    br.mapstruct.mgrid_1 = mgrid_1;

    //Detector Lat and Lon
    Eigen::VectorXd lat, lon;
    double maxlat = 0;
    double minlat = 0;
    double maxlon = 0;
    double minlon = 0;

    offsets.resize(2,ndet);
    offsets.setZero();

    //Just hard coded for now for simplicity
    offsets.row(0) << -46.7880,
        -22.1615,
        -27.5385,
        -44.1943,
        -52.4330,
        -17.7477,
        -31.8161,
        -26.2618,
        -23.4685,
        -57.9020,
        -34.4158,
        -32.9506,
        -38.4429,
        -53.8667,
        -48.1658,
        -42.9117,
        27.6698,
        17.6056,
        2.1175,
        -12.1645,
        5.4031,
        -1.3428,
        22.2773,
        23.0054,
        0.0000,
        12.0803,
        -10.8672,
        -3.5259,
        -7.7369,
        13.0807,
        7.7058,
        -2.2558,
        -16.5836,
        6.5872,
        3.2281,
        57.5866,
        51.3470,
        46.5800,
        30.6140,
        31.6050,
        15.6096,
        32.5743,
        9.7967,
        52.9672,
        40.9711,
        57.0099,
        61.8633,
        36.1012,
        21.4061,
        26.9416,
        52.1007,
        41.7338,
        37.0830,
        37.8372,
        42.5945,
        47.2382,
        49.6778,
        23.4983,
        28.9190,
        13.8987,
        29.5835,
        14.1927,
        45.8894,
        55.5009,
        18.6960,
        33.8372,
        28.4526,
        24.0219,
        9.0492,
        61.3059,
        25.0032,
        35.3958,
        39.5045,
        34.5794,
        40.2977,
        56.3914,
        50.5016,
        19.0881,
        44.9974,
        -9.2009,
        -27.5735,
        -17.2041,
        -16.2837,
        -1.7072,
        -0.7216,
        13.1131,
        2.3705,
        -11.8413,
        -10.7722,
        12.4788,
        3.8226,
        8.4420,
        -7.0876,
        2.8665,
        18.2610,
        7.7018,
        -6.2725,
        -2.4516,
        -56.8127,
        -49.8685,
        -29.9931,
        -32.3442,
        -52.3440,
        -40.0469,
        -55.2944,
        -59.6142,
        -35.5632,
        -19.9083,
        -47.9394,
        -20.9059,
        -61.0681,
        -37.9667,
        -42.2780;

    offsets.row(1) << 40.9281,
        46.6905,
        38.3251,
        12.8161,
        32.4890,
        37.0918,
        48.2152,
        56.6698,
        28.7097,
        23.7743,
        11.7973,
        29.5689,
        21.3060,
        13.7270,
        22.6426,
        31.0178,
        40.9415,
        41.8794,
        34.9571,
        45.5471,
        8.1813,
        61.7465,
        32.6698,
        50.2471,
        0.0000,
        33.6414,
        63.2222,
        26.5885,
        35.9722,
        51.1805,
        42.9892,
        44.2741,
        55.0812,
        25.6630,
        52.1353,
        22.5333,
        -1.7594,
        6.5067,
        -1.1434,
        15.2782,
        7.7489,
        31.8832,
        -0.6813,
        30.7403,
        -1.5114,
        6.5707,
        -1.7566,
        6.7576,
        15.7883,
        23.8362,
        14.6216,
        14.9405,
        23.3879,
        39.9447,
        31.3342,
        22.9165,
        -34.8072,
        -43.9026,
        -35.0828,
        -26.5522,
        -17.9785,
        -8.9599,
        -9.8874,
        -26.3238,
        -35.1698,
        -43.6042,
        -52.5802,
        -26.4258,
        -17.5126,
        -17.8745,
        -9.2757,
        -9.6317,
        -34.8776,
        -26.4878,
        -17.9941,
        -9.8868,
        -18.0887,
        -17.8379,
        -26.4317,
        -61.3365,
        -43.1756,
        -43.5551,
        -25.4130,
        -35.1422,
        -17.0059,
        -43.9287,
        -61.3283,
        -34.8619,
        -16.6177,
        -61.3027,
        -26.1819,
        -35.1167,
        -43.9529,
        -43.8270,
        -52.6500,
        -52.7389,
        -25.8759,
        -52.7270,
        -22.0529,
        4.1117,
        2.2251,
        -33.8629,
        -31.5614,
        2.9882,
        -4.4984,
        5.2589,
        -6.4893,
        1.2428,
        -40.9366,
        -16.2956,
        -13.5431,
        -42.8164,
        -33.4187;


    //Get max and min lat and lon values out of all detectors.  Maybe parallelize?
    for (int i=0;i<ndet;i++) {
        mapmaking::getPointing(bd.telescope_data, lat, lon, offsets,i);
        if(lat.maxCoeff() > maxlat){
            maxlat = lat.maxCoeff();
        }
        if(lat.minCoeff() < minlat){
            minlat = lat.minCoeff();
        }
        if(lon.maxCoeff() > maxlon){
            maxlon = lon.maxCoeff();
        }
        if(lon.minCoeff() < minlon){
            minlon = lon.minCoeff();
        }
    }

    //Get nrows, ncols
    mapmaking::internal::getRowCol(br.mapstruct, br.mapstruct.mgrid_0, br.mapstruct.mgrid_1, maxlat, minlat, maxlon, minlon);

    //MapStruct class function to resize map tensors.
    br.mapstruct.resize(ndet);

    //Set up noisemap matrices (move to inside mapstruct.resize()
    br.mapstruct.noisemaps.resize(br.mapstruct.NNoiseMapsPerObs,br.mapstruct.nrows,br.mapstruct.ncols);
    br.mapstruct.noisemaps.setZero();

    //Parameter matrix
    br.pp.resize(6,ndet);
    br.pp.setOnes();
}

//This function runs the actual pipeline
auto laliclass::process(){
    ndet = bd.meta.get_typed<int>("ndetectors");
    nscans = bd.meta.get_typed<int>("nscans");

    rtc_times.resize(nscans);
    ptc_times.resize(nscans);
    map_times.resize(nscans);

    auto process = grppi::pipeline(
        //grrppi call to parallelize inputs
        grppi::farm(n,[&](auto in) -> TCData<LaliDataKind::PTC,Eigen::MatrixXd> {
            //create an RTC to hold the reference to BeammapData scans
            RTCProc rtcproc(config);
            //Create a PTC to hold processed data
            TCData<LaliDataKind::PTC,Eigen::MatrixXd> out;
            //Process the data
            {
            logging::scoped_timeit timer("RTCProc",rtc_times.data() + in.index.data);
            SPDLOG_INFO("RTC in {}", in.scans.data);
            rtcproc.process(in,out);
            }

            //Sizes for the kernel
            Eigen::VectorXd beamSigAz(ndet);
            Eigen::VectorXd beamSigEl(ndet);

            //Hard coded for now
            beamSigAz.setConstant(4);
            beamSigEl.setConstant(4);

            //{
            //logging::scoped_timeit timer("makeKernelTimestream");
            for(Eigen::Index det=0;det<ndet;det++) {
                Eigen::VectorXd lat, lon;

                //Map to kernel scan so no copying
                Eigen::Map<Eigen::VectorXd> scans(out.kernelscans.data.col(det).data(),out.kernelscans.data.rows());

                //Need to get pointing for each scan
                mapmaking::getPointing(bd.telescope_data, lat, lon, offsets, det, out.scanindex.data(0), out.scanindex.data(1),dsf);
                //Make the kernel scan
                timestream::makeKernelTimestream(scans,lat,lon,beamSigAz(det),beamSigEl(det));

            }

            TCData<LaliDataKind::PTC,Eigen::MatrixXd> out2;
            PTCProc ptcproc(config);
            {
                logging::scoped_timeit timer("PTCProc",ptc_times.data() + out.index.data);
                //Run PCA clean
                ptcproc.process(out,out2);
                SPDLOG_INFO("PTC in {}", out.scans.data);
            }


            Eigen::VectorXd tmpwt(out2.scans.data.cols());
            //Need to loop through detectors since we are parallelized on scans
            for(int i=0; i<out2.scans.data.cols();i++)
                //Generate weight matrix
                tmpwt[i] = mapmaking::internal::calcScanWeight(out2.scans.data.col(i), out2.flags.data.col(i), samplerate);

            //Random matrix for noisemaps
            Eigen::MatrixXi noisemaps;
            {
                //Do this in a scoped lock to prevent parallelization problems with random number generator
                std::scoped_lock lock(random_mutex);
                noisemaps = Eigen::MatrixXi::Zero(br.mapstruct.NNoiseMapsPerObs,1).unaryExpr([&](int dummy){return rands(rng);});
                noisemaps = (2.*(noisemaps.template cast<double>().array() - 0.5)).template cast<int>();
            }

            {
                //Make the actual science maps
                logging::scoped_timeit timer("generate_scimaps",map_times.data() + out2.index.data);
                mapmaking::generate_scimaps(out2,bd.telescope_data, br.mapstruct.mgrid_0, br.mapstruct.mgrid_1,
                                            samplerate, br.mapstruct, offsets, tmpwt, br.mapstruct.NNoiseMapsPerObs, noisemaps, dsf);
            }

            return out2;
         }));/*,

        grppi::farm(n2,[&](auto in) -> TCData<LaliDataKind::PTC,Eigen::MatrixXd> {
            //Create a TCData with PTC kind to hold output
            TCData<LaliDataKind::PTC,Eigen::MatrixXd> out;
            PTCProc ptcproc(config);
            {
            logging::scoped_timeit timer("PTCProc",ptc_times.data() + in.index.data);
            //Run PCA clean
            ptcproc.process(in,out);
            SPDLOG_INFO("PTC in {}", in.scans.data);
            }
        return out;
        }),

        grppi::farm(n3,[&](auto in) {
            Eigen::VectorXd tmpwt(in.scans.data.cols());
            //Need to loop through detectors since we are parallelized on scans
            for(int i=0; i<in.scans.data.cols();i++)
                //Generate weight matrix
                tmpwt[i] = mapmaking::internal::calcScanWeight(in.scans.data.col(i), in.flags.data.col(i), samplerate);

            //Random matrix for noisemaps
            Eigen::MatrixXi noisemaps;
            {
             //Do this in a scoped lock to prevent parallelization problems with random number generator
            std::scoped_lock lock(random_mutex);
            noisemaps = Eigen::MatrixXi::Zero(br.mapstruct.NNoiseMapsPerObs,1).unaryExpr([&](int dummy){return rands(rng);});
            noisemaps = (2.*(noisemaps.template cast<double>().array() - 0.5)).template cast<int>();
            }

            {
            //Make the actual science maps
            logging::scoped_timeit timer("generate_scimaps",map_times.data() + in.index.data);
            mapmaking::generate_scimaps(in,bd.telescope_data, br.mapstruct.mgrid_0, br.mapstruct.mgrid_1,
                                    samplerate, br.mapstruct, offsets, tmpwt, br.mapstruct.NNoiseMapsPerObs, noisemaps, dsf);
            }
           }));
    */

    //We need to return the actual pipeline here so it will run when the function is called in another pipeline.
    return process;
}
}

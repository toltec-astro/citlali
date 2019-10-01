#pragma once

//Needed for config file input
//Eigen Includes
#include <Eigen/Dense>

//Timestream includes
#include "despike.h"
#include "arraydespike.h"
#include "filter.h"
#include "downsample.h"

//PCA Clean Includes
//#include "cleanpca.h"
#include "cleanpca2.h"
#include "kernel.h"

#include "rtcdata.h"

namespace timestream {

using config::Config;
using timestream::RTCData;
using timestream::LaliDataKind;

/*Class to hold timestream data*/

class RTCData2
{
public:
    RTCData2() {}

    Eigen::MatrixXd scans;
    Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic> flags;
    Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1> scanindex;
};

/*Class to hold processed timestream data*/

class PTCData
{
public:
    PTCData() {}

    Eigen::MatrixXd scans;
    Eigen::MatrixXd kernelscans;
    Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic> flags;
    Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1> scanindex;

    bool isLowpassed = 0;
    bool isDespiked = 0;
};


/*Class to hold kernel timestream data*/
class KTCData
{
public:
    KTCData() {}

    Eigen::MatrixXd kernelscans;
    Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic> kernelflags;
    Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1> kernelindex;
};

/*Class to process timestream data*/

class RTCProc
{
public:
    RTCProc(std::shared_ptr<Config> config_): config(std::move(config_)) {}
    std::shared_ptr<Config> config;
    void process (RTCData<LaliDataKind::SolvedTimeStream>&, PTCData&);
};

 void RTCProc::process(RTCData<LaliDataKind::SolvedTimeStream>& in, PTCData& out) {
     in.flags.data.resize(in.scans.data.rows(),in.scans.data.cols());
     in.flags.data.setOnes();

     auto run_despike = this->config->get_typed<int>("proc.rtc.despike");
     auto run_filter = this->config->get_typed<int>("proc.rtc.filter");
     auto run_downsample = this->config->get_typed<int>("proc.rtc.downsample");

     auto samplerate = this->config->get_typed<double>("proc.rtc.samplerate");
     auto nterms = this->config->get_typed<int>("proc.rtc.filter.nterms");

     //Get reference to inner scan region
     auto inner_scans = in.scans.data.block(nterms,0,in.scans.data.rows() - 2*nterms,in.scans.data.cols());
     auto inner_flags = in.flags.data.block(nterms,0,in.flags.data.rows() - 2*nterms,in.scans.data.cols());

     if (run_despike) {
         auto sigma = this->config->get_typed<double>("proc.rtc.despike.sigma");
         auto despikewindow = this->config->get_typed<int>("proc.rtc.despike.despikewindow");
         auto timeconstant = this->config->get_typed<double>("proc.rtc.despike.timeconstant");

         despike(inner_scans,inner_flags,sigma,samplerate,despikewindow,timeconstant,out.isLowpassed);

         Eigen::VectorXi responsivity(in.scans.data.cols());
         responsivity.setOnes();
         fakeFlaggedData(inner_scans, inner_flags, responsivity);
     }

     if (run_filter) {
         auto flow = this->config->get_typed<double>("proc.rtc.filter.flow");
         auto fhigh = this->config->get_typed<double>("proc.rtc.filter.fhigh");
         auto agibbs = this->config->get_typed<double>("proc.rtc.filter.agibbs");

         timestream::filter<extend>(in.scans.data, flow, fhigh, agibbs, nterms, samplerate);
     }

     if (run_downsample) {
         auto dsf = this->config->get_typed<int>("proc.rtc.downsample.downsamplefactor");
         downsample(inner_scans, inner_flags, in.scanindex.data, out.scans, out.flags, out.scanindex, dsf);
     }

     out.kernelscans.resize(out.scans.rows(),out.scans.cols());
     out.kernelscans.setZero();
     //out.scans.noalias() = inner_scans;
     //out.flags.noalias() = inner_flags;
     out.scanindex.noalias() = in.scanindex.data;
 }

/*Class to clean timestream data*/

class PTCProc
{
public:
    PTCProc(std::shared_ptr<Config> config_): config(std::move(config_)) {}
    std::shared_ptr<Config> config;
    void process (PTCData&,PTCData&);
};

void PTCProc::process(PTCData& in, PTCData& out) {
    auto run_cleanpca = this->config->get_typed<int>("proc.ptc.cleanpca");
    if (run_cleanpca){
        auto neigToCut = this->config->get_typed<int>("proc.ptc.pcaclean.neigToCut");
        auto cutStd = this->config->get_typed<double>("proc.ptc.pcaclean.cutStd");

        pcaclean2<timestream::SpectraBackend>(in.scans,in.kernelscans,in.flags,out.scans,out.kernelscans,neigToCut,cutStd);
    }
    else {
        out.scans = in.scans;
        out.kernelscans = in.kernelscans;
    }
    out.flags.noalias() = in.flags;
    out.scanindex.noalias() = in.scanindex;
}
} //namespace

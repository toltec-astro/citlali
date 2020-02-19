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

#include "TCData.h"

namespace timestream {

using lali::YamlConfig;
using timestream::TCData;
using timestream::LaliDataKind;

/*Class to process timestream data*/

class RTCProc
{
public:
    RTCProc(std::shared_ptr<YamlConfig> config_): config(std::move(config_)) {}
    std::shared_ptr<YamlConfig> config;

    //template <typename RTC_type, typename PTC_type>
    void process (TCData<LaliDataKind::RTC,Eigen::MatrixXd>&, TCData<LaliDataKind::PTC,Eigen::MatrixXd>&);
    bool isLowpassed = 0;
    bool isDespiked = 0;
};

//template <typename RTC_type, typename PTC_type>
void RTCProc::process(TCData<LaliDataKind::RTC,Eigen::MatrixXd>& in, TCData<LaliDataKind::PTC,Eigen::MatrixXd>& out) {
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

         despike(inner_scans,inner_flags,sigma,samplerate,despikewindow,timeconstant,isLowpassed);

         Eigen::VectorXi responsivity(in.scans.data.cols());
         responsivity.setOnes();
         fakeFlaggedData(inner_scans, inner_flags, responsivity);

         isDespiked = 1;
     }

     if (run_filter) {
         auto flow = this->config->get_typed<double>("proc.rtc.filter.flow");
         auto fhigh = this->config->get_typed<double>("proc.rtc.filter.fhigh");
         auto agibbs = this->config->get_typed<double>("proc.rtc.filter.agibbs");

         timestream::filter(in.scans.data, flow, fhigh, agibbs, nterms, samplerate);

         isLowpassed = 1;
     }

     if (run_downsample) {
         auto dsf = this->config->get_typed<int>("proc.rtc.downsample.downsamplefactor");
         downsample(inner_scans, inner_flags, in.scanindex.data, out.scans.data, out.flags.data, out.scanindex.data, dsf);
     }

     else{
         out.scans.data = in.scans.data;
         out.flags.data = in.flags.data;
     }

     //out.scans.data = inner_scans;
     //out.flags.data = inner_flags;
     out.kernelscans.data.resize(out.scans.data.rows(),out.scans.data.cols());
     out.kernelscans.data.setZero();
     out.scanindex.data.noalias() = in.scanindex.data;
     out.index.data = in.index.data;

     SPDLOG_INFO("inner scans final {} {}", inner_scans,in.index.data);
     SPDLOG_INFO("out scan {} {}", out.scans.data,out.index.data);

 }

/*Class to clean timestream data*/

class PTCProc
{
public:
    PTCProc(std::shared_ptr<YamlConfig> config_): config(std::move(config_)) {}
    std::shared_ptr<YamlConfig> config;
    //template <typename PTC_type>
    void process (TCData<LaliDataKind::PTC,Eigen::MatrixXd>&,TCData<LaliDataKind::PTC,Eigen::MatrixXd>&);
};

//template <typename PTC_type>
void PTCProc::process(TCData<LaliDataKind::PTC,Eigen::MatrixXd> &in, TCData<LaliDataKind::PTC,Eigen::MatrixXd> &out) {
    auto run_cleanpca = this->config->get_typed<int>("proc.ptc.cleanpca");
    if (run_cleanpca){
        auto neigToCut = this->config->get_typed<int>("proc.ptc.pcaclean.neigToCut");
        auto cutStd = this->config->get_typed<double>("proc.ptc.pcaclean.cutStd");

        pcaclean2<timestream::SpectraBackend>(in.scans.data,in.kernelscans.data,in.flags.data,out.scans.data,out.kernelscans.data,neigToCut,cutStd);
    }
    else {
        out.scans.data = in.scans.data;
        out.kernelscans.data = in.kernelscans.data;
    }

    //Copy flags and scan indices over to output PTCData
    out.flags.data.noalias() = in.flags.data;
    out.scanindex.data.noalias() = in.scanindex.data;
    out.index.data = in.index.data;
}
} //namespace

#pragma once

#include "timestream_utils.h"

#include "../TCData.h"
#include "../lali.h"
#include "cleanPCA.h"
#include "despike.h"
#include "downsample.h"
#include "filter.h"
#include "kernel.h"
#include "calibrate.h"
#include "polarization.h"

using lali::YamlConfig;

namespace timestream {

class RTCProc {
public:
  RTCProc(YamlConfig config_) : config(std::move(config_)) {}
  YamlConfig config;

  template <class engineType>
  auto run(TCData<LaliDataKind::RTC, Eigen::MatrixXd> &,
           TCData<LaliDataKind::PTC, Eigen::MatrixXd> &,
           engineType);

  template <class engineType>
  auto runDespike(TCData<LaliDataKind::RTC, MatrixXd> &, engineType);

  template <class engineType>
  auto runFilter(TCData<LaliDataKind::RTC, MatrixXd> &, engineType);

  auto runDownsample(TCData<LaliDataKind::RTC, MatrixXd> &,
                     TCData<LaliDataKind::PTC, MatrixXd> &);

  template<class engineType>
  auto runKernel(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &, engineType, YamlConfig);

  auto runCalibration();

  auto runPolarization();
};

// Run the despiker
template <class engineType>
auto RTCProc::runDespike(TCData<LaliDataKind::RTC, MatrixXd> &in, engineType engine) {
  // Get parameters
  auto sigma = config.get_typed<double>(std::tuple{"tod","despike","sigma"});
  auto despikewindow =
      config.get_typed<int>(std::tuple{"tod","despike","despikewindow"});
  auto timeconstant =
      config.get_typed<double>(std::tuple{"tod","despike","timeconstant"});
  auto samplerate = engine->samplerate;

  // Setup despiker with config values
  Despiker despiker(sigma, timeconstant, samplerate, despikewindow);
  // Run despiking
  despiker.despike(in.scans.data, in.flags.data);

  Eigen::VectorXd responsivity(in.scans.data.cols());
  responsivity.setOnes();
  // Replace flagged data with interpolation
  despiker.replaceSpikes(in.scans.data, in.flags.data, responsivity);
}

// Run the lowpass + highpass filter
template <class engineType>
auto RTCProc::runFilter(TCData<LaliDataKind::RTC, MatrixXd> &in, engineType engine) {

  // Get parameters
  auto flow = config.get_typed<double>(std::tuple{"tod","filter","flow"});
  auto fhigh = config.get_typed<double>(std::tuple{"tod","filter","fhigh"});
  auto agibbs = config.get_typed<double>(std::tuple{"tod","filter","agibbs"});

  // Run the Filter class declared in lali.h
  engine->filter.convolveFilter(in.scans.data);
}

// Run the downsampler
 auto RTCProc::runDownsample(TCData<LaliDataKind::RTC, Eigen::MatrixXd> &in,
                            TCData<LaliDataKind::PTC, Eigen::MatrixXd> &out) {
  // Get parameters
  auto dsf =
      config.get_typed<int>(std::tuple{"tod","downsample","downsamplefactor"});

  // Run the downsampler
  downsample(in, out, dsf);
}

// Run the calibrator for the timestreams to get science units
auto RTCProc::runCalibration() {
  // calibrate();
}

// Run the calibrator for the timestreams to get science units
auto RTCProc::runPolarization() {
  // polarizationDemodulation();
}

// Run the kernel to get kernel timestreams.  Called after
// downsampling to save time and memory
template<class engineType>
auto RTCProc::runKernel(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in, engineType engine, YamlConfig config) {
    makeKernel(in, engine->offsets, config);
}

 // The main RTC timestream run function.  It checks the config
 // file for each reduction step and calls the corresponding
 // runStep() function.  If none is requested, set out data
 // to in data
 template <class engineType>
  auto RTCProc::run(TCData<LaliDataKind::RTC, Eigen::MatrixXd> &in,
                   TCData<LaliDataKind::PTC, Eigen::MatrixXd> &out,
                   engineType engine) {


   // Check if despiker is requested and run if so
     if (config.get_typed<bool>(std::tuple{"tod","despike","enabled"})) {
     SPDLOG_INFO("Despiking scan {}...", in.index.data);
     runDespike(in, engine);
   }

   // Check if filter is requested and run if so
   if (config.get_typed<bool>(std::tuple{"tod","filter","enabled"})) {
     SPDLOG_INFO("Filtering scan {}...", in.index.data);
     runFilter(in, engine);
   }

   // Check if downsampler is requested and run if so
   if (config.get_typed<bool>(std::tuple{"tod","downsample","enabled"})) {
     SPDLOG_INFO("Downsampling scan {}...", in.index.data);
     runDownsample(in, out);
   }

   // If downsampler is not run, we need to set out's data
   // to in's data
   else {
     out.scans.data = in.scans.data;
     out.flags.data = in.flags.data;

     out.telLat.data = in.telLat.data;
     out.telLon.data = in.telLon.data;
     out.telElDes.data = in.telElDes.data;
     out.ParAng.data = in.ParAng.data;
   }

   // Run calibration
   SPDLOG_INFO("Calibrating scan {}...", in.index.data);
   runCalibration();

   // Check if kernel is requested and run if so
   if (config.get_typed<bool>(std::tuple{"tod","kernel","enabled"})) {
     SPDLOG_INFO("Generating kernel timestream for scan {}...", in.index.data);
     runKernel(out, engine, config);
   }

   // Set the scan indices and current scan number of out to those of in
   // These are unused elsewhere in this method
   out.scanindex.data = in.scanindex.data;
   out.index.data = in.index.data;
}

} // namespace

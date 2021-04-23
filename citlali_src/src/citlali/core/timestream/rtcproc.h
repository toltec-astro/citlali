#pragma once

#include "timestream_utils.h"

#include "../TCData.h"
#include "../lali.h"
#include "cleanPCA.h"
#include "despike.h"
#include "downsample.h"
#include "filter.h"
#include "kernel.h"

using lali::YamlConfig;

namespace timestream {

class RTCProc {
public:
  RTCProc(YamlConfig config_) : config(std::move(config_)) {}
  YamlConfig config;

  template <class L>
  void run(TCData<LaliDataKind::RTC, Eigen::MatrixXd> &,
           TCData<LaliDataKind::PTC, Eigen::MatrixXd> &,
           L);

  template <class L>
  void runDespike(TCData<LaliDataKind::RTC, MatrixXd> &, L);

  template <class L>
  void runFilter(TCData<LaliDataKind::RTC, MatrixXd> &, L);

  void runDownsample(TCData<LaliDataKind::RTC, MatrixXd> &,
                     TCData<LaliDataKind::PTC, MatrixXd> &);

  template<class L>
  void runKernel(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &, L, YamlConfig);

  void runCalibration();
};

// The main RTC timestream run function.  It checks the config
// file for each reduction step and calls the corresponding
// runStep() function.  If none is requested, set out data
// to in data
template <class L>
void RTCProc::run(TCData<LaliDataKind::RTC, Eigen::MatrixXd> &in,
                  TCData<LaliDataKind::PTC, Eigen::MatrixXd> &out,
                  L LC) {


  // Check if despiker is requested and run if so
    if (config.get_typed<bool>(std::tuple{"tod","despike","enabled"})) {
    SPDLOG_INFO("Despiking scan {}...", in.index.data);
    runDespike(in, LC);
  }

  // Check if filter is requested and run if so
  if (config.get_typed<bool>(std::tuple{"tod","filter","enabled"})) {
    SPDLOG_INFO("Filtering scan {}...", in.index.data);
    runFilter(in, LC);
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
    runKernel(out, LC, config);
  }

  // Set the scan indices and current scan number of out to those of in
  // These are unused elsewhere in this method
  out.scanindex.data = in.scanindex.data;
  out.index.data = in.index.data;
}

// Run the despiker
template <class L>
void RTCProc::runDespike(TCData<LaliDataKind::RTC, MatrixXd> &in, L LC) {
  // Get parameters
  auto sigma = config.get_typed<double>(std::tuple{"tod","despike","sigma"});
  auto despikewindow =
      config.get_typed<int>(std::tuple{"tod","despike","despikewindow"});
  auto timeconstant =
      config.get_typed<double>(std::tuple{"tod","despike","timeconstant"});
  auto samplerate = LC->samplerate;

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
template <class L>
void RTCProc::runFilter(TCData<LaliDataKind::RTC, MatrixXd> &in, L LC) {

  // Get parameters
  auto flow = config.get_typed<double>(std::tuple{"tod","filter","flow"});
  auto fhigh = config.get_typed<double>(std::tuple{"tod","filter","fhigh"});
  auto agibbs = config.get_typed<double>(std::tuple{"tod","filter","agibbs"});

  // Run the Filter class declared in lali.h
  LC->filter.convolveFilter(in.scans.data);
}

// Run the downsampler
void RTCProc::runDownsample(TCData<LaliDataKind::RTC, Eigen::MatrixXd> &in,
                            TCData<LaliDataKind::PTC, Eigen::MatrixXd> &out) {
  // Get parameters
  auto dsf =
      config.get_typed<int>(std::tuple{"tod","downsample","downsamplefactor"});

  // Run the downsampler
  downsample(in, out, dsf);
}

// Run the calibrator for the timestreams to get science units
void RTCProc::runCalibration() {

  // calibrate();
}

// Run the kernel to get kernel timestreams.  Called after
// downsampling to save time and memory
template<class L>
void RTCProc::runKernel(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in, L LC, YamlConfig config) {
    makeKernel(in, LC->offsets, config);
}
} // namespace

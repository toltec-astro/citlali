#pragma once

#include "timestream_utils.h"

#include "cleanPCA.h"
#include "despike.h"
#include "filter.h"
#include "downsample.h"
#include "kernel.h"

using lali::YamlConfig;

namespace timestream {

class RTCProc {
public:
  RTCProc(std::shared_ptr<YamlConfig> config_) : config(std::move(config_)) {}
  std::shared_ptr<YamlConfig> config;

  template <class L>
  void run(TCData<LaliDataKind::RTC, Eigen::MatrixXd> &,
           TCData<LaliDataKind::PTC, Eigen::MatrixXd> &,
           L);
  void runDespike(TCData<LaliDataKind::RTC, MatrixXd> &);

  template <class L>
  void runFilter(TCData<LaliDataKind::RTC, MatrixXd> &, L);

  void runDownsample(TCData<LaliDataKind::RTC, MatrixXd> &,
                     TCData<LaliDataKind::PTC, MatrixXd> &);

  template<class L>
  void runKernel(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &, L, std::shared_ptr<YamlConfig>);

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
  if (this->config->get_typed<int>("proc.rtc.despike")) {
    SPDLOG_INFO("Despiking scan {}...", in.index.data);
    runDespike(in);
  }

  // Check if filter is requested and run if so
  if (this->config->get_typed<int>("proc.rtc.filter")) {
    SPDLOG_INFO("Filtering scan {}...", in.index.data);
    runFilter(in, LC);
  }

  // Check if downsampler is requested and run if so
  if (this->config->get_typed<int>("proc.rtc.downsample")) {
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
  if (this->config->get_typed<int>("proc.rtc.kernel")) {
    SPDLOG_INFO("Generating kernel timestream for scan {}...", in.index.data);
    runKernel(out, LC, this->config);
  }

  // Set the scan indices and current scan number of out to those of in
  // These are unused elsewhere in this method
  out.scanindex.data = in.scanindex.data;
  out.index.data = in.index.data;
}

// Run the despiker
void RTCProc::runDespike(TCData<LaliDataKind::RTC, MatrixXd> &in) {
  // Get parameters
  auto sigma = this->config->get_typed<double>("proc.rtc.despike.sigma");
  auto despikewindow =
      this->config->get_typed<int>("proc.rtc.despike.despikewindow");
  auto timeconstant =
      this->config->get_typed<double>("proc.rtc.despike.timeconstant");
  auto samplerate = this->config->get_typed<double>("proc.rtc.samplerate");

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
  auto flow = this->config->get_typed<double>("proc.rtc.filter.flow");
  auto fhigh = this->config->get_typed<double>("proc.rtc.filter.fhigh");
  auto agibbs = this->config->get_typed<double>("proc.rtc.filter.agibbs");

  // Run the Filter class declared in lali.h
  LC->filter.convolveFilter(in.scans.data);
}

// Run the downsampler
void RTCProc::runDownsample(TCData<LaliDataKind::RTC, Eigen::MatrixXd> &in,
                            TCData<LaliDataKind::PTC, Eigen::MatrixXd> &out) {
  // Get parameters
  auto dsf =
      this->config->get_typed<int>("proc.rtc.downsample.downsamplefactor");

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
void RTCProc::runKernel(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in, L LC, std::shared_ptr<YamlConfig> config) {
    makeKernel(in, LC->offsets, config);
}

class PTCProc {
public:
  PTCProc(std::shared_ptr<YamlConfig> config_) : config(std::move(config_)) {}
  std::shared_ptr<YamlConfig> config;

  // template <typename Derived>
  void run(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &,
           TCData<LaliDataKind::PTC, Eigen::MatrixXd> &);

  void runClean(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &,
                TCData<LaliDataKind::PTC, Eigen::MatrixXd> &);

  void getWeights(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &);
};

// The main PTC timestream run function.  It checks the config
// file for each reduction step and calls the corresponding
// runStep() function.  If none is requested, set out data
// to in data
// template <typename Derived>
void PTCProc::run(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in,
                  TCData<LaliDataKind::PTC, Eigen::MatrixXd> &out) {
  if (this->config->get_typed<int>("proc.ptc.clean")) {
    SPDLOG_INFO("Cleaning signal and kernel timestreams for scan {}...",
                in.index.data);
    //Run clean
    runClean(in, out);
  }

  // If we don't clean, set out to in.  Not necessary, but done to be safe
  else {
    out = in;
  }

  // Get scan weights
  getWeights(out);
}

//run the PCA cleaner
void PTCProc::runClean(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in,
                       TCData<LaliDataKind::PTC, Eigen::MatrixXd> &out) {

  // Get parameters
  auto neigToCut = this->config->get_typed<int>("proc.ptc.pcaclean.neigToCut");
  auto cutStd = this->config->get_typed<double>("proc.ptc.pcaclean.cutStd");

  // Setup PCA cleaner class
  pcaCleaner cleaner(neigToCut, cutStd);

  // Calculate the eigenvalues from the scan signal timestream
  cleaner.calcEigs<SpectraBackend>(in.scans.data, in.flags.data);

  // Remove neigToCut eigenvalues from scan signal timestream
  cleaner.removeEigs<SpectraBackend, DataType>(cleaner.det, out.scans.data);

  // We don't need det anymore, so delete it to save some space
  cleaner.det.resize(0, 0);

  // Check if kernel is requested
  if (this->config->get_typed<int>("proc.rtc.kernel")) {
      // Remove neigToCut eigenvalues from scan kernel timestream
      cleaner.removeEigs<SpectraBackend, KernelType>(in.kernelscans.data,
                                                     out.kernelscans.data);
  }
}

// Get the weights of each scan and replace outliers
void PTCProc::getWeights(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in) {

  // Scan weight is a vector that is ndetectors long
  in.weights.data = Eigen::VectorXd::Zero(in.scans.data.cols());

  // This is for approximate weights
  if (this->config->get_typed<int>("proc.ptc.approximateWeights")) {
    SPDLOG_INFO("Using Approximate Weights for scan {}...", in.index.data);
  }

  // Calculating the weights for each detector for this scan
  else {
    SPDLOG_INFO("Calculating Weights for scan {}...", in.index.data);
    for (Eigen::Index det = 0; det < in.scans.data.cols(); det++) {

      // Make Eigen::Maps for each detector's scan
      Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
          in.flags.data.col(det).data(), in.flags.data.rows());
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
          in.scans.data.col(det).data(), in.scans.data.rows());

      // Get standard deviation excluding flagged samples
      auto [tmp, ngood] = timestream_utils::stddev(scans, flags);

      // Check for NaNs and too short scans
      if (tmp != tmp ||
          ngood < this->config->get_typed<double>("proc.rtc.samplerate")) {
        in.weights.data(det) = 0.0;
      } else {
        tmp = pow(tmp, -2.0);
        // Check if weight is too large and normalize if so
        if (tmp > 2.0 * 3.95274e+09) {
          in.weights.data(det) = 3.95274e+09 / 2.0;
        } else {
          in.weights.data(det) = tmp;
        }
      }
    }
  }
}

} // namespace timestream

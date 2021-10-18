#pragma once

#include "timestream_utils.h"

#include "../TCData.h"
#include "cleanPCA.h"

using tula::config::YamlConfig;

namespace timestream {

class PTCProc {
public:
  PTCProc(YamlConfig config_) : config(std::move(config_)) {}
  YamlConfig config;

  // template <typename Derived>
  template <class engineType>
  auto run(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &,
           TCData<LaliDataKind::PTC, Eigen::MatrixXd> &, engineType);

  auto runClean(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &,
                TCData<LaliDataKind::PTC, Eigen::MatrixXd> &,
                std::vector<std::tuple<int,int>> &);

  template< class engineType>
  auto getWeights(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &, engineType);
};

//run the PCA cleaner
auto PTCProc::runClean(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in,
                       TCData<LaliDataKind::PTC, Eigen::MatrixXd> &out,
                       std::vector<std::tuple<int,int>> &array_index) {

  // Get parameters
  auto neigToCut = config.get_typed<int>(std::tuple{"tod","pcaclean","neigToCut"});
  auto cutStd = config.get_typed<double>(std::tuple{"tod","pcaclean","cutStd"});

  auto map_count = array_index.size();

  if (out.scans.data.isZero(0)){
      out.scans.data.resize(in.scans.data.rows(), in.scans.data.cols());

      if (config.get_typed<bool>(std::tuple{"tod","kernel","enabled"})) {
          out.kernelscans.data.resize(in.kernelscans.data.rows(), in.kernelscans.data.cols());
      }
  }

  auto run_kernel = config.get_typed<bool>(std::tuple{"tod","kernel","enabled"});

  for (Eigen::Index mc=0;mc<map_count;mc++){

      auto det = std::get<0>(array_index.at(mc));
      auto ndetectors = std::get<1>(array_index.at(mc)) - std::get<0>(array_index.at(mc));

      Eigen::Ref<Eigen::MatrixXd> in_scans_block = in.scans.data.block(0,det,in.scans.data.rows(),ndetectors);
      Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<> >
         in_scans(in_scans_block.data(), in_scans_block.rows(), in_scans_block.cols(),
                  Eigen::OuterStride<>(in_scans_block.outerStride()));

      Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags_block = in.flags.data.block(0,det,in.flags.data.rows(),ndetectors);
      Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>, 0, Eigen::OuterStride<> >
         in_flags(in_flags_block.data(), in_flags_block.rows(), in_flags_block.cols(),
                  Eigen::OuterStride<>(in_flags_block.outerStride()));

      Eigen::Ref<Eigen::MatrixXd> out_scans_block = out.scans.data.block(0,det,out.scans.data.rows(),ndetectors);
      Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<> >
         out_scans(out_scans_block.data(), out_scans_block.rows(), out_scans_block.cols(),
                  Eigen::OuterStride<>(out_scans_block.outerStride()));

      // Setup PCA cleaner class
      pcaCleaner cleaner(neigToCut, cutStd);

      // Calculate the eigenvalues from the scan signal timestream
      cleaner.calcEigs<SpectraBackend>(in_scans, in_flags);

      // Remove neigToCut eigenvalues from scan signal timestream
      cleaner.removeEigs<SpectraBackend, DataType>(cleaner.det, out_scans);

      // We don't need det anymore, so delete it to save some space
      cleaner.det.resize(0, 0);

      // Check if kernel is requested
      if (run_kernel) {
          // Remove neigToCut eigenvalues from scan kernel timestream

          Eigen::Ref<Eigen::MatrixXd> in_kernelscans_block = in.kernelscans.data.block(0,det,in.kernelscans.data.rows(),ndetectors);
          Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<> >
             in_kernelscans(in_kernelscans_block.data(), in_kernelscans_block.rows(), in_kernelscans_block.cols(),
                      Eigen::OuterStride<>(in_kernelscans_block.outerStride()));

          Eigen::Ref<Eigen::MatrixXd> out_kernelscans_block = out.kernelscans.data.block(0,det,out.kernelscans.data.rows(),ndetectors);
          Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<> >
             out_kernelscans(out_kernelscans_block.data(), out_kernelscans_block.rows(), out_kernelscans_block.cols(),
                      Eigen::OuterStride<>(out_kernelscans_block.outerStride()));

          cleaner.removeEigs<SpectraBackend, KernelType>(in_kernelscans,
                                                         out_kernelscans);
      }
   }
}

// Get the weights of each scan and replace outliers
template <class engineType>
auto PTCProc::getWeights(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in, engineType engine) {

  // Scan weight is a vector that is ndetectors long
  in.weights.data = Eigen::VectorXd::Zero(in.scans.data.cols());

  // This is for approximate weights
  if (config.get_typed<bool>(std::tuple{"tod","pcaclean","approximateWeights"})) {
    SPDLOG_INFO("Using Approximate Weights for scan {}...", in.index.data);

    // temporary set sensitivity vector to 1
    Eigen::VectorXd sens(in.weights.data.size());
    sens.setOnes();
    in.weights.data = pow(sqrt(engine->samplerate)*sens.array(),-2.0);
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
          ngood < engine->samplerate || tmp == 0) {
        in.weights.data(det) = 0.0;
      } else {
        tmp = pow(tmp, -2.0);
        // Check if weight is too large and normalize if so (temporary solution)
        if (tmp > 2.0 * 3.95274e+09) {
            in.weights.data(det) = tmp;
            //in.weights.data(det) = 3.95274e+09 / 2.0;
        } else {
          in.weights.data(det) = tmp;
        }
      }
    }
  }
}

// The main PTC timestream run function.  It checks the config
// file for each reduction step and calls the corresponding
// runStep() function.  If none is requested, set out data
// to in data
template <class engineType>
auto PTCProc::run(TCData<LaliDataKind::PTC, Eigen::MatrixXd> &in,
                  TCData<LaliDataKind::PTC, Eigen::MatrixXd> &out, engineType engine) {
  if (config.get_typed<bool>(std::tuple{"tod","pcaclean","enabled"})) {
    SPDLOG_INFO("Cleaning signal and kernel timestreams for scan {}...",
                in.index.data);
    //Run clean
    runClean(in, out, engine->array_index);
  }

  // If we don't clean, set out to in.  Not necessary, but done to be safe
  else {
    out = in;
  }

  // Get scan weights
  getWeights(out, engine);
}

} // namespace timestream

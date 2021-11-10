#pragma once

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/utils/utils.h>

namespace timestream {


class PTCProc {
public:
    template <class Engine>
    void run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             Engine);
};

template <class Engine>
void PTCProc::run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
         TCData<TCDataKind::PTC, Eigen::MatrixXd> &out,
         Engine engine) {

    // run timestream cleaning
     if (engine->run_clean) {

         // need this if out != in
         if (out.scans.data.isZero(0)){
             out.scans.data.resize(in.scans.data.rows(), in.scans.data.cols());

             if (engine->run_kernel) {
                 out.kernel_scans.data.resize(in.kernel_scans.data.rows(), in.kernel_scans.data.cols());
             }
         }

         // loop through the arrays
         for (Eigen::Index mi=0; mi<engine->array_indices.size(); mi++) {

             // current detector
             auto det = std::get<0>(engine->array_indices.at(mi));
             // size of block for each grouping
             auto ndet = std::get<1>(engine->array_indices.at(mi)) - std::get<0>(engine->array_indices.at(mi)) + 1;

             // get the block of in scans that corresponds to the current array
             Eigen::Ref<Eigen::MatrixXd> in_scans_block = in.scans.data.block(0,det,in.scans.data.rows(),ndet);

             Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>
                     in_scans(in_scans_block.data(), in_scans_block.rows(), in_scans_block.cols(),
                         Eigen::OuterStride<>(in_scans_block.outerStride()));

             // get the block of in flags that corresponds to the current array
             Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags_block =
                     in.flags.data.block(0,det,in.flags.data.rows(),ndet);

             Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>, 0, Eigen::OuterStride<> >
                in_flags(in_flags_block.data(), in_flags_block.rows(), in_flags_block.cols(),
                         Eigen::OuterStride<>(in_flags_block.outerStride()));

             // get the block of out scans that corresponds to the current array
             Eigen::Ref<Eigen::MatrixXd> out_scans_block = out.scans.data.block(0,det,out.scans.data.rows(),ndet);

             Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<> >
                out_scans(out_scans_block.data(), out_scans_block.rows(), out_scans_block.cols(),
                         Eigen::OuterStride<>(out_scans_block.outerStride()));

             // calculate the eigenvalues from the signal timestream
             SPDLOG_INFO("calculating eigenvalues for scan {}", in.index.data);
             auto [det_scan, evals, evecs] = engine->cleaner.template calcEigs<SpectraBackend>(in_scans, in_flags);

             // remove eigenvalues from signal timestream
             SPDLOG_INFO("removing eigenvalues from scan {}", in.index.data);
             engine->cleaner.template removeEigs<SpectraBackend, DataType>(det_scan, out_scans,
                     evals, evecs);

             // we don't need det anymore, so delete it to save some space
             //engine->cleaner.det.resize(0, 0);

             if (engine->run_kernel) {
                 // get the block of in kernel that corresponds to the current array
                 Eigen::Ref<Eigen::MatrixXd> in_kernel_scans_block =
                         in.kernel_scans.data.block(0,det,in.kernel_scans.data.rows(),ndet);

                 Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<> >
                    in_kernel_scans(in_kernel_scans_block.data(), in_kernel_scans_block.rows(), in_kernel_scans_block.cols(),
                             Eigen::OuterStride<>(in_kernel_scans_block.outerStride()));

                 // get the block of out kernel that corresponds to the current array
                 Eigen::Ref<Eigen::MatrixXd> out_kernel_scans_block =
                         out.kernel_scans.data.block(0,det,out.kernel_scans.data.rows(),ndet);

                 Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<> >
                    out_kernel_scans(out_kernel_scans_block.data(), out_kernel_scans_block.rows(), out_kernel_scans_block.cols(),
                             Eigen::OuterStride<>(out_kernel_scans_block.outerStride()));

                 // remove kernel scan eigenvalues
                 SPDLOG_INFO("removing kernel eigenvalues from scan {}", in.index.data);
                 engine->cleaner.template removeEigs<SpectraBackend, KernelType>(in_kernel_scans,
                                                                out_kernel_scans, evals, evecs);
             }
         }
     }

     else {
         out = in;
     }

     // calculate approximate weights from sensitivity and sample rate
     if (engine->approx_weights == "approximate") {
         out.weights.data = pow(sqrt(engine->fsmp)*engine->sensitivity.array(),-2.0);
     }

     // get weights from stddev of each scan for each detector
     else {
         out.weights.data = Eigen::VectorXd::Zero(out.scans.data.cols());
         for (Eigen::Index i=0; i<out.scans.data.cols(); i++) {

             // Make Eigen::Maps for each detector's scan
             Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                         out.flags.data.col(i).data(), out.flags.data.rows());
             Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                         out.scans.data.col(i).data(), out.scans.data.rows());

             // Get standard deviation excluding flagged samples
             auto [tmp, ngood] = engine_utils::stddev(scans, flags);
             if (tmp != tmp || ngood < engine->fsmp || tmp == 0) {
                 out.weights.data(i) = 0.0;
             }
             else {
                 out.weights.data(i) = pow(tmp, -2.0);
             }
         }
     }
}

} // namespace
#pragma once

#include <cmath>
#include <memory>
#include <mutex>

#include <Eigen/Sparse>
#include <fftw3.h>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/pointing.h>

using timestream::TCData;

// selects the type of TCData
using timestream::TCDataKind;

namespace mapmaking {

class ConjugateGradient {

};

class MLMapmaker {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
    std::unique_ptr<std::mutex> ml_mutex = std::make_unique<std::mutex>();

    double tolerance;
    int max_iterations;

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename calib_t>
    void populate_maps_ml(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                            Eigen::DenseBase<Derived> &, std::string &, calib_t &, double, bool, bool);
};

template<class map_buffer_t, typename Derived, typename calib_t>
void MLMapmaker::populate_maps_ml(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, map_buffer_t &omb, map_buffer_t &cmb,
                                  Eigen::DenseBase<Derived> &map_indices, std::string &pixel_axes, calib_t &calib, double d_fsmp,
                                  bool run_omb, bool run_noise) {

    const bool use_cmb = !cmb.noise.empty();
    const bool use_omb = !omb.noise.empty();
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();

    Eigen::Index n_pixels = omb.n_rows * omb.n_cols;

    for (Eigen::Index arr=0; arr<calib.n_arrays; ++arr) {
        logger->info("making map for array {}/{}",arr + 1,calib.n_arrays);
        auto array = calib.arrays[arr];
        Eigen::Index map_index = arr;

        // start indices for current array
        Eigen::Index start = std::get<0>(calib.array_limits[array]);
        // end indices for current array
        Eigen::Index end = std::get<1>(calib.array_limits[array]);

        // hold the values for the pointing matrix and timestream vectors.
        const Eigen::Index n_pts_max = (end - start) * in.scans.data.rows();
        std::vector<Eigen::Triplet<double>> triplet_list;
        triplet_list.reserve(n_pts_max);
        std::vector<double> b_vals;
        b_vals.reserve(n_pts_max);
        std::vector<double> b2_vals;
        if (run_kernel) {
            b2_vals.reserve(n_pts_max);
        }

        const double row_center = (omb.n_rows - 1) / 2.0;
        const double col_center = (omb.n_cols - 1) / 2.0;

        for (Eigen::Index i=start; i<end; ++i) {
            auto det_index = i;
            if (calib.apt["flag"](det_index)==0) {
                // get detector pointing
                auto [lat,lon] = engine_utils::calc_det_pointing(in.tel_data.data, calib.apt["x_t"](det_index), calib.apt["y_t"](det_index),
                                                                 pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

                // get map buffer row and col indices for lat and lon vectors
                Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + row_center;
                Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + col_center;

                // loop through current detector chunk
                for (Eigen::Index j=0; j<in.scans.data.rows(); ++j) {
                    // skip flagged samples
                    if (in.flags.data(j,i)) {
                        continue;
                    }
                    Eigen::Index omb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                    Eigen::Index omb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));
                    if ((omb_ir < 0) || (omb_ir >= omb.n_rows) || (omb_ic < 0) || (omb_ic >= omb.n_cols)) {
                        continue;
                    }
                    Eigen::Index index = omb.n_rows * omb_ic + omb_ir;
                    Eigen::Index row = static_cast<Eigen::Index>(b_vals.size());
                    // get pointing matrix value
                    triplet_list.push_back(Eigen::Triplet<double>(row,index,in.weights.data(i)));
                    b_vals.push_back(in.scans.data(j,i)*in.weights.data(i));
                    if (run_kernel) {
                        b2_vals.push_back(in.kernel.data(j,i)*in.weights.data(i));
                    }
                }
            }
        }

        Eigen::Index n_pts = static_cast<Eigen::Index>(b_vals.size());
        logger->info("start {} end {} valid_n_pts {} n_pixels {} n_rows {}", start, end, n_pts, n_pixels, in.scans.data.rows());
        if (n_pts == 0) {
            logger->warn("no unflagged in-bounds samples for array {}; skipping ML solve", array);
            continue;
        }

        // signal and kernel timestreams
        Eigen::VectorXd b = Eigen::Map<const Eigen::VectorXd>(b_vals.data(), n_pts);
        Eigen::VectorXd b2;
        if (run_kernel) {
            b2 = Eigen::Map<const Eigen::VectorXd>(b2_vals.data(), n_pts);
        }
        // pointing matrix
        Eigen::SparseMatrix<double> A(n_pts,n_pixels);

        // initialize sparse matrix
        A.setFromTriplets(triplet_list.begin(), triplet_list.end());

        logger->info("running conjugate gradient for array {}/{}", arr + 1, calib.n_arrays);

        //Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double> > cg;
        cg.setMaxIterations(max_iterations);
        cg.setTolerance(tolerance);

        // compute pointing matrix
        cg.compute(A);

        // solve for signal map
        auto signal_x = cg.solve(b).eval();
        logger->info("signal iterations {}",cg.iterations());
        logger->info("signal error {}",cg.error());

        Eigen::VectorXd kernel_x;
        if (run_kernel) {
            // solve for kernel map
            kernel_x = cg.solve(b2).eval();
            logger->info("kernel iterations {}",cg.iterations());
            logger->info("kernel error {}",cg.error());
        }

        Eigen::VectorXd ones = Eigen::VectorXd::Ones(n_pts);
        // solve for weight map
        auto weight_x = cg.solve(ones).eval();
        logger->info("weight iterations {}",cg.iterations());
        logger->info("weight error {}",cg.error());

        // protect shared map accumulation when scans are processed concurrently.
        {
            std::scoped_lock<std::mutex> lk(*ml_mutex);
            omb.signal[map_index] += Eigen::Map<const Eigen::MatrixXd>(signal_x.data(),omb.n_rows, omb.n_cols);
            if (run_kernel) {
                omb.kernel[map_index] += Eigen::Map<const Eigen::MatrixXd>(kernel_x.data(),omb.n_rows, omb.n_cols);
            }
            omb.weight[map_index] += Eigen::Map<const Eigen::MatrixXd>(weight_x.data(),omb.n_rows, omb.n_cols);
        }

        logger->info("signal[{}] {}",map_index,omb.signal[map_index]);
        if (run_kernel) {
            logger->info("kernel[{}] {}",map_index,omb.kernel[map_index]);
        }
        logger->info("weight[{}] {}",map_index,omb.weight[map_index]);
    }

    // free fftw vectors
    /*fftw_free(fftw_a);
    fftw_free(fftw_b);
    // destroy fftw plan
    fftw_destroy_plan(pf);*/
}
} // namespace mapmaking

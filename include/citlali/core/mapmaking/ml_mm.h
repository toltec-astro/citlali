// pragma once

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

    double tolerance;
    int max_iterations;

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename calib_t>
    void populate_maps_ml(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                            Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string &,
                            calib_t &, double, bool, bool);
};

template<class map_buffer_t, typename Derived, typename calib_t>
void MLMapmaker::populate_maps_ml(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, map_buffer_t &omb, map_buffer_t &cmb,
                                  Eigen::DenseBase<Derived> &map_indices, Eigen::DenseBase<Derived> &det_indices,
                                  std::string &pixel_axes, calib_t &calib, double d_fsmp, bool run_omb, bool run_noise) {

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

        // number of good detectors
        Eigen::Index n_good_det = (calib.apt["flag"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                          std::get<1>(calib.array_limits[array])-1)).array()==0).count();

        //for (Eigen::Index i=start; i<end; ++i) {
        //    if ((in.flags.data.col(i).array() ==1).all()) {
        //        n_good_det--;
        //    }
        //}

        // total number of data points
        Eigen::Index n_pts = n_good_det*in.scans.data.rows();

        logger->info("start {} end {} n_pts {} n_pixels {} n_good_det {} n_rows {}", start, end, n_pts, n_pixels, n_good_det, in.scans.data.rows());

        // signal and kernel timestreams
        Eigen::VectorXd b(n_pts), b2(n_pts);
        // pointing matrix
        Eigen::SparseMatrix<double> A(n_pts,n_pixels);

        // hold the values for the pointing matrix
        std::vector<Eigen::Triplet<double>> triplet_list;
        triplet_list.reserve(n_pts);

        // keep track of what index next detector starts at
        Eigen::Index k = 0;

        for (Eigen::Index i=start; i<end; ++i) {
            auto det_index = det_indices(i);
            if (calib.apt["flag"](det_index)==0) {// && !((in.flags.data.col(i).array()==1).all())) {
                // get detector pointing
                auto [lat,lon] = engine_utils::calc_det_pointing(in.tel_data.data, calib.apt["x_t"](det_index), calib.apt["y_t"](det_index),
                                                                 pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

                // get map buffer row and col indices for lat and lon vectors
                Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows)/2.;
                Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols)/2.;

                // loop through current detector chunk
                for (Eigen::Index j=0; j<in.scans.data.rows(); ++j) {
                    Eigen::Index omb_ir = omb_irow(j);
                    Eigen::Index omb_ic = omb_icol(j);
                    Eigen::Index index = omb.n_rows * omb_ic + omb_ir;
                    // get pointing matrix value
                    triplet_list.push_back(Eigen::Triplet<double>(j + k,index,in.weights.data(i)));
                }
                // get signal values
                b.segment(k,in.scans.data.rows()) = in.scans.data.col(i)*in.weights.data(i);

                if (run_kernel) {
                    // get kernel values
                    b2.segment(k,in.scans.data.rows()) = in.kernel.data.col(i)*in.weights.data(i);
                }
                // move onto the next detector
                k += in.scans.data.rows();
            }
        }

        // initialize sparse matrix
        A.setFromTriplets(triplet_list.begin(), triplet_list.end());

        logger->info("running conjugate gradient for array {}/{}",arr + 1,calib.n_arrays);

        //Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double> > cg;
        cg.setMaxIterations(max_iterations);
        cg.setTolerance(tolerance);

        // compute pointing matrix
        cg.compute(A);

        // solve for signal map
        auto x = cg.solve(b).eval();
        // populate signal map
        omb.signal[map_index] += Eigen::Map<Eigen::MatrixXd>(x.data(),omb.n_rows, omb.n_cols);
        logger->info("signal iterations {}",cg.iterations());
        logger->info("signal error {}",cg.error());
        logger->info("signal[{}] {}",map_index,omb.signal[map_index]);

        if (run_kernel) {
            // solve for kernel map
            x = cg.solve(b2).eval();
            // populate kernel map
            omb.kernel[map_index] += Eigen::Map<Eigen::MatrixXd>(x.data(),omb.n_rows, omb.n_cols);
            logger->info("kernel iterations {}",cg.iterations());
            logger->info("kernel error {}",cg.error());
            logger->info("kernel[{}] {}",map_index,omb.signal[map_index]);
        }

        b2.setOnes();
        // solve for weight map
        x = cg.solve(b2).eval();
        // populate weight map
        omb.weight[map_index] += Eigen::Map<Eigen::MatrixXd>(x.data(),omb.n_rows, omb.n_cols);
        logger->info("weight iterations {}",cg.iterations());
        logger->info("weight error {}",cg.error());
        logger->info("weight[{}] {}",map_index,omb.signal[map_index]);
    }

    // free fftw vectors
    /*fftw_free(fftw_a);
    fftw_free(fftw_b);
    // destroy fftw plan
    fftw_destroy_plan(pf);*/
}
} // namespace mapmaking

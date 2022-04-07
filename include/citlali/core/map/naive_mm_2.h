#pragma once

#include <cstddef>
#include <Eigen/Core>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <omp.h>

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/utils/pointing.h>

#include <thread>
#include <time.h>

using timestream::TCData;
using timestream::TCDataKind;

using map_count_t = std::size_t;
using det_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;

template <typename Derived, class Engine>
void populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &map_index_vector,
                         Eigen::DenseBase<Derived> &det_index_vector, Engine engine) {
    SPDLOG_INFO("populating map with scan {}", in.index.data);

    Eigen::MatrixXi noise_rand;

    Eigen::Index ndets = in.scans.data.cols();

    if (engine->run_coadd) {
        if (engine->run_noise) {
            // declare random number generator
            thread_local boost::random::mt19937 eng;
            boost::random::uniform_int_distribution<> rands{0,1};

            // generate the random number matrix
            noise_rand =
                Eigen::MatrixXi::Zero(engine->cmb.nnoise, ndets).unaryExpr([&](int dummy){return rands(eng);});
            noise_rand = (2.*(noise_rand.template cast<double>().array() - 0.5)).template cast<int>();
        }
    }


    // loop through the detector indices
    for (Eigen::Index mi = 0; mi < ndets; mi++) {

        Eigen::Index mc = 0;
        Eigen::Index di = det_index_vector(mi);

        if (engine->reduction_type == "science" || engine->reduction_type == "pointing") {
            mc = map_index_vector(mi);
        }

        else if (engine->reduction_type == "beammap") {
            mc = mi;
        }

        Eigen::Index noise_det = 0;

        // current detector offsets
        double azoff, eloff;

        // if in science/pointing mode, get offsets from apt table
        if (engine->reduction_type == "science" || engine->reduction_type == "pointing") {
            azoff = engine->calib_data["x_t"](di);
            eloff = engine->calib_data["y_t"](di);
        }

        // else if in beammap mode, offsets are zero
        else if (engine->reduction_type == "beammap") {
            azoff = 0;
            eloff = 0;
        }

        // get detector pointing (lat/lon = rows/cols -> icrs or altaz)
        auto [lat, lon] = engine_utils::get_det_pointing(in.tel_meta_data.data, azoff, eloff, engine->map_type);

        // get map buffer row and col indices for lat and lon vectors
        Eigen::VectorXd mb_irow = lat.array()/engine->pixel_size + (engine->mb.nrows)/2.;
        Eigen::VectorXd mb_icol = lon.array()/engine->pixel_size + (engine->mb.ncols)/2.;

        Eigen::VectorXd cmb_irow, cmb_icol;
        if (engine->run_coadd) {
            // get coadded map buffer row and col indices for lat and lon vectors
            cmb_irow = lat.array()/engine->pixel_size + (engine->cmb.nrows)/2.;
            cmb_icol = lon.array()/engine->pixel_size + (engine->cmb.ncols)/2.;
        }

        // loop through scan
        for (Eigen::Index si = 0; si < in.scans.data.rows(); si++) {
            // row and col pixel for map buffer
            Eigen::Index mb_ir = (mb_irow(si));
            Eigen::Index mb_ic = (mb_icol(si));

            Eigen::Index cmb_ir, cmb_ic;
            if (engine->run_coadd) {
                // row and col pixel for coadded map buffer
                cmb_ir = (cmb_irow(si));
                cmb_ic = (cmb_icol(si));
            }

            // check if sample is flagged as good
            if (in.flags.data(si, mi)) {
                auto sig = in.scans.data(si, mi)*in.weights.data(mi);

                if ((mb_ir >= 0) && (mb_ir < engine->mb.nrows) && (mb_ic >= 0) && (mb_ic < engine->mb.ncols)) {
                    // weight map
                    engine->mb.weight.at(mc)(mb_ir,mb_ic) += in.weights.data(mi);

                    // signal map
                    engine->mb.signal.at(mc)(mb_ir,mb_ic) += sig;

                    // check if kernel map is requested
                    if (engine->run_kernel) {
                        // kernel map
                        auto ker = in.kernel_scans.data(si, mi)*in.weights.data(mi);
                        engine->mb.kernel.at(mc)(mb_ir,mb_ic) += ker;
                    }

                    // coverage map if not in beammap mode
                    if (engine->reduction_type != "beammap") {
                        // coverage map
                        engine->mb.coverage.at(mc)(mb_ir,mb_ic) += 1./engine->dfsmp;
                    }
                }

                // noise maps
                if (engine->run_coadd) {
                    // check if noise maps requested
                    if (engine->run_noise) {
                        // loop through noise maps
                        for (Eigen::Index nn=0; nn<engine->cmb.nnoise; nn++) {
                            // coadd into current noise map
                            if ((cmb_ir >= 0) && (cmb_ir < engine->cmb.nrows) && (cmb_ic >= 0) && (cmb_ic < engine->cmb.ncols)) {
                                engine->cmb.noise.at(mc)(cmb_ir,cmb_ic,nn) += noise_rand(nn, noise_det)*sig;
                            }
                        }
                    }
                }
            }
        }
        noise_det++;
    }
}

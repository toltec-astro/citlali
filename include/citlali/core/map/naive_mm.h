#pragma once

#include <cstddef>
#include <Eigen/Core>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <omp.h>

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/utils/pointing.h>

using timestream::TCData;
using timestream::TCDataKind;

using map_count_t = std::size_t;
using det_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;

template <class Engine>
void populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, Engine engine) {
    SPDLOG_INFO("populating map with scan {}", in.index.data);

    Eigen::MatrixXi noise_rand;

    if (engine->run_noise) {
        // declare random number generator for each thread
        boost::random::mt19937 eng(omp_get_thread_num());
        boost::random::uniform_int_distribution<> rands{0,1};

        // generate the random number matrix
        noise_rand =
                Eigen::MatrixXi::Zero(engine->cmb.nnoise,1).unaryExpr([&](int dummy){return rands(eng);});
        noise_rand = (2.*(noise_rand.template cast<double>().array() - 0.5)).template cast<int>();
    }

    // loop through the detector indices
    for (Eigen::Index mi = 0; mi < engine->det_indices.size(); mi++) {
        for (Eigen::Index di = std::get<0>(engine->det_indices.at(mi)); di < std::get<1>(engine->det_indices.at(mi)); di++) {

            // current detector's offsets
            double azoff, eloff;

            // if in science/pointing mode, get offsets from apt table
            if (engine->reduction_type == "science" || engine->reduction_type == "pointing") {
                azoff = engine->calib_data["x_t"](di);
                eloff = engine->calib_data["y_t"](di);
            }

            // if in beammap mode, offsets are zero
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
                if (in.flags.data(si, di)) {
                    // weight map
                    engine->mb.weight.at(mi)(mb_ir,mb_ic) += in.weights.data(di);

                    // signal map
                    auto sig = in.scans.data(si, di)*in.weights.data(di);
                    engine->mb.signal.at(mi)(mb_ir,mb_ic) += sig;

                    // check if kernel map is requested
                    if (engine->run_kernel) {
                        // kernel map
                        auto ker = in.kernel_scans.data(si, di)*in.weights.data(di);
                        engine->mb.kernel.at(mi)(mb_ir,mb_ic) += ker;
                    }

                    // coverage map if not in beammap mode
                    if (engine->reduction_type != "beammap") {
                        // coverage map
                        engine->mb.coverage.at(mi)(mb_ir,mb_ic) += 1./engine->fsmp;
                    }

                    // noise maps
                    if (engine->run_noise) {
                        for (Eigen::Index nn=0; nn<engine->cmb.nnoise; nn++) {
                            engine->cmb.noise.at(mi)(cmb_ir,cmb_ic,nn) += noise_rand(nn)*sig;
                        }
                    }
                }
            }
        }
    }
}

#pragma once

#include <cstddef>
#include <Eigen/Core>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <omp.h>

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/utils/pointing.h>

#include <citlali/core/utils/utils.h>

#include <thread>
#include <time.h>

using timestream::TCData;
using timestream::TCDataKind;

using map_count_t = std::size_t;
using det_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;

// jinc function
auto jinc_func(double r, double a, double b, double c, double r_max, double l_d_i) {
    r = r/l_d_i;
    auto arg0 = 2*boost::math::cyl_bessel_j(1, 2*pi*r/a)/(2*pi*r/a);
    auto arg1 = exp(-pow(2*r/b,c));
    auto arg2 = 2*boost::math::cyl_bessel_j(1,3.831706*r/r_max)/(3.831706*r/r_max);

    return arg0*arg1*arg2;
}

template <typename Derived, class Engine>
void populate_maps_jinc(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                         Eigen::DenseBase<Derived> &map_index_vector,
                         Eigen::DenseBase<Derived> &det_index_vector,
                         Engine engine) {

    SPDLOG_INFO("rad {} jw {}", engine->radius, engine->jinc_weights);
    engine_utils::SplineFunction jinc_spline(engine->radius, engine->jinc_weights);

    // jinc filter parameters
    double r_max = 1.5;
    double a = 1.1;
    double b = 4.75;
    double c = 2.;

    // lambda over diameter
    Eigen::VectorXd l_d(3);
    l_d << (1.1/1000)/50, (1.4/1000)/50, (2.0/1000)/50;

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

        SPDLOG_INFO("DET {}", mi);

        Eigen::Index mc;
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

        // loop through scan
        for (Eigen::Index si = 0; si < in.scans.data.rows(); si++) {
            // loop through rows and cols
            // check if sample is flagged as good
            if (in.flags.data(si, mi)) {
                // sig x weight
                double sig = in.scans.data(si, mi)*in.weights.data(mi);
                double ker = 0;
                if (engine->run_kernel) {
                    // kernel map
                    ker = in.kernel_scans.data(si, mi)*in.weights.data(mi);
                }

                for (Eigen::Index i=0; i<engine->mb.nrows; i++) {
                    for (Eigen::Index j=0; j<engine->mb.ncols; j++) {

                        // calculate distance
                        auto r = sqrt(pow((lat(si) - engine->mb.rcphys(i)),2) + pow((lon(si) - engine->mb.ccphys(j)),2));
                        if (r/l_d(mc) < r_max) {

                            //SPDLOG_INFO("i {} j {} si {} r {}", i, j, si, r);

                            // calculate Jinc weight for pixel
                            auto jinc_weight = jinc_func(r,a,b,c,r_max,l_d(mc));
                            //auto jinc_weight = jinc_spline(r/l_d(mc));

                            //SPDLOG_INFO("r {} r/ld {} jw {}", r, r/l_d(mi), jinc_weight);

                            if ((i >= 0) && (i < engine->mb.nrows) && (j >= 0) && (j < engine->mb.ncols)) {
                                // weight map
                                engine->mb.weight.at(mc)(i,j) += in.weights.data(mi)*jinc_weight;

                                // signal map
                                engine->mb.signal.at(mc)(i,j) += sig*jinc_weight;

                                // check if kernel map is requested
                                if (engine->run_kernel) {
                                    // kernel map
                                    engine->mb.kernel.at(mc)(i,j) += ker*jinc_weight;
                                }

                                // coverage map if not in beammap mode
                                if (engine->reduction_type != "beammap") {
                                    // coverage map
                                    engine->mb.coverage.at(mc)(i,j) += 1./engine->dfsmp*jinc_weight;
                                }
                            }
                        }
                    }
                }
            }

            // noise maps
            if (engine->run_coadd) {
                // check if noise maps requested
                if (engine->run_noise) {
                    if (in.flags.data(si, mi)) {
                        // sig x weight
                        auto sig = in.scans.data(si, mi)*in.weights.data(mi);

                        for (Eigen::Index i=0; i<engine->cmb.nrows; i++) {
                            for (Eigen::Index j=0; j<engine->cmb.ncols; j++) {
                                // check if sample is flagged as good
                                auto r = sqrt(pow((lat(si) - engine->cmb.rcphys(i)),2) + pow((lon(si) - engine->cmb.ccphys(j)),2));
                                if (r/l_d(mi)<r_max) {
                                    // calculate Jinc weight for pixel
                                    auto jinc_weight = jinc_func(r,a,b,c,r_max,l_d(mc));

                                    // loop through noise maps
                                    for (Eigen::Index nn=0; nn<engine->cmb.nnoise; nn++) {
                                        // coadd into current noise map
                                        if ((i >= 0) && (i < engine->cmb.nrows) && (j >= 0) && (j < engine->cmb.ncols)) {
                                            engine->cmb.noise.at(mc)(i,j,nn) += noise_rand(nn, noise_det)*sig*jinc_weight;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        noise_det++;
    }
}

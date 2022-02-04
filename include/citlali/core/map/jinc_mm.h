#pragma once

#include <cstddef>
#include <Eigen/Core>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <omp.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/timestream/timestream.h>
#include <citlali/core/utils/pointing.h>

using timestream::TCData;
using timestream::TCDataKind;

using map_count_t = std::size_t;
using det_indices_t = std::vector<std::tuple<Eigen::Index, Eigen::Index>>;

auto jinc_func(double r, double a, double b, double c, double r_max, double l_d_i) {
    r = r/l_d_i;
    auto arg0 = 2*boost::math::cyl_bessel_j(1, 2*pi*r/a)/(2*pi*r/a);
    auto arg1 = exp(-pow(2*r/b,c));
    auto arg2 = 2*boost::math::cyl_bessel_j(1,3.831706*r/r_max)/(3.831706*r/r_max);

    return arg0*arg1*arg2;
}

template <class Engine>
void populate_maps_jinc(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, Engine engine) {
    SPDLOG_INFO("populating map with scan {}", in.index.data);

    double r_max = 3;
    double a = 1.1;
    double b = 4.75;
    double c = 2.;

    Eigen::VectorXd l_d(3);
    l_d << (1.1/1000)/50, (1.4/1000)/50, (2.0/1000)/50;

    // loop through the detector indices
    for (Eigen::Index mi = 0; mi < engine->det_indices.size(); mi++) {
        for (Eigen::Index di = std::get<0>(engine->det_indices.at(mi)); di < std::get<1>(engine->det_indices.at(mi)); di++) {
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

            SPDLOG_INFO("DI {}",di);

            // loop through scan
            for (Eigen::Index si = 0; si < in.scans.data.rows(); si++) {
                int blah = 0;
                for (Eigen::Index i=0; i<engine->mb.nrows; i++) {
                    for (Eigen::Index j=0; j<engine->mb.ncols; j++) {
                        auto r = sqrt(pow((lat(si) - engine->mb.rcphys(i)),2) + pow((lon(si) - engine->mb.ccphys(j)),2));

                        if (r/l_d(mi)<r_max) {
                            blah++;
                            SPDLOG_INFO("r {} di {} mi {} si {}",r/l_d(mi), di, mi, si);
                            auto sig = in.scans.data(si, di)*in.weights.data(di);
                            auto jinc_weight = jinc_func(r,a,b,c,r_max,l_d(mi));

                            // weight map
                            engine->mb.weight.at(mi)(i,j) += in.weights.data(di)*jinc_weight;

                            // signal map
                            engine->mb.signal.at(mi)(i,j) += sig*jinc_weight;

                            // check if kernel map is requested
                            if (engine->run_kernel) {
                                engine->mb.kernel.at(mi)(i,j) += in.kernel_scans.data(si,di)*jinc_weight;
                            }
                        }
                    }
                }
                SPDLOG_INFO("blah {}", blah);
            }
        }
    }
}

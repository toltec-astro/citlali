#pragma once

#include <thread>

#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include <unsupported/Eigen/Splines>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/pointing.h>


using timestream::TCData;

// selects the type of TCData
using timestream::TCDataKind;

namespace mapmaking {

// jinc function
auto jinc_func(double r, double a, double b, double c, double r_max, double l_d) {
    r = r/l_d;
    auto arg0 = 2*boost::math::cyl_bessel_j(1, 2*pi*r/a)/(2*pi*r/a);
    auto arg1 = exp(-pow(2*r/b,c));
    auto arg2 = 2*boost::math::cyl_bessel_j(1,3.831706*r/r_max)/(3.831706*r/r_max);

    return arg0*arg1*arg2;
}

template<class map_buffer_t, typename Derived, typename apt_t, typename pointing_offset_t>
void populate_maps_jinc(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                        map_buffer_t &omb, map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices, Eigen::DenseBase<Derived> &det_indices,
                        std::string &pixel_axes, std::string &redu_type, apt_t &apt,
                        pointing_offset_t &pointing_offsets_arcsec, double d_fsmp, bool run_noise,
                        double r_max, double a, double b, double c) {

    // lambda over diameter
    std::map<Eigen::Index,double> l_d;
    l_d[0] = (1.1/1000)/50;
    l_d[1] = (1.4/1000)/50;
    l_d[2] = (2.0/1000)/50;

    auto radius = Eigen::VectorXd::LinSpaced(1000, 1e-10, r_max);
    std::map<Eigen::Index,Eigen::VectorXd> jinc_weights;

    for (const auto &ld: l_d) {
        jinc_weights[ld.first].resize(radius.size());
        Eigen::Index j = 0;
        for (const auto &r: radius) {
            jinc_weights[ld.first](j) = jinc_func(r,a,b,c,r_max,ld.second);
            j++;
        }
    }

    std::map<Eigen::Index, engine_utils::SplineFunction2> jinc_splines;

    for (const auto &ld: l_d) {
        engine_utils::SplineFunction2 s;
        s.interpolate(radius, jinc_weights[ld.first]);
        jinc_splines[ld.first] = s;
    }

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // pointer to map buffer with noise maps
    ObsMapBuffer* nmb = NULL;

    // matrix to hold random noise value
    Eigen::MatrixXi noise;

    if (run_noise) {
        // set pointer to cmb if it has noise maps
        if (!cmb.noise.empty()) {
            nmb = &cmb;
        }

        // otherwise set pointer to omb if it has noise maps
        else if (!omb.noise.empty()) {
            nmb = &omb;
        }

        // declare random number generator
        thread_local boost::random::mt19937 eng;

        // boost random number generator (0,1)
        boost::random::uniform_int_distribution<> rands{0,1};

        // rescale random values to -1 or 1
        noise =
            Eigen::MatrixXi::Zero(nmb->n_noise, n_pts).unaryExpr([&](int dummy){return rands(eng);});
        noise = (2.*(noise.template cast<double>().array() - 0.5)).template cast<int>();
    }

    for (Eigen::Index i=0; i<n_dets; i++) {
        // skip completely flagged detectors
        if ((in.flags.data.col(i).array()==true).any()) {
            double az_off = 0;
            double el_off = 0;

            // get detector positions from apt table if not in detector mapmaking mode
            if (omb.map_grouping!="detector" || redu_type!="beammap") {
                auto det_index = det_indices(i);
                az_off = apt["x_t"](det_index);
                el_off = apt["y_t"](det_index);
            }

            // which map to assign detector to
            Eigen::Index map_index = map_indices(i);

            // get detector pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off,
                                                              pixel_axes, pointing_offsets_arcsec);

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (!cmb.noise.empty()) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols)/2.;
            }

            double r_max_pix = r_max*l_d[apt["array"](det_indices(i))]/omb.pixel_size_rad;

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; j++) {
                // check if sample is flagged, ignore if so
                if (in.flags.data(j,i)) {
                    Eigen::Index omb_ir = omb_irow(j);
                    Eigen::Index omb_ic = omb_icol(j);

                    // signal map value
                    double signal;

                    // make sure the data point is within the map
                    if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {
                        // find minimum row
                        auto row_min = std::max(0.0, omb_ir - r_max_pix);
                        // find maximum row
                        auto row_max = std::min(omb.n_rows - 1.0, omb_ir + r_max_pix + 1);

                        // find minimum col
                        auto col_min = std::max(0.0, omb_ic - r_max_pix);
                        // find maximum col
                        auto col_max = std::min(omb.n_cols - 1.0, omb_ic + r_max_pix + 1);

                        //SPDLOG_INFO("omb_ir {}, omb_ic {}, r_max_pix {}, r_max {}", omb_ir, omb_ic, r_max_pix, r_max);
                        //SPDLOG_INFO("row_min {}, row_max {}, col_min {}, col_max {}", row_min, row_max, col_min, col_max);

                        // loop through nearby rows and cols
                        for (Eigen::Index r=row_min; r<row_max; r++) {
                            for (Eigen::Index c=col_min; c<col_max; c++) {
                                // distance from current sample to pixel
                                auto radius = sqrt(std::pow(lat(j) - omb.rows_tan_vec(r),2) + std::pow(lon(j) - omb.cols_tan_vec(c),2));
                                //SPDLOG_INFO("radius {}", radius);
                                if (radius<r_max*l_d[map_index]) {
                                    // jinc weighting function
                                    auto jinc_weight = jinc_func(radius,a,b,c,r_max,l_d[map_index]);
                                    //auto jinc_weight = jinc_splines[map_index](radius);

                                    // populate signal map
                                    signal = in.scans.data(j,i)*in.weights.data(i)*jinc_weight;
                                    omb.signal[map_index](r,c) += signal;

                                    // populate weight map
                                    omb.weight[map_index](r,c) += in.weights.data(i)*jinc_weight;

                                    // populate kernel map
                                    if (!omb.kernel.empty()) {
                                        auto kernel = in.kernel.data(j,i)*in.weights.data(i)*jinc_weight;
                                        omb.kernel[map_index](r,c) += kernel;
                                    }

                                    // populate coverage map
                                    if (!omb.coverage.empty()) {
                                        omb.coverage[map_index](r,c) += (jinc_weight/d_fsmp);
                                    }
                                }
                            }
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        Eigen::Index nmb_ir, nmb_ic;

                        // if coaddition is enabled
                        if (!cmb.noise.empty()) {
                            nmb_ir = cmb_irow(j);
                            nmb_ic = cmb_icol(j);
                        }

                        // else make noise maps for obs
                        else if (!omb.noise.empty()) {
                            nmb_ir = omb_irow(j);
                            nmb_ic = omb_icol(j);
                        }

                        // loop through noise maps
                        //for (Eigen::Index nn=0; nn<nmb->n_noise; nn++) {
                        // coadd into current noise map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                            // find minimum row
                            auto row_min = std::max(0.0,nmb_ir - r_max*l_d[map_index]/nmb->pixel_size_rad);
                            // find maximum row
                            auto row_max = std::min(nmb->n_rows - 1.0 ,nmb_ir + r_max*l_d[map_index]/nmb->pixel_size_rad + 1);

                            // find minimum col
                            auto col_min = std::max(0.0,nmb_ic - r_max*l_d[map_index]/nmb->pixel_size_rad);
                            // find maximum col
                            auto col_max = std::min(nmb->n_cols - 1.0 ,nmb_ic + r_max*l_d[map_index]/nmb->pixel_size_rad + 1);

                            // loop through nearby rows and cols
                            for (Eigen::Index r=row_min; r<row_max; r++) {
                                for (Eigen::Index c=col_min; c<col_max; c++) {
                                    // distance from current sample to pixel
                                    auto radius = sqrt(std::pow(lat(j) - nmb->rows_tan_vec(r),2) + std::pow(lon(j) - nmb->cols_tan_vec(c),2));
                                    //SPDLOG_INFO("radius {}", radius);
                                    if (radius<r_max*l_d[map_index]) {
                                        // jinc weighting function
                                        auto jinc_weight = jinc_func(radius,a,b,c,r_max,l_d[map_index]);
                                        signal = in.scans.data(j,i)*in.weights.data(i)*jinc_weight;
                                        for (Eigen::Index nn=0; nn<nmb->n_noise; nn++) {
                                            nmb->noise[map_index](r,c,nn) += noise(nn,j)*signal;
                                        }
                                    }
                                }
                            }
                        }
                        //}
                    }
                }
            }
        }
    }
}
} // namespace mapmaking

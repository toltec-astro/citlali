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

    if (r!=0) {
        r = r/l_d;
        auto arg0 = 2*boost::math::cyl_bessel_j(1,2*pi*r/a)/(2*pi*r/a);
        auto arg1 = exp(-pow(2*r/b,c));
        auto arg2 = 2*boost::math::cyl_bessel_j(1,3.831706*r/r_max)/(3.831706*r/r_max);

        return arg0*arg1*arg2;
    }
    else {
        return 1.0;
    }
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

    std::map<Eigen::Index,Eigen::VectorXd> jinc_weights;
    std::map<Eigen::Index, engine_utils::SplineFunction2> jinc_splines;

    for (const auto &ld: l_d) {
        auto radius = Eigen::VectorXd::LinSpaced(1000, 0, r_max*ld.second);
        jinc_weights[ld.first].resize(radius.size());
        Eigen::Index j = 0;
        for (const auto &r: radius) {
            jinc_weights[ld.first](j) = jinc_func(r,a,b,c,r_max,ld.second);
            j++;
        }
        engine_utils::SplineFunction2 s;
        s.interpolate(radius, jinc_weights[ld.first]);
        jinc_splines[ld.first] = s;
    }

    std::map<Eigen::Index,Eigen::MatrixXd> jinc_weights_mat;
    for (const auto &ld: l_d) {
        double r_max_pix = std::floor(r_max*ld.second/omb.pixel_size_rad);
        Eigen::VectorXd pixels = Eigen::VectorXd::LinSpaced(2*r_max_pix + 1,-r_max_pix, r_max_pix);

        jinc_weights_mat[ld.first].setZero(2*r_max_pix + 1,2*r_max_pix + 1);

        for (Eigen::Index i=0; i<pixels.size(); i++) {
            for (Eigen::Index j=0; j<pixels.size(); j++) {
                double r = omb.pixel_size_rad*sqrt(pow(pixels(i),2) + pow(pixels(j),2));
                jinc_weights_mat[ld.first](i,j) = jinc_func(r,a,b,c,r_max,ld.second);
            }
        }
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
        if ((in.flags.data.col(i).array()==0).any()) {
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

            Eigen::Index r_max_pix = std::floor(r_max*l_d[apt["array"](det_indices(i))]/omb.pixel_size_rad);

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; j++) {
                // check if sample is flagged, ignore if so
                if (!in.flags.data(j,i)) {
                    Eigen::Index omb_ir = omb_irow(j);
                    Eigen::Index omb_ic = omb_icol(j);

                    // signal map value
                    double signal;

                    // make sure the data point is within the map
                    if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {
                        // find minimum row
                        /*auto row_min = std::max(static_cast<Eigen::Index>(0), omb_ir - r_max_pix);
                        // find maximum row
                        auto row_max = std::min(omb.n_rows, omb_ir + r_max_pix);

                        // find minimum col
                        auto col_min = std::max(static_cast<Eigen::Index>(0), omb_ic - r_max_pix);
                        // find maximum col
                        auto col_max = std::min(omb.n_cols, omb_ic + r_max_pix);

                        // loop through nearby rows and cols
                        for (Eigen::Index r=row_min; r<row_max+1; r++) {
                            for (Eigen::Index c=col_min; c<col_max+1; c++) {
                                // distance from current sample to pixel
                                auto radius = sqrt(std::pow(lat(j) - omb.rows_tan_vec(r),2) + std::pow(lon(j) - omb.cols_tan_vec(c),2));
                                //SPDLOG_INFO("radius {}", radius);
                                if (radius<r_max*l_d[apt["array"](det_indices(i))]) {
                                    // jinc weighting function
                                    //auto jinc_weight = jinc_func(radius,a,b,c,r_max,l_d[apt["array"](det_indices(i))]);
                                    auto jinc_weight = jinc_splines[apt["array"](det_indices(i))](radius);
                                    auto weight = in.weights.data(i)*jinc_weight;

                                    // populate signal map
                                    signal = in.scans.data(j,i)*weight;//in.weights.data(i)*jinc_weight;
                                    omb.signal[map_index](r,c) += signal;

                                    // populate weight map
                                    omb.weight[map_index](r,c) += in.weights.data(i);//in.weights.data(i)*jinc_weight;

                                    // populate kernel map
                                    if (!omb.kernel.empty()) {
                                        auto kernel = in.kernel.data(j,i)*weight;//in.weights.data(i)*jinc_weight;
                                        omb.kernel[map_index](r,c) += kernel;
                                    }

                                    // populate coverage map
                                    if (!omb.coverage.empty()) {
                                        omb.coverage[map_index](r,c) += (jinc_weight/d_fsmp);
                                    }
                                }
                            }
                        }*/

                        // loop through nearby rows and cols
                        for (Eigen::Index r=-r_max_pix; r<r_max_pix+1; r++) {
                            for (Eigen::Index c=-r_max_pix; c<r_max_pix+1; c++) {
                                // distance from current sample to pixel
                                //auto radius = sqrt(std::pow(lat(j) - omb.rows_tan_vec(r),2) + std::pow(lon(j) - omb.cols_tan_vec(c),2));
                                //SPDLOG_INFO("radius {}", radius);
                                //if (radius<r_max*l_d[apt["array"](det_indices(i))]) {
                                // jinc weighting function

                                Eigen::Index ri = omb_ir + r;
                                Eigen::Index ci = omb_ic + c;

                                if (ri > 0 && ci > 0 && ri < omb.n_rows && ci < omb.n_cols) {

                                    Eigen::Index ji = r_max_pix + r;
                                    Eigen::Index jj = r_max_pix + c;

                                    auto jinc_weight = jinc_weights_mat[apt["array"](det_indices(i))](ji,jj);
                                    auto weight = in.weights.data(i)*jinc_weight;

                                    // populate signal map
                                    signal = in.scans.data(j,i)*weight;//in.weights.data(i)*jinc_weight;
                                    omb.signal[map_index](ri,ci) += signal;

                                    // populate weight map
                                    omb.weight[map_index](ri,ci) += weight;//in.weights.data(i)*jinc_weight;

                                    // populate kernel map
                                    if (!omb.kernel.empty()) {
                                        auto kernel = in.kernel.data(j,i)*weight;//in.weights.data(i)*jinc_weight;
                                        omb.kernel[map_index](ri,ci) += kernel;
                                    }

                                    // populate coverage map
                                    if (!omb.coverage.empty()) {
                                        omb.coverage[map_index](ri,ci) += (jinc_weight/d_fsmp);
                                    }
                                }
                            //}
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

                            double r_max_pix = std::floor(r_max*l_d[apt["array"](det_indices(i))]/nmb->pixel_size_rad);

                            // find minimum row
                            auto row_min = std::max(0.0,nmb_ir - r_max_pix);
                            // find maximum row
                            auto row_max = std::min(nmb->n_rows - 1.0 ,nmb_ir + r_max_pix + 1);

                            // find minimum col
                            auto col_min = std::max(0.0,nmb_ic - r_max_pix);
                            // find maximum col
                            auto col_max = std::min(nmb->n_cols - 1.0 ,nmb_ic + r_max_pix + 1);

                            // loop through nearby rows and cols
                            for (Eigen::Index r=row_min; r<row_max; r++) {
                                for (Eigen::Index c=col_min; c<col_max; c++) {
                                    // distance from current sample to pixel
                                    auto radius = sqrt(std::pow(lat(j) - nmb->rows_tan_vec(r),2) + std::pow(lon(j) - nmb->cols_tan_vec(c),2));
                                    //SPDLOG_INFO("radius {}", radius);
                                    if (radius<r_max*l_d[map_index]) {
                                        // jinc weighting function
                                        auto jinc_weight = jinc_func(radius,a,b,c,r_max,l_d[apt["array"](det_indices(i))]);
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

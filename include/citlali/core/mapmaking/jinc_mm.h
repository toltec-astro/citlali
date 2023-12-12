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

class JincMapmaker {
public:
    // run polarization?
    bool run_polarization;

    // parallel policy
    std::string parallel_policy;

    // method to calculate jinc weights
    std::string mode = "matrix";

    // lambda over diameter
    std::map<Eigen::Index,double> l_d;

    // maximum radius
    double r_max;

    // number of points for spline
    int n_pts_splines = 1000;

    // jinc filter shape parameters
    std::map<Eigen::Index,Eigen::VectorXd> shape_params;

    // matrices to hold precomputed jinc function
    std::map<Eigen::Index,Eigen::MatrixXd> jinc_weights_mat;

    // splines for jinc function
    std::map<Eigen::Index, engine_utils::SplineFunction> jinc_splines;

    // calculate jinc weight at a given radius
    auto jinc_func(double, double, double, double, double, double);

    // precompute jinc weight matrix
    void allocate_jinc_matrix(double);

    // calculate spline function for jinc weights
    void calculate_jinc_splines();

    // allocate pointing matrix for polarization reduction
    template <class map_buffer_t>
    void allocate_pointing(map_buffer_t &, double, double, double, Eigen::Index, int, int);

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_jinc(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                            Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string &,
                            apt_t &, double, bool, bool);
};

auto JincMapmaker::jinc_func(double r, double a, double b, double c, double r_max, double l_d) {
    if (r!=0) {
        // unitless radius
        r = r/l_d;
        // first jinc function
        auto jinc_1 = 2.*boost::math::cyl_bessel_j(1,2.*pi*r/a)/(2.*pi*r/a);
        // exponential
        auto exp_func = exp(-pow(2.*r/b,c));
        // second jinc function
        auto jinc_2 = 2.*boost::math::cyl_bessel_j(1,3.831706*r/r_max)/(3.831706*r/r_max);
        // jinc1 x exp x jinc2
        return jinc_1*exp_func*jinc_2;
    }
    else {
        return 1.0;
    }
}

void JincMapmaker::allocate_jinc_matrix(double pixel_size_rad) {
    l_d[0] = (1.1/1000)/50;
    l_d[1] = (1.4/1000)/50;
    l_d[2] = (2.0/1000)/50;

    // loop through lambda/diameters
    for (const auto &ld: l_d) {
        // get shape params
        auto a = shape_params[ld.first](0);
        auto b = shape_params[ld.first](1);
        auto c = shape_params[ld.first](2);

        // maximum radius in pixels
        int r_max_pix = std::floor(r_max*ld.second/pixel_size_rad);

        // pixel centers within max radius
        Eigen::VectorXd pixels = Eigen::VectorXd::LinSpaced(2*r_max_pix + 1,-r_max_pix, r_max_pix);

        // allocate jinc weights
        jinc_weights_mat[ld.first].setZero(2*r_max_pix + 1,2*r_max_pix + 1);

        // loop through matrix rows
        for (Eigen::Index i=0; i<pixels.size(); ++i) {
            // loop through matrix cols
            for (Eigen::Index j=0; j<pixels.size(); ++j) {
                // radius of current pixel in radians
                double r = pixel_size_rad*sqrt(pow(pixels(i),2) + pow(pixels(j),2));
                // calculate jinc weight at pixel
                jinc_weights_mat[ld.first](i,j) = jinc_func(r,a,b,c,r_max,ld.second);
            }
        }
    }
}

void JincMapmaker::calculate_jinc_splines() {
    l_d[0] = (1.1/1000)/50;
    l_d[1] = (1.4/1000)/50;
    l_d[2] = (2.0/1000)/50;

    // loop through lambda/diameters
    for (const auto &ld: l_d) {
        // get shape params
        auto a = shape_params[ld.first](0);
        auto b = shape_params[ld.first](1);
        auto c = shape_params[ld.first](2);

        // radius vector in radians
        auto radius = Eigen::VectorXd::LinSpaced(n_pts_splines, 0, r_max*ld.second);
        // jinc weights on dense vector
        Eigen::VectorXd jinc_weights(radius.size());

        Eigen::Index j = 0;

        for (const auto &r: radius) {
            // calculate jinc weights
            jinc_weights(j) = jinc_func(r,a,b,c,r_max,ld.second);
            ++j;
        }
        // create spline class
        engine_utils::SplineFunction s;
        // spline interpolate
        s.interpolate(radius, jinc_weights);
        // store jinc spline
        jinc_splines[ld.first] = s;
    }
}

template <class map_buffer_t>
void JincMapmaker::allocate_pointing(map_buffer_t &mb, double weight, double cos_2angle, double sin_2angle,
                                     Eigen::Index map_index, int ir, int ic) {
    // update pointing matrix
    mb.pointing[map_index](ir,ic,0) += weight;
    mb.pointing[map_index](ir,ic,1) += weight*cos_2angle;
    mb.pointing[map_index](ir,ic,2) += weight*sin_2angle;
    mb.pointing[map_index](ir,ic,3) = mb.pointing[map_index](ir,ic,1);//weight*cos(2*angle);
    mb.pointing[map_index](ir,ic,4) += weight*pow(cos_2angle,2.);
    mb.pointing[map_index](ir,ic,5) += weight*cos_2angle*sin_2angle;
    mb.pointing[map_index](ir,ic,6) = mb.pointing[map_index](ir,ic,2);//weight*sin(2*angle);
    mb.pointing[map_index](ir,ic,7) = mb.pointing[map_index](ir,ic,5);//weight*cos(2*angle)*sin(2*angle);
    mb.pointing[map_index](ir,ic,8) += weight*pow(sin_2angle,2.);
}

template<class map_buffer_t, typename Derived, typename apt_t>
void JincMapmaker::populate_maps_jinc(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                        map_buffer_t &omb, map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
                        Eigen::DenseBase<Derived> &det_indices, std::string &pixel_axes, apt_t &apt,
                        double d_fsmp, bool run_omb, bool run_noise) {

    const bool use_cmb = !cmb.noise.empty();
    const bool use_omb = !omb.noise.empty();
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();

    // dimensions of data
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // step to skip to reach next stokes param
    int step = omb.pointing.size();

    // pointer to map buffer with noise maps
    ObsMapBuffer* nmb = nullptr;

    if (run_noise) {
        // set pointer to cmb or omb for noise maps
        nmb = use_cmb ? &cmb : (use_omb ? &omb : nullptr);
    }

    // placeholder vectors for grppi loop
    std::vector<int> det_in_vec(n_dets);
    std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
    std::vector<int> det_out_vec(n_dets);

    // parallelize over detectors
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
    //for (Eigen::Index i=0; i<n_dets; ++i) {
        // skip completely flagged detectors
        if ((in.flags.data.col(i).array()==0).any()) {
            // get detector positions from apt table if not in detector mapmaking mode
            auto det_index = det_indices(i);

            // which map to assign detector to
            Eigen::Index map_index = map_indices(i);
            // indices for Q and U maps
            int q_index = map_index + step;
            int u_index = map_index + 2 * step;
            Eigen::Index array_index = apt["array"](det_index);
            Eigen::Index mat_rows = jinc_weights_mat[array_index].rows();
            Eigen::Index mat_cols = jinc_weights_mat[array_index].cols();
            Eigen::Index mat_rows_center = (mat_rows - 1.)/2.;
            Eigen::Index mat_cols_center = (mat_cols - 1.)/2.;

            // get detector pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](det_index), apt["y_t"](det_index),
                                                              pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (use_cmb) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols)/2.;
            }

            // signal map value
            double signal, kernel;

            // noise map value
            double noise_v;

            // noise map indices
            Eigen::Index nmb_ir, nmb_ic;

            // cosine and sine of angles
            double cos_2angle, sin_2angle;

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; ++j) {
                // check if sample is flagged, ignore if so
                if (!in.flags.data(j,i)) {
                    Eigen::Index omb_ir = omb_irow(j);
                    Eigen::Index omb_ic = omb_icol(j);

                    if (run_polarization) {
                        cos_2angle = cos(2.*in.angle.data(j,i));
                        sin_2angle = sin(2.*in.angle.data(j,i));
                    }

                    if (run_omb) {
                        // make sure the data point is within the map
                        if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {

                            int lower_row = omb_ir - mat_rows_center;
                            int upper_row = omb_ir + mat_rows - 1 - mat_rows_center;
                            int lower_col = omb_ic - mat_cols_center;
                            int upper_col = omb_ic + mat_cols - 1 - mat_cols_center;

                            int jinc_lower_row = abs(std::min(0, lower_row));
                            int jinc_lower_col = abs(std::min(0, lower_col));

                            lower_row = std::max(0,lower_row);
                            upper_row = std::min(static_cast<int>(omb.n_rows - 1),upper_row);
                            lower_col = std::max(0,lower_col);
                            upper_col = std::min(static_cast<int>(omb.n_cols - 1),upper_col);

                            int size_rows = upper_row - lower_row + 1;
                            int size_cols = upper_col - lower_col + 1;

                            auto sig_block = omb.signal[map_index].block(lower_row,lower_col,size_rows,size_cols);
                            auto wt_block = omb.weight[map_index].block(lower_row,lower_col,size_rows,size_cols);
                            auto cov_block = omb.coverage[map_index].block(lower_row,lower_col,size_rows,size_cols);

                            const auto mat_block = jinc_weights_mat[array_index].block(jinc_lower_row,jinc_lower_col,size_rows,size_cols);

                            // populate signal map
                            sig_block += mat_block*in.weights.data(i)*in.scans.data(j,i);
                            // populate weight map
                            wt_block += mat_block*in.weights.data(i);

                            // populate coverage map
                            if (run_coverage) {
                                cov_block += mat_block/d_fsmp;
                            }

                            // populate kernel map
                            if (run_kernel) {
                                auto ker_block = omb.kernel[map_index].block(lower_row,lower_col,size_rows,size_cols);
                                ker_block += mat_block*in.weights.data(i)*in.kernel.data(j,i);
                            }

                            // loop through nearby rows and cols
                            /*for (Eigen::Index r=0; r<mat_rows; ++r) {
                                for (Eigen::Index c=0; c<mat_cols; ++c) {
                                    // get pixel in map
                                    Eigen::Index ri = omb_ir + r - mat_rows_center;
                                    Eigen::Index ci = omb_ic + c - mat_cols_center;

                                    // make sure pixel is in the map
                                    if (ri >= 0 && ci >= 0 && ri < omb.n_rows && ci < omb.n_cols) {
                                        // get radius from data point to pixel
                                        //auto radius = sqrt(std::pow(lat(j) - omb.rows_tan_vec(ri),2) + std::pow(lon(j) - omb.cols_tan_vec(ci),2));
                                        //auto weight = in.weights.data(i)*jinc_splines[array_index](radius);

                                        // det weight x jinc weight
                                        auto weight = in.weights.data(i)*jinc_weights_mat[array_index](r,c);

                                        // data x weight
                                        signal = in.scans.data(j,i)*weight;
                                        // populate signal map
                                        omb.signal[map_index](ri,ci) += signal;

                                        // populate weight map
                                        omb.weight[map_index](ri,ci) += weight;

                                        // populate kernel map
                                        if (run_kernel) {
                                            kernel = in.kernel.data(j,i)*weight;
                                            omb.kernel[map_index](ri,ci) += kernel;
                                        }

                                        // populate coverage map
                                        if (run_coverage) {
                                            omb.coverage[map_index](ri,ci) += jinc_weights_mat[array_index](r,c)/d_fsmp;
                                        }

                                        if (run_polarization) {
                                            // calculate pointing matrix
                                            allocate_pointing(omb, weight, cos_2angle, sin_2angle, map_index, ri, ci);

                                            // update signal map Q and U
                                            omb.signal[q_index](ri,ci) += signal*cos_2angle;
                                            omb.signal[u_index](ri,ci) += signal*sin_2angle;

                                            // update kernel map Q and U
                                            if (run_kernel) {
                                                omb.kernel[q_index](ri,ci) += kernel*cos_2angle;
                                                omb.kernel[u_index](ri,ci) += kernel*sin_2angle;
                                            }
                                        }
                                    }
                                }
                            }*/
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        // if coaddition is enabled
                        if (use_cmb) {
                            nmb_ir = cmb_irow(j);
                            nmb_ic = cmb_icol(j);
                        }

                        // else make noise maps for obs
                        else if (use_omb) {
                            nmb_ir = omb_irow(j);
                            nmb_ic = omb_icol(j);
                        }

                        // make sure pixel is in the map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {

                            int lower_row = nmb_ir - mat_rows_center;
                            int upper_row = nmb_ir + mat_rows - 1 - mat_rows_center;
                            int lower_col = nmb_ic - mat_cols_center;
                            int upper_col = nmb_ic + mat_cols - 1 - mat_cols_center;

                            int jinc_lower_row = abs(std::min(0, lower_row));
                            int jinc_lower_col = abs(std::min(0, lower_col));

                            lower_row = std::max(0,lower_row);
                            upper_row = std::min(static_cast<int>(nmb->n_rows - 1),upper_row);
                            lower_col = std::max(0,lower_col);
                            upper_col = std::min(static_cast<int>(nmb->n_cols - 1),upper_col);

                            int size_rows = upper_row - lower_row + 1;
                            int size_cols = upper_col - lower_col + 1;

                            const auto mat_block = jinc_weights_mat[array_index].block(jinc_lower_row,jinc_lower_col,size_rows,size_cols);
                            signal = in.scans.data(j,i)*in.weights.data(i);

                            for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                // randomizing on dets
                                if (nmb->randomize_dets) {
                                    noise_v = in.noise.data(nn,i)*signal;
                                }
                                else {
                                    noise_v = in.noise.data(nn)*signal;
                                }
                                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(nmb->noise[map_index].data() + nn * nmb->n_rows * nmb->n_cols,
                                                                                                               nmb->n_rows, nmb->n_cols);
                                auto noise_block = noise_matrix.block(lower_row,lower_col,size_rows,size_cols);
                                noise_block += mat_block*noise_v;
                            }

                            // loop through nearby rows and cols
                            /*for (Eigen::Index r=0; r<mat_rows; ++r) {
                                for (Eigen::Index c=0; c<mat_cols; ++c) {
                                    // get pixel in map
                                    Eigen::Index ri = nmb_ir + r - mat_rows_center;
                                    Eigen::Index ci = nmb_ic + c - mat_cols_center;

                                    // make sure pixel is in the map
                                    if (ri >= 0 && ci >= 0 && ri < nmb->n_rows && ci < nmb->n_cols) {
                                        // get radius from data point to pixel
                                        //auto radius = sqrt(std::pow(lat(j) - nmb->rows_tan_vec(ri),2) + std::pow(lon(j) - nmb->cols_tan_vec(ci),2));
                                        //auto weight = in.weights.data(i)*jinc_splines[array_index](radius);

                                        // det weight x jinc weight
                                        auto weight = in.weights.data(i)*jinc_weights_mat[array_index](r,c);
                                        // data x weight
                                        signal = in.scans.data(j,i)*weight;

                                        if (run_polarization) {
                                            if (!cmb.pointing.empty()) {
                                                // calculate pointing matrix for cmb
                                                allocate_pointing(cmb, weight, cos_2angle, sin_2angle, map_index, ri, ci);
                                            }
                                        }
                                        // populate noise maps
                                        for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                            // randomizing on dets
                                            if (nmb->randomize_dets) {
                                                noise_v = in.noise.data(nn,i)*signal;
                                            }
                                            else {
                                                noise_v = in.noise.data(nn)*signal;
                                            }
                                            // allocate pixel for current noise map
                                            nmb->noise[map_index](ri,ci,nn) += noise_v;

                                            if (run_polarization) {
                                                // update noise map Q
                                                nmb->noise[q_index](ri,ci,nn) += noise_v*cos_2angle;
                                                // update noise map U
                                                nmb->noise[u_index](ri,ci,nn) += noise_v*sin_2angle;
                                            }
                                        }
                                    }
                                }
                            }*/
                        }
                    }
                }
            }
        }
        return 0;
    });
}
} // namespace mapmaking

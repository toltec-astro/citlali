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

/*class JincMapmaker {
public:
    std::map<Eigen::Index, Eigen::VectorXd> shape_params;
    double r_max;

    // lambda over diameter
    std::map<Eigen::Index,double> l_d = {
        {0,(1.1/1000)/50},
        {1,(1.4/1000)/50},
        {2,(2.0/1000)/50}
    };

    auto jinc_filter(double, Eigen::Index);

    template<class map_buffer_t, typename Derived, typename apt_t, typename pointing_offset_t>
    void populate_maps(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                       Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string &,
                       std::string &, apt_t &, pointing_offset_t &, double, bool );
};

auto JincMapmaker::jinc_filter(double r, Eigen::Index arr) {
    if (r!=0) {
        double a = shape_params[arr](0);
        double b = shape_params[arr](1);
        double c = shape_params[arr](2);

        r = r/l_d[arr];
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
void JincMapmaker::populate_maps(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                        map_buffer_t &omb, map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices, Eigen::DenseBase<Derived> &det_indices,
                        std::string &pixel_axes, std::string &redu_type, apt_t &apt,
                        pointing_offset_t &pointing_offsets_arcsec, double d_fsmp, bool run_noise) {

}*/

// jinc function
auto jinc_func(double r, double a, double b, double c, double r_max, double l_d) {

    if (r!=0) {
        r = r/l_d;
        auto arg0 = 2.*boost::math::cyl_bessel_j(1,2.*pi*r/a)/(2.*pi*r/a);
        auto arg1 = exp(-pow(2.*r/b,c));
        auto arg2 = 2.*boost::math::cyl_bessel_j(1,3.831706*r/r_max)/(3.831706*r/r_max);

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
                        double r_max, std::map<Eigen::Index,Eigen::VectorXd> &shape_params, std::string parallel_policy) {

    // lambda over diameter
    std::map<Eigen::Index,double> l_d;
    l_d[0] = (1.1/1000)/50;
    l_d[1] = (1.4/1000)/50;
    l_d[2] = (2.0/1000)/50;

    std::map<Eigen::Index,Eigen::VectorXd> jinc_weights;
    /*std::map<Eigen::Index, engine_utils::SplineFunction2> jinc_splines;

    for (const auto &ld: l_d) {
        auto a = shape_params[ld.first](0);
        auto b = shape_params[ld.first](1);
        auto c = shape_params[ld.first](2);

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
    }*/

    std::map<Eigen::Index,Eigen::MatrixXd> jinc_weights_mat;
    for (const auto &ld: l_d) {
        auto a = shape_params[ld.first](0);
        auto b = shape_params[ld.first](1);
        auto c = shape_params[ld.first](2);

        int r_max_pix = std::floor(r_max*ld.second/omb.pixel_size_rad);
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
    Eigen::Matrix<int,Eigen::Dynamic, Eigen::Dynamic> noise;

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
        if (nmb->randomize_dets) {
            noise =
                Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>::Zero(nmb->n_noise, n_dets).unaryExpr([&](int dummy){return rands(eng);});
            noise = (2.*(noise.template cast<double>().array() - 0.5)).template cast<int>();
        }
        else {
            noise =
                Eigen::Matrix<int,Eigen::Dynamic,1>::Zero(nmb->n_noise).unaryExpr([&](int dummy){return rands(eng);});
            noise = (2.*(noise.template cast<double>().array() - 0.5)).template cast<int>();
        }
    }

    std::vector<int> det_in_vec, det_out_vec;

    det_in_vec.resize(n_dets);
    std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
    det_out_vec.resize(n_dets);

    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
    //for (Eigen::Index i=0; i<n_dets; i++) {
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

            auto a = shape_params[apt["array"](det_indices(i))](0);
            auto b = shape_params[apt["array"](det_indices(i))](1);
            auto c = shape_params[apt["array"](det_indices(i))](2);

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

                        Eigen::Index mat_rows = (jinc_weights_mat[apt["array"](det_indices(i))].rows() - 1.)/2.;
                        Eigen::Index mat_cols = (jinc_weights_mat[apt["array"](det_indices(i))].cols() - 1.)/2.;

                        // loop through nearby rows and cols
                        for (Eigen::Index r=0; r<jinc_weights_mat[apt["array"](det_indices(i))].rows(); r++) {
                            for (Eigen::Index c=0; c<jinc_weights_mat[apt["array"](det_indices(i))].cols(); c++) {

                                // get pixel in map
                                Eigen::Index ri = omb_ir + r - mat_rows;
                                Eigen::Index ci = omb_ic + c - mat_cols;

                                if (ri >= 0 && ci >= 0 && ri < omb.n_rows && ci < omb.n_cols) {

                                    auto jinc_weight = jinc_weights_mat[apt["array"](det_indices(i))](r,c);
                                    auto weight = in.weights.data(i)*jinc_weight;

                                    // populate signal map
                                    signal = in.scans.data(j,i)*weight;
                                    omb.signal[map_index](ri,ci) += signal;

                                    // populate weight map
                                    omb.weight[map_index](ri,ci) += weight;

                                    // populate kernel map
                                    if (!omb.kernel.empty()) {
                                        auto kernel = in.kernel.data(j,i)*weight;
                                        omb.kernel[map_index](ri,ci) += kernel;
                                    }

                                    // populate coverage map
                                    if (!omb.coverage.empty()) {
                                        omb.coverage[map_index](ri,ci) += (jinc_weight/d_fsmp);
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

                        // loop through noise maps and coadd into current noise map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {

                            double r_max_pix = std::floor(r_max*l_d[apt["array"](det_indices(i))]/nmb->pixel_size_rad);

                            Eigen::Index mat_rows = (jinc_weights_mat[apt["array"](det_indices(i))].rows() - 1.)/2.;
                            Eigen::Index mat_cols = (jinc_weights_mat[apt["array"](det_indices(i))].cols() - 1.)/2.;

                            for (Eigen::Index r=0; r<jinc_weights_mat[apt["array"](det_indices(i))].rows(); r++) {
                                for (Eigen::Index c=0; c<jinc_weights_mat[apt["array"](det_indices(i))].cols(); c++) {

                                    // get pixel in map
                                    Eigen::Index ri = omb_ir + r - mat_rows;
                                    Eigen::Index ci = omb_ic + c - mat_cols;

                                    if (ri >= 0 && ci >= 0 && ri < nmb->n_rows && ci < nmb->n_cols) {

                                        auto jinc_weight = jinc_weights_mat[apt["array"](det_indices(i))](r,c);
                                        auto weight = in.weights.data(i)*jinc_weight;
                                        signal = in.scans.data(j,i)*weight;

                                        // populate signal map
                                        for (Eigen::Index nn=0; nn<nmb->n_noise; nn++) {
                                            if (nmb->randomize_dets) {
                                                nmb->noise[map_index](ri,ci,nn) += noise(nn,i)*signal;
                                            }
                                            else {
                                                nmb->noise[map_index](ri,ci,nn) += noise(nn)*signal;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return 0;
    });
}
} // namespace mapmaking

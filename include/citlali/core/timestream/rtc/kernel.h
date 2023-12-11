#pragma once

#include <string>

#include <boost/math/special_functions/bessel.hpp>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/pointing.h>
#include <citlali/core/utils/fits_io.h>

namespace timestream {

class Kernel {
public:
    std::string filepath, type;
    std::vector<std::string> img_ext_names;

    // input kernel images
    std::vector<Eigen::MatrixXd> images;

    // sigma and fwhm from config
    double sigma_rad, fwhm_rad;

    // limit on distance to calc sigma to
    double sigma_limit = 3;

    // map grouping
    std::string map_grouping;

    // initial setup
    void setup(Eigen::Index);

    // symmetric gaussian kernel
    template<typename apt_t, typename Derived>
    void create_symmetric_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &,
                                          std::string &, apt_t &, Eigen::DenseBase<Derived> &);

    // asymmetric elliptical gaussian kernel
    template<typename apt_t, typename Derived>
    void create_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &,
                                std::string &, apt_t &, Eigen::DenseBase<Derived> &);
    // airy pattern kernel
    template<typename apt_t, typename Derived>
    void create_airy_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &,
                            std::string &, apt_t &, Eigen::DenseBase<Derived> &);

    // kernel from fits file
    template<typename apt_t, typename Derived>
    void create_kernel_from_fits(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &,
                                 std::string &, apt_t &, double, Eigen::DenseBase<Derived> &,
                                 Eigen::DenseBase<Derived> &);
};

void Kernel::setup(Eigen::Index n_maps) {
    if (type == "fits") {
        if (img_ext_names.size()!=n_maps && img_ext_names.size()!=1) {
            SPDLOG_INFO("mismatch for number of kernel images");
            std::exit(EXIT_FAILURE);
        }

        fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*> fits_io(filepath);
        for (auto & img_ext_name : img_ext_names) {
            images.push_back(fits_io.get_hdu(img_ext_name));
        }
    }
}

template<typename apt_t, typename Derived>
void Kernel::create_symmetric_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes,
                                              std::string &redu_type, apt_t &apt, Eigen::DenseBase<Derived> &det_indices) {

    // dimensions of scan
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // resize kernel to match data size
    in.kernel.data.resize(n_pts,n_dets);

    double sigma = sigma_rad;

    for (Eigen::Index i=0; i<n_dets; ++i) {
        // detector in apt
        auto det_index = det_indices(i);

        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](det_index), apt["y_t"](det_index),
                                                          pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        // distance to source to truncate it
        auto dist = ((lat.array()).pow(2) + (lon.array()).pow(2)).sqrt();

        // calculate stddev from apt table if config stddev <=0
        if (sigma_rad <= 0) {
            sigma = FWHM_TO_STD * ASEC_TO_RAD*(apt["a_fwhm"](det_index) + apt["b_fwhm"](det_index))/2;
        }

        // loop through samples and calculate
        for (Eigen::Index j=0; j<n_pts; ++j) {
            // truncate within radius
            if (dist(j) <= sigma_limit*sigma) {
                in.kernel.data(j,i) = exp(-0.5*pow(dist(j)/sigma,2));
            }
            else {
                in.kernel.data(j,i) = 0;
            }
        }
    }
}

template<typename apt_t, typename Derived>
void Kernel::create_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes,
                                    std::string &redu_type, apt_t &apt, Eigen::DenseBase<Derived> &det_indices) {

    // dimensions of scan
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // resize kernel to match data size
    in.kernel.data.resize(n_pts,n_dets);

    // get parameters for current detector
    double amp = 1.0;
    double off_lat = 0.0;
    double off_lon = 0.0;

    // beam standard deviations
    double sigma_lat = sigma_rad;
    double sigma_lon = sigma_rad;

    for (Eigen::Index i=0; i<n_dets; ++i) {
        // detector in apt
        auto det_index = det_indices(i);

        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](det_index), apt["y_t"](det_index),
                                                          pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        // distance to source to truncate it
        auto dist = ((lat.array()).pow(2) + (lon.array()).pow(2)).sqrt();

        // calculate stddev from apt table if config stddev <=0
        if (sigma_rad <= 0) {
            sigma_lat = FWHM_TO_STD * ASEC_TO_RAD * apt["b_fwhm"](det_index);
            sigma_lon = FWHM_TO_STD * ASEC_TO_RAD * apt["a_fwhm"](det_index);
        }

        // rotation angle
        double rot_ang = apt["angle"](det_index);

        auto cost2 = cos(rot_ang) * cos(rot_ang);
        auto sint2 = sin(rot_ang) * sin(rot_ang);
        auto sin2t = sin(2. * rot_ang);
        auto xstd2 = sigma_lon * sigma_lon;
        auto ystd2 = sigma_lat * sigma_lat;
        auto a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        auto b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        auto c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

        double sigma_limit_det = sigma_limit * (sigma_lat + sigma_lon)/2;

        // make elliptical gaussian
        for (Eigen::Index j=0; j<n_pts; ++j) {
            // truncate within radius
            if (dist(j) <= sigma_limit_det) {
                in.kernel.data(j,i) = amp*exp(pow(lon(j) - off_lon, 2) * a +
                                     (lon(j) - off_lon) * (lat(j) - off_lat) * b +
                                     pow(lat(j) - off_lat, 2) * c);
            }
            else {
                in.kernel.data(j,i) = 0;
            }
        }
    }
}

template<typename apt_t, typename Derived>
void Kernel::create_airy_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes,
                                std::string &redu_type, apt_t &apt, Eigen::DenseBase<Derived> &det_indices) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.kernel.data.resize(n_pts,n_dets);

    double fwhm = fwhm_rad;

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // current detector in apt
        auto det_index = det_indices(i);

        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](det_index), apt["y_t"](det_index),
                                                          pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        // distance to source to truncate it
        auto dist = ((lat.array()).pow(2) + (lon.array()).pow(2)).sqrt();

        // get fwhm from apt if config file fwhm is <= 0
        if (fwhm_rad <= 0) {
            fwhm = ASEC_TO_RAD*(apt["a_fwhm"](det_index) + apt["b_fwhm"](det_index))/2;
        }

        // airy pattern factor
        double factor = pi*(1.028/fwhm);

        for (Eigen::Index j=0; j<n_pts; ++j) {
            if (dist(j) <= sigma_limit*fwhm) {
                in.kernel.data(j,i) = pow(2*boost::math::cyl_bessel_j(1,factor*dist(j))/(factor*dist(j)),2);
            }
            else {
                in.kernel.data(j,i) = 0;
            }
        }
    }
}

template<typename apt_t, typename Derived>
void Kernel::create_kernel_from_fits(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes,
                                     std::string &redu_type, apt_t &apt, double pixel_size_rad,
                                     Eigen::DenseBase<Derived> &map_indices, Eigen::DenseBase<Derived> &det_indices) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.kernel.data.resize(n_pts,n_dets);

    Eigen::Index map_index = 0;

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // current detector index in apt
        auto det_index = det_indices(i);

        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](det_index), apt["y_t"](det_index),
                                                          pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        if (images.size() > 1) {
            map_index = map_indices(i);
        }

        // get map buffer row and col indices for lat and lon vectors
        Eigen::VectorXd irows = lat.array()/pixel_size_rad + (images[map_index].rows())/2.;
        Eigen::VectorXd icols = lon.array()/pixel_size_rad + (images[map_index].cols())/2.;

        for (Eigen::Index j = 0; j<n_pts; ++j) {
            // row and col pixel for kernel image
            Eigen::Index ir = irows(j);
            Eigen::Index ic = icols(j);

            // check if current sample is on the image and add to the timestream
            if ((ir >= 0) && (ir < images[map_index].rows()) && (ic >= 0) && (ic < images[map_index].cols())) {
                in.kernel.data(j,i) = images[map_index](ir,ic);
            }
        }
    }
}
} // namespace timestream

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

    std::vector<Eigen::MatrixXd> images;

    void setup(Eigen::Index n_maps) {
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

    template<typename apt_t, typename pointing_offset_t, typename Derived>
    void create_symmetric_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &,
                                          std::string &, apt_t &, pointing_offset_t &,
                                          Eigen::DenseBase<Derived> &);

    template<typename apt_t, typename pointing_offset_t, typename Derived>
    void create_airy_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &,
                            std::string &, apt_t &, pointing_offset_t &,
                            Eigen::DenseBase<Derived> &);

    template<typename apt_t, typename pointing_offset_t>
    void create_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &);

    template<typename apt_t, typename pointing_offset_t, typename Derived>
    void create_kernel_from_fits(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &,
                                          std::string &, apt_t &, pointing_offset_t &, double,
                                          Eigen::DenseBase<Derived> &);

};


template<typename apt_t, typename pointing_offset_t, typename Derived>
void Kernel::create_symmetric_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes,
                                              std::string &redu_type, apt_t &apt,
                                              pointing_offset_t &pointing_offsets_arcsec, Eigen::DenseBase<Derived> &det_indices) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.kernel.data.resize(n_pts, n_dets);

    for (Eigen::Index i=0; i<n_dets; i++) {

        auto det_index = det_indices(i);

        double az_off = 0;
        double el_off = 0;

        if (redu_type!="beammap") {
            az_off = apt["x_t"](det_index);
            el_off = apt["y_t"](det_index);
        }

        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off,
                                                          pixel_axes, pointing_offsets_arcsec);

        auto distance = (lat.array().pow(2) + lon.array().pow(2)).sqrt();

        double sigma = ASEC_TO_RAD*(apt["a_fwhm"](det_index) + apt["b_fwhm"](det_index))/STD_TO_FWHM;

        for (Eigen::Index j=0; j<n_pts; j++) {
            if (distance(j) <= 3.*sigma) {
                in.kernel.data(j,i) = exp(-0.5*pow(distance(j)/sigma,2));
            }
            else {
                in.kernel.data(j,i) = 0;
            }
        }
    }
}

template<typename apt_t, typename pointing_offset_t, typename Derived>
void Kernel::create_airy_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes,
                                              std::string &redu_type, apt_t &apt, pointing_offset_t &pointing_offsets_arcsec,
                                              Eigen::DenseBase<Derived> &det_indices) {
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.kernel.data.resize(n_pts, n_dets);

    for (Eigen::Index i=0; i<n_dets; i++) {

        auto det_index = det_indices(i);

        double az_off = 0;
        double el_off = 0;

        if (redu_type!="beammap") {
            az_off = apt["x_t"](det_index);
            el_off = apt["y_t"](det_index);
        }

        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off,
                                                          pixel_axes, pointing_offsets_arcsec);

        auto distance = (lat.array().pow(2) + lon.array().pow(2)).sqrt();

        double fwhm = ASEC_TO_RAD*(apt["a_fwhm"](det_index) + apt["b_fwhm"](det_index));

        double factor = pi*(1.028/fwhm);

        for (Eigen::Index j=0; j<n_pts; j++) {
            if (distance(j) <= 3.*fwhm) {
                in.kernel.data(j,i) = pow(2*boost::math::cyl_bessel_j(1,factor*distance(j))/(factor*distance(j)),2);
            }
            else {
                in.kernel.data(j,i) = 0;
            }
        }
    }
}

template<typename apt_t, typename pointing_offset_t, typename Derived>
void Kernel::create_kernel_from_fits(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes,
                                     std::string &redu_type, apt_t &apt,
                                     pointing_offset_t &pointing_offsets_arcsec,
                                     double pixel_size_rad,
                                     Eigen::DenseBase<Derived> &map_indices) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.kernel.data.resize(n_pts, n_dets);

    for (Eigen::Index i=0; i<n_dets; i++) {
        double az_off = 0;
        double el_off = 0;

        if (redu_type!="beammap") {
            az_off = apt["x_t"](i);
            el_off = apt["y_t"](i);
        }

        Eigen::Index map_index = 0;

        if (images.size() > 1) {
            map_index = map_indices(i);
        }

        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off,
                                                          pixel_axes, pointing_offsets_arcsec);

        // get map buffer row and col indices for lat and lon vectors
        Eigen::VectorXd irows = lat.array()/pixel_size_rad + (images[map_index].rows())/2.;
        Eigen::VectorXd icols = lon.array()/pixel_size_rad + (images[map_index].cols())/2.;

        for (Eigen::Index si = 0; si<n_pts; si++) {
            // row and col pixel for kernel image
            Eigen::Index ir = (irows(si));
            Eigen::Index ic = (icols(si));

            // check if current sample is on the image and add to the timestream
            if ((ir >= 0) && (ir < images[map_index].rows()) && (ic >= 0) && (ic < images[map_index].cols())) {
                in.kernel.data(si,i) = images[map_index](ir,ic);
            }
        }
    }

}

} // namespace timestream

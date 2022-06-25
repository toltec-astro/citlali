#pragma once

#include <Eigen/Core>
#include <boost/math/special_functions/bessel.hpp>

#include <citlali/core/utils/fits_io.h>

#include <citlali/core/utils/pointing.h>
#include <citlali/core/timestream/timestream.h>

namespace timestream {

class Kernel {
public:
    std::string filepath, kernel_type, hdu_name;

    Eigen::MatrixXd image;

    std::vector<std::string> hdu_names;
    std::vector<Eigen::MatrixXd> images;

    void setup() {
        if (kernel_type == "image") {
            FitsIO<fileType::read_fits, CCfits::ExtHDU*> fits_io(filepath);
            image = fits_io.get_hdu(hdu_name);
        }
    }

    template <class Engine, typename Derived>
    void gaussian_kernel(Engine, TCData<TCDataKind::RTC, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &);

    template <class Engine, typename Derived>
    void airy_kernel(Engine, TCData<TCDataKind::RTC, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &);

    template <class Engine, typename Derived>
    void kernel_from_fits(Engine, TCData<TCDataKind::RTC, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &);
};

template <class Engine, typename Derived>
void Kernel::gaussian_kernel(Engine engine, TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &det_index_vector) {

    // resize kernel scans
    in.kernel_scans.data.resize(in.scans.data.rows(), in.scans.data.cols());

    // loop through detectors and make a kernel timestream
    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {

        Eigen::Index di = det_index_vector(i);

        // get kernel standard deviation from apt table
        double sigma = ASEC_TO_RAD*(engine->calib_data["a_fwhm"](di) + engine->calib_data["b_fwhm"](di))/2.;
        sigma = sigma/STD_TO_FWHM;

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
        // get pointing
        auto [lat, lon] = engine_utils::get_det_pointing(in.tel_meta_data.data, azoff, eloff, engine->map_type, engine->pointing_offsets);

        // distance from map center
        auto dist = (lat.array().pow(2) + lon.array().pow(2)).sqrt();

        {
            //tula::logging::scoped_timeit timer("kernel method 2");
             in.kernel_scans.data.col(i) =
             (dist.array() <= 3.*sigma).select(exp(-0.5*(dist.array()/sigma).pow(2)), 0);
        }

        {
            //tula::logging::scoped_timeit timer("kernel method 1");
            // is this faster?
            for (Eigen::Index j=0; j<in.scans.data.rows(); j++) {
                if (dist(j) <= 3.*sigma) {
                    in.kernel_scans.data(j,i) = exp(-0.5*pow(dist(j)/sigma,2));
                }
                else {
                    in.kernel_scans.data(j,i) = 0;
                }
            }
        }
    }
}

template <class Engine, typename Derived>
void Kernel::airy_kernel(Engine engine, TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &det_index_vector) {
    // resize kernel scans
    in.kernel_scans.data.resize(in.scans.data.rows(), in.scans.data.cols());

    // loop through detectors and make a kernel timestream
    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {

        Eigen::Index di = det_index_vector(i);

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

        // get pointing
        auto [lat, lon] = engine_utils::get_det_pointing(in.tel_meta_data.data, azoff, eloff, engine->map_type, engine->pointing_offsets);

        // distance from map center
        auto dist = (lat.array().pow(2) + lon.array().pow(2)).sqrt();
        auto fwhm = ASEC_TO_RAD*(engine->calib_data["a_fwhm"](di) + engine->calib_data["b_fwhm"](di))/2.;
        auto factor = pi*(1.028/fwhm);

        for (Eigen::Index j=0; j<in.scans.data.rows(); j++) {
            in.kernel_scans.data(j,i) = pow(2*boost::math::cyl_bessel_j(1,factor*dist(j))/(factor*dist(j)),2);
        }
    }
}

template <class Engine, typename Derived>
void Kernel::kernel_from_fits(Engine engine, TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &det_index_vector) {

    in.kernel_scans.data.setZero(in.scans.data.rows(), in.scans.data.cols());

    // loop through detectors and make a kernel timestream
    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {

        Eigen::Index di = det_index_vector(i);

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

        // get pointing
        auto [lat, lon] = engine_utils::get_det_pointing(in.tel_meta_data.data, azoff, eloff, engine->map_type, engine->pointing_offsets);

        // get map buffer row and col indices for lat and lon vectors
        Eigen::VectorXd mb_irow = lat.array()/engine->pixel_size + (image.rows())/2.;
        Eigen::VectorXd mb_icol = lon.array()/engine->pixel_size + (image.cols())/2.;

        for (Eigen::Index si = 0; si < in.scans.data.rows(); si++) {
            // row and col pixel for kernel image
            Eigen::Index mb_ir = (mb_irow(si));
            Eigen::Index mb_ic = (mb_icol(si));

            // check if current sample is on the image and add to the timestream
            if ((mb_ir >= 0) && (mb_ir < image.rows()) && (mb_ic >= 0) && (mb_ic < image.cols())) {
                in.kernel_scans.data(si,i) = image(mb_ir,mb_ic);
            }
        }
    }
}

} // namespace

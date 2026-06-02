#pragma once

#include <cmath>
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

    // Optional detector-map source centers in map-frame radians.  Beammap
    // iteration 0 leaves these empty so the kernel is centered; later
    // iterations can populate them from previous fitted source locations.
    Eigen::VectorXd source_lat;
    Eigen::VectorXd source_lon;
    Eigen::VectorXi source_valid;

    void clear_source_centers();
    void set_source_centers(const Eigen::VectorXd &, const Eigen::VectorXd &,
                            const Eigen::VectorXi &);
    bool has_source_centers() const;
    bool source_center_for_map(Eigen::Index, double &, double &) const;

    // initial setup
    void setup(Eigen::Index);

    // symmetric gaussian kernel
    template<typename apt_t>
    void create_symmetric_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &,
                                          apt_t &);
    // asymmetric elliptical gaussian kernel
    template<typename apt_t>
    void create_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &, apt_t &);
    // airy pattern kernel
    template<typename apt_t>
    void create_airy_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &, apt_t &);

    // kernel from fits file
    template<typename apt_t, typename Derived>
    void create_kernel_from_fits(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string &,
                                 apt_t &, double, Eigen::DenseBase<Derived> &);
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

inline void Kernel::clear_source_centers() {
    source_lat.resize(0);
    source_lon.resize(0);
    source_valid.resize(0);
}

inline void Kernel::set_source_centers(const Eigen::VectorXd &lat,
                                       const Eigen::VectorXd &lon,
                                       const Eigen::VectorXi &valid) {
    source_lat = lat;
    source_lon = lon;
    source_valid = valid;
}

inline bool Kernel::has_source_centers() const {
    return map_grouping == "detector" &&
           source_valid.size() > 0 &&
           source_lat.size() == source_valid.size() &&
           source_lon.size() == source_valid.size() &&
           (source_valid.array() != 0).any();
}

inline bool Kernel::source_center_for_map(Eigen::Index map_index,
                                          double &lat,
                                          double &lon) const {
    lat = 0.0;
    lon = 0.0;
    if (map_grouping != "detector" ||
        map_index < 0 ||
        map_index >= source_valid.size() ||
        map_index >= source_lat.size() ||
        map_index >= source_lon.size() ||
        source_valid(map_index) == 0) {
        return false;
    }
    const double src_lat = source_lat(map_index);
    const double src_lon = source_lon(map_index);
    if (!std::isfinite(src_lat) || !std::isfinite(src_lon)) {
        return false;
    }
    lat = src_lat;
    lon = src_lon;
    return true;
}

template<typename apt_t>
void Kernel::create_symmetric_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes, apt_t &apt) {

    // dimensions of scan
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // resize kernel to match data size
    in.kernel.data.resize(n_pts,n_dets);

    double sigma = sigma_rad;

    for (Eigen::Index i=0; i<n_dets; ++i) {
        double source_lat_rad = 0.0;
        double source_lon_rad = 0.0;
        source_center_for_map(i, source_lat_rad, source_lon_rad);

        // calc tangent plane pointing for a unit source kernel.  Detector
        // beammaps use fitted map-frame source centers when available; iter 0
        // and non-detector maps keep the source at the map center.
        auto [lat, lon] = engine_utils::calc_det_pointing(
            in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
            pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        // distance to source to truncate it
        auto dist = ((lat.array() - source_lat_rad).pow(2) +
                     (lon.array() - source_lon_rad).pow(2)).sqrt();

        // calculate stddev from apt table if config stddev <=0
        if (sigma_rad <= 0) {
            sigma = FWHM_TO_STD * ASEC_TO_RAD*(apt["a_fwhm"](i) + apt["b_fwhm"](i))/2;
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

template<typename apt_t>
void Kernel::create_gaussian_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes, apt_t &apt) {

    // dimensions of scan
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // resize kernel to match data size
    in.kernel.data.resize(n_pts,n_dets);

    // get parameters for current detector
    double amp = 1.0;

    // beam standard deviations
    double sigma_lat = sigma_rad;
    double sigma_lon = sigma_rad;

    for (Eigen::Index i=0; i<n_dets; ++i) {
        double source_lat_rad = 0.0;
        double source_lon_rad = 0.0;
        source_center_for_map(i, source_lat_rad, source_lon_rad);

        // calc tangent plane pointing for a unit source kernel.  Detector
        // beammaps use fitted map-frame source centers when available; iter 0
        // and non-detector maps keep the source at the map center.
        auto [lat, lon] = engine_utils::calc_det_pointing(
            in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
            pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        // distance to source to truncate it
        auto dist = ((lat.array() - source_lat_rad).pow(2) +
                     (lon.array() - source_lon_rad).pow(2)).sqrt();

        // calculate stddev from apt table if config stddev <=0
        if (sigma_rad <= 0) {
            sigma_lat = FWHM_TO_STD * ASEC_TO_RAD * apt["b_fwhm"](i);
            sigma_lon = FWHM_TO_STD * ASEC_TO_RAD * apt["a_fwhm"](i);
        }

        // rotation angle
        double rot_ang = apt["angle"](i);

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
                in.kernel.data(j,i) = amp*exp(pow(lon(j) - source_lon_rad, 2) * a +
                                     (lon(j) - source_lon_rad) * (lat(j) - source_lat_rad) * b +
                                     pow(lat(j) - source_lat_rad, 2) * c);
            }
            else {
                in.kernel.data(j,i) = 0;
            }
        }
    }
}

template<typename apt_t>
void Kernel::create_airy_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes, apt_t &apt) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.kernel.data.resize(n_pts,n_dets);
    in.kernel.data.setZero();

    double fwhm = fwhm_rad;

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        double source_lat_rad = 0.0;
        double source_lon_rad = 0.0;
        source_center_for_map(i, source_lat_rad, source_lon_rad);

        // calc tangent plane pointing for a unit source kernel.  Detector
        // beammaps use fitted map-frame source centers when available; iter 0
        // and non-detector maps keep the source at the map center.
        auto [lat, lon] = engine_utils::calc_det_pointing(
            in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
            pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        // distance to source to truncate it
        auto dist = ((lat.array() - source_lat_rad).pow(2) +
                     (lon.array() - source_lon_rad).pow(2)).sqrt();

        // get fwhm from apt if config file fwhm is <= 0
        if (fwhm_rad <= 0) {
            fwhm = ASEC_TO_RAD*(apt["a_fwhm"](i) + apt["b_fwhm"](i))/2;
        }

        // airy pattern factor
        double factor = pi*(1.028/fwhm);

        for (Eigen::Index j=0; j<n_pts; ++j) {
            if (dist(j) <= sigma_limit*fwhm) {
                const double x = factor * dist(j);
                if (std::abs(x) < 1e-12) {
                    in.kernel.data(j,i) = 1.0;
                }
                else {
                    in.kernel.data(j,i) =
                        pow(2 * boost::math::cyl_bessel_j(1, x) / x, 2);
                }
            }
            else {
                in.kernel.data(j,i) = 0;
            }
        }
    }
}

template<typename apt_t, typename Derived>
void Kernel::create_kernel_from_fits(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes, apt_t &apt,
                                     double pixel_size_rad, Eigen::DenseBase<Derived> &map_indices) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.kernel.data.resize(n_pts,n_dets);
    in.kernel.data.setZero();

    Eigen::Index map_index = 0;

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        const Eigen::Index source_map_index =
            (i >= 0 && i < map_indices.size()) ? map_indices(i) : i;
        double source_lat_rad = 0.0;
        double source_lon_rad = 0.0;
        source_center_for_map(source_map_index, source_lat_rad, source_lon_rad);

        // calc tangent plane pointing for a unit source kernel.  Detector
        // beammaps use fitted map-frame source centers when available; iter 0
        // and non-detector maps keep the source at the map center.
        auto [lat, lon] = engine_utils::calc_det_pointing(
            in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
            pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        if (images.size() > 1) {
            map_index = source_map_index;
        }

        // get map buffer row and col indices for lat and lon vectors
        const double row_center = (images[map_index].rows() - 1) / 2.0;
        const double col_center = (images[map_index].cols() - 1) / 2.0;
        Eigen::VectorXd irows = (lat.array() - source_lat_rad)/pixel_size_rad + row_center;
        Eigen::VectorXd icols = (lon.array() - source_lon_rad)/pixel_size_rad + col_center;

        for (Eigen::Index j = 0; j<n_pts; ++j) {
            // row and col pixel for kernel image
            Eigen::Index ir = static_cast<Eigen::Index>(std::llround(irows(j)));
            Eigen::Index ic = static_cast<Eigen::Index>(std::llround(icols(j)));

            // check if current sample is on the image and add to the timestream
            if ((ir >= 0) && (ir < images[map_index].rows()) && (ic >= 0) && (ic < images[map_index].cols())) {
                in.kernel.data(j,i) = images[map_index](ir,ic);
            }
        }
    }
}
} // namespace timestream

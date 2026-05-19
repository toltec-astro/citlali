#pragma once

#include <string>
#include <cmath>
#include <cctype>

#include <boost/math/special_functions/bessel.hpp>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/pointing.h>
#include <citlali/core/utils/fits_io.h>

namespace timestream {

class Kernel {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    std::string filepath, type;
    std::vector<std::string> img_ext_names;
    bool use_wcs_reprojection = true;

    // input kernel images
    std::vector<Eigen::MatrixXd> images;
    std::vector<Eigen::MatrixXd> reprojected_images;

    struct KernelWcs {
        bool has_wcs = false;
        double crpix1_pix = 0.0;
        double crpix2_pix = 0.0;
        double crval1_rad = 0.0;
        double crval2_rad = 0.0;
        double cdelt1_rad = 0.0;
        double cdelt2_rad = 0.0;
    };
    std::vector<KernelWcs> image_wcs;
    bool reprojection_ready = false;
    double reproj_pixel_size_rad = -1.0;

    // sigma and fwhm from config
    double sigma_rad, fwhm_rad;

    // limit on distance to calc sigma to
    double sigma_limit = 3;

    // map grouping
    std::string map_grouping;

    // initial setup
    void setup(Eigen::Index, std::string);

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

    double unit_to_rad(std::string unit) {
        for (auto &c: unit) {
            c = static_cast<char>(std::tolower(c));
        }
        if (unit == "rad" || unit == "radian" || unit == "radians") {
            return 1.0;
        }
        if (unit == "deg" || unit == "degree" || unit == "degrees") {
            return DEG_TO_RAD;
        }
        if (unit == "arcsec") {
            return ASEC_TO_RAD;
        }
        if (unit == "arcmin") {
            return 60.0 * ASEC_TO_RAD;
        }
        return DEG_TO_RAD;
    }

    double bilinear_sample(const Eigen::MatrixXd &img, double row, double col) {
        if (img.rows() <= 1 || img.cols() <= 1) {
            return 0.0;
        }
        if (row < 0.0 || col < 0.0 || row > static_cast<double>(img.rows() - 1) ||
            col > static_cast<double>(img.cols() - 1)) {
            return 0.0;
        }
        auto r0 = static_cast<Eigen::Index>(std::floor(row));
        auto c0 = static_cast<Eigen::Index>(std::floor(col));
        auto r1 = std::min<Eigen::Index>(r0 + 1, img.rows() - 1);
        auto c1 = std::min<Eigen::Index>(c0 + 1, img.cols() - 1);
        double fr = row - static_cast<double>(r0);
        double fc = col - static_cast<double>(c0);
        double v00 = img(r0, c0);
        double v01 = img(r0, c1);
        double v10 = img(r1, c0);
        double v11 = img(r1, c1);
        return (1.0 - fr) * (1.0 - fc) * v00 +
               (1.0 - fr) * fc * v01 +
               fr * (1.0 - fc) * v10 +
               fr * fc * v11;
    }

    double nearest_sample(const Eigen::MatrixXd &img, double row, double col) {
        auto ir = static_cast<Eigen::Index>(std::llround(row));
        auto ic = static_cast<Eigen::Index>(std::llround(col));
        if (ir < 0 || ic < 0 || ir >= img.rows() || ic >= img.cols()) {
            return 0.0;
        }
        return img(ir, ic);
    }

    auto reproject_image_to_target(const Eigen::MatrixXd &src, const KernelWcs &wcs, double target_pixel_size_rad) {
        if (!wcs.has_wcs || target_pixel_size_rad <= 0) {
            return src;
        }

        auto src_drow = std::abs(wcs.cdelt2_rad);
        auto src_dcol = std::abs(wcs.cdelt1_rad);
        if (src_drow <= 0 || src_dcol <= 0) {
            return src;
        }

        auto n_rows = std::max<Eigen::Index>(1, static_cast<Eigen::Index>(std::llround(
            static_cast<double>(src.rows()) * src_drow / target_pixel_size_rad)));
        auto n_cols = std::max<Eigen::Index>(1, static_cast<Eigen::Index>(std::llround(
            static_cast<double>(src.cols()) * src_dcol / target_pixel_size_rad)));

        Eigen::MatrixXd out(n_rows, n_cols);
        out.setZero();

        double row0 = (static_cast<double>(n_rows) - 1.0) / 2.0;
        double col0 = (static_cast<double>(n_cols) - 1.0) / 2.0;

        for (Eigen::Index r = 0; r < n_rows; ++r) {
            double lat = (static_cast<double>(r) - row0) * target_pixel_size_rad;
            for (Eigen::Index c = 0; c < n_cols; ++c) {
                double lon = (static_cast<double>(c) - col0) * target_pixel_size_rad;
                double src_col = (lon - wcs.crval1_rad) / wcs.cdelt1_rad + wcs.crpix1_pix;
                double src_row = (lat - wcs.crval2_rad) / wcs.cdelt2_rad + wcs.crpix2_pix;
                out(r, c) = bilinear_sample(src, src_row, src_col);
            }
        }
        return out;
    }

    void ensure_reprojected_images(double target_pixel_size_rad) {
        if (reprojection_ready && std::abs(reproj_pixel_size_rad - target_pixel_size_rad) <= 1e-18) {
            return;
        }
        reprojected_images.clear();
        for (Eigen::Index i = 0; i < images.size(); ++i) {
            reprojected_images.push_back(reproject_image_to_target(images[i], image_wcs[i], target_pixel_size_rad));
        }
        reprojection_ready = true;
        reproj_pixel_size_rad = target_pixel_size_rad;
    }
};

void Kernel::setup(Eigen::Index n_maps, std::string pixel_axes) {
    if (type == "fits") {
        images.clear();
        image_wcs.clear();
        reprojected_images.clear();
        reprojection_ready = false;
        reproj_pixel_size_rad = -1.0;

        if (img_ext_names.size()!=n_maps && img_ext_names.size()!=1) {
            logger->error("mismatch for number of kernel images");
            std::exit(EXIT_FAILURE);
        }

        fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*> fits_io(filepath, pixel_axes);
        for (auto & img_ext_name : img_ext_names) {
            images.push_back(fits_io.get_hdu(img_ext_name));

            KernelWcs wcs;
            try {
                CCfits::ExtHDU& hdu = fits_io.pfits->extension(img_ext_name);
                double crpix1 = 0.0, crpix2 = 0.0, crval1 = 0.0, crval2 = 0.0, cdelt1 = 0.0, cdelt2 = 0.0;
                std::string cunit1 = "deg", cunit2 = "deg";
                hdu.readKey("CRPIX1", crpix1);
                hdu.readKey("CRPIX2", crpix2);
                hdu.readKey("CRVAL1", crval1);
                hdu.readKey("CRVAL2", crval2);
                hdu.readKey("CDELT1", cdelt1);
                hdu.readKey("CDELT2", cdelt2);
                try {
                    hdu.readKey("CUNIT1", cunit1);
                } catch (...) {}
                try {
                    hdu.readKey("CUNIT2", cunit2);
                } catch (...) {}

                auto s1 = unit_to_rad(cunit1);
                auto s2 = unit_to_rad(cunit2);

                wcs.crpix1_pix = crpix1 - 1.0;
                wcs.crpix2_pix = crpix2 - 1.0;
                wcs.crval1_rad = crval1 * s1;
                wcs.crval2_rad = crval2 * s2;
                wcs.cdelt1_rad = cdelt1 * s1;
                wcs.cdelt2_rad = cdelt2 * s2;
                wcs.has_wcs = (std::abs(wcs.cdelt1_rad) > 0 && std::abs(wcs.cdelt2_rad) > 0);

                // fitsIO::get_hdu flips x for non-altaz; mirror WCS to match internal matrix orientation.
                if (pixel_axes != "altaz") {
                    auto nx = static_cast<double>(images.back().cols());
                    auto old_crpix1 = wcs.crpix1_pix;
                    wcs.crpix1_pix = (nx - 1.0) - old_crpix1;
                    wcs.cdelt1_rad *= -1.0;
                }
            } catch (...) {
                wcs.has_wcs = false;
            }
            image_wcs.push_back(wcs);
        }
    }
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
        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
                                                          pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        // distance to source to truncate it
        auto dist = ((lat.array()).pow(2) + (lon.array()).pow(2)).sqrt();

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
    double off_lat = 0.0;
    double off_lon = 0.0;

    // beam standard deviations
    double sigma_lat = sigma_rad;
    double sigma_lon = sigma_rad;

    for (Eigen::Index i=0; i<n_dets; ++i) {
        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
                                                          pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        // distance to source to truncate it
        auto dist = ((lat.array()).pow(2) + (lon.array()).pow(2)).sqrt();

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

template<typename apt_t>
void Kernel::create_airy_kernel(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes, apt_t &apt) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.kernel.data.resize(n_pts,n_dets);

    double fwhm = fwhm_rad;

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
                                                          pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        // distance to source to truncate it
        auto dist = ((lat.array()).pow(2) + (lon.array()).pow(2)).sqrt();

        // get fwhm from apt if config file fwhm is <= 0
        if (fwhm_rad <= 0) {
            fwhm = ASEC_TO_RAD*(apt["a_fwhm"](i) + apt["b_fwhm"](i))/2;
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
void Kernel::create_kernel_from_fits(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string &pixel_axes, apt_t &apt,
                                     double pixel_size_rad, Eigen::DenseBase<Derived> &map_indices) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.kernel.data.resize(n_pts,n_dets);
    in.kernel.data.setZero();
    if (use_wcs_reprojection) {
        ensure_reprojected_images(pixel_size_rad);
    }

    Eigen::Index map_index = 0;

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
                                                          pixel_axes, in.pointing_offsets_arcsec.data, map_grouping);

        if (images.size() > 1) {
            map_index = map_indices(i);
        }

        const auto &img = use_wcs_reprojection ? reprojected_images[map_index] : images[map_index];
        double row0 = (static_cast<double>(img.rows()) - 1.0) / 2.0;
        double col0 = (static_cast<double>(img.cols()) - 1.0) / 2.0;

        // get map buffer row and col indices for lat and lon vectors
        Eigen::VectorXd irows = lat.array()/pixel_size_rad + row0;
        Eigen::VectorXd icols = lon.array()/pixel_size_rad + col0;

        for (Eigen::Index j = 0; j<n_pts; ++j) {
            if (use_wcs_reprojection) {
                in.kernel.data(j,i) = bilinear_sample(img, irows(j), icols(j));
            }
            else {
                in.kernel.data(j,i) = nearest_sample(img, irows(j), icols(j));
            }
        }
    }
}
} // namespace timestream

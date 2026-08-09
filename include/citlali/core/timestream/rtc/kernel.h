#pragma once

#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

#include <boost/math/special_functions/bessel.hpp>

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/timestream_invariant_validation.h>
#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/pointing.h>
#include <citlali/core/utils/fits_io.h>
#include <citlali/core/utils/sha256.h>

namespace timestream {

enum class RTCSourceMaskAdmissionStatus {
    not_requested,
    admitted,
    unavailable_mode,
    unavailable_radius,
    unavailable_frame,
    unavailable_shape,
    unavailable_coordinates,
    unavailable_detector_identity,
    unavailable_validity,
};

inline constexpr std::string_view rtc_source_mask_admission_status_name(
    RTCSourceMaskAdmissionStatus status) {
    switch (status) {
        case RTCSourceMaskAdmissionStatus::not_requested:
            return "not_requested";
        case RTCSourceMaskAdmissionStatus::admitted:
            return "admitted";
        case RTCSourceMaskAdmissionStatus::unavailable_mode:
            return "unavailable_mode";
        case RTCSourceMaskAdmissionStatus::unavailable_radius:
            return "unavailable_radius";
        case RTCSourceMaskAdmissionStatus::unavailable_frame:
            return "unavailable_frame";
        case RTCSourceMaskAdmissionStatus::unavailable_shape:
            return "unavailable_shape";
        case RTCSourceMaskAdmissionStatus::unavailable_coordinates:
            return "unavailable_coordinates";
        case RTCSourceMaskAdmissionStatus::unavailable_detector_identity:
            return "unavailable_detector_identity";
        case RTCSourceMaskAdmissionStatus::unavailable_validity:
            return "unavailable_validity";
    }
    return "unavailable_validity";
}

struct RTCSourceMaskAdmission {
    RTCSourceMaskAdmissionStatus status =
        RTCSourceMaskAdmissionStatus::not_requested;
    std::string identity;
    std::string frame;
    std::string detector_ordering = "apt_uid_column_order";
    std::string reason;

    [[nodiscard]] bool admitted() const {
        return status == RTCSourceMaskAdmissionStatus::admitted;
    }
};

template <class Values>
bool rtc_source_mask_values_exact_shape_and_finite(
    const Values &values, Eigen::Index expected_size) {
    return values.size() == expected_size && values.allFinite();
}

template <class TCData, class Apt>
RTCSourceMaskAdmission admit_rtc_source_mask(
    const TCData &in, const Apt &apt, const std::string &pixel_axes,
    const std::string &map_grouping, const std::string &mode,
    double radius_arcsec) {
    RTCSourceMaskAdmission result;
    result.frame = pixel_axes;
    if (mode.empty() || mode == "none") {
        return result;
    }
    if (mode != "map_center_radius" &&
        mode != "pointing_center_radius") {
        result.status = RTCSourceMaskAdmissionStatus::unavailable_mode;
        result.reason = "unsupported source-mask mode";
        return result;
    }
    if (!std::isfinite(radius_arcsec) || radius_arcsec <= 0.0) {
        result.status = RTCSourceMaskAdmissionStatus::unavailable_radius;
        result.reason = "source-mask radius must be finite and positive";
        return result;
    }
    const bool known_frame =
        citlali::config::is_radec_map_pixel_axes(pixel_axes) ||
        citlali::config::is_altaz_map_pixel_axes(pixel_axes) ||
        citlali::config::is_galactic_map_pixel_axes(pixel_axes);
    const bool known_grouping =
        citlali::config::is_network_map_grouping(map_grouping) ||
        citlali::config::is_array_map_grouping(map_grouping) ||
        citlali::config::is_detector_map_grouping(map_grouping) ||
        citlali::config::is_frequency_group_map_grouping(map_grouping);
    if (!known_frame || !known_grouping) {
        result.status = RTCSourceMaskAdmissionStatus::unavailable_frame;
        result.reason = "source-mask frame or grouping is unavailable";
        return result;
    }

    const Eigen::Index n_samples = in.scans.data.rows();
    const Eigen::Index n_detectors = in.scans.data.cols();
    if (n_samples <= 0 || n_detectors <= 0) {
        result.status = RTCSourceMaskAdmissionStatus::unavailable_shape;
        result.reason = "source-mask signal shape is empty";
        return result;
    }
    const auto require_apt = [&](const char *name)
        -> const typename Apt::mapped_type * {
        const auto it = apt.find(name);
        return it == apt.end() ? nullptr : &it->second;
    };
    const auto *uid = require_apt("uid");
    const auto *flag = require_apt("flag");
    const auto *x_t = require_apt("x_t");
    const auto *y_t = require_apt("y_t");
    if (uid == nullptr || flag == nullptr || x_t == nullptr ||
        y_t == nullptr || uid->size() != n_detectors ||
        flag->size() != n_detectors || x_t->size() != n_detectors ||
        y_t->size() != n_detectors) {
        result.status =
            RTCSourceMaskAdmissionStatus::unavailable_detector_identity;
        result.reason = "source-mask detector identity shape is unavailable";
        return result;
    }
    if (!uid->allFinite() || !flag->allFinite() || !x_t->allFinite() ||
        !y_t->allFinite()) {
        result.status = RTCSourceMaskAdmissionStatus::unavailable_validity;
        result.reason = "source-mask detector identity is non-finite";
        return result;
    }
    std::set<long long> detector_uids;
    for (Eigen::Index detector = 0; detector < n_detectors; ++detector) {
        const double value = (*uid)(detector);
        if (value < static_cast<double>(
                        std::numeric_limits<long long>::min()) ||
            value > static_cast<double>(
                        std::numeric_limits<long long>::max())) {
            result.status =
                RTCSourceMaskAdmissionStatus::unavailable_detector_identity;
            result.reason = "source-mask detector UID is out of range";
            return result;
        }
        const auto integer_uid = static_cast<long long>(std::llround(value));
        if (static_cast<double>(integer_uid) != value ||
            !detector_uids.insert(integer_uid).second) {
            result.status =
                RTCSourceMaskAdmissionStatus::unavailable_detector_identity;
            result.reason = "source-mask detector UID is non-integral or duplicate";
            return result;
        }
    }

    std::vector<std::string> telescope_fields{"TelElAct"};
    if (citlali::config::is_radec_map_pixel_axes(pixel_axes)) {
        telescope_fields.insert(
            telescope_fields.end(), {"ActParAng", "dec_phys", "ra_phys"});
    }
    else if (citlali::config::is_altaz_map_pixel_axes(pixel_axes)) {
        telescope_fields.insert(
            telescope_fields.end(), {"alt_phys", "az_phys"});
    }
    else {
        telescope_fields.insert(
            telescope_fields.end(),
            {"ActParAng", "ActGalAng", "b_phys", "l_phys"});
    }
    for (const auto &field : telescope_fields) {
        const auto it = in.tel_data.data.find(field);
        if (it == in.tel_data.data.end() ||
            !rtc_source_mask_values_exact_shape_and_finite(
                it->second, n_samples)) {
            result.status =
                RTCSourceMaskAdmissionStatus::unavailable_coordinates;
            result.reason = "source-mask telescope coordinate is unavailable: " +
                            field;
            return result;
        }
    }
    for (const std::string field : {"az", "alt"}) {
        const auto it = in.pointing_offsets_arcsec.data.find(field);
        if (it == in.pointing_offsets_arcsec.data.end() ||
            !rtc_source_mask_values_exact_shape_and_finite(
                it->second, n_samples)) {
            result.status =
                RTCSourceMaskAdmissionStatus::unavailable_coordinates;
            result.reason = "source-mask pointing coordinate is unavailable: " +
                            field;
            return result;
        }
    }

    std::ostringstream identity;
    identity << "SCI-RTC-001-source-mask-v1|mode=" << mode
             << "|frame=" << pixel_axes
             << "|grouping=" << map_grouping
             << "|radius_arcsec=" << std::hexfloat << radius_arcsec
             << "|samples=" << n_samples
             << "|detectors=" << n_detectors;
    const auto append_values = [&](std::string_view name,
                                   const auto &values) {
        identity << '|' << name << '=';
        for (Eigen::Index index = 0; index < values.size(); ++index) {
            identity << std::hexfloat << values(index) << ',';
        }
    };
    append_values("uid", *uid);
    append_values("flag", *flag);
    append_values("x_t", *x_t);
    append_values("y_t", *y_t);
    for (const auto &field : telescope_fields) {
        append_values(field, in.tel_data.data.at(field));
    }
    append_values("pointing_az", in.pointing_offsets_arcsec.data.at("az"));
    append_values("pointing_alt", in.pointing_offsets_arcsec.data.at("alt"));

    result.status = RTCSourceMaskAdmissionStatus::admitted;
    result.identity = "sha256:" + citlali::utils::sha256(identity.str());
    return result;
}

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
    Eigen::VectorXd source_a_fwhm_rad;
    Eigen::VectorXd source_b_fwhm_rad;
    Eigen::VectorXi source_valid;

    void clear_source_centers();
    void set_source_centers(const Eigen::VectorXd &, const Eigen::VectorXd &,
                            const Eigen::VectorXi &);
    void set_source_centers(const Eigen::VectorXd &, const Eigen::VectorXd &,
                            const Eigen::VectorXi &, const Eigen::VectorXd &,
                            const Eigen::VectorXd &);
    bool has_source_centers() const;
    bool source_center_for_map(Eigen::Index, double &, double &) const;
    bool source_fwhm_for_map(Eigen::Index, double &, double &) const;

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
    // Observation setup rebuilds the effective response inventory. Retaining
    // FITS images from a prior observation would make an otherwise identical
    // response acquire a different bundle identity and could apply stale
    // response planes.
    images.clear();
    if (type == "fits") {
        if (img_ext_names.size()!=n_maps && img_ext_names.size()!=1) {
            SPDLOG_INFO("mismatch for number of kernel images");
            citlali::pipeline::require_kernel_image_cardinality(
                img_ext_names.size(), n_maps);
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
    source_a_fwhm_rad.resize(0);
    source_b_fwhm_rad.resize(0);
    source_valid.resize(0);
}

inline void Kernel::set_source_centers(const Eigen::VectorXd &lat,
                                       const Eigen::VectorXd &lon,
                                       const Eigen::VectorXi &valid) {
    source_lat = lat;
    source_lon = lon;
    source_valid = valid;
    source_a_fwhm_rad.resize(0);
    source_b_fwhm_rad.resize(0);
}

inline void Kernel::set_source_centers(const Eigen::VectorXd &lat,
                                       const Eigen::VectorXd &lon,
                                       const Eigen::VectorXi &valid,
                                       const Eigen::VectorXd &a_fwhm_rad,
                                       const Eigen::VectorXd &b_fwhm_rad) {
    source_lat = lat;
    source_lon = lon;
    source_valid = valid;
    source_a_fwhm_rad = a_fwhm_rad;
    source_b_fwhm_rad = b_fwhm_rad;
}

inline bool Kernel::has_source_centers() const {
    return citlali::config::is_detector_map_grouping(map_grouping) &&
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
    if (!citlali::config::is_detector_map_grouping(map_grouping) ||
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

inline bool Kernel::source_fwhm_for_map(Eigen::Index map_index,
                                        double &a_fwhm_rad,
                                        double &b_fwhm_rad) const {
    a_fwhm_rad = std::numeric_limits<double>::quiet_NaN();
    b_fwhm_rad = std::numeric_limits<double>::quiet_NaN();
    if (!citlali::config::is_detector_map_grouping(map_grouping) ||
        map_index < 0 ||
        map_index >= source_valid.size() ||
        map_index >= source_a_fwhm_rad.size() ||
        map_index >= source_b_fwhm_rad.size() ||
        source_valid(map_index) == 0) {
        return false;
    }
    const double a = source_a_fwhm_rad(map_index);
    const double b = source_b_fwhm_rad(map_index);
    if (!std::isfinite(a) || !std::isfinite(b) || a <= 0.0 || b <= 0.0) {
        return false;
    }
    a_fwhm_rad = a;
    b_fwhm_rad = b;
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
            double source_a_fwhm = 0.0;
            double source_b_fwhm = 0.0;
            if (source_fwhm_for_map(i, source_a_fwhm, source_b_fwhm)) {
                sigma = FWHM_TO_STD * (source_a_fwhm + source_b_fwhm) / 2.0;
            }
            else {
                sigma = FWHM_TO_STD * ASEC_TO_RAD*(apt["a_fwhm"](i) + apt["b_fwhm"](i))/2;
            }
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
            double source_a_fwhm = 0.0;
            double source_b_fwhm = 0.0;
            if (source_fwhm_for_map(i, source_a_fwhm, source_b_fwhm)) {
                sigma_lat = FWHM_TO_STD * source_b_fwhm;
                sigma_lon = FWHM_TO_STD * source_a_fwhm;
            }
            else {
                sigma_lat = FWHM_TO_STD * ASEC_TO_RAD * apt["b_fwhm"](i);
                sigma_lon = FWHM_TO_STD * ASEC_TO_RAD * apt["a_fwhm"](i);
            }
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
            double source_a_fwhm = 0.0;
            double source_b_fwhm = 0.0;
            if (source_fwhm_for_map(i, source_a_fwhm, source_b_fwhm)) {
                fwhm = (source_a_fwhm + source_b_fwhm) / 2.0;
            }
            else {
                fwhm = ASEC_TO_RAD*(apt["a_fwhm"](i) + apt["b_fwhm"](i))/2;
            }
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

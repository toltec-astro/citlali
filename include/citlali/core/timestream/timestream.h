#pragma once

#include <map>

#include <tula/enum.h>
#include <tula/nddata/labelmapper.h>
#include <tula/formatter/enum.h>
#include <tula/formatter/matrix.h>
#include <tula/formatter/utils.h>
#include <tula/logging.h>
#include <kids/core/wcs.h>

#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>
#include <citlali/core/utils/toltec_io.h>

namespace timestream {

enum TimestreamFlags {
    Good = 0,
    D21FitsBetter   = 1 << 0,
    D21LargeOffset  = 1 << 1,
    D21NotConverged = 1 << 2,
    D21OutOfRange   = 1 << 3,
    D21QrOutOfRange = 1 << 4,
    LargeOffset     = 1 << 5,
    NotConverged    = 1 << 6,
    OutOfRange      = 1 << 7,
    QrOutOfRange    = 1 << 8,
    LowGain         = 1 << 9,
    APT             = 1 << 10,
    Spike           = 1 << 11,
    Freq            = 1 << 12
};

namespace wcs = kids::wcs;

// clang-format off
TULA_BITFLAG(TCDataKind, int,  0xFFFF,
             RTC               = 1 << 0,
             PTC               = 1 << 1,
             Any               = RTC | PTC
             );
// clang-format on

/// @brief TC data class.
template <TCDataKind kind_ = TCDataKind::Any, typename = void>
struct TCData;

} // namespace timestream

namespace std {

// register TCData as a variant type
// below are mandatory to inherit from variant on gcc
template <timestream::TCDataKind kind>
struct variant_size<timestream::TCData<kind>>
    : variant_size<typename timestream::TCData<kind>::variant_t> {};

template <size_t _Np, auto kind>
struct variant_alternative<_Np, timestream::TCData<kind>>
    : variant_alternative<_Np, typename timestream::TCData<kind>::variant_t> {};

#if defined(__GNUC__) && !defined(__clang__)
#if (__GNUC__ >= 9)
// this is need to allow inherit from std::variant on GCC
namespace __detail {
namespace __variant {

template <typename _Ret, typename _Visitor, auto kind, size_t __first>
struct _Multi_array<_Ret (*)(_Visitor, timestream::TCData<kind>), __first>
    : _Multi_array<_Ret (*)(_Visitor, typename timestream::TCData<kind>::variant_t),
                   __first> {
    static constexpr int __do_cookie = 0;
};
template <typename _Maybe_variant_cookie, auto kind>
struct _Extra_visit_slot_needed<_Maybe_variant_cookie, timestream::TCData<kind>>
    : _Extra_visit_slot_needed<_Maybe_variant_cookie,
                               typename timestream::TCData<kind>::variant_t> {};

template <typename _Maybe_variant_cookie, auto kind>
struct _Extra_visit_slot_needed<_Maybe_variant_cookie, timestream::TCData<kind> &>
    : _Extra_visit_slot_needed<_Maybe_variant_cookie,
                               typename timestream::TCData<kind>::variant_t &> {};
} // namespace __variant
} // namespace __detail
#else
#endif
#endif

} // namespace std

namespace timestream {

namespace internal {

template <typename Derived>
struct TCDataBase;

template <TCDataKind kind_>
struct TCDataBase<TCData<kind_>> {
    static constexpr auto kind() { return kind_; }
    //using meta_t = typename internal::impl_traits<TCData<kind_>>::meta_t;
    //meta_t meta;
};

} // namespace internal

// wcs objects
struct DetectorAxis : wcs::Axis<DetectorAxis, wcs::CoordsKind::Column>,
                      wcs::LabeledData<DetectorAxis> {
    DetectorAxis() = default;
    DetectorAxis(Eigen::MatrixXd data_, tula::nddata::LabelMapper<DetectorAxis> row_labels_)
        : data{std::move(data_)}, row_labels{std::move(row_labels_)} {}
    std::string_view name{"detector"};
    Eigen::MatrixXd data;
    tula::nddata::LabelMapper<DetectorAxis> row_labels;
};

struct TimeAxis : wcs::Axis<TimeAxis, wcs::CoordsKind::Row>,
                  wcs::LabeledData<TimeAxis> {
    TimeAxis() = default;
    TimeAxis(Eigen::MatrixXd data_, tula::nddata::LabelMapper<TimeAxis> col_labels_)
        : data{std::move(data_)}, col_labels{std::move(col_labels_)} {}
    std::string_view name{"time"};
    Eigen::MatrixXd data;
    tula::nddata::LabelMapper<TimeAxis> col_labels;
};

struct TimeStreamFrame : wcs::Frame2D<TimeStreamFrame, TimeAxis, DetectorAxis> {
    TimeAxis time_axis;
    DetectorAxis detector_axis;

    // Frame2D impl
    const TimeAxis &row_axis() const { return time_axis; }
    const DetectorAxis &col_axis() const { return detector_axis; }
};

// data objects

/// @brief base class for time stream data
template <typename Derived>
struct TimeStream : internal::TCDataBase<Derived>,
                    tula::nddata::NDData<TimeStream<Derived>> {
    TimeStreamFrame wcs;
    // The timestream is stored in row major for efficient r/w
    template <typename PlainObject>
    struct dataref_t : tula::nddata::NDData<dataref_t<PlainObject>> {
        PlainObject data{nullptr, 0, 0};
    };

    template <typename PlainObject>
    struct data_t : tula::nddata::NDData<data_t<PlainObject>> {
        PlainObject data;
    };

    // time of creation
    std::string creation_time = engine_utils::current_date_time();

    // data status
    bool demodulated = false;
    bool kernel_generated = false;
    bool despiked = false;
    bool tod_filtered = false;
    bool downsampled = false;
    bool calibrated = false;
    bool cleaned = false;

    // number of detectors lower than weight limit
    int n_low_dets;
    // number of detectors higher than weight limit
    int n_high_dets;

    // kernel timestreams
    data_t<Eigen::MatrixXd> kernel;
    // flag timestream
    data_t<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> flags;
    // bitwise flags
    data_t<Eigen::Matrix<TimestreamFlags,Eigen::Dynamic,Eigen::Dynamic>> flags2;
    // current scan indices
    data_t<Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1>> scan_indices;
    // scan index
    data_t<Eigen::Index> index;
    // telescope data for scan
    data_t<std::map<std::string, Eigen::VectorXd>> tel_data;
    // pointing offsets
    data_t<std::map<std::string, Eigen::VectorXd>> pointing_offsets_arcsec;
    // hwp angle for scan
    data_t<Eigen::VectorXd> hwp_angle;
    // fcf
    data_t<Eigen::VectorXd> fcf;
    // vectors for mapping apt table onto timestreams
    data_t<Eigen::VectorXI> det_indices, nw_indices, array_indices, map_indices;
};

template <typename RefType>
struct TCData<TCDataKind::RTC,RefType>
    : TimeStream<TCData<TCDataKind::RTC>> {
    using Base = TimeStream<TCData<TCDataKind::RTC>>;
    using data_t = std::conditional_t<tula::eigen_utils::is_plain_v<RefType>,Base::data_t<RefType>,
                                      Base::dataref_t<RefType>>;
    // time chunk type
    std::string_view name{"RTC"};
    // data timestreams
    data_t scans;
};

template <typename RefType>
struct TCData<TCDataKind::PTC, RefType>
    : TimeStream<TCData<TCDataKind::PTC>> {
    using Base = TimeStream<TCData<TCDataKind::PTC>>;
    using data_t = std::conditional_t<tula::eigen_utils::is_plain_v<RefType>,Base::data_t<RefType>,
                                      Base::dataref_t<RefType>>;
    // time chunk type
    std::string_view name{"PTC"};
    // data timestreams
    data_t scans;
    // weights for current scan
    Base::data_t<Eigen::VectorXd> weights;
    // eigenvalues for scan
    Base::data_t<std::vector<std::vector<Eigen::VectorXd>>> evals;
    // eigenvectors for scan
    Base::data_t<std::vector<std::vector<Eigen::MatrixXd>>> evecs;
    // medians of good detector weights
    Base::data_t<std::vector<double>> median_weights;
};

/// @brief data class of runtime variant kind.
template <TCDataKind kind_>
struct TCData<kind_, std::enable_if_t<tula::enum_utils::is_compound_v<kind_>>>
    : tula::enum_utils::enum_to_variant_t<kind_, TCData> {
    using Base = tula::enum_utils::enum_to_variant_t<kind_, TCData>;
    using variant_t = tula::enum_utils::enum_to_variant_t<kind_, TCData>;

    const variant_t &variant() const { return *this; }
    static constexpr auto kind() { return kind_; }
};

// class for tod processing
class TCProc {
public:
    // toltec io class for array names
    engine_utils::toltecIO toltec_io;

    // add or subtract gaussian source
    enum GaussType {
        add = 0,
        subtract = 1
    };

    // number of weight outlier iterations
    int iter_lim = 0;

    // upper and lower limits for outliers
    double lower_weight_factor, upper_weight_factor;

    // get limits for a particular grouping
    template <typename Derived, class calib_t>
    auto get_grouping(std::string, Eigen::DenseBase<Derived> &, calib_t &, int);

    // remove detectors with outlier weights
    template <TCDataKind tcdata_t, typename calib_t, typename Derived>
    auto remove_bad_dets(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                         Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);

    // add or subtract gaussian to timestream
    template <GaussType gauss_type, TCDataKind tcdata_t, typename DerivedB, typename DerivedC, typename apt_t, typename pointing_offset_t>
    void add_gaussian(TCData<tcdata_t, Eigen::MatrixXd> &, Eigen::DenseBase<DerivedB> &, std::string &,
                      std::string &, apt_t &, pointing_offset_t &, double,
                      Eigen::Index, Eigen::Index, Eigen::DenseBase<DerivedC> &, Eigen::DenseBase<DerivedC> &);
};

template <typename Derived, class calib_t>
auto TCProc::get_grouping(std::string grp, Eigen::DenseBase<Derived> &det_indices, calib_t &calib, int n_dets) {
    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

    Eigen::Index grp_i = calib.apt[grp](det_indices(0));
    grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};
    Eigen::Index j = 0;
    // loop through apt table arrays, get highest index for current array
    for (Eigen::Index i=0; i<n_dets; i++) {
        auto det_index = det_indices(i);
        if (calib.apt[grp](det_index) == grp_i) {
            std::get<1>(grp_limits[grp_i]) = i + 1;
        }
        else {
            grp_i = calib.apt[grp](det_index);
            j += 1;
            grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{i, 0};
        }
    }
    return grp_limits;
}

template <TCDataKind tcdata_t, typename calib_t, typename Derived>
auto TCProc::remove_bad_dets(TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                              Eigen::DenseBase<Derived> &nw_indices, Eigen::DenseBase<Derived> &array_indices, std::string redu_type,
                              std::string map_grouping) {


    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // only run if limits are not zero
    if (lower_weight_factor !=0 || upper_weight_factor !=0) {
        // number of detectors
        Eigen::Index n_dets = in.scans.data.cols();

        std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

        Eigen::Index grp_i = calib.apt["array"](det_indices(0));
        grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};
        Eigen::Index j = 0;
        // loop through apt table arrays, get highest index for current array
        for (Eigen::Index i=0; i<n_dets; i++) {
            auto det_index = det_indices(i);
            if (calib.apt["array"](det_index) == grp_i) {
                std::get<1>(grp_limits[grp_i]) = i + 1;
            }
            else {
                grp_i = calib.apt["array"](det_index);
                j += 1;
                grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{i, 0};
            }
        }

        in.n_low_dets = 0;
        in.n_high_dets = 0;

        for (auto const& [key, val] : grp_limits) {

            bool keep_going = true;
            Eigen::Index n_iter = 0;

            while (keep_going) {
                // number of unflagged detectors
                Eigen::Index n_good_dets = 0;

                for (Eigen::Index j=std::get<0>(grp_limits[key]); j<std::get<1>(grp_limits[key]); j++) {
                    if (calib.apt["flag"](det_indices(j))==0) {
                        n_good_dets++;
                    }
                }

                Eigen::VectorXd det_std_dev(n_good_dets);
                Eigen::VectorXI dets(n_good_dets);
                Eigen::Index k = 0;

                // collect standard deviation from good detectors
                for (Eigen::Index j=std::get<0>(grp_limits[key]); j<std::get<1>(grp_limits[key]); j++) {
                    Eigen::Index det_index = det_indices(j);
                    if (calib.apt["flag"](det_index)==0) {
                        // make Eigen::Maps for each detector's scan
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                            in.scans.data.col(j).data(), in.scans.data.rows());
                        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                            in.flags.data.col(j).data(), in.flags.data.rows());

                        // calc standard deviation
                        det_std_dev(k) = engine_utils::calc_std_dev(scans, flags);

                        // convert to 1/variance
                        if (det_std_dev(k) !=0) {
                            det_std_dev(k) = std::pow(det_std_dev(k),-2);
                        }
                        else {
                            det_std_dev(k) = 0;
                        }

                        dets(k) = j;
                        k++;
                    }
                }

                // get median standard deviation
                double mean_std_dev = tula::alg::median(det_std_dev);

                int n_low_dets = 0;
                int n_high_dets = 0;

                // loop through good detectors and flag those that have std devs beyond the limits
                for (Eigen::Index j=0; j<n_good_dets; j++) {
                    Eigen::Index det_index = det_indices(dets(j));
                    // only run if unflagged already
                    if (calib.apt["flag"](det_index)==0) {
                        // flag those below limit
                        if ((det_std_dev(j) < (lower_weight_factor*mean_std_dev)) && lower_weight_factor!=0) {
                            if (map_grouping!="detector") {
                                in.flags.data.col(dets(j)).setOnes();
                            }
                            else {
                                calib_scan.apt["flag"](det_index) = 1;
                            }
                            in.n_low_dets++;
                            n_low_dets++;
                        }

                        // flag those above limit
                        if ((det_std_dev(j) > (upper_weight_factor*mean_std_dev)) && upper_weight_factor!=0) {
                            if (map_grouping!="detector") {
                                in.flags.data.col(dets(j)).setOnes();
                            }
                            else {
                                calib_scan.apt["flag"](det_index) = 1;
                            }
                            in.n_high_dets++;
                            n_high_dets++;
                        }
                    }
                }

                SPDLOG_INFO("array {} iter {}: {}/{} dets below limit. {}/{} dets above limit.", key, n_iter,
                            n_low_dets, n_good_dets, n_high_dets, n_good_dets);

                // increment iteration
                n_iter++;
                // check if no more detectors are above limit
                if ((n_low_dets==0 && n_high_dets==0) || n_iter > iter_lim) {
                    keep_going = false;
                }
            }
        }
    }

    // set up scan calib
    calib_scan.setup();

    return std::move(calib_scan);
}

template <TCProc::GaussType gauss_type, TCDataKind tcdata_t, typename DerivedB, typename DerivedC, typename apt_t, typename pointing_offset_t>
void TCProc::add_gaussian(TCData<tcdata_t, Eigen::MatrixXd> &in, Eigen::DenseBase<DerivedB> &params, std::string &pixel_axes,
                           std::string &map_grouping, apt_t &apt, pointing_offset_t &pointing_offsets_arcsec,
                           double pixel_size_rad, Eigen::Index n_rows, Eigen::Index n_cols,
                           Eigen::DenseBase<DerivedC> &map_indices, Eigen::DenseBase<DerivedC> &det_indices) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; i++) {
        double azoff, eloff;

        auto det_index = det_indices(i);
        auto map_index = map_indices(i);

        double az_off = 0;
        double el_off = 0;

        if (map_grouping!="detector") {
            az_off = apt["x_t"](det_index);
            el_off = apt["y_t"](det_index);
        }

        // get pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off,
                                                          pixel_axes, pointing_offsets_arcsec);

        // get parameters for current detector
        double amp = params(map_index,0);
        double off_lat = params(map_index,2);
        double off_lon = params(map_index,1);
        double sigma_lat = params(map_index,4);
        double sigma_lon = params(map_index,3);
        double rot_ang = params(map_index,5);

        // use maximum of sigmas due to atmospheric cleaning
        double sigma = std::max(sigma_lat, sigma_lon);

        // subtract source
        if constexpr (gauss_type==subtract) {
            amp = -amp;
        }

        // rescale offsets and stddev to on-sky units
        off_lat = pixel_size_rad*(off_lat - (n_rows)/2);
        off_lon = pixel_size_rad*(off_lon - (n_cols)/2);

        // convert to on-sky units
        sigma_lon = pixel_size_rad*sigma;
        sigma_lat = pixel_size_rad*sigma;
        sigma = pixel_size_rad*sigma;

        // get angles
        auto cost2 = cos(rot_ang) * cos(rot_ang);
        auto sint2 = sin(rot_ang) * sin(rot_ang);
        auto sin2t = sin(2. * rot_ang);
        auto xstd2 = sigma * sigma;
        auto ystd2 = sigma * sigma;
        auto a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        auto b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        auto c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

        // calculate distance to source to truncate it
        auto dist = ((lat.array() - off_lat).pow(2) + (lon.array() - off_lon).pow(2)).sqrt();

        Eigen::VectorXd gauss(n_pts);
        // make gaussian
        for (Eigen::Index j=0; j<n_pts; j++) {
            gauss(j) = amp*exp(pow(lon(j) - off_lon, 2) * a +
                                 (lon(j) - off_lon) * (lat(j) - off_lat) * b +
                                 pow(lat(j) - off_lat, 2) * c);
        }

        if (!gauss.array().isNaN().any()) {
            // add gaussian to detector scan
            in.scans.data.col(i) = in.scans.data.col(i).array() + gauss.array();
        }
    }
}

} // namespace timestream

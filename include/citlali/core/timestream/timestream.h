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
    struct {
        bool demodulated = false;
        bool kernel_generated = false;
        bool despiked = false;
        bool tod_filtered = false;
        bool downsampled = false;
        bool calibrated = false;
        bool extinction_corrected = false;
        bool cleaned = false;
    } status;

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
    data_t<Eigen::VectorXd> hwpr_angle;
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
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

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

    // mask radius in arcseconds
    double mask_radius_arcsec;

    // get limits for a particular grouping
    template <typename Derived, class calib_t>
    auto get_grouping(std::string, Eigen::DenseBase<Derived> &, calib_t &, int);

    // remove detectors with outlier weights
    template <TCDataKind tcdata_t, typename calib_t, typename Derived>
    auto remove_bad_dets(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                         Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);

    // add or subtract gaussian to timestream
    template <GaussType gauss_type, TCDataKind tcdata_t, typename DerivedB, typename DerivedC, typename apt_t,
             typename pointing_offset_t>
    void add_gaussian(TCData<tcdata_t, Eigen::MatrixXd> &, Eigen::DenseBase<DerivedB> &, std::string &,
                      std::string &, apt_t &, pointing_offset_t &, double, Eigen::Index, Eigen::Index,
                      Eigen::DenseBase<DerivedC> &, Eigen::DenseBase<DerivedC> &);

    // flag a region around the center of the map
    template <TCDataKind tcdata_t, class calib_t, typename Derived>
    auto mask_region(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                     std::string, std::string, int, int, int);

    // append time chunk params common to rtcs and ptcs
    template <TCDataKind tcdata_t, typename Derived, typename calib_t, typename pointing_offset_t>
    void append_base_to_netcdf(netCDF::NcFile &, TCData<tcdata_t, Eigen::MatrixXd> &, std::string,
                               std::string &, pointing_offset_t &, Eigen::DenseBase<Derived> &,
                               calib_t &);
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

        // get grouping
        auto grp_limits = get_grouping("array",det_indices,calib,n_dets);

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

                logger->info("array {} iter {}: {}/{} dets below limit. {}/{} dets above limit.", key, n_iter,
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
        // detector index in apt
        auto det_index = det_indices(i);
        // map index
        auto map_index = map_indices(i);

        double az_off = apt["x_t"](det_index);
        double el_off = apt["y_t"](det_index);

        // get pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off, pixel_axes,
                                                          pointing_offsets_arcsec, map_grouping);

        // get parameters from current map
        double amp = params(map_index,0);
        // rows
        double off_lat = params(map_index,2);
        // cols
        double off_lon = params(map_index,1);
        // row fwhm
        double sigma_lat = params(map_index,4);
        // col fwhm
        double sigma_lon = params(map_index,3);
        // rot angle
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

        // get 2d elliptical gaussian angles
        auto cost2 = cos(rot_ang) * cos(rot_ang);
        auto sint2 = sin(rot_ang) * sin(rot_ang);
        auto sin2t = sin(2. * rot_ang);
        auto xstd2 = sigma * sigma;
        auto ystd2 = sigma * sigma;
        auto a = - 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        auto b = - 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        auto c = - 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

        // calculate distance to source to truncate it
        //auto dist = ((lat.array() - off_lat).pow(2) + (lon.array() - off_lon).pow(2)).sqrt();

        Eigen::VectorXd gauss(n_pts);
        // make timestream from 2d gaussian
        for (Eigen::Index j=0; j<n_pts; j++) {
            gauss(j) = amp*exp(pow(lon(j) - off_lon, 2) * a +
                                 (lon(j) - off_lon) * (lat(j) - off_lat) * b +
                                 pow(lat(j) - off_lat, 2) * c);
        }

        // check for bad fit?
        if (!gauss.array().isNaN().any()) {
            // add gaussian to detector scan
            in.scans.data.col(i) = in.scans.data.col(i).array() + gauss.array();
        }
    }
}

template <TCDataKind tcdata_t, class calib_t, typename Derived>
auto TCProc::mask_region(TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                         std::string pixel_axes, std::string map_grouping, int n_pts, int n_dets, int start_index) {

    // copy of tel data
    std::map<std::string, Eigen::VectorXd> tel_data_copy;

    // populate copy of tel data
    for (const auto &[key,val]: in.tel_data.data) {
        tel_data_copy[key] = in.tel_data.data[key].segment(0,n_pts);
    }

    // make a copy of det indices
    Eigen::VectorXI det_indices_copy = det_indices.segment(start_index,n_dets);

    // copy of pointing offsets
    std::map<std::string, Eigen::VectorXd> pointing_offset_copy;

    // populate copy of pointing offsets
    for (const auto &[key,val]: in.pointing_offsets_arcsec.data) {
        pointing_offset_copy[key] = in.pointing_offsets_arcsec.data[key].segment(0,n_pts);
    }

    // make a copy of the timestream flags
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> masked_flags = in.flags.data.block(0, start_index, n_pts, n_dets);

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; i++) {
        // current detector index in apt
        auto det_index = det_indices_copy(i);

        double az_off = calib.apt["x_t"](det_index);
        double el_off = calib.apt["y_t"](det_index);

        // calc tangent plane pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(tel_data_copy, az_off, el_off, pixel_axes,
                                                          pointing_offset_copy, map_grouping);

        // distance to center of map
        auto dist = (lat.array().pow(2) + lon.array().pow(2)).sqrt();

        // loop through samples
        for (Eigen::Index j=0; j<n_pts; j++) {
            // flag samples within radius as bad
            if (dist(j) < mask_radius_arcsec*ASEC_TO_RAD) {
                masked_flags(j,i) = 1;
            }
        }
    }

    return std::move(masked_flags);
}

template <TCDataKind tcdata_t, typename Derived, typename calib_t, typename pointing_offset_t>
void TCProc::append_base_to_netcdf(netCDF::NcFile &fo, TCData<tcdata_t, Eigen::MatrixXd> &in, std::string map_grouping,
                                   std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec,
                                   Eigen::DenseBase<Derived> &det_indices, calib_t &calib) {
    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    // tangent plane pointing for each detector
    Eigen::MatrixXd lat(n_pts,n_dets), lon(n_pts,n_dets);

    // loop through detectors and get tangent plane pointing
    for (Eigen::Index i=0; i<n_dets; i++) {
        // detector index in apt
        auto det_index = det_indices(i);
        double az_off = calib.apt["x_t"](det_index);
        double el_off = calib.apt["y_t"](det_index);

        // get tangent pointing
        auto [det_lat, det_lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off, pixel_axes,
                                                                  pointing_offsets_arcsec, map_grouping);
        lat.col(i) = std::move(det_lat);
        lon.col(i) = std::move(det_lon);
    }

    // get variables
    auto vars = fo.getVars();

    // get absolute coords
    double cra, cdec;
    vars.find("SourceRa")->second.getVar(&cra);
    vars.find("SourceDec")->second.getVar(&cdec);

    // get dimensions
    NcDim n_pts_dim = fo.getDim("n_pts");
    NcDim n_dets_dim = fo.getDim("n_dets");

    // number of samples currently in file
    unsigned long n_pts_exists = n_pts_dim.getSize();
    // number of detectors currently in file
    unsigned long n_dets_exists = n_dets_dim.getSize();

    // start indices for data
    std::vector<std::size_t> start_index = {n_pts_exists, 0};
    // size for data
    std::vector<std::size_t> size = {1, TULA_SIZET(n_dets)};

    // start index for telescope data
    std::vector<std::size_t> start_index_tel = {n_pts_exists};
    // size for telescope data
    std::vector<std::size_t> size_tel = {TULA_SIZET(n_pts)};

    // start index for apt table
    std::vector<std::size_t> start_index_apt = {0};
    // size for apt
    std::vector<std::size_t> size_apt = {1};

    // get timestream variables
    NcVar signal_v = fo.getVar("signal");
    NcVar flags_v = fo.getVar("flags");
    NcVar kernel_v = fo.getVar("kernel");

    // detector tangent plane pointing
    NcVar det_lat_v = fo.getVar("det_lat");
    NcVar det_lon_v = fo.getVar("det_lon");

    // detector absolute pointing
    NcVar det_ra_v = fo.getVar("det_ra");
    NcVar det_dec_v = fo.getVar("det_dec");

    // append data (doing this per row is way faster than transposing
    // and populating them at once)
    for (std::size_t i=0; i<TULA_SIZET(n_pts); ++i) {
        start_index[0] = n_pts_exists + i;
        // append scans
        Eigen::VectorXd scans = in.scans.data.row(i);
        signal_v.putVar(start_index, size, scans.data());

        // append flags
        Eigen::VectorXi flags_int = in.flags.data.row(i).template cast<int> ();
        flags_v.putVar(start_index, size, flags_int.data());

        // append kernel
        if (!kernel_v.isNull()) {
            Eigen::VectorXd kernel = in.kernel.data.row(i);
            kernel_v.putVar(start_index, size, kernel.data());
        }

        // append detector latitudes
        Eigen::VectorXd lat_row = lat.row(i);
        det_lat_v.putVar(start_index, size, lat_row.data());

        // append detector longitudes
        Eigen::VectorXd lon_row = lon.row(i);
        det_lon_v.putVar(start_index, size, lon_row.data());

        if (pixel_axes == "icrs") {
            // get absolute pointing
            auto [dec, ra] = engine_utils::tangent_to_abs(lat_row, lon_row, cra, cdec);

            // append detector ra
            det_ra_v.putVar(start_index, size, ra.data());

            // append detector dec
            det_dec_v.putVar(start_index, size, dec.data());
        }
    }

    // append telescope
    for (auto const& x: in.tel_data.data) {
        NcVar tel_data_v = fo.getVar(x.first);
        tel_data_v.putVar(start_index_tel, size_tel, x.second.data());
    }

    // append pointing offsets
    for (auto const& x: in.pointing_offsets_arcsec.data) {
        NcVar offset_v = fo.getVar("pointing_offset_"+x.first);
        offset_v.putVar(start_index_tel, size_tel, x.second.data());
    }

    // append hwpr angle
    if (calib.run_hwpr) {
        NcVar hwpr_v = fo.getVar("hwpr");
        hwpr_v.putVar(start_index_tel, size_tel, in.hwpr_angle.data.data());
    }

    // overwrite apt table (can be updated between beammap iterations)
    for (auto const& x: calib.apt) {
        netCDF::NcVar apt_v = fo.getVar("apt_" + x.first);
        for (std::size_t i=0; i<TULA_SIZET(n_dets_exists); ++i) {
            start_index_apt[0] = i;
            apt_v.putVar(start_index_apt, size_apt, &calib.apt[x.first](det_indices(i)));
        }
    }

    // vector to hold current scan indices
    Eigen::VectorXd scan_indices(2);

    // if not on first scan, grab last scan and add size of current scan
    if (in.index.data > 0) {
        // start indices for data
        std::vector<std::size_t> scan_indices_start_index = {TULA_SIZET(in.index.data-1), 0};
        // size for data
        std::vector<std::size_t> scan_indices_size = {1, 2};
        vars.find("scan_indices")->second.getVar(scan_indices_start_index,scan_indices_size,scan_indices.data());

        scan_indices = scan_indices.array() + in.scans.data.rows();
    }
    // otherwise, use size of this scan
    else {
        scan_indices(0) = 0;
        scan_indices(1) = in.scans.data.rows() - 1;
    }

    // add current scan indices row
    std::vector<std::size_t> scan_indices_start_index = {TULA_SIZET(in.index.data), 0};
    std::vector<std::size_t> scan_indices_size = {1, 2};
    NcVar scan_indices_v = fo.getVar("scan_indices");
    scan_indices_v.putVar(scan_indices_start_index, scan_indices_size,scan_indices.data());
}

} // namespace timestream

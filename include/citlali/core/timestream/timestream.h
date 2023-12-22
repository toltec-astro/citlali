#pragma once

#include <map>
#include <filesystem>

#include <unsupported/Eigen/CXX11/Tensor>

#include <CCfits/CCfits>

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

#include <citlali/core/mapmaking/map.h>

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

// data status
struct Status {
    bool demodulated = false;
    bool kernel_generated = false;
    bool despiked = false;
    bool tod_filtered = false;
    bool downsampled = false;
    bool calibrated = false;
    bool extinction_corrected = false;
    bool cleaned = false;
};

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
    // the timestream is stored in row major for efficient r/w
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

    // number of detectors lower than weight limit
    int n_dets_low, n_dets_high;

    // data status struct
    Status status;

    // kernel timestreams
    data_t<Eigen::MatrixXd> kernel;
    // flag timestream
    data_t<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> flags;
    // noise timestreams
    data_t<Eigen::MatrixXi> noise;
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
    // hwpr angle for scan
    data_t<Eigen::VectorXd> hwpr_angle;
    // detector angle
    data_t<Eigen::MatrixXd> angle;
    // fcf
    data_t<Eigen::VectorXd> fcf;
    // vectors for mapping apt table onto timestreams
    data_t<Eigen::VectorXI> det_indices, nw_indices, array_indices, map_indices;
    // detector pointing
    data_t<std::map<std::string, Eigen::MatrixXd>> pointing;
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
    enum SourceType {
        Gaussian = 0,
        NegativeGaussian = 1,
        Airy = 2,
        NegativeAiry = 3,
        Map = 4,
        NegativeMap = 5
    };

    // tod output
    bool run_tod_output, write_evals;

    // run fruit loops
    bool run_fruit_loops;
    // path for input images
    std::string fruit_loops_path;
    // paths for first set of images
    std::vector<std::string> init_fruit_loops_path;
    // fruit loops type
    std::string fruit_loops_type;
    // number of fruit loops iterations
    int fruit_loops_iters;
    // signal-to-noise cut for fruit loops algorithm
    double fruit_loops_sig2noise;
    // flux density cut for fruit loops algorithm
    double fruit_loops_flux;
    // save all iterations
    bool save_all_iters;

    // map buffer for map to tod approach
    mapmaking::ObsMapBuffer tod_mb;

    // number of weight outlier iterations
    int iter_lim = 0;

    // upper and lower limits for outliers
    double lower_weight_factor, upper_weight_factor;

    // mask radius in arcseconds
    double mask_radius_arcsec;

    // create a map buffer from a citlali reduction directory
    template <class calib_t>
    void load_mb(std::string, std::string, calib_t &);

    // get limits for a particular grouping
    template <typename Derived, class calib_t>
    auto get_grouping(std::string, Eigen::DenseBase<Derived> &, calib_t &, int);

    // compute and store pointing of all detectors
    template <TCDataKind tcdata_t, class calib_t, typename Derived>
    void precompute_pointing(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                             std::string, std::string);

    // translate citlali map buffer to timestream and add/subtract from TCData scans
    template <TCProc::SourceType source_type, class mb_t, TCDataKind tcdata_t, class calib_t, typename Derived>
    void map_to_tod(mb_t &, TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                    Eigen::DenseBase<Derived> &, std::string, std::string);

    // remove detectors with outlier weights
    template <TCDataKind tcdata_t, class calib_t, typename Derived>
    auto remove_bad_dets(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                         std::string);

    // add or subtract gaussian to timestream
    template <SourceType source_type, TCDataKind tcdata_t, typename Derived, typename apt_t>
    void add_gaussian(TCData<tcdata_t, Eigen::MatrixXd> &, Eigen::DenseBase<Derived> &, std::string &,
                      std::string &, apt_t &, double, Eigen::Index, Eigen::Index);

    // flag a region around the center of the map
    template <TCDataKind tcdata_t, class calib_t, typename Derived>
    auto mask_region(TCData<tcdata_t, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                     std::string, std::string, int, int, int);

    // append time chunk params common to rtcs and ptcs
    template <TCDataKind tcdata_t, typename Derived, class calib_t, typename pointing_offset_t>
    void append_base_to_netcdf(netCDF::NcFile &, TCData<tcdata_t, Eigen::MatrixXd> &, std::string,
                               std::string &, pointing_offset_t &, Eigen::DenseBase<Derived> &,
                               calib_t &);
};

template <class calib_t>
void TCProc::load_mb(std::string filepath, std::string noise_filepath, calib_t &calib) {

    namespace fs = std::filesystem;

    // clear map buffer
    std::vector<Eigen::MatrixXd>().swap(tod_mb.signal);
    std::vector<Eigen::MatrixXd>().swap(tod_mb.weight);
    std::vector<Eigen::MatrixXd>().swap(tod_mb.kernel);
    std::vector<Eigen::Tensor<double,3>>().swap(tod_mb.noise);
    std::vector<std::string>().swap(tod_mb.wcs.cunit);

    tod_mb.median_rms.resize(0);

    // resize wcs params
    tod_mb.wcs.naxis.resize(4,0.);
    tod_mb.wcs.crpix.resize(4,0.);
    tod_mb.wcs.crval.resize(4,0.);
    tod_mb.wcs.cdelt.resize(4,0.);
    tod_mb.wcs.cunit.push_back("N/A");
    tod_mb.wcs.cunit.push_back("N/A");

    // vector to hold mean rms
    std::vector<double> median_rms_vec;

    // loop through arrays in current obs
    for (const auto &arr: calib.arrays) {
        try {
            // loop through files in redu directory
            for (const auto& entry : fs::directory_iterator(filepath)) {
                // check if fits file
                bool fits_file = entry.path().string().find(".fits") != std::string::npos;
                // find current array obs map
                if (entry.path().string().find(toltec_io.array_name_map[arr]) != std::string::npos && fits_file) {

                    // get filename
                    std::string filename;
                    size_t last_slash_pos = entry.path().string().find_last_of("/");
                    if (last_slash_pos != std::string::npos) {
                        filename = entry.path().string().substr(last_slash_pos + 1);
                    }

                    // get maps (if noise not in filename)
                    if (filename.find("noise") == std::string::npos) {
                        // get fits file
                        fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*> fits_io(entry.path().string());

                        // get number of extensions other than primary extension
                        int num_extensions = 0;
                        bool keep_going = true;
                        while (keep_going) {
                            try {
                                // attempt to access an HDU (ignore primary hdu)
                                CCfits::ExtHDU& ext = fits_io.pfits->extension(num_extensions + 1);
                                num_extensions++;
                            } catch (CCfits::FITS::NoSuchHDU) {
                                // NoSuchHDU exception is thrown when there are no more HDUs
                                keep_going = false;
                            }
                        }

                        if (num_extensions == 0) {
                            logger->error("{} is empty",filename);
                            std::exit(EXIT_FAILURE);
                        }

                        // get wcs (should be same for all maps)
                        CCfits::ExtHDU& extension = fits_io.pfits->extension(1);

                        // get naxis
                        extension.readKey("NAXIS1", tod_mb.wcs.naxis[0]);
                        extension.readKey("NAXIS2", tod_mb.wcs.naxis[1]);
                        // get crpix
                        extension.readKey("CRPIX1", tod_mb.wcs.crpix[0]);
                        extension.readKey("CRPIX2", tod_mb.wcs.crpix[1]);
                        // get crval
                        extension.readKey("CRVAL1", tod_mb.wcs.crval[0]);
                        extension.readKey("CRVAL2", tod_mb.wcs.crval[1]);
                        // get cdelt
                        extension.readKey("CDELT1", tod_mb.wcs.cdelt[0]);
                        extension.readKey("CDELT2", tod_mb.wcs.cdelt[1]);
                        // get cunit
                        extension.readKey("CUNIT1", tod_mb.wcs.cunit[0]);
                        extension.readKey("CUNIT2", tod_mb.wcs.cunit[1]);

                        // get I maps, including all fg maps
                        for (int i=0; i<num_extensions; ++i) {
                            CCfits::ExtHDU& ext = fits_io.pfits->extension(i+1);
                            std::string extName;
                            ext.readKey("EXTNAME", extName);
                            // get signal I map
                            if (extName.find("signal") != std::string::npos && extName.find("_I") != std::string::npos) {
                                tod_mb.signal.push_back(fits_io.get_hdu(extName));
                                logger->info("found {} [{}]", filename, extName);
                            }
                            // get weight I map
                            else if (extName.find("weight") != std::string::npos && extName.find("_I") != std::string::npos) {
                                tod_mb.weight.push_back(fits_io.get_hdu(extName));
                                logger->info("found {} [{}]", filename, extName);
                            }
                            // get kernel I map
                            else if (extName.find("kernel") != std::string::npos && extName.find("_I") != std::string::npos) {
                                tod_mb.kernel.push_back(fits_io.get_hdu(extName));
                                logger->info("found {} [{}]", filename, extName);
                            }
                        }
                    }
                }
            }

            // get noise maps
            // loop through files in redu directory
            for (const auto& entry : fs::directory_iterator(noise_filepath)) {
                // check if fits file
                bool fits_file = entry.path().string().find(".fits") != std::string::npos;
                // find current array obs map
                if (entry.path().string().find(toltec_io.array_name_map[arr]) != std::string::npos && fits_file) {

                    // get filename
                    std::string filename;
                    size_t lastSlashPos = entry.path().string().find_last_of("/");
                    if (lastSlashPos != std::string::npos) {
                        filename = entry.path().string().substr(lastSlashPos + 1);
                    }

                    // check if the current file is a noise map
                    if (filename.find("_noise_citlali.fits") != std::string::npos) {
                        // get fits file
                        fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*> fits_io(entry.path().string());

                        // get number of noise maps
                        int num_extensions = 0;
                        bool keep_going = true;
                        while (keep_going) {
                            try {
                                // attempt to access an HDU (ignore primary hdu)
                                CCfits::ExtHDU& ext = fits_io.pfits->extension(num_extensions + 1);
                                std::string extName;
                                ext.readKey("EXTNAME", extName);

                                // only get stokes I
                                if (extName.find("_I") != std::string::npos) {
                                    tod_mb.n_noise++;
                                }
                                // if extension found, add to total number
                                num_extensions++;
                            } catch (CCfits::FITS::NoSuchHDU) {
                                // NoSuchHDU exception is thrown when there are no more HDUs
                                keep_going = false;
                            }
                        }

                        if (num_extensions == 0) {
                            logger->error("{} is empty",filename);
                            std::exit(EXIT_FAILURE);
                        }

                        // loop through noise maps for current array
                        for (int i=0; i<num_extensions; ++i) {
                            // get current extension
                            CCfits::ExtHDU& ext = fits_io.pfits->extension(i+1);
                            // get extension name
                            std::string extName;
                            ext.readKey("EXTNAME", extName);
                            // if signal I's first noise map (ignore Q, U, and extra noise maps)
                            if (extName.find("signal") != std::string::npos && extName.find("_0_I") != std::string::npos) {                                    // get mean rms for current extension
                                // get median rms from current extension
                                double median_rms;
                                ext.readKey("MEDRMS", median_rms);
                                median_rms_vec.push_back(median_rms);
                                logger->info("found {} [{}]", filename, extName);
                            }
                        }
                    }
                }
            }

        } catch (const fs::filesystem_error& err) {
            logger->error("{}", err.what());
            std::exit(EXIT_FAILURE);
        }
    }

    // check if we found any maps
    if (tod_mb.signal.empty() || tod_mb.weight.empty()) {
        logger->error("no signal maps found in {}", filepath);
        std::exit(EXIT_FAILURE);
    }

    if (!median_rms_vec.empty()) {
        // map median rms from fits files vector to map buffer vector
        tod_mb.median_rms = Eigen::Map<Eigen::VectorXd>(median_rms_vec.data(),median_rms_vec.size());
    }

    // set dimensions
    tod_mb.n_cols = tod_mb.wcs.naxis[0];
    tod_mb.n_rows = tod_mb.wcs.naxis[1];

    // get pixel size in radians
    if (tod_mb.wcs.cunit[0] == "deg") {
        tod_mb.pixel_size_rad = abs(tod_mb.wcs.cdelt[0])*DEG_TO_RAD;
    }
    else if (tod_mb.wcs.cunit[0] == "arcsec") {
        tod_mb.pixel_size_rad = abs(tod_mb.wcs.cdelt[0])*ASEC_TO_RAD;
    }

    // calculate coverage bool map
    for (int i=0; i<tod_mb.weight.size(); ++i) {
        Eigen::MatrixXd ones, zeros;
        ones.setOnes(tod_mb.weight[i].rows(), tod_mb.weight[i].cols());
        zeros.setZero(tod_mb.weight[i].rows(), tod_mb.weight[i].cols());

        // get weight threshold for current map
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = tod_mb.calc_cov_region(i);
        // if weight is less than threshold, set to zero, otherwise set to one
        auto cov_bool = (tod_mb.weight[i].array() < weight_threshold).select(zeros,ones);
        tod_mb.signal[i] = tod_mb.signal[i].array() * cov_bool.array();
        if (!tod_mb.kernel.empty()) {
            tod_mb.kernel[i] = tod_mb.kernel[i].array() * cov_bool.array();
        }
    }
    // clear weight maps to save memory
    std::vector<Eigen::MatrixXd>().swap(tod_mb.weight);
}

template <typename Derived, class calib_t>
auto TCProc::get_grouping(std::string grp, Eigen::DenseBase<Derived> &det_indices, calib_t &calib, int n_dets) {
    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

    // initial group value is value for the first det index
    Eigen::Index grp_i = calib.apt[grp](det_indices(0));
    // set up first group
    grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};
    Eigen::Index j = 0;
    // loop through apt table arrays, get highest index for current array
    for (Eigen::Index i=0; i<n_dets; ++i) {
        auto det_index = det_indices(i);
        // if we're still on the current group
        if (calib.apt[grp](det_index) == grp_i) {
            std::get<1>(grp_limits[grp_i]) = i + 1;
        }
        // otherwise increment and start the next group
        else {
            grp_i = calib.apt[grp](det_index);
            j += 1;
            grp_limits[grp_i] = std::tuple<Eigen::Index, Eigen::Index>{i,0};
        }
    }
    return grp_limits;
}

// compute pointing for all detectors
template <TCDataKind tcdata_t, class calib_t, typename Derived>
void TCProc::precompute_pointing(TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                 std::string pixel_axes, std::string map_grouping) {

    // dimensions of data
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    in.pointing.data["lat"].resize(n_pts,n_dets);
    in.pointing.data["lon"].resize(n_pts,n_dets);

    for (Eigen::Index i=0; i<n_dets; ++i) {
        // current detector index in apt
        auto det_index = det_indices(i);
        double az_off = calib.apt["x_t"](det_index);
        double el_off = calib.apt["y_t"](det_index);

        // get detector pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off, pixel_axes,
                                                          in.pointing_offsets_arcsec.data, map_grouping);

        in.pointing.data["lat"].col(i) = std::move(lat);
        in.pointing.data["lon"].col(i) = std::move(lon);
    }
}

template <TCProc::SourceType source_type, class mb_t, TCDataKind tcdata_t, class calib_t, typename Derived>
void TCProc::map_to_tod(mb_t &mb, TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                        Eigen::DenseBase<Derived> &map_indices, std::string pixel_axes, std::string map_grouping) {

    // dimensions of data
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // add or subtract timestream
    int factor = 1;
    if constexpr (source_type==NegativeMap) {
        factor = -1;
    }

    // run kernel through fruit loops
    bool run_kernel = in.kernel.data.size() !=0;
    // if mean rms is filled use S/N limit
    bool run_noise = mb.median_rms.size() != 0;

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // current detector index in apt
        auto det_index = det_indices(i);
        auto map_index = map_indices(i);

        // check if detector is not flagged
        if (calib.apt["flag"](det_index) == 0 && (in.flags.data.col(i).array() == 0).any()) {
            double az_off = calib.apt["x_t"](det_index);
            double el_off = calib.apt["y_t"](det_index);

            // calc tangent plane pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off, pixel_axes,
                                                              in.pointing_offsets_arcsec.data, map_grouping);

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd irows = lat.array()/mb.pixel_size_rad + (mb.n_rows)/2.;
            Eigen::VectorXd icols = lon.array()/mb.pixel_size_rad + (mb.n_cols)/2.;

            // loop through data points
            for (Eigen::Index j=0; j<n_pts; ++j) {
                // row and col pixel from signal image
                Eigen::Index ir = irows(j);
                Eigen::Index ic = icols(j);

                // check if current sample is on the image and add to the timestream
                if ((ir >= 0) && (ir < mb.n_rows) && (ic >= 0) && (ic < mb.n_cols)) {
                    double signal = mb.signal[map_index](ir,ic);
                    // check whether we should include pixel
                    bool run_pix_s2n = run_noise && (signal / mb.median_rms(map_index) >= fruit_loops_sig2noise);
                    bool run_pix_flux = signal >= fruit_loops_flux;

                    // if signal flux is higher than S/N limit, flux limit
                    if (run_pix_s2n && run_pix_flux) {
                        // add/subtract signal pixel from signal timestream
                        in.scans.data(j,i) += factor * signal;
                        // add/subtract kernel pixel from kernel timestream
                        if (run_kernel) {
                            in.kernel.data(j,i) += factor * mb.kernel[map_index](ir,ic);
                        }
                    }
                }
            }
        }
    }
}

template <TCDataKind tcdata_t, class calib_t, typename Derived>
auto TCProc::remove_bad_dets(TCData<tcdata_t, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                             std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // only run if limits are not zero
    if (lower_weight_factor !=0 || upper_weight_factor !=0) {
        logger->info("removing outlier dets");
        // number of detectors
        Eigen::Index n_dets = in.scans.data.cols();

        // get grouping
        auto grp_limits = get_grouping("array",det_indices,calib,n_dets);

        in.n_dets_low = 0;
        in.n_dets_high = 0;

        // loop through group limits
        for (auto const& [key, val] : grp_limits) {
            // control for iteration
            bool keep_going = true;
            Eigen::Index n_iter = 0;

            while (keep_going) {
                // number of unflagged detectors
                Eigen::Index n_good_dets = 0;

                // get good dets in group
                for (Eigen::Index j=std::get<0>(grp_limits[key]); j<std::get<1>(grp_limits[key]); ++j) {
                    if (calib.apt["flag"](det_indices(j))==0 && (in.flags.data.col(j).array()==0).any()) {
                        n_good_dets++;
                    }
                }

                Eigen::VectorXd det_std_dev(n_good_dets);
                Eigen::VectorXI dets(n_good_dets);
                Eigen::Index k = 0;

                // collect standard deviation from good detectors
                for (Eigen::Index j=std::get<0>(grp_limits[key]); j<std::get<1>(grp_limits[key]); ++j) {
                    Eigen::Index det_index = det_indices(j);
                    if (calib.apt["flag"](det_index)==0 && (in.flags.data.col(j).array()==0).any()) {
                        // make Eigen::Maps for each detector's scan
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                            in.scans.data.col(j).data(), in.scans.data.rows());
                        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                            in.flags.data.col(j).data(), in.flags.data.rows());

                        // calc standard deviation
                        det_std_dev(k) = engine_utils::calc_std_dev(scans, flags);

                        // convert to 1/variance so it is a weight
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

                int n_dets_low = 0;
                int n_dets_high = 0;

                // loop through good detectors and flag those that have std devs beyond the limits
                for (Eigen::Index j=0; j<n_good_dets; ++j) {
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
                            in.n_dets_low++;
                            n_dets_low++;
                        }

                        // flag those above limit
                        if ((det_std_dev(j) > (upper_weight_factor*mean_std_dev)) && upper_weight_factor!=0) {
                            if (map_grouping!="detector") {
                                in.flags.data.col(dets(j)).setOnes();
                            }
                            else {
                                calib_scan.apt["flag"](det_index) = 1;
                            }
                            in.n_dets_high++;
                            n_dets_high++;
                        }
                    }
                }

                logger->info("array {} iter {}: {}/{} dets below limit. {}/{} dets above limit.", key, n_iter,
                            n_dets_low, n_good_dets, n_dets_high, n_good_dets);

                // increment iteration
                n_iter++;
                // check if no more detectors are above limit
                if ((n_dets_low==0 && n_dets_high==0) || n_iter > iter_lim) {
                    keep_going = false;
                }
            }
        }
        // set up scan calib
        calib_scan.setup();
    }

    return std::move(calib_scan);
}

template <TCProc::SourceType source_type, TCDataKind tcdata_t, typename Derived, typename apt_t>
void TCProc::add_gaussian(TCData<tcdata_t, Eigen::MatrixXd> &in, Eigen::DenseBase<Derived> &params, std::string &pixel_axes,
                          std::string &map_grouping, apt_t &apt, double pixel_size_rad, Eigen::Index n_rows, Eigen::Index n_cols) {

    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // loop through detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // detector index in apt
        auto det_index = in.det_indices.data(i);
        // map index
        auto map_index = in.map_indices.data(i);

        double az_off = apt["x_t"](det_index);
        double el_off = apt["y_t"](det_index);

        // get pointing
        auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off, pixel_axes,
                                                          in.pointing_offsets_arcsec.data, map_grouping);

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
        if constexpr (source_type==NegativeGaussian) {
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
        for (Eigen::Index j=0; j<n_pts; ++j) {
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
    for (Eigen::Index i=0; i<n_dets; ++i) {
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
        for (Eigen::Index j=0; j<n_pts; ++j) {
            // flag samples within radius as bad
            if (dist(j) < mask_radius_arcsec*ASEC_TO_RAD) {
                masked_flags(j,i) = 1;
            }
        }
    }

    return std::move(masked_flags);
}

template <TCDataKind tcdata_t, typename Derived, class calib_t, typename pointing_offset_t>
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
    for (Eigen::Index i=0; i<n_dets; ++i) {
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

        if (pixel_axes == "radec") {
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

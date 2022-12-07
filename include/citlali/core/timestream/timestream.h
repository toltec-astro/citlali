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

namespace timestream {

namespace wcs = kids::wcs;

// clang-format off
TULA_BITFLAG(TCDataKind, int,        0xFFFF,
             RTC                     = 1 << 0,
             PTC                     = 1 << 1,
             Any                     = RTC | PTC
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

template <typename Derived> struct TCDataBase;
template <TCDataKind kind_> struct TCDataBase<TCData<kind_>> {
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
};

template <typename RefType>
struct TCData<TCDataKind::RTC,RefType>
    : TimeStream<TCData<TCDataKind::RTC>> {
    using Base = TimeStream<TCData<TCDataKind::RTC>>;
    using data_t = std::conditional_t<tula::eigen_utils::is_plain_v<RefType>,Base::data_t<RefType>, Base::dataref_t<RefType>>;
    // data timestreams
    data_t scans;
    // kernel timestreams
    Base::data_t<Eigen::MatrixXd> kernel;
    // flag timestream
    Base::data_t<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> flags;
    // current scan indices
    Base::data_t<Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1>> scan_indices;
    // scan index
    Base::data_t<Eigen::Index> index;
    // telescope data for scan
    Base::data_t<std::map<std::string, Eigen::VectorXd>> tel_data;
    // hwp angle for scan
    Base::data_t<Eigen::VectorXd> hwp_angle;
    // time of rtc creation
    std::string creation_time = engine_utils::current_date_time();

    bool demodulated = false;
    bool kernel_generated = false;
    bool despiked = false;
    bool tod_filtered = false;
    bool downsampled = false;
    bool calibrated = false;
    bool cleaned = false;
};

template <typename RefType>
struct TCData<TCDataKind::PTC, RefType>
    : TimeStream<TCData<TCDataKind::PTC>> {
    using Base = TimeStream<TCData<TCDataKind::PTC>>;
    using data_t = std::conditional_t<tula::eigen_utils::is_plain_v<RefType>,Base::data_t<RefType>, Base::dataref_t<RefType>>;
    // data timestreams
    data_t scans;
    // weights for current scan
    Base::data_t<Eigen::MatrixXd> weights;
    // kernel timestreams
    Base::data_t<Eigen::MatrixXd> kernel;
    // flag timestream
    Base::data_t<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> flags;
    // scan index
    Base::data_t<Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1>> scan_indices;
    // scan index
    Base::data_t<Eigen::Index> index;
    // telescope data for scan
    Base::data_t<std::map<std::string, Eigen::VectorXd>> tel_data;
    // hwp angle for scan
    Base::data_t<Eigen::VectorXd> hwp_angle;
    // eigenvalues for scan
    Base::data_t<Eigen::VectorXd> evals;
    // vectors for mapping apt table onto timestreams
    Base::data_t<Eigen::VectorXI> det_indices, nw_indices, array_indices, map_indices;
    // number of detectors lower than stddev limit
    int n_low_dets;
    // number of detectors lower than stddev limit
    int n_high_dets;
    // time of ptc creation
    std::string creation_time = engine_utils::current_date_time();

    bool demodulated = false;
    bool kernel_generated = false;
    bool despiked = false;
    bool tod_filtered = false;
    bool downsampled = false;
    bool calibrated = false;
    bool cleaned = false;
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

} // namespace timestream

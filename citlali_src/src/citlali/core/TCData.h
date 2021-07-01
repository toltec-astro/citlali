#pragma once

#include "../common_utils/src/utils/config.h"
#include "../common_utils/src/utils/enum.h"
#include "../common_utils/src/utils/formatter/enum.h"
#include "../common_utils/src/utils/formatter/matrix.h"
#include "../common_utils/src/utils/formatter/utils.h"
#include "../common_utils/src/utils/logging.h"
#include "../common_utils/src/utils/nddata/nddata.h"

namespace timestream {

// clang-format off
BITMASK_(LaliDataKind, int,        0xFFFF,
         RTC                     = 1 << 0,
         PTC                     = 1 << 1,
         Any                     = RTC | PTC
         );
// clang-format on

/// @brief TC data class.
template <LaliDataKind kind_ = LaliDataKind::Any, typename = void>
struct TCData;
} // namespace timestream

namespace std {

// Register TCData as a variant type
// Below are mandatory to inherit from variant on gcc
template <timestream::LaliDataKind kind>
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

/*template <typename Derived> struct impl_traits {
    using meta_t = config::Config;
    define_has_member_traits(Derived, fs);
    define_has_member_traits(Derived, iqs);
    define_has_member_traits(Derived, eiqs);
    define_has_member_traits(Derived, cal);
    define_has_member_traits(Derived, is);
    define_has_member_traits(Derived, qs);
    define_has_member_traits(Derived, xs);
    define_has_member_traits(Derived, rs);
};*/

template <typename Derived> struct TCDataBase;
template <LaliDataKind kind_> struct TCDataBase<TCData<kind_>> {
    static constexpr auto kind() { return kind_; }
    //using meta_t = typename internal::impl_traits<TCData<kind_>>::meta_t;
    //meta_t meta;
};

} // namespace internal

// WCS objects
struct DetectorAxis : wcs::Axis<DetectorAxis, wcs::CoordsKind::Column>,
                  nddata::LabeledData<DetectorAxis> {
    DetectorAxis() = default;
    DetectorAxis(Eigen::MatrixXd data_, nddata::LabelMapper<DetectorAxis> row_labels_)
        : data{std::move(data_)}, row_labels{std::move(row_labels_)} {}
    std::string_view name{"detector"};
    Eigen::MatrixXd data;
    nddata::LabelMapper<DetectorAxis> row_labels;
};


struct TimeAxis : wcs::Axis<TimeAxis, wcs::CoordsKind::Row>,
                  nddata::LabeledData<TimeAxis> {
    TimeAxis() = default;
    TimeAxis(Eigen::MatrixXd data_, nddata::LabelMapper<TimeAxis> col_labels_)
        : data{std::move(data_)}, col_labels{std::move(col_labels_)} {}
    std::string_view name{"time"};
    Eigen::MatrixXd data;
    nddata::LabelMapper<TimeAxis> col_labels;
};

struct TimeStreamFrame : wcs::Frame2D<TimeStreamFrame, TimeAxis, DetectorAxis> {
    TimeAxis time_axis;
    DetectorAxis detector_axis;

    // Frame2D impl
    const TimeAxis &row_axis() const { return time_axis; }
    const DetectorAxis &col_axis() const { return detector_axis; }
};

// Data objects

/// @brief Base class for time stream data
template <typename Derived>
struct TimeStream : internal::TCDataBase<Derived>,
                    nddata::NDData<TimeStream<Derived>> {
    TimeStreamFrame wcs;
    // The timestream is stored in row major for efficient r/w
    template <typename PlainObject>
    struct dataref_t : nddata::NDData<dataref_t<PlainObject>> {
        PlainObject data{nullptr, 0, 0};
    };

    template <typename PlainObject>
    struct data_t : nddata::NDData<data_t<PlainObject>> {
        PlainObject data;
    };
};

template <typename RefType>
struct TCData<LaliDataKind::RTC,RefType>
    : TimeStream<TCData<LaliDataKind::RTC>> {
    using Base = TimeStream<TCData<LaliDataKind::RTC>>;
    //TCData(RefType r): {}
    //TCData(RefType r) {
      //  scans.data = r;
    //}
    //using data_t = std::conditional_t<meta::is_template<RefType, Eigen::Map>, Base::dataref_t, Base::data_t>;
    using data_t = std::conditional_t<eigen_utils::is_plain_v<RefType>,Base::data_t<RefType>, Base::dataref_t<RefType>>;
    data_t scans;
    // NDData impl
    Base::data_t<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> flags;
    Base::data_t<Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1>> scanindex;
    Base::data_t<Eigen::Index> index;
    Base::data_t<Eigen::VectorXd> telLat;
    Base::data_t<Eigen::VectorXd> telLon;
    Base::data_t<Eigen::VectorXd> telElDes;
    Base::data_t<Eigen::VectorXd> ParAng;

};

template <typename RefType>
struct TCData<LaliDataKind::PTC, RefType>
    : TimeStream<TCData<LaliDataKind::PTC>> {
    using Base = TimeStream<TCData<LaliDataKind::PTC>>;
    using data_t = std::conditional_t<eigen_utils::is_plain_v<RefType>,Base::data_t<RefType>, Base::dataref_t<RefType>>;
    // NDData impl
    data_t scans;
    Base::data_t<Eigen::MatrixXd> kernelscans;
    Base::data_t<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> flags;
    Base::data_t<Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1>> scanindex;
    Base::data_t<Eigen::Index> index;
    Base::data_t<Eigen::VectorXd> weights;
    Base::data_t<Eigen::VectorXd> telLat;
    Base::data_t<Eigen::VectorXd> telLon;
    Base::data_t<Eigen::VectorXd> telElDes;
    Base::data_t<Eigen::VectorXd> ParAng;
};


/// @brief data class of runtime variant kind.
template <LaliDataKind kind_>
struct TCData<kind_, std::enable_if_t<enum_utils::is_compound_v<kind_>>>
    : enum_utils::enum_to_variant_t<kind_, TCData> {
    using Base = enum_utils::enum_to_variant_t<kind_, TCData>;
    using variant_t = enum_utils::enum_to_variant_t<kind_, TCData>;
    //using meta_t = typename internal::impl_traits<TCData<kind_>>::meta_t;

    // construct from primitive type
    // template <LaliDataKind kind1,
    // REQUIRES_V(!enum_utils::is_compound_v<kind1>)> TCData(TCData<kind1>
    // other) : Base(std::move(other)) {}

    const variant_t &variant() const { return *this; }
    static constexpr auto kind() { return kind_; }
};

} // namespace kids

namespace fmt {

template <>
struct formatter<timestream::DetectorAxis> : formatter<wcs::FrameBase<timestream::DetectorAxis>> {
    using Base = formatter<wcs::FrameBase<timestream::DetectorAxis>>;
    template <typename FormatContext>
    auto format(const timestream::DetectorAxis &data, FormatContext &ctx) {
        auto it = Base::format(data, ctx);
        return it = format_to(it, " labels={}", data.row_labels());
    }
};

template <>
struct formatter<timestream::TimeAxis> : formatter<wcs::FrameBase<timestream::TimeAxis>> {
    using Base = formatter<wcs::FrameBase<timestream::TimeAxis>>;
    template <typename FormatContext>
    auto format(const timestream::TimeAxis &data, FormatContext &ctx) {
        return Base::format(data, ctx);
    }
};

template <timestream::LaliDataKind kind_>
struct formatter<timestream::TCData<kind_>>
    : fmt_utils::charspec_formatter_base<'l', 's'> {
    // s: the short form
    // l: the long form
    using Data = timestream::TCData<kind_>;

    template <typename FormatContext>
    auto format(const Data &data, FormatContext &ctx) {
        //using data_traits = timestream::internal::impl_traits<Data>;
        auto it = ctx.out();
        constexpr auto kind = Data::kind();
        if constexpr (enum_utils::is_compound_v<kind>) {
            // format compound kind as variant
            return format_to(it, "({}) {:0}", kind, data.variant());
        }/* else {
            /// format simple kind type
            auto spec = spec_handler();
            // meta
            switch (spec) {
            case 's': {
                it = format_to(it, "({})", kind);
                if constexpr (data_traits::has_iqs::value) {
                    return format_member(it, "iqs", data.iqs());
                } else if constexpr (data_traits::has_xs::value) {
                    return format_member(it, "xs", data.xs());
                }
            }
            case 'l': {
                it = format_to(it, "kind={} meta={}", kind, data.meta.pformat());
                bool sep = false;
                if constexpr (data_traits::has_detectors::value) {
                    it = format_member(it, "detectors", data.detectors(), &sep);
                }
                if constexpr (data_traits::has_sweeps::value) {
                    it = format_member(it, "sweeps", data.sweeps(), &sep);
                }
                if constexpr (data_traits::has_fs::value) {
                    it = format_member(it, "fs", data.fs(), &sep);
                }
                if constexpr (data_traits::has_iqs::value) {
                    it = format_member(it, "iqs", data.iqs(), &sep);
                }
                if constexpr (data_traits::has_eiqs::value) {
                    it = format_member(it, "eiqs", data.eiqs(), &sep);
                }
                if constexpr (data_traits::has_is::value) {
                    it = format_member(it, "is", data.is(), &sep);
                }
                if constexpr (data_traits::has_qs::value) {
                    it = format_member(it, "qs", data.qs(), &sep);
                }
                if constexpr (data_traits::has_xs::value) {
                    it = format_member(it, "xs", data.xs(), &sep);
                }
                if constexpr (data_traits::has_rs::value) {
                    it = format_member(it, "rs", data.rs(), &sep);
                }
                return it;
            }
            }
            return it;
        }*/
    }
    template <typename T, typename FormatContextOut>
    auto format_member(FormatContextOut &it, std::string_view name, const T &m,
                       bool *sep = nullptr) {
        auto spec = spec_handler();
        switch (spec) {
        case 's': {
            if ((m.rows() == 1) || (m.cols() == 1)) {
                return format_to(it, "[{}]", m.size());
            }
            return format_to(it, "[{}, {}]", m.rows(), m.cols());
        }
        case 'l': {
            it = format_to(it, "{}{}={:r0}", sep ? " " : "", name, m);
            // as soon as this it called, sep is set
            *sep = true;
            return it;
        }
        }
        return it;
    }
};
} // namespace fmt

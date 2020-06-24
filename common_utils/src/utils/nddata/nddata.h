#pragma once

#include "../container.h"
#include "../eigen.h"
#include "../enum.h"
#include "../meta.h"
#include <Eigen/Core>

namespace nddata {

namespace internal {

// Define some of the implementation related types
template <typename T, typename = void> struct impl_traits {
    using index_t = Eigen::Index;
    using physical_type_t = std::optional<std::string>;
    using unit_t = std::optional<std::string>;
    using label_t = std::string;
};

} // namespace internal

template <typename Derived> struct NDData {
    define_has_member_traits(Derived, data);
    define_has_member_traits(Derived, uncertainty);
    define_has_member_traits(Derived, mask);
    define_has_member_traits(Derived, flags);
    define_has_member_traits(Derived, wcs);
    define_has_member_traits(Derived, meta);
    define_has_member_traits(Derived, unit);
    define_has_member_traits(Derived, physical_type);
    using physical_type_t =
        typename internal::impl_traits<Derived>::physical_type_t;
    using unit_t = typename internal::impl_traits<Derived>::unit_t;

    const auto &derived() const { return static_cast<const Derived &>(*this); }
    auto &derived() { return static_cast<Derived &>(*this); }

    template <typename U = Derived, REQUIRES_V_(U::has_data::value)>
    const auto &operator()() const {
        return derived().data;
    }
    template <typename U = Derived, REQUIRES_V_(U::has_data::value)>
    auto &operator()() {
        return derived().data;
    }
};

/// @brief Data object that evaluate only on first call.
template <typename DataType, typename Evaluator>
struct CachedData : NDData<CachedData<DataType, Evaluator>> {
    mutable DataType data;
    mutable bool initialized{false};
    mutable std::mutex m_mutex;
    CachedData() = default;
    CachedData(const CachedData &other)
        : data(other.data), initialized(other.initialized) {}
    CachedData(CachedData &&other)
        : data(std::move(other.data)), initialized(other.initialized) {}
    CachedData &operator=(const CachedData &other) {
        data = other.data;
        initialized = other.initialized;
        return *this;
    }
    CachedData &operator=(CachedData &&other) {
        data = std::move(other.data);
        initialized = other.initialized;
        return *this;
    }

    template <typename T> const auto &operator()(const T &parent) const {
        static_assert(
            std::is_constructible_v<
                DataType,
                std::invoke_result_t<decltype(Evaluator::evaluate), T>>,
            "EVALUATOR RETURN TYPE MISMATCH ATTRIBUTE DATA TYPE");
        std::scoped_lock lock(m_mutex);
        if (!initialized) {
            data = Evaluator::evaluate(parent);
            initialized = true;
        }
        return data;
    }
};

template <typename Derived> struct LabelMapper {
    using index_t = typename internal::impl_traits<Derived>::index_t;
    using label_t = typename internal::impl_traits<Derived>::label_t;
    using labels_t = std::vector<label_t>;
    LabelMapper() = default;
    LabelMapper(labels_t labels_) : m_labels{std::move(labels_)} {}
    index_t size() const { return meta::size_cast<index_t>(m_labels.size()); }
    index_t index(const label_t &label) const {
        if (auto opt_i = container_utils::indexof(m_labels, label);
            opt_i.has_value()) {
            return opt_i.value();
        } else {
            throw std::runtime_error(
                fmt::format("label {} not found in {}", label, m_labels));
        }
    }
    const label_t &label(index_t i) const { return labels().at(SIZET(i)); }
    const labels_t &labels() const { return m_labels; }
    const auto &operator()() const { return labels(); }

private:
    labels_t m_labels;
};

template <typename Derived> struct LabeledData : NDData<Derived> {
    define_has_member_traits(Derived, row_labels);
    define_has_member_traits(Derived, col_labels);

    using Base = NDData<Derived>;
    using Base::derived;
    using index_t = typename internal::impl_traits<Derived>::index_t;
    using label_t = typename internal::impl_traits<Derived>::label_t;

    template <typename U = Derived, REQUIRES_V_(U::has_row_labels::value)>
    auto operator()(const label_t &name) const {
        return derived().data.row(derived().row_labels.index(name));
    }
    template <typename U = Derived, REQUIRES_V_(U::has_row_labels::value)>
    auto operator()(const label_t &name) {
        return derived().data.row(derived().row_labels.index(name));
    }

    template <typename U = Derived, REQUIRES_V_(U::has_col_labels::value)>
    auto operator()(const label_t &name) const {
        return derived().data.col(derived().col_labels.index(name));
    }
    template <typename U = Derived, REQUIRES_V_(U::has_col_labels::value)>
    auto operator()(const label_t &name) {
        return derived().data.col(derived().col_labels.index(name));
    }
};

/*
/// @brief The base class of a data object
template <typename Derived, typename = void> struct DataObject;

/// @brief Specialization for plain object
template <template <typename, typename, typename...> typename U,
          typename DataType, typename Tag, typename SFAINE, typename... Rest>
struct DataObject<U<DataType, Tag, Rest...>, SFAINE> {
    using Derived = U<DataType, Tag, Rest...>;
    using data_t = DataType;
    using physical_type_t = typename internal::traits<Derived>::physical_type_t;
    using unit_t = typename internal::traits<Derived>::unit_t;
    const auto &operator()() const {
        return static_cast<Derived &>(*this).data;
    }
    auto &operator()() { return static_cast<Derived &>(*this).data; }

    // unit_t unit{};
    // physical_type_t physical_type{};
};

/// @brief Convenient macro to define data objects and the getter.
#define DEFINE_NDDATA_OBJECT(class_name, member_name)                          \
    template <typename DataType, typename Tag = void>                          \
    struct class_name : DataObject<class_name<DataType, Tag>> {                \
        const auto &operator()() const { return this->data; }                  \
        auto &operator()() { return this->data; }                              \
    }
DEFINE_NDDATA_OBJECT(Data, data);
DEFINE_NDDATA_OBJECT(Uncertainty, uncertainty);
DEFINE_NDDATA_OBJECT(Mask, mask);
DEFINE_NDDATA_OBJECT(Flags, flags);
DEFINE_NDDATA_OBJECT(WCS, wcs);
DEFINE_NDDATA_OBJECT(Meta, meta);

/// @brief Data object that evaluate only on first call.
template <typename DataType, typename Evaluator>
struct CachedDataObject : DataObject<CachedDataObject<DataType, Evaluator>> {
    mutable DataType data;
    mutable bool initialized{false};
    template <typename Derived>
    const auto &operator()(const Derived &derived) const {
        static_assert(
            std::is_constructible_v<
                DataType,
                std::invoke_result_t<decltype(Evaluator::evaluate), Derived>>,
            "EVALUATOR RETURN TYPE MISMATCH ATTRIBUTE DATA TYPE");
        if (!initialized) {
            data = Evaluator::evalutate(derived);
            initialized = true;
        }
        return data;
    }
};

// This unfortunately requires c++20
// template <typename DataType, auto evaluator>
// struct CachedAttr : NDDataAttr<CachedAttr<DataType, evaluator>> {
//     mutable DataType data;
//     mutable bool initialized{false};
//     template <typename T> const auto &get(const T &derived) const {
//         if (!initialized) {
//             data = evaluator(derived);
//             initialized = true;
//         }
//         return this->data;
//     }
// };
//
template <typename Derived> struct NDDataBase;
template <typename... Attrs> struct NDData : NDDataBase<NDData<Attrs...>> {};
template <typename... Attrs> struct NDDataBase<NDData<Attrs...>> : Attrs... {};

*/
} // namespace nddata

namespace wcs {

namespace internal {

struct frame_impl_traits {
    using index_t = nddata::internal::impl_traits<frame_impl_traits>::index_t;
    using physical_type_t =
        nddata::internal::impl_traits<frame_impl_traits>::physical_type_t;
    using unit_t = nddata::internal::impl_traits<frame_impl_traits>::unit_t;
    template <typename data_t>
    static constexpr auto is_valid_data_type =
        std::is_base_of_v<nddata::NDData<data_t>, data_t>;
};

} // namespace internal

/// @brief Frame base interface.
template <typename Derived> struct FrameBase {
    define_has_member_traits(Derived, name);
    using index_t = internal::frame_impl_traits::index_t;
    constexpr index_t array_n_dim() const { return Derived::frame_dimension; }
    const auto &derived() const { return static_cast<const Derived &>(*this); }
    auto &derived() { return static_cast<Derived &>(*this); }
};

enum class CoordsKind { Scalar, Row, Column };

/// @brief Axis base interface that represent one single dimension.
template <typename Derived, CoordsKind coords_kind>
struct Axis : FrameBase<Derived> {
    using Base = FrameBase<Derived>;
    using index_t = typename Base::index_t;
    using Base::derived;

    // FrameBase impl
    static constexpr index_t frame_dimension = 1;

    // WCS impl
    const auto &array_index_to_world_values(index_t i) const {
        if constexpr (coords_kind == CoordsKind::Scalar) {
            return derived().data.coeff(i);
        } else if constexpr (coords_kind == CoordsKind::Row) {
            return derived().data.row(i);
        } else if constexpr (coords_kind == CoordsKind::Column) {
            return derived().data.col(i);
        }
    }
    void resize(index_t n) {
        if constexpr (coords_kind == CoordsKind::Scalar) {
            return derived().data.resize(n);
        } else if constexpr (coords_kind == CoordsKind::Row) {
            return derived().data.resize(n, world_n_dim());
        } else if constexpr (coords_kind == CoordsKind::Column) {
            return derived().data.resize(world_n_dim(), n);
        }
    }
    index_t array_shape() const {
        if constexpr (coords_kind == CoordsKind::Scalar) {
            return derived().data.size();
        } else if constexpr (coords_kind == CoordsKind::Row) {
            return derived().data.rows();
        } else if constexpr (coords_kind == CoordsKind::Column) {
            return derived().data.cols();
        }
    }
    index_t world_n_dim() const {
        const index_t one = 1;
        if constexpr (coords_kind == CoordsKind::Scalar) {
            return one;
        } else if constexpr (coords_kind == CoordsKind::Row) {
            return derived().data.cols();
        } else if constexpr (coords_kind == CoordsKind::Column) {
            return derived().data.rows();
        }
    }
};

/// @brief Two dimensional frame that consists of two axes.
template <typename Derived, typename RowAxis, typename ColAxis>
struct Frame2D : FrameBase<Derived> {
    using Base = FrameBase<Derived>;
    using index_t = typename Base::index_t;
    using Base::derived;

    // FrameBase impl
    static constexpr index_t dimension = 2;

    auto array_index_to_world_values(index_t i, index_t j) const {
        return std::make_tuple(
            derived().row_axis().array_index_to_world_values(i),
            derived().col_axis().array_index_to_world_values(j));
    }
    std::pair<index_t, index_t> array_shape() {
        return {derived().row_axis().array_shape(),
                derived().col_axis().array_shape()};
    }
    index_t world_n_dim() {
        return derived().row_axis().world_n_dim() +
            derived().col_axis().world_n_dim();
    }
};

/*
template <typename Derived> struct BaseLowLevelWCS {
    using Self = BaseLowLevelWCS<Derived>;
    using size_t = typename internal::impl_traits<Self>::size_t;
    using index_t = typename internal::impl_traits<Self>::index_t;
    using physical_type_t =
        typename internal::impl_traits<Self>::physical_type_t;
    using unit_t = typename internal::impl_traits<Self>::unit_t;

    size_t pixel_n_dim() const {
        return static_cast<const Derived &>(*this).pixel_n_dim();
    }
    size_t world_n_dim() const {
        return static_cast<const Derived &>(*this).world_n_dim();
    }
    std::vector<physical_type_t> world_axis_physical_types() const {
        return static_cast<const Derived &>(*this)._world_axis_physical_types;
    }
    std::vector<unit_t> world_axis_units() const {
        return static_cast<const Derived &>(*this)._world_axis_units;
    }

    template <typename... Args> auto pixel_to_world_values(Args... args) {
        return static_cast<const Derived &>(*this)._pixel_to_world_values(
            args...);
    }
    template <typename... Args> auto array_index_to_world_values(Args... args) {
        return static_cast<const Derived &>(*this)._array_index_to_world_values(
            args...);
    }
    template <typename... Args> auto world_to_pixel_values(Args... args) {
        return static_cast<const Derived &>(*this)._world_to_pixel_values(
            args...);
    }
    template <typename... Args> auto world_to_array_index_values(Args... args) {
        return static_cast<const Derived &>(*this)._world_to_array_index_values(
            args...);
    }

    void world_axis_object_components() {}
    void world_axis_object_classes() {}

    std::optional<std::pair<Eigen::Index, Eigen::Index>> _array_shape() {
        return std::nullopt{};
    }
    std::optional<std::pair<Eigen::Index, Eigen::Index>> _pixel_shape() {
        return std::nullopt{};
    }
    std::optional<std::vector<std::pair<>>> _pixel_bounds() {
        return std::nullopt{};
    }

*/

} // namespace wcs

namespace fmt {

template <typename Derived, typename Char>
struct formatter<wcs::FrameBase<Derived>, Char>
    : fmt_utils::charspec_formatter_base<'l', 's'> {
    // s: the short form
    // l: the long form

    template <typename FormatContext>
    auto format(const wcs::FrameBase<Derived> &data_, FormatContext &ctx) {
        const auto &data = data_.derived();
        auto it = ctx.out();
        /// format simple kind type
        auto spec = spec_handler();
        std::string name = "unnamed";
        if constexpr (Derived::has_name::value) {
            name = data.name;
        }
        // meta
        switch (spec) {
        case 's': {
            it = format_to(it, "{}({}d{})[{}]", name, data.array_n_dim(),
                           data.world_n_dim(), data.array_shape());
        }
        case 'l': {
            it = format_to(it, "wcs{}d name={} shape=[{}] coords_dim={}",
                           data.array_n_dim(), name, data.array_shape(),
                           data.world_n_dim());
        }
        }
        return it;
    }
};

} // namespace fmt

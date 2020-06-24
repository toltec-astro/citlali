#pragma once
#include <Eigen/Core>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <type_traits>

namespace Eigen {

using VectorXI = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
using MatrixXI = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
using MatrixXb = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace Eigen

namespace eigen_utils {

namespace internal {
// Eigen CRTP does not work for vector block
template <typename T> struct is_vector_block : std::false_type {};
template <typename VectorType, int Size>
struct is_vector_block<Eigen::VectorBlock<VectorType, Size>> : std::true_type {
};

} // namespace internal

template <typename T>
inline constexpr bool is_vblock_v =
    internal::is_vector_block<std::decay_t<T>>::value;

template <typename T>
inline constexpr bool is_eigen_v =
    std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>> ||
    is_vblock_v<std::decay_t<T>>;

template <typename T>
inline constexpr bool is_dense_v =
    std::is_base_of_v<Eigen::DenseBase<std::decay_t<T>>, std::decay_t<T>>;

/// True if type manages its own data, e.g. MatrixXd, etc.
template <typename T>
inline constexpr bool is_plain_v =
    std::is_base_of_v<Eigen::PlainObjectBase<std::decay_t<T>>, std::decay_t<T>>;

template <typename T, typename = void> struct type_traits : std::false_type {};

template <typename T>
struct type_traits<T, std::enable_if_t<is_eigen_v<T>>> : std::true_type {
    using Derived = typename std::decay_t<T>;
    constexpr static Eigen::StorageOptions order =
        Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor;
    // related types
    using Vector = std::conditional_t<
        order == Eigen::RowMajor,
        Eigen::Matrix<typename Derived::Scalar, 1, Derived::ColsAtCompileTime,
                      Eigen::RowMajor>,
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1,
                      Eigen::ColMajor>>;
    using Matrix =
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime,
                      Derived::ColsAtCompileTime, order>;
    using VecMap = Eigen::Map<Vector, Eigen::AlignmentType::Unaligned>;
    using MatMap = Eigen::Map<Matrix, Eigen::AlignmentType::Unaligned>;
};

template <typename Derived>
bool is_contiguous(const Eigen::DenseBase<Derived> &m) {
    //     SPDLOG_TRACE(
    //         "size={} outerstride={} innerstride={} outersize={}
    //         innersize={}", m.size(), m.outerStride(), m.innerStride(),
    //         m.outerSize(), m.innerSize());
    if (m.innerStride() != 1) {
        return false;
    }
    return (m.size() <= m.innerSize()) || (m.outerStride() == m.innerSize());
}

// https://stackoverflow.com/a/21918950/1824372
template <typename T> struct PreAllocator {
    using size_type = std::size_t;
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    // using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    PreAllocator(pointer memory_ptr, size_type memory_size)
        : memory_ptr(memory_ptr), memory_size(memory_size) {}

    PreAllocator(const_pointer memory_ptr, size_type memory_size)
        : PreAllocator(const_cast<pointer>(memory_ptr), memory_size) {}

    PreAllocator(const PreAllocator &other) noexcept
        : memory_ptr(other.memory_ptr), memory_size(other.memory_size) {}

    template <typename U>
    PreAllocator(const PreAllocator<U> &other) noexcept
        : memory_ptr(other.memory_ptr), memory_size(other.memory_size) {}

    template <typename U> PreAllocator &operator=(const PreAllocator<U> &) {
        return *this;
    }
    PreAllocator<T> &operator=(const PreAllocator &) { return *this; }
    ~PreAllocator() = default;

    pointer allocate(size_type, const void * = nullptr) { return memory_ptr; }
    void deallocate(pointer, size_type) {}
    size_type max_size() const { return memory_size; }

    template <typename U, typename ... Args>
    void construct(U*, Args&&...) noexcept {}

private:
    pointer memory_ptr;
    std::size_t memory_size;
};

/**
 * @brief Create std::vector that views the data held by Eigen types.
 * @param m_ Input data of Eigen type.
 * Default is the same as input.
 */
template <typename Derived> auto asstd(const Eigen::DenseBase<Derived> &m_) {
    using Eigen::Dynamic;
    using Scalar = typename Derived::Scalar;
    using Allocator = PreAllocator<Scalar>;
    // Derived &m = const_cast<Eigen::DenseBase<Derived> &>(m_).derived();
    auto &m = m_.derived();
    return std::vector<Scalar, Allocator>(m.size(), Allocator{
        m.data(), static_cast<typename Allocator::size_type>((m.size()))});
}

/**
 * @brief Create std::vector from data held by Eigen types. Copies the data.
 * @param m The Eigen type.
 * @tparam order The storage order to follow when copying the data.
 * Default is the same as input.
 */
template <typename Derived,
          Eigen::StorageOptions order = type_traits<Derived>::order>
auto tostd(const Eigen::DenseBase<Derived> &m) {
    using Eigen::Dynamic;
    using Scalar = typename Derived::Scalar;
    std::vector<Scalar> vec(m.size());
    Eigen::Map<Eigen::Matrix<Scalar, Dynamic, Dynamic, order>>(
        vec.data(), m.rows(), m.cols()) = m;
    return vec;
}

/**
 * @brief Create std::vector from data held by Eigen types. Copies the data.
 * @param m The Eigen type.
 * @param order The storage order to follow when copying the data.
 */
template <typename Derived>
auto tostd(const Eigen::DenseBase<Derived> &m, Eigen::StorageOptions order) {
    using Eigen::ColMajor;
    using Eigen::RowMajor;
    if (order & RowMajor) {
        return tostd<Derived, RowMajor>(m.derived());
    }
    return tostd<Derived, ColMajor>(m.derived());
}

/**
 * @brief Create Eigen::Map from std::vector.
 * @tparam order The storage order to follow when mapping the data.
 * Default is Eigen::ColMajor.
 */
template <Eigen::StorageOptions order = Eigen::ColMajor, typename Scalar,
          typename... Rest>
auto asvec(std::vector<Scalar, Rest...> &v) {
    return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1, order>>(
        v.data(), v.size());
}
/**
 * @brief Create Eigen::Map from std::vector.
 * @tparam order The storage order to follow when mapping the data.
 * Default is Eigen::ColMajor.
 */
template <Eigen::StorageOptions order = Eigen::ColMajor, typename Scalar,
          typename... Rest>
auto asvec(const std::vector<Scalar, Rest...> &v) {
    return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1, order>>(
        v.data(), v.size());
}

/**
 * @brief Create Eigen::Matrix from std::vector of std::pair.
 */
template <typename Scalar, typename... Rest>
auto tomat(const std::vector<std::pair<Scalar, Scalar>, Rest...> &v) {
    Eigen::Matrix<Scalar, 2, Eigen::Dynamic> m(2, v.size());
    for (Eigen::Index i = 0; i < m.cols(); ++i) {
        m.coeffRef(0, i) = v[static_cast<std::size_t>(i)].first;
        m.coeffRef(1, i) = v[static_cast<std::size_t>(i)].second;
    }
    return m;
}

/**
 * @brief Create Eigen::Map from Eigen Matrix.
 * @tparam order The storage order to follow when mapping the data.
 * Default is Eigen::ColMajor.
 */
template <Eigen::StorageOptions order = Eigen::ColMajor, typename Derived>
auto asvec(const Eigen::DenseBase<Derived> &v_) {
    static_assert(is_plain_v<Derived>,
                  "ASVEC ONLY IMPLEMENTED FOR PLAIN OBJECT");
    auto &v = const_cast<Eigen::DenseBase<Derived> &>(v_).derived();
    if (is_contiguous(v)) {
        using Scalar = typename Derived::Scalar;
        return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1, order>>(
            v.data(), v.size());
    }
    throw std::runtime_error(
        "not able to create vector view for data in non-contiguous memory");
}

/**
 * @brief Create Eigen::Map from std::vector with shape
 * @tparam order The storage order to follow when mapping the data.
 * Default is Eigen::ColMajor.
 */
template <Eigen::StorageOptions order = Eigen::ColMajor, typename Scalar,
          typename... Rest>
auto asmat(std::vector<Scalar, Rest...> &v, Eigen::Index nrows,
           Eigen::Index ncols) {
    using Eigen::Dynamic;
    assert(nrows * ncols == v.size());
    return Eigen::Map<Eigen::Matrix<Scalar, Dynamic, Dynamic, order>>(
        v.data(), nrows, ncols);
}

} // namespace eigen_utils
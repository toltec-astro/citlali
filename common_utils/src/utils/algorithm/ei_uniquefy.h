#pragma once
#include <Eigen/Core>
#include <vector>

namespace alg {
/**
 * @brief Locates and removes duplicated values in *sorted* vector.
 * @return A vector of n_unique bin edge indexes (hence n_unique + 1 items in
 * total). THe i-th bin edge [i, i + 1) stores the indexes [si, ei) which
 * identifies the i-th unique element in the original vector.
 * @note The input vector is not resized, therefore elements starting at index
 * n_unique are in undetermined state.
 */
template <typename Derived> auto uniquefy(const Eigen::DenseBase<Derived> &m) {
    using Eigen::Index;
    auto size = m.size();
    std::vector<Index> uindex;
    uindex.push_back(0);
    auto old = m(0);
    for (Index i = 0; i < size; ++i) {
        if (m(i) != old) {
            // encountered a new unique value
            old = m(i); // update old
            const_cast<Eigen::DenseBase<Derived> &>(m)(uindex.size()) =
                old;             // store the unqiue value to upfront
            uindex.push_back(i); // store the unique index
        }
    }
    uindex.push_back(
        size); // this extra allows easier loop through the uindex with [i, i+1)
    return uindex;
}

}  // namespace alg

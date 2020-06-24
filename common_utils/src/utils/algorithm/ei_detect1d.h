#pragma once
#include "../container.h"
#include "../eigen.h"
#include "../eigeniter.h"
#include "../grppiex.h"
#include "../logging.h"
#include "../meta.h"
#include <set>

namespace alg {

namespace detect1d {

using Index = Eigen::Index;

/**
 * @brief Class to store intermediate results of \ref divconqfinder.
 * This is optionally created, populated, and reuturned
 * from the \ref divconqfinder function if the
 * template parameter \p ReturnStateCache is set.
 * @see divconqfinder
 */
template <typename F1, typename F2, typename F3, typename R1, typename R2,
          typename R3>
struct DivConqFinderStateCache {
    /// xdata vector
    Eigen::VectorXd xdata;
    /// ydata vector
    Eigen::VectorXd ydata;
    /// Index of chunks from \p chunkfunc.
    R1 chunkindex;
    /// Results of \p findfunc per chunk.
    std::vector<R2> findfunc_results;
    /// Index of segments of found features.
    R1 segmentindex;
    // Results of \p propfunc per segment.
    std::vector<R3> propfunc_results;
    // Results of \p propfunc per segment.
    std::vector<typename R3::value_type> results;
};

/**
 * @brief A divide-and-conquer feature detection algorithm.
 *  The returned functor takes a pair of xdata and ydata, produces
 *  a subset of xdata such that the corresponding ydata represent certain
 *  feature.
 * @param chunkfunc The functor that divides the input data into chunks for
 * individual processing. Expected signature: vector<pair<Index, Index>>(Index).
 *  - Params: size of input vector.
 *  - Return: Vector of pairs (si, ei) such that each defines a
 *      data chunk data.segment(si, ei-si).
 * @param findfunc The functor to identify feature indexes in each chunk.
 *  Expected signature: optional<tuple<vector<Index>, ...>>(Data, Data).
 *  - Params: xdata and ydata for one chunk
 *  - Return: Optional tuple. The first item of the tuple is a vector of
 * indexes at which data are identified as feature, which, as the next step, is
 * merged with those from other chunks into a unified feature segments vector.
 * Additional useful objects may be returned in the tuple.
 * @param propfunc The functor to operate on the found feature segments.
 *  Expected signature: optional<tuple<Scalar, ...>>(Data, Data).
 *  - Params: xdata and ydata for one selected feature segment.
 *  - Return: Optional tuple. The tuple contains properties computed
 *  for each feature segment. The first item of the properties should
 *  be of scalar type such that an Eigen vector could be constructed.
 * @param exmode The GRPPI execution mode to use.
 * @tparam ReturnStateCache If true, a DivConqFinderStateCache object contains
 *  the intermediate results is created, populated, and returned.
 * @return Functor that perform the feature detection.
 *  Expected signature: vector<tuple<Scalar, ...>>(Data, Data)
 *  - Params: xdata and ydata that is sorted, and mappable (contiguous in
 * memory) to Eigen::Vector.
 *  - Return: vector of non-null propfunc results.
 */
template <bool ReturnStateCache = false, typename F1, typename F2, typename F3,
          // set up compile type constraits for the functors
          typename R1 = REQUIRES_RT(
              meta::rt_is_instance<std::vector, std::pair, F1, Index>),
          typename R2 = REQUIRES_RT(
              meta::rt_is_instance<std::optional, std::tuple, F2,
                                   Eigen::VectorXd, Eigen::VectorXd>),
          typename R3 = REQUIRES_RT(
              meta::rt_is_instance<std::optional, std::tuple, F3,
                                   Eigen::VectorXd, Eigen::VectorXd>)>
auto divconqfinder(F1 &&chunkfunc, F2 &&findfunc, F3 &&propfunc,
                   grppiex::Mode exmode = grppiex::default_mode()) {
    return [fargs = FWD_CAPTURE(chunkfunc, findfunc, propfunc),
            exmode](const auto &xdata, const auto &ydata) {
        // unpack the functors
        auto &&[chunkfunc, findfunc, propfunc] = fargs;
        // check if sorted
        auto validate_sorted = [](const auto &m) {
            if (!eigeniter::iterapply(m, [](auto &&... args) {
                    return std::is_sorted(args...);
                })) {
                throw std::runtime_error("input data has to be sorted");
            }
            return true;
        };
        assert(validate_sorted(xdata));
        // map xdata and ydata to vectors
        namespace eiu = eigen_utils;
        auto xvec = eiu::asvec(xdata);
        auto yvec = eiu::asvec(ydata);

        // algorithm starts here
        // create empty cache object
        auto cache = DivConqFinderStateCache<F1, F2, F3, R1, R2, R3>();
        auto ex = grppiex::dyn_ex(exmode);
        // create chunks
        auto chunkindex = FWD(chunkfunc)(xvec.size());
        // update cache
        if constexpr (ReturnStateCache) {
            auto nchunks = chunkindex.size();
            SPDLOG_TRACE("cache: xdata {}", xvec);
            SPDLOG_TRACE("cache: ydata {}", yvec);
            SPDLOG_TRACE("cache: nchunks={}", nchunks);
            cache.xdata = xvec;
            cache.ydata = yvec;
            cache.chunkindex = chunkindex;
            // allocate findfunc_results
            cache.findfunc_results.resize(nchunks);
        }
        // find features
        auto featureindex = grppi::map_reduce(
            ex,
            // for each chunk, run findfunc and aggregate the result to a set.
            container_utils::unordered_enumerate(chunkindex),
            std::set<Index>{},
            [&xvec, &yvec, &cache, fargs = FWD_CAPTURE(findfunc)](
                const auto &chunk_) -> std::vector<Index> {
                auto &&[findfunc] = fargs;
                const auto &[ichunk, chunk] = chunk_;
                auto size = chunk.second - chunk.first;
                //SPDLOG_TRACE("findfunc on chunk #{} [{}, {}) size={}",
                //             ichunk, chunk.first, chunk.second, size);
                auto result = FWD(findfunc)(xvec.segment(chunk.first, size),
                                            yvec.segment(chunk.first, size));
                // shortcut to return empty if nothing found
                if (!result.has_value()) {
                    return {};
                }
                // get the feature index vector
                auto &index = std::get<0>(result.value());
                SPDLOG_TRACE("feature of length {} found in chunk #{} [{}, "
                             "{}) size={}",
                             index.size(), ichunk, chunk.first, chunk.second,
                             size);
                // restore the correct index w.r.t. the original data
                for (auto &i : index) {
                    i += chunk.first;
                }
                // update cache
                if constexpr (ReturnStateCache) {
                    cache.findfunc_results[ichunk] = result;
                }
                return std::move(index);
            },
            // reduction op to merge the results to a set
            [](auto &&lhs, auto &&rhs) {
                lhs.insert(std::make_move_iterator(rhs.begin()),
                           std::make_move_iterator(rhs.end()));
                return lhs;
            });
        // merge the peak of consecutive feature index to one to create
        // segmentindex
        std::vector<std::pair<Index, Index>> segmentindex;
        for (auto i : featureindex) {
            if (segmentindex.empty() || segmentindex.back().second < i) {
                segmentindex.emplace_back(i, i + 1); // [begin, past-last] range
            } else {
                ++(segmentindex.back().second);
            }
        }
        SPDLOG_DEBUG("found {} feature segments", segmentindex.size());
        // update cache
        if constexpr (ReturnStateCache) {
            cache.segmentindex = segmentindex;
            cache.propfunc_results.resize(segmentindex.size());
        }
        // grppi call for each feature
        // aggregate the found props to a vector
        using Prop = typename R3::value_type; // type of the valid property
        auto segmentindex_ = container_utils::unordered_enumerate(segmentindex);
        auto results = grppi::map_reduce(
            ex,
            // for each feature, run propfunc and aggregate the result to a
            // vector
            segmentindex_.begin(), segmentindex_.end(), std::vector<Prop>{},
            [&xvec, &yvec, &cache,
             fargs = FWD_CAPTURE(propfunc)](const auto &segment_) {
                auto &&[propfunc] = fargs;
                const auto &[isegment, segment] = segment_;
                auto size = segment.second - segment.first;
                // SPDLOG_TRACE("checking feature #{} [{}, {}]", isegment,
                // segment.first, segment.second); compute property
                auto result = FWD(propfunc)(xvec.segment(segment.first, size),
                                            yvec.segment(segment.first, size));
                // update cache
                if constexpr (ReturnStateCache) {
                    cache.propfunc_results[isegment] = result;
                }
                if (result.has_value()) {
                    return std::vector<Prop>{result.value()};
                }
                return std::vector<Prop>{};
            },
            // reduction op to merge the results if has value
            [](auto &&lhs, auto &&rhs) {
                lhs.insert(lhs.end(),
                           std::make_move_iterator(rhs.begin()),
                           std::make_move_iterator(rhs.end()));
                return lhs;
            });
        SPDLOG_TRACE("number of results: {}", results.size());
        if constexpr (ReturnStateCache) {
            cache.results = results;
            return std::make_tuple(std::move(results), std::move(cache));
        } else {
            return results;
        }
    };
}

} // namespace detect1d
} // namespace finder
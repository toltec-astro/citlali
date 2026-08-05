#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>

#include <Eigen/Sparse>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/pointing.h>

using timestream::TCData;

// selects the type of TCData
using timestream::TCDataKind;

namespace mapmaking {

class NaiveMapmaker {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
    std::unique_ptr<std::mutex> naive_mutex = std::make_unique<std::mutex>();

    // toltec array mounting angle
    std::map<int, double> install_ang = {
        {-1,-1},
        {0,pi/2},
        {1,-pi/2},
        {2,-pi/2},
    };

    // toltec detector orientation angles
    std::map<int, double> fgs = {
        {-1,-1},
        {0,0},
        {1,pi/4},
        {2,pi/2},
        {3,3*pi/4}
    };

    template <typename Scalar, typename Derived>
    void add_sparse_to_dense(std::vector<Eigen::Triplet<Scalar>> &triplets,
                             Eigen::DenseBase<Derived> &dense_matrix) {
        Eigen::SparseMatrix<Scalar> sparse_matrix(dense_matrix.rows(),dense_matrix.cols());
        sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
        dense_matrix += sparse_matrix;
        std::vector<Eigen::Triplet<Scalar>>().swap(triplets);
    }

private:
    template <typename Scalar>
    static Scalar checked_aggregate_add(Scalar lhs, Scalar rhs,
                                        const char *quantity) {
        if constexpr (std::is_floating_point_v<Scalar>) {
            if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
                throw std::runtime_error(
                    std::string("ordinary naive ") + quantity +
                    " aggregate is non-finite");
            }
            const Scalar result = lhs + rhs;
            if (!std::isfinite(result)) {
                throw std::runtime_error(
                    std::string("ordinary naive ") + quantity +
                    " aggregate overflowed");
            }
            return result;
        }
        else {
            static_assert(std::is_integral_v<Scalar> &&
                          std::is_signed_v<Scalar>);
            if ((rhs > 0 &&
                 lhs > std::numeric_limits<Scalar>::max() - rhs) ||
                (rhs < 0 &&
                 lhs < std::numeric_limits<Scalar>::lowest() - rhs)) {
                throw std::runtime_error(
                    std::string("ordinary naive ") + quantity +
                    " aggregate overflowed");
            }
            return static_cast<Scalar>(lhs + rhs);
        }
    }

    template <typename Scalar>
    static Eigen::SparseMatrix<Scalar> make_checked_sparse_delta(
        const std::vector<Eigen::Triplet<Scalar>> &triplets,
        Eigen::Index rows, Eigen::Index cols, const char *quantity) {
        for (const auto &triplet : triplets) {
            if (triplet.row() < 0 || triplet.row() >= rows ||
                triplet.col() < 0 || triplet.col() >= cols) {
                throw std::runtime_error(
                    std::string("ordinary naive ") + quantity +
                    " aggregate index is outside the target plane");
            }
            (void)checked_aggregate_add(
                Scalar{}, triplet.value(), quantity);
        }

        Eigen::SparseMatrix<Scalar> delta(rows, cols);
        delta.setFromTriplets(
            triplets.begin(), triplets.end(),
            [quantity](Scalar lhs, Scalar rhs) {
                return checked_aggregate_add(lhs, rhs, quantity);
            });
        return delta;
    }

    template <typename Scalar, typename Derived>
    static void preflight_sparse_commit(
        const Eigen::SparseMatrix<Scalar> &delta,
        const Eigen::DenseBase<Derived> &dense, const char *quantity) {
        if (delta.rows() != dense.rows() || delta.cols() != dense.cols()) {
            throw std::runtime_error(
                std::string("ordinary naive ") + quantity +
                " aggregate shape differs from the target plane");
        }
        for (Eigen::Index outer = 0; outer < delta.outerSize(); ++outer) {
            for (typename Eigen::SparseMatrix<Scalar>::InnerIterator value(
                     delta, outer);
                 value; ++value) {
                (void)checked_aggregate_add(
                    static_cast<Scalar>(dense(value.row(), value.col())),
                    value.value(), quantity);
            }
        }
    }

    static Eigen::Index checked_projected_index(double coordinate) {
        if (!std::isfinite(coordinate)) {
            throw std::runtime_error(
                "ordinary naive projected coordinate is non-finite");
        }
        const long double projected = static_cast<long double>(coordinate);
        if (projected < static_cast<long double>(
                            std::numeric_limits<Eigen::Index>::lowest()) ||
            projected > static_cast<long double>(
                            std::numeric_limits<Eigen::Index>::max()) ||
            projected < static_cast<long double>(
                            std::numeric_limits<long long>::lowest()) ||
            projected > static_cast<long double>(
                            std::numeric_limits<long long>::max())) {
            throw std::runtime_error(
                "ordinary naive projected coordinate is outside the representable index domain");
        }
        // Keep the established conversion for every accepted coordinate.
        return static_cast<Eigen::Index>(std::llround(coordinate));
    }

public:

    // run polarization?
    bool run_polarization;

    // allocate pointing matrix for polarization reduction
    template <class map_buffer_t>
    void allocate_pointing(map_buffer_t &, double, double, double, Eigen::Index, int, int);

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                             Eigen::DenseBase<Derived> &, std::string &, apt_t &, double, bool, bool,
                             const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps = nullptr);

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_naive_parallel(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                                      Eigen::DenseBase<Derived> &, std::string &, apt_t &, double, bool, bool,
                                      const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps = nullptr);

    // SCI-MAP-001 ordinary Stokes-I primitive. Contributions are gathered in
    // deterministic detector/sample order within each scan and committed
    // under the existing merge lock. Sequential and requested-parallel entry
    // points share the same race-free implementation and arithmetic; the
    // existing farm's cross-scan completion order remains a declared bounded
    // floating-equivalence boundary.
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_naive_science_contract(
        TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &,
        map_buffer_t &, Eigen::DenseBase<Derived> &, std::string &, apt_t &,
        double, bool, bool,
        const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps = nullptr);
};

template <class map_buffer_t>
void NaiveMapmaker::allocate_pointing(map_buffer_t &mb, double weight, double cos_2angle, double sin_2angle,
                                      Eigen::Index map_index, int ir, int ic) {
    int pix = mb.n_rows * ic + ir;
    Eigen::MatrixXd& matrix = mb.pointing[map_index];

    // calculate reused expressions
    double weight_cos_2angle = weight * cos_2angle;
    double weight_sin_2angle = weight * sin_2angle;
    double weight_cos2_2angle = weight * std::pow(cos_2angle, 2);
    double weight_sin2_2angle = weight * std::pow(sin_2angle, 2);
    double weight_cos_sin_2angle = weight * cos_2angle * sin_2angle;

    // update pointing matrix
    matrix(pix, 0) += weight;
    matrix(pix, 1) += weight_cos_2angle;
    matrix(pix, 2) += weight_sin_2angle;
    matrix(pix, 3) = matrix(pix, 1);  // previously set to += weight*cos_2angle, then directly assigned here
    matrix(pix, 4) += weight_cos2_2angle;
    matrix(pix, 5) += weight_cos_sin_2angle;
    matrix(pix, 6) = matrix(pix, 2);  // previously set to += weight*sin_2angle, then directly assigned here
    matrix(pix, 7) = matrix(pix, 5);  // reuse the result from matrix(pix,5)
    matrix(pix, 8) += weight_sin2_2angle;
}

template<class map_buffer_t, typename Derived, typename apt_t>
void NaiveMapmaker::populate_maps_naive_science_contract(
    TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, map_buffer_t &omb,
    map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
    std::string &pixel_axes, apt_t &apt, double d_fsmp, bool run_omb,
    bool run_noise,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps) {
    using DoubleTriplet = Eigen::Triplet<double>;
    using CountTriplet = Eigen::Triplet<std::int64_t>;
    struct ContributionRecord {
        Eigen::Index map_index;
        Eigen::Index row;
        Eigen::Index col;
        double signal;
        double coefficient;
        int detector_uid;
        int scan;
        int sample;
    };

    if (!std::isfinite(d_fsmp) || d_fsmp <= 0.0) {
        throw std::runtime_error(
            "ordinary naive mapmaking requires a finite positive sample rate");
    }
    if (in.scans.data.rows() != in.flags.data.rows() ||
        in.scans.data.cols() != in.flags.data.cols() ||
        map_indices.size() != in.scans.data.cols()) {
        throw std::runtime_error(
            "ordinary naive mapmaking input shape mismatch");
    }

    const Eigen::Index n_pts = in.scans.data.rows();
    const Eigen::Index n_dets = in.scans.data.cols();
    const bool run_kernel = run_omb && !omb.kernel.empty();
    const bool run_coverage = run_omb && !omb.coverage.empty();
    const bool science_products =
        run_omb && omb.science_products.initialized &&
        !omb.science_products.geometric_hits.empty();
    const bool use_omb_noise = !omb.noise.empty();
    const bool use_cmb_noise = !cmb.noise.empty() && !use_omb_noise;
    map_buffer_t *nmb = use_omb_noise ? &omb :
                        (use_cmb_noise ? &cmb : nullptr);

    if (run_kernel &&
        (in.kernel.data.rows() != n_pts ||
         in.kernel.data.cols() != n_dets)) {
        throw std::runtime_error(
            "ordinary naive required kernel shape differs from samples");
    }

    if (run_noise && nmb == nullptr) {
        throw std::runtime_error(
            "ordinary naive noise mapping requires an observation-owned noise buffer");
    }
    if (run_noise && (nmb->n_noise < 0 ||
                      static_cast<Eigen::Index>(nmb->noise.size()) !=
                          static_cast<Eigen::Index>(omb.signal.size()))) {
        throw std::runtime_error(
            "ordinary naive noise-map inventory differs from map inventory");
    }

    const auto n_maps = static_cast<Eigen::Index>(omb.signal.size());
    std::vector<std::vector<DoubleTriplet>> signals(
        static_cast<std::size_t>(n_maps));
    std::vector<std::vector<DoubleTriplet>> weights(
        static_cast<std::size_t>(n_maps));
    std::vector<std::vector<DoubleTriplet>> kernels(
        run_kernel ? static_cast<std::size_t>(n_maps) : 0U);
    std::vector<std::vector<DoubleTriplet>> coverages(
        run_coverage ? static_cast<std::size_t>(n_maps) : 0U);
    std::vector<std::vector<CountTriplet>> geometric_hits(
        science_products ? static_cast<std::size_t>(n_maps) : 0U);
    std::vector<std::vector<CountTriplet>> contributing_hits(
        science_products ? static_cast<std::size_t>(n_maps) : 0U);
    std::vector<std::vector<DoubleTriplet>> eligible_exposure(
        science_products ? static_cast<std::size_t>(n_maps) : 0U);
    std::vector<std::vector<DoubleTriplet>> retained_exposure(
        science_products ? static_cast<std::size_t>(n_maps) : 0U);
    std::vector<ContributionRecord> contribution_records;

    std::vector<Eigen::Tensor<double, 3>> noise_delta;
    if (run_noise) {
        noise_delta.reserve(nmb->noise.size());
        for (std::size_t slot = 0; slot < nmb->noise.size(); ++slot) {
            if (nmb->noise[slot].dimension(0) != nmb->n_rows ||
                nmb->noise[slot].dimension(1) != nmb->n_cols ||
                nmb->noise[slot].dimension(2) != nmb->n_noise) {
                throw std::runtime_error(
                    "ordinary naive noise-map shape mismatch");
            }
            noise_delta.emplace_back(nmb->n_rows, nmb->n_cols,
                                     nmb->n_noise);
            noise_delta.back().setZero();
        }
    }

    const double sample_seconds = 1.0 / d_fsmp;
    for (Eigen::Index det = 0; det < n_dets; ++det) {
        const Eigen::Index map_index = map_indices(det);
        if (map_index < 0 || map_index >= n_maps) {
            throw std::runtime_error(
                "ordinary naive map index is outside the allocated bundle");
        }
        if (active_maps != nullptr &&
            (map_index >= active_maps->size() || !(*active_maps)(map_index))) {
            continue;
        }
        int detector_uid = static_cast<int>(det);
        const auto uid_it = apt.find("uid");
        if (uid_it != apt.end() && det < uid_it->second.size() &&
            std::isfinite(uid_it->second(det))) {
            detector_uid =
                static_cast<int>(std::llround(uid_it->second(det)));
        }

        auto [lat, lon] = engine_utils::calc_det_pointing(
            in.tel_data.data, apt["x_t"](det), apt["y_t"](det), pixel_axes,
            in.pointing_offsets_arcsec.data, omb.map_grouping);
        if (lat.size() != n_pts || lon.size() != n_pts) {
            throw std::runtime_error(
                "ordinary naive projection cardinality differs from samples");
        }

        const Eigen::VectorXd omb_irow =
            lat.array() / omb.pixel_size_rad + (omb.n_rows - 1) / 2.0;
        const Eigen::VectorXd omb_icol =
            lon.array() / omb.pixel_size_rad + (omb.n_cols - 1) / 2.0;
        Eigen::VectorXd cmb_irow;
        Eigen::VectorXd cmb_icol;
        if (use_cmb_noise) {
            cmb_irow =
                lat.array() / cmb.pixel_size_rad + (cmb.n_rows - 1) / 2.0;
            cmb_icol =
                lon.array() / cmb.pixel_size_rad + (cmb.n_cols - 1) / 2.0;
        }

        const bool detector_eligible = apt["flag"](det) == 0;
        for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
            const bool projection_finite =
                std::isfinite(omb_irow(sample)) &&
                std::isfinite(omb_icol(sample));
            Eigen::Index omb_row = -1;
            Eigen::Index omb_col = -1;
            bool omb_in_bounds = false;
            if (projection_finite) {
                omb_row = checked_projected_index(omb_irow(sample));
                omb_col = checked_projected_index(omb_icol(sample));
                omb_in_bounds = omb_row >= 0 && omb_row < omb.n_rows &&
                                omb_col >= 0 && omb_col < omb.n_cols;
                if (science_products && omb_in_bounds) {
                    geometric_hits[static_cast<std::size_t>(map_index)]
                        .emplace_back(omb_row, omb_col, 1);
                }
            }

            // Detector/sample flags are the upstream explicit-invalid
            // authority. Do not inspect a flagged sample's payload.
            if (!detector_eligible || in.flags.data(sample, det)) {
                continue;
            }
            if (!projection_finite) {
                throw std::runtime_error(
                    "eligible ordinary naive sample has non-finite projection");
            }
            if (run_omb && !omb_in_bounds) {
                continue;
            }
            if (science_products && omb_in_bounds) {
                eligible_exposure[static_cast<std::size_t>(map_index)]
                    .emplace_back(omb_row, omb_col, sample_seconds);
            }

            if (det >= in.weights.data.size()) {
                throw std::runtime_error(
                    "eligible ordinary naive detector lacks a coefficient");
            }
            const double coefficient = in.weights.data(det);
            if (!std::isfinite(coefficient)) {
                throw std::runtime_error(
                    "eligible ordinary naive coefficient is non-finite");
            }
            if (coefficient <= 0.0) {
                continue;
            }

            const double sample_signal = in.scans.data(sample, det);
            if (!std::isfinite(sample_signal)) {
                throw std::runtime_error(
                    "contributing ordinary naive signal is non-finite");
            }
            const double weighted_signal = sample_signal * coefficient;
            if (!std::isfinite(weighted_signal)) {
                throw std::runtime_error(
                    "ordinary naive weighted signal overflowed");
            }

            double weighted_kernel = 0.0;
            if (run_kernel) {
                const double kernel_value = in.kernel.data(sample, det);
                if (!std::isfinite(kernel_value)) {
                    throw std::runtime_error(
                        "contributing ordinary naive kernel is non-finite");
                }
                weighted_kernel = kernel_value * coefficient;
                if (!std::isfinite(weighted_kernel)) {
                    throw std::runtime_error(
                        "ordinary naive weighted kernel overflowed");
                }
            }

            Eigen::Index noise_row = omb_row;
            Eigen::Index noise_col = omb_col;
            if (use_cmb_noise) {
                if (!std::isfinite(cmb_irow(sample)) ||
                    !std::isfinite(cmb_icol(sample))) {
                    throw std::runtime_error(
                        "eligible ordinary naive coadd-noise projection is non-finite");
                }
                noise_row = checked_projected_index(cmb_irow(sample));
                noise_col = checked_projected_index(cmb_icol(sample));
            }
            const bool noise_in_bounds =
                !run_noise ||
                (noise_row >= 0 && noise_row < nmb->n_rows &&
                 noise_col >= 0 && noise_col < nmb->n_cols);
            std::vector<double> noise_values;
            if (run_noise && noise_in_bounds) {
                noise_values.reserve(static_cast<std::size_t>(nmb->n_noise));
                for (Eigen::Index realization = 0;
                     realization < nmb->n_noise; ++realization) {
                    if (realization >= in.noise.data.rows() ||
                        (nmb->randomize_dets && det >= in.noise.data.cols()) ||
                        (!nmb->randomize_dets && in.noise.data.cols() < 1)) {
                        throw std::runtime_error(
                            "ordinary naive realization-sign inventory mismatch");
                    }
                    const double sign = nmb->randomize_dets
                        ? static_cast<double>(in.noise.data(realization, det))
                        : static_cast<double>(in.noise.data(realization, 0));
                    const double noise_value = sign * weighted_signal;
                    if (!std::isfinite(sign) || !std::isfinite(noise_value)) {
                        throw std::runtime_error(
                            "contributing ordinary naive realization is non-finite");
                    }
                    noise_values.push_back(noise_value);
                }
            }

            if (run_omb && omb_in_bounds) {
                // Preserve the accepted arithmetic order in the staged merge.
                signals[static_cast<std::size_t>(map_index)].emplace_back(
                    omb_row, omb_col, weighted_signal);
                weights[static_cast<std::size_t>(map_index)].emplace_back(
                    omb_row, omb_col, coefficient);
                if (run_kernel) {
                    kernels[static_cast<std::size_t>(map_index)].emplace_back(
                        omb_row, omb_col, weighted_kernel);
                }
                if (run_coverage) {
                    coverages[static_cast<std::size_t>(map_index)].emplace_back(
                        omb_row, omb_col, sample_seconds);
                }
                if (science_products) {
                    contributing_hits[static_cast<std::size_t>(map_index)]
                        .emplace_back(omb_row, omb_col, 1);
                    retained_exposure[static_cast<std::size_t>(map_index)]
                        .emplace_back(omb_row, omb_col, sample_seconds);
                }
                if (omb.contribution_diag_enabled) {
                    contribution_records.push_back(ContributionRecord{
                        map_index, omb_row, omb_col, weighted_signal,
                        coefficient, detector_uid,
                        static_cast<int>(in.index.data),
                        static_cast<int>(sample)});
                }
            }
            for (Eigen::Index realization = 0;
                 realization < static_cast<Eigen::Index>(noise_values.size());
                 ++realization) {
                auto &aggregate =
                    noise_delta.at(static_cast<std::size_t>(map_index))(
                        noise_row, noise_col, realization);
                aggregate = checked_aggregate_add(
                    aggregate,
                    noise_values.at(static_cast<std::size_t>(realization)),
                    "realization");
            }
        }
    }

    std::vector<Eigen::SparseMatrix<double>> signal_delta;
    std::vector<Eigen::SparseMatrix<double>> weight_delta;
    std::vector<Eigen::SparseMatrix<double>> kernel_delta;
    std::vector<Eigen::SparseMatrix<double>> coverage_delta;
    std::vector<Eigen::SparseMatrix<std::int64_t>> geometric_hit_delta;
    std::vector<Eigen::SparseMatrix<std::int64_t>> contributing_hit_delta;
    std::vector<Eigen::SparseMatrix<double>> eligible_exposure_delta;
    std::vector<Eigen::SparseMatrix<double>> retained_exposure_delta;
    if (run_omb) {
        const auto count = static_cast<std::size_t>(n_maps);
        signal_delta.reserve(count);
        weight_delta.reserve(count);
        if (run_kernel) {
            kernel_delta.reserve(count);
        }
        if (run_coverage) {
            coverage_delta.reserve(count);
        }
        if (science_products) {
            geometric_hit_delta.reserve(count);
            contributing_hit_delta.reserve(count);
            eligible_exposure_delta.reserve(count);
            retained_exposure_delta.reserve(count);
        }
        for (Eigen::Index map_index = 0; map_index < n_maps; ++map_index) {
            const auto slot = static_cast<std::size_t>(map_index);
            signal_delta.push_back(make_checked_sparse_delta(
                signals[slot], omb.signal[slot].rows(),
                omb.signal[slot].cols(), "signal"));
            weight_delta.push_back(make_checked_sparse_delta(
                weights[slot], omb.weight[slot].rows(),
                omb.weight[slot].cols(), "coefficient"));
            if (run_kernel) {
                kernel_delta.push_back(make_checked_sparse_delta(
                    kernels[slot], omb.kernel[slot].rows(),
                    omb.kernel[slot].cols(), "kernel"));
            }
            if (run_coverage) {
                coverage_delta.push_back(make_checked_sparse_delta(
                    coverages[slot], omb.coverage[slot].rows(),
                    omb.coverage[slot].cols(), "coverage"));
            }
            if (science_products) {
                auto &products = omb.science_products;
                geometric_hit_delta.push_back(make_checked_sparse_delta(
                    geometric_hits[slot],
                    products.geometric_hits[slot].rows(),
                    products.geometric_hits[slot].cols(),
                    "geometric-hit count"));
                contributing_hit_delta.push_back(make_checked_sparse_delta(
                    contributing_hits[slot],
                    products.contributing_hits[slot].rows(),
                    products.contributing_hits[slot].cols(),
                    "contributing-hit count"));
                eligible_exposure_delta.push_back(make_checked_sparse_delta(
                    eligible_exposure[slot],
                    products.upstream_eligible_exposure[slot].rows(),
                    products.upstream_eligible_exposure[slot].cols(),
                    "upstream-eligible exposure"));
                retained_exposure_delta.push_back(make_checked_sparse_delta(
                    retained_exposure[slot],
                    products.retained_exposure[slot].rows(),
                    products.retained_exposure[slot].cols(),
                    "retained exposure"));
            }
        }
    }

    std::scoped_lock<std::mutex> lock(*naive_mutex);
    if (run_omb) {
        for (Eigen::Index map_index = 0; map_index < n_maps; ++map_index) {
            const auto slot = static_cast<std::size_t>(map_index);
            preflight_sparse_commit(signal_delta[slot], omb.signal[slot],
                                    "signal");
            preflight_sparse_commit(weight_delta[slot], omb.weight[slot],
                                    "coefficient");
            if (run_kernel) {
                preflight_sparse_commit(kernel_delta[slot], omb.kernel[slot],
                                        "kernel");
            }
            if (run_coverage) {
                preflight_sparse_commit(coverage_delta[slot],
                                        omb.coverage[slot], "coverage");
            }
            if (science_products) {
                auto &products = omb.science_products;
                preflight_sparse_commit(
                    geometric_hit_delta[slot], products.geometric_hits[slot],
                    "geometric-hit count");
                preflight_sparse_commit(
                    contributing_hit_delta[slot],
                    products.contributing_hits[slot],
                    "contributing-hit count");
                preflight_sparse_commit(
                    eligible_exposure_delta[slot],
                    products.upstream_eligible_exposure[slot],
                    "upstream-eligible exposure");
                preflight_sparse_commit(
                    retained_exposure_delta[slot],
                    products.retained_exposure[slot],
                    "retained exposure");
            }
        }
    }
    if (run_noise) {
        for (std::size_t map_index = 0; map_index < noise_delta.size();
             ++map_index) {
            for (Eigen::Index realization = 0;
                 realization < nmb->n_noise; ++realization) {
                for (Eigen::Index col = 0; col < nmb->n_cols; ++col) {
                    for (Eigen::Index row = 0; row < nmb->n_rows; ++row) {
                        (void)checked_aggregate_add(
                            nmb->noise[map_index](row, col, realization),
                            noise_delta[map_index](row, col, realization),
                            "realization");
                    }
                }
            }
        }
    }

    if (run_omb) {
        if (omb.contribution_diag_enabled) {
            omb.ensure_contribution_diag(n_maps);
        }
        for (Eigen::Index map_index = 0; map_index < n_maps; ++map_index) {
            const auto slot = static_cast<std::size_t>(map_index);
            // These are the established accepted-input commit operations and
            // order; every bundle member has already passed preflight.
            omb.signal[slot] += signal_delta[slot];
            omb.weight[slot] += weight_delta[slot];
            if (run_kernel) {
                omb.kernel[slot] += kernel_delta[slot];
            }
            if (run_coverage) {
                omb.coverage[slot] += coverage_delta[slot];
            }
            if (science_products) {
                auto &products = omb.science_products;
                products.geometric_hits[slot] += geometric_hit_delta[slot];
                products.contributing_hits[slot] +=
                    contributing_hit_delta[slot];
                products.upstream_eligible_exposure[slot] +=
                    eligible_exposure_delta[slot];
                products.retained_exposure[slot] +=
                    retained_exposure_delta[slot];
            }
        }
        for (const auto &record : contribution_records) {
            omb.record_contribution(
                record.map_index, record.row, record.col, record.signal,
                record.coefficient, record.detector_uid, record.scan,
                record.sample);
        }
    }
    if (run_noise) {
        for (std::size_t map_index = 0; map_index < noise_delta.size();
             ++map_index) {
            nmb->noise[map_index] += noise_delta[map_index];
        }
    }
}

template<class map_buffer_t, typename Derived, typename apt_t>
void NaiveMapmaker::populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, map_buffer_t &omb,
                                        map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
                                        std::string &pixel_axes, apt_t &apt, double d_fsmp,
                                        bool run_omb, bool run_noise,
                                        const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps) {

    if (!run_polarization) {
        populate_maps_naive_science_contract(
            in, omb, cmb, map_indices, pixel_axes, apt, d_fsmp, run_omb,
            run_noise, active_maps);
        return;
    }

    typedef Eigen::Triplet<double> T;
    std::vector<std::vector<T>> signals, weights, kernels, coverages;
    std::vector<std::vector<T>> cmb_signals, cmb_weights, cmb_kernels, cmb_coverages;

    const bool use_cmb = !cmb.noise.empty();
    const bool use_omb = !omb.noise.empty();
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();
    const bool run_hwpr = in.hwpr_angle.data.size()!=0;

    if (run_omb) {
        omb.ensure_contribution_diag(static_cast<Eigen::Index>(omb.signal.size()));
        signals.resize(omb.signal.size());
        weights.resize(omb.signal.size());

        if (run_kernel) {
            kernels.resize(omb.signal.size());
        }
        if (run_coverage) {
            coverages.resize(omb.signal.size());
        }
    }

    if (run_polarization && !cmb.signal.empty()) {
        cmb.ensure_contribution_diag(static_cast<Eigen::Index>(cmb.signal.size()));
        cmb_signals.resize(cmb.signal.size());
        cmb_weights.resize(cmb.signal.size());

        if (run_kernel) {
            cmb_kernels.resize(cmb.signal.size());
        }
        if (run_coverage) {
            cmb_coverages.resize(cmb.signal.size());
        }
    }

    map_buffer_t omb_copy, cmb_copy;
    // pointer to map buffer with noise maps
    map_buffer_t *nmb, *nmb_copy;

    omb_copy.n_rows = omb.n_rows;
    omb_copy.n_cols = omb.n_cols;

    cmb_copy.n_rows = cmb.n_rows;
    cmb_copy.n_cols = cmb.n_cols;

    if (run_noise) {
        if (use_omb) {
            omb_copy.noise = omb.noise;

            for (Eigen::Index i=0; i<omb.noise.size(); ++i) {
                omb_copy.noise[i].setZero();
            }
        }

        else {
            cmb_copy.noise = cmb.noise;

            for (Eigen::Index i=0; i<cmb.noise.size(); ++i) {
                cmb_copy.noise[i].setZero();
            }
        }
        nmb = use_cmb ? &cmb : (use_omb ? &omb : nullptr);
        nmb_copy = use_cmb ? &cmb_copy : (use_omb ? &omb_copy : nullptr);
    }

    // step to skip to reach next stokes param
    int step = omb.pointing.size();

    if (!omb.pointing.empty() && run_omb) {
        for (Eigen::Index i=0; i<omb.pointing.size(); ++i) {
            omb_copy.pointing.emplace_back(Eigen::MatrixXd::Zero(omb.pointing[i].rows(), omb.pointing[i].cols()));
        }
    }

    if (!cmb.pointing.empty() && run_omb) {
        for (Eigen::Index i=0; i<cmb.pointing.size(); ++i) {
            cmb_copy.pointing.emplace_back(Eigen::MatrixXd::Zero(cmb.pointing[i].rows(), cmb.pointing[i].cols()));
        }
    }

    // dimensions of data
    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    // signal, kernel and noise map values
    double signal, kernel, noise_v;

    // noise map indices
    Eigen::Index nmb_ir, nmb_ic;

    // cosine and sine of angles
    double angle, cos_2angle, sin_2angle;

    // add detector to map?
    bool run_det;

    for (Eigen::Index i=0; i<n_dets; ++i) {
        // skip fg = -1 if in polarization mode
        if (run_polarization && apt["loc"](i)==-1) {
            run_det = false;
        }
        else {
            run_det = true;
        }

        // skip completely flagged detectors
        if (apt["flag"](i)==0 && (in.flags.data.col(i).array()==0).any() && run_det) {
            // which map to assign detector to
            Eigen::Index map_index = map_indices(i);
            if (active_maps != nullptr &&
                (map_index < 0 || map_index >= active_maps->size() || !(*active_maps)(map_index))) {
                continue;
            }

            // indices for Q and U maps
            int q_index = map_index + step;
            int u_index = map_index + 2 * step;

            // array index
            Eigen::Index array_index = apt["array"](i);
            int det_uid = static_cast<int>(i);
            auto uid_it = apt.find("uid");
            if (uid_it != apt.end() && i < uid_it->second.size() &&
                std::isfinite(uid_it->second(i))) {
                det_uid = static_cast<int>(std::llround(uid_it->second(i)));
            }
            // get detector pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
                                                              pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

            Eigen::VectorXd alt;

            if (run_polarization) {
                std::tuple<Eigen::VectorXd,Eigen::VectorXd> altaz_tuple = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i),
                                                                                                           apt["y_t"](i), "altaz",
                                                                                                           in.pointing_offsets_arcsec.data,
                                                                                                           omb.map_grouping);
                alt = std::get<0>(altaz_tuple);
            }

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows - 1)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols - 1)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (use_cmb || (run_polarization && !cmb.signal.empty())) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows - 1)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols - 1)/2.;
            }

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; ++j) {
                // check if sample is flagged, ignore if so
                if (!in.flags.data(j,i) &&
                    std::isfinite(in.scans.data(j,i)) &&
                    i < in.weights.data.size() &&
                    std::isfinite(in.weights.data(i)) &&
                    in.weights.data(i) > 0.0) {
                    Eigen::Index omb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                    Eigen::Index omb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));

                    if (run_polarization) {
                        auto fg_index = apt["fg"](i);
                        if (run_hwpr) {
                            angle = 2*in.hwpr_angle.data(j) - (in.angle.data(j) + alt(j) + fgs[fg_index] + install_ang[array_index]);
                        }
                        else {
                            angle = in.angle.data(j) + alt(j) + fgs[fg_index] + install_ang[array_index];
                        }

                        cos_2angle = cos(2.*angle);
                        sin_2angle = sin(2.*angle);
                    }

                    if (run_omb) {
                        // make sure the data point is within the map
                        if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {
                            // populate signal map
                            signal = in.scans.data(j,i)*in.weights.data(i);
                            signals[map_index].push_back(T(omb_ir,omb_ic,signal));
                            if (omb.contribution_diag_enabled) {
                                std::scoped_lock<std::mutex> lk(*naive_mutex);
                                omb.record_contribution(map_index, omb_ir, omb_ic, signal,
                                                        in.weights.data(i), det_uid,
                                                        static_cast<int>(in.index.data),
                                                        static_cast<int>(j));
                            }

                            // populate weight map
                            weights[map_index].push_back(T(omb_ir,omb_ic,in.weights.data(i)));

                            // populate kernel map
                            if (run_kernel) {
                                kernel = in.kernel.data(j,i)*in.weights.data(i);
                                kernels[map_index].push_back(T(omb_ir,omb_ic,kernel));
                            }

                            // populate coverage map
                            if (run_coverage) {
                                coverages[map_index].push_back(T(omb_ir,omb_ic,1./d_fsmp));
                            }

                            if (run_polarization) {
                                // calculate pointing matrix
                                allocate_pointing(omb_copy, in.weights.data(i), cos_2angle, sin_2angle, map_index, omb_ir, omb_ic);

                                // update signal map Q and U
                                signals[q_index].push_back(T(omb_ir,omb_ic,signal*cos_2angle));
                                signals[u_index].push_back(T(omb_ir,omb_ic,signal*sin_2angle));
                                if (omb.contribution_diag_enabled) {
                                    std::scoped_lock<std::mutex> lk(*naive_mutex);
                                    omb.record_contribution(q_index, omb_ir, omb_ic,
                                                            signal*cos_2angle,
                                                            in.weights.data(i), det_uid,
                                                            static_cast<int>(in.index.data),
                                                            static_cast<int>(j));
                                    omb.record_contribution(u_index, omb_ir, omb_ic,
                                                            signal*sin_2angle,
                                                            in.weights.data(i), det_uid,
                                                            static_cast<int>(in.index.data),
                                                            static_cast<int>(j));
                                }

                                // update kernel map Q and U
                                if (run_kernel) {
                                    kernels[q_index].push_back(T(omb_ir,omb_ic,kernel*cos_2angle));
                                    kernels[u_index].push_back(T(omb_ir,omb_ic,kernel*sin_2angle));
                                }
                            }
                        }
                    }

                    if (run_polarization && !cmb.signal.empty() && run_omb) {
                        Eigen::Index cmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                        Eigen::Index cmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));

                        // make sure the data point is within the map
                        if ((cmb_ir >= 0) && (cmb_ir < cmb.n_rows) && (cmb_ic >= 0) && (cmb_ic < cmb.n_cols)) {
                            // populate signal map
                            signal = in.scans.data(j,i)*in.weights.data(i);
                            cmb_signals[map_index].push_back(T(cmb_ir,cmb_ic,signal));
                            if (cmb.contribution_diag_enabled) {
                                std::scoped_lock<std::mutex> lk(*naive_mutex);
                                cmb.record_contribution(map_index, cmb_ir, cmb_ic, signal,
                                                        in.weights.data(i), det_uid,
                                                        static_cast<int>(in.index.data),
                                                        static_cast<int>(j));
                            }

                            // populate weight map
                            cmb_weights[map_index].push_back(T(cmb_ir,cmb_ic,in.weights.data(i)));

                            // populate kernel map
                            if (run_kernel) {
                                kernel = in.kernel.data(j,i)*in.weights.data(i);
                                cmb_kernels[map_index].push_back(T(cmb_ir,cmb_ic,kernel));
                            }

                            // populate coverage map
                            if (run_coverage) {
                                cmb_coverages[map_index].push_back(T(cmb_ir,cmb_ic,1./d_fsmp));
                            }

                            // calculate pointing matrix
                            allocate_pointing(cmb_copy, in.weights.data(i), cos_2angle, sin_2angle, map_index, cmb_ir, cmb_ic);

                            // update signal map Q and U
                            cmb_signals[q_index].push_back(T(cmb_ir,cmb_ic,signal*cos_2angle));
                            cmb_signals[u_index].push_back(T(cmb_ir,cmb_ic,signal*sin_2angle));
                            if (cmb.contribution_diag_enabled) {
                                std::scoped_lock<std::mutex> lk(*naive_mutex);
                                cmb.record_contribution(q_index, cmb_ir, cmb_ic,
                                                        signal*cos_2angle,
                                                        in.weights.data(i), det_uid,
                                                        static_cast<int>(in.index.data),
                                                        static_cast<int>(j));
                                cmb.record_contribution(u_index, cmb_ir, cmb_ic,
                                                        signal*sin_2angle,
                                                        in.weights.data(i), det_uid,
                                                        static_cast<int>(in.index.data),
                                                        static_cast<int>(j));
                            }

                            // update kernel map Q and U
                            if (run_kernel) {
                                cmb_kernels[q_index].push_back(T(cmb_ir,cmb_ic,kernel*cos_2angle));
                                cmb_kernels[u_index].push_back(T(cmb_ir,cmb_ic,kernel*sin_2angle));
                            }
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        // if coaddition is enabled
                        if (use_cmb) {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));
                        }
                        // else make noise maps for obs
                        else {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));
                        }

                        // coadd into current noise map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                            //if (run_polarization) {
                                //if (use_cmb) {
                                    // calculate pointing matrix for cmb
                                  //  allocate_pointing(cmb_copy, in.weights.data(i), cos_2angle, sin_2angle, map_index, nmb_ir, nmb_ic);
                                //}
                            //}
                            // loop through noise maps
                            for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                // randomizing on dets
                                if (nmb->randomize_dets) {
                                    noise_v = in.noise.data(nn,i)*in.scans.data(j,i)*in.weights.data(i);
                                }
                                else {
                                    noise_v = in.noise.data(nn)*in.scans.data(j,i)*in.weights.data(i);
                                }
                                // add noise value to current noise map
                                nmb_copy->noise[map_index](nmb_ir,nmb_ic,nn) += noise_v;

                                if (run_polarization) {
                                    // update noise map Q
                                    nmb_copy->noise[q_index](nmb_ir,nmb_ic,nn) += noise_v*cos_2angle;
                                    // update noise map U
                                    nmb_copy->noise[u_index](nmb_ir,nmb_ic,nn) += noise_v*sin_2angle;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    {
        std::scoped_lock<std::mutex> lk(*naive_mutex);
        if (run_omb) {
            for (int i=0; i<omb.signal.size(); ++i) {
                add_sparse_to_dense(signals[i],omb.signal[i]);
                add_sparse_to_dense(weights[i],omb.weight[i]);

                if (run_kernel) {
                    add_sparse_to_dense(kernels[i],omb.kernel[i]);
                }

                if (run_coverage) {
                    add_sparse_to_dense(coverages[i],omb.coverage[i]);
                }
            }
            if (!omb.pointing.empty()) {
                for (int i=0; i<omb.pointing.size(); ++i) {
                    omb.pointing[i] += omb_copy.pointing[i];
                }
            }
        }

        if (run_polarization && !cmb.signal.empty()) {
            for (int i=0; i<cmb.signal.size(); ++i) {
                add_sparse_to_dense(cmb_signals[i],cmb.signal[i]);
                add_sparse_to_dense(cmb_weights[i],cmb.weight[i]);

                if (run_kernel) {
                    add_sparse_to_dense(cmb_kernels[i],cmb.kernel[i]);
                }

                if (run_coverage) {
                    add_sparse_to_dense(cmb_coverages[i],cmb.coverage[i]);
                }
            }
        }

        if (run_noise) {
            for (int i=0; i<nmb->noise.size(); ++i) {
                nmb->noise[i] += nmb_copy->noise[i];
            }
        }

        if (!cmb.pointing.empty() && run_omb) {
            for (int i=0; i<cmb.pointing.size(); ++i) {
                cmb.pointing[i] += cmb_copy.pointing[i];
            }
        }
    }

    nmb = nullptr;
    nmb_copy = nullptr;
}

template<class map_buffer_t, typename Derived, typename apt_t>
void NaiveMapmaker::populate_maps_naive_parallel(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, map_buffer_t &omb,
                                                 map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
                                                 std::string &pixel_axes, apt_t &apt, double d_fsmp,
                                                 bool run_omb, bool run_noise,
                                                 const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps) {

    if (!run_polarization) {
        populate_maps_naive_science_contract(
            in, omb, cmb, map_indices, pixel_axes, apt, d_fsmp, run_omb,
            run_noise, active_maps);
        return;
    }

    const bool use_cmb = !cmb.noise.empty();
    const bool use_omb = !omb.noise.empty();
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();
    const bool run_hwpr = in.hwpr_angle.data.size()!=0;

    // pointer to map buffer with noise maps
    map_buffer_t *nmb;

    if (run_noise) {
        nmb = use_cmb ? &cmb : (use_omb ? &omb : nullptr);
    }

    if (run_omb) {
        omb.ensure_contribution_diag(static_cast<Eigen::Index>(omb.signal.size()));
    }
    if (run_polarization && !cmb.signal.empty()) {
        cmb.ensure_contribution_diag(static_cast<Eigen::Index>(cmb.signal.size()));
    }

    // step to skip to reach next stokes param
    int step = omb.pointing.size();

    // dimensions of data
    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    bool unique_map_indices =
        (map_indices.size() == n_dets && omb.signal.size() == static_cast<std::size_t>(n_dets));
    if (unique_map_indices) {
        std::vector<unsigned char> seen(omb.signal.size(), 0);
        for (Eigen::Index i = 0; i < n_dets; ++i) {
            const auto idx = map_indices(i);
            if (idx < 0 || idx >= static_cast<Eigen::Index>(seen.size()) || seen[static_cast<std::size_t>(idx)] != 0) {
                unique_map_indices = false;
                break;
            }
            seen[static_cast<std::size_t>(idx)] = 1;
        }
    }

    if (!unique_map_indices) {
        logger->warn("populate_maps_naive_parallel requires unique map indices; falling back to populate_maps_naive");
        populate_maps_naive(in, omb, cmb, map_indices, pixel_axes, apt, d_fsmp, run_omb, run_noise, active_maps);
        return;
    }

    // signal, kernel and noise map values
    double signal, kernel, noise_v;

    // noise map indices
    Eigen::Index nmb_ir, nmb_ic;

    // cosine and sine of angles
    double angle, cos_2angle, sin_2angle;

    // add detector to map?
    bool run_det;

    // placeholder vectors of size ndet for grppi maps
    std::vector<int> map_in_vec, map_out_vec;
    map_in_vec.resize(omb.signal.size());
    std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    map_out_vec.resize(map_in_vec.size());

    // parallelize over detectors
    //for (Eigen::Index i=0; i<n_dets; ++i) {
    grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
    //for (Eigen::Index i=0; i<n_dets; ++i) {
        // skip fg = -1 if in polarization mode
        if (run_polarization && apt["loc"](i)==-1) {
            run_det = false;
        }
        else {
            run_det = true;
        }

        // skip completely flagged detectors
        if (apt["flag"](i)==0 && (in.flags.data.col(i).array()==0).any() && run_det) {
            // which map to assign detector to
            Eigen::Index map_index = map_indices(i);
            if (active_maps != nullptr &&
                (map_index < 0 || map_index >= active_maps->size() || !(*active_maps)(map_index))) {
                return 0;
            }

            // indices for Q and U maps
            int q_index = map_index + step;
            int u_index = map_index + 2 * step;

            // array index
            Eigen::Index array_index = apt["array"](i);
            int det_uid = static_cast<int>(i);
            auto uid_it = apt.find("uid");
            if (uid_it != apt.end() && i < uid_it->second.size() &&
                std::isfinite(uid_it->second(i))) {
                det_uid = static_cast<int>(std::llround(uid_it->second(i)));
            }
            // get detector pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
                                                              pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

            Eigen::VectorXd alt;

            if (run_polarization) {
                std::tuple<Eigen::VectorXd,Eigen::VectorXd> altaz_tuple = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i),
                                                                                                           apt["y_t"](i), "altaz",
                                                                                                           in.pointing_offsets_arcsec.data,
                                                                                                           omb.map_grouping);
                alt = std::get<0>(altaz_tuple);
            }

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows - 1)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols - 1)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (use_cmb || (run_polarization && !cmb.signal.empty())) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows - 1)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols - 1)/2.;
            }

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; ++j) {
                // check if sample is flagged, ignore if so
                if (!in.flags.data(j,i) &&
                    std::isfinite(in.scans.data(j,i)) &&
                    i < in.weights.data.size() &&
                    std::isfinite(in.weights.data(i)) &&
                    in.weights.data(i) > 0.0) {
                    Eigen::Index omb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                    Eigen::Index omb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));

                    if (run_polarization) {
                        auto fg_index = apt["fg"](i);
                        if (run_hwpr) {
                            angle = 2*in.hwpr_angle.data(j) - (in.angle.data(j) + alt(j) + fgs[fg_index] + install_ang[array_index]);
                        }
                        else {
                            angle = in.angle.data(j) + alt(j) + fgs[fg_index] + install_ang[array_index];
                        }

                        cos_2angle = cos(2.*angle);
                        sin_2angle = sin(2.*angle);
                    }

                    if (run_omb) {
                        // make sure the data point is within the map
                        if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {
                            // populate signal map
                            signal = in.scans.data(j,i)*in.weights.data(i);
                            omb.signal[map_index](omb_ir, omb_ic) += signal;
                            if (omb.contribution_diag_enabled) {
                                std::scoped_lock<std::mutex> lk(*naive_mutex);
                                omb.record_contribution(map_index, omb_ir, omb_ic, signal,
                                                        in.weights.data(i), det_uid,
                                                        static_cast<int>(in.index.data),
                                                        static_cast<int>(j));
                            }
                            //signals[map_index].push_back(T(omb_ir,omb_ic,signal));

                            // populate weight map
                            //weights[map_index].push_back(T(omb_ir,omb_ic,in.weights.data(i)));
                            omb.weight[map_index](omb_ir, omb_ic) += in.weights.data(i);

                            // populate kernel map
                            if (run_kernel) {
                                kernel = in.kernel.data(j,i)*in.weights.data(i);
                                //kernels[map_index].push_back(T(omb_ir,omb_ic,kernel));
                                omb.kernel[map_index](omb_ir, omb_ic) += kernel;
                            }

                            // populate coverage map
                            if (run_coverage) {
                                //coverages[map_index].push_back(T(omb_ir,omb_ic,1./d_fsmp));
                                omb.coverage[map_index](omb_ir, omb_ic) += 1./d_fsmp;
                            }

                            if (run_polarization) {
                                // calculate pointing matrix
                                allocate_pointing(omb, in.weights.data(i), cos_2angle, sin_2angle, map_index, omb_ir, omb_ic);

                                // update signal map Q and U
                                //signals[q_index].push_back(T(omb_ir,omb_ic,signal*cos_2angle));
                                //signals[u_index].push_back(T(omb_ir,omb_ic,signal*sin_2angle));

                                omb.signal[q_index](omb_ir, omb_ic) += signal*cos_2angle;
                                omb.signal[u_index](omb_ir, omb_ic) += signal*sin_2angle;
                                if (omb.contribution_diag_enabled) {
                                    std::scoped_lock<std::mutex> lk(*naive_mutex);
                                    omb.record_contribution(q_index, omb_ir, omb_ic,
                                                            signal*cos_2angle,
                                                            in.weights.data(i), det_uid,
                                                            static_cast<int>(in.index.data),
                                                            static_cast<int>(j));
                                    omb.record_contribution(u_index, omb_ir, omb_ic,
                                                            signal*sin_2angle,
                                                            in.weights.data(i), det_uid,
                                                            static_cast<int>(in.index.data),
                                                            static_cast<int>(j));
                                }


                                // update kernel map Q and U
                                if (run_kernel) {
                                    //kernels[q_index].push_back(T(omb_ir,omb_ic,kernel*cos_2angle));
                                    //kernels[u_index].push_back(T(omb_ir,omb_ic,kernel*sin_2angle));
                                    omb.kernel[q_index](omb_ir, omb_ic) += kernel*cos_2angle;
                                    omb.kernel[u_index](omb_ir, omb_ic) += kernel*sin_2angle;
                                }
                            }
                        }
                    }

                    if (run_polarization && !cmb.signal.empty() && run_omb) {
                        Eigen::Index cmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                        Eigen::Index cmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));

                        // make sure the data point is within the map
                        if ((cmb_ir >= 0) && (cmb_ir < cmb.n_rows) && (cmb_ic >= 0) && (cmb_ic < cmb.n_cols)) {
                            // populate signal map
                            signal = in.scans.data(j,i)*in.weights.data(i);
                            //cmb_signals[map_index].push_back(T(cmb_ir,cmb_ic,signal));
                            cmb.signal[map_index](cmb_ir, cmb_ic) += signal;
                            if (cmb.contribution_diag_enabled) {
                                std::scoped_lock<std::mutex> lk(*naive_mutex);
                                cmb.record_contribution(map_index, cmb_ir, cmb_ic, signal,
                                                        in.weights.data(i), det_uid,
                                                        static_cast<int>(in.index.data),
                                                        static_cast<int>(j));
                            }

                            // populate weight map
                            //cmb_weights[map_index].push_back(T(cmb_ir,cmb_ic,in.weights.data(i)));
                            cmb.weight[map_index](cmb_ir, cmb_ic) += in.weights.data(i);

                            // populate kernel map
                            if (run_kernel) {
                                kernel = in.kernel.data(j,i)*in.weights.data(i);
                                //cmb_kernels[map_index].push_back(T(cmb_ir,cmb_ic,kernel));
                                cmb.kernel[map_index](cmb_ir, cmb_ic) += kernel;
                            }

                            // populate coverage map
                            if (run_coverage) {
                                //cmb_coverages[map_index].push_back(T(cmb_ir,cmb_ic,1./d_fsmp));
                                cmb.coverage[map_index](cmb_ir, cmb_ic) += 1./d_fsmp;
                            }

                            // calculate pointing matrix
                            allocate_pointing(cmb, in.weights.data(i), cos_2angle, sin_2angle, map_index, cmb_ir, cmb_ic);

                            // update signal map Q and U
                            //cmb_signals[q_index].push_back(T(cmb_ir,cmb_ic,signal*cos_2angle));
                            //cmb_signals[u_index].push_back(T(cmb_ir,cmb_ic,signal*sin_2angle));

                            cmb.signal[q_index](cmb_ir, cmb_ic) += signal*cos_2angle;
                            cmb.signal[u_index](cmb_ir, cmb_ic) += signal*sin_2angle;
                            if (cmb.contribution_diag_enabled) {
                                std::scoped_lock<std::mutex> lk(*naive_mutex);
                                cmb.record_contribution(q_index, cmb_ir, cmb_ic,
                                                        signal*cos_2angle,
                                                        in.weights.data(i), det_uid,
                                                        static_cast<int>(in.index.data),
                                                        static_cast<int>(j));
                                cmb.record_contribution(u_index, cmb_ir, cmb_ic,
                                                        signal*sin_2angle,
                                                        in.weights.data(i), det_uid,
                                                        static_cast<int>(in.index.data),
                                                        static_cast<int>(j));
                            }


                            // update kernel map Q and U
                            if (run_kernel) {
                                //cmb_kernels[q_index].push_back(T(cmb_ir,cmb_ic,kernel*cos_2angle));
                                //cmb_kernels[u_index].push_back(T(cmb_ir,cmb_ic,kernel*sin_2angle));
                                cmb.kernel[q_index](cmb_ir, cmb_ic) += kernel*cos_2angle;
                                cmb.kernel[u_index](cmb_ir, cmb_ic) += kernel*sin_2angle;
                            }
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        // if coaddition is enabled
                        if (use_cmb) {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));
                        }
                        // else make noise maps for obs
                        else {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));
                        }

                        // coadd into current noise map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                            //if (run_polarization) {
                                //if (use_cmb) {
                                    // calculate pointing matrix for cmb
                                  //  allocate_pointing(cmb, in.weights.data(i), cos_2angle, sin_2angle, map_index, nmb_ir, nmb_ic);
                                //}
                            //}
                            // loop through noise maps
                            for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                // randomizing on dets
                                if (nmb->randomize_dets) {
                                    noise_v = in.noise.data(nn,i)*in.scans.data(j,i)*in.weights.data(i);
                                }
                                else {
                                    noise_v = in.noise.data(nn)*in.scans.data(j,i)*in.weights.data(i);
                                }
                                // add noise value to current noise map
                                nmb->noise[map_index](nmb_ir,nmb_ic,nn) += noise_v;

                                if (run_polarization) {
                                    // update noise map Q
                                    nmb->noise[q_index](nmb_ir,nmb_ic,nn) += noise_v*cos_2angle;
                                    // update noise map U
                                    nmb->noise[u_index](nmb_ir,nmb_ic,nn) += noise_v*sin_2angle;
                                }
                            }
                        }
                    }
                }
            }
        }
        return 0;
    });

    nmb = nullptr;
}
} // namespace mapmaking

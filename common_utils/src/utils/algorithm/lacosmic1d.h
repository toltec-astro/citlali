#pragma once

#include "../container.h"
#include "../eigen.h"
#include "../grppiex.h"
#include "ei_stats.h"

namespace alg {

template <typename DerivedA, typename DerivedB, typename DerivedC>
auto lacosmic1d(const Eigen::DenseBase<DerivedA> &data,
                const Eigen::DenseBase<DerivedB> &uncertainty,
                const Eigen::DenseBase<DerivedC> &mask, double sigclip = 4.5,
                double sigfrac = 0.3, double objlim = 5., int maxiter = 4,
                double fill_value = 0.,
                const grppi::dynamic_execution &ex = grppiex::dyn_ex()) {
    static_assert(DerivedA::IsVectorAtCompileTime, "EXPECT VECTOR");
    constexpr auto block_size = 2;
    const auto sigcliplow = sigclip * sigfrac;

    using Eigen::Index;
    using data_t = typename DerivedA::PlainObject;

    data_t cleaned_data = data;
    const auto n = cleaned_data.size();

    Eigen::VectorXb cosmics(n); // stores result
    cosmics.setZero();
    Index ncosmics{0};

    // allocate data buffer
    data_t laplacian_data(n);
    const auto n_sampled = n * block_size;
    data_t sampled_data(n_sampled);
    data_t convolved_data(n_sampled);

    constexpr auto laplace_size = 3;

    auto border_patches = [](const auto &data, Index window, auto &&patchfunc) {
        const auto n = data.size();
        const auto i0 = (window - 1) / 2;
        using Scalar = typename DECAY(data)::Scalar;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> patches(
            window, window - 1);
        Eigen::Matrix<Index, Eigen::Dynamic, 1> ipatches(window - 1);
        for (Index j = 0; j < patches.cols(); ++j) {
            const auto i = j < i0 ? j : n - window + 1 + j;
            ipatches.coeffRef(j) = i;
            patches.col(j).array() = FWD(patchfunc)(data, i, i0, window);
        }
        return std::make_pair(patches, ipatches);
    };
    auto border_patches_nearest = [&border_patches](auto &&... args) {
        return border_patches(
            FWD(args)..., [](const auto &data, auto i, auto i0, auto &&...) {
                return i < i0 ? data.coeff(0) : data.coeff(data.size() - 1);
            });
    };
    auto border_patches_mirror = [&border_patches](auto &&... args) {
        return border_patches(
            FWD(args)..., [](const auto &data, auto i, auto i0, auto window) {
                using Scalar = typename DECAY(data)::Scalar;
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> patch(window);
                const auto n = data.size();
                if (i < i0) {
                    patch.head(i0).reverse() = data.segment(1, i0);
                    patch.tail(window - i0) = data.head(window - i0);
                } else {
                    patch.head(i0) = data.tail(i0);
                    patch.tail(window - i0).reverse() =
                        data.segment(n - window - i0 - 1, window - i0);
                }
                return patch;
            });
    };
    auto windowed_apply = [&](const auto &data, auto &&func, Index size,
                              auto &output, auto &&borderfunc) {
        auto valid_segment = [](auto &data, Index window) {
            const auto size = data.size();
            assert(size >= window);
            const auto i0 = (window - 1) / 2;
            return data.segment(i0, size - window + 1);
        };
        auto valid_indices = container_utils::index(valid_segment(data, size));
        assert(output.size() == data.size());
        grppi::map(ex, valid_indices, valid_indices, [&](auto i) {
            FWD(func)
            (data.segment(i, size), valid_segment(output, size).coeffRef(i));
            return i;
        });
        auto patches = borderfunc(data, size);
        auto border_indices = container_utils::index(patches.first.cols());
        grppi::map(ex, border_indices, border_indices, [&](auto i) {
            FWD(func)
            (patches.first.col(i), output.coeffRef(patches.second.coeff(i)));
            return i;
        });
    };
    auto medfilt = [&windowed_apply, &border_patches_mirror](const auto &data,
                                                             Index size) {
        typename DECAY(data)::PlainObject output(data.size());
        windowed_apply(
            data,
            [](const auto &patch, auto &output) {
                output = alg::median(patch);
            },
            size, output, border_patches_mirror);
        return output;
    };
    auto dilate = [&windowed_apply, &border_patches_nearest](const auto &data,
                                                             Index size) {
        typename DECAY(data)::PlainObject output(data.size());
        windowed_apply(
            data, [](const auto &patch, auto &output) { output = patch.sum(); },
            size, output, border_patches_nearest);
        return output;
    };
    auto indices = container_utils::index(n);
    auto sampled_indices = container_utils::index(n_sampled);
    for (Index it = 0; it < maxiter; ++it) {
        logging::scoped_timeit _0("lacosmic1d iter");
        grppi::map(ex, indices, indices, [&](auto i) {
            sampled_data.segment(i * block_size, block_size)
                .setConstant(cleaned_data.coeff(i) / block_size);
            return i;
        });
        windowed_apply(
            sampled_data,
            [](const auto &patch, auto &output) {
                // [-1, 2, -1]
                output = patch.coeff(1) * 2. - patch.coeff(0) - patch.coeff(1);
            },
            laplace_size, convolved_data, border_patches_mirror);
        grppi::map(ex, indices, indices, [&](auto i) {
            laplacian_data.coeffRef(i) =
                convolved_data.segment(i * block_size, block_size).sum();
            return i;
        });
        auto snr =
            laplacian_data
                .cwiseQuotient(double(block_size) * uncertainty.derived())
                .array()
                .eval();
        // remove large structure
        snr -= medfilt(snr, 5);
        // fine structure
        auto m3 = medfilt(cleaned_data, 3);
        auto fine = (m3 - medfilt(m3, 7))
                        .cwiseQuotient(uncertainty.derived())
                        .array()
                        .eval();
        fine = (fine < 0.01).select(0.01, fine);
        // set cosmics
        auto cr = ((snr > sigclip) && (snr / fine > objlim) &&
                   (!mask.derived().array()))
                      .eval();
        // special treatment for neighbors, with more relaxed constraints
        // grow these cosmics a first time to determine the immediate
        // neighborhood
        // keep those that have sp > sigmalim
        cr = dilate(cr, 3) && (snr > sigclip) && (!mask.derived().array());
        // repeat, but lower the detection limit to siglow
        cr = dilate(cr, 3) && (snr > sigcliplow) && (!mask.derived().array());

        auto ncrs = cr.template cast<int>().sum();
        ncosmics += ncrs;
        SPDLOG_TRACE("iter={} ncrs={} ncosmics={}", it, ncrs, ncosmics);
        if (ncosmics == 0) {
            SPDLOG_TRACE("stop iter");
            break;
        }
        // update result
        cosmics = cosmics.array() || cr.array();
        {
            // replace cr with good data
            const auto rsize = 5;
            const auto j0 = (rsize - 1) / 2;
            for (Index j = 0; j < n; ++j) {
                if (!cr.coeff(j)) {
                    continue;
                }
                auto k0 = j - j0;
                auto k1 = k0 + rsize;
                if (k0 < 0) {
                    k0 = 0;
                }
                if (k1 > n) {
                    k1 = n;
                }
                assert(k1 - k0 > 0);
                Index n_good = 0;
                double v = 0.;
                // sum up good pixs
                for (Index k = k0; k < k1; ++k) {
                    if (!(cr.coeff(k) || mask.coeff(k))) {
                        ++n_good;
                        v += cleaned_data.coeff(k);
                    }
                }
                if (n_good == 0) {
                    v = fill_value;
                } else {
                    v /= n_good;
                }
                SPDLOG_TRACE("clean at pos j={} k0={} k1={} n_good={} v={}", j,
                             k0, k1, n_good, v);
            }
            if (it < maxiter - 1) {
                SPDLOG_TRACE("continue iter");
            } else {
                SPDLOG_DEBUG("stop at maximum iter={}", maxiter);
            }
        }
        SPDLOG_TRACE("found ncosmics={}", ncosmics);
        SPDLOG_TRACE("cosmics{}", cosmics);
    }
    return std::make_pair(std::move(cleaned_data), std::move(cosmics));
}

} // namespace alg

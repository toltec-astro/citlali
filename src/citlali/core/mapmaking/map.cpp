#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <Eigen/Sparse>
#include <spdlog/spdlog.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/toltec_io.h>

namespace mapmaking {

namespace {

[[noreturn]] void throw_psd_support_error(const mapmaking::MapBuffer &mb, Eigen::Index map_index,
                                          double weight_threshold, const Eigen::MatrixXd &cov_ranges,
                                          Eigen::Index cov_n_rows, Eigen::Index cov_n_cols,
                                          const char *reason) {
    auto logger = spdlog::get("citlali_logger");

    const auto &weight = mb.weight[map_index];
    const Eigen::Index n_positive_weight = (weight.array() > 0.0).count();
    const Eigen::Index n_finite_weight = weight.array().isFinite().count();
    const Eigen::Index n_above_threshold =
        (weight.array().isFinite() && (weight.array() >= weight_threshold)).count();
    const double weight_min = weight.size() > 0 ? weight.minCoeff() : 0.0;
    const double weight_max = weight.size() > 0 ? weight.maxCoeff() : 0.0;

    std::ostringstream os;
    os << "cannot calculate map PSD: " << reason
       << " support for map_index=" << map_index
       << " map_buffer=" << mb.name
       << " map_grouping=" << mb.map_grouping
       << " obsnums=";
    if (mb.obsnums.empty()) {
        os << "<none>";
    }
    else {
        for (size_t i = 0; i < mb.obsnums.size(); ++i) {
            if (i != 0) {
                os << ",";
            }
            os << mb.obsnums[i];
        }
    }
    os << " weight_threshold=" << weight_threshold
       << " cov_ranges=[(" << cov_ranges(0,0) << "," << cov_ranges(0,1) << "),("
       << cov_ranges(1,0) << "," << cov_ranges(1,1) << ")]"
       << " support_rows=" << cov_n_rows
       << " support_cols=" << cov_n_cols
       << " map_rows=" << mb.n_rows
       << " map_cols=" << mb.n_cols
       << " positive_weight_pixels=" << n_positive_weight
       << " finite_weight_pixels=" << n_finite_weight
       << " above_threshold_pixels=" << n_above_threshold
       << " weight_min=" << weight_min
       << " weight_max=" << weight_max
       << " coverage_cut=" << mb.cov_cut;

    if (logger) {
        logger->error("{}", os.str());
    }

    throw std::runtime_error(os.str());
}

} // namespace

// constructor
MapBuffer::MapBuffer() {}

// constructor
MapBuffer::MapBuffer(std::string _n): name(_n) {}

void MapBuffer::normalize_maps(const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps) {
    // vectors for maps
    const bool use_grid_weight = grid_weight.size() == signal.size();
    const bool realize_science_products =
        science_products.initialized && !science_products.normalization_support.empty();
    if (realize_science_products) {
        const auto &products = science_products;
        if (!products.identity_admitted || !products.bundle_identity ||
            products.bundle_identity->rows != n_rows ||
            products.bundle_identity->cols != n_cols ||
            products.bundle_identity->ordered_slots.size() != signal.size()) {
            throw std::runtime_error(
                "science-map normalization requires an admitted complete bundle identity");
        }
        std::vector<std::string> expected_companions;
        if (!kernel.empty()) {
            if (kernel.size() != signal.size()) {
                throw std::runtime_error(
                    "science-map normalization kernel inventory mismatch");
            }
            expected_companions.push_back("kernel_I");
        }
        if (!noise.empty()) {
            if (noise.size() != signal.size() || n_noise < 0) {
                throw std::runtime_error(
                    "science-map normalization realization inventory mismatch");
            }
            for (Eigen::Index realization = 0; realization < n_noise;
                 ++realization) {
                expected_companions.push_back(
                    "noise_realization_" + std::to_string(realization) +
                    "_I");
            }
        }
        if (products.bundle_identity->required_companions !=
            expected_companions) {
            throw std::runtime_error(
                "science-map normalization declared-companion inventory mismatch");
        }
    }
    normalize_support_diag.assign(signal.size(), NormalizeSupportDiag{});
    if (active_maps == nullptr) {
        map_in_vec.resize(signal.size());
        std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    }
    else {
        map_in_vec.clear();
        map_in_vec.reserve(signal.size());
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(signal.size()); ++i) {
            if (i < active_maps->size() && (*active_maps)(i)) {
                map_in_vec.push_back(static_cast<int>(i));
            }
        }
    }
    map_out_vec.resize(map_in_vec.size());

    if (map_in_vec.empty()) {
        if (use_grid_weight) {
            std::vector<Eigen::MatrixXd>().swap(grid_weight);
        }
        return;
    }

    // normalize science and kernel mpas
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
        Eigen::ArrayXXd mask(n_rows, n_cols);
        Eigen::ArrayXXd science_policy_mask(n_rows, n_cols);
        Eigen::MatrixXd finalized_weight = Eigen::MatrixXd::Zero(n_rows, n_cols);
        engine_utils::WeightThresholdSelection normalization_selection;
        engine_utils::WeightThresholdSelection science_policy_selection;

        auto apply_normalization_threshold =
            [&](const Eigen::MatrixXd &weight_map) {
            // Preserve the historical pre-normalization coverage_cut/10 role
            // while making its predicate finite-positive even at zero cut.
            normalization_selection =
                engine_utils::find_weight_threshold_selection(
                    weight_map, cov_cut / 10.0);
            const auto finite_positive =
                weight_map.array().isFinite() && (weight_map.array() > 0.0);
            mask = (finite_positive &&
                    (weight_map.array() >= normalization_selection.threshold))
                       .template cast<double>();
            return normalization_selection.threshold;
        };

        auto classify_masked_cause = [](bool has_accum_weight,
                                        bool has_valid_grid_weight,
                                        bool use_grid_weight,
                                        bool was_pre_threshold_supported) {
            if (!has_accum_weight) {
                return 1;
            }
            if (use_grid_weight && !has_valid_grid_weight) {
                return 2;
            }
            if (was_pre_threshold_supported) {
                return 3;
            }
            return 0;
        };

        auto record_support_diag = [&](const Eigen::MatrixXd &raw_signal,
                                       const Eigen::ArrayXXd &accum_weight_valid,
                                       const Eigen::ArrayXXd &grid_weight_valid,
                                       const Eigen::ArrayXXd &pre_threshold_support,
                                       double support_weight_threshold,
                                       bool diag_use_grid_weight) {
            NormalizeSupportDiag diag;
            diag.map_index = i;
            diag.n_total = n_rows * n_cols;
            diag.use_grid_weight = diag_use_grid_weight;
            diag.support_weight_threshold = support_weight_threshold;
            diag.n_retained = (mask > 0.0).count();
            diag.n_masked = (mask <= 0.0).count();
            diag.n_masked_no_accum_weight =
                ((mask <= 0.0) && (accum_weight_valid <= 0.0)).count();
            diag.n_masked_bad_grid_weight_with_accum_weight =
                ((mask <= 0.0) && (accum_weight_valid > 0.0) &&
                 (grid_weight_valid <= 0.0)).count();
            diag.n_masked_by_support_threshold =
                ((mask <= 0.0) && (pre_threshold_support > 0.0)).count();
            diag.n_masked_raw_signal_nonzero =
                ((mask <= 0.0) && raw_signal.array().isFinite() &&
                 (raw_signal.array().abs() > 0.0)).count();

            const bool inspect_neighbors =
                diag.n_masked > 0 && diag.n_retained > 0 &&
                (diag.n_masked_bad_grid_weight_with_accum_weight > 0 ||
                 diag.n_masked_by_support_threshold > 0 ||
                 diag.n_masked_raw_signal_nonzero > 0);

            if (inspect_neighbors) {
                double max_abs_raw_signal = 0.0;
                double max_neighbor_weight = -1.0;
                for (Eigen::Index row = 0; row < n_rows; ++row) {
                    for (Eigen::Index col = 0; col < n_cols; ++col) {
                        if (mask(row, col) > 0.0) {
                            continue;
                        }
                        if (std::isfinite(raw_signal(row, col))) {
                            max_abs_raw_signal =
                                std::max(max_abs_raw_signal, std::abs(raw_signal(row, col)));
                        }

                        double neighbor_weight = -1.0;
                        if (row > 0 && mask(row - 1, col) > 0.0) {
                            neighbor_weight =
                                std::max(neighbor_weight, finalized_weight(row - 1, col));
                        }
                        if ((row + 1) < n_rows && mask(row + 1, col) > 0.0) {
                            neighbor_weight =
                                std::max(neighbor_weight, finalized_weight(row + 1, col));
                        }
                        if (col > 0 && mask(row, col - 1) > 0.0) {
                            neighbor_weight =
                                std::max(neighbor_weight, finalized_weight(row, col - 1));
                        }
                        if ((col + 1) < n_cols && mask(row, col + 1) > 0.0) {
                            neighbor_weight =
                                std::max(neighbor_weight, finalized_weight(row, col + 1));
                        }
                        if (neighbor_weight > 0.0) {
                            diag.n_masked_adjacent_support++;
                            if (neighbor_weight > max_neighbor_weight) {
                                max_neighbor_weight = neighbor_weight;
                                diag.max_neighbor_row = row;
                                diag.max_neighbor_col = col;
                                diag.max_neighbor_cause =
                                    classify_masked_cause(accum_weight_valid(row, col) > 0.0,
                                                          grid_weight_valid(row, col) > 0.0,
                                                          diag_use_grid_weight,
                                                          pre_threshold_support(row, col) > 0.0);
                            }
                        }
                    }
                }
                diag.max_masked_abs_raw_signal = max_abs_raw_signal;
                if (max_neighbor_weight >= 0.0) {
                    diag.max_masked_neighbor_weight = max_neighbor_weight;
                }
            }

            normalize_support_diag[i] = diag;
        };

        if (use_grid_weight) {
            const auto &denom = grid_weight[i];
            const auto denom_valid = denom.array().isFinite() && (denom.array().abs() > 1e-8);
            const auto accum_weight_valid = weight[i].array().isFinite() && (weight[i].array() > 0.0);
            const auto valid_support = denom_valid && accum_weight_valid;
            const Eigen::ArrayXXd safe_denom = denom_valid.select(denom.array(), 1.0);
            const Eigen::MatrixXd raw_signal = signal[i];

            finalized_weight =
                ((denom.array().square() / weight[i].array().max(1e-30)) *
                 valid_support.template cast<double>()).matrix();
            const Eigen::ArrayXXd pre_threshold_support =
                (finalized_weight.array() > 0.0).template cast<double>();
            const double support_weight_threshold =
                apply_normalization_threshold(finalized_weight);
            record_support_diag(raw_signal,
                                accum_weight_valid.template cast<double>(),
                                denom_valid.template cast<double>(),
                                pre_threshold_support,
                                support_weight_threshold,
                                true);

            signal[i] = (mask > 0.0)
                .select(signal[i].array() / safe_denom, 0.0).matrix();

            if (!kernel.empty()) {
                kernel[i] = (mask > 0.0)
                    .select(kernel[i].array() / safe_denom, 0.0).matrix();
            }

            if (!noise.empty()) {
                for (Eigen::Index n = 0; n < n_noise; ++n) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
                        noise[i].data() + n * n_rows * n_cols, n_rows, n_cols);
                    noise_matrix = (mask > 0.0)
                        .select(noise_matrix.array() / safe_denom, 0.0).matrix();
                }
            }

            weight[i] = (mask > 0.0)
                .select(finalized_weight.array(), 0.0).matrix();
        }
        else {
            const Eigen::MatrixXd raw_signal = signal[i];
            finalized_weight =
                (weight[i].array().isFinite() && (weight[i].array() > 0.0))
                    .select(weight[i].array(), 0.0).matrix();
            const Eigen::ArrayXXd pre_threshold_support =
                (finalized_weight.array() > 0.0).template cast<double>();
            const double support_weight_threshold =
                apply_normalization_threshold(finalized_weight);
            record_support_diag(raw_signal,
                                pre_threshold_support,
                                Eigen::ArrayXXd::Ones(n_rows, n_cols),
                                pre_threshold_support,
                                support_weight_threshold,
                                false);
            const Eigen::ArrayXXd safe_weight = (finalized_weight.array() > 0.0)
                                                    .select(finalized_weight.array(), 1.0);

            signal[i] = (mask > 0.0)
                .select(signal[i].array() / safe_weight, 0.0).matrix();

            if (!kernel.empty()) {
                kernel[i] = (mask > 0.0)
                    .select(kernel[i].array() / safe_weight, 0.0).matrix();
            }

            if (!noise.empty()) {
                for (Eigen::Index n = 0; n < n_noise; ++n) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
                        noise[i].data() + n * n_rows * n_cols, n_rows, n_cols);
                    noise_matrix = (mask > 0.0)
                        .select(noise_matrix.array() / safe_weight, 0.0).matrix();
                }
            }

            weight[i] = (mask > 0.0)
                .select(finalized_weight.array(), 0.0).matrix();
        }

        // The science-policy role historically consumes the finalized
        // post-normalization coefficient plane. A later positive global
        // empirical rescale refreshes this threshold and its exact provenance
        // without renormalizing any map plane.
        science_policy_selection =
            engine_utils::find_weight_threshold_selection(weight[i], cov_cut);
        const auto final_finite_positive =
            weight[i].array().isFinite() && (weight[i].array() > 0.0);
        science_policy_mask =
            (final_finite_positive &&
             (weight[i].array() >= science_policy_selection.threshold))
                .template cast<double>();

        if (realize_science_products) {
            auto &products = science_products;
            auto &record = products.realized.at(static_cast<std::size_t>(i));
            products.normalization_support.at(static_cast<std::size_t>(i)) =
                (mask > 0.0).template cast<std::uint8_t>().matrix();
            products.science_policy_support.at(static_cast<std::size_t>(i)) =
                (science_policy_mask > 0.0)
                    .template cast<std::uint8_t>().matrix();

            auto &retained =
                products.retained_exposure.at(static_cast<std::size_t>(i));
            retained = (mask > 0.0).select(retained.array(), 0.0).matrix();
            if (!coverage.empty()) {
                coverage.at(static_cast<std::size_t>(i)) = retained;
            }

            Eigen::ArrayXX<bool> companions_finite =
                signal.at(static_cast<std::size_t>(i)).array().isFinite() &&
                weight.at(static_cast<std::size_t>(i)).array().isFinite() &&
                (weight.at(static_cast<std::size_t>(i)).array() > 0.0);
            if (!kernel.empty()) {
                companions_finite =
                    companions_finite &&
                    kernel.at(static_cast<std::size_t>(i)).array().isFinite();
            }
            if (!noise.empty()) {
                for (Eigen::Index realization = 0; realization < n_noise;
                     ++realization) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                                             Eigen::Dynamic>> noise_matrix(
                        noise.at(static_cast<std::size_t>(i)).data() +
                            realization * n_rows * n_cols,
                        n_rows, n_cols);
                    companions_finite =
                        companions_finite && noise_matrix.array().isFinite();
                }
            }
            if (((mask > 0.0) && !companions_finite).any()) {
                throw std::runtime_error(
                    "science-map normalization produced a non-finite declared companion on numerical support");
            }
            const bool identity_available =
                products.identity_admitted && products.bundle_identity.has_value();
            if (identity_available) {
                products.science_valid.at(static_cast<std::size_t>(i)) =
                    ((mask > 0.0) && (science_policy_mask > 0.0) &&
                     companions_finite)
                        .template cast<std::uint8_t>().matrix();
            }
            else {
                products.science_valid.at(static_cast<std::size_t>(i)).setZero();
            }

            auto populate_threshold = [&](ScienceMapThresholdRealization &target,
                                          const auto &selection,
                                          const char *algorithm,
                                          const char *coefficient_stage,
                                          double cut) {
                target.support_algorithm = algorithm;
                target.coefficient_stage = coefficient_stage;
                target.requested_cut = cut;
                target.realized_cut = cut;
                target.realized_threshold = selection.threshold;
                target.selected_positive_value =
                    selection.selected_positive_value;
                target.positive_value_count = selection.positive_value_count;
                target.selected_zero_based_index =
                    selection.selected_zero_based_index;
                target.selected_index_available =
                    selection.selected_index_available;
            };
            populate_threshold(
                record.normalization, normalization_selection,
                science_map_normalization_support_version,
                products.is_coadd
                    ? science_map_coadd_normalization_coefficient_stage
                    : science_map_observation_normalization_coefficient_stage,
                cov_cut / 10.0);
            populate_threshold(
                record.science_policy, science_policy_selection,
                science_map_policy_support_version,
                products.coefficient_stage.c_str(), cov_cut);

            science_map_finalize_realized_product_facts(
                *this, static_cast<std::size_t>(i));
        }
        else if (!coverage.empty()) {
            coverage[i] = (mask > 0.0)
                .select(coverage[i].array(), 0.0).matrix();
        }
        return 0;
    });

    if (use_grid_weight) {
        std::vector<Eigen::MatrixXd>().swap(grid_weight);
    }
}

void MapBuffer::freeze_raw_science_parent() {
    if (raw_science_parent) {
        return;
    }
    if (!science_products.initialized) {
        return;
    }
    const bool any_product_available = std::any_of(
        science_products.realized.begin(), science_products.realized.end(),
        [](const auto &record) {
            return std::any_of(record.product_available.begin(),
                               record.product_available.end(),
                               [](bool available) { return available; });
        });
    if (!any_product_available) {
        return;
    }
    if (!science_products.identity_admitted ||
        !science_products.bundle_identity ||
        science_products.bundle_identity->rows != n_rows ||
        science_products.bundle_identity->cols != n_cols ||
        science_products.bundle_identity->ordered_slots.size() !=
            signal.size() ||
        science_products.realized.size() != signal.size()) {
        throw std::runtime_error(
            "cannot freeze an incomplete raw science-map parent identity");
    }
    for (std::size_t slot = 0; slot < signal.size(); ++slot) {
        bool coverage_alias_equal =
            slot < coverage.size() &&
            slot < science_products.retained_exposure.size() &&
            coverage[slot].rows() ==
                science_products.retained_exposure[slot].rows() &&
            coverage[slot].cols() ==
                science_products.retained_exposure[slot].cols();
        if (coverage_alias_equal) {
            for (Eigen::Index index = 0; index < coverage[slot].size();
                 ++index) {
                if (!science_map_exact_double_equal(
                        coverage[slot].data()[index],
                        science_products.retained_exposure[slot]
                            .data()[index])) {
                    coverage_alias_equal = false;
                    break;
                }
            }
        }
        if (!science_map_realized_product_facts_match(*this, slot) ||
            science_products.realized[slot].raw_parent_digest !=
                science_map_raw_parent_digest(*this, slot) ||
            !coverage_alias_equal) {
            throw std::runtime_error(
                "cannot freeze a stale raw science-map product bundle");
        }
    }
    raw_science_parent =
        std::make_shared<const ScienceMapProducts>(science_products);
}

void MapBuffer::refresh_science_products_after_coefficient_rescale(
    Eigen::Index map_index) {
    if (raw_science_parent || !science_products.initialized ||
        map_index < 0 ||
        map_index >= static_cast<Eigen::Index>(signal.size()) ||
        map_index >= static_cast<Eigen::Index>(weight.size()) ||
        map_index >= static_cast<Eigen::Index>(science_products.realized.size())) {
        return;
    }
    const auto slot = static_cast<std::size_t>(map_index);
    auto &products = science_products;
    auto &record = products.realized[slot];
    if (!record.initialized ||
        slot >= products.normalization_support.size() ||
        slot >= products.science_policy_support.size() ||
        slot >= products.science_valid.size()) {
        return;
    }

    const auto selection = engine_utils::find_weight_threshold_selection(
        weight[slot], cov_cut);
    const auto finite_positive =
        weight[slot].array().isFinite() && (weight[slot].array() > 0.0);
    products.science_policy_support[slot] =
        (finite_positive &&
         (weight[slot].array() >= selection.threshold))
            .template cast<std::uint8_t>().matrix();

    Eigen::ArrayXX<bool> companions_finite =
        signal[slot].array().isFinite() && finite_positive;
    if (!kernel.empty()) {
        companions_finite = companions_finite && kernel.at(slot).array().isFinite();
    }
    if (!noise.empty()) {
        for (Eigen::Index realization = 0; realization < n_noise;
             ++realization) {
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                           Eigen::Dynamic>> noise_matrix(
                noise.at(slot).data() + realization * n_rows * n_cols,
                n_rows, n_cols);
            companions_finite =
                companions_finite && noise_matrix.array().isFinite();
        }
    }
    if (products.identity_admitted && products.bundle_identity) {
        products.science_valid[slot] =
            ((products.normalization_support[slot].array() != 0) &&
             (products.science_policy_support[slot].array() != 0) &&
             companions_finite)
                .template cast<std::uint8_t>().matrix();
    }
    else {
        products.science_valid[slot].setZero();
    }

    auto &threshold = record.science_policy;
    threshold.order_statistic_algorithm = science_map_order_statistic_version;
    threshold.support_algorithm = science_map_policy_support_version;
    threshold.coefficient_product = "weight_I";
    threshold.coefficient_stage = products.coefficient_stage;
    threshold.requested_cut = cov_cut;
    threshold.realized_cut = cov_cut;
    threshold.realized_threshold = selection.threshold;
    threshold.selected_positive_value = selection.selected_positive_value;
    threshold.positive_value_count = selection.positive_value_count;
    threshold.selected_zero_based_index = selection.selected_zero_based_index;
    threshold.selected_index_available = selection.selected_index_available;
    threshold.finite_convention = "coefficient must be finite";
    threshold.positivity_convention = "coefficient > 0";
    threshold.comparison_convention = ">=";

    science_map_finalize_realized_product_facts(*this, slot);
}

void MapBuffer::ensure_contribution_diag(Eigen::Index n_maps) {
    if (!contribution_diag_enabled) {
        return;
    }
    if (n_maps <= 0 || n_rows <= 0 || n_cols <= 0) {
        return;
    }
    const auto target = static_cast<std::size_t>(n_maps);
    if (contribution_max_abs.size() == target &&
        !contribution_max_abs.empty() &&
        contribution_max_abs.front().rows() == n_rows &&
        contribution_max_abs.front().cols() == n_cols) {
        return;
    }

    contribution_max_abs.assign(
        target, Eigen::MatrixXd::Constant(
                    n_rows, n_cols, -std::numeric_limits<double>::infinity()));
    contribution_signal.assign(
        target, Eigen::MatrixXd::Constant(
                    n_rows, n_cols, std::numeric_limits<double>::quiet_NaN()));
    contribution_weight.assign(
        target, Eigen::MatrixXd::Constant(
            n_rows, n_cols, std::numeric_limits<double>::quiet_NaN()));
    contribution_variance_weight.assign(
        target, Eigen::MatrixXd::Constant(
                    n_rows, n_cols, std::numeric_limits<double>::quiet_NaN()));
    contribution_total_signal.assign(
        target, Eigen::MatrixXd::Zero(n_rows, n_cols));
    contribution_total_weight.assign(
        target, Eigen::MatrixXd::Zero(n_rows, n_cols));
    contribution_total_variance_weight.assign(
        target, Eigen::MatrixXd::Zero(n_rows, n_cols));
    contribution_uid.assign(
        target, Eigen::MatrixXi::Constant(n_rows, n_cols, -2147483647));
    contribution_scan.assign(
        target, Eigen::MatrixXi::Constant(n_rows, n_cols, -2147483647));
    contribution_sample.assign(
        target, Eigen::MatrixXi::Constant(n_rows, n_cols, -2147483647));
}

void MapBuffer::clear_contribution_diag() {
    contribution_max_abs.clear();
    contribution_signal.clear();
    contribution_weight.clear();
    contribution_variance_weight.clear();
    contribution_total_signal.clear();
    contribution_total_weight.clear();
    contribution_total_variance_weight.clear();
    contribution_uid.clear();
    contribution_scan.clear();
    contribution_sample.clear();
    clear_contribution_targets();
}

void MapBuffer::set_contribution_targets(
    Eigen::Index n_maps,
    const std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> &targets) {
    clear_contribution_targets();
    if (n_maps <= 0 || targets.empty()) {
        return;
    }
    contribution_targets.assign(static_cast<std::size_t>(n_maps), {});
    for (const auto &[map_index, row, col] : targets) {
        if (map_index < 0 || map_index >= n_maps || row < 0 || col < 0 ||
            row >= n_rows || col >= n_cols) {
            continue;
        }
        auto &map_targets = contribution_targets[static_cast<std::size_t>(map_index)];
        const auto pixel = std::make_pair(row, col);
        if (std::find(map_targets.begin(), map_targets.end(), pixel) == map_targets.end()) {
            map_targets.push_back(pixel);
        }
    }
    contribution_diag_targeted = false;
    for (const auto &map_targets : contribution_targets) {
        if (!map_targets.empty()) {
            contribution_diag_targeted = true;
            break;
        }
    }
}

void MapBuffer::clear_contribution_targets() {
    contribution_targets.clear();
    contribution_diag_targeted = false;
}

bool MapBuffer::contribution_target_enabled(Eigen::Index map_index,
                                            Eigen::Index row,
                                            Eigen::Index col) const {
    if (!contribution_diag_targeted) {
        return true;
    }
    if (map_index < 0 || map_index >= static_cast<Eigen::Index>(contribution_targets.size())) {
        return false;
    }
    const auto &map_targets = contribution_targets[static_cast<std::size_t>(map_index)];
    return std::find(map_targets.begin(), map_targets.end(),
                     std::make_pair(row, col)) != map_targets.end();
}

void MapBuffer::record_contribution(Eigen::Index map_index, Eigen::Index row,
                                    Eigen::Index col, double signal_contribution,
                                    double weight_contribution, int uid,
                                    int scan, int sample) {
    record_contribution(map_index, row, col, signal_contribution,
                        weight_contribution, weight_contribution, uid, scan, sample);
}

void MapBuffer::record_contribution(Eigen::Index map_index, Eigen::Index row,
                                    Eigen::Index col, double signal_contribution,
                                    double weight_contribution,
                                    double variance_weight_contribution,
                                    int uid, int scan, int sample) {
    if (!contribution_diag_enabled) {
        return;
    }
    if (map_index < 0 || row < 0 || col < 0 ||
        map_index >= static_cast<Eigen::Index>(contribution_max_abs.size()) ||
        row >= n_rows || col >= n_cols ||
        !std::isfinite(signal_contribution)) {
        return;
    }
    if (!contribution_target_enabled(map_index, row, col)) {
        return;
    }
    const double safe_weight =
        std::isfinite(weight_contribution) ? weight_contribution : 0.0;
    const double safe_variance_weight =
        std::isfinite(variance_weight_contribution)
            ? variance_weight_contribution
            : safe_weight;
    contribution_total_signal[map_index](row, col) += signal_contribution;
    contribution_total_weight[map_index](row, col) += safe_weight;
    contribution_total_variance_weight[map_index](row, col) += safe_variance_weight;
    const double abs_contribution = std::abs(signal_contribution);
    if (!std::isfinite(abs_contribution) ||
        abs_contribution <= contribution_max_abs[map_index](row, col)) {
        return;
    }
    contribution_max_abs[map_index](row, col) = abs_contribution;
    contribution_signal[map_index](row, col) = signal_contribution;
    contribution_weight[map_index](row, col) = weight_contribution;
    contribution_variance_weight[map_index](row, col) = safe_variance_weight;
    contribution_uid[map_index](row, col) = uid;
    contribution_scan[map_index](row, col) = scan;
    contribution_sample[map_index](row, col) = sample;
}

void MapBuffer::calculate_stokes(std::vector<Eigen::MatrixXd>& map_vec, const Eigen::MatrixXd& m, Eigen::Index i, Eigen::Index j,
                                 int index, int step) {
    Eigen::VectorXd d(3);
    d(0) = map_vec[index](i, j);
    d(1) = map_vec[index + step](i, j);
    d(2) = map_vec[index + 2 * step](i, j);
    Eigen::VectorXd v = m.ldlt().solve(d);//m.colPivHouseholderQr().solve(d);
    map_vec[index](i, j) = v(0);
    map_vec[index + step](i, j) = v(1);
    map_vec[index + 2 * step](i, j) = v(2);
}

void MapBuffer::calculate_stokes(std::vector<Eigen::Tensor<double,3>>& map_vec, const Eigen::MatrixXd& m, Eigen::Index i,
                                 Eigen::Index j, int index, int step) {
    Eigen::VectorXd d(3);
    for (Eigen::Index n = 0; n < n_noise; ++n) {
        d(0) = map_vec[index](i, j, n);
        d(1) = map_vec[index + step](i, j, n);
        d(2) = map_vec[index + 2 * step](i, j, n);
        Eigen::VectorXd v = m.colPivHouseholderQr().solve(d);
        map_vec[index](i, j, n) = v(0);
        map_vec[index + step](i, j, n) = v(1);
        map_vec[index + 2 * step](i, j, n) = v(2);
    }
}

void MapBuffer::process_maps_for_pixel(Eigen::Index i, Eigen::Index j, int a, int step, const Eigen::MatrixXd& m) {
    calculate_stokes(signal, m, i, j, a, step);
    if (!kernel.empty()) {
        calculate_stokes(kernel, m, i, j, a, step);
    }
    if (!noise.empty()) {
        calculate_stokes(noise, m, i, j, a, step);
    }
}

void MapBuffer::zero_out_maps(Eigen::Index i, Eigen::Index j, int index, int step) {
    signal[index](i, j) = 0;
    signal[index + step](i, j) = 0;
    signal[index + 2 * step](i, j) = 0;
    if (!kernel.empty()) {
        kernel[index](i, j) = 0;
        kernel[index + step](i, j) = 0;
        kernel[index + 2 * step](i, j) = 0;
    }
    if (!noise.empty()) {
        for (Eigen::Index n = 0; n < n_noise; ++n) {
            noise[index](i, j, n) = 0;
            noise[index + step](i, j, n) = 0;
            noise[index + 2 * step](i, j, n) = 0;
        }
    }
}

void MapBuffer::normalize_polarized_maps() {
    int step = pointing.size();
    for (Eigen::Index index = 0; index < pointing.size(); ++index) {
        Eigen::MatrixXd m(3, 3);
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            for (Eigen::Index j = 0; j < n_cols; ++j) {
                int pix = n_rows * j + i;
                Eigen::VectorXd temp = pointing[index].row(pix);
                m = Eigen::Map<Eigen::MatrixXd>(temp.data(), 3, 3);
                Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(m);
                if ((m.array() != 0).all() && lu_decomp.isInvertible()) {
                    process_maps_for_pixel(i, j, index, step, m);
                }
                else {
                    zero_out_maps(i, j, index, step);
                }
            }
        }
        weight[index + step] = weight[index];
        weight[index + 2 * step] = weight[index];

        coverage[index + step] = coverage[index];
        coverage[index + 2 * step] = coverage[index];
    }

    if (!grid_weight.empty()) {
        std::vector<Eigen::MatrixXd>().swap(grid_weight);
    }
}

std::tuple<double, Eigen::MatrixXd, Eigen::Index, Eigen::Index> MapBuffer::calc_cov_region(Eigen::Index i) {
    // calculate weight threshold
    double weight_threshold = engine_utils::find_weight_threshold(weight[i], cov_cut);

    // calculate coverage ranges
    Eigen::MatrixXd cov_ranges = engine_utils::set_cov_cov_ranges(weight[i], weight_threshold);

    // rows and cols of region above weight threshold
    Eigen::Index cov_n_rows = cov_ranges(1,0) - cov_ranges(0,0) + 1;
    Eigen::Index cov_n_cols = cov_ranges(1,1) - cov_ranges(0,1) + 1;

    return std::tuple<double, Eigen::MatrixXd, Eigen::Index, Eigen::Index>(weight_threshold, cov_ranges,
                                                                           cov_n_rows, cov_n_cols);
}

// loop through maps
void MapBuffer::calc_map_psd() {
    // clear psd vectors
    std::vector<Eigen::VectorXd>().swap(psds);
    std::vector<Eigen::VectorXd>().swap(psd_freqs);
    std::vector<Eigen::MatrixXd>().swap(psd_2ds);
    std::vector<Eigen::MatrixXd>().swap(psd_2d_freqs);

    // clear noise psd vectors
    std::vector<Eigen::VectorXd>().swap(noise_psds);
    std::vector<Eigen::VectorXd>().swap(noise_psd_freqs);
    std::vector<Eigen::MatrixXd>().swap(noise_psd_2ds);
    std::vector<Eigen::MatrixXd>().swap(noise_psd_2d_freqs);

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); ++i) {
        // calculate weight threshold
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(i);

        if (!std::isfinite(weight_threshold) || weight_threshold < 0.0) {
            throw_psd_support_error(*this, i, weight_threshold, cov_ranges, cov_n_rows, cov_n_cols,
                                    "invalid weight threshold");
        }

        if (!cov_ranges.array().isFinite().all()) {
            throw_psd_support_error(*this, i, weight_threshold, cov_ranges, cov_n_rows, cov_n_cols,
                                    "non-finite coverage bounds");
        }

        if (cov_ranges(0,0) < 0 || cov_ranges(0,1) < 0 ||
            cov_ranges(1,0) >= n_rows || cov_ranges(1,1) >= n_cols ||
            cov_ranges(0,0) > cov_ranges(1,0) || cov_ranges(0,1) > cov_ranges(1,1)) {
            throw_psd_support_error(*this, i, weight_threshold, cov_ranges, cov_n_rows, cov_n_cols,
                                    "invalid coverage bounds");
        }

        // ensure even rows
        if (cov_n_rows % 2 == 1) {
            cov_ranges(1,0)--;
            cov_n_rows--;
        }

        // ensure even cols
        if (cov_n_cols % 2 == 1) {
            cov_ranges(1,1)--;
            cov_n_cols--;
        }

        if (cov_n_rows < 2 || cov_n_cols < 2) {
            throw_psd_support_error(*this, i, weight_threshold, cov_ranges, cov_n_rows, cov_n_cols,
                                    "coverage support smaller than 2x2 after PSD trimming");
        }

        // explicit copy signal map within coverage region
        Eigen::MatrixXd sig = signal[i].block(cov_ranges(0,0), cov_ranges(0,1), cov_n_rows, cov_n_cols);

        // calculate psds
        auto [p, pf, p_2d, pf_2d] = engine_utils::calc_2D_psd(sig, rows_tan_vec, cols_tan_vec, smooth_window, parallel_policy);
        // move current map psd values into vectors
        psds.push_back(std::move(p));
        psd_freqs.push_back(std::move(pf));

        psd_2ds.push_back(std::move(p_2d));
        psd_2d_freqs.push_back(std::move(pf_2d));

        // get average noise psd if noise maps are requested
        if (!noise.empty()) {
            for (Eigen::Index j=0; j<n_noise; ++j) {
                // get noise map
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(noise[i].data() + j * n_rows * n_cols,
                                                                                               n_rows, n_cols);
                sig = noise_matrix.block(cov_ranges(0,0), cov_ranges(0,1), cov_n_rows, cov_n_cols);

                // calculate psds
                auto [noise_p, noise_pf, noise_p_2d, noise_pf_2d] = engine_utils::calc_2D_psd(sig, rows_tan_vec, cols_tan_vec,
                                                                                              smooth_window, parallel_policy);

                // just copy if on first noise map
                if (j==0) {
                    noise_psds.push_back(std::move(noise_p));
                    noise_psd_freqs.push_back(std::move(noise_pf));

                    noise_psd_2ds.push_back(std::move(noise_p_2d));
                    noise_psd_2d_freqs.push_back(std::move(noise_pf_2d));
                }

                // otherwise add to existing vector
                else {
                    noise_psds.back() = noise_psds.back() + noise_p;
                    noise_psd_2ds.back() = noise_psd_2ds.back() + noise_p_2d;
                    noise_psd_2d_freqs.back() = noise_psd_2d_freqs.back() + noise_pf_2d;
                }
            }
            noise_psds.back() = noise_psds.back()/n_noise;
            noise_psd_2ds.back() = noise_psd_2ds.back()/n_noise;
            noise_psd_2d_freqs.back() = noise_psd_2d_freqs.back()/n_noise;
        }
    }
}

void MapBuffer::calc_map_hist() {
    // clear vectors
    std::vector<Eigen::VectorXd>().swap(hists);
    std::vector<Eigen::VectorXd>().swap(hist_bins);
    std::vector<Eigen::VectorXd>().swap(noise_hists);
    std::vector<Eigen::VectorXd>().swap(noise_hist_bins);

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); ++i) {
        // calculate weight threshold
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(i);

        // setup input signal data
        Eigen::MatrixXd sig = signal[i].block(cov_ranges(0,0),cov_ranges(0,1), cov_n_rows, cov_n_cols);

        // calculate histogram and bins
        auto [h, h_bins] = engine_utils::calc_hist(sig, hist_n_bins);

        hists.push_back(std::move(h));
        hist_bins.push_back(std::move(h_bins));

        // get average noise psd if noise maps are requested
        if (!noise.empty()) {
            for (Eigen::Index j=0; j<n_noise; ++j) {
                // get noise map
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(noise[i].data() + j * n_rows * n_cols,
                                                                                               n_rows, n_cols);
                sig = noise_matrix.block(cov_ranges(0,0), cov_ranges(0,1), cov_n_rows, cov_n_cols);

                // calculate histogram and bins
                auto [noise_h, noise_h_bins] = engine_utils::calc_hist(sig, hist_n_bins);

                // just copy if on first noise map
                if (j==0) {
                    noise_hists.push_back(std::move(noise_h));
                    noise_hist_bins.push_back(std::move(noise_h_bins));
                }
                // otherwise add to existing vector
                else {
                    noise_hists.back() = noise_hists.back() + noise_h;
                }
            }
            noise_hists.back() = noise_hists.back()/n_noise;
        }
    }
}

void MapBuffer::calc_median_err() {
    // resize mean errors
    median_err.setZero(weight.size());
    for (Eigen::Index i=0; i<weight.size(); ++i) {
        // calculate weight threshold
	if (weight[i].maxCoeff() == weight[i].minCoeff()) {
		median_err(i) = 0;
	}
	else {
        	auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(i);
        	Eigen::MatrixXd mean_sqerr = ((weight[i].array()>=weight_threshold).select(1/weight[i].array(),0));
        	// construct a sparse matrix
        	Eigen::SparseMatrix<double> sparse_err(mean_sqerr.sparseView());
        	// construct a dense map
        	Eigen::Map<Eigen::VectorXd> dense_map(sparse_err.valuePtr(), sparse_err.nonZeros());
        	// construct a dense vector
        	Eigen::VectorXd dense_vector(dense_map);
        	// get mean square error
        	median_err(i) = tula::alg::median(dense_vector);
	}
    }
}

void MapBuffer::calc_median_rms() {
    // average filtered rms vector
    median_rms.setZero(noise.size());

    // loop through arrays/polarizations
    for (Eigen::Index i=0; i<noise.size(); ++i) {
        const double weight_threshold = std::get<0>(calc_cov_region(i));
        const auto valid_mask = (weight[i].array()>=weight_threshold);
        const int counter = valid_mask.count();
        if (counter <= 0) {
            median_rms(i) = 0.0;
            continue;
        }

        // vector of rms of noise maps
        Eigen::VectorXd noise_rms(n_noise);
        for (Eigen::Index j=0; j<n_noise; ++j) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(noise[i].data() + j * n_rows * n_cols,
                                                                                           n_rows, n_cols);
            double rms = (valid_mask.select(noise_matrix.array().square(), 0.0)).sum();

            noise_rms(j) = sqrt(rms/counter);
        }
        // get mean rms
        median_rms(i) = tula::alg::median(noise_rms);
    }
}

void MapBuffer::clear_noise_products() {
    std::vector<Eigen::MatrixXd>().swap(weight_formal);
    std::vector<Eigen::MatrixXd>().swap(noise_mean);
    std::vector<Eigen::MatrixXd>().swap(noise_variance);
    std::vector<Eigen::MatrixXd>().swap(weight_empirical);
    std::vector<Eigen::MatrixXd>().swap(sig2noise_pixel);
    std::vector<Eigen::MatrixXd>().swap(point_source_uncertainty);
    std::vector<Eigen::MatrixXd>().swap(sig2noise_point_source);
    noise_weight_median_ratio.resize(0);
    noise_weight_scale.resize(0);
    noise_s2n_sigma.resize(0);
    noise_valid_pixels.resize(0);
}

void MapBuffer::calc_noise_products(bool apply_empirical_weight_scale, bool mean_subtract) {
    clear_noise_products();

    const Eigen::Index n_maps = static_cast<Eigen::Index>(weight.size());
    if (n_maps <= 0 || noise.empty() || n_noise <= 0) {
        return;
    }

    weight_formal.resize(static_cast<size_t>(n_maps));
    noise_mean.resize(static_cast<size_t>(n_maps));
    noise_variance.resize(static_cast<size_t>(n_maps));
    weight_empirical.resize(static_cast<size_t>(n_maps));
    sig2noise_pixel.resize(static_cast<size_t>(n_maps));
    point_source_uncertainty.resize(static_cast<size_t>(n_maps));
    sig2noise_point_source.resize(static_cast<size_t>(n_maps));
    noise_weight_median_ratio.setZero(n_maps);
    noise_weight_scale.setOnes(n_maps);
    noise_s2n_sigma.setZero(n_maps);
    noise_valid_pixels.setZero(n_maps);

    for (Eigen::Index i=0; i<n_maps; ++i) {
        calc_noise_products(i, apply_empirical_weight_scale, mean_subtract);
    }

    calc_median_err();
    calc_median_rms();
}

void MapBuffer::calc_noise_products(Eigen::Index i, bool apply_empirical_weight_scale, bool mean_subtract) {
    const Eigen::Index n_maps = static_cast<Eigen::Index>(weight.size());
    if (i < 0 || i >= n_maps || i >= static_cast<Eigen::Index>(noise.size()) || n_noise <= 0) {
        return;
    }

    auto ensure_matrix_vec = [&](std::vector<Eigen::MatrixXd> &vec) {
        if (static_cast<Eigen::Index>(vec.size()) != n_maps) {
            vec.resize(static_cast<size_t>(n_maps));
        }
    };
    ensure_matrix_vec(weight_formal);
    ensure_matrix_vec(noise_mean);
    ensure_matrix_vec(noise_variance);
    ensure_matrix_vec(weight_empirical);
    ensure_matrix_vec(sig2noise_pixel);
    ensure_matrix_vec(point_source_uncertainty);
    ensure_matrix_vec(sig2noise_point_source);

    if (noise_weight_median_ratio.size() != n_maps) {
        noise_weight_median_ratio.setZero(n_maps);
    }
    if (noise_weight_scale.size() != n_maps) {
        noise_weight_scale.setOnes(n_maps);
    }
    if (noise_s2n_sigma.size() != n_maps) {
        noise_s2n_sigma.setZero(n_maps);
    }
    if (noise_valid_pixels.size() != n_maps) {
        noise_valid_pixels.setZero(n_maps);
    }

    weight_formal[static_cast<size_t>(i)] = weight[i];
    noise_mean[static_cast<size_t>(i)] = Eigen::MatrixXd::Zero(n_rows, n_cols);
    noise_variance[static_cast<size_t>(i)] = Eigen::MatrixXd::Zero(n_rows, n_cols);

    for (Eigen::Index j=0; j<n_noise; ++j) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
            noise[i].data() + j * n_rows * n_cols, n_rows, n_cols);
        noise_mean[static_cast<size_t>(i)].array() += noise_matrix.array();
        noise_variance[static_cast<size_t>(i)].array() += noise_matrix.array().square();
    }
    noise_mean[static_cast<size_t>(i)].array() /= static_cast<double>(n_noise);
    noise_variance[static_cast<size_t>(i)].array() /= static_cast<double>(n_noise);
    if (mean_subtract) {
        noise_variance[static_cast<size_t>(i)].array() -=
            noise_mean[static_cast<size_t>(i)].array().square();
        noise_variance[static_cast<size_t>(i)] =
            noise_variance[static_cast<size_t>(i)].array().max(0.0).matrix();
    }

    double weight_threshold = 0.0;
    if (cov_cut > 0.0) {
        weight_threshold = engine_utils::find_weight_threshold(weight_formal[static_cast<size_t>(i)], cov_cut);
    }
    if (!std::isfinite(weight_threshold) || weight_threshold < 0.0) {
        weight_threshold = 0.0;
    }

    Eigen::Index n_valid = 0;
    for (Eigen::Index r=0; r<n_rows; ++r) {
        for (Eigen::Index c=0; c<n_cols; ++c) {
            const double w = weight_formal[static_cast<size_t>(i)](r,c);
            const double v = noise_variance[static_cast<size_t>(i)](r,c);
            if (std::isfinite(w) && w > 0.0 && w >= weight_threshold &&
                std::isfinite(v) && v > 0.0) {
                n_valid++;
            }
        }
    }
    noise_valid_pixels(i) = static_cast<double>(n_valid);

    double scale = 1.0;
    if (n_valid > 0) {
        Eigen::VectorXd ratios(n_valid);
        Eigen::Index idx = 0;
        double ns_sum = 0.0;
        double ns_sum_sq = 0.0;
        Eigen::Index ns_count = 0;
        for (Eigen::Index j=0; j<n_noise; ++j) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
                noise[i].data() + j * n_rows * n_cols, n_rows, n_cols);
            for (Eigen::Index r=0; r<n_rows; ++r) {
                for (Eigen::Index c=0; c<n_cols; ++c) {
                    const double w = weight_formal[static_cast<size_t>(i)](r,c);
                    const double v = noise_variance[static_cast<size_t>(i)](r,c);
                    if (std::isfinite(w) && w > 0.0 && w >= weight_threshold &&
                        std::isfinite(v) && v > 0.0) {
                        if (j == 0) {
                            ratios(idx++) = w * v;
                        }
                        const double ns = noise_matrix(r,c) * std::sqrt(w);
                        ns_sum += ns;
                        ns_sum_sq += ns * ns;
                        ns_count++;
                    }
                }
            }
        }

        const double med_ratio = tula::alg::median(ratios);
        if (std::isfinite(med_ratio) && med_ratio > 0.0) {
            noise_weight_median_ratio(i) = med_ratio;
            scale = 1.0 / med_ratio;
        }
        if (ns_count > 1) {
            const double mean = ns_sum / static_cast<double>(ns_count);
            const double var = (ns_sum_sq - static_cast<double>(ns_count) * mean * mean) /
                               static_cast<double>(ns_count - 1);
            noise_s2n_sigma(i) = std::sqrt(std::max(0.0, var));
        }
    }

    if (!std::isfinite(scale) || scale <= 0.0) {
        throw std::runtime_error(
            "empirical coefficient rescale is non-finite or nonpositive");
    }

    noise_weight_scale(i) = scale;
    weight_empirical[static_cast<size_t>(i)] =
        weight_formal[static_cast<size_t>(i)] * scale;
    if (!weight_empirical[static_cast<size_t>(i)].array().isFinite().all()) {
        throw std::runtime_error(
            "empirical coefficient rescale produced a non-finite plane");
    }

    if (apply_empirical_weight_scale) {
        if (science_products.initialized && !raw_science_parent) {
            science_products.coefficient_stage =
                science_products.is_coadd
                    ? science_map_coadd_empirical_coefficient_stage
                    : science_map_observation_empirical_coefficient_stage;
        }
        weight[i] = weight_empirical[static_cast<size_t>(i)];
        refresh_science_products_after_coefficient_rescale(i);
    }

    sig2noise_pixel[static_cast<size_t>(i)] =
        (signal[i].array() * weight_empirical[static_cast<size_t>(i)].array().max(0.0).sqrt()).matrix();
    point_source_uncertainty[static_cast<size_t>(i)] =
        noise_variance[static_cast<size_t>(i)].array().max(0.0).sqrt().matrix();
    sig2noise_point_source[static_cast<size_t>(i)] =
        (point_source_uncertainty[static_cast<size_t>(i)].array() > 0.0)
            .select(signal[i].array() / point_source_uncertainty[static_cast<size_t>(i)].array(), 0.0)
            .matrix();
}

void MapBuffer::calc_median_rms_annulus(double inner_radius_rad, double outer_radius_rad) {
    // average filtered rms vector
    median_rms.setZero(weight.size());

    // distance to each pixel
    Eigen::MatrixXd dist(n_rows,n_cols);

    // calculate distance to each pixel from center (same for all maps)
    for (Eigen::Index i=0; i<n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            dist(i,j) = sqrt(pow(rows_tan_vec(i),2) + pow(cols_tan_vec(j),2));
        }
    }

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); ++i) {
        int n_pts = 0;
        // loop through pixels
        for (Eigen::Index j=0; j<n_rows; ++j) {
            for (Eigen::Index k=0; k<n_cols; ++k) {
                if (dist(j,k) > inner_radius_rad && dist(j,k) <= outer_radius_rad) {
                    n_pts++;
                    median_rms(i) += signal[i](j,k);
                }
            }
        }
        // get mean
        median_rms(i) /= n_pts;
    }
}

bool MapBuffer::find_sources(Eigen::Index map_index) {
    // calc coverage bool map
    Eigen::MatrixXd ones, zeros;

    ones.setOnes(n_rows, n_cols);
    zeros.setZero(n_rows, n_cols);

    // get weight threshold for current map
    auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(map_index);
    // if weight is less than threshold, set to zero, otherwise set to one
    Eigen::MatrixXd cov_bool = (weight[map_index].array() < weight_threshold).select(zeros,ones);

    Eigen::MatrixXd source_signal = signal[map_index];
    if (source_finder_mode=="negative") {
        source_signal = -source_signal;
    }

    // s/n map
    Eigen::MatrixXd sig2noise = sqrt(weight[map_index].array())*source_signal.array();

    // find pixels equal or above source sigma
    std::vector<int> row_index, col_index;

    // search both positive and negatives
    if (source_finder_mode=="both") {
        for (Eigen::Index i=0; i<n_rows; ++i) {
            for (Eigen::Index j=0; j<n_cols; ++j) {
                if (cov_bool(i,j) == 1) {
                    if (abs(sig2noise(i,j)) >= source_sigma) {
                        row_index.push_back(i);
                        col_index.push_back(j);
                    }
                }
            }
        }
    }
    else {
        for (Eigen::Index i=0; i<n_rows; ++i) {
            for (Eigen::Index j=0; j<n_cols; ++j) {
                if (cov_bool(i,j) == 1) {
                    if (sig2noise(i,j) >= source_sigma) {
                        row_index.push_back(i);
                        col_index.push_back(j);
                    }
                }
            }
        }
    }

    // if no sources found
    if (row_index.size()==0) {
        return false;
    }

    // make sure source extremum is within good coverage region by
    // searching in index boxes of +/- 1 pixel around hot pixels
    std::vector<int> row_source_index, col_source_index;
    for (unsigned int i=0; i<row_index.size(); ++i) {
        double extremum;
        const Eigen::Index row0 = std::max<Eigen::Index>(0, row_index[i] - 1);
        const Eigen::Index row1 = std::min<Eigen::Index>(n_rows - 1, row_index[i] + 1);
        const Eigen::Index col0 = std::max<Eigen::Index>(0, col_index[i] - 1);
        const Eigen::Index col1 = std::min<Eigen::Index>(n_cols - 1, col_index[i] + 1);
        if (source_finder_mode=="both" && source_signal(row_index[i],col_index[i]) < 0.0) {
            extremum = source_signal(row_index[i],col_index[i]);
            // find minimum within index box
            for (Eigen::Index j=row0; j<=row1; ++j) {
                for (Eigen::Index k=col0; k<=col1; ++k) {
                    if (source_signal(j,k) < extremum) {
                        extremum = source_signal(j,k);
                    }
                }
            }
        }
        else {
            extremum = source_signal(row_index[i],col_index[i]);
            // find maximum within index box
            for (Eigen::Index j=row0; j<=row1; ++j) {
                for (Eigen::Index k=col0; k<=col1; ++k) {
                    if (source_signal(j,k) > extremum) {
                        extremum = source_signal(j,k);
                    }
                }
            }
        }
        // only keep the hot pixel if it is the extremum
        if (source_signal(row_index[i],col_index[i]) == extremum) {
            row_source_index.push_back(row_index[i]);
            col_source_index.push_back(col_index[i]);
        }
    }

    int n_raw_sources = row_source_index.size();
    // done with vectors of all hot pixels
    row_index.clear();
    col_index.clear();

    // if no sources found
    if (n_raw_sources == 0) {
        return false;
    }

    // find indices of hot pixels close together
    std::vector<int> row_dist_index, col_dist_index;

    for (Eigen::Index i=0; i<n_raw_sources; ++i) {
        for (Eigen::Index j=0; j<n_raw_sources; ++j) {
            unsigned int row_sep = pow(row_source_index[i] - row_source_index[j],2);
            unsigned int col_sep = pow(col_source_index[i] - col_source_index[j],2);
            double hot_dist = sqrt(row_sep + col_sep);
            if (hot_dist <= (source_window_rad/pixel_size_rad) && hot_dist != 0.0) {
                row_dist_index.push_back(i);
                col_dist_index.push_back(j);
            }
        }
    }

    // flag non-maximum hot pixel indices
    if (row_dist_index.size() != 0) {
        for (unsigned int i=0; i<row_dist_index.size(); ++i) {
            if (row_source_index[row_dist_index[i]] == -1 || col_source_index[col_dist_index[i]] == -1) {
                continue;
            }
            double f1 = source_signal(row_source_index[row_dist_index[i]],col_source_index[row_dist_index[i]]);
            double f2 = source_signal(row_source_index[col_dist_index[i]],col_source_index[col_dist_index[i]]);
            // determine if same sign and which sign
            if (f1 < 0.0 && f2 < 0.0) {
                // negative case
                if (f1 <= f2) {
                    row_source_index[col_dist_index[i]] = -1;
                    col_source_index[col_dist_index[i]] = -1;
                }
                else {
                    row_source_index[row_dist_index[i]] = -1;
                    col_source_index[row_dist_index[i]] = -1;
                }
            }
            else{
                // positive case
                if (f1 >= f2) {
                    row_source_index[col_dist_index[i]] = -1;
                    col_source_index[col_dist_index[i]] = -1;
                }
                else{
                    row_source_index[row_dist_index[i]] = -1;
                    col_source_index[row_dist_index[i]] = -1;
                }
            }
        }
    }

    row_dist_index.clear();
    col_dist_index.clear();

    // get rows/cols of each source
    std::vector<int> row_source_loc, col_source_loc;
    for (Eigen::Index i=0; i<n_raw_sources; ++i) {
        if ((row_source_index[i] != -1) && (col_source_index[i] != -1)) {
            row_source_loc.push_back(row_source_index[i]);
            col_source_loc.push_back(col_source_index[i]);
            n_sources[map_index]++;
        }
    }

    // done with flag filled source index vectors
    row_source_index.clear();
    col_source_index.clear();

    // copy locations for current map
    row_source_locs[map_index] = Eigen::Map<Eigen::VectorXi>(row_source_loc.data(),
                                                             row_source_loc.size());
    col_source_locs[map_index] = Eigen::Map<Eigen::VectorXi>(col_source_loc.data(),
                                                             col_source_loc.size());

    // if no sources found
    if (n_sources[map_index] == 0) {
        return false;
    }

    return true;
}

}  // namespace mapmaking

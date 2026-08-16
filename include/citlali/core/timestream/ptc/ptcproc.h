#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <complex>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unsupported/Eigen/FFT>

#include <tula/logging.h>
#include <tula/nc.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/engine/io.h>
#include <citlali/core/pipeline/timestream_native_consumer_bridge.h>
#include <citlali/core/pipeline/timestream_invariant_validation.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>
#include <citlali/core/utils/sha256.h>

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/timestream/ptc/clean.h>
#include <citlali/core/timestream/rtc/despike.h>

#include <citlali/core/utils/toltec_io.h>

namespace timestream {

using timestream::TCData;

template <class Matrix>
std::string ptc_realization_matrix_digest(const Matrix &matrix) {
    citlali::utils::Sha256 digest;
    auto add = [&](const std::string &value) {
        digest.update(std::to_string(value.size()));
        digest.update(":");
        digest.update(value);
        digest.update(";");
    };
    add(std::to_string(matrix.rows()));
    add(std::to_string(matrix.cols()));
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            using Scalar = std::remove_cv_t<typename Matrix::Scalar>;
            if constexpr (std::is_integral_v<Scalar>) {
                add(std::to_string(static_cast<long long>(matrix(row, col))));
            }
            else {
                std::ostringstream value;
                value << std::hexfloat
                      << static_cast<double>(matrix(row, col));
                add(value.str());
            }
        }
    }
    return "sha256:" + digest.finish();
}

class PTCProc: public TCProc {
public:
    struct PCARealizationSummary {
        std::string grouping;
        Eigen::Index group_key = -1;
        Eigen::Index array_index = -1;
        Eigen::Index configured_cut = 0;
        Eigen::Index applied_cut = 0;
        Eigen::Index forced_limit_index = -1;
        std::string eigenvalue_digest;
        std::string eigenvector_digest;
    };

    struct MeanRealizationSummary {
        bool mean_subtracted = false;
        bool source_mask_applied = false;
        std::size_t masked_sample_count = 0;
        std::string mask_digest = "unavailable";
    };

    std::map<Eigen::Index, std::vector<PCARealizationSummary>>
        pca_realization_summary_by_scan;
    std::map<Eigen::Index, MeanRealizationSummary>
        mean_realization_summary_by_scan;

    std::vector<PCARealizationSummary>
    snapshot_pca_realization_summary(Eigen::Index scan_id) {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto it = pca_realization_summary_by_scan.find(scan_id);
        return it == pca_realization_summary_by_scan.end()
                   ? std::vector<PCARealizationSummary>{}
                   : it->second;
    }

    std::optional<MeanRealizationSummary>
    snapshot_mean_realization_summary(Eigen::Index scan_id) {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto it = mean_realization_summary_by_scan.find(scan_id);
        return it == mean_realization_summary_by_scan.end()
                   ? std::optional<MeanRealizationSummary>{}
                   : std::optional<MeanRealizationSummary>{it->second};
    }
    // controls for timestream reduction
    bool run_clean;
    // median weight factor
    double med_weight_factor;
    // upper and lower weight limits for outliers
    double lower_weight_factor, upper_weight_factor;
    // weight type (full, approximate, const)
    std::string weighting_type;
    // source exclusion radius used only for full-weight variance estimation
    double source_mask_radius_arcsec = 0.0;
    // bounds for the dimensionless residual-variance correction in hybrid weighting
    double hybrid_correction_min_factor = 0.5;
    double hybrid_correction_max_factor = 2.0;

    struct WeightValidationOptions {
        bool enabled = false;
        int accumulation_iters = 1;
        int apply_start_iter = 1;
        int min_valid_scans = 1;
        double min_factor = 0.1;
        double unvalidated_factor = 1.0;
        bool require_fruitloops_model = true;
        bool transient_ratio_enabled = false;
        double ratio_power = 1.0;
        double transient_ratio_power = 1.0;
        bool upward_enabled = false;
        double upward_max_factor = 1.10;
        double upward_power = 1.0;
        double upward_min_base_factor = 0.95;
        bool upward_require_atmospheric = true;
        double upward_min_atmospheric_factor = 0.9;
        bool atmospheric_correlation_enabled = true;
        std::string atmospheric_grouping = "array";
        int atmospheric_min_detectors = 8;
        double atmospheric_ref = 0.0;
        double atmospheric_span = 0.15;
        double atmospheric_power = 1.0;
        double min_good_frac = 0.5;
        int min_overlap = 200;
        int max_samples = 5000;
        bool high_weight_validation_enabled = true;
        bool high_weight_apply_caps = true;
        std::string high_weight_grouping = "array";
        int high_weight_min_group_detectors = 20;
        double high_weight_log_robust_z = 6.0;
        double high_weight_max_median_factor = 8.0;
        double high_weight_cap_median_factor = 4.0;
        double high_weight_min_validated_factor = 0.95;
    };

    WeightValidationOptions weight_validation;
    int weight_validation_current_iter = 0;
    int weight_validation_accumulated_iters = 0;
    int weight_validation_current_iter_contribution_count = 0;
    bool weight_validation_finalized = false;
    Eigen::VectorXd weight_validation_ratio_penalty_sum;
    Eigen::VectorXd weight_validation_ratio_value_sum;
    Eigen::VectorXi weight_validation_ratio_value_count;
    Eigen::VectorXi weight_validation_ratio_count;
    Eigen::VectorXd weight_validation_atm_penalty_sum;
    Eigen::VectorXd weight_validation_atm_corr_sum;
    Eigen::VectorXi weight_validation_atm_count;
    Eigen::VectorXd weight_validation_detector_penalty;
    Eigen::VectorXi weight_validation_detector_validated;
    std::shared_ptr<std::mutex> weight_validation_mutex = std::make_shared<std::mutex>();

    struct HighWeightDiagSummary {
        int iter = -1;
        Eigen::Index scan = -1;
        Eigen::Index det = -1;
        int uid = kTransientFillInt;
        Eigen::Index nw = -1;
        Eigen::Index array = -1;
        std::string grouping = "array";
        std::string reason = "high_weight";
        double approximate_weight = std::numeric_limits<double>::quiet_NaN();
        double final_weight = std::numeric_limits<double>::quiet_NaN();
        double group_median_weight = std::numeric_limits<double>::quiet_NaN();
        double robust_z = std::numeric_limits<double>::quiet_NaN();
        double applied_cap = std::numeric_limits<double>::quiet_NaN();
        double validation_factor = std::numeric_limits<double>::quiet_NaN();
        bool cap_recommended = false;
        bool cap_applied = false;
        bool validated = false;
    };
    std::map<Eigen::Index, std::vector<HighWeightDiagSummary>> high_weight_summary_by_scan;
    std::shared_ptr<std::mutex> diag_summary_mutex = std::make_shared<std::mutex>();

    std::vector<HighWeightDiagSummary> snapshot_high_weight_summary(Eigen::Index scan_id) {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto it = high_weight_summary_by_scan.find(scan_id);
        if (it == high_weight_summary_by_scan.end()) {
            return {};
        }
        return it->second;
    }

    // ptc tod proc
    timestream::Cleaner cleaner;

    // Phase B1 production-consumer seam.  It constructs the existing
    // effective detector grouping (including exact corr_nw memberships and
    // pass-through columns), gathers a private finite rectangular view from
    // measured native rows, and checks optional-mode compatibility against
    // the actual detector-cell exclusion mask.  It deliberately does not
    // activate the bridge in run() at this checkpoint.
    template <class calib_type>
    citlali::pipeline::NativePreparedPcaOperation
    prepare_native_consumer_pca(
        const citlali::pipeline::NativeDetectorLedger &ledger,
        const citlali::pipeline::NativeCohortSelection &selection,
        calib_type &calib,
        const std::string &grouping,
        const citlali::pipeline::NativeDetectorFlagBitsMatrix &
            actual_exclusion_bits,
        citlali::pipeline::FinitePcaPlaceholder excluded_placeholder,
        const citlali::pipeline::AptDetectorRelation *typed_relation =
            nullptr) {
        using namespace citlali::pipeline;

        const auto detector_count =
            static_cast<Eigen::Index>(calib.n_dets);
        const auto &cohort = selection.cohort();
        if (detector_count <= 0 ||
            actual_exclusion_bits.rows() !=
                static_cast<Eigen::Index>(cohort.slot_count()) ||
            actual_exclusion_bits.cols() != detector_count) {
            throw std::invalid_argument(
                "native PTC exclusion mask must match the production cohort and detector count");
        }
        const auto requested_grouping =
            Cleaner::normalize_group_name(grouping);
        if (!Cleaner::is_supported_clean_group(requested_grouping)) {
            throw std::invalid_argument(
                "native PTC requested an unsupported production grouping");
        }
        const bool corr_nw_requested =
            Cleaner::is_corr_nw_clean_group(requested_grouping);
        const bool corr_nw_enabled =
            corr_nw_requested && cleaner.corr_grouping.enabled;
        std::string effective_grouping = requested_grouping;
        if (corr_nw_requested && !corr_nw_enabled) {
            if (logger != nullptr) {
                logger->warn(
                    "cleaning group 'corr_nw' requested but clean.corr_grouping.enabled=false; falling back to nw");
            }
            effective_grouping = "nw";
        }

        const auto &apt_flag = calib.apt.at("flag");
        if (apt_flag.size() != detector_count ||
            (typed_relation != nullptr &&
             typed_relation->bindings().size() !=
                 static_cast<std::size_t>(detector_count))) {
            throw std::invalid_argument(
                "native PTC detector bindings require a complete typed relation and flag policy");
        }
        const auto *apt_nw = typed_relation == nullptr
            ? &calib.apt.at("nw") : nullptr;
        const auto *apt_uid = typed_relation == nullptr
            ? &calib.apt.at("uid") : nullptr;
        if (typed_relation == nullptr &&
            (apt_nw->size() != detector_count ||
             apt_uid->size() != detector_count)) {
            throw std::invalid_argument(
                "legacy native PTC test binding requires complete nw and uid APT columns");
        }

        NativePreparedPcaOperation prepared{
            cohort.operation(), effective_grouping, detector_count};
        PcaCompatibilityInputs compatibility;
        compatibility.null_model_active_for_operation =
            cleaner.null_model.enabled &&
            cleaner.null_model_enabled_for_group(effective_grouping);
        compatibility.marchenko_pastur_active_for_operation =
            cleaner.marchenko_pastur.enabled &&
            cleaner.marchenko_pastur_enabled_for_group(effective_grouping);
        compatibility.marchenko_pastur_band_requested =
            cleaner.marchenko_pastur.band_low_Hz > 0.0 ||
            cleaner.marchenko_pastur.band_high_Hz > 0.0;
        compatibility.adaptive_selector_active_for_operation =
            cleaner.adaptive_selector.enabled &&
            cleaner.adaptive_selector_enabled_for_group(effective_grouping) &&
            !corr_nw_enabled;

        std::vector<TimestreamNetworkId> detector_networks(
            static_cast<std::size_t>(detector_count));
        std::vector<TimestreamDetectorUid> detector_uids(
            static_cast<std::size_t>(detector_count));
        Eigen::VectorXi detector_apt_flags(detector_count);
        std::set<TimestreamDetectorUid> seen_uids;
        for (Eigen::Index detector = 0; detector < detector_count;
             ++detector) {
            const long double apt_flag_value =
                static_cast<long double>(apt_flag(detector));
            if (!std::isfinite(apt_flag_value) ||
                std::floor(apt_flag_value) != apt_flag_value ||
                apt_flag_value < static_cast<long double>(
                    std::numeric_limits<int>::min()) ||
                apt_flag_value > static_cast<long double>(
                    std::numeric_limits<int>::max())) {
                throw std::invalid_argument(
                    "production APT detector flag must be a representable integer");
            }
            TimestreamNetworkId network_id = -1;
            TimestreamDetectorUid uid = -1;
            if (typed_relation != nullptr) {
                const auto reference =
                    typed_relation->binding_reference_for_column(
                        static_cast<std::size_t>(detector));
                const auto &binding =
                    typed_relation->require_binding(reference);
                network_id = static_cast<TimestreamNetworkId>(
                    binding.network);
                uid = static_cast<TimestreamDetectorUid>(binding.uid);
            }
            else {
                const long double network_value =
                    static_cast<long double>((*apt_nw)(detector));
                const long double uid_value =
                    static_cast<long double>((*apt_uid)(detector));
                if (!std::isfinite(network_value) ||
                    std::floor(network_value) != network_value ||
                    network_value < 0.0L ||
                    network_value > static_cast<long double>(
                        std::numeric_limits<TimestreamNetworkId>::max())) {
                    throw std::invalid_argument(
                        "production APT nw binding must be a nonnegative integer network ID");
                }
                if (!std::isfinite(uid_value) ||
                    std::floor(uid_value) != uid_value ||
                    uid_value < 0.0L ||
                    uid_value > static_cast<long double>(
                        std::numeric_limits<TimestreamDetectorUid>::max())) {
                    throw std::invalid_argument(
                        "production APT uid binding must be a nonnegative representable integer");
                }
                network_id =
                    static_cast<TimestreamNetworkId>(network_value);
                uid = static_cast<TimestreamDetectorUid>(uid_value);
            }
            if (!seen_uids.insert(uid).second) {
                throw std::invalid_argument(
                    "production APT detector UID must be injective");
            }
            detector_networks.at(static_cast<std::size_t>(detector)) =
                network_id;
            detector_uids.at(static_cast<std::size_t>(detector)) = uid;
            detector_apt_flags(detector) =
                static_cast<int>(apt_flag_value);
        }

        auto make_prepared_group =
            [&](std::vector<TimestreamDetectorColumn> detector_columns,
                Eigen::Index group_key,
                Eigen::Index subgroup_index,
                NativePreparedPcaGroupRole role) {
                if (detector_columns.empty()) {
                    throw std::logic_error(
                        "native PTC prepared group must contain detectors");
                }
                const auto group_detector_count =
                    static_cast<Eigen::Index>(detector_columns.size());
                std::vector<NativeDetectorColumnBinding> bindings;
                std::vector<TimestreamDetectorUid> group_uids;
                bindings.reserve(detector_columns.size());
                group_uids.reserve(detector_columns.size());
                Eigen::VectorXi group_apt_flags(group_detector_count);
                NativeDetectorFlagBitsMatrix group_exclusions(
                    actual_exclusion_bits.rows(), group_detector_count);
                for (Eigen::Index local = 0;
                     local < group_detector_count; ++local) {
                    const auto detector_column = detector_columns.at(
                        static_cast<std::size_t>(local));
                    if (detector_column < 0 ||
                        detector_column >= detector_count) {
                        throw std::logic_error(
                            "native PTC detector subgroup column is out of range");
                    }
                    const auto index =
                        static_cast<std::size_t>(detector_column);
                    bindings.push_back(NativeDetectorColumnBinding{
                        detector_column, detector_uids.at(index),
                        detector_networks.at(index)});
                    group_uids.push_back(detector_uids.at(index));
                    group_apt_flags(local) =
                        detector_apt_flags(detector_column);
                    group_exclusions.col(local) =
                        actual_exclusion_bits.col(detector_column);
                }
                auto working = gather_native_detector_pca_working_set(
                    ledger, selection, std::move(bindings),
                    group_exclusions, excluded_placeholder);
                finalize_native_detector_pca_binding(
                    working, group_apt_flags, excluded_placeholder);
                require_native_detector_pca_compatibility(
                    classify_native_detector_pca_compatibility(
                        working, compatibility));
                return NativePreparedPcaGroup{
                    effective_grouping, group_key, subgroup_index,
                    role, std::move(detector_columns),
                    std::move(group_uids), std::move(group_apt_flags),
                    std::move(working)};
            };

        std::map<Eigen::Index,
                 std::tuple<Eigen::Index, Eigen::Index>> group_limits;
        if (corr_nw_enabled) {
            if (typed_relation != nullptr) {
                Eigen::Index first = 0;
                while (first < detector_count) {
                    const auto network = detector_networks.at(
                        static_cast<std::size_t>(first));
                    Eigen::Index past = first + 1;
                    while (past < detector_count &&
                           detector_networks.at(
                               static_cast<std::size_t>(past)) == network) {
                        ++past;
                    }
                    if (!group_limits
                             .emplace(network,
                                      std::make_tuple(first, past))
                             .second) {
                        throw std::logic_error(
                            "typed PTC network detector columns are not contiguous");
                    }
                    first = past;
                }
            }
            else {
                group_limits = get_grouping("nw", calib, detector_count);
            }
            for (const auto &[network_key, limits] : group_limits) {
                const auto first = std::get<0>(limits);
                const auto past = std::get<1>(limits);
                if (first < 0 || past <= first || past > detector_count) {
                    throw std::logic_error(
                        "production corr_nw base grouping produced an invalid network interval");
                }
                for (auto detector = first; detector < past; ++detector) {
                    if (detector_networks.at(static_cast<std::size_t>(
                            detector)) != network_key) {
                        throw std::logic_error(
                            "production corr_nw base interval and APT network identity disagree");
                    }
                }
                std::vector<TimestreamDetectorColumn> base_columns;
                base_columns.reserve(static_cast<std::size_t>(past - first));
                for (auto detector = first; detector < past; ++detector) {
                    base_columns.push_back(detector);
                }
                auto base_group = make_prepared_group(
                    base_columns, network_key, 0,
                    NativePreparedPcaGroupRole::pca_clean);
                const auto corr_groups = cleaner.get_corr_groups(
                    base_group.working_set.values(),
                    base_group.working_set.exclusion_flags(),
                    base_group.apt_flags);
                std::vector<bool> grouped(
                    static_cast<std::size_t>(past - first), false);
                for (Eigen::Index subgroup_index = 0;
                     subgroup_index < static_cast<Eigen::Index>(
                         corr_groups.groups.size()); ++subgroup_index) {
                    const auto &local_columns = corr_groups.groups.at(
                        static_cast<std::size_t>(subgroup_index));
                    if (local_columns.size() < 2) {
                        continue;
                    }
                    std::vector<TimestreamDetectorColumn> global_columns;
                    global_columns.reserve(local_columns.size());
                    for (const auto local_column : local_columns) {
                        if (local_column < 0 ||
                            local_column >= past - first ||
                            grouped.at(static_cast<std::size_t>(
                                local_column))) {
                            throw std::logic_error(
                                "production corr_nw subgroup is not an injective network-local binding");
                        }
                        grouped.at(static_cast<std::size_t>(local_column)) =
                            true;
                        global_columns.push_back(first + local_column);
                    }
                    prepared.groups.push_back(make_prepared_group(
                        std::move(global_columns), network_key,
                        subgroup_index,
                        NativePreparedPcaGroupRole::pca_clean));
                }
                std::vector<TimestreamDetectorColumn> pass_through_columns;
                for (Eigen::Index local = 0; local < past - first; ++local) {
                    if (!grouped.at(static_cast<std::size_t>(local))) {
                        pass_through_columns.push_back(first + local);
                    }
                }
                if (!pass_through_columns.empty()) {
                    prepared.groups.push_back(make_prepared_group(
                        std::move(pass_through_columns), network_key,
                        static_cast<Eigen::Index>(corr_groups.groups.size()),
                        NativePreparedPcaGroupRole::pass_through));
                }
            }
            prepared.require_complete_detector_partition();
            return prepared;
        }

        if (Cleaner::is_all_clean_group(effective_grouping)) {
            group_limits[0] = std::make_tuple(0, detector_count);
        }
        else if (typed_relation != nullptr &&
                 citlali::config::is_network_map_grouping(
                     effective_grouping)) {
            Eigen::Index first = 0;
            while (first < detector_count) {
                const auto network = detector_networks.at(
                    static_cast<std::size_t>(first));
                Eigen::Index past = first + 1;
                while (past < detector_count &&
                       detector_networks.at(
                           static_cast<std::size_t>(past)) == network) {
                    ++past;
                }
                if (!group_limits
                         .emplace(network, std::make_tuple(first, past))
                         .second) {
                    throw std::logic_error(
                        "typed PTC network detector columns are not contiguous");
                }
                first = past;
            }
        }
        else {
            const auto grouping_column = calib.apt.find(effective_grouping);
            if (grouping_column == calib.apt.end() ||
                grouping_column->second.size() != detector_count) {
                throw std::invalid_argument(
                    "native PTC production grouping APT column is missing or incomplete");
            }
            group_limits =
                get_grouping(effective_grouping, calib, detector_count);
        }

        for (const auto &[group_key, limits] : group_limits) {
            const auto first = std::get<0>(limits);
            const auto past = std::get<1>(limits);
            if (first < 0 || past <= first || past > detector_count) {
                throw std::logic_error(
                    "production PTC grouping produced an invalid detector interval");
            }
            std::vector<TimestreamDetectorColumn> detector_columns;
            detector_columns.reserve(
                static_cast<std::size_t>(past - first));
            for (auto detector_column = first;
                 detector_column < past; ++detector_column) {
                detector_columns.push_back(detector_column);
            }
            prepared.groups.push_back(make_prepared_group(
                std::move(detector_columns), group_key, 0,
                NativePreparedPcaGroupRole::pca_clean));
        }
        prepared.require_complete_detector_partition();
        return prepared;
    }

    struct CorrNWDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_candidates = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index n_det_grouped = 0;
        Eigen::Index n_det_ungrouped = 0;
        Eigen::Index n_groups_raw = 0;
        Eigen::Index n_groups_final = 0;
        Eigen::Index sample_step = 1;
    };

    struct WeightCorrPenaltyTermOptions {
        bool enabled = true;
        double ref = 0.05;
        double span = 0.15;
        double weight = 1.0;
    };

    struct WeightCorrPenaltyBandOptions {
        bool enabled = false;
        double ref = 0.6;
        double span = 2.0;
        double weight = 0.5;
        double low_min_Hz = 0.05;
        double low_max_Hz = 0.5;
        double mid_min_Hz = 0.5;
        double mid_max_Hz = 2.0;
    };

    struct WeightCorrPenaltyOptions {
        bool enabled = false;
        double min_good_frac = 0.7;
        int min_overlap = 200;
        int max_samples = 20000;
        int max_pairs = 4000;
        std::uint32_t seed = 12345;
        double floor = 0.05;
        double exponent = 2.0;
        WeightCorrPenaltyTermOptions pair_corr;
        WeightCorrPenaltyTermOptions cm_el_corr{false, 0.05, 0.25, 0.5};
        WeightCorrPenaltyBandOptions cm_low_mid_ratio;
    };

    struct WeightCorrPenaltyDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_candidates = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index n_det_weighted = 0;
        Eigen::Index sample_step = 1;
        double pair_med_abs_corr = std::numeric_limits<double>::quiet_NaN();
        double cm_el_abs_corr = std::numeric_limits<double>::quiet_NaN();
        double cm_low_mid_ratio = std::numeric_limits<double>::quiet_NaN();
        double severity = 0.0;
        double penalty_factor = 1.0;
    };

    struct BusyRowSuppressionOptions {
        bool enabled = false;
        bool require_busy_veto = true;
        int min_candidate_clusters = 5;
        double min_max_unflagged_residual_z = 25.0;
        double factor = 0.0;
    };

    struct BusyRowSuppressionDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_weighted = 0;
        Eigen::Index n_candidate_clusters = 0;
        bool busy_network_vetoed = false;
        bool applied = false;
        double max_unflagged_residual_z = std::numeric_limits<double>::quiet_NaN();
        double factor = 1.0;
    };

    WeightCorrPenaltyOptions weight_corr_penalty;
    BusyRowSuppressionOptions busy_row_suppression;
    std::map<Eigen::Index, Eigen::VectorXi> corr_nw_group_ids_by_scan;
    std::map<Eigen::Index, std::vector<CorrNWDiagSummary>> corr_nw_summary_by_scan;
    std::map<Eigen::Index, std::vector<WeightCorrPenaltyDiagSummary>> weight_corr_penalty_summary_by_scan;
    std::map<Eigen::Index, std::vector<BusyRowSuppressionDiagSummary>> busy_row_suppression_summary_by_scan;

    struct AdaptiveSelectorDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index n_time_used = 0;
        Eigen::Index sample_step = 1;
        Eigen::Index baseline_k = 0;
        Eigen::Index chosen_k = 0;
        Eigen::Index runnerup_k = -1;
        Eigen::Index n_candidates = 0;
        int selector_used = 0;
        int selector_fallback = 0;
        double chosen_score = std::numeric_limits<double>::quiet_NaN();
        double runnerup_score = std::numeric_limits<double>::quiet_NaN();
        double score_margin = std::numeric_limits<double>::quiet_NaN();
        double chosen_med_abs_corr = std::numeric_limits<double>::quiet_NaN();
        double chosen_cm_low_mid_ratio = std::numeric_limits<double>::quiet_NaN();
        double chosen_tail4_binom_z = std::numeric_limits<double>::quiet_NaN();
        double chosen_top_mode_frac = std::numeric_limits<double>::quiet_NaN();
        double eig_solve_msec = std::numeric_limits<double>::quiet_NaN();
        double candidate_eval_msec = std::numeric_limits<double>::quiet_NaN();
        double total_msec = std::numeric_limits<double>::quiet_NaN();
    };

    std::map<Eigen::Index, std::vector<AdaptiveSelectorDiagSummary>> adaptive_selector_summary_by_scan;

    struct SecondPassLocalOptions {
        bool enabled = false;
        double min_spike_sigma = 8.0;
        double min_good_frac = 0.5;
        double baseline_window_sec = 0.25;
        double sigma_scale = 0.75;
        double delta_sigma_scale = 0.75;
        double raw_candidate_rel_sigma_scale = 1.0;
        double raw_window_sec = 0.18;
        double raw_half_peak_frac = 0.5;
        double raw_max_width_sec = 0.18;
        double delta_window_sec = 0.12;
        double delta_half_peak_frac = 0.5;
        double delta_max_width_sec = 0.10;
        double max_step_shift_z = 3.0;
        double high_score_event_override = 20.0;
        double merge_within_detector_sec = 0.08;
        double cluster_events_sec = 0.08;
        int min_cluster_detectors = 3;
        double high_score_cluster_override = 9.0;
        int max_auto_flag_clusters_per_network = 3;
        bool selective_busy_network_acceptance_enabled = true;
        bool source_protection_config_enabled = true;
        bool source_protection_enabled = false;
        double source_protection_radius_arcsec = 20.0;
    };

    struct SecondPassCandidateEvent {
        int uid = kTransientFillInt;
        int kind = 0;
        int sample = kTransientFillInt;
        int start_sample = kTransientFillInt;
        int end_sample = kTransientFillInt;
        double score = std::numeric_limits<double>::quiet_NaN();
        double cluster_score = std::numeric_limits<double>::quiet_NaN();
        int cluster_sample = kTransientFillInt;
        int cluster_n_detectors = 0;
        int cluster_n_events = 0;
        bool busy_network_vetoed = false;
        bool accepted = false;
        bool source_protected = false;
    };

    struct SecondPassDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det = 0;
        Eigen::Index n_pts = 0;
        Eigen::Index n_merged_events_total = 0;
        Eigen::Index n_clusters_total = 0;
        Eigen::Index n_candidate_events = 0;
        Eigen::Index n_candidate_clusters = 0;
        Eigen::Index n_accepted_events = 0;
        Eigen::Index n_accepted_clusters = 0;
        Eigen::Index n_rejected_events = 0;
        Eigen::Index n_rejected_clusters = 0;
        Eigen::Index n_source_protected_events = 0;
        Eigen::Index n_source_protected_clusters = 0;
        Eigen::Index n_det_with_added_flags = 0;
        bool busy_network_vetoed = false;
        double existing_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
        double proposed_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
        double newly_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
        double max_unflagged_residual_z = std::numeric_limits<double>::quiet_NaN();
        int max_unflagged_residual_uid = kTransientFillInt;
        double top_candidate_cluster_peak_score = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index top_candidate_cluster_n_detectors = 0;
        Eigen::Index top_candidate_cluster_n_events = 0;
        int top_candidate_cluster_sample = kTransientFillInt;
        int top_event_uid = kTransientFillInt;
        TransientEvent top_event;
        std::vector<SecondPassCandidateEvent> candidate_events;
    };

    SecondPassLocalOptions second_pass_local;
    std::map<Eigen::Index, std::vector<SecondPassDiagSummary>> second_pass_summary_by_scan;
    std::map<Eigen::Index, Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic>> second_pass_added_flags_by_scan;

    std::vector<SecondPassDiagSummary> snapshot_second_pass_summary(Eigen::Index scan_id) {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto it = second_pass_summary_by_scan.find(scan_id);
        if (it == second_pass_summary_by_scan.end()) {
            return {};
        }
        return it->second;
    }

    // subtract detector means
    void subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
                       const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> *flags_override = nullptr);

    struct RunDiagnosticTransaction {
        std::optional<Eigen::VectorXi> corr_nw_group_ids;
        std::optional<std::vector<CorrNWDiagSummary>> corr_nw_summary;
        std::optional<std::vector<AdaptiveSelectorDiagSummary>>
            adaptive_selector_summary;
        std::optional<MeanRealizationSummary> mean_realization;
        std::optional<std::vector<PCARealizationSummary>>
            pca_realizations;
    };

    // run main processing stage
    template <class calib_type>
    void run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             calib_type &, std::string, std::string);

    template <class calib_type>
    void run_impl(TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
                  calib_type &, std::string, std::string,
                  const citlali::pipeline::AptDetectorRelation * = nullptr,
                  RunDiagnosticTransaction * = nullptr);

    template <class calib_type>
    void apply_second_pass_local(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_type &,
                                 std::string, std::string);

    template <typename calib_t>
    void append_diag_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, calib_t &,
                               Eigen::Index scan_row_index = -1);

    void clear_cached_diagnostics(Eigen::Index scan_id);

    void begin_weight_validation_iteration(int iter);
    void finalize_weight_validation_iteration(int iter);
    bool weight_validation_is_enabled() const;
    bool should_accumulate_weight_validation(bool source_subtracted) const;
    void ensure_weight_validation_storage(Eigen::Index n_uids);

    template <typename apt_type>
    void accumulate_weight_validation_atmosphere(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_type &);

    // calculate detector weights
    template <typename apt_type, class tel_type>
    void calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_type &, tel_type &,
                      bool source_subtracted_for_weight_validation = false);

    // reset outlier weights to the median
    template <typename calib_t>
    auto reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, std::string);

    // append time chunk to tod netcdf file
    template <typename calib_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, calib_t &, bool apply_det_offsets = false,
                          Eigen::Index scan_row_index = -1, bool mini_output = false);
};

inline bool PTCProc::weight_validation_is_enabled() const {
    return weight_validation.enabled ||
           citlali::config::is_validated_processed_weighting_type(weighting_type);
}

inline bool PTCProc::should_accumulate_weight_validation(bool source_subtracted) const {
    if (!weight_validation_is_enabled()) {
        return false;
    }
    std::lock_guard<std::mutex> lk(*weight_validation_mutex);
    if (weight_validation_finalized) {
        return false;
    }
    if (weight_validation_accumulated_iters >= weight_validation.accumulation_iters) {
        return false;
    }
    if (weight_validation.require_fruitloops_model && !source_subtracted) {
        return false;
    }
    return true;
}

inline void PTCProc::ensure_weight_validation_storage(Eigen::Index n_uids) {
    if (n_uids <= 0) {
        return;
    }
    if (weight_validation_detector_penalty.size() >= n_uids &&
        weight_validation_detector_validated.size() >= n_uids) {
        return;
    }

    const Eigen::Index old_size =
        std::max(weight_validation_detector_penalty.size(),
                 weight_validation_detector_validated.size());
    auto grow_double = [&](Eigen::VectorXd &vec, double fill) {
        Eigen::VectorXd old = vec;
        vec = Eigen::VectorXd::Constant(n_uids, fill);
        if (old.size() > 0) {
            vec.head(old.size()) = old;
        }
    };
    auto grow_int = [&](Eigen::VectorXi &vec, int fill) {
        Eigen::VectorXi old = vec;
        vec = Eigen::VectorXi::Constant(n_uids, fill);
        if (old.size() > 0) {
            vec.head(old.size()) = old;
        }
    };

    grow_double(weight_validation_ratio_penalty_sum, 0.0);
    grow_double(weight_validation_ratio_value_sum, 0.0);
    grow_int(weight_validation_ratio_value_count, 0);
    grow_int(weight_validation_ratio_count, 0);
    grow_double(weight_validation_atm_penalty_sum, 0.0);
    grow_double(weight_validation_atm_corr_sum, 0.0);
    grow_int(weight_validation_atm_count, 0);
    grow_double(weight_validation_detector_penalty,
                std::clamp(weight_validation.unvalidated_factor,
                           weight_validation.min_factor, 1.0));
    grow_int(weight_validation_detector_validated, 0);

    logger->debug("weight validation storage resized from {} to {} detector slots",
                  old_size, n_uids);
}

inline void PTCProc::begin_weight_validation_iteration(int iter) {
    std::lock_guard<std::mutex> lk(*weight_validation_mutex);
    weight_validation_current_iter = iter;
    weight_validation_current_iter_contribution_count = 0;

    if (iter == 0) {
        weight_validation_accumulated_iters = 0;
        weight_validation_finalized = false;
        weight_validation_ratio_penalty_sum.resize(0);
        weight_validation_ratio_value_sum.resize(0);
        weight_validation_ratio_value_count.resize(0);
        weight_validation_ratio_count.resize(0);
        weight_validation_atm_penalty_sum.resize(0);
        weight_validation_atm_corr_sum.resize(0);
        weight_validation_atm_count.resize(0);
        weight_validation_detector_penalty.resize(0);
        weight_validation_detector_validated.resize(0);
    }
}

inline void PTCProc::finalize_weight_validation_iteration(int iter) {
    if (!weight_validation_is_enabled()) {
        return;
    }

    std::lock_guard<std::mutex> lk(*weight_validation_mutex);
    if (weight_validation_finalized) {
        return;
    }
    if (iter != weight_validation_current_iter) {
        weight_validation_current_iter = iter;
    }
    if (weight_validation_current_iter_contribution_count <= 0) {
        logger->info("weight validation iteration {} had no source-subtracted validation contributions; not finalizing",
                     iter);
        return;
    }

    weight_validation_accumulated_iters++;
    if (weight_validation_accumulated_iters < weight_validation.accumulation_iters) {
        logger->info("weight validation accumulated {}/{} requested iterations",
                     weight_validation_accumulated_iters,
                     weight_validation.accumulation_iters);
        return;
    }

    const Eigen::Index n_uids = std::max(weight_validation_ratio_count.size(),
                                        weight_validation_atm_count.size());
    if (n_uids <= 0) {
        logger->warn("weight validation had contributions but no detector slots; leaving validation inactive");
        return;
    }
    ensure_weight_validation_storage(n_uids);

    const double min_factor = std::clamp(weight_validation.min_factor, 0.0, 1.0);
    const double unvalidated_factor = std::clamp(weight_validation.unvalidated_factor,
                                                min_factor, 1.0);
    const double upward_max_factor =
        weight_validation.upward_enabled
            ? std::max(1.0, weight_validation.upward_max_factor)
            : 1.0;
    const double upward_power = std::max(weight_validation.upward_power, 0.0);
    const double upward_min_base =
        std::clamp(weight_validation.upward_min_base_factor, 0.0, 1.0);
    const double upward_min_atm =
        std::clamp(weight_validation.upward_min_atmospheric_factor, 0.0, 1.0);
    const int min_valid = std::max(1, weight_validation.min_valid_scans);
    weight_validation_detector_penalty =
        Eigen::VectorXd::Constant(n_uids, unvalidated_factor);
    weight_validation_detector_validated =
        Eigen::VectorXi::Zero(n_uids);

    std::vector<double> factors;
    factors.reserve(static_cast<std::size_t>(n_uids));
    Eigen::Index n_ratio_valid = 0;
    Eigen::Index n_ratio_upward_valid = 0;
    Eigen::Index n_atm_valid = 0;
    Eigen::Index n_penalized = 0;
    Eigen::Index n_boosted = 0;
    for (Eigen::Index uid = 0; uid < n_uids; ++uid) {
        double factor = unvalidated_factor;
        bool have_factor = false;
        bool have_atm_factor = false;
        double atm_factor = std::numeric_limits<double>::quiet_NaN();

        if (uid < weight_validation_ratio_count.size() &&
            weight_validation_ratio_count(uid) >= min_valid) {
            const double avg =
                weight_validation_ratio_penalty_sum(uid) /
                static_cast<double>(weight_validation_ratio_count(uid));
            if (std::isfinite(avg)) {
                factor = std::clamp(avg, min_factor, 1.0);
                have_factor = true;
                n_ratio_valid++;
            }
        }

        if (uid < weight_validation_atm_count.size() &&
            weight_validation_atm_count(uid) >= min_valid) {
            const double avg =
                weight_validation_atm_penalty_sum(uid) /
                static_cast<double>(weight_validation_atm_count(uid));
            if (std::isfinite(avg)) {
                atm_factor = std::clamp(avg, min_factor, 1.0);
                factor = have_factor ? std::min(factor, atm_factor) : atm_factor;
                have_factor = true;
                have_atm_factor = true;
                n_atm_valid++;
            }
        }

        if (!have_factor) {
            factor = unvalidated_factor;
        }
        bool detector_validated = have_factor;
        if (weight_validation.upward_enabled &&
            uid < weight_validation_ratio_value_count.size() &&
            weight_validation_ratio_value_count(uid) >= min_valid &&
            factor >= upward_min_base) {
            const double avg_correction =
                weight_validation_ratio_value_sum(uid) /
                static_cast<double>(weight_validation_ratio_value_count(uid));
            const bool atm_ok =
                !weight_validation.upward_require_atmospheric ||
                (have_atm_factor && atm_factor >= upward_min_atm);
            if (atm_ok && std::isfinite(avg_correction) && avg_correction > 1.0) {
                const double raw_upward =
                    std::clamp(avg_correction, 1.0, upward_max_factor);
                const double powered_upward =
                    std::clamp(std::pow(raw_upward, upward_power),
                               1.0, upward_max_factor);
                double atm_quality = 1.0;
                if (weight_validation.upward_require_atmospheric) {
                    const double atm_span = std::max(1.0 - upward_min_atm, 1e-12);
                    atm_quality = std::clamp((atm_factor - upward_min_atm) / atm_span,
                                             0.0, 1.0);
                }
                const double upward_factor =
                    1.0 + (powered_upward - 1.0) * atm_quality;
                factor = std::max(factor, upward_factor);
                detector_validated = true;
                n_ratio_upward_valid++;
            }
        }
        factor = std::clamp(factor, min_factor, upward_max_factor);
        weight_validation_detector_penalty(uid) = factor;
        weight_validation_detector_validated(uid) = detector_validated ? 1 : 0;
        if (factor < 0.999) {
            n_penalized++;
        }
        if (factor > 1.001) {
            n_boosted++;
        }
        factors.push_back(factor);
    }

    double median_factor = std::numeric_limits<double>::quiet_NaN();
    double p10_factor = std::numeric_limits<double>::quiet_NaN();
    if (!factors.empty()) {
        auto quantile = [](std::vector<double> values, double q) {
            std::sort(values.begin(), values.end());
            q = std::clamp(q, 0.0, 1.0);
            const double pos = q * static_cast<double>(values.size() - 1);
            const auto lo = static_cast<std::size_t>(std::floor(pos));
            const auto hi = static_cast<std::size_t>(std::ceil(pos));
            if (lo == hi) {
                return values[lo];
            }
            const double frac = pos - static_cast<double>(lo);
            return values[lo] * (1.0 - frac) + values[hi] * frac;
        };
        median_factor = quantile(factors, 0.5);
        p10_factor = quantile(factors, 0.1);
    }

    weight_validation_finalized = true;
    logger->info(
        "weight validation finalized at fruitloops iter {} after {} contributing iteration(s): "
        "detectors={} ratio_valid={} ratio_upward_valid={} atmosphere_valid={} penalized={} boosted={} "
        "factor_median={} factor_p10={}",
        iter, weight_validation_accumulated_iters, n_uids, n_ratio_valid,
        n_ratio_upward_valid, n_atm_valid, n_penalized, n_boosted,
        median_factor, p10_factor);
}

void PTCProc::subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                            const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> *flags_override) {
    const auto &flags_ref = flags_override ? *flags_override : in.flags.data;
    // cast flags to double and flip 1's and 0's so we can multiply by the data
    auto f = (flags_ref.derived().array().cast <double> ().array() - 1).abs();
    // mean of each detector
    Eigen::RowVectorXd col_mean = (in.scans.data.derived().array()*f).colwise().sum()/
                                   f.colwise().sum();

    // remove nans from completely flagged detectors
    Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

    // subtract mean from data and copy into det matrix
    in.scans.data.noalias() = in.scans.data.derived().rowwise() - dm;

    // subtract kernel mean
    if (in.kernel.data.size()!=0) {
        Eigen::RowVectorXd col_mean = (in.kernel.data.derived().array()*f).colwise().sum()/
                                      f.colwise().sum();

        // remove nans from completely flagged detectors
        Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

        // subtract mean from data and copy into det matrix
        in.kernel.data.noalias() = in.kernel.data.derived().rowwise() - dm;
    }
}

template <class calib_type>
void PTCProc::run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, TCData<TCDataKind::PTC, Eigen::MatrixXd> &out,
                  calib_type &calib, std::string pixel_axes, std::string map_grouping) {

    in.require_native_science_mode_consistent();
    if (!in.native_science_required()) {
        run_impl(in, out, calib, std::move(pixel_axes),
                 std::move(map_grouping));
        return;
    }
    if (&in != &out) {
        throw std::runtime_error(
            "native-required PTC requires one transaction-owned in-place scan object");
    }
    if (weight_validation_is_enabled()) {
        throw std::runtime_error(
            "native-required PTC does not permit dense legacy-UID weight validation");
    }

    const auto &native = in.require_native_scan();
    native.require_compatible(
        in.scans.data.rows(), in.scans.data.cols(), in.index.data);
    if (!native.is_processed_projection() ||
        native.rtc_output_rows().size() !=
            static_cast<std::size_t>(in.scans.data.rows())) {
        throw std::runtime_error(
            "native-required PTC requires exact prior RTC output provenance");
    }
    bool has_typed_relation = false;
    std::shared_ptr<const citlali::pipeline::AptDetectorRelation>
        relation_handle;
    if constexpr (requires(calib_type &candidate) {
                      candidate.has_apt_detector_relation();
                      candidate.apt_detector_relation_handle();
                  }) {
        has_typed_relation = calib.has_apt_detector_relation();
        relation_handle = calib.apt_detector_relation_handle();
    }
    if (!has_typed_relation) {
        throw std::runtime_error(
            "native-required PTC requires a typed artifact-scoped detector relation");
    }
    if (!relation_handle ||
        relation_handle.get() != native.detector_relation_handle().get()) {
        throw std::runtime_error(
            "native-required PTC detector relation is stale or cross-scope");
    }
    if (run_tod_output || write_evals) {
        throw std::runtime_error(
            "native-required PTC output awaits B3 artifact-occurrence provenance synchronization");
    }
    if (second_pass_local.enabled) {
        throw std::runtime_error(
            "native-required PTC second-pass windowing lacks a bounded native-run carrier");
    }
    if (in.scans.data.rows() <= 0 || in.scans.data.cols() <= 0 ||
        in.flags.data.rows() != in.scans.data.rows() ||
        in.flags.data.cols() != in.scans.data.cols() ||
        !in.scans.data.array().isFinite().all()) {
        throw std::runtime_error(
            "native-required PTC candidate must be a finite measured matrix with exact flags");
    }
    if (in.kernel.data.size() != 0 &&
        (in.kernel.data.rows() != in.scans.data.rows() ||
         in.kernel.data.cols() != in.scans.data.cols() ||
         !in.kernel.data.array().isFinite().all())) {
        throw std::runtime_error(
            "native-required PTC kernel candidate differs from its measured matrix");
    }

    using namespace citlali::pipeline;
    const auto n_rows = in.scans.data.rows();
    const auto n_dets = in.scans.data.cols();
    const auto apt_flag = calib.apt.find("flag");
    if (apt_flag == calib.apt.end() ||
        apt_flag->second.size() != n_dets) {
        throw std::runtime_error(
            "native-required PTC detector flag policy differs from the typed detector relation");
    }
    const auto &participants =
        native.alignment_plan_handle()->participant_network_ids();
    std::map<TimestreamNetworkId, Eigen::Index> first_column_by_network;
    for (Eigen::Index detector = 0; detector < n_dets; ++detector) {
        const auto &binding = relation_handle->require_binding(
            relation_handle->binding_reference_for_column(
                static_cast<std::size_t>(detector)));
        first_column_by_network.try_emplace(
            static_cast<TimestreamNetworkId>(binding.network), detector);
    }
    if (first_column_by_network.size() != participants.size()) {
        throw std::runtime_error(
            "native-required PTC typed detector networks differ from the cohort participants");
    }

    std::optional<TimestreamNativeRevision> input_revision;
    std::optional<std::size_t> previous_common_slot;
    for (Eigen::Index row = 0; row < n_rows; ++row) {
        const auto common_slot = native.relational_common_slot(row);
        if (previous_common_slot.has_value() &&
            common_slot <= *previous_common_slot) {
            throw std::runtime_error(
                "native-required PTC relational grouping provenance is not strictly ordered");
        }
        previous_common_slot = common_slot;
        for (const auto network_id : participants) {
            const auto column = first_column_by_network.at(network_id);
            const auto cell = native.require_cell(row, column);
            if (cell.identity.network_id() != network_id) {
                throw std::runtime_error(
                    "native-required PTC cohort identity changed network");
            }
            if (!input_revision.has_value()) {
                input_revision = cell.expected_revision;
            }
            else if (*input_revision != cell.expected_revision) {
                throw std::runtime_error(
                    "native-required PTC input revisions are not coherent");
            }
        }
        for (Eigen::Index detector = 0; detector < n_dets; ++detector) {
            const auto cell = native.require_cell(row, detector);
            const auto &binding = relation_handle->require_binding(
                cell.detector);
            if (binding.detector_column !=
                    static_cast<std::size_t>(detector) ||
                binding.network != cell.identity.network_id()) {
                throw std::runtime_error(
                    "native-required PTC typed detector binding changed during gather");
            }
        }
    }
    if (!input_revision.has_value()) {
        throw std::runtime_error(
            "native-required PTC lacks an input revision");
    }
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> operation_mask =
        in.flags.data;
    if (run_clean && mask_radius_arcsec > 0) {
        operation_mask = mask_region(
            in, calib, pixel_axes, map_grouping, n_rows, n_dets, 0);
    }
    if (run_clean) {
        for (const auto &grouping : cleaner.grouping) {
            const auto effective_grouping =
                Cleaner::is_corr_nw_clean_group(grouping) &&
                        !cleaner.corr_grouping.enabled
                    ? std::string{"nw"}
                    : Cleaner::normalize_group_name(grouping);
            PcaCompatibilityInputs compatibility;
            compatibility.null_model_active_for_operation =
                cleaner.null_model.enabled &&
                cleaner.null_model_enabled_for_group(effective_grouping);
            compatibility.marchenko_pastur_active_for_operation =
                cleaner.marchenko_pastur.enabled &&
                cleaner.marchenko_pastur_enabled_for_group(
                    effective_grouping);
            compatibility.marchenko_pastur_band_requested =
                cleaner.marchenko_pastur.band_low_Hz > 0.0 ||
                cleaner.marchenko_pastur.band_high_Hz > 0.0;
            compatibility.adaptive_selector_active_for_operation =
                cleaner.adaptive_selector.enabled &&
                cleaner.adaptive_selector_enabled_for_group(
                    effective_grouping) &&
                !Cleaner::is_corr_nw_clean_group(effective_grouping);
            const bool has_excluded_cells =
                operation_mask.count() != 0 ||
                (apt_flag->second.array() != 0).any();
            require_native_detector_pca_compatibility(
                classify_native_detector_pca_compatibility(
                    has_excluded_cells, compatibility));
        }
    }

    const auto scan_id = in.index.data;
    RunDiagnosticTransaction diagnostics;
    auto candidate = in;
    run_impl(candidate, candidate, calib, pixel_axes, map_grouping,
             relation_handle.get(), &diagnostics);
    for (Eigen::Index detector = 0; detector < n_dets; ++detector) {
        const bool detector_excluded =
            apt_flag->second(detector) != 0;
        for (Eigen::Index row = 0; row < n_rows; ++row) {
            if (!detector_excluded && !operation_mask(row, detector)) {
                if (!std::isfinite(candidate.scans.data(row, detector)) ||
                    (candidate.kernel.data.size() != 0 &&
                     !std::isfinite(candidate.kernel.data(
                         row, detector)))) {
                    throw std::runtime_error(
                        "native-required PTC candidate produced a nonfinite scientific value");
                }
                continue;
            }
            candidate.scans.data(row, detector) =
                in.scans.data(row, detector);
            candidate.flags.data(row, detector) =
                in.flags.data(row, detector);
            if (candidate.kernel.data.size() != 0) {
                candidate.kernel.data(row, detector) =
                    in.kernel.data(row, detector);
            }
        }
    }
    if (candidate.native_scan.get() != in.native_scan.get() ||
        candidate.native_science_mode != in.native_science_mode) {
        throw std::runtime_error(
            "native-required PTC candidate changed an immutable input authority");
    }
    candidate.native_scan =
        NativeMeasuredScanState::advance_revision(
            candidate.native_scan);
    candidate.require_native_science_mode_consistent();
    candidate.require_native_scan().require_compatible(
        candidate.scans.data.rows(), candidate.scans.data.cols(),
        candidate.index.data);

    auto stage_entry = [scan_id](auto &destination,
                                 const auto &source) {
        if (source.has_value()) {
            destination[scan_id] = *source;
        }
        else {
            destination.erase(scan_id);
        }
    };
    std::lock_guard<std::mutex> lock(*diag_summary_mutex);
    auto pca_candidate = pca_realization_summary_by_scan;
    auto mean_candidate = mean_realization_summary_by_scan;
    auto corr_ids_candidate = corr_nw_group_ids_by_scan;
    auto corr_summary_candidate = corr_nw_summary_by_scan;
    auto adaptive_candidate = adaptive_selector_summary_by_scan;
    stage_entry(pca_candidate, diagnostics.pca_realizations);
    stage_entry(mean_candidate, diagnostics.mean_realization);
    stage_entry(corr_ids_candidate, diagnostics.corr_nw_group_ids);
    stage_entry(corr_summary_candidate, diagnostics.corr_nw_summary);
    stage_entry(adaptive_candidate,
                diagnostics.adaptive_selector_summary);
    in = std::move(candidate);
    pca_realization_summary_by_scan.swap(pca_candidate);
    mean_realization_summary_by_scan.swap(mean_candidate);
    corr_nw_group_ids_by_scan.swap(corr_ids_candidate);
    corr_nw_summary_by_scan.swap(corr_summary_candidate);
    adaptive_selector_summary_by_scan.swap(adaptive_candidate);
}

template <class calib_type>
void PTCProc::run_impl(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, TCData<TCDataKind::PTC, Eigen::MatrixXd> &out,
                       calib_type &calib, std::string pixel_axes, std::string map_grouping,
                       const citlali::pipeline::AptDetectorRelation *typed_relation,
                       RunDiagnosticTransaction *diagnostics) {

    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();
    auto get_runtime_grouping = [&](const std::string &grouping) {
        if (typed_relation == nullptr ||
            !citlali::config::is_network_map_grouping(
                Cleaner::normalize_group_name(grouping))) {
            return get_grouping(grouping, calib, n_dets);
        }
        if (typed_relation->bindings().size() !=
            static_cast<std::size_t>(n_dets)) {
            throw std::runtime_error(
                "native-required PTC typed relation is incomplete for network grouping");
        }
        std::map<Eigen::Index,
                 std::tuple<Eigen::Index, Eigen::Index>> group_limits;
        std::set<Eigen::Index> seen;
        Eigen::Index first = 0;
        while (first < n_dets) {
            const auto &first_binding = typed_relation->require_binding(
                typed_relation->binding_reference_for_column(
                    static_cast<std::size_t>(first)));
            const auto network =
                static_cast<Eigen::Index>(first_binding.network);
            if (!seen.insert(network).second) {
                throw std::runtime_error(
                    "native-required PTC typed network detector columns are not contiguous");
            }
            Eigen::Index past = first + 1;
            while (past < n_dets) {
                const auto &binding = typed_relation->require_binding(
                    typed_relation->binding_reference_for_column(
                        static_cast<std::size_t>(past)));
                if (static_cast<Eigen::Index>(binding.network) != network) {
                    break;
                }
                ++past;
            }
            group_limits.emplace(network, std::make_tuple(first, past));
            first = past;
        }
        return group_limits;
    };
    MeanRealizationSummary mean_realization;
    mean_realization.mean_subtracted = true;
    std::vector<PCARealizationSummary> pca_realizations;

    log_kernel_matrix_diag(logger, "ptc run input", in.kernel.data, in.index.data);

    // subtract mean from data and kernel, optionally masking the source region
    if (run_clean && mask_radius_arcsec > 0) {
        auto mean_flags = mask_region(in, calib, pixel_axes, map_grouping, n_pts, n_dets, 0);
        mean_realization.source_mask_applied = true;
        mean_realization.masked_sample_count =
            static_cast<std::size_t>(mean_flags.array().count());
        mean_realization.mask_digest =
            ptc_realization_matrix_digest(mean_flags);
        subtract_mean(in, &mean_flags);
    }
    else {
        mean_realization.masked_sample_count =
            static_cast<std::size_t>(in.flags.data.array().count());
        mean_realization.mask_digest =
            ptc_realization_matrix_digest(in.flags.data);
        subtract_mean(in);
    }
    log_kernel_matrix_diag(logger, "ptc after subtract_mean", in.kernel.data, in.index.data);

    if (run_clean) {
        logger->info("cleaning");
        // Use a local copy so per-pass state does not leak across concurrent run() calls.
        auto cleaner_local = cleaner;
        // number of samples
        Eigen::Index n_pts = in.scans.data.rows();
        // index for number of cleaning groups in vectors
        Eigen::Index indx = 0;
        std::vector<AdaptiveSelectorDiagSummary> adaptive_summary_scan;
        const bool want_eigs = (run_tod_output || write_evals);
        const bool store_eigs = want_eigs && (cleaner_local.n_calc > 0);
        bool warned_eigs = false;

        // loop through config groupings
        const bool null_model_enabled_global = cleaner_local.null_model.enabled;
        const bool marchenko_pastur_enabled_global = cleaner_local.marchenko_pastur.enabled;
        const bool adaptive_selector_enabled_global = cleaner_local.adaptive_selector.enabled;
        for (const auto & group: cleaner_local.grouping) {
            std::string effective_group = group;
            const bool group_is_corr_nw =
                Cleaner::is_corr_nw_clean_group(group);
            if (group_is_corr_nw && !cleaner_local.corr_grouping.enabled) {
                logger->warn("cleaning group 'corr_nw' requested but clean.corr_grouping.enabled=false; falling back to nw");
                effective_group = "nw";
            }
            const auto effective_group_normalized =
                Cleaner::normalize_group_name(effective_group);
            const bool effective_group_is_corr_nw =
                Cleaner::is_corr_nw_clean_group(effective_group_normalized);
            // optional per-group null-model gating
            const bool null_model_for_group =
                null_model_enabled_global && cleaner_local.null_model_enabled_for_group(effective_group);
            if (null_model_enabled_global && !null_model_for_group) {
                logger->debug("null_model disabled for {} grouping", effective_group);
            }
            const bool marchenko_pastur_for_group =
                marchenko_pastur_enabled_global && cleaner_local.marchenko_pastur_enabled_for_group(effective_group);
            if (marchenko_pastur_enabled_global && !marchenko_pastur_for_group) {
                logger->debug("marchenko_pastur disabled for {} grouping", effective_group);
            }
            const bool adaptive_selector_for_group =
                adaptive_selector_enabled_global &&
                cleaner_local.adaptive_selector_enabled_for_group(effective_group) &&
                !effective_group_is_corr_nw;
            if (adaptive_selector_enabled_global && effective_group_is_corr_nw &&
                cleaner_local.adaptive_selector_enabled_for_group(effective_group)) {
                logger->warn("clean.adaptive_selector currently skips corr_nw sub-groups; using configured fixed cut instead");
            }
            if (adaptive_selector_enabled_global &&
                !cleaner_local.adaptive_selector_enabled_for_group(effective_group)) {
                logger->debug("adaptive_selector disabled for {} grouping", effective_group);
            }

            auto get_forced_limit_index_safe = [&](const auto &scans_view,
                                                   const auto &flags_view,
                                                   const auto &apt_flags_view,
                                                   const std::string &group_name_log,
                                                   const Eigen::Index group_key_log,
                                                   const Eigen::Index arr_index_log) {
                try {
                    if (null_model_for_group) {
                        return cleaner_local.get_null_model_index(scans_view, flags_view, apt_flags_view);
                    }
                    if (marchenko_pastur_for_group) {
                        return cleaner_local.get_marchenko_pastur_index(scans_view, flags_view, apt_flags_view);
                    }
                }
                catch (const std::exception &e) {
                    logger->warn(
                        "adaptive cleaner {} failed for grouping={} key={} array={} n_pts={} n_dets={}; "
                        "falling back to configured PCA cut: {}",
                        cleaner_local.active_cleaner_label(), group_name_log, group_key_log, arr_index_log,
                        scans_view.rows(), scans_view.cols(), e.what());
                }
                return Eigen::Index{-1};
            };

            logger->debug("cleaning with {} grouping", effective_group);

            if (store_eigs) {
                // add current group to eval/evec vectors
                out.evals.data.emplace_back();
                out.evecs.data.emplace_back();
            }
            else if (want_eigs && !warned_eigs) {
                logger->debug("n_calc=0; skipping eval/evec output");
                warned_eigs = true;
            }

            // map of tuples to hold detector limits
            std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

            if (group_is_corr_nw && cleaner_local.corr_grouping.enabled) {
                    Eigen::VectorXi corr_group_ids_scan = Eigen::VectorXi::Constant(in.scans.data.cols(), -1);
                    std::vector<CorrNWDiagSummary> corr_summary_scan;
                    corr_summary_scan.reserve(static_cast<std::size_t>(calib.n_nws));
                    grp_limits = get_runtime_grouping("nw");
                    for (auto const& [key, val] : grp_limits) {
                        const Eigen::Index nw_index = key;
                        const Eigen::Index arr_index = toltec_io.nw_to_array_map[key];
                        auto [start_index, n_dets] = std::make_tuple(std::get<0>(val), std::get<1>(val) - std::get<0>(val));

                        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> masked_flags;
                        if (mask_radius_arcsec > 0) {
                            masked_flags = mask_region(in, calib, pixel_axes, map_grouping, n_pts, n_dets, start_index);
                        }
                        else {
                            masked_flags = in.flags.data.block(0, start_index, n_pts, n_dets);
                        }

                        auto in_scans_block = in.scans.data.block(0, start_index, n_pts, n_dets);
                        auto out_scans_block = out.scans.data.block(0, start_index, n_pts, n_dets);
                        out_scans_block = in_scans_block;

                        auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);

                        if (in.kernel.data.size()!=0) {
                            auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                            auto out_kernel_block = out.kernel.data.block(0, start_index, n_pts, n_dets);
                            out_kernel_block = in_kernel_block;
                        }

                        auto corr_groups = cleaner_local.get_corr_groups(in_scans_block, masked_flags, apt_flags);
                        logger->info("cleaning corr_nw {} groups={} grouped={} ungrouped={} candidates={} used={} step={}",
                                     key, corr_groups.n_groups_final, corr_groups.n_det_grouped, corr_groups.n_det_ungrouped,
                                     corr_groups.n_det_candidates, corr_groups.n_det_used, corr_groups.sample_step);
                        corr_summary_scan.push_back(CorrNWDiagSummary{
                            .nw = nw_index,
                            .n_det_input = corr_groups.n_det_input,
                            .n_det_candidates = corr_groups.n_det_candidates,
                            .n_det_used = corr_groups.n_det_used,
                            .n_det_grouped = corr_groups.n_det_grouped,
                            .n_det_ungrouped = corr_groups.n_det_ungrouped,
                            .n_groups_raw = corr_groups.n_groups_raw,
                            .n_groups_final = corr_groups.n_groups_final,
                            .sample_step = corr_groups.sample_step,
                        });

                        auto extract_scans_cols = [&](const auto &m, const std::vector<Eigen::Index> &cols) {
                            Eigen::MatrixXd out_m(m.rows(), static_cast<Eigen::Index>(cols.size()));
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                out_m.col(c) = m.col(cols[static_cast<std::size_t>(c)]);
                            }
                            return out_m;
                        };
                        auto extract_flag_cols = [&](const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> &m,
                                                     const std::vector<Eigen::Index> &cols) {
                            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> out_m(
                                m.rows(), static_cast<Eigen::Index>(cols.size()));
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                out_m.col(c) = m.col(cols[static_cast<std::size_t>(c)]);
                            }
                            return out_m;
                        };
                        auto extract_apt_cols = [&](const auto &v, const std::vector<Eigen::Index> &cols) {
                            Eigen::VectorXd out_v(static_cast<Eigen::Index>(cols.size()));
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                out_v(c) = v(cols[static_cast<std::size_t>(c)]);
                            }
                            return out_v;
                        };
                        auto scatter_cols = [&](auto &dst, const Eigen::MatrixXd &src, const std::vector<Eigen::Index> &cols) {
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                dst.col(cols[static_cast<std::size_t>(c)]) = src.col(c);
                            }
                        };

                        for (Eigen::Index gidx = 0; gidx < static_cast<Eigen::Index>(corr_groups.groups.size()); ++gidx) {
                            const auto &cols = corr_groups.groups[static_cast<std::size_t>(gidx)];
                            if (cols.size() < 2) {
                                continue;
                            }
                            for (const auto &local_col : cols) {
                                corr_group_ids_scan(start_index + local_col) = gidx;
                            }

                            auto in_scans_sub = extract_scans_cols(in_scans_block, cols);
                            auto out_scans_sub = in_scans_sub;
                            auto flags_sub = extract_flag_cols(masked_flags, cols);
                            auto apt_flags_sub = extract_apt_cols(apt_flags, cols);

                            if (!(apt_flags_sub.array() == 0).any()) {
                                continue;
                            }

                            auto [evals, evecs] = cleaner_local.calc_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_scans_sub, flags_sub, apt_flags_sub, cleaner_local.n_eig_to_cut[arr_index](indx));
                            Eigen::Index forced_limit_index = get_forced_limit_index_safe(
                                in_scans_sub, flags_sub, apt_flags_sub, group, nw_index, arr_index);
                            const Eigen::Index configured_cut =
                                cleaner_local.n_eig_to_cut[arr_index](indx);

                            if (store_eigs) {
                                Eigen::Index n_keep = std::min<Eigen::Index>(cleaner_local.n_calc, evals.size());
                                if (n_keep > 0) {
                                    Eigen::VectorXd ev = evals.head(n_keep);
                                    Eigen::MatrixXd evc = evecs.leftCols(n_keep);
                                    out.evals.data[indx].push_back(std::move(ev));
                                    out.evecs.data[indx].push_back(std::move(evc));
                                }
                            }

                            const Eigen::Index applied_cut =
                                cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_scans_sub, flags_sub, evals, evecs, out_scans_sub,
                                cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
                                group, nw_index, arr_index);
                            pca_realizations.push_back(PCARealizationSummary{
                                effective_group, gidx, arr_index,
                                configured_cut, applied_cut,
                                forced_limit_index,
                                ptc_realization_matrix_digest(evals),
                                ptc_realization_matrix_digest(evecs)});
                            scatter_cols(out_scans_block, out_scans_sub, cols);

                            if (in.kernel.data.size()!=0) {
                                auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                                auto out_kernel_block = out.kernel.data.block(0, start_index, n_pts, n_dets);
                                auto in_kernel_sub = extract_scans_cols(in_kernel_block, cols);
                                auto out_kernel_sub = in_kernel_sub;
                                cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                    in_kernel_sub, flags_sub, evals, evecs, out_kernel_sub,
                                    cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
                                    group, nw_index, arr_index);
                                scatter_cols(out_kernel_block, out_kernel_sub, cols);
                            }
                        }
                    }
                    if (diagnostics != nullptr) {
                        diagnostics->corr_nw_group_ids =
                            std::move(corr_group_ids_scan);
                        diagnostics->corr_nw_summary =
                            std::move(corr_summary_scan);
                    }
                    else {
                        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
                        corr_nw_group_ids_by_scan[in.index.data] = std::move(corr_group_ids_scan);
                        corr_nw_summary_by_scan[in.index.data] = std::move(corr_summary_scan);
                    }
                    indx++;
                    out.status.cleaned = true;
                    log_kernel_matrix_diag(logger, "ptc after clean group=corr_nw", out.kernel.data, in.index.data);
                    continue;
            }

            // use all detectors for cleaning
            if (Cleaner::is_all_clean_group(effective_group_normalized)) {
                grp_limits[0] = std::make_tuple(0,in.scans.data.cols());
            }
            else {
                // get group limits
                grp_limits = get_runtime_grouping(effective_group);
            }
            // loop through cleaning groups
            for (auto const& [key, val] : grp_limits) {
                Eigen::Index arr_index;
                Eigen::Index nw_index = -1;
                // use all detectors
                if (Cleaner::is_all_clean_group(effective_group_normalized)) {
                    arr_index = calib.arrays(0);
                }
                // use network grouping
                else if (citlali::config::is_network_map_grouping(
                             effective_group_normalized)) {
                    nw_index = key;
                    arr_index = toltec_io.nw_to_array_map[key];
                }
                // use array grouping
                else if (citlali::config::is_array_map_grouping(
                             effective_group_normalized)) {
                    arr_index = key;
                }

                // start index and number of detectors
                auto [start_index, n_dets] = std::make_tuple(std::get<0>(val), std::get<1>(val) - std::get<0>(val));

                // matrix for flags so we don't overwrite the raw flags
                Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> masked_flags;

                // mask region if radius is >0
                if (mask_radius_arcsec > 0) {
                    // samples that were masked will be flagged
                    masked_flags = mask_region(in, calib, pixel_axes, map_grouping, n_pts, n_dets, start_index);
                }
                // otherwise just use input flags
                else {
                    masked_flags = in.flags.data.block(0, start_index, n_pts, n_dets);
                }

                auto in_scans_block = in.scans.data.block(0, start_index, n_pts, n_dets);
                auto out_scans_block = out.scans.data.block(0, start_index, n_pts, n_dets);

                // get the block of out scans that corresponds to the current array
                auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);

                // check if any good flags
                if ((apt_flags.array()==0).any()) {
                    logger->info("cleaning {} {}", effective_group, key);
                    const Eigen::Index baseline_k = cleaner_local.n_eig_to_cut[arr_index](indx);
                    Eigen::Index solve_n_eig = baseline_k;
                    if (adaptive_selector_for_group) {
                        auto candidate_ks = cleaner_local.adaptive_selector_candidate_cuts(
                            baseline_k, n_dets - 1);
                        if (!candidate_ks.empty()) {
                            solve_n_eig = candidate_ks.back();
                        }
                    }
                    // calculate eigenvalues and eigenvalues
                    const auto eig_t0 = std::chrono::steady_clock::now();
                    auto [evals, evecs] = cleaner_local.calc_eig_values<timestream::Cleaner::SpectraBackend>(
                        in_scans_block, masked_flags, apt_flags, solve_n_eig);
                    const auto eig_t1 = std::chrono::steady_clock::now();
                    const double eig_solve_msec =
                        std::chrono::duration<double, std::milli>(eig_t1 - eig_t0).count();
                    Eigen::Index forced_limit_index = get_forced_limit_index_safe(
                        in_scans_block, masked_flags, apt_flags, effective_group, key, arr_index);
                    timestream::Cleaner::AdaptiveSelectorResult adaptive_result;
                    if (adaptive_selector_for_group) {
                        adaptive_result = cleaner_local.select_adaptive_cut(
                            in_scans_block, masked_flags, apt_flags, evecs,
                            baseline_k, effective_group, key, arr_index);
                    }

                    if (store_eigs) {
                        // get first n_calc eigenvalues and eigenvectors
                        Eigen::Index n_keep = std::min<Eigen::Index>(cleaner_local.n_calc, evals.size());
                        if (n_keep > 0) {
                            Eigen::VectorXd ev = evals.head(n_keep);
                            Eigen::MatrixXd evc = evecs.leftCols(n_keep);

                            // avoid dumping full matrices in debug; can be huge and unstable
                            const Eigen::Index n_show = std::min<Eigen::Index>(n_keep, 8);
                            logger->debug("evals n={} head({})={}", n_keep, n_show, ev.head(n_show).transpose());
                            logger->debug("evecs shape={}x{} (values omitted)", evc.rows(), evc.cols());

                            // copy evals and evecs to ptcdata
                            out.evals.data[indx].push_back(std::move(ev));
                            out.evecs.data[indx].push_back(std::move(evc));
                        }
                    }

                    Eigen::Index k_to_apply = baseline_k;
                    Eigen::Index applied_cut = 0;
                    if (adaptive_selector_for_group && adaptive_result.used &&
                        adaptive_result.chosen_cleaned_scans.rows() == out_scans_block.rows() &&
                        adaptive_result.chosen_cleaned_scans.cols() == out_scans_block.cols()) {
                        k_to_apply = adaptive_result.chosen_k;
                        out_scans_block = adaptive_result.chosen_cleaned_scans;
                        applied_cut = std::max<Eigen::Index>(
                            0, std::min<Eigen::Index>(
                                   k_to_apply, evecs.cols()));
                    }
                    else if (adaptive_selector_for_group && adaptive_result.used) {
                        k_to_apply = adaptive_result.chosen_k;
                        applied_cut =
                            cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                            in_scans_block, masked_flags, evals, evecs, out_scans_block,
                            k_to_apply, forced_limit_index,
                            effective_group, nw_index, arr_index);
                    }
                    else {
                        applied_cut =
                            cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                            in_scans_block, masked_flags, evals, evecs, out_scans_block,
                            baseline_k, forced_limit_index,
                            effective_group, nw_index, arr_index);
                    }

                    pca_realizations.push_back(PCARealizationSummary{
                        effective_group, key, arr_index, baseline_k,
                        applied_cut, forced_limit_index,
                        ptc_realization_matrix_digest(evals),
                        ptc_realization_matrix_digest(evecs)});

                    if (adaptive_selector_for_group) {
                        const double total_selector_msec = eig_solve_msec +
                            (std::isfinite(adaptive_result.candidate_eval_msec)
                                 ? adaptive_result.candidate_eval_msec
                                 : 0.0);
                        logger->info(
                            "adaptive_selector timing grouping={} key={} nw={} baseline_k={} chosen_k={} eig_ms={} candidate_ms={} total_ms={} margin={}",
                            effective_group, key, nw_index, baseline_k, k_to_apply,
                            eig_solve_msec, adaptive_result.candidate_eval_msec,
                            total_selector_msec, adaptive_result.score_margin);
                        adaptive_summary_scan.push_back(AdaptiveSelectorDiagSummary{
                            .nw = nw_index,
                            .n_det_input = in_scans_block.cols(),
                            .n_det_used = adaptive_result.chosen_diag.n_det_used,
                            .n_time_used = adaptive_result.chosen_diag.n_time_used,
                            .sample_step = adaptive_result.chosen_diag.sample_step,
                            .baseline_k = baseline_k,
                            .chosen_k = k_to_apply,
                            .runnerup_k = adaptive_result.runnerup_k,
                            .n_candidates = adaptive_result.n_candidates,
                            .selector_used = adaptive_result.used ? 1 : 0,
                            .selector_fallback = adaptive_result.fallback ? 1 : 0,
                            .chosen_score = adaptive_result.chosen_score,
                            .runnerup_score = adaptive_result.runnerup_score,
                            .score_margin = adaptive_result.score_margin,
                            .chosen_med_abs_corr = adaptive_result.chosen_diag.med_abs_corr,
                            .chosen_cm_low_mid_ratio = adaptive_result.chosen_diag.cm_low_mid_ratio,
                            .chosen_tail4_binom_z = adaptive_result.chosen_diag.tail4_binom_z,
                            .chosen_top_mode_frac = adaptive_result.chosen_diag.top_mode_frac,
                            .eig_solve_msec = eig_solve_msec,
                            .candidate_eval_msec = adaptive_result.candidate_eval_msec,
                            .total_msec = total_selector_msec,
                        });
                    }

                    if (in.kernel.data.size()!=0) {
                        // check if any good flags
                            logger->debug("cleaning kernel");
                            auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                            auto out_kernel_block = out.kernel.data.block(0, start_index, n_pts, n_dets);

                            // remove eigenvalues from the kernel and reconstruct the tod
                            cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_kernel_block, masked_flags, evals, evecs, out_kernel_block,
                                k_to_apply, forced_limit_index,
                                effective_group, nw_index, arr_index);
                    }
                }
                // otherwise just copy the data
                else {
                    logger->debug("no good detectors found. skipping clean.");
                    // copy scans
                    out.scans.data.block(0, start_index, n_pts, n_dets) = in.scans.data.block(0, start_index, n_pts, n_dets);
                    // copy kernel
                    if (in.kernel.data.size()!=0) {
                        out.kernel.data.block(0, start_index, n_pts, n_dets) = in.kernel.data.block(0, start_index, n_pts, n_dets);
                    }
                }
            }
            indx++;
            // set as cleaned
            out.status.cleaned = true;
            log_kernel_matrix_diag(logger, "ptc after clean group=" + effective_group, out.kernel.data, in.index.data);
        }
        if (!adaptive_summary_scan.empty()) {
            if (diagnostics != nullptr) {
                diagnostics->adaptive_selector_summary =
                    std::move(adaptive_summary_scan);
            }
            else {
                std::lock_guard<std::mutex> lock(*diag_summary_mutex);
                adaptive_selector_summary_by_scan[in.index.data] =
                    std::move(adaptive_summary_scan);
            }
        }
    }

    if (second_pass_local.enabled) {
        if (!run_clean) {
            logger->warn("processed_time_chunk.flagging.second_pass_local enabled but clean.enabled=false; skipping PTC second-pass residual flagging");
        }
        else {
            apply_second_pass_local(out, calib, pixel_axes, map_grouping);
        }
    }
    if (diagnostics != nullptr) {
        diagnostics->mean_realization = std::move(mean_realization);
        diagnostics->pca_realizations = std::move(pca_realizations);
    }
    else {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        mean_realization_summary_by_scan[in.index.data] =
            std::move(mean_realization);
        pca_realization_summary_by_scan[in.index.data] =
            std::move(pca_realizations);
    }
    log_kernel_matrix_diag(logger, "ptc run output", out.kernel.data, in.index.data);
}

template <class calib_type>
void PTCProc::apply_second_pass_local(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_type &calib,
                                      std::string pixel_axes, std::string map_grouping) {

    engine_utils::reject_native_science_consumer(
        in, "PTC second-pass windowing across native run support");

    struct DetectorEventRow {
        Eigen::Index nw = -1;
        Eigen::Index uid = -1;
        Eigen::Index det_index = -1;
        TransientEventKind kind = TransientEventKind::unknown;
        Eigen::Index sample = -1;
        double score = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index start_sample = -1;
        Eigen::Index end_sample = -1;
        Eigen::Index width_samples = 0;
        double baseline_shift_z = std::numeric_limits<double>::quiet_NaN();
        double dt_sec = 1.0;
    };

    struct EventCluster {
        Eigen::Index sample = -1;
        Eigen::Index start_sample = -1;
        Eigen::Index end_sample = -1;
        double peak_score = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index top_uid = -1;
        TransientEventKind top_kind = TransientEventKind::unknown;
        Eigen::Index n_detector_events = 0;
        Eigen::Index n_detectors = 0;
        Eigen::Index n_source_protected_events = 0;
        Eigen::Index n_source_protected_detectors = 0;
        std::vector<DetectorEventRow> rows;
    };

    const Eigen::Index n_pts = in.scans.data.rows();
    const Eigen::Index n_dets_total = in.scans.data.cols();
    if (n_pts < 3 || n_dets_total <= 0) {
        return;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_protection_mask;
    Eigen::Index n_source_detectors = 0;
    if (second_pass_local.source_protection_enabled) {
        auto [source_mask, source_info] = engine_utils::calc_source_protection_mask(
            in, calib.apt, pixel_axes, map_grouping,
            "map_center_radius", second_pass_local.source_protection_radius_arcsec);
        source_protection_mask = std::move(source_mask);
        n_source_detectors = source_info.detectors_with_source;
        logger->debug(
            "processed_time_chunk.flagging.second_pass_local source protection scan={} mode={} radius_arcsec={:.4g} protected_samples={} detectors_with_source={}",
            in.index.data, source_info.mode, source_info.radius_arcsec,
            source_info.protected_samples, n_source_detectors);
    }
    const bool have_source_protection =
        second_pass_local.source_protection_enabled &&
        source_protection_mask.rows() == n_pts &&
        source_protection_mask.cols() == n_dets_total;

    const double fsmp = (cleaner.sample_rate_Hz > 0.0) ? cleaner.sample_rate_Hz : 1.0;
    const double dt_sec = 1.0 / fsmp;
    int smooth_window = static_cast<int>(std::lround(second_pass_local.baseline_window_sec * fsmp));
    smooth_window = std::max(3, smooth_window);
    if ((smooth_window % 2) == 0) {
        ++smooth_window;
    }
    const Eigen::Index raw_gate_half_window = std::max<Eigen::Index>(
        4, static_cast<Eigen::Index>(std::llround(second_pass_local.raw_window_sec * fsmp)));
    const Eigen::Index raw_max_width_samples = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(std::llround(second_pass_local.raw_max_width_sec * fsmp)));
    const Eigen::Index delta_gate_half_window = std::max<Eigen::Index>(
        4, static_cast<Eigen::Index>(std::llround(second_pass_local.delta_window_sec * fsmp)));
    const Eigen::Index delta_max_width_samples = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(std::llround(second_pass_local.delta_max_width_sec * fsmp)));
    const Eigen::Index merge_samples = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(std::llround(second_pass_local.merge_within_detector_sec * fsmp)));
    const Eigen::Index cluster_samples = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(std::llround(second_pass_local.cluster_events_sec * fsmp)));

    auto robust_center_scale = [&](const Eigen::VectorXd &x,
                                   const Eigen::Matrix<bool, Eigen::Dynamic, 1> &flag_mask) {
        std::vector<double> vals;
        vals.reserve(static_cast<std::size_t>(x.size()));
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            if (!flag_mask(i) && std::isfinite(x(i))) {
                vals.push_back(x(i));
            }
        }
        if (vals.size() < 8) {
            vals.clear();
            vals.reserve(static_cast<std::size_t>(x.size()));
            for (Eigen::Index i = 0; i < x.size(); ++i) {
                if (std::isfinite(x(i))) {
                    vals.push_back(x(i));
                }
            }
        }
        if (vals.size() < 8) {
            return std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN());
        }
        Eigen::Map<const Eigen::VectorXd> vals_map(vals.data(), static_cast<Eigen::Index>(vals.size()));
        const double med = tula::alg::median(vals_map);
        Eigen::VectorXd abs_dev = (vals_map.array() - med).abs();
        double sigma = 1.4826 * tula::alg::median(abs_dev);
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            sigma = engine_utils::calc_std_dev(abs_dev);
        }
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            return std::make_pair(med, std::numeric_limits<double>::quiet_NaN());
        }
        return std::make_pair(med, sigma);
    };

    auto characterize_event =
        [&](const Eigen::VectorXd &resid,
            const Eigen::VectorXd &metric_abs_z,
            const Eigen::Matrix<bool, Eigen::Dynamic, 1> &base_flags,
            Eigen::Index metric_peak_index,
            Eigen::Index peak_sample,
            Eigen::Index gate_half_window,
            Eigen::Index max_width_samples,
            double half_peak_frac,
            double resid_sigma,
            double max_step_shift_z,
            TransientEventKind kind,
            bool metric_is_delta) {
            TransientEvent event;
            event.kind = kind;
            event.sample = static_cast<int>(peak_sample);
            if (!(std::isfinite(resid_sigma) && resid_sigma > 0.0) ||
                metric_peak_index < 0 || metric_peak_index >= metric_abs_z.size() ||
                peak_sample < 0 || peak_sample >= resid.size()) {
                return event;
            }

            const double peak_z = metric_abs_z(metric_peak_index);
            if (!std::isfinite(peak_z) || peak_z <= 0.0) {
                return event;
            }

            event.score = peak_z;
            if (kind == TransientEventKind::raw_like) {
                event.peak_abs_z = peak_z;
            }
            else if (kind == TransientEventKind::delta_like) {
                event.peak_delta_abs_z = peak_z;
            }

            const Eigen::Index left_bound = std::max<Eigen::Index>(0, metric_peak_index - gate_half_window);
            const Eigen::Index right_bound =
                std::min<Eigen::Index>(metric_abs_z.size() - 1, metric_peak_index + gate_half_window);
            const double width_thresh =
                std::max(half_peak_frac * peak_z, std::min(peak_z, 1.5));

            Eigen::Index left = metric_peak_index;
            while (left - 1 >= left_bound &&
                   std::isfinite(metric_abs_z(left - 1)) &&
                   metric_abs_z(left - 1) >= width_thresh) {
                --left;
            }
            Eigen::Index right = metric_peak_index;
            while (right + 1 <= right_bound &&
                   std::isfinite(metric_abs_z(right + 1)) &&
                   metric_abs_z(right + 1) >= width_thresh) {
                ++right;
            }

            const Eigen::Index event_start = std::max<Eigen::Index>(0, left);
            const Eigen::Index event_end = metric_is_delta
                ? std::min<Eigen::Index>(resid.size() - 1, right + 1)
                : std::min<Eigen::Index>(resid.size() - 1, right);
            const Eigen::Index width_samples = std::max<Eigen::Index>(0, event_end - event_start + 1);
            event.start_sample = static_cast<int>(event_start);
            event.end_sample = static_cast<int>(event_end);
            event.width_samples = static_cast<double>(width_samples);

            const Eigen::Index pre_lo = std::max<Eigen::Index>(0, peak_sample - gate_half_window);
            const Eigen::Index pre_hi = std::max<Eigen::Index>(pre_lo, peak_sample - (metric_is_delta ? 2 : 1));
            const Eigen::Index post_lo = std::min<Eigen::Index>(resid.size(), peak_sample + 2);
            const Eigen::Index post_hi = std::min<Eigen::Index>(resid.size(), peak_sample + gate_half_window + 1);
            std::vector<double> pre_vals;
            std::vector<double> post_vals;
            for (Eigen::Index i = pre_lo; i < pre_hi; ++i) {
                if (!base_flags(i) && std::isfinite(resid(i))) {
                    pre_vals.push_back(resid(i));
                }
            }
            for (Eigen::Index i = post_lo; i < post_hi; ++i) {
                if (!base_flags(i) && std::isfinite(resid(i))) {
                    post_vals.push_back(resid(i));
                }
            }
            if (pre_vals.size() >= 4 && post_vals.size() >= 4) {
                Eigen::Map<const Eigen::VectorXd> pre_map(pre_vals.data(), static_cast<Eigen::Index>(pre_vals.size()));
                Eigen::Map<const Eigen::VectorXd> post_map(post_vals.data(), static_cast<Eigen::Index>(post_vals.size()));
                const double pre_med = tula::alg::median(pre_map);
                const double post_med = tula::alg::median(post_map);
                event.baseline_shift_z = std::abs(post_med - pre_med) / resid_sigma;
            }

            const bool compact_event = width_samples <= max_width_samples;
            const bool baseline_ok =
                std::isfinite(event.baseline_shift_z) &&
                event.baseline_shift_z <= max_step_shift_z;
            const bool high_score_override =
                std::isfinite(second_pass_local.high_score_event_override) &&
                second_pass_local.high_score_event_override > 0.0 &&
                std::isfinite(event.score) &&
                event.score >= second_pass_local.high_score_event_override;
            event.accepted = compact_event && (baseline_ok || high_score_override);
            return event;
        };

    auto cluster_runs = [](const std::vector<Eigen::Index> &indices) {
        std::vector<std::pair<Eigen::Index, Eigen::Index>> runs;
        if (indices.empty()) {
            return runs;
        }
        Eigen::Index lo = indices.front();
        Eigen::Index hi = indices.front();
        for (std::size_t i = 1; i < indices.size(); ++i) {
            const auto idx = indices[i];
            if (idx <= hi + 1) {
                hi = idx;
            }
            else {
                runs.emplace_back(lo, hi);
                lo = idx;
                hi = idx;
            }
        }
        runs.emplace_back(lo, hi);
        return runs;
    };

    auto median_sample = [](std::vector<Eigen::Index> samples) {
        if (samples.empty()) {
            return Eigen::Index{-1};
        }
        const auto mid = samples.begin() + static_cast<std::ptrdiff_t>(samples.size() / 2);
        std::nth_element(samples.begin(), mid, samples.end());
        return *mid;
    };

    auto merge_detector_rows = [&](std::vector<DetectorEventRow> rows) {
        std::vector<DetectorEventRow> merged;
        if (rows.empty()) {
            return merged;
        }
        std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
            if (a.uid != b.uid) {
                return a.uid < b.uid;
            }
            return a.sample < b.sample;
        });
        std::vector<DetectorEventRow> group{rows.front()};
        auto flush = [&](const std::vector<DetectorEventRow> &current) {
            auto best_it = std::max_element(current.begin(), current.end(), [](const auto &a, const auto &b) {
                return a.score < b.score;
            });
            DetectorEventRow out = *best_it;
            std::vector<Eigen::Index> samples;
            samples.reserve(current.size());
            Eigen::Index start_sample = current.front().start_sample;
            Eigen::Index end_sample = current.front().end_sample;
            for (const auto &row : current) {
                samples.push_back(row.sample);
                start_sample = std::min(start_sample, row.start_sample);
                end_sample = std::max(end_sample, row.end_sample);
            }
            out.start_sample = start_sample;
            out.end_sample = end_sample;
            out.sample = median_sample(samples);
            out.width_samples = out.end_sample - out.start_sample + 1;
            merged.push_back(out);
        };
        for (std::size_t i = 1; i < rows.size(); ++i) {
            if (rows[i].uid == group.back().uid && rows[i].sample <= group.back().sample + merge_samples) {
                group.push_back(rows[i]);
            }
            else {
                flush(group);
                group.assign(1, rows[i]);
            }
        }
        flush(group);
        return merged;
    };

    auto cluster_event_rows = [&](std::vector<DetectorEventRow> rows) {
        std::vector<EventCluster> clusters;
        if (rows.empty()) {
            return clusters;
        }
        std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
            return a.sample < b.sample;
        });
        std::vector<DetectorEventRow> group{rows.front()};
        auto flush = [&](const std::vector<DetectorEventRow> &current) {
            auto best_it = std::max_element(current.begin(), current.end(), [](const auto &a, const auto &b) {
                return a.score < b.score;
            });
            EventCluster cluster;
            cluster.peak_score = best_it->score;
            cluster.top_uid = best_it->uid;
            cluster.top_kind = best_it->kind;
            cluster.rows = current;
            cluster.n_detector_events = static_cast<Eigen::Index>(current.size());
            std::vector<Eigen::Index> samples;
            std::unordered_set<Eigen::Index> uids;
            cluster.start_sample = current.front().start_sample;
            cluster.end_sample = current.front().end_sample;
            samples.reserve(current.size());
            for (const auto &row : current) {
                samples.push_back(row.sample);
                uids.insert(row.uid);
                cluster.start_sample = std::min(cluster.start_sample, row.start_sample);
                cluster.end_sample = std::max(cluster.end_sample, row.end_sample);
            }
            cluster.sample = median_sample(samples);
            cluster.n_detectors = static_cast<Eigen::Index>(uids.size());
            clusters.push_back(cluster);
        };
        for (std::size_t i = 1; i < rows.size(); ++i) {
            Eigen::Index group_max_sample = group.front().sample;
            for (const auto &row : group) {
                group_max_sample = std::max(group_max_sample, row.sample);
            }
            if (rows[i].sample <= group_max_sample + cluster_samples) {
                group.push_back(rows[i]);
            }
            else {
                flush(group);
                group.assign(1, rows[i]);
            }
        }
        flush(group);
        std::sort(clusters.begin(), clusters.end(), [](const auto &a, const auto &b) {
            if (a.peak_score != b.peak_score) {
                return a.peak_score > b.peak_score;
            }
            if (a.sample != b.sample) {
                return a.sample < b.sample;
            }
            return a.top_uid < b.top_uid;
        });
        return clusters;
    };

    auto analyze_detector =
        [&](const Eigen::VectorXd &signal,
            const Eigen::Matrix<bool, Eigen::Dynamic, 1> &base_flags) {
            std::vector<TransientEvent> events;
            Eigen::Matrix<bool, Eigen::Dynamic, 1> final_flags =
                Eigen::Matrix<bool, Eigen::Dynamic, 1>::Zero(n_pts);
            Eigen::VectorXd resid_z = Eigen::VectorXd::Constant(
                n_pts, std::numeric_limits<double>::quiet_NaN());

            Eigen::Index n_good = 0;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (!base_flags(i) && std::isfinite(signal(i))) {
                    ++n_good;
                }
            }
            const double good_frac = static_cast<double>(n_good) / static_cast<double>(n_pts);
            if (good_frac < second_pass_local.min_good_frac) {
                return std::make_tuple(events, final_flags, resid_z);
            }

            auto [med, sigma] = robust_center_scale(signal, base_flags);
            if (!std::isfinite(sigma) || sigma <= 0.0) {
                return std::make_tuple(events, final_flags, resid_z);
            }

            Eigen::VectorXd baseline_input = signal;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (base_flags(i) || !std::isfinite(baseline_input(i))) {
                    baseline_input(i) = med;
                }
            }
            Eigen::VectorXd smooth = Eigen::VectorXd::Zero(n_pts);
            engine_utils::smooth<engine_utils::SmoothType::edge_truncate>(
                baseline_input, smooth, smooth_window);
            Eigen::VectorXd resid = signal - smooth;

            auto [resid_med, resid_sigma] = robust_center_scale(resid, base_flags);
            if (!std::isfinite(resid_sigma) || resid_sigma <= 0.0) {
                return std::make_tuple(events, final_flags, resid_z);
            }

            Eigen::VectorXd abs_dev = (resid.array() - resid_med).abs();
            Eigen::VectorXd local_abs_z = abs_dev / resid_sigma;
            resid_z = resid / resid_sigma;
            const double raw_candidate_z =
                second_pass_local.raw_candidate_rel_sigma_scale *
                second_pass_local.sigma_scale *
                second_pass_local.min_spike_sigma;

            Eigen::Matrix<bool, Eigen::Dynamic, 1> raw_flags =
                Eigen::Matrix<bool, Eigen::Dynamic, 1>::Zero(n_pts);
            std::vector<Eigen::Index> candidate_samples;
            candidate_samples.reserve(static_cast<std::size_t>(n_pts));
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (!base_flags(i) && std::isfinite(local_abs_z(i)) && local_abs_z(i) > raw_candidate_z) {
                    candidate_samples.push_back(i);
                }
            }
            for (const auto &[lo, hi] : cluster_runs(candidate_samples)) {
                Eigen::Index best_sample = lo;
                double best_z = -1.0;
                for (Eigen::Index sample = lo; sample <= hi; ++sample) {
                    if (std::isfinite(local_abs_z(sample)) && local_abs_z(sample) > best_z) {
                        best_z = local_abs_z(sample);
                        best_sample = sample;
                    }
                }
                auto event = characterize_event(
                    resid, local_abs_z, base_flags, best_sample, best_sample,
                    raw_gate_half_window, raw_max_width_samples,
                    second_pass_local.raw_half_peak_frac, resid_sigma,
                    second_pass_local.max_step_shift_z,
                    TransientEventKind::raw_like, false);
                if (event.accepted) {
                    raw_flags.segment(event.start_sample, event.end_sample - event.start_sample + 1).setOnes();
                    events.push_back(event);
                }
            }

            final_flags = raw_flags;
            std::vector<double> local_delta_vals;
            std::vector<Eigen::Index> local_delta_edges;
            local_delta_vals.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(n_pts - 1, 0)));
            local_delta_edges.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(n_pts - 1, 0)));
            for (Eigen::Index i = 0; i < n_pts - 1; ++i) {
                if (base_flags(i) || base_flags(i + 1) || raw_flags(i) || raw_flags(i + 1)) {
                    continue;
                }
                if (!std::isfinite(resid(i)) || !std::isfinite(resid(i + 1))) {
                    continue;
                }
                local_delta_vals.push_back(resid(i + 1) - resid(i));
                local_delta_edges.push_back(i);
            }

            if (local_delta_vals.size() >= 8) {
                Eigen::Map<const Eigen::VectorXd> delta_map(
                    local_delta_vals.data(), static_cast<Eigen::Index>(local_delta_vals.size()));
                const double delta_med = tula::alg::median(delta_map);
                Eigen::VectorXd delta_abs_dev = (delta_map.array() - delta_med).abs();
                double delta_sigma = 1.4826 * tula::alg::median(delta_abs_dev);
                if (!std::isfinite(delta_sigma) || delta_sigma <= 0.0) {
                    delta_sigma = engine_utils::calc_std_dev(delta_abs_dev);
                }
                if (std::isfinite(delta_sigma) && delta_sigma > 0.0) {
                    Eigen::VectorXd local_delta_abs_z =
                        Eigen::VectorXd::Constant(std::max<Eigen::Index>(n_pts - 1, 0),
                                                  std::numeric_limits<double>::quiet_NaN());
                    std::vector<Eigen::Index> candidate_edges;
                    const double local_delta_cutoff =
                        second_pass_local.delta_sigma_scale *
                        second_pass_local.min_spike_sigma * delta_sigma;
                    for (std::size_t i = 0; i < local_delta_edges.size(); ++i) {
                        const auto edge = local_delta_edges[i];
                        const double abs_delta = std::abs(local_delta_vals[i] - delta_med);
                        local_delta_abs_z(edge) = abs_delta / delta_sigma;
                        if (abs_delta > local_delta_cutoff) {
                            candidate_edges.push_back(edge);
                        }
                    }
                    for (const auto &[lo, hi] : cluster_runs(candidate_edges)) {
                        Eigen::Index best_edge = lo;
                        double best_z = -1.0;
                        for (Eigen::Index edge = lo; edge <= hi; ++edge) {
                            if (edge >= 0 && edge < local_delta_abs_z.size() &&
                                std::isfinite(local_delta_abs_z(edge)) &&
                                local_delta_abs_z(edge) > best_z) {
                                best_z = local_delta_abs_z(edge);
                                best_edge = edge;
                            }
                        }
                        auto event = characterize_event(
                            resid, local_delta_abs_z, base_flags, best_edge, best_edge + 1,
                            delta_gate_half_window, delta_max_width_samples,
                            second_pass_local.delta_half_peak_frac, resid_sigma,
                            second_pass_local.max_step_shift_z,
                            TransientEventKind::delta_like, true);
                        if (event.accepted) {
                            final_flags(best_edge) = true;
                            if (best_edge + 1 < n_pts) {
                                final_flags(best_edge + 1) = true;
                            }
                            events.push_back(event);
                        }
                    }
                }
            }

            return std::make_tuple(events, final_flags, resid_z);
        };

    auto group_limits = get_grouping("nw", calib, in.scans.data.cols());
    std::vector<SecondPassDiagSummary> summaries;
    summaries.reserve(group_limits.size());
    Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic> added_flags_out;
    if (run_tod_output) {
        added_flags_out = Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_pts, n_dets_total);
    }

    for (const auto &[key, val] : group_limits) {
        const Eigen::Index nw_index = key;
        const auto start_index = std::get<0>(val);
        const auto n_dets = std::get<1>(val) - std::get<0>(val);
        if (n_dets <= 0) {
            continue;
        }

        const auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);
        auto flags_block = in.flags.data.block(0, start_index, n_pts, n_dets);
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> existing_flags_block = flags_block;
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_flags_block;
        if (have_source_protection) {
            source_flags_block = source_protection_mask.block(0, start_index, n_pts, n_dets);
        }
        std::unordered_map<Eigen::Index, Eigen::Index> local_det_lookup;
        local_det_lookup.reserve(static_cast<std::size_t>(n_dets));

        std::vector<DetectorEventRow> detector_rows;
        double residual_peak = std::numeric_limits<double>::quiet_NaN();
        int residual_peak_uid = kTransientFillInt;

        for (Eigen::Index local_j = 0; local_j < n_dets; ++local_j) {
            const Eigen::Index det_col = start_index + local_j;
            local_det_lookup[det_col] = local_j;
            if (apt_flags(local_j) != 0) {
                continue;
            }
            auto signal = in.scans.data.col(det_col);
            Eigen::Matrix<bool, Eigen::Dynamic, 1> det_flags = in.flags.data.col(det_col);
            Eigen::Matrix<bool, Eigen::Dynamic, 1> base_flags = det_flags;
            if (have_source_protection) {
                auto source_flags = source_protection_mask.col(det_col);
                if ((source_flags.array() == true).any()) {
                    base_flags = (base_flags.array() || source_flags.array()).matrix();
                }
            }
            auto [events, det_prop_flags, det_resid_z] = analyze_detector(signal, base_flags);

            bool det_has_resid = false;
            double det_peak = std::numeric_limits<double>::quiet_NaN();
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (!base_flags(i) && std::isfinite(det_resid_z(i))) {
                    const double v = std::abs(det_resid_z(i));
                    if (!det_has_resid || v > det_peak) {
                        det_peak = v;
                        det_has_resid = true;
                    }
                }
            }
            if (det_has_resid && (!std::isfinite(residual_peak) || det_peak > residual_peak)) {
                residual_peak = det_peak;
                residual_peak_uid = static_cast<int>(calib.apt["uid"](det_col));
            }

            for (const auto &event : events) {
                detector_rows.push_back(DetectorEventRow{
                    .nw = nw_index,
                    .uid = static_cast<Eigen::Index>(calib.apt["uid"](det_col)),
                    .det_index = det_col,
                    .kind = event.kind,
                    .sample = event.sample,
                    .score = event.score,
                    .start_sample = event.start_sample,
                    .end_sample = event.end_sample,
                    .width_samples = std::max(0, event.end_sample - event.start_sample + 1),
                    .baseline_shift_z = event.baseline_shift_z,
                    .dt_sec = dt_sec,
                });
            }
        }

        auto merged_events = merge_detector_rows(detector_rows);
        auto clusters = cluster_event_rows(merged_events);
        std::vector<EventCluster> candidate_clusters;
        for (const auto &cluster : clusters) {
            if (cluster.n_detectors >= second_pass_local.min_cluster_detectors ||
                cluster.peak_score >= second_pass_local.high_score_cluster_override) {
                candidate_clusters.push_back(cluster);
            }
        }
        std::sort(candidate_clusters.begin(), candidate_clusters.end(), [](const auto &a, const auto &b) {
            if (a.peak_score != b.peak_score) {
                return a.peak_score > b.peak_score;
            }
            if (a.sample != b.sample) {
                return a.sample < b.sample;
            }
            return a.top_uid < b.top_uid;
        });

        const bool busy_network_vetoed =
            static_cast<int>(candidate_clusters.size()) > second_pass_local.max_auto_flag_clusters_per_network;
        auto row_source_protected = [&](const DetectorEventRow &row) {
            if (!have_source_protection) {
                return false;
            }
            const auto it = local_det_lookup.find(row.det_index);
            if (it == local_det_lookup.end()) {
                return false;
            }
            const Eigen::Index start =
                std::max<Eigen::Index>(0, row.start_sample);
            const Eigen::Index stop =
                std::min<Eigen::Index>(n_pts - 1, row.end_sample);
            for (Eigen::Index sample = start; sample <= stop; ++sample) {
                if (source_flags_block(sample, it->second)) {
                    return true;
                }
            }
            return false;
        };

        for (auto &cluster : candidate_clusters) {
            std::unordered_set<Eigen::Index> protected_uids;
            for (const auto &row : cluster.rows) {
                if (row_source_protected(row)) {
                    ++cluster.n_source_protected_events;
                    protected_uids.insert(row.uid);
                }
            }
            cluster.n_source_protected_detectors =
                static_cast<Eigen::Index>(protected_uids.size());
        }

        auto has_off_source_rows = [](const EventCluster &cluster) {
            return cluster.n_detector_events > cluster.n_source_protected_events;
        };

        auto high_confidence_cluster = [&](const EventCluster &cluster) {
            const bool high_cluster_score =
                std::isfinite(second_pass_local.high_score_cluster_override) &&
                second_pass_local.high_score_cluster_override > 0.0 &&
                std::isfinite(cluster.peak_score) &&
                cluster.peak_score >= second_pass_local.high_score_cluster_override;
            const bool multi_detector_cluster =
                cluster.n_detectors >= second_pass_local.min_cluster_detectors;
            return high_cluster_score || multi_detector_cluster;
        };

        const Eigen::Index accepted_cluster_cap =
            std::max<Eigen::Index>(0, second_pass_local.max_auto_flag_clusters_per_network);
        std::vector<EventCluster> accepted_clusters;
        accepted_clusters.reserve(static_cast<std::size_t>(std::min<Eigen::Index>(
            accepted_cluster_cap, static_cast<Eigen::Index>(candidate_clusters.size()))));
        Eigen::Index n_rejected_clusters = 0;
        Eigen::Index n_rejected_events = 0;
        Eigen::Index n_source_protected_clusters = 0;
        Eigen::Index n_source_protected_events = 0;

        for (const auto &cluster : candidate_clusters) {
            if (cluster.n_source_protected_events > 0) {
                ++n_source_protected_clusters;
                n_source_protected_events += cluster.n_source_protected_events;
            }

            const bool accept_cluster =
                has_off_source_rows(cluster) &&
                (!busy_network_vetoed ||
                 (second_pass_local.selective_busy_network_acceptance_enabled &&
                  high_confidence_cluster(cluster))) &&
                static_cast<Eigen::Index>(accepted_clusters.size()) < accepted_cluster_cap;

            if (accept_cluster) {
                accepted_clusters.push_back(cluster);
            }
            else {
                ++n_rejected_clusters;
                n_rejected_events +=
                    std::max<Eigen::Index>(0, cluster.n_detector_events -
                                              cluster.n_source_protected_events);
            }
        }

        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> accepted_flags_block =
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_pts, n_dets);
        std::vector<DetectorEventRow> accepted_rows;
        accepted_rows.reserve(detector_rows.size());
        for (const auto &cluster : accepted_clusters) {
            for (const auto &row : cluster.rows) {
                if (row_source_protected(row)) {
                    continue;
                }
                accepted_rows.push_back(row);
                const auto it = local_det_lookup.find(row.det_index);
                if (it == local_det_lookup.end()) {
                    continue;
                }
                accepted_flags_block.block(
                    row.start_sample, it->second, row.end_sample - row.start_sample + 1, 1).setOnes();
            }
        }
        if (have_source_protection) {
            accepted_flags_block =
                (accepted_flags_block.array() && (source_flags_block.array() == false)).matrix();
        }
        std::sort(accepted_rows.begin(), accepted_rows.end(), [](const auto &a, const auto &b) {
            return a.score > b.score;
        });

        flags_block = existing_flags_block.array() || accepted_flags_block.array();
        if (run_tod_output) {
            added_flags_out.block(0, start_index, n_pts, n_dets) =
                accepted_flags_block.cast<signed char>();
        }

        Eigen::Index n_det_with_added_flags = 0;
        for (Eigen::Index j = 0; j < n_dets; ++j) {
            bool any = false;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (accepted_flags_block(i, j)) {
                    any = true;
                    break;
                }
            }
            if (any) {
                ++n_det_with_added_flags;
            }
        }

        SecondPassDiagSummary summary;
        summary.nw = nw_index;
        summary.n_det = n_dets;
        summary.n_pts = n_pts;
        summary.n_merged_events_total = static_cast<Eigen::Index>(merged_events.size());
        summary.n_clusters_total = static_cast<Eigen::Index>(clusters.size());
        summary.n_candidate_clusters = static_cast<Eigen::Index>(candidate_clusters.size());
        summary.n_candidate_events = 0;
        for (const auto &cluster : candidate_clusters) {
            summary.n_candidate_events += cluster.n_detector_events;
        }
        summary.n_accepted_clusters = static_cast<Eigen::Index>(accepted_clusters.size());
        summary.n_accepted_events = static_cast<Eigen::Index>(accepted_rows.size());
        summary.n_rejected_clusters = n_rejected_clusters;
        summary.n_rejected_events = n_rejected_events;
        summary.n_source_protected_clusters = n_source_protected_clusters;
        summary.n_source_protected_events = n_source_protected_events;
        summary.n_det_with_added_flags = n_det_with_added_flags;
        summary.busy_network_vetoed = busy_network_vetoed;
        summary.existing_flagged_fraction = existing_flags_block.cast<double>().mean();
        summary.proposed_flagged_fraction = accepted_flags_block.cast<double>().mean();
        summary.newly_flagged_fraction =
            (accepted_flags_block.array() && !existing_flags_block.array())
                .template cast<double>().mean();
        summary.max_unflagged_residual_z = residual_peak;
        summary.max_unflagged_residual_uid = residual_peak_uid;
        if (!candidate_clusters.empty()) {
            summary.top_candidate_cluster_peak_score = candidate_clusters.front().peak_score;
            summary.top_candidate_cluster_n_detectors = candidate_clusters.front().n_detectors;
            summary.top_candidate_cluster_n_events = candidate_clusters.front().n_detector_events;
            summary.top_candidate_cluster_sample = static_cast<int>(candidate_clusters.front().sample);
        }
        const Eigen::Index max_learning_candidate_clusters =
            std::max<Eigen::Index>(8, second_pass_local.max_auto_flag_clusters_per_network + 1);
        for (Eigen::Index cluster_i = 0;
             cluster_i < static_cast<Eigen::Index>(candidate_clusters.size()) &&
             cluster_i < max_learning_candidate_clusters;
             ++cluster_i) {
            const auto &cluster = candidate_clusters[static_cast<std::size_t>(cluster_i)];
            for (const auto &row : cluster.rows) {
                const bool event_source_protected = row_source_protected(row);
                const bool event_accepted =
                    std::any_of(accepted_clusters.begin(), accepted_clusters.end(),
                                [&](const auto &accepted_cluster) {
                                    return accepted_cluster.sample == cluster.sample &&
                                           accepted_cluster.top_uid == cluster.top_uid &&
                                           accepted_cluster.peak_score == cluster.peak_score;
                                }) && !event_source_protected;
                summary.candidate_events.push_back(SecondPassCandidateEvent{
                    .uid = static_cast<int>(row.uid),
                    .kind = static_cast<int>(row.kind),
                    .sample = static_cast<int>(row.sample),
                    .start_sample = static_cast<int>(row.start_sample),
                    .end_sample = static_cast<int>(row.end_sample),
                    .score = row.score,
                    .cluster_score = cluster.peak_score,
                    .cluster_sample = static_cast<int>(cluster.sample),
                    .cluster_n_detectors = static_cast<int>(cluster.n_detectors),
                    .cluster_n_events = static_cast<int>(cluster.n_detector_events),
                    .busy_network_vetoed = busy_network_vetoed,
                    .accepted = event_accepted,
                    .source_protected = event_source_protected,
                });
            }
        }
        if (!accepted_rows.empty()) {
            summary.top_event_uid = static_cast<int>(accepted_rows.front().uid);
            summary.top_event.kind = accepted_rows.front().kind;
            summary.top_event.sample = static_cast<int>(accepted_rows.front().sample);
            summary.top_event.start_sample = static_cast<int>(accepted_rows.front().start_sample);
            summary.top_event.end_sample = static_cast<int>(accepted_rows.front().end_sample);
            summary.top_event.score = accepted_rows.front().score;
            summary.top_event.width_samples = static_cast<double>(accepted_rows.front().width_samples);
            summary.top_event.baseline_shift_z = accepted_rows.front().baseline_shift_z;
            summary.top_event.accepted = true;
        }
        summaries.push_back(summary);

        if (!candidate_clusters.empty()) {
            logger->info(
                "PTC second pass scan {} nw {} candidate_clusters={} accepted_clusters={} rejected_clusters={} source_protected_events={} busy_veto={} newly_flagged_fraction={:.4f} top_candidate_peak_score={:.4g} top_candidate_n_detectors={}",
                static_cast<long long>(in.index.data) + 1, static_cast<long long>(nw_index),
                static_cast<long long>(summary.n_candidate_clusters),
                static_cast<long long>(summary.n_accepted_clusters),
                static_cast<long long>(summary.n_rejected_clusters),
                static_cast<long long>(summary.n_source_protected_events),
                summary.busy_network_vetoed ? 1 : 0,
                summary.newly_flagged_fraction,
                summary.top_candidate_cluster_peak_score,
                static_cast<long long>(summary.top_candidate_cluster_n_detectors));
        }
    }

    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        second_pass_summary_by_scan[in.index.data] = std::move(summaries);
        if (run_tod_output) {
            second_pass_added_flags_by_scan[in.index.data] = std::move(added_flags_out);
        }
    }
}

template <typename apt_type>
void PTCProc::accumulate_weight_validation_atmosphere(
    TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_type &apt) {

    in.require_native_science_mode_consistent();
    if (in.native_science_required() &&
        weight_validation_is_enabled()) {
        throw std::runtime_error(
            "native-required PTC does not permit legacy-UID atmosphere weight validation");
    }
    if (!weight_validation_is_enabled() ||
        !weight_validation.atmospheric_correlation_enabled) {
        return;
    }

    const bool source_subtracted = run_fruit_loops && !tod_mb.signal.empty();
    if (!should_accumulate_weight_validation(source_subtracted)) {
        return;
    }

    const Eigen::Index n_dets = in.scans.data.cols();
    const Eigen::Index n_pts_full = in.scans.data.rows();
    if (n_dets <= 0 || n_pts_full <= 1) {
        return;
    }

    auto uid_for = [&](Eigen::Index i) {
        auto uid_it = apt.find("uid");
        if (uid_it != apt.end() && i < uid_it->second.size() &&
            std::isfinite(uid_it->second(i))) {
            const auto uid = static_cast<Eigen::Index>(std::llround(uid_it->second(i)));
            if (uid >= 0) {
                return uid;
            }
        }
        return i;
    };
    auto group_for = [&](Eigen::Index i) {
        if (citlali::config::is_all_processed_weight_grouping(
                weight_validation.atmospheric_grouping)) {
            return static_cast<Eigen::Index>(0);
        }
        const char *key = citlali::config::is_network_processed_weight_grouping(
                              weight_validation.atmospheric_grouping)
                              ? "nw"
                              : "array";
        auto grp_it = apt.find(key);
        if (grp_it != apt.end() && i < grp_it->second.size() &&
            std::isfinite(grp_it->second(i))) {
            return static_cast<Eigen::Index>(std::llround(grp_it->second(i)));
        }
        return static_cast<Eigen::Index>(0);
    };

    Eigen::Index max_uid = -1;
    std::vector<Eigen::Index> det_uids(static_cast<std::size_t>(n_dets), -1);
    std::map<Eigen::Index, std::vector<Eigen::Index>> dets_by_group;
    for (Eigen::Index i = 0; i < n_dets; ++i) {
        const Eigen::Index uid = uid_for(i);
        det_uids[static_cast<std::size_t>(i)] = uid;
        max_uid = std::max(max_uid, uid);
        if (apt["flag"](i) != 0) {
            continue;
        }
        dets_by_group[group_for(i)].push_back(i);
    }
    if (max_uid < 0 || dets_by_group.empty()) {
        return;
    }

    Eigen::Index sample_step = 1;
    if (weight_validation.max_samples > 0 &&
        n_pts_full > static_cast<Eigen::Index>(weight_validation.max_samples)) {
        sample_step = static_cast<Eigen::Index>(std::ceil(
            static_cast<double>(n_pts_full) /
            static_cast<double>(weight_validation.max_samples)));
    }
    sample_step = std::max<Eigen::Index>(sample_step, 1);
    const Eigen::Index n_pts = (n_pts_full + sample_step - 1) / sample_step;
    const Eigen::Index min_overlap =
        std::max<Eigen::Index>(2, weight_validation.min_overlap);
    const Eigen::Index min_group_det =
        std::max<Eigen::Index>(2, weight_validation.atmospheric_min_detectors);
    const double min_good_frac =
        std::clamp(weight_validation.min_good_frac, 0.0, 1.0);
    const double min_factor = std::clamp(weight_validation.min_factor, 0.0, 1.0);
    const double ref = std::clamp(weight_validation.atmospheric_ref, 0.0, 1.0);
    const double span = std::max(weight_validation.atmospheric_span, 1e-12);
    const double power = std::max(weight_validation.atmospheric_power, 0.0);

    Eigen::VectorXd atm_penalty =
        Eigen::VectorXd::Constant(n_dets, std::numeric_limits<double>::quiet_NaN());
    Eigen::VectorXd atm_corr =
        Eigen::VectorXd::Constant(n_dets, std::numeric_limits<double>::quiet_NaN());

    for (const auto &[group_id, group_dets] : dets_by_group) {
        (void) group_id;
        if (group_dets.size() < static_cast<std::size_t>(min_group_det)) {
            continue;
        }

        std::vector<Eigen::Index> used_dets;
        std::vector<double> det_mean;
        std::vector<double> det_std;
        used_dets.reserve(group_dets.size());
        det_mean.reserve(group_dets.size());
        det_std.reserve(group_dets.size());

        for (const auto det : group_dets) {
            double sum = 0.0;
            double sum2 = 0.0;
            Eigen::Index count = 0;
            for (Eigen::Index is = 0; is < n_pts; ++is) {
                const Eigen::Index row = is * sample_step;
                if (row >= n_pts_full) {
                    break;
                }
                if (in.flags.data(row, det)) {
                    continue;
                }
                const double v = in.scans.data(row, det);
                if (!std::isfinite(v)) {
                    continue;
                }
                sum += v;
                sum2 += v * v;
                count++;
            }
            if (count <= 1) {
                continue;
            }
            const double frac = static_cast<double>(count) /
                                static_cast<double>(n_pts);
            if (frac < min_good_frac) {
                continue;
            }
            const double mean = sum / static_cast<double>(count);
            const double var_num =
                sum2 - (sum * sum) / static_cast<double>(count);
            const double var_den = static_cast<double>(count - 1);
            if (var_den <= 0.0) {
                continue;
            }
            const double var = var_num / var_den;
            if (!(var > 0.0) || !std::isfinite(var)) {
                continue;
            }
            const double std = std::sqrt(var);
            if (!(std > 0.0) || !std::isfinite(std)) {
                continue;
            }
            used_dets.push_back(det);
            det_mean.push_back(mean);
            det_std.push_back(std);
        }

        const Eigen::Index n_used = static_cast<Eigen::Index>(used_dets.size());
        if (n_used < min_group_det) {
            continue;
        }

        Eigen::MatrixXd z =
            Eigen::MatrixXd::Constant(n_pts, n_used,
                                      std::numeric_limits<double>::quiet_NaN());
        Eigen::VectorXd sum_z = Eigen::VectorXd::Zero(n_pts);
        Eigen::VectorXi count_z = Eigen::VectorXi::Zero(n_pts);

        for (Eigen::Index k = 0; k < n_used; ++k) {
            const Eigen::Index det = used_dets[static_cast<std::size_t>(k)];
            for (Eigen::Index is = 0; is < n_pts; ++is) {
                const Eigen::Index row = is * sample_step;
                if (row >= n_pts_full) {
                    break;
                }
                if (in.flags.data(row, det)) {
                    continue;
                }
                const double v = in.scans.data(row, det);
                if (!std::isfinite(v)) {
                    continue;
                }
                const double zv =
                    (v - det_mean[static_cast<std::size_t>(k)]) /
                    det_std[static_cast<std::size_t>(k)];
                if (!std::isfinite(zv)) {
                    continue;
                }
                z(is, k) = zv;
                sum_z(is) += zv;
                count_z(is)++;
            }
        }

        for (Eigen::Index k = 0; k < n_used; ++k) {
            double sx = 0.0;
            double sy = 0.0;
            double sxx = 0.0;
            double syy = 0.0;
            double sxy = 0.0;
            Eigen::Index n_overlap = 0;
            for (Eigen::Index is = 0; is < n_pts; ++is) {
                const double x = z(is, k);
                if (!std::isfinite(x) || count_z(is) <= 1) {
                    continue;
                }
                const double y =
                    (sum_z(is) - x) /
                    static_cast<double>(count_z(is) - 1);
                if (!std::isfinite(y)) {
                    continue;
                }
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
                n_overlap++;
            }
            if (n_overlap < min_overlap) {
                continue;
            }
            const double n = static_cast<double>(n_overlap);
            const double vx = sxx - (sx * sx) / n;
            const double vy = syy - (sy * sy) / n;
            if (!(vx > 0.0) || !(vy > 0.0) ||
                !std::isfinite(vx) || !std::isfinite(vy)) {
                continue;
            }
            const double cov = sxy - (sx * sy) / n;
            double corr = cov / std::sqrt(vx * vy);
            if (!std::isfinite(corr)) {
                continue;
            }
            corr = std::clamp(corr, -1.0, 1.0);
            const double quality =
                std::clamp((corr - ref) / span, 0.0, 1.0);
            const double factor =
                min_factor + (1.0 - min_factor) * std::pow(quality, power);
            const Eigen::Index det = used_dets[static_cast<std::size_t>(k)];
            atm_corr(det) = corr;
            atm_penalty(det) = std::clamp(factor, min_factor, 1.0);
        }
    }

    Eigen::Index n_contrib = 0;
    double penalty_sum = 0.0;
    double corr_sum = 0.0;
    {
        std::lock_guard<std::mutex> lk(*weight_validation_mutex);
        ensure_weight_validation_storage(max_uid + 1);
        for (Eigen::Index i = 0; i < n_dets; ++i) {
            if (!std::isfinite(atm_penalty(i))) {
                continue;
            }
            const Eigen::Index uid = det_uids[static_cast<std::size_t>(i)];
            if (uid < 0 || uid >= weight_validation_atm_count.size()) {
                continue;
            }
            weight_validation_atm_penalty_sum(uid) += atm_penalty(i);
            weight_validation_atm_corr_sum(uid) += atm_corr(i);
            weight_validation_atm_count(uid)++;
            penalty_sum += atm_penalty(i);
            corr_sum += atm_corr(i);
            n_contrib++;
        }
        if (n_contrib > 0) {
            weight_validation_current_iter_contribution_count++;
        }
    }

    if (n_contrib > 0) {
        logger->info(
            "weight validation atmosphere scan={} grouping={} detectors={} sample_step={} "
            "mean_factor={} mean_corr={}",
            static_cast<long long>(in.index.data) + 1,
            weight_validation.atmospheric_grouping,
            n_contrib,
            sample_step,
            penalty_sum / static_cast<double>(n_contrib),
            corr_sum / static_cast<double>(n_contrib));
    }
}

template <typename apt_type, class tel_type>
void PTCProc::calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_type &apt, tel_type &telescope,
                           bool source_subtracted_for_weight_validation) {
    engine_utils::require_native_science_matrix(in);
    if (in.native_science_required() &&
        weight_validation_is_enabled()) {
        throw std::runtime_error(
            "native-required PTC does not permit dense legacy-UID weight validation");
    }
    if (in.native_science_required() &&
        (weight_corr_penalty.enabled ||
         busy_row_suppression.enabled)) {
        throw std::runtime_error(
            "native-required PTC does not permit nontransactional processed-weight diagnostic penalties");
    }
    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();
    const auto scan_index_1based = static_cast<long long>(in.index.data) + 1;
    const bool uses_approximate_weighting =
        citlali::config::is_approximate_processed_weighting_type(
            weighting_type);
    const bool uses_full_weighting =
        citlali::config::is_full_processed_weighting_type(weighting_type);
    const bool uses_hybrid_weighting =
        citlali::config::is_hybrid_processed_weighting_type(weighting_type);
    const bool uses_validated_weighting =
        citlali::config::is_validated_processed_weighting_type(
            weighting_type);
    const bool uses_constant_weighting =
        citlali::config::is_constant_processed_weighting_type(weighting_type);

    // resize weights to number of detectors
    in.weights.data = Eigen::VectorXd::Zero(n_dets);

    // approximate weighting
    if (uses_approximate_weighting) {
        logger->debug("calculating weights using detector sensitivities");
        // unit conversion x flux calibration factor x 1/exp(-tau)
        double conversion_factor;

        // loop through detectors and calculate weights
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // current detector index
            Eigen::Index det_index = i;
            if (apt["flag"](det_index)!=0) {
                in.weights.data(i) = 0;
                continue;
            }
            // if flux calibrated, get flux conversion factor
            if (in.status.calibrated) {
                conversion_factor = in.fcf.data(i);
            }
            // otherwise fcf is unity
            else {
                conversion_factor = 1;
            }
            // make sure flux conversion is not zero (otherwise weight=0)
            const double denom = conversion_factor * apt["sens"](det_index);
            const double weight_scale = std::sqrt(telescope.d_fsmp) * denom;
            if (std::isfinite(weight_scale) && weight_scale != 0.0) {
                // calculate weights while applying flux calibration
                in.weights.data(i) = pow(weight_scale,-2.0);
            }
            else {
                in.weights.data(i) = 0;
            }
        }
    }
    // use full weighting
    else if (uses_full_weighting) {
        logger->debug("calculating weights using timestream variance");
        const bool use_source_weight_mask =
            source_mask_radius_arcsec > 0.0 &&
            fruit_loops_source_valid.size() > 0 &&
            fruit_loops_source_lat.size() == fruit_loops_source_valid.size() &&
            fruit_loops_source_lon.size() == fruit_loops_source_valid.size();
        const double source_mask_radius_rad = source_mask_radius_arcsec * ASEC_TO_RAD;

        if (use_source_weight_mask) {
            logger->info("calculating full weights with source mask (radius {:.3f} arcsec) for scan {}",
                         source_mask_radius_arcsec, scan_index_1based);
        }

        // loop through detectors
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // only calculate weights if detector is unflagged
            if (apt["flag"](i)==0) {
                // make Eigen::Maps for each detector's scan
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                    in.scans.data.col(i).data(), in.scans.data.rows());
                Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> base_flags(
                    in.flags.data.col(i).data(), in.flags.data.rows());

                // unflagged detector stddev
                double det_std_dev = 0.0;
                if (use_source_weight_mask &&
                    i < in.map_indices.data.size()) {
                    const auto map_index = in.map_indices.data(i);
                    if (map_index >= 0 &&
                        map_index < fruit_loops_source_valid.size() &&
                        fruit_loops_source_valid(map_index)) {
                        Eigen::Matrix<bool, Eigen::Dynamic, 1> weight_flags = base_flags;
                        auto [lat, lon] =
                            engine_utils::calc_det_pointing_for_science_sample(
                                in, i, apt["x_t"](i), apt["y_t"](i),
                                telescope.pixel_axes,
                                active_map_grouping);
                        const double source_lat = fruit_loops_source_lat(map_index);
                        const double source_lon = fruit_loops_source_lon(map_index);
                        for (Eigen::Index j = 0; j < weight_flags.size(); ++j) {
                            const double dlat = lat(j) - source_lat;
                            const double dlon = lon(j) - source_lon;
                            if (std::sqrt(dlat * dlat + dlon * dlon) < source_mask_radius_rad) {
                                weight_flags(j) = 1;
                            }
                        }
                        det_std_dev = engine_utils::calc_std_dev(scans, weight_flags);
                    }
                    else {
                        det_std_dev = engine_utils::calc_std_dev(scans, base_flags);
                    }
                }
                else {
                    det_std_dev = engine_utils::calc_std_dev(scans, base_flags);
                }
                // if stddev is not zero
                if (std::isfinite(det_std_dev) && det_std_dev > 0.0) {
                    // weight = 1/(stddev)^2
                    in.weights.data(i) = pow(det_std_dev,-2);
                }
                // otherwise weight = 0 (not included in maps)
                else {
                    in.weights.data(i) = 0;
                }
            }
            // otherwise weight = 0 (not included in maps)
            else {
                in.weights.data(i) = 0;
            }
        }
    }
    // hybrid/validated weighting: approximate calibration prior plus residual-variance diagnostics
    else if (uses_hybrid_weighting || uses_validated_weighting) {
        if (uses_hybrid_weighting) {
            logger->debug("calculating hybrid weights using detector sensitivities and residual variance");
        }
        else {
            logger->debug("calculating validated weights using detector sensitivities and learned detector penalties");
        }
        Eigen::VectorXd approximate_weights = Eigen::VectorXd::Zero(n_dets);
        Eigen::VectorXd full_weights = Eigen::VectorXd::Zero(n_dets);
        Eigen::VectorXd ratio_penalty = Eigen::VectorXd::Ones(n_dets);
        std::vector<Eigen::Index> det_uids(static_cast<std::size_t>(n_dets), -1);
        Eigen::Index max_uid = -1;

        auto calc_approx_weight = [&](Eigen::Index i) {
            if (apt["flag"](i)!=0) {
                return 0.0;
            }
            const double conversion_factor = in.status.calibrated ? in.fcf.data(i) : 1.0;
            const double denom = conversion_factor * apt["sens"](i);
            const double weight_scale = std::sqrt(telescope.d_fsmp) * denom;
            if (std::isfinite(weight_scale) && weight_scale != 0.0) {
                return std::pow(weight_scale, -2.0);
            }
            return 0.0;
        };

        auto uid_for = [&](Eigen::Index i) {
            if (in.native_science_required()) {
                const auto uid =
                    in.require_native_scan()
                        .require_detector_binding(i).uid;
                if (uid < 0 ||
                    static_cast<std::uint64_t>(uid) >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<Eigen::Index>::max())) {
                    throw std::runtime_error(
                        "native-required validated weight UID is not exactly representable");
                }
                return static_cast<Eigen::Index>(uid);
            }
            auto uid_it = apt.find("uid");
            if (uid_it != apt.end() && i < uid_it->second.size() &&
                std::isfinite(uid_it->second(i))) {
                const auto uid = static_cast<Eigen::Index>(std::llround(uid_it->second(i)));
                if (uid >= 0) {
                    return uid;
                }
            }
            return i;
        };

        const bool use_source_weight_mask =
            source_mask_radius_arcsec > 0.0 &&
            fruit_loops_source_valid.size() > 0 &&
            fruit_loops_source_lat.size() == fruit_loops_source_valid.size() &&
            fruit_loops_source_lon.size() == fruit_loops_source_valid.size();
        const double source_mask_radius_rad = source_mask_radius_arcsec * ASEC_TO_RAD;
        if (use_source_weight_mask) {
            logger->info("calculating hybrid full-weight correction with source mask (radius {:.3f} arcsec) for scan {}",
                         source_mask_radius_arcsec, scan_index_1based);
        }

        auto calc_full_weight = [&](Eigen::Index i) {
            if (apt["flag"](i)!=0) {
                return 0.0;
            }
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                in.scans.data.col(i).data(), in.scans.data.rows());
            Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> base_flags(
                in.flags.data.col(i).data(), in.flags.data.rows());

            double det_std_dev = 0.0;
            if (use_source_weight_mask &&
                i < in.map_indices.data.size()) {
                const auto map_index = in.map_indices.data(i);
                if (map_index >= 0 &&
                    map_index < fruit_loops_source_valid.size() &&
                    fruit_loops_source_valid(map_index)) {
                    Eigen::Matrix<bool, Eigen::Dynamic, 1> weight_flags = base_flags;
                    auto [lat, lon] =
                        engine_utils::calc_det_pointing_for_science_sample(
                            in, i, apt["x_t"](i), apt["y_t"](i),
                            telescope.pixel_axes,
                            active_map_grouping);
                    const double source_lat = fruit_loops_source_lat(map_index);
                    const double source_lon = fruit_loops_source_lon(map_index);
                    for (Eigen::Index j = 0; j < weight_flags.size(); ++j) {
                        const double dlat = lat(j) - source_lat;
                        const double dlon = lon(j) - source_lon;
                        if (std::sqrt(dlat * dlat + dlon * dlon) < source_mask_radius_rad) {
                            weight_flags(j) = 1;
                        }
                    }
                    det_std_dev = engine_utils::calc_std_dev(scans, weight_flags);
                }
                else {
                    det_std_dev = engine_utils::calc_std_dev(scans, base_flags);
                }
            }
            else {
                det_std_dev = engine_utils::calc_std_dev(scans, base_flags);
            }
            if (std::isfinite(det_std_dev) && det_std_dev > 0.0) {
                return std::pow(det_std_dev, -2.0);
            }
            return 0.0;
        };

        std::map<Eigen::Index, std::vector<double>> ratios_by_array;
        for (Eigen::Index i=0; i<n_dets; ++i) {
            approximate_weights(i) = calc_approx_weight(i);
            full_weights(i) = calc_full_weight(i);
            if (approximate_weights(i) > 0.0 && full_weights(i) > 0.0 &&
                std::isfinite(approximate_weights(i)) && std::isfinite(full_weights(i))) {
                const Eigen::Index array_index =
                    static_cast<Eigen::Index>(std::lround(apt["array"](i)));
                ratios_by_array[array_index].push_back(full_weights(i) / approximate_weights(i));
            }
        }

        std::map<Eigen::Index, double> median_ratio_by_array;
        for (auto const& [array_index, ratios] : ratios_by_array) {
            Eigen::VectorXd ratio_vec(static_cast<Eigen::Index>(ratios.size()));
            for (Eigen::Index i=0; i<ratio_vec.size(); ++i) {
                ratio_vec(i) = ratios[static_cast<std::size_t>(i)];
            }
            const double median_ratio = tula::alg::median(ratio_vec);
            median_ratio_by_array[array_index] =
                (std::isfinite(median_ratio) && median_ratio > 0.0) ? median_ratio : 1.0;
        }

        std::vector<double> high_weight_group_median(
            static_cast<std::size_t>(n_dets),
            std::numeric_limits<double>::quiet_NaN());
        std::vector<double> high_weight_robust_z(
            static_cast<std::size_t>(n_dets),
            std::numeric_limits<double>::quiet_NaN());
        std::vector<double> high_weight_cap(
            static_cast<std::size_t>(n_dets),
            std::numeric_limits<double>::quiet_NaN());
        std::vector<unsigned char> high_weight_extreme(
            static_cast<std::size_t>(n_dets), 0);
        std::vector<HighWeightDiagSummary> high_weight_records;

        auto safe_apt_int = [&](const std::string &key, Eigen::Index i, Eigen::Index fallback) {
            if (key == "nw" && in.native_science_required()) {
                const auto network =
                    in.require_native_scan()
                        .require_detector_binding(i).network;
                if (network < 0 ||
                    static_cast<std::uint64_t>(network) >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<Eigen::Index>::max())) {
                    throw std::runtime_error(
                        "native-required validated weight network is not exactly representable");
                }
                return static_cast<Eigen::Index>(network);
            }
            auto it = apt.find(key);
            if (it != apt.end() && i >= 0 && i < it->second.size() &&
                std::isfinite(it->second(i))) {
                return static_cast<Eigen::Index>(std::llround(it->second(i)));
            }
            return fallback;
        };

        auto high_weight_group_for = [&](Eigen::Index i) {
            if (citlali::config::is_all_processed_weight_grouping(
                    weight_validation.high_weight_grouping)) {
                return static_cast<Eigen::Index>(0);
            }
            if (citlali::config::is_network_processed_weight_grouping(
                    weight_validation.high_weight_grouping)) {
                return safe_apt_int("nw", i, 0);
            }
            return safe_apt_int("array", i, 0);
        };

        if (uses_validated_weighting &&
            weight_validation.high_weight_validation_enabled) {
            std::map<Eigen::Index, std::vector<std::pair<Eigen::Index, double>>> log_weights_by_group;
            for (Eigen::Index i=0; i<n_dets; ++i) {
                if (apt["flag"](i) != 0 ||
                    approximate_weights(i) <= 0.0 ||
                    !std::isfinite(approximate_weights(i))) {
                    continue;
                }
                log_weights_by_group[high_weight_group_for(i)].push_back(
                    {i, std::log(approximate_weights(i))});
            }

            const Eigen::Index min_group =
                std::max<Eigen::Index>(2, weight_validation.high_weight_min_group_detectors);
            const double robust_z_thresh =
                std::max(0.0, weight_validation.high_weight_log_robust_z);
            const double max_median_factor =
                std::max(1.0, weight_validation.high_weight_max_median_factor);
            const double cap_median_factor =
                std::max(1.0, weight_validation.high_weight_cap_median_factor);
            for (const auto &[group_id, entries] : log_weights_by_group) {
                (void) group_id;
                if (entries.size() < static_cast<std::size_t>(min_group)) {
                    continue;
                }
                Eigen::VectorXd logs(static_cast<Eigen::Index>(entries.size()));
                for (Eigen::Index k=0; k<logs.size(); ++k) {
                    logs(k) = entries[static_cast<std::size_t>(k)].second;
                }
                const double med_log = tula::alg::median(logs);
                if (!std::isfinite(med_log)) {
                    continue;
                }
                Eigen::VectorXd abs_dev = (logs.array() - med_log).abs();
                double sigma_log = 1.4826 * tula::alg::median(abs_dev);
                if (!std::isfinite(sigma_log) ||
                    sigma_log <= std::numeric_limits<double>::epsilon()) {
                    sigma_log = std::numeric_limits<double>::infinity();
                }
                const double median_weight = std::exp(med_log);
                if (!std::isfinite(median_weight) || median_weight <= 0.0) {
                    continue;
                }
                const double cap_weight = median_weight * cap_median_factor;
                for (const auto &[det, log_weight] : entries) {
                    const double robust_z =
                        std::isfinite(sigma_log)
                            ? std::max(0.0, (log_weight - med_log) / sigma_log)
                            : 0.0;
                    const double median_factor = approximate_weights(det) / median_weight;
                    const bool extreme =
                        (robust_z_thresh > 0.0 && robust_z >= robust_z_thresh) ||
                        median_factor >= max_median_factor;
                    const auto idx = static_cast<std::size_t>(det);
                    high_weight_group_median[idx] = median_weight;
                    high_weight_robust_z[idx] = robust_z;
                    high_weight_cap[idx] = cap_weight;
                    high_weight_extreme[idx] = extreme ? 1 : 0;
                }
            }
        }

        auto normalized_full_over_approx = [&](Eigen::Index i, bool &valid) {
            valid = false;
            if (full_weights(i) > 0.0 && std::isfinite(full_weights(i))) {
                const Eigen::Index array_index =
                    static_cast<Eigen::Index>(std::lround(apt["array"](i)));
                auto med_it = median_ratio_by_array.find(array_index);
                const double median_ratio =
                    (med_it != median_ratio_by_array.end()) ? med_it->second : 1.0;
                if (std::isfinite(median_ratio) && median_ratio > 0.0) {
                    valid = true;
                    return (full_weights(i) / approximate_weights(i)) / median_ratio;
                }
            }
            return 1.0;
        };

        if (uses_hybrid_weighting) {
            for (Eigen::Index i=0; i<n_dets; ++i) {
                if (approximate_weights(i) <= 0.0 || !std::isfinite(approximate_weights(i))) {
                    in.weights.data(i) = 0.0;
                    continue;
                }
                bool valid_ratio = false;
                double correction = normalized_full_over_approx(i, valid_ratio);
                (void) valid_ratio;
                if (!std::isfinite(correction) || correction <= 0.0) {
                    correction = 1.0;
                }
                correction = std::clamp(
                    correction, hybrid_correction_min_factor, hybrid_correction_max_factor);
                in.weights.data(i) = approximate_weights(i) * correction;
            }

            logger->info(
                "hybrid weight correction scan={} source_mask_radius_arcsec={} min_factor={} max_factor={} arrays={}",
                scan_index_1based, source_mask_radius_arcsec,
                hybrid_correction_min_factor, hybrid_correction_max_factor,
                median_ratio_by_array.size());
        }
        else {
            const double min_factor = std::clamp(weight_validation.min_factor, 0.0, 1.0);
            const double unvalidated_factor =
                std::clamp(weight_validation.unvalidated_factor, min_factor, 1.0);
            const double ratio_power = std::max(weight_validation.ratio_power, 0.0);
            const double transient_power = std::max(weight_validation.transient_ratio_power, 0.0);

            for (Eigen::Index i=0; i<n_dets; ++i) {
                const Eigen::Index uid = uid_for(i);
                det_uids[static_cast<std::size_t>(i)] = uid;
                max_uid = std::max(max_uid, uid);

                if (approximate_weights(i) <= 0.0 || !std::isfinite(approximate_weights(i))) {
                    in.weights.data(i) = 0.0;
                    ratio_penalty(i) = 0.0;
                    continue;
                }

                bool valid_ratio = false;
                double correction = normalized_full_over_approx(i, valid_ratio);
                if (!valid_ratio || !std::isfinite(correction) || correction <= 0.0) {
                    correction = min_factor;
                }
                double penalty = std::min(1.0, correction);
                penalty = std::pow(std::clamp(penalty, 0.0, 1.0), ratio_power);
                ratio_penalty(i) = std::clamp(penalty, min_factor, 1.0);
            }

            if (should_accumulate_weight_validation(source_subtracted_for_weight_validation) &&
                max_uid >= 0) {
                Eigen::Index n_contrib = 0;
                Eigen::Index n_ratio_valid = 0;
                double penalty_sum = 0.0;
                double ratio_sum = 0.0;
                {
                    std::lock_guard<std::mutex> lk(*weight_validation_mutex);
                    ensure_weight_validation_storage(max_uid + 1);
                    for (Eigen::Index i=0; i<n_dets; ++i) {
                        if (approximate_weights(i) <= 0.0 || !std::isfinite(approximate_weights(i)) ||
                            !std::isfinite(ratio_penalty(i))) {
                            continue;
                        }
                        const Eigen::Index uid = det_uids[static_cast<std::size_t>(i)];
                        if (uid < 0 || uid >= weight_validation_ratio_count.size()) {
                            continue;
                        }
                        bool valid_ratio = false;
                        const double correction = normalized_full_over_approx(i, valid_ratio);
                        weight_validation_ratio_penalty_sum(uid) += ratio_penalty(i);
                        if (valid_ratio && std::isfinite(correction)) {
                            weight_validation_ratio_value_sum(uid) += correction;
                            weight_validation_ratio_value_count(uid)++;
                            ratio_sum += correction;
                            n_ratio_valid++;
                        }
                        weight_validation_ratio_count(uid)++;
                        penalty_sum += ratio_penalty(i);
                        n_contrib++;
                    }
                    if (n_contrib > 0) {
                        weight_validation_current_iter_contribution_count++;
                    }
                }
                if (n_contrib > 0) {
                    logger->info(
                        "weight validation ratio scan={} detectors={} source_mask_radius_arcsec={} "
                        "mean_factor={} mean_full_over_approx={}",
                        scan_index_1based, n_contrib, source_mask_radius_arcsec,
                        penalty_sum / static_cast<double>(n_contrib),
                        n_ratio_valid > 0
                            ? ratio_sum / static_cast<double>(n_ratio_valid)
                            : std::numeric_limits<double>::quiet_NaN());
                }
            }

            Eigen::VectorXd detector_penalty;
            Eigen::VectorXi detector_validated;
            bool use_learned_penalty = false;
            int current_iter = 0;
            {
                std::lock_guard<std::mutex> lk(*weight_validation_mutex);
                current_iter = weight_validation_current_iter;
                use_learned_penalty =
                    weight_validation_finalized &&
                    current_iter >= weight_validation.apply_start_iter &&
                    weight_validation_detector_penalty.size() > 0;
                if (use_learned_penalty) {
                    detector_penalty = weight_validation_detector_penalty;
                    detector_validated = weight_validation_detector_validated;
                }
            }

            Eigen::Index n_weighted = 0;
            Eigen::Index n_penalized = 0;
            Eigen::Index n_boosted = 0;
            Eigen::Index n_high_weight_extreme = 0;
            Eigen::Index n_high_weight_cap_recommended = 0;
            Eigen::Index n_high_weight_cap_applied = 0;
            const double max_applied_factor =
                weight_validation.upward_enabled
                    ? std::max(1.0, weight_validation.upward_max_factor)
                    : 1.0;
            const bool allow_transient_ratio =
                !weight_validation.require_fruitloops_model ||
                source_subtracted_for_weight_validation;
            for (Eigen::Index i=0; i<n_dets; ++i) {
                if (approximate_weights(i) <= 0.0 || !std::isfinite(approximate_weights(i))) {
                    in.weights.data(i) = 0.0;
                    continue;
                }
                double factor = unvalidated_factor;
                const Eigen::Index uid = det_uids[static_cast<std::size_t>(i)];
                if (use_learned_penalty && uid >= 0 && uid < detector_penalty.size() &&
                    std::isfinite(detector_penalty(uid))) {
                    factor = detector_penalty(uid);
                }
                else if (!use_learned_penalty) {
                    factor = 1.0;
                }
                if (allow_transient_ratio && weight_validation.transient_ratio_enabled &&
                    std::isfinite(ratio_penalty(i))) {
                    const double transient_factor =
                        std::pow(std::clamp(ratio_penalty(i), 0.0, 1.0), transient_power);
                    factor = std::min(factor, std::clamp(transient_factor, min_factor, 1.0));
                }
                factor = std::clamp(factor, min_factor, max_applied_factor);
                double final_weight = approximate_weights(i) * factor;
                bool high_weight_validated = false;
                bool high_weight_cap_recommended = false;
                bool high_weight_cap_applied = false;
                const auto det_index = static_cast<std::size_t>(i);
                if (weight_validation.high_weight_validation_enabled &&
                    det_index < high_weight_extreme.size() &&
                    high_weight_extreme[det_index] != 0) {
                    ++n_high_weight_extreme;
                    const bool detector_has_validation =
                        use_learned_penalty &&
                        uid >= 0 &&
                        uid < detector_validated.size() &&
                        detector_validated(uid) != 0;
                    high_weight_validated =
                        detector_has_validation &&
                        factor >= weight_validation.high_weight_min_validated_factor;
                    high_weight_cap_recommended = !high_weight_validated;
                    if (high_weight_cap_recommended) {
                        ++n_high_weight_cap_recommended;
                    }
                    if (high_weight_cap_recommended &&
                        weight_validation.high_weight_apply_caps &&
                        use_learned_penalty &&
                        std::isfinite(high_weight_cap[det_index]) &&
                        high_weight_cap[det_index] > 0.0 &&
                        final_weight > high_weight_cap[det_index]) {
                        final_weight = high_weight_cap[det_index];
                        factor = final_weight / approximate_weights(i);
                        high_weight_cap_applied = true;
                        ++n_high_weight_cap_applied;
                    }

                    HighWeightDiagSummary record;
                    record.iter = current_iter;
                    record.scan = in.index.data;
                    record.det = i;
                    record.uid = static_cast<int>(uid);
                    record.nw = safe_apt_int("nw", i, -1);
                    record.array = safe_apt_int("array", i, -1);
                    record.grouping = weight_validation.high_weight_grouping;
                    record.reason = high_weight_cap_applied
                        ? "high_weight_cap_applied"
                        : (high_weight_cap_recommended
                               ? "high_weight_cap_recommended"
                               : "high_weight_validated");
                    record.approximate_weight = approximate_weights(i);
                    record.final_weight = final_weight;
                    record.group_median_weight = high_weight_group_median[det_index];
                    record.robust_z = high_weight_robust_z[det_index];
                    record.applied_cap = high_weight_cap[det_index];
                    record.validation_factor = factor;
                    record.cap_recommended = high_weight_cap_recommended;
                    record.cap_applied = high_weight_cap_applied;
                    record.validated = high_weight_validated;
                    high_weight_records.push_back(std::move(record));
                }
                in.weights.data(i) = final_weight;
                n_weighted++;
                if (factor < 0.999) {
                    n_penalized++;
                }
                if (factor > 1.001) {
                    n_boosted++;
                }
            }

            logger->info(
                "validated weight correction scan={} iter={} source_mask_radius_arcsec={} "
                "learned_applied={} weighted_detectors={} penalized_detectors={} boosted_detectors={} arrays={} "
                "high_weight_extreme={} high_weight_cap_recommended={} high_weight_cap_applied={}",
                scan_index_1based, current_iter, source_mask_radius_arcsec,
                use_learned_penalty, n_weighted, n_penalized, n_boosted,
                median_ratio_by_array.size(), n_high_weight_extreme,
                n_high_weight_cap_recommended, n_high_weight_cap_applied);
        }
        if (uses_validated_weighting &&
            weight_validation.high_weight_validation_enabled) {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            high_weight_summary_by_scan[in.index.data] = std::move(high_weight_records);
        }
    }
    // constant weighting
    else if (uses_constant_weighting) {
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // only calculate weights if detector is unflagged
            if (apt["flag"](i)==0) {
                in.weights.data(i) = 1;
            }
            // otherwise set to zero
            else {
                in.weights.data(i) = 0;
            }
        }
    }

    auto finite_or_nan = [](double v) {
        if (std::isfinite(v)) {
            return v;
        }
        return std::numeric_limits<double>::quiet_NaN();
    };

    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> nw_limits;
    if (weight_corr_penalty.enabled || busy_row_suppression.enabled) {
        if (n_dets > 0) {
            auto network_for = [&](Eigen::Index detector) {
                if (!in.native_science_required()) {
                    return static_cast<Eigen::Index>(
                        apt["nw"](detector));
                }
                const auto network =
                    in.require_native_scan()
                        .require_detector_binding(detector).network;
                if (network < 0 ||
                    static_cast<std::uint64_t>(network) >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<Eigen::Index>::max())) {
                    throw std::runtime_error(
                        "native-required weight-correlation network is not exactly representable");
                }
                return static_cast<Eigen::Index>(network);
            };
            Eigen::Index nw_i = network_for(0);
            nw_limits[nw_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 1};
            std::unordered_set<Eigen::Index> seen;
            seen.insert(nw_i);
            for (Eigen::Index i = 1; i < n_dets; ++i) {
                auto nw_v = network_for(i);
                if (nw_v == nw_i) {
                    std::get<1>(nw_limits[nw_i]) = i + 1;
                }
                else {
                    if (seen.find(nw_v) != seen.end()) {
                        logger->error("non-contiguous grouping detected for 'nw' value {}", nw_v);
                        citlali::pipeline::require_group_value_not_seen(
                            true, "nw", nw_v);
                    }
                    seen.insert(nw_v);
                    nw_i = nw_v;
                    nw_limits[nw_i] = std::tuple<Eigen::Index, Eigen::Index>{i, i + 1};
                }
            }
        }
    }

    if (weight_corr_penalty.enabled) {
        auto clamp01 = [](double v) {
            return std::clamp(v, 0.0, 1.0);
        };
        auto median_from_values = [](std::vector<double> values) {
            if (values.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const auto mid = values.size() / 2;
            std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
            double med = values[mid];
            if ((values.size() % 2) == 0) {
                auto max_it = std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
                med = 0.5 * (med + *max_it);
            }
            return med;
        };
        auto pearson_corr = [](const std::vector<double> &x, const std::vector<double> &y) {
            if (x.size() != y.size() || x.size() < 2) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            double sx = 0.0;
            double sy = 0.0;
            double sxx = 0.0;
            double syy = 0.0;
            double sxy = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double xv = x[i];
                const double yv = y[i];
                sx += xv;
                sy += yv;
                sxx += xv * xv;
                syy += yv * yv;
                sxy += xv * yv;
            }
            const double n = static_cast<double>(x.size());
            const double vx = sxx - (sx * sx) / n;
            const double vy = syy - (sy * sy) / n;
            if (vx <= 0.0 || vy <= 0.0 || !std::isfinite(vx) || !std::isfinite(vy)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double cov = sxy - (sx * sy) / n;
            const double corr = cov / std::sqrt(vx * vy);
            if (!std::isfinite(corr)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return std::clamp(corr, -1.0, 1.0);
        };
        auto score_metric = [&](double metric, const auto &term) {
            if (!term.enabled || term.weight <= 0.0 || !std::isfinite(metric)) {
                return std::pair<double, double>{0.0, 0.0};
            }
            const double span = std::max(term.span, 1e-12);
            const double score = clamp01((metric - term.ref) / span);
            return std::pair<double, double>{term.weight * score, term.weight};
        };

        const Eigen::Index n_pts_full = in.scans.data.rows();
        std::vector<WeightCorrPenaltyDiagSummary> penalty_summary;
        penalty_summary.reserve(static_cast<std::size_t>(nw_limits.size()));

        for (const auto &[nw, limits] : nw_limits) {
            const auto [start_index, end_index] = limits;
            const Eigen::Index n_det_group = end_index - start_index;

            Eigen::Index sample_step = 1;
            if (weight_corr_penalty.max_samples > 0 &&
                n_pts_full > static_cast<Eigen::Index>(weight_corr_penalty.max_samples)) {
                sample_step = static_cast<Eigen::Index>(std::ceil(
                    static_cast<double>(n_pts_full) / static_cast<double>(weight_corr_penalty.max_samples)));
            }
            sample_step = std::max<Eigen::Index>(sample_step, 1);
            const Eigen::Index n_pts = (n_pts_full + sample_step - 1) / sample_step;

            std::vector<Eigen::Index> det_keep;
            std::vector<double> det_mean;
            std::vector<double> det_std;
            det_keep.reserve(static_cast<std::size_t>(n_det_group));
            det_mean.reserve(static_cast<std::size_t>(n_det_group));
            det_std.reserve(static_cast<std::size_t>(n_det_group));

            Eigen::Index n_candidates = 0;
            for (Eigen::Index j = start_index; j < end_index; ++j) {
                if (apt["flag"](j) != 0) {
                    continue;
                }
                double sum = 0.0;
                double sum2 = 0.0;
                double count = 0.0;
                for (Eigen::Index is = 0; is < n_pts; ++is) {
                    const Eigen::Index i = is * sample_step;
                    if (i >= n_pts_full) {
                        break;
                    }
                    if (in.flags.data(i, j)) {
                        continue;
                    }
                    const double v = in.scans.data(i, j);
                    if (!std::isfinite(v)) {
                        continue;
                    }
                    sum += v;
                    sum2 += v * v;
                    count += 1.0;
                }
                if (count <= 1.0) {
                    continue;
                }
                const double frac = count / static_cast<double>(n_pts);
                if (frac < weight_corr_penalty.min_good_frac) {
                    continue;
                }
                n_candidates++;
                const double mean = sum / count;
                const double var_num = sum2 - (sum * sum) / count;
                const double var_den = count - 1.0;
                if (var_den <= 0.0) {
                    continue;
                }
                const double var = var_num / var_den;
                if (!(var > 0.0) || !std::isfinite(var)) {
                    continue;
                }
                const double std = std::sqrt(var);
                if (!(std > 0.0) || !std::isfinite(std)) {
                    continue;
                }
                det_keep.push_back(j);
                det_mean.push_back(mean);
                det_std.push_back(std);
            }
            const Eigen::Index n_used = static_cast<Eigen::Index>(det_keep.size());

            auto pair_corr_for = [&](Eigen::Index det_a, Eigen::Index det_b) {
                double sx = 0.0;
                double sy = 0.0;
                double sxx = 0.0;
                double syy = 0.0;
                double sxy = 0.0;
                Eigen::Index n_ov = 0;
                for (Eigen::Index is = 0; is < n_pts; ++is) {
                    const Eigen::Index i = is * sample_step;
                    if (i >= n_pts_full) {
                        break;
                    }
                    if (in.flags.data(i, det_a) || in.flags.data(i, det_b)) {
                        continue;
                    }
                    const double x = in.scans.data(i, det_a);
                    const double y = in.scans.data(i, det_b);
                    if (!std::isfinite(x) || !std::isfinite(y)) {
                        continue;
                    }
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    syy += y * y;
                    sxy += x * y;
                    n_ov++;
                }
                const Eigen::Index min_overlap = std::max<Eigen::Index>(2, weight_corr_penalty.min_overlap);
                if (n_ov < min_overlap) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const double n = static_cast<double>(n_ov);
                const double vx = sxx - (sx * sx) / n;
                const double vy = syy - (sy * sy) / n;
                if (!(vx > 0.0) || !(vy > 0.0) || !std::isfinite(vx) || !std::isfinite(vy)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const double cov = sxy - (sx * sy) / n;
                const double corr = cov / std::sqrt(vx * vy);
                if (!std::isfinite(corr)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                return std::clamp(corr, -1.0, 1.0);
            };

            double pair_med_abs_corr = std::numeric_limits<double>::quiet_NaN();
            if (weight_corr_penalty.pair_corr.enabled && n_used >= 2) {
                const std::uint64_t n_pairs_total = static_cast<std::uint64_t>(n_used) *
                                                    static_cast<std::uint64_t>(n_used - 1) / 2ULL;
                std::uint64_t target_pairs = n_pairs_total;
                if (weight_corr_penalty.max_pairs > 0) {
                    target_pairs = std::min<std::uint64_t>(
                        n_pairs_total, static_cast<std::uint64_t>(weight_corr_penalty.max_pairs));
                }
                std::vector<double> abs_corrs;
                abs_corrs.reserve(static_cast<std::size_t>(target_pairs));

                if (target_pairs == n_pairs_total) {
                    for (Eigen::Index i = 0; i < n_used; ++i) {
                        for (Eigen::Index j = i + 1; j < n_used; ++j) {
                            const double c = pair_corr_for(
                                det_keep[static_cast<std::size_t>(i)],
                                det_keep[static_cast<std::size_t>(j)]);
                            if (std::isfinite(c)) {
                                abs_corrs.push_back(std::abs(c));
                            }
                        }
                    }
                }
                else if (target_pairs > 0) {
                    const std::uint64_t seed_mix =
                        static_cast<std::uint64_t>(weight_corr_penalty.seed) ^
                        (static_cast<std::uint64_t>(scan_index_1based + 1) * 1315423911ULL) ^
                        (static_cast<std::uint64_t>(nw + 1) * 2654435761ULL);
                    std::mt19937 rng_nw(static_cast<std::uint32_t>(seed_mix & 0xffffffffULL));
                    std::uniform_int_distribution<Eigen::Index> det_dist(0, n_used - 1);
                    std::unordered_set<std::uint64_t> seen_pairs;
                    seen_pairs.reserve(static_cast<std::size_t>(target_pairs * 2 + 1));
                    std::uint64_t tries = 0;
                    const std::uint64_t max_tries = std::max<std::uint64_t>(target_pairs * 32ULL, 1024ULL);
                    while (seen_pairs.size() < target_pairs && tries < max_tries) {
                        tries++;
                        Eigen::Index a = det_dist(rng_nw);
                        Eigen::Index b = det_dist(rng_nw);
                        if (a == b) {
                            continue;
                        }
                        if (a > b) {
                            std::swap(a, b);
                        }
                        const auto key = (static_cast<std::uint64_t>(a) << 32ULL) |
                                         static_cast<std::uint64_t>(b);
                        if (!seen_pairs.insert(key).second) {
                            continue;
                        }
                        const double c = pair_corr_for(
                            det_keep[static_cast<std::size_t>(a)],
                            det_keep[static_cast<std::size_t>(b)]);
                        if (std::isfinite(c)) {
                            abs_corrs.push_back(std::abs(c));
                        }
                    }
                }
                pair_med_abs_corr = median_from_values(std::move(abs_corrs));
            }

            Eigen::VectorXd cm = Eigen::VectorXd::Constant(n_pts, std::numeric_limits<double>::quiet_NaN());
            std::vector<double> cm_valid;
            std::vector<double> el_valid;
            double cm_el_abs_corr = std::numeric_limits<double>::quiet_NaN();
            double cm_low_mid_ratio = std::numeric_limits<double>::quiet_NaN();

            const bool need_cm = (weight_corr_penalty.cm_el_corr.enabled ||
                                  weight_corr_penalty.cm_low_mid_ratio.enabled) && (n_used > 0);
            if (need_cm) {
                for (Eigen::Index is = 0; is < n_pts; ++is) {
                    const Eigen::Index i = is * sample_step;
                    if (i >= n_pts_full) {
                        break;
                    }
                    double sum = 0.0;
                    Eigen::Index count = 0;
                    for (Eigen::Index k = 0; k < n_used; ++k) {
                        const Eigen::Index det = det_keep[static_cast<std::size_t>(k)];
                        if (in.flags.data(i, det)) {
                            continue;
                        }
                        const double v = in.scans.data(i, det);
                        if (!std::isfinite(v)) {
                            continue;
                        }
                        const double z = (v - det_mean[static_cast<std::size_t>(k)]) /
                                         det_std[static_cast<std::size_t>(k)];
                        if (!std::isfinite(z)) {
                            continue;
                        }
                        sum += z;
                        count++;
                    }
                    if (count >= 2) {
                        cm(is) = sum / static_cast<double>(count);
                    }
                }

                if (weight_corr_penalty.cm_el_corr.enabled) {
                    std::optional<Eigen::VectorXd> native_tel_el;
                    const Eigen::VectorXd *tel_el = nullptr;
                    if (in.native_science_required()) {
                        const auto telescope_data =
                            in.require_native_scan()
                                .telescope_data_for_detector(start_index);
                        const auto el_it =
                            telescope_data.find("TelElAct");
                        if (el_it == telescope_data.end() ||
                            el_it->second.size() != n_pts_full) {
                            throw std::runtime_error(
                                "native-required weight correlation lacks network-native elevation");
                        }
                        native_tel_el = el_it->second;
                        tel_el = &*native_tel_el;
                    }
                    else {
                        const auto el_it =
                            in.tel_data.data.find("TelElAct");
                        if (el_it != in.tel_data.data.end()) {
                            tel_el = &el_it->second;
                        }
                    }
                    if (tel_el != nullptr) {
                        cm_valid.reserve(static_cast<std::size_t>(n_pts));
                        el_valid.reserve(static_cast<std::size_t>(n_pts));
                        for (Eigen::Index is = 0; is < n_pts; ++is) {
                            const Eigen::Index i = is * sample_step;
                            if (i >= n_pts_full || i >= tel_el->size()) {
                                break;
                            }
                            const double c = cm(is);
                            const double e = (*tel_el)(i);
                            if (!std::isfinite(c) || !std::isfinite(e)) {
                                continue;
                            }
                            cm_valid.push_back(c);
                            el_valid.push_back(e);
                        }
                        const double c = pearson_corr(cm_valid, el_valid);
                        if (std::isfinite(c)) {
                            cm_el_abs_corr = std::abs(c);
                        }
                    }
                }

                if (weight_corr_penalty.cm_low_mid_ratio.enabled) {
                    std::vector<double> cm_pts;
                    cm_pts.reserve(static_cast<std::size_t>(n_pts));
                    for (Eigen::Index is = 0; is < n_pts; ++is) {
                        const double c = cm(is);
                        if (std::isfinite(c)) {
                            cm_pts.push_back(c);
                        }
                    }
                    if (cm_pts.size() >= 8) {
                        const double cm_mean = std::accumulate(cm_pts.begin(), cm_pts.end(), 0.0) /
                                               static_cast<double>(cm_pts.size());
                        Eigen::VectorXd x = Eigen::VectorXd::Zero(n_pts);
                        for (Eigen::Index is = 0; is < n_pts; ++is) {
                            const double c = cm(is);
                            if (std::isfinite(c)) {
                                x(is) = c - cm_mean;
                            }
                        }
                        // mild taper to reduce leakage from scan edges
                        if (n_pts > 1) {
                            constexpr double two_pi = 6.283185307179586476925286766559;
                            for (Eigen::Index is = 0; is < n_pts; ++is) {
                                const double w = 0.5 * (1.0 - std::cos(
                                    two_pi * static_cast<double>(is) /
                                    static_cast<double>(n_pts - 1)));
                                x(is) *= w;
                            }
                        }

                        Eigen::FFT<double> fft;
                        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
                        fft.SetFlag(Eigen::FFT<double>::Unscaled);
                        Eigen::VectorXcd freq;
                        fft.fwd(freq, x);

                        const double fs_eff = telescope.d_fsmp / static_cast<double>(sample_step);
                        if (fs_eff > 0.0 && freq.size() > 0) {
                            double p_low = 0.0;
                            double p_mid = 0.0;
                            const auto &band = weight_corr_penalty.cm_low_mid_ratio;
                            for (Eigen::Index k = 1; k < freq.size(); ++k) {
                                const double f = static_cast<double>(k) * fs_eff / static_cast<double>(n_pts);
                                const double p = std::norm(freq(k));
                                if (f >= band.low_min_Hz && f < band.low_max_Hz) {
                                    p_low += p;
                                }
                                if (f >= band.mid_min_Hz && f < band.mid_max_Hz) {
                                    p_mid += p;
                                }
                            }
                            if (p_mid > 0.0 && std::isfinite(p_low) && std::isfinite(p_mid)) {
                                cm_low_mid_ratio = p_low / p_mid;
                            }
                        }
                    }
                }
            }

            double score_num = 0.0;
            double score_den = 0.0;

            {
                const auto [n, d] = score_metric(pair_med_abs_corr, weight_corr_penalty.pair_corr);
                score_num += n;
                score_den += d;
            }
            {
                const auto [n, d] = score_metric(cm_el_abs_corr, weight_corr_penalty.cm_el_corr);
                score_num += n;
                score_den += d;
            }
            {
                const auto [n, d] = score_metric(cm_low_mid_ratio, weight_corr_penalty.cm_low_mid_ratio);
                score_num += n;
                score_den += d;
            }

            double severity = 0.0;
            if (score_den > 0.0 && std::isfinite(score_num)) {
                severity = clamp01(score_num / score_den);
            }

            const double floor = clamp01(weight_corr_penalty.floor);
            const double exponent = std::max(0.0, weight_corr_penalty.exponent);
            double penalty_factor = 1.0;
            if (score_den > 0.0) {
                penalty_factor = floor + (1.0 - floor) * std::pow(clamp01(1.0 - severity), exponent);
            }
            if (!std::isfinite(penalty_factor)) {
                penalty_factor = 1.0;
            }
            penalty_factor = std::clamp(penalty_factor, floor, 1.0);

            Eigen::Index n_weighted = 0;
            for (Eigen::Index j = start_index; j < end_index; ++j) {
                if (apt["flag"](j) != 0) {
                    continue;
                }
                if (!std::isfinite(in.weights.data(j)) || in.weights.data(j) <= 0.0) {
                    continue;
                }
                in.weights.data(j) *= penalty_factor;
                n_weighted++;
            }

            penalty_summary.push_back(WeightCorrPenaltyDiagSummary{
                .nw = nw,
                .n_det_input = n_det_group,
                .n_det_candidates = n_candidates,
                .n_det_used = n_used,
                .n_det_weighted = n_weighted,
                .sample_step = sample_step,
                .pair_med_abs_corr = finite_or_nan(pair_med_abs_corr),
                .cm_el_abs_corr = finite_or_nan(cm_el_abs_corr),
                .cm_low_mid_ratio = finite_or_nan(cm_low_mid_ratio),
                .severity = severity,
                .penalty_factor = penalty_factor,
            });

            logger->info(
                "weight corr_penalty scan={} nw={} dets_in={} candidates={} used={} weighted={} "
                "pair_med_abs_corr={} cm_el_abs_corr={} cm_low_mid_ratio={} severity={} factor={}",
                scan_index_1based, nw, n_det_group, n_candidates, n_used, n_weighted,
                finite_or_nan(pair_med_abs_corr), finite_or_nan(cm_el_abs_corr),
                finite_or_nan(cm_low_mid_ratio), severity, penalty_factor);
        }
        {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            weight_corr_penalty_summary_by_scan[in.index.data] = std::move(penalty_summary);
        }
    }

    if (busy_row_suppression.enabled) {
        std::unordered_map<Eigen::Index, const SecondPassDiagSummary *> second_pass_by_nw;
        const auto second_pass_summary = snapshot_second_pass_summary(in.index.data);
        if (!second_pass_summary.empty()) {
            second_pass_by_nw.reserve(second_pass_summary.size());
            for (const auto &row : second_pass_summary) {
                second_pass_by_nw[row.nw] = &row;
            }
        } else {
            logger->warn(
                "weighting.busy_row_suppression enabled but no second-pass diagnostics were available for scan={}",
                scan_index_1based);
        }

        const double suppression_factor = std::clamp(busy_row_suppression.factor, 0.0, 1.0);
        std::vector<BusyRowSuppressionDiagSummary> suppression_summary;
        suppression_summary.reserve(static_cast<std::size_t>(nw_limits.size()));

        for (const auto &[nw, limits] : nw_limits) {
            const auto [start_index, end_index] = limits;
            BusyRowSuppressionDiagSummary summary;
            summary.nw = nw;

            const auto second_pass_nw_it = second_pass_by_nw.find(nw);
            if (second_pass_nw_it != second_pass_by_nw.end() && second_pass_nw_it->second != nullptr) {
                const auto &diag = *second_pass_nw_it->second;
                summary.busy_network_vetoed = diag.busy_network_vetoed;
                summary.n_candidate_clusters = diag.n_candidate_clusters;
                summary.max_unflagged_residual_z = finite_or_nan(diag.max_unflagged_residual_z);
            }

            const bool busy_ok = !busy_row_suppression.require_busy_veto || summary.busy_network_vetoed;
            const bool candidate_ok = summary.n_candidate_clusters >= busy_row_suppression.min_candidate_clusters;
            const bool residual_ok = std::isfinite(summary.max_unflagged_residual_z) &&
                                     summary.max_unflagged_residual_z >= busy_row_suppression.min_max_unflagged_residual_z;
            const bool should_suppress = busy_ok && candidate_ok && residual_ok && suppression_factor < 1.0;

            if (should_suppress) {
                for (Eigen::Index j = start_index; j < end_index; ++j) {
                    if (apt["flag"](j) != 0) {
                        continue;
                    }
                    if (!std::isfinite(in.weights.data(j)) || in.weights.data(j) <= 0.0) {
                        continue;
                    }
                    in.weights.data(j) *= suppression_factor;
                    summary.n_det_weighted++;
                }
            }
            summary.applied = should_suppress && summary.n_det_weighted > 0;
            summary.factor = summary.applied ? suppression_factor : 1.0;

            if (summary.applied) {
                logger->info(
                    "weight busy_row_suppression scan={} nw={} busy={} n_candidate_clusters={} "
                    "max_unflagged_residual_z={} factor={} weighted={}",
                    scan_index_1based, nw, summary.busy_network_vetoed, summary.n_candidate_clusters,
                    summary.max_unflagged_residual_z, summary.factor, summary.n_det_weighted);
            }

            suppression_summary.push_back(summary);
        }

        {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            busy_row_suppression_summary_by_scan[in.index.data] = std::move(suppression_summary);
        }
    }

    Eigen::Index n_apt_unflagged = 0;
    Eigen::Index n_nonfinite = 0;
    Eigen::Index n_positive = 0;
    Eigen::Index n_zero = 0;
    Eigen::Index n_negative = 0;
    for (Eigen::Index i = 0; i < n_dets; ++i) {
        if (apt["flag"](i) == 0) {
            n_apt_unflagged++;
        }
        const auto w = in.weights.data(i);
        if (!std::isfinite(w)) {
            n_nonfinite++;
        } else if (w > 0) {
            n_positive++;
        } else if (w == 0) {
            n_zero++;
        } else {
            n_negative++;
        }
    }
    logger->info(
        "weight calc summary scan={} type={} n_dets={} apt_unflagged={} "
        "positive={} zero={} negative={} nonfinite={}",
        scan_index_1based, weighting_type, n_dets, n_apt_unflagged, n_positive,
        n_zero, n_negative, n_nonfinite);
}

template <typename calib_t>
auto PTCProc::reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    const auto scan_index_1based = static_cast<long long>(in.index.data) + 1;
    static std::atomic<long long> reset_weights_call_counter{0};
    const auto reset_call_id = ++reset_weights_call_counter;

    // only need to run if median weight factor >=1
    if (med_weight_factor >= 1 || lower_weight_factor > 0 || upper_weight_factor > 0) {
        // number of detectors
        Eigen::Index n_dets = in.scans.data.cols();

        // get group limits
        auto grp_limits = get_grouping("array", calib, n_dets);

        logger->info(
            "resetting weights call={} scan={} map_grouping={} n_dets={} "
            "med_weight_factor={} lower_weight_factor={} upper_weight_factor={}",
            reset_call_id, scan_index_1based, map_grouping, n_dets,
            med_weight_factor, lower_weight_factor, upper_weight_factor);

        // collect detectors that are un-flagged and have non-zero weights
        for (auto const& [key, val] : grp_limits) {
            // weights for current group
            auto grp_weights = in.weights.data(Eigen::seq(std::get<0>(grp_limits[key]),
                                                         std::get<1>(grp_limits[key])-1));
            const auto group_start = std::get<0>(grp_limits[key]);
            const auto group_end = std::get<1>(grp_limits[key]);
            const auto n_group_dets = group_end - group_start;
            // number of unflagged detectors, and unflagged with positive weights
            Eigen::Index n_unflagged = 0;
            Eigen::Index n_good_dets = 0;
            Eigen::Index n_nonfinite_weights = 0;
            Eigen::Index n_nonpositive_unflagged = 0;
            // start index of current group
            Eigen::Index j = group_start;

            // loop through detectors in current group
            for (Eigen::Index m=0; m<grp_weights.size(); ++m) {
                if (!std::isfinite(grp_weights(m))) {
                    n_nonfinite_weights++;
                }
                // count unflagged detectors
                if (calib.apt["flag"](j)==0) {
                    n_unflagged++;
                    if (std::isfinite(grp_weights(m)) && grp_weights(m) > 0) {
                        n_good_dets++;
                    } else {
                        n_nonpositive_unflagged++;
                    }
                }
                j++;
            }

            // to hold good detectors
            Eigen::VectorXd good_wt;

            // if good detectors were found
            if (n_good_dets>0) {
                good_wt.resize(n_good_dets);

                // remove flagged dets
                j = std::get<0>(grp_limits[key]);
                Eigen::Index k = 0;
                for (Eigen::Index m=0; m<grp_weights.size(); ++m) {
                    if (calib.apt["flag"](j)==0 &&
                        std::isfinite(grp_weights(m)) && grp_weights(m)>0) {
                        good_wt(k) = grp_weights(m);
                        k++;
                    }
                    j++;
                }
            }
            // otherwise leave the group without a median; non-finite medians make
            // the threshold cuts non-deterministic.
            else {
                good_wt.resize(0);
            }

            if (good_wt.size() <= 0) {
                logger->warn(
                    "weight audit call={} scan={} array={} skipped: no finite positive unflagged weights",
                    reset_call_id, scan_index_1based, key);
                continue;
            }
            // get median weight
            auto med_wt = tula::alg::median(good_wt);
            if (!std::isfinite(med_wt) || med_wt <= 0.0) {
                logger->warn(
                    "weight audit call={} scan={} array={} skipped: non-finite median weight",
                    reset_call_id, scan_index_1based, key);
                continue;
            }
            const auto lower_limit =
                lower_weight_factor != 0 ? lower_weight_factor * med_wt : 0.0;
            const auto upper_limit =
                upper_weight_factor != 0 ? upper_weight_factor * med_wt : 0.0;
            // store median weights
            in.median_weights.data.push_back(med_wt);

            int outliers = 0;
            int n_dets_low = 0;
            int n_dets_high = 0;

            // start index of current group
            j = group_start;
            // loop through detectors in current group
            for (Eigen::Index m=0; m<grp_weights.size(); ++m) {
                // if detector weight is med_weight_factor times larger than med_wt
                if (med_weight_factor >=1 && in.weights.data(j) > med_weight_factor*med_wt) {
                    // reset high weights to median
                    in.weights.data(j) = med_wt;
                    outliers++;
                }

                // only run if unflagged already
                if (calib.apt["flag"](j)==0) {
                    // flag those below limit
                    if ((in.weights.data(j) < (lower_weight_factor*med_wt)) && lower_weight_factor!=0) {
                        in.flags.data.col(j).setOnes();
                        if (citlali::config::is_detector_map_grouping(map_grouping)) {
                            calib_scan.apt["flag"](j) = 1;
                        }
                        in.n_dets_low++;
                        n_dets_low++;
                    }

                    // flag those above limit
                    if ((in.weights.data(j) > (upper_weight_factor*med_wt)) && upper_weight_factor!=0) {
                        in.flags.data.col(j).setOnes();
                        if (citlali::config::is_detector_map_grouping(map_grouping)) {
                            calib_scan.apt["flag"](j) = 1;
                        }
                        in.n_dets_high++;
                        n_dets_high++;
                    }
                }
                j++;
            }
            logger->info(
                "weight audit call={} scan={} array={} idx_range=[{}, {}) "
                "group_dets={} apt_unflagged={} apt_flagged={} "
                "positive_unflagged={} nonpositive_unflagged={} nonfinite_weights={} "
                "median_weight={:.4g} lower_limit={:.4g} upper_limit={:.4g}",
                reset_call_id, scan_index_1based, key, group_start, group_end,
                n_group_dets, n_unflagged, n_group_dets - n_unflagged, n_good_dets,
                n_nonpositive_unflagged, n_nonfinite_weights, med_wt, lower_limit,
                upper_limit);
            logger->info(
                "weight flags call={} scan={} array={} outlier_resets={} "
                "below_limit={}/{} above_limit={}/{}",
                reset_call_id, scan_index_1based, key, outliers, n_dets_low,
                n_unflagged, n_dets_high, n_unflagged);

            // sanity checks for impossible counter combinations
            try {
                citlali::pipeline::require_valid_weight_counters(
                    n_group_dets, n_unflagged, n_good_dets,
                    n_dets_low, n_dets_high);
            } catch (const citlali::error::Error &) {
                logger->error(
                    "weight counter invariant failure call={} scan={} array={} "
                    "group_dets={} apt_unflagged={} positive_unflagged={} "
                    "below_count={} above_count={} outlier_count={}",
                    reset_call_id, scan_index_1based, key, n_group_dets,
                    n_unflagged, n_good_dets, n_dets_low, n_dets_high, outliers);
                const auto n_dump = std::min<Eigen::Index>(grp_weights.size(), 10);
                for (Eigen::Index m = 0; m < n_dump; ++m) {
                    const auto det_index = group_start + m;
                    logger->error(
                        "weight counter dump call={} scan={} array={} m={} det_index={} apt_flag={} weight={}",
                        reset_call_id, scan_index_1based, key, m, det_index,
                        calib.apt["flag"](det_index), in.weights.data(det_index));
                }
                throw;
            }
        }

        // set up scan calib
        calib_scan.setup();
    }
    return std::move(calib_scan);
}

template <typename calib_t, typename pointing_offset_t>
void PTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath, std::string map_grouping,
                              std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, calib_t &calib,
                              bool apply_det_offsets, Eigen::Index scan_row_index, bool mini_output) {

    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        // open netcdf file
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        NcFile fo(filepath, netCDF::NcFile::write);
        const auto n_pts_before_append = fo.getDim("n_pts").getSize();
        const auto n_dets_before_append = fo.getDim("n_dets").getSize();

        // append common time chunk variables
        append_base_to_netcdf(fo, in, map_grouping, pixel_axes, pointing_offsets_arcsec, calib, apply_det_offsets,
                              scan_row_index, false, mini_output);

        // get dimensions
        NcDim n_dets_dim = fo.getDim("n_dets");

        // number of detectors currently in file
        unsigned long n_dets_exists = n_dets_dim.getSize();

        // append weights
        const auto scan_row = static_cast<unsigned long>((scan_row_index >= 0) ? scan_row_index : in.index.data);
        std::vector<std::size_t> start_index_weights = {scan_row, 0};
        std::vector<std::size_t> size_weights = {1, n_dets_exists};

        // get weight variable
        NcVar weights_v = fo.getVar("weights");

        // add weights to tod output
        weights_v.putVar(start_index_weights, size_weights, in.weights.data.data());

        const auto second_pass_summary = snapshot_second_pass_summary(in.index.data);
        Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic> second_pass_added_flags;
        bool have_second_pass_added_flags = false;
        {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            const auto second_pass_added_it = second_pass_added_flags_by_scan.find(in.index.data);
            if (second_pass_added_it != second_pass_added_flags_by_scan.end()) {
                second_pass_added_flags = second_pass_added_it->second;
                have_second_pass_added_flags = true;
            }
        }
        NcVar second_pass_added_v = fo.getVar("ptc_second_pass_added_flag");
        if (!second_pass_added_v.isNull() && have_second_pass_added_flags) {
            std::vector<std::size_t> start_index = {n_pts_before_append, 0};
            std::vector<std::size_t> size = {1, n_dets_before_append};
            const auto &added = second_pass_added_flags;
            const auto n_rows = std::min<unsigned long>(
                static_cast<unsigned long>(added.rows()),
                static_cast<unsigned long>(in.scans.data.rows()));
            for (unsigned long i = 0; i < n_rows; ++i) {
                start_index[0] = n_pts_before_append + i;
                Eigen::Matrix<signed char, 1, Eigen::Dynamic> row = added.row(static_cast<Eigen::Index>(i));
                second_pass_added_v.putVar(start_index, size, row.data());
            }
        }

        Eigen::VectorXi corr_group_ids;
        bool have_corr_group_ids = false;
        std::vector<CorrNWDiagSummary> corr_summary;
        std::vector<WeightCorrPenaltyDiagSummary> weight_corr_penalty_summary;
        std::vector<BusyRowSuppressionDiagSummary> busy_row_suppression_summary;
        std::vector<AdaptiveSelectorDiagSummary> adaptive_selector_summary;
        {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            const auto corr_groups_it = corr_nw_group_ids_by_scan.find(in.index.data);
            if (corr_groups_it != corr_nw_group_ids_by_scan.end()) {
                corr_group_ids = corr_groups_it->second;
                have_corr_group_ids = true;
            }
            const auto corr_summary_it = corr_nw_summary_by_scan.find(in.index.data);
            if (corr_summary_it != corr_nw_summary_by_scan.end()) {
                corr_summary = corr_summary_it->second;
            }
            const auto weight_corr_penalty_it = weight_corr_penalty_summary_by_scan.find(in.index.data);
            if (weight_corr_penalty_it != weight_corr_penalty_summary_by_scan.end()) {
                weight_corr_penalty_summary = weight_corr_penalty_it->second;
            }
            const auto busy_row_suppression_it = busy_row_suppression_summary_by_scan.find(in.index.data);
            if (busy_row_suppression_it != busy_row_suppression_summary_by_scan.end()) {
                busy_row_suppression_summary = busy_row_suppression_it->second;
            }
            const auto adaptive_selector_it = adaptive_selector_summary_by_scan.find(in.index.data);
            if (adaptive_selector_it != adaptive_selector_summary_by_scan.end()) {
                adaptive_selector_summary = adaptive_selector_it->second;
            }
        }
        const int corr_fill_value = -2147483647;

        // optional corr_nw diagnostics: detector group IDs per scan x detector
        NcVar corr_group_id_v = fo.getVar("corr_nw_group_id");
        if (!corr_group_id_v.isNull()) {
            std::vector<int> group_ids(static_cast<std::size_t>(n_dets_exists), corr_fill_value);
            if (have_corr_group_ids) {
                const auto n_copy = std::min<unsigned long>(
                    n_dets_exists,
                    static_cast<unsigned long>(corr_group_ids.size()));
                for (unsigned long i = 0; i < n_copy; ++i) {
                    group_ids[static_cast<std::size_t>(i)] =
                        static_cast<int>(corr_group_ids(static_cast<Eigen::Index>(i)));
                }
            }
            corr_group_id_v.putVar(start_index_weights, size_weights, group_ids.data());
        }

        // optional corr_nw diagnostics: per-network summaries per scan
        NcVar corr_n_groups_v = fo.getVar("corr_nw_n_groups");
        if (!corr_n_groups_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_corr");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                std::vector<int> v_n_groups(n_nws, corr_fill_value);
                std::vector<int> v_n_groups_raw(n_nws, corr_fill_value);
                std::vector<int> v_n_det_input(n_nws, corr_fill_value);
                std::vector<int> v_n_det_candidates(n_nws, corr_fill_value);
                std::vector<int> v_n_det_used(n_nws, corr_fill_value);
                std::vector<int> v_n_det_grouped(n_nws, corr_fill_value);
                std::vector<int> v_n_det_ungrouped(n_nws, corr_fill_value);
                std::vector<int> v_sample_step(n_nws, corr_fill_value);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }

                if (!corr_summary.empty()) {
                    for (const auto &row : corr_summary) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_n_groups[j] = static_cast<int>(row.n_groups_final);
                        v_n_groups_raw[j] = static_cast<int>(row.n_groups_raw);
                        v_n_det_input[j] = static_cast<int>(row.n_det_input);
                        v_n_det_candidates[j] = static_cast<int>(row.n_det_candidates);
                        v_n_det_used[j] = static_cast<int>(row.n_det_used);
                        v_n_det_grouped[j] = static_cast<int>(row.n_det_grouped);
                        v_n_det_ungrouped[j] = static_cast<int>(row.n_det_ungrouped);
                        v_sample_step[j] = static_cast<int>(row.sample_step);
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};

                corr_n_groups_v.putVar(start_scan_nw, size_scan_nw, v_n_groups.data());
                fo.getVar("corr_nw_n_groups_raw").putVar(start_scan_nw, size_scan_nw, v_n_groups_raw.data());
                fo.getVar("corr_nw_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
                fo.getVar("corr_nw_n_det_candidates").putVar(start_scan_nw, size_scan_nw, v_n_det_candidates.data());
                fo.getVar("corr_nw_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
                fo.getVar("corr_nw_n_det_grouped").putVar(start_scan_nw, size_scan_nw, v_n_det_grouped.data());
                fo.getVar("corr_nw_n_det_ungrouped").putVar(start_scan_nw, size_scan_nw, v_n_det_ungrouped.data());
                fo.getVar("corr_nw_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
            }
        }

        // optional diagnostics: per-network weight penalty summaries per scan
        NcVar wcorr_factor_v = fo.getVar("weight_corr_penalty_factor");
        if (!wcorr_factor_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_wcorr");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                const double fill_double = std::numeric_limits<double>::quiet_NaN();
                std::vector<double> v_factor(n_nws, fill_double);
                std::vector<double> v_severity(n_nws, fill_double);
                std::vector<double> v_pair_corr(n_nws, fill_double);
                std::vector<double> v_cm_el_corr(n_nws, fill_double);
                std::vector<double> v_cm_low_mid(n_nws, fill_double);
                std::vector<int> v_n_det_input(n_nws, corr_fill_value);
                std::vector<int> v_n_det_candidates(n_nws, corr_fill_value);
                std::vector<int> v_n_det_used(n_nws, corr_fill_value);
                std::vector<int> v_n_det_weighted(n_nws, corr_fill_value);
                std::vector<int> v_sample_step(n_nws, corr_fill_value);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }

                if (!weight_corr_penalty_summary.empty()) {
                    for (const auto &row : weight_corr_penalty_summary) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_factor[j] = row.penalty_factor;
                        v_severity[j] = row.severity;
                        v_pair_corr[j] = row.pair_med_abs_corr;
                        v_cm_el_corr[j] = row.cm_el_abs_corr;
                        v_cm_low_mid[j] = row.cm_low_mid_ratio;
                        v_n_det_input[j] = static_cast<int>(row.n_det_input);
                        v_n_det_candidates[j] = static_cast<int>(row.n_det_candidates);
                        v_n_det_used[j] = static_cast<int>(row.n_det_used);
                        v_n_det_weighted[j] = static_cast<int>(row.n_det_weighted);
                        v_sample_step[j] = static_cast<int>(row.sample_step);
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};

                wcorr_factor_v.putVar(start_scan_nw, size_scan_nw, v_factor.data());
                fo.getVar("weight_corr_penalty_severity").putVar(start_scan_nw, size_scan_nw, v_severity.data());
                fo.getVar("weight_corr_penalty_pair_med_abs_corr").putVar(start_scan_nw, size_scan_nw, v_pair_corr.data());
                fo.getVar("weight_corr_penalty_cm_el_abs_corr").putVar(start_scan_nw, size_scan_nw, v_cm_el_corr.data());
                fo.getVar("weight_corr_penalty_cm_low_mid_ratio").putVar(start_scan_nw, size_scan_nw, v_cm_low_mid.data());
                fo.getVar("weight_corr_penalty_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
                fo.getVar("weight_corr_penalty_n_det_candidates").putVar(start_scan_nw, size_scan_nw, v_n_det_candidates.data());
                fo.getVar("weight_corr_penalty_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
                fo.getVar("weight_corr_penalty_n_det_weighted").putVar(start_scan_nw, size_scan_nw, v_n_det_weighted.data());
                fo.getVar("weight_corr_penalty_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
            }
        }

        NcVar wbusy_applied_v = fo.getVar("weight_busy_row_suppression_applied");
        if (!wbusy_applied_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_busy_row_suppression");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                const double fill_double = std::numeric_limits<double>::quiet_NaN();
                std::vector<int> v_applied(n_nws, corr_fill_value);
                std::vector<int> v_busy(n_nws, corr_fill_value);
                std::vector<int> v_n_candidate_clusters(n_nws, corr_fill_value);
                std::vector<int> v_n_det_weighted(n_nws, corr_fill_value);
                std::vector<double> v_factor(n_nws, fill_double);
                std::vector<double> v_max_resid_z(n_nws, fill_double);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }

                if (!busy_row_suppression_summary.empty()) {
                    for (const auto &row : busy_row_suppression_summary) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_applied[j] = row.applied ? 1 : 0;
                        v_busy[j] = row.busy_network_vetoed ? 1 : 0;
                        v_n_candidate_clusters[j] = static_cast<int>(row.n_candidate_clusters);
                        v_n_det_weighted[j] = static_cast<int>(row.n_det_weighted);
                        v_factor[j] = row.factor;
                        v_max_resid_z[j] = row.max_unflagged_residual_z;
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};
                wbusy_applied_v.putVar(start_scan_nw, size_scan_nw, v_applied.data());
                fo.getVar("weight_busy_row_suppression_busy_network_vetoed").putVar(start_scan_nw, size_scan_nw, v_busy.data());
                fo.getVar("weight_busy_row_suppression_n_candidate_clusters").putVar(start_scan_nw, size_scan_nw, v_n_candidate_clusters.data());
                fo.getVar("weight_busy_row_suppression_n_det_weighted").putVar(start_scan_nw, size_scan_nw, v_n_det_weighted.data());
                fo.getVar("weight_busy_row_suppression_factor").putVar(start_scan_nw, size_scan_nw, v_factor.data());
                fo.getVar("weight_busy_row_suppression_max_unflagged_residual_z").putVar(start_scan_nw, size_scan_nw, v_max_resid_z.data());
            }
        }

        NcVar adaptive_chosen_k_v = fo.getVar("adaptive_pca_chosen_k");
        if (!adaptive_chosen_k_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_adaptive_pca");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                const double fill_double = std::numeric_limits<double>::quiet_NaN();
                std::vector<int> v_selector_used(n_nws, corr_fill_value);
                std::vector<int> v_selector_fallback(n_nws, corr_fill_value);
                std::vector<int> v_baseline_k(n_nws, corr_fill_value);
                std::vector<int> v_chosen_k(n_nws, corr_fill_value);
                std::vector<int> v_runnerup_k(n_nws, corr_fill_value);
                std::vector<int> v_n_candidates(n_nws, corr_fill_value);
                std::vector<int> v_n_det_input(n_nws, corr_fill_value);
                std::vector<int> v_n_det_used(n_nws, corr_fill_value);
                std::vector<int> v_n_time_used(n_nws, corr_fill_value);
                std::vector<int> v_sample_step(n_nws, corr_fill_value);
                std::vector<double> v_chosen_score(n_nws, fill_double);
                std::vector<double> v_runnerup_score(n_nws, fill_double);
                std::vector<double> v_score_margin(n_nws, fill_double);
                std::vector<double> v_chosen_med_abs_corr(n_nws, fill_double);
                std::vector<double> v_chosen_cm_low_mid_ratio(n_nws, fill_double);
                std::vector<double> v_chosen_tail4_binom_z(n_nws, fill_double);
                std::vector<double> v_chosen_top_mode_frac(n_nws, fill_double);
                std::vector<double> v_eig_solve_msec(n_nws, fill_double);
                std::vector<double> v_candidate_eval_msec(n_nws, fill_double);
                std::vector<double> v_total_msec(n_nws, fill_double);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }

                if (!adaptive_selector_summary.empty()) {
                    for (const auto &row : adaptive_selector_summary) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_selector_used[j] = row.selector_used;
                        v_selector_fallback[j] = row.selector_fallback;
                        v_baseline_k[j] = static_cast<int>(row.baseline_k);
                        v_chosen_k[j] = static_cast<int>(row.chosen_k);
                        v_runnerup_k[j] = static_cast<int>(row.runnerup_k);
                        v_n_candidates[j] = static_cast<int>(row.n_candidates);
                        v_n_det_input[j] = static_cast<int>(row.n_det_input);
                        v_n_det_used[j] = static_cast<int>(row.n_det_used);
                        v_n_time_used[j] = static_cast<int>(row.n_time_used);
                        v_sample_step[j] = static_cast<int>(row.sample_step);
                        v_chosen_score[j] = row.chosen_score;
                        v_runnerup_score[j] = row.runnerup_score;
                        v_score_margin[j] = row.score_margin;
                        v_chosen_med_abs_corr[j] = row.chosen_med_abs_corr;
                        v_chosen_cm_low_mid_ratio[j] = row.chosen_cm_low_mid_ratio;
                        v_chosen_tail4_binom_z[j] = row.chosen_tail4_binom_z;
                        v_chosen_top_mode_frac[j] = row.chosen_top_mode_frac;
                        v_eig_solve_msec[j] = row.eig_solve_msec;
                        v_candidate_eval_msec[j] = row.candidate_eval_msec;
                        v_total_msec[j] = row.total_msec;
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};
                fo.getVar("adaptive_pca_selector_used").putVar(start_scan_nw, size_scan_nw, v_selector_used.data());
                fo.getVar("adaptive_pca_selector_fallback").putVar(start_scan_nw, size_scan_nw, v_selector_fallback.data());
                fo.getVar("adaptive_pca_baseline_k").putVar(start_scan_nw, size_scan_nw, v_baseline_k.data());
                adaptive_chosen_k_v.putVar(start_scan_nw, size_scan_nw, v_chosen_k.data());
                fo.getVar("adaptive_pca_runnerup_k").putVar(start_scan_nw, size_scan_nw, v_runnerup_k.data());
                fo.getVar("adaptive_pca_n_candidates").putVar(start_scan_nw, size_scan_nw, v_n_candidates.data());
                fo.getVar("adaptive_pca_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
                fo.getVar("adaptive_pca_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
                fo.getVar("adaptive_pca_n_time_used").putVar(start_scan_nw, size_scan_nw, v_n_time_used.data());
                fo.getVar("adaptive_pca_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
                fo.getVar("adaptive_pca_chosen_score").putVar(start_scan_nw, size_scan_nw, v_chosen_score.data());
                fo.getVar("adaptive_pca_runnerup_score").putVar(start_scan_nw, size_scan_nw, v_runnerup_score.data());
                fo.getVar("adaptive_pca_score_margin").putVar(start_scan_nw, size_scan_nw, v_score_margin.data());
                fo.getVar("adaptive_pca_chosen_med_abs_corr").putVar(start_scan_nw, size_scan_nw, v_chosen_med_abs_corr.data());
                fo.getVar("adaptive_pca_chosen_cm_low_mid_ratio").putVar(start_scan_nw, size_scan_nw, v_chosen_cm_low_mid_ratio.data());
                fo.getVar("adaptive_pca_chosen_tail4_binom_z").putVar(start_scan_nw, size_scan_nw, v_chosen_tail4_binom_z.data());
                fo.getVar("adaptive_pca_chosen_top_mode_frac").putVar(start_scan_nw, size_scan_nw, v_chosen_top_mode_frac.data());
                fo.getVar("adaptive_pca_eig_solve_msec").putVar(start_scan_nw, size_scan_nw, v_eig_solve_msec.data());
                fo.getVar("adaptive_pca_candidate_eval_msec").putVar(start_scan_nw, size_scan_nw, v_candidate_eval_msec.data());
                fo.getVar("adaptive_pca_total_msec").putVar(start_scan_nw, size_scan_nw, v_total_msec.data());
            }
        }

        NcVar second_pass_busy_v = fo.getVar("ptc_second_pass_busy_network_vetoed");
        if (!second_pass_busy_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_ptc_second_pass");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                const double fill_double = std::numeric_limits<double>::quiet_NaN();
                std::vector<int> v_busy(n_nws, corr_fill_value);
                std::vector<int> v_n_candidate_clusters(n_nws, corr_fill_value);
                std::vector<int> v_n_candidate_events(n_nws, corr_fill_value);
                std::vector<int> v_n_accepted_clusters(n_nws, corr_fill_value);
                std::vector<int> v_n_accepted_events(n_nws, corr_fill_value);
                std::vector<int> v_n_rejected_clusters(n_nws, corr_fill_value);
                std::vector<int> v_n_rejected_events(n_nws, corr_fill_value);
                std::vector<int> v_n_source_protected_clusters(n_nws, corr_fill_value);
                std::vector<int> v_n_source_protected_events(n_nws, corr_fill_value);
                std::vector<int> v_n_det_with_added_flags(n_nws, corr_fill_value);
                std::vector<int> v_max_resid_uid(n_nws, corr_fill_value);
                std::vector<int> v_top_cluster_sample(n_nws, corr_fill_value);
                std::vector<int> v_top_cluster_n_detectors(n_nws, corr_fill_value);
                std::vector<int> v_top_cluster_n_events(n_nws, corr_fill_value);
                std::vector<int> v_top_event_kind(n_nws, corr_fill_value);
                std::vector<int> v_top_event_uid(n_nws, corr_fill_value);
                std::vector<int> v_top_event_sample(n_nws, corr_fill_value);
                std::vector<double> v_existing_frac(n_nws, fill_double);
                std::vector<double> v_proposed_frac(n_nws, fill_double);
                std::vector<double> v_new_frac(n_nws, fill_double);
                std::vector<double> v_max_resid_z(n_nws, fill_double);
                std::vector<double> v_top_cluster_peak(n_nws, fill_double);
                std::vector<double> v_top_event_score(n_nws, fill_double);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }
                if (!second_pass_summary.empty()) {
                    for (const auto &row : second_pass_summary) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_busy[j] = row.busy_network_vetoed ? 1 : 0;
                        v_n_candidate_clusters[j] = static_cast<int>(row.n_candidate_clusters);
                        v_n_candidate_events[j] = static_cast<int>(row.n_candidate_events);
                        v_n_accepted_clusters[j] = static_cast<int>(row.n_accepted_clusters);
                        v_n_accepted_events[j] = static_cast<int>(row.n_accepted_events);
                        v_n_rejected_clusters[j] = static_cast<int>(row.n_rejected_clusters);
                        v_n_rejected_events[j] = static_cast<int>(row.n_rejected_events);
                        v_n_source_protected_clusters[j] = static_cast<int>(row.n_source_protected_clusters);
                        v_n_source_protected_events[j] = static_cast<int>(row.n_source_protected_events);
                        v_n_det_with_added_flags[j] = static_cast<int>(row.n_det_with_added_flags);
                        v_max_resid_uid[j] = row.max_unflagged_residual_uid;
                        v_top_cluster_sample[j] = row.top_candidate_cluster_sample;
                        v_top_cluster_n_detectors[j] = static_cast<int>(row.top_candidate_cluster_n_detectors);
                        v_top_cluster_n_events[j] = static_cast<int>(row.top_candidate_cluster_n_events);
                        v_top_event_kind[j] = row.top_event.kind_code();
                        v_top_event_uid[j] = row.top_event_uid;
                        v_top_event_sample[j] = row.top_event.sample;
                        v_existing_frac[j] = row.existing_flagged_fraction;
                        v_proposed_frac[j] = row.proposed_flagged_fraction;
                        v_new_frac[j] = row.newly_flagged_fraction;
                        v_max_resid_z[j] = row.max_unflagged_residual_z;
                        v_top_cluster_peak[j] = row.top_candidate_cluster_peak_score;
                        v_top_event_score[j] = row.top_event.score;
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};
                second_pass_busy_v.putVar(start_scan_nw, size_scan_nw, v_busy.data());
                fo.getVar("ptc_second_pass_n_candidate_clusters").putVar(start_scan_nw, size_scan_nw, v_n_candidate_clusters.data());
                fo.getVar("ptc_second_pass_n_candidate_events").putVar(start_scan_nw, size_scan_nw, v_n_candidate_events.data());
                fo.getVar("ptc_second_pass_n_accepted_clusters").putVar(start_scan_nw, size_scan_nw, v_n_accepted_clusters.data());
                fo.getVar("ptc_second_pass_n_accepted_events").putVar(start_scan_nw, size_scan_nw, v_n_accepted_events.data());
                fo.getVar("ptc_second_pass_n_rejected_clusters").putVar(start_scan_nw, size_scan_nw, v_n_rejected_clusters.data());
                fo.getVar("ptc_second_pass_n_rejected_events").putVar(start_scan_nw, size_scan_nw, v_n_rejected_events.data());
                fo.getVar("ptc_second_pass_n_source_protected_clusters").putVar(start_scan_nw, size_scan_nw, v_n_source_protected_clusters.data());
                fo.getVar("ptc_second_pass_n_source_protected_events").putVar(start_scan_nw, size_scan_nw, v_n_source_protected_events.data());
                fo.getVar("ptc_second_pass_n_det_with_added_flags").putVar(start_scan_nw, size_scan_nw, v_n_det_with_added_flags.data());
                fo.getVar("ptc_second_pass_max_unflagged_residual_uid").putVar(start_scan_nw, size_scan_nw, v_max_resid_uid.data());
                fo.getVar("ptc_second_pass_top_candidate_cluster_sample").putVar(start_scan_nw, size_scan_nw, v_top_cluster_sample.data());
                fo.getVar("ptc_second_pass_top_candidate_cluster_n_detectors").putVar(start_scan_nw, size_scan_nw, v_top_cluster_n_detectors.data());
                fo.getVar("ptc_second_pass_top_candidate_cluster_n_events").putVar(start_scan_nw, size_scan_nw, v_top_cluster_n_events.data());
                fo.getVar("ptc_second_pass_top_event_kind").putVar(start_scan_nw, size_scan_nw, v_top_event_kind.data());
                fo.getVar("ptc_second_pass_top_event_uid").putVar(start_scan_nw, size_scan_nw, v_top_event_uid.data());
                fo.getVar("ptc_second_pass_top_event_sample").putVar(start_scan_nw, size_scan_nw, v_top_event_sample.data());
                fo.getVar("ptc_second_pass_existing_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_existing_frac.data());
                fo.getVar("ptc_second_pass_proposed_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_proposed_frac.data());
                fo.getVar("ptc_second_pass_newly_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_new_frac.data());
                fo.getVar("ptc_second_pass_max_unflagged_residual_z").putVar(start_scan_nw, size_scan_nw, v_max_resid_z.data());
                fo.getVar("ptc_second_pass_top_candidate_cluster_peak_score").putVar(start_scan_nw, size_scan_nw, v_top_cluster_peak.data());
                fo.getVar("ptc_second_pass_top_event_score").putVar(start_scan_nw, size_scan_nw, v_top_event_score.data());
            }
        }

        // drop per-scan diagnostics once persisted to netCDF
        {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            corr_nw_group_ids_by_scan.erase(in.index.data);
            corr_nw_summary_by_scan.erase(in.index.data);
            weight_corr_penalty_summary_by_scan.erase(in.index.data);
            busy_row_suppression_summary_by_scan.erase(in.index.data);
            adaptive_selector_summary_by_scan.erase(in.index.data);
            second_pass_summary_by_scan.erase(in.index.data);
            second_pass_added_flags_by_scan.erase(in.index.data);
        }

        if (write_evals) {
            if (cleaner.n_calc <= 0 || in.evals.data.empty()) {
                logger->debug("n_calc=0 or evals empty; skipping eval/evec output");
                // sync file to make sure it gets updated
                fo.sync();
                // close file
                fo.close();
                logger->info("tod chunk written to {}", filepath);
                return;
            }
            // get number of eigenvalues to save
            NcDim n_eigs_dim = fo.getDim("n_eigs");
            netCDF::NcDim n_eig_grp_dim = fo.getDim("n_eig_grp");

            // if eigenvalue dimension is null, add it
            if (n_eig_grp_dim.isNull()) {
                n_eig_grp_dim = fo.addDim("n_eig_grp",in.evals.data[0].size());
            }

            // dimensions for eigenvalue data
            std::vector<netCDF::NcDim> eval_dims = {n_eig_grp_dim, n_eigs_dim};

            // loop through cleaner gropuing
            for (Eigen::Index i=0; i<in.evals.data.size(); ++i) {
                NcVar eval_v = fo.addVar("evals_" + cleaner.grouping[i] + "_" + std::to_string(i) +
                                             "_chunk_" + std::to_string(in.index.data), netCDF::ncDouble,eval_dims);
                std::vector<std::size_t> start_eig_index = {0, 0};
                std::vector<std::size_t> size = {1, TULA_SIZET(cleaner.n_calc)};

                // loop through eigenvalues in current group
                for (const auto &evals: in.evals.data[i]) {
                    eval_v.putVar(start_eig_index,size,evals.data());
                    start_eig_index[0] += 1;
                }
            }

            // number of dimensions for eigenvectors
            std::vector<netCDF::NcDim> eig_dims = {n_dets_dim, n_eigs_dim};

            // loop through cleaner gropuing
            for (Eigen::Index i=0; i<in.evecs.data.size(); ++i) {
                // start at first row and col
                std::vector<std::size_t> start_eig_index = {0, 0};

                NcVar evec_v = fo.addVar("evecs_" + cleaner.grouping[i] + "_" + std::to_string(i) + "_chunk_" +
                                             std::to_string(in.index.data),netCDF::ncDouble,eig_dims);

                // loop through eigenvectors in current group
                for (const auto &evecs: in.evecs.data[i]) {
                    std::vector<std::size_t> size = {TULA_SIZET(evecs.rows()), TULA_SIZET(cleaner.n_calc)};

                    // transpose eigenvectors
                    Eigen::MatrixXd ev = evecs.transpose();
                    evec_v.putVar(start_eig_index, size, ev.data());

                    // increment start
                    start_eig_index[0] += TULA_SIZET(evecs.rows());
                }
            }
        }

        // sync file to make sure it gets updated
        fo.sync();
        // close file
        fo.close();
        logger->info("tod chunk written to {}", filepath);

    } catch (NcException &e) {
        logger->error(
            "required PTC TOD write failed; partial output may remain at {}: {}",
            filepath, e.what());
        throw;
    }
}

template <typename calib_t>
void PTCProc::append_diag_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath,
                                    calib_t &calib, Eigen::Index scan_row_index) {
    engine_utils::reject_native_science_consumer(
        in, "PTC diagnostic output before B3 native provenance synchronization");
    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        NcFile fo(filepath, netCDF::NcFile::write);
        const auto scan_row = static_cast<unsigned long>((scan_row_index >= 0) ? scan_row_index : in.index.data);
        const auto n_dets = fo.getDim("n_dets").getSize();
        std::vector<std::size_t> start_index_det = {scan_row, 0};
        std::vector<std::size_t> size_det = {1, n_dets};

        std::vector<double> weights(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> rms(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> stddev(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> median(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> flagged_frac(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        std::vector<RemoveBadDetsWindowDiagSummary> window_diag;
        {
            std::lock_guard<std::mutex> lock(*diag_cache_mutex);
            const auto window_diag_it = remove_bad_dets_window_summary_by_scan.find(in.index.data);
            if (window_diag_it != remove_bad_dets_window_summary_by_scan.end()) {
                window_diag = window_diag_it->second;
            }
        }
        const double n_pts = static_cast<double>(in.scans.data.rows());
        const auto n_copy = std::min<unsigned long>(n_dets, static_cast<unsigned long>(in.scans.data.cols()));
        for (unsigned long i = 0; i < n_copy; ++i) {
            const auto det = static_cast<Eigen::Index>(i);
            Eigen::VectorXd scans = in.scans.data.col(det);
            Eigen::Matrix<bool, Eigen::Dynamic, 1> flags = in.flags.data.col(det);
            weights[static_cast<std::size_t>(i)] = (det < in.weights.data.size()) ? in.weights.data(det) : std::numeric_limits<double>::quiet_NaN();
            rms[static_cast<std::size_t>(i)] = engine_utils::calc_rms(scans);
            stddev[static_cast<std::size_t>(i)] = engine_utils::calc_std_dev(scans);
            median[static_cast<std::size_t>(i)] = tula::alg::median(scans);
            flagged_frac[static_cast<std::size_t>(i)] =
                (n_pts > 0.0) ? flags.cast<double>().sum() / n_pts : std::numeric_limits<double>::quiet_NaN();
        }

        if (window_diag.empty()) {
            auto infer_dt_sec = [&]() {
                auto it = in.tel_data.data.find("TelTime");
                if (it == in.tel_data.data.end() || it->second.size() < 2) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                std::vector<double> dt;
                dt.reserve(static_cast<std::size_t>(it->second.size() - 1));
                for (Eigen::Index i = 1; i < it->second.size(); ++i) {
                    const double delta = it->second(i) - it->second(i - 1);
                    if (std::isfinite(delta) && delta > 0.0) {
                        dt.push_back(delta);
                    }
                }
                if (dt.empty()) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                return tula::alg::median(Eigen::Map<Eigen::VectorXd>(dt.data(), dt.size()));
            };
            auto vector_quantile = [](std::vector<double> values, double q) {
                if (values.empty()) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                std::sort(values.begin(), values.end());
                q = std::clamp(q, 0.0, 1.0);
                const double pos = q * static_cast<double>(values.size() - 1);
                const auto lo = static_cast<std::size_t>(std::floor(pos));
                const auto hi = static_cast<std::size_t>(std::ceil(pos));
                if (lo == hi) {
                    return values[lo];
                }
                const double frac = pos - static_cast<double>(lo);
                return values[lo] * (1.0 - frac) + values[hi] * frac;
            };
            const double dt_sec = infer_dt_sec();
            Eigen::Index window_samples = in.scans.data.rows();
            if (std::isfinite(dt_sec) && dt_sec > 0.0 && remove_bad_dets_window_sec > 0.0) {
                window_samples = std::max<Eigen::Index>(
                    8, static_cast<Eigen::Index>(std::llround(remove_bad_dets_window_sec / dt_sec)));
            }
            window_samples = std::min<Eigen::Index>(window_samples, in.scans.data.rows());
            window_samples = std::max<Eigen::Index>(1, window_samples);

            auto summarize_windows = [&](Eigen::Index det_index) {
                RemoveBadDetsWindowDiagSummary summary;
                if (det_index < 0 || det_index >= in.scans.data.cols()) {
                    return summary;
                }
                Eigen::VectorXd scans = in.scans.data.col(det_index);
                Eigen::Matrix<bool, Eigen::Dynamic, 1> flags = in.flags.data.col(det_index);
                if (citlali::config::is_detector_map_grouping(active_map_grouping) &&
                    mask_radius_arcsec > 0.0) {
                    Eigen::Matrix<bool, Eigen::Dynamic, 1> masked_flags = flags;
                    double az_off = calib.apt["x_t"](det_index);
                    double el_off = calib.apt["y_t"](det_index);
                    auto [lat, lon] = engine_utils::calc_det_pointing(
                        in.tel_data.data,
                        az_off,
                        el_off,
                        std::string{"altaz"},
                        in.pointing_offsets_arcsec.data,
                        active_map_grouping);
                    double source_lat = 0.0;
                    double source_lon = 0.0;
                    resolve_mask_center_rad(in, calib, active_map_grouping, det_index,
                                            source_lat, source_lon);
                    const double radius_rad = mask_radius_arcsec * ASEC_TO_RAD;
                    for (Eigen::Index sample = 0; sample < masked_flags.size(); ++sample) {
                        const double dlat = lat(sample) - source_lat;
                        const double dlon = lon(sample) - source_lon;
                        if (std::sqrt(dlat * dlat + dlon * dlon) < radius_rad) {
                            masked_flags(sample) = true;
                        }
                    }
                    flags = masked_flags;
                }

                summary.n_total_windows = static_cast<int>((scans.size() + window_samples - 1) / window_samples);
                std::vector<double> inv_vars;
                std::vector<double> flagged_fracs;
                inv_vars.reserve(static_cast<std::size_t>(summary.n_total_windows));
                flagged_fracs.reserve(static_cast<std::size_t>(summary.n_total_windows));

                for (Eigen::Index start = 0; start < scans.size(); start += window_samples) {
                    const Eigen::Index stop = std::min<Eigen::Index>(scans.size(), start + window_samples);
                    const Eigen::Index len = stop - start;
                    if (len <= 0) {
                        continue;
                    }
                    int n_flagged = 0;
                    for (Eigen::Index i = start; i < stop; ++i) {
                        if (flags(i)) {
                            ++n_flagged;
                        }
                    }
                    const double flagged_window_frac =
                        static_cast<double>(n_flagged) / static_cast<double>(len);
                    flagged_fracs.push_back(flagged_window_frac);

                    Eigen::VectorXd scan_window = scans.segment(start, len);
                    Eigen::Matrix<bool, Eigen::Dynamic, 1> flag_window = flags.segment(start, len);
                    const double sigma = engine_utils::calc_std_dev(scan_window, flag_window);
                    if (std::isfinite(sigma) && sigma > 0.0) {
                        inv_vars.push_back(std::pow(sigma, -2));
                    }
                }

                summary.n_valid_windows = static_cast<int>(inv_vars.size());
                if (summary.n_total_windows > 0) {
                    summary.valid_window_fraction =
                        static_cast<double>(summary.n_valid_windows) /
                        static_cast<double>(summary.n_total_windows);
                }
                if (!inv_vars.empty()) {
                    summary.inv_var_median = vector_quantile(inv_vars, 0.5);
                    summary.inv_var_q10 = vector_quantile(inv_vars, 0.1);
                    summary.inv_var_q90 = vector_quantile(inv_vars, 0.9);
                }
                if (!flagged_fracs.empty()) {
                    summary.flagged_frac_median = vector_quantile(flagged_fracs, 0.5);
                    summary.flagged_frac_max = *std::max_element(flagged_fracs.begin(), flagged_fracs.end());
                    const auto n_heavy = std::count_if(
                        flagged_fracs.begin(), flagged_fracs.end(),
                        [](double v) { return std::isfinite(v) && v >= 0.5; });
                    summary.heavily_flagged_window_fraction =
                        static_cast<double>(n_heavy) /
                        static_cast<double>(flagged_fracs.size());
                }
                return summary;
            };

            window_diag.assign(static_cast<std::size_t>(in.scans.data.cols()),
                               RemoveBadDetsWindowDiagSummary{});
            for (Eigen::Index det = 0; det < in.scans.data.cols(); ++det) {
                window_diag[static_cast<std::size_t>(det)] = summarize_windows(det);
            }
            {
                std::lock_guard<std::mutex> lock(*diag_cache_mutex);
                remove_bad_dets_window_summary_by_scan[in.index.data] = window_diag;
            }
        }

        fo.getVar("ptc_detector_weight").putVar(start_index_det, size_det, weights.data());
        fo.getVar("ptc_detector_rms").putVar(start_index_det, size_det, rms.data());
        fo.getVar("ptc_detector_stddev").putVar(start_index_det, size_det, stddev.data());
        fo.getVar("ptc_detector_median").putVar(start_index_det, size_det, median.data());
        fo.getVar("ptc_detector_flagged_fraction").putVar(start_index_det, size_det, flagged_frac.data());

        auto window_double_values = [&](auto getter) {
            std::vector<double> values(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
            if (!window_diag.empty()) {
                const auto n_copy_diag = std::min<std::size_t>(
                    static_cast<std::size_t>(n_dets), window_diag.size());
                for (std::size_t i = 0; i < n_copy_diag; ++i) {
                    values[i] = getter(window_diag[i]);
                }
            }
            return values;
        };
        auto window_int_values = [&](auto getter) {
            std::vector<int> values(static_cast<std::size_t>(n_dets), -2147483647);
            if (!window_diag.empty()) {
                const auto n_copy_diag = std::min<std::size_t>(
                    static_cast<std::size_t>(n_dets), window_diag.size());
                for (std::size_t i = 0; i < n_copy_diag; ++i) {
                    values[i] = getter(window_diag[i]);
                }
            }
            return values;
        };
        auto write_window_double = [&](const std::string &name, auto getter) {
            NcVar v = fo.getVar(name);
            if (!v.isNull()) {
                auto values = window_double_values(getter);
                v.putVar(start_index_det, size_det, values.data());
            }
        };
        auto write_window_int = [&](const std::string &name, auto getter) {
            NcVar v = fo.getVar(name);
            if (!v.isNull()) {
                auto values = window_int_values(getter);
                v.putVar(start_index_det, size_det, values.data());
            }
        };

        write_window_int("ptc_invvar_window_n_total",
                         [](const auto &row) { return row.n_total_windows; });
        write_window_int("ptc_invvar_window_n_valid",
                         [](const auto &row) { return row.n_valid_windows; });
        write_window_double("ptc_invvar_window_valid_fraction",
                            [](const auto &row) { return row.valid_window_fraction; });
        write_window_double("ptc_invvar_window_median",
                            [](const auto &row) { return row.inv_var_median; });
        write_window_double("ptc_invvar_window_q10",
                            [](const auto &row) { return row.inv_var_q10; });
        write_window_double("ptc_invvar_window_q90",
                            [](const auto &row) { return row.inv_var_q90; });
        write_window_double("ptc_invvar_window_flagged_frac_median",
                            [](const auto &row) { return row.flagged_frac_median; });
        write_window_double("ptc_invvar_window_flagged_frac_max",
                            [](const auto &row) { return row.flagged_frac_max; });
        write_window_double("ptc_invvar_window_heavy_flagged_fraction",
                            [](const auto &row) { return row.heavily_flagged_window_fraction; });

        const auto second_pass_summary = snapshot_second_pass_summary(in.index.data);
        std::vector<CorrNWDiagSummary> corr_summary;
        std::vector<WeightCorrPenaltyDiagSummary> weight_corr_penalty_summary;
        std::vector<BusyRowSuppressionDiagSummary> busy_row_suppression_summary;
        std::vector<AdaptiveSelectorDiagSummary> adaptive_selector_summary;
        {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            const auto corr_summary_it = corr_nw_summary_by_scan.find(in.index.data);
            if (corr_summary_it != corr_nw_summary_by_scan.end()) {
                corr_summary = corr_summary_it->second;
            }
            const auto weight_corr_penalty_it = weight_corr_penalty_summary_by_scan.find(in.index.data);
            if (weight_corr_penalty_it != weight_corr_penalty_summary_by_scan.end()) {
                weight_corr_penalty_summary = weight_corr_penalty_it->second;
            }
            const auto busy_row_suppression_it = busy_row_suppression_summary_by_scan.find(in.index.data);
            if (busy_row_suppression_it != busy_row_suppression_summary_by_scan.end()) {
                busy_row_suppression_summary = busy_row_suppression_it->second;
            }
            const auto adaptive_selector_it = adaptive_selector_summary_by_scan.find(in.index.data);
            if (adaptive_selector_it != adaptive_selector_summary_by_scan.end()) {
                adaptive_selector_summary = adaptive_selector_it->second;
            }
        }
        const int fill_int = -2147483647;
        const double fill_double = std::numeric_limits<double>::quiet_NaN();

        auto build_nw_index = [&]() {
            std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
            nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
            for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
            }
            return nw_to_index;
        };

        const auto nw_to_index = build_nw_index();

        auto put_corr_nw = [&]() {
            NcVar corr_n_groups_v = fo.getVar("corr_nw_n_groups");
            if (corr_n_groups_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_corr");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<int> v_n_groups(n_nws, fill_int);
            std::vector<int> v_n_groups_raw(n_nws, fill_int);
            std::vector<int> v_n_det_input(n_nws, fill_int);
            std::vector<int> v_n_det_candidates(n_nws, fill_int);
            std::vector<int> v_n_det_used(n_nws, fill_int);
            std::vector<int> v_n_det_grouped(n_nws, fill_int);
            std::vector<int> v_n_det_ungrouped(n_nws, fill_int);
            std::vector<int> v_sample_step(n_nws, fill_int);
            if (!corr_summary.empty()) {
                for (const auto &row : corr_summary) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_n_groups[j] = static_cast<int>(row.n_groups_final);
                    v_n_groups_raw[j] = static_cast<int>(row.n_groups_raw);
                    v_n_det_input[j] = static_cast<int>(row.n_det_input);
                    v_n_det_candidates[j] = static_cast<int>(row.n_det_candidates);
                    v_n_det_used[j] = static_cast<int>(row.n_det_used);
                    v_n_det_grouped[j] = static_cast<int>(row.n_det_grouped);
                    v_n_det_ungrouped[j] = static_cast<int>(row.n_det_ungrouped);
                    v_sample_step[j] = static_cast<int>(row.sample_step);
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            corr_n_groups_v.putVar(start_scan_nw, size_scan_nw, v_n_groups.data());
            fo.getVar("corr_nw_n_groups_raw").putVar(start_scan_nw, size_scan_nw, v_n_groups_raw.data());
            fo.getVar("corr_nw_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
            fo.getVar("corr_nw_n_det_candidates").putVar(start_scan_nw, size_scan_nw, v_n_det_candidates.data());
            fo.getVar("corr_nw_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
            fo.getVar("corr_nw_n_det_grouped").putVar(start_scan_nw, size_scan_nw, v_n_det_grouped.data());
            fo.getVar("corr_nw_n_det_ungrouped").putVar(start_scan_nw, size_scan_nw, v_n_det_ungrouped.data());
            fo.getVar("corr_nw_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
        };

        auto put_weight_corr = [&]() {
            NcVar wcorr_factor_v = fo.getVar("weight_corr_penalty_factor");
            if (wcorr_factor_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_wcorr");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<double> v_factor(n_nws, fill_double);
            std::vector<double> v_severity(n_nws, fill_double);
            std::vector<double> v_pair_corr(n_nws, fill_double);
            std::vector<double> v_cm_el_corr(n_nws, fill_double);
            std::vector<double> v_cm_low_mid(n_nws, fill_double);
            std::vector<int> v_n_det_input(n_nws, fill_int);
            std::vector<int> v_n_det_candidates(n_nws, fill_int);
            std::vector<int> v_n_det_used(n_nws, fill_int);
            std::vector<int> v_n_det_weighted(n_nws, fill_int);
            std::vector<int> v_sample_step(n_nws, fill_int);
            if (!weight_corr_penalty_summary.empty()) {
                for (const auto &row : weight_corr_penalty_summary) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_factor[j] = row.penalty_factor;
                    v_severity[j] = row.severity;
                    v_pair_corr[j] = row.pair_med_abs_corr;
                    v_cm_el_corr[j] = row.cm_el_abs_corr;
                    v_cm_low_mid[j] = row.cm_low_mid_ratio;
                    v_n_det_input[j] = static_cast<int>(row.n_det_input);
                    v_n_det_candidates[j] = static_cast<int>(row.n_det_candidates);
                    v_n_det_used[j] = static_cast<int>(row.n_det_used);
                    v_n_det_weighted[j] = static_cast<int>(row.n_det_weighted);
                    v_sample_step[j] = static_cast<int>(row.sample_step);
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            wcorr_factor_v.putVar(start_scan_nw, size_scan_nw, v_factor.data());
            fo.getVar("weight_corr_penalty_severity").putVar(start_scan_nw, size_scan_nw, v_severity.data());
            fo.getVar("weight_corr_penalty_pair_med_abs_corr").putVar(start_scan_nw, size_scan_nw, v_pair_corr.data());
            fo.getVar("weight_corr_penalty_cm_el_abs_corr").putVar(start_scan_nw, size_scan_nw, v_cm_el_corr.data());
            fo.getVar("weight_corr_penalty_cm_low_mid_ratio").putVar(start_scan_nw, size_scan_nw, v_cm_low_mid.data());
            fo.getVar("weight_corr_penalty_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
            fo.getVar("weight_corr_penalty_n_det_candidates").putVar(start_scan_nw, size_scan_nw, v_n_det_candidates.data());
            fo.getVar("weight_corr_penalty_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
            fo.getVar("weight_corr_penalty_n_det_weighted").putVar(start_scan_nw, size_scan_nw, v_n_det_weighted.data());
            fo.getVar("weight_corr_penalty_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
        };

        auto put_busy_row = [&]() {
            NcVar wbusy_applied_v = fo.getVar("weight_busy_row_suppression_applied");
            if (wbusy_applied_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_busy_row_suppression");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<int> v_applied(n_nws, fill_int);
            std::vector<int> v_busy(n_nws, fill_int);
            std::vector<int> v_n_candidate_clusters(n_nws, fill_int);
            std::vector<int> v_n_det_weighted(n_nws, fill_int);
            std::vector<double> v_factor(n_nws, fill_double);
            std::vector<double> v_max_resid_z(n_nws, fill_double);
            if (!busy_row_suppression_summary.empty()) {
                for (const auto &row : busy_row_suppression_summary) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_applied[j] = row.applied ? 1 : 0;
                    v_busy[j] = row.busy_network_vetoed ? 1 : 0;
                    v_n_candidate_clusters[j] = static_cast<int>(row.n_candidate_clusters);
                    v_n_det_weighted[j] = static_cast<int>(row.n_det_weighted);
                    v_factor[j] = row.factor;
                    v_max_resid_z[j] = row.max_unflagged_residual_z;
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            wbusy_applied_v.putVar(start_scan_nw, size_scan_nw, v_applied.data());
            fo.getVar("weight_busy_row_suppression_busy_network_vetoed").putVar(start_scan_nw, size_scan_nw, v_busy.data());
            fo.getVar("weight_busy_row_suppression_n_candidate_clusters").putVar(start_scan_nw, size_scan_nw, v_n_candidate_clusters.data());
            fo.getVar("weight_busy_row_suppression_n_det_weighted").putVar(start_scan_nw, size_scan_nw, v_n_det_weighted.data());
            fo.getVar("weight_busy_row_suppression_factor").putVar(start_scan_nw, size_scan_nw, v_factor.data());
            fo.getVar("weight_busy_row_suppression_max_unflagged_residual_z").putVar(start_scan_nw, size_scan_nw, v_max_resid_z.data());
        };

        auto put_adaptive = [&]() {
            NcVar adaptive_chosen_k_v = fo.getVar("adaptive_pca_chosen_k");
            if (adaptive_chosen_k_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_adaptive_pca");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<int> v_selector_used(n_nws, fill_int);
            std::vector<int> v_selector_fallback(n_nws, fill_int);
            std::vector<int> v_baseline_k(n_nws, fill_int);
            std::vector<int> v_chosen_k(n_nws, fill_int);
            std::vector<int> v_runnerup_k(n_nws, fill_int);
            std::vector<int> v_n_candidates(n_nws, fill_int);
            std::vector<int> v_n_det_input(n_nws, fill_int);
            std::vector<int> v_n_det_used(n_nws, fill_int);
            std::vector<int> v_n_time_used(n_nws, fill_int);
            std::vector<int> v_sample_step(n_nws, fill_int);
            std::vector<double> v_chosen_score(n_nws, fill_double);
            std::vector<double> v_runnerup_score(n_nws, fill_double);
            std::vector<double> v_score_margin(n_nws, fill_double);
            std::vector<double> v_chosen_med_abs_corr(n_nws, fill_double);
            std::vector<double> v_chosen_cm_low_mid_ratio(n_nws, fill_double);
            std::vector<double> v_chosen_tail4_binom_z(n_nws, fill_double);
            std::vector<double> v_chosen_top_mode_frac(n_nws, fill_double);
            std::vector<double> v_eig_solve_msec(n_nws, fill_double);
            std::vector<double> v_candidate_eval_msec(n_nws, fill_double);
            std::vector<double> v_total_msec(n_nws, fill_double);
            if (!adaptive_selector_summary.empty()) {
                for (const auto &row : adaptive_selector_summary) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_selector_used[j] = row.selector_used;
                    v_selector_fallback[j] = row.selector_fallback;
                    v_baseline_k[j] = static_cast<int>(row.baseline_k);
                    v_chosen_k[j] = static_cast<int>(row.chosen_k);
                    v_runnerup_k[j] = static_cast<int>(row.runnerup_k);
                    v_n_candidates[j] = static_cast<int>(row.n_candidates);
                    v_n_det_input[j] = static_cast<int>(row.n_det_input);
                    v_n_det_used[j] = static_cast<int>(row.n_det_used);
                    v_n_time_used[j] = static_cast<int>(row.n_time_used);
                    v_sample_step[j] = static_cast<int>(row.sample_step);
                    v_chosen_score[j] = row.chosen_score;
                    v_runnerup_score[j] = row.runnerup_score;
                    v_score_margin[j] = row.score_margin;
                    v_chosen_med_abs_corr[j] = row.chosen_med_abs_corr;
                    v_chosen_cm_low_mid_ratio[j] = row.chosen_cm_low_mid_ratio;
                    v_chosen_tail4_binom_z[j] = row.chosen_tail4_binom_z;
                    v_chosen_top_mode_frac[j] = row.chosen_top_mode_frac;
                    v_eig_solve_msec[j] = row.eig_solve_msec;
                    v_candidate_eval_msec[j] = row.candidate_eval_msec;
                    v_total_msec[j] = row.total_msec;
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            fo.getVar("adaptive_pca_selector_used").putVar(start_scan_nw, size_scan_nw, v_selector_used.data());
            fo.getVar("adaptive_pca_selector_fallback").putVar(start_scan_nw, size_scan_nw, v_selector_fallback.data());
            fo.getVar("adaptive_pca_baseline_k").putVar(start_scan_nw, size_scan_nw, v_baseline_k.data());
            adaptive_chosen_k_v.putVar(start_scan_nw, size_scan_nw, v_chosen_k.data());
            fo.getVar("adaptive_pca_runnerup_k").putVar(start_scan_nw, size_scan_nw, v_runnerup_k.data());
            fo.getVar("adaptive_pca_n_candidates").putVar(start_scan_nw, size_scan_nw, v_n_candidates.data());
            fo.getVar("adaptive_pca_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
            fo.getVar("adaptive_pca_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
            fo.getVar("adaptive_pca_n_time_used").putVar(start_scan_nw, size_scan_nw, v_n_time_used.data());
            fo.getVar("adaptive_pca_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
            fo.getVar("adaptive_pca_chosen_score").putVar(start_scan_nw, size_scan_nw, v_chosen_score.data());
            fo.getVar("adaptive_pca_runnerup_score").putVar(start_scan_nw, size_scan_nw, v_runnerup_score.data());
            fo.getVar("adaptive_pca_score_margin").putVar(start_scan_nw, size_scan_nw, v_score_margin.data());
            fo.getVar("adaptive_pca_chosen_med_abs_corr").putVar(start_scan_nw, size_scan_nw, v_chosen_med_abs_corr.data());
            fo.getVar("adaptive_pca_chosen_cm_low_mid_ratio").putVar(start_scan_nw, size_scan_nw, v_chosen_cm_low_mid_ratio.data());
            fo.getVar("adaptive_pca_chosen_tail4_binom_z").putVar(start_scan_nw, size_scan_nw, v_chosen_tail4_binom_z.data());
            fo.getVar("adaptive_pca_chosen_top_mode_frac").putVar(start_scan_nw, size_scan_nw, v_chosen_top_mode_frac.data());
            fo.getVar("adaptive_pca_eig_solve_msec").putVar(start_scan_nw, size_scan_nw, v_eig_solve_msec.data());
            fo.getVar("adaptive_pca_candidate_eval_msec").putVar(start_scan_nw, size_scan_nw, v_candidate_eval_msec.data());
            fo.getVar("adaptive_pca_total_msec").putVar(start_scan_nw, size_scan_nw, v_total_msec.data());
        };

        auto put_second_pass = [&]() {
            NcVar second_pass_busy_v = fo.getVar("ptc_second_pass_busy_network_vetoed");
            if (second_pass_busy_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_ptc_second_pass");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<int> v_busy(n_nws, fill_int);
            std::vector<int> v_n_candidate_clusters(n_nws, fill_int);
            std::vector<int> v_n_candidate_events(n_nws, fill_int);
            std::vector<int> v_n_accepted_clusters(n_nws, fill_int);
            std::vector<int> v_n_accepted_events(n_nws, fill_int);
            std::vector<int> v_n_rejected_clusters(n_nws, fill_int);
            std::vector<int> v_n_rejected_events(n_nws, fill_int);
            std::vector<int> v_n_source_protected_clusters(n_nws, fill_int);
            std::vector<int> v_n_source_protected_events(n_nws, fill_int);
            std::vector<int> v_n_det_with_added_flags(n_nws, fill_int);
            std::vector<int> v_max_resid_uid(n_nws, fill_int);
            std::vector<int> v_top_cluster_sample(n_nws, fill_int);
            std::vector<int> v_top_cluster_n_detectors(n_nws, fill_int);
            std::vector<int> v_top_cluster_n_events(n_nws, fill_int);
            std::vector<int> v_top_event_kind(n_nws, fill_int);
            std::vector<int> v_top_event_uid(n_nws, fill_int);
            std::vector<int> v_top_event_sample(n_nws, fill_int);
            std::vector<double> v_existing_frac(n_nws, fill_double);
            std::vector<double> v_proposed_frac(n_nws, fill_double);
            std::vector<double> v_new_frac(n_nws, fill_double);
            std::vector<double> v_max_resid_z(n_nws, fill_double);
            std::vector<double> v_top_cluster_peak(n_nws, fill_double);
            std::vector<double> v_top_event_score(n_nws, fill_double);
            if (!second_pass_summary.empty()) {
                for (const auto &row : second_pass_summary) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_busy[j] = row.busy_network_vetoed ? 1 : 0;
                    v_n_candidate_clusters[j] = static_cast<int>(row.n_candidate_clusters);
                    v_n_candidate_events[j] = static_cast<int>(row.n_candidate_events);
                    v_n_accepted_clusters[j] = static_cast<int>(row.n_accepted_clusters);
                    v_n_accepted_events[j] = static_cast<int>(row.n_accepted_events);
                    v_n_rejected_clusters[j] = static_cast<int>(row.n_rejected_clusters);
                    v_n_rejected_events[j] = static_cast<int>(row.n_rejected_events);
                    v_n_source_protected_clusters[j] = static_cast<int>(row.n_source_protected_clusters);
                    v_n_source_protected_events[j] = static_cast<int>(row.n_source_protected_events);
                    v_n_det_with_added_flags[j] = static_cast<int>(row.n_det_with_added_flags);
                    v_max_resid_uid[j] = row.max_unflagged_residual_uid;
                    v_top_cluster_sample[j] = row.top_candidate_cluster_sample;
                    v_top_cluster_n_detectors[j] = static_cast<int>(row.top_candidate_cluster_n_detectors);
                    v_top_cluster_n_events[j] = static_cast<int>(row.top_candidate_cluster_n_events);
                    v_top_event_kind[j] = row.top_event.kind_code();
                    v_top_event_uid[j] = row.top_event_uid;
                    v_top_event_sample[j] = row.top_event.sample;
                    v_existing_frac[j] = row.existing_flagged_fraction;
                    v_proposed_frac[j] = row.proposed_flagged_fraction;
                    v_new_frac[j] = row.newly_flagged_fraction;
                    v_max_resid_z[j] = row.max_unflagged_residual_z;
                    v_top_cluster_peak[j] = row.top_candidate_cluster_peak_score;
                    v_top_event_score[j] = row.top_event.score;
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            second_pass_busy_v.putVar(start_scan_nw, size_scan_nw, v_busy.data());
            fo.getVar("ptc_second_pass_n_candidate_clusters").putVar(start_scan_nw, size_scan_nw, v_n_candidate_clusters.data());
            fo.getVar("ptc_second_pass_n_candidate_events").putVar(start_scan_nw, size_scan_nw, v_n_candidate_events.data());
            fo.getVar("ptc_second_pass_n_accepted_clusters").putVar(start_scan_nw, size_scan_nw, v_n_accepted_clusters.data());
            fo.getVar("ptc_second_pass_n_accepted_events").putVar(start_scan_nw, size_scan_nw, v_n_accepted_events.data());
            fo.getVar("ptc_second_pass_n_rejected_clusters").putVar(start_scan_nw, size_scan_nw, v_n_rejected_clusters.data());
            fo.getVar("ptc_second_pass_n_rejected_events").putVar(start_scan_nw, size_scan_nw, v_n_rejected_events.data());
            fo.getVar("ptc_second_pass_n_source_protected_clusters").putVar(start_scan_nw, size_scan_nw, v_n_source_protected_clusters.data());
            fo.getVar("ptc_second_pass_n_source_protected_events").putVar(start_scan_nw, size_scan_nw, v_n_source_protected_events.data());
            fo.getVar("ptc_second_pass_n_det_with_added_flags").putVar(start_scan_nw, size_scan_nw, v_n_det_with_added_flags.data());
            fo.getVar("ptc_second_pass_max_unflagged_residual_uid").putVar(start_scan_nw, size_scan_nw, v_max_resid_uid.data());
            fo.getVar("ptc_second_pass_top_candidate_cluster_sample").putVar(start_scan_nw, size_scan_nw, v_top_cluster_sample.data());
            fo.getVar("ptc_second_pass_top_candidate_cluster_n_detectors").putVar(start_scan_nw, size_scan_nw, v_top_cluster_n_detectors.data());
            fo.getVar("ptc_second_pass_top_candidate_cluster_n_events").putVar(start_scan_nw, size_scan_nw, v_top_cluster_n_events.data());
            fo.getVar("ptc_second_pass_top_event_kind").putVar(start_scan_nw, size_scan_nw, v_top_event_kind.data());
            fo.getVar("ptc_second_pass_top_event_uid").putVar(start_scan_nw, size_scan_nw, v_top_event_uid.data());
            fo.getVar("ptc_second_pass_top_event_sample").putVar(start_scan_nw, size_scan_nw, v_top_event_sample.data());
            fo.getVar("ptc_second_pass_existing_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_existing_frac.data());
            fo.getVar("ptc_second_pass_proposed_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_proposed_frac.data());
            fo.getVar("ptc_second_pass_newly_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_new_frac.data());
            fo.getVar("ptc_second_pass_max_unflagged_residual_z").putVar(start_scan_nw, size_scan_nw, v_max_resid_z.data());
            fo.getVar("ptc_second_pass_top_candidate_cluster_peak_score").putVar(start_scan_nw, size_scan_nw, v_top_cluster_peak.data());
            fo.getVar("ptc_second_pass_top_event_score").putVar(start_scan_nw, size_scan_nw, v_top_event_score.data());
        };

        put_corr_nw();
        put_weight_corr();
        put_busy_row();
        put_adaptive();
        put_second_pass();

        fo.sync();
        fo.close();
        logger->info("ptc diagnostics sidecar chunk written to {}", filepath);
    } catch (NcException &e) {
        logger->error(
            "required PTC diagnostics write failed; partial output may remain at {}: {}",
            filepath, e.what());
        throw;
    }
}

inline void PTCProc::clear_cached_diagnostics(Eigen::Index scan_id) {
    {
        std::lock_guard<std::mutex> lock(*diag_cache_mutex);
        remove_bad_dets_window_summary_by_scan.erase(scan_id);
    }
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        corr_nw_group_ids_by_scan.erase(scan_id);
        corr_nw_summary_by_scan.erase(scan_id);
        weight_corr_penalty_summary_by_scan.erase(scan_id);
        busy_row_suppression_summary_by_scan.erase(scan_id);
        adaptive_selector_summary_by_scan.erase(scan_id);
        pca_realization_summary_by_scan.erase(scan_id);
        mean_realization_summary_by_scan.erase(scan_id);
        second_pass_summary_by_scan.erase(scan_id);
        second_pass_added_flags_by_scan.erase(scan_id);
        high_weight_summary_by_scan.erase(scan_id);
    }
}

} // namespace timestream

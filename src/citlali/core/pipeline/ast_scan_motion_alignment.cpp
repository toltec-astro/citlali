#include <citlali/core/pipeline/ast_scan_motion_alignment.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace citlali::pipeline {
namespace {

bool usable_source_pair(const AstScanMotionProduct &product,
                        std::size_t lower, std::size_t upper) noexcept {
    if (upper != lower + 1 || upper >= product.record_count()) return false;
    const auto &left = product.record_at_local(lower);
    const auto &right = product.record_at_local(upper);
    return left.derivative_valid() && right.derivative_valid() &&
           left.continuity_run() >= 0 &&
           left.continuity_run() == right.continuity_run();
}

AstScanMotionCause source_pair_causes(const AstScanMotionProduct &product,
                                      std::size_t lower,
                                      std::size_t upper) noexcept {
    AstScanMotionCause causes =
        AstScanMotionCause::network_mapping_support_unavailable;
    if (lower < product.record_count()) {
        causes |= product.record_at_local(lower).causes();
    }
    if (upper < product.record_count()) {
        causes |= product.record_at_local(upper).causes();
    }
    return causes;
}

AstScanMotionCause source_neighborhood_causes(
    const AstScanMotionProduct &product, std::size_t center) noexcept {
    AstScanMotionCause causes =
        AstScanMotionCause::network_mapping_support_unavailable;
    if (center > 0) causes |= product.record_at_local(center - 1).causes();
    if (center < product.record_count()) {
        causes |= product.record_at_local(center).causes();
    }
    if (center + 1 < product.record_count()) {
        causes |= product.record_at_local(center + 1).causes();
    }
    return causes;
}

}  // namespace

bool AstScanMotionMappedRecord::available() const noexcept {
    return available_;
}

AstScanMotionCause AstScanMotionMappedRecord::causes() const noexcept {
    return causes_;
}

double AstScanMotionMappedRecord::scalar_speed_arcsec_per_sec()
    const noexcept {
    return scalar_speed_arcsec_per_sec_;
}

std::shared_ptr<const AstScanMotionNetworkView>
AstScanMotionNetworkView::admit(
    std::shared_ptr<const AstScanMotionProduct> raw_product,
    std::shared_ptr<const NativeNetworkAlignment> network_timing) {
    if (!raw_product || !network_timing) {
        throw std::invalid_argument(
            "AST network mapping requires product and native timing handles");
    }

    std::vector<AstScanMotionMappedRecord> mapped(
        static_cast<std::size_t>(network_timing->row_count()));
    if (!raw_product->source_time_axis_mapping_eligible()) {
        return std::shared_ptr<const AstScanMotionNetworkView>(
            new AstScanMotionNetworkView{
                std::move(raw_product), std::move(network_timing),
                std::move(mapped)});
    }

    const auto &source_times =
        raw_product->source_handle()->producer_times_unix_sec();
    const auto source_count = raw_product->record_count();
    for (std::size_t target_index = 0; target_index < mapped.size();
         ++target_index) {
        auto &result = mapped[target_index];
        const auto native_row = network_timing->first_native_row() +
            static_cast<TimestreamNativeRow>(target_index);
        const double target_time =
            network_timing->identity(native_row).reconstructed_time_unix_sec();
        const double *begin = source_times.data();
        const double *end = begin + source_times.size();
        const double *found = std::lower_bound(begin, end, target_time);
        if (found == end || target_time < *begin) continue;

        const auto found_index = static_cast<std::size_t>(found - begin);
        std::size_t lower = source_count;
        std::size_t upper = source_count;
        if (*found == target_time) {
            if (found_index + 1 < source_count &&
                usable_source_pair(*raw_product, found_index,
                                   found_index + 1)) {
                lower = found_index;
                upper = found_index + 1;
            }
            else if (found_index > 0 &&
                     usable_source_pair(*raw_product, found_index - 1,
                                        found_index)) {
                lower = found_index - 1;
                upper = found_index;
            }
            else {
                result.causes_ = source_neighborhood_causes(
                    *raw_product, found_index);
                continue;
            }
        }
        else {
            if (found_index == 0) continue;
            lower = found_index - 1;
            upper = found_index;
            if (!usable_source_pair(*raw_product, lower, upper)) {
                result.causes_ =
                    source_pair_causes(*raw_product, lower, upper);
                continue;
            }
        }

        const double lower_time =
            source_times(static_cast<Eigen::Index>(lower));
        const double upper_time =
            source_times(static_cast<Eigen::Index>(upper));
        const double upper_weight =
            (target_time - lower_time) / (upper_time - lower_time);
        const double lower_weight = 1.0 - upper_weight;
        const double speed =
            lower_weight *
                raw_product->record_at_local(lower)
                    .scalar_speed_arcsec_per_sec() +
            upper_weight *
                raw_product->record_at_local(upper)
                    .scalar_speed_arcsec_per_sec();
        if (!std::isfinite(lower_weight) || !std::isfinite(upper_weight) ||
            lower_weight < 0.0 || upper_weight < 0.0 ||
            lower_weight > 1.0 || upper_weight > 1.0 ||
            !std::isfinite(speed)) {
            result.causes_ = source_pair_causes(*raw_product, lower, upper);
            continue;
        }
        result.available_ = true;
        result.causes_ = AstScanMotionCause::none;
        result.lower_source_local_index_ = lower;
        result.upper_source_local_index_ = upper;
        result.lower_weight_ = lower_weight;
        result.upper_weight_ = upper_weight;
        result.scalar_speed_arcsec_per_sec_ = speed;
    }

    return std::shared_ptr<const AstScanMotionNetworkView>(
        new AstScanMotionNetworkView{
            std::move(raw_product), std::move(network_timing),
            std::move(mapped)});
}

AstScanMotionNetworkView::AstScanMotionNetworkView(
    std::shared_ptr<const AstScanMotionProduct> raw_product,
    std::shared_ptr<const NativeNetworkAlignment> network_timing,
    std::vector<AstScanMotionMappedRecord> records)
    : raw_product_{std::move(raw_product)},
      network_timing_{std::move(network_timing)},
      records_{std::move(records)} {}

TimestreamNetworkId AstScanMotionNetworkView::network_id() const noexcept {
    return network_timing_->network_id();
}

TimestreamNativeRow AstScanMotionNetworkView::first_native_row()
    const noexcept {
    return network_timing_->first_native_row();
}

TimestreamNativeRow AstScanMotionNetworkView::past_last_native_row()
    const noexcept {
    return network_timing_->past_last_native_row();
}

std::size_t AstScanMotionNetworkView::occurrence_count() const noexcept {
    return records_.size();
}

const std::shared_ptr<const AstScanMotionProduct> &
AstScanMotionNetworkView::raw_product_handle() const noexcept {
    return raw_product_;
}

const std::shared_ptr<const NativeNetworkAlignment> &
AstScanMotionNetworkView::network_timing_handle() const noexcept {
    return network_timing_;
}

NativeSampleIdentity AstScanMotionNetworkView::identity(
    TimestreamNativeRow native_row) const {
    (void)local_index(native_row);
    return network_timing_->identity(native_row);
}

const AstScanMotionMappedRecord &AstScanMotionNetworkView::record(
    TimestreamNativeRow native_row) const {
    return records_.at(local_index(native_row));
}

std::optional<double>
AstScanMotionNetworkView::scalar_speed_arcsec_per_sec(
    TimestreamNativeRow native_row) const {
    const auto &mapped = record(native_row);
    if (!mapped.available()) return std::nullopt;
    return mapped.scalar_speed_arcsec_per_sec();
}

std::optional<AstScanMotionMappedSupport> AstScanMotionNetworkView::support(
    TimestreamNativeRow native_row) const {
    const auto &mapped = record(native_row);
    if (!mapped.available()) return std::nullopt;
    const auto &source = *raw_product_->source_handle();
    const auto lower_record =
        raw_product_->record_identity(mapped.lower_source_local_index_);
    const auto upper_record =
        raw_product_->record_identity(mapped.upper_source_local_index_);
    return AstScanMotionMappedSupport{
        identity(native_row), source.identity(lower_record),
        source.identity(upper_record),
        source.producer_times_unix_sec()(
            static_cast<Eigen::Index>(mapped.lower_source_local_index_)),
        source.producer_times_unix_sec()(
            static_cast<Eigen::Index>(mapped.upper_source_local_index_)),
        mapped.lower_weight_, mapped.upper_weight_};
}

AstScanMotionMappedMemoryEvidence
AstScanMotionNetworkView::memory_evidence() const noexcept {
    return {records_.size() * sizeof(AstScanMotionMappedRecord), 1, 1};
}

std::size_t AstScanMotionNetworkView::local_index(
    TimestreamNativeRow native_row) const {
    if (native_row < first_native_row() ||
        native_row >= past_last_native_row()) {
        throw std::out_of_range(
            "native row is outside AST network-mapped support");
    }
    return static_cast<std::size_t>(native_row - first_native_row());
}

std::shared_ptr<const AstScanMotionNetworkViews>
AstScanMotionNetworkViews::admit(
    NativeObservationScope expected_scope,
    std::shared_ptr<const AstScanMotionProduct> raw_product,
    std::vector<std::shared_ptr<const NativeNetworkAlignment>>
        network_timings) {
    if (!raw_product || network_timings.empty() ||
        !(raw_product->scope() == expected_scope)) {
        throw std::invalid_argument(
            "AST network views require matching scope and participants");
    }
    std::sort(network_timings.begin(), network_timings.end(),
              [](const auto &lhs, const auto &rhs) {
                  if (!lhs || !rhs) return static_cast<bool>(lhs);
                  return lhs->network_id() < rhs->network_id();
              });
    std::vector<std::shared_ptr<const AstScanMotionNetworkView>> networks;
    std::vector<TimestreamNetworkId> participant_network_ids;
    std::map<TimestreamNetworkId, std::size_t> network_index_by_id;
    networks.reserve(network_timings.size());
    participant_network_ids.reserve(network_timings.size());
    for (auto &timing : network_timings) {
        if (!timing ||
            !network_index_by_id
                 .emplace(timing->network_id(), networks.size())
                 .second) {
            throw std::invalid_argument(
                "AST network views contain an absent or repeated participant");
        }
        participant_network_ids.push_back(timing->network_id());
        networks.push_back(AstScanMotionNetworkView::admit(
            raw_product, std::move(timing)));
    }
    return std::shared_ptr<const AstScanMotionNetworkViews>(
        new AstScanMotionNetworkViews{
            expected_scope, std::move(raw_product), std::move(networks),
            std::move(participant_network_ids),
            std::move(network_index_by_id)});
}

AstScanMotionNetworkViews::AstScanMotionNetworkViews(
    NativeObservationScope scope,
    std::shared_ptr<const AstScanMotionProduct> raw_product,
    std::vector<std::shared_ptr<const AstScanMotionNetworkView>> networks,
    std::vector<TimestreamNetworkId> participant_network_ids,
    std::map<TimestreamNetworkId, std::size_t> network_index_by_id)
    : scope_{scope}, raw_product_{std::move(raw_product)},
      networks_{std::move(networks)},
      participant_network_ids_{std::move(participant_network_ids)},
      network_index_by_id_{std::move(network_index_by_id)} {}

const NativeObservationScope &AstScanMotionNetworkViews::scope()
    const noexcept {
    return scope_;
}

const std::shared_ptr<const AstScanMotionProduct> &
AstScanMotionNetworkViews::raw_product_handle() const noexcept {
    return raw_product_;
}

std::span<const TimestreamNetworkId>
AstScanMotionNetworkViews::participant_network_ids() const noexcept {
    return participant_network_ids_;
}

const AstScanMotionNetworkView &AstScanMotionNetworkViews::network(
    TimestreamNetworkId network_id) const {
    const auto found = network_index_by_id_.find(network_id);
    if (found == network_index_by_id_.end()) {
        throw std::out_of_range(
            "network is absent from AST network-mapped views");
    }
    return *networks_.at(found->second);
}

}  // namespace citlali::pipeline

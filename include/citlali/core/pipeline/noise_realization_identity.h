#pragma once

#include <citlali/core/utils/sha256.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

// The pre-repair implementation used the default mt19937 seed (5489). Keep
// that value as a named master input, but bind it to a versioned namespace and
// counter algorithm instead of mutable generator draw order.
inline constexpr std::uint32_t noise_random_seed = 5489U;
inline constexpr const char *noise_realization_key_policy_version =
    "citlali-noise-realization-key-v1";
inline constexpr const char *noise_realization_generator_version =
    "citlali-sha256-splitmix64-sign-v1";
inline constexpr const char *noise_random_engine_name =
    "sha256_namespace_splitmix64_counter";
inline constexpr const char *noise_seed_policy_name =
    "versioned_internal_master_seed_v1";
inline constexpr const char *noise_generator_scope_name =
    "observation_conditioning_pass";
inline constexpr const char *noise_ensemble_mode_source_imprinted_current =
    "source_imprinted_current";
inline constexpr const char *noise_channel_identity_policy =
    "observation_scoped_zero_based_channel_ordinal";
inline constexpr const char *noise_coherence_unit_identity_policy =
    "observation_scoped_zero_based_scan_or_chunk_ordinal";
inline constexpr const char *noise_assignment_ordering_policy =
    "realization_then_coherence_unit_then_channel";

inline void append_noise_string_field(
    std::string &canonical, std::string_view name, std::string_view value) {
    canonical.append(std::to_string(name.size()));
    canonical.push_back(':');
    canonical.append(name);
    canonical.push_back('=');
    canonical.append(std::to_string(value.size()));
    canonical.push_back(':');
    canonical.append(value);
    canonical.push_back(';');
}

template <class Integer>
void append_noise_integer_field(
    std::string &canonical, std::string_view name, Integer value) {
    append_noise_string_field(canonical, name, std::to_string(value));
}

inline std::uint64_t noise_hex_word(
    std::string_view digest, std::size_t offset) {
    if (offset + 16 > digest.size()) {
        throw std::logic_error("noise namespace digest is incomplete");
    }
    std::uint64_t value = 0;
    for (std::size_t index = offset; index < offset + 16; ++index) {
        const char digit = digest[index];
        std::uint8_t nibble = 0;
        if (digit >= '0' && digit <= '9') {
            nibble = static_cast<std::uint8_t>(digit - '0');
        }
        else if (digit >= 'a' && digit <= 'f') {
            nibble = static_cast<std::uint8_t>(digit - 'a' + 10);
        }
        else {
            throw std::logic_error(
                "noise namespace digest contains a non-hex digit");
        }
        value = (value << 4U) | nibble;
    }
    return value;
}

inline std::uint64_t noise_splitmix64(std::uint64_t value) noexcept {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

struct NoiseAssignmentContext {
    std::string observation_id;
    std::string ensemble_mode =
        noise_ensemble_mode_source_imprinted_current;
    int conditioning_iteration = 0;
    std::string pass_id;
    int pass_ordinal = 0;
    int n_realizations = 0;
    std::size_t coherence_unit_count = 0;
    std::size_t channel_count = 0;
    bool randomize_channels = true;
    std::string namespace_digest;
    std::uint64_t namespace_word_0 = 0;
    std::uint64_t namespace_word_1 = 0;
};

inline std::string noise_assignment_namespace_canonical(
    const NoiseAssignmentContext &context) {
    std::string canonical;
    append_noise_string_field(
        canonical, "key_policy_version",
        noise_realization_key_policy_version);
    append_noise_string_field(
        canonical, "generator_version", noise_realization_generator_version);
    append_noise_integer_field(canonical, "master_seed", noise_random_seed);
    append_noise_string_field(
        canonical, "seed_policy", noise_seed_policy_name);
    append_noise_string_field(
        canonical, "observation_id", context.observation_id);
    append_noise_string_field(
        canonical, "ensemble_mode", context.ensemble_mode);
    append_noise_integer_field(
        canonical, "conditioning_iteration",
        context.conditioning_iteration);
    append_noise_string_field(canonical, "pass_id", context.pass_id);
    append_noise_integer_field(
        canonical, "pass_ordinal", context.pass_ordinal);
    append_noise_string_field(
        canonical, "coherence_unit_identity_policy",
        noise_coherence_unit_identity_policy);
    append_noise_string_field(
        canonical, "channel_identity_policy", noise_channel_identity_policy);
    append_noise_integer_field(
        canonical, "randomize_channels", context.randomize_channels ? 1 : 0);
    return canonical;
}

inline std::string noise_assignment_namespace_digest(
    const NoiseAssignmentContext &context) {
    if (!context.namespace_digest.empty()) {
        return context.namespace_digest;
    }
    return citlali::utils::sha256(
        noise_assignment_namespace_canonical(context));
}

inline NoiseAssignmentContext make_noise_assignment_context(
    std::string observation_id, int conditioning_iteration,
    std::string pass_id, int n_realizations,
    std::size_t coherence_unit_count, std::size_t channel_count,
    bool randomize_channels, int pass_ordinal = 0) {
    if (observation_id.empty() || pass_id.empty() ||
        conditioning_iteration < 0 || pass_ordinal < 0 ||
        n_realizations < 1 ||
        coherence_unit_count < 1 || channel_count < 1) {
        throw std::logic_error(
            "noise assignment context has an invalid identity or cardinality");
    }
    NoiseAssignmentContext context;
    context.observation_id = std::move(observation_id);
    context.conditioning_iteration = conditioning_iteration;
    context.pass_id = std::move(pass_id);
    context.pass_ordinal = pass_ordinal;
    context.n_realizations = n_realizations;
    context.coherence_unit_count = coherence_unit_count;
    context.channel_count = channel_count;
    context.randomize_channels = randomize_channels;
    context.namespace_digest = citlali::utils::sha256(
        noise_assignment_namespace_canonical(context));
    context.namespace_word_0 = noise_hex_word(context.namespace_digest, 0);
    context.namespace_word_1 = noise_hex_word(context.namespace_digest, 16);
    return context;
}

inline std::size_t noise_effective_channel_identity(
    const NoiseAssignmentContext &context, std::size_t channel) noexcept {
    return context.randomize_channels ? channel : std::size_t{0};
}

inline void require_noise_realization_key_indices(
    const NoiseAssignmentContext &context, int realization,
    std::size_t coherence_unit, std::size_t channel) {
    if (realization < 0 || realization >= context.n_realizations ||
        coherence_unit >= context.coherence_unit_count ||
        channel >= context.channel_count) {
        throw std::out_of_range("noise realization key index is out of range");
    }
}

inline int noise_realization_sign(
    const NoiseAssignmentContext &context, int realization,
    std::size_t coherence_unit, std::size_t channel) {
    require_noise_realization_key_indices(
        context, realization, coherence_unit, channel);
    const auto stable_channel =
        noise_effective_channel_identity(context, channel);
    std::uint64_t value = context.namespace_word_0;
    value ^= noise_splitmix64(
        context.namespace_word_1 ^
        (static_cast<std::uint64_t>(realization) +
         UINT64_C(0x243f6a8885a308d3)));
    value ^= noise_splitmix64(
        static_cast<std::uint64_t>(coherence_unit) +
        UINT64_C(0x13198a2e03707344));
    value ^= noise_splitmix64(
        static_cast<std::uint64_t>(stable_channel) +
        UINT64_C(0xa4093822299f31d0));
    return (noise_splitmix64(value) & UINT64_C(1)) != 0 ? 1 : -1;
}

inline std::string noise_realization_key_digest(
    const NoiseAssignmentContext &context, int realization,
    std::size_t coherence_unit, std::size_t channel) {
    require_noise_realization_key_indices(
        context, realization, coherence_unit, channel);
    std::string canonical;
    append_noise_string_field(
        canonical, "namespace_digest",
        noise_assignment_namespace_digest(context));
    append_noise_integer_field(canonical, "realization", realization);
    append_noise_integer_field(
        canonical, "coherence_unit", coherence_unit);
    append_noise_integer_field(
        canonical, "channel",
        noise_effective_channel_identity(context, channel));
    return citlali::utils::sha256(canonical);
}

inline std::string noise_assignment_partition_digest(
    const NoiseAssignmentContext &context) {
    std::string canonical;
    append_noise_string_field(
        canonical, "namespace_digest",
        noise_assignment_namespace_digest(context));
    append_noise_string_field(
        canonical, "ordering_policy", noise_assignment_ordering_policy);
    append_noise_integer_field(
        canonical, "n_realizations", context.n_realizations);
    append_noise_integer_field(
        canonical, "coherence_unit_count", context.coherence_unit_count);
    append_noise_integer_field(
        canonical, "channel_count", context.channel_count);
    return citlali::utils::sha256(canonical);
}

struct NoiseAssignmentRecord {
    std::string key_policy_version =
        noise_realization_key_policy_version;
    std::string generator_version = noise_realization_generator_version;
    std::string observation_id;
    std::string ensemble_mode;
    int conditioning_iteration = 0;
    std::string pass_id;
    int pass_ordinal = 0;
    bool randomize_channels = true;
    std::string coherence_unit_identity_policy =
        noise_coherence_unit_identity_policy;
    std::string channel_identity_policy = noise_channel_identity_policy;
    std::string ordering_policy = noise_assignment_ordering_policy;
    std::size_t coherence_unit_count = 0;
    std::size_t channel_count = 0;
    std::vector<std::size_t> completed_realization_ids;
    std::string namespace_digest;
    std::string partition_digest;
    std::string reconstruction_digest;

    bool compact() const {
        return !observation_id.empty() && !ensemble_mode.empty() &&
               !pass_id.empty() && coherence_unit_count > 0 &&
               channel_count > 0 && !completed_realization_ids.empty() &&
               !namespace_digest.empty() && !partition_digest.empty() &&
               !reconstruction_digest.empty();
    }
};

inline std::string noise_assignment_record_reconstruction_digest(
    const NoiseAssignmentRecord &record) {
    std::string canonical;
    append_noise_string_field(
        canonical, "key_policy_version", record.key_policy_version);
    append_noise_string_field(
        canonical, "generator_version", record.generator_version);
    append_noise_string_field(
        canonical, "observation_id", record.observation_id);
    append_noise_string_field(
        canonical, "ensemble_mode", record.ensemble_mode);
    append_noise_integer_field(
        canonical, "conditioning_iteration", record.conditioning_iteration);
    append_noise_string_field(canonical, "pass_id", record.pass_id);
    append_noise_integer_field(
        canonical, "pass_ordinal", record.pass_ordinal);
    append_noise_integer_field(
        canonical, "randomize_channels",
        record.randomize_channels ? 1 : 0);
    append_noise_string_field(
        canonical, "coherence_unit_identity_policy",
        record.coherence_unit_identity_policy);
    append_noise_string_field(
        canonical, "channel_identity_policy",
        record.channel_identity_policy);
    append_noise_string_field(
        canonical, "ordering_policy", record.ordering_policy);
    append_noise_integer_field(
        canonical, "coherence_unit_count", record.coherence_unit_count);
    append_noise_integer_field(
        canonical, "channel_count", record.channel_count);
    for (const auto realization : record.completed_realization_ids) {
        append_noise_integer_field(
            canonical, "completed_realization_id", realization);
    }
    append_noise_string_field(
        canonical, "namespace_digest", record.namespace_digest);
    append_noise_string_field(
        canonical, "partition_digest", record.partition_digest);
    return citlali::utils::sha256(canonical);
}

inline NoiseAssignmentRecord make_noise_assignment_record(
    const NoiseAssignmentContext &context) {
    NoiseAssignmentRecord record;
    record.observation_id = context.observation_id;
    record.ensemble_mode = context.ensemble_mode;
    record.conditioning_iteration = context.conditioning_iteration;
    record.pass_id = context.pass_id;
    record.pass_ordinal = context.pass_ordinal;
    record.randomize_channels = context.randomize_channels;
    record.coherence_unit_count = context.coherence_unit_count;
    record.channel_count = context.channel_count;
    record.completed_realization_ids.reserve(
        static_cast<std::size_t>(context.n_realizations));
    for (int realization = 0; realization < context.n_realizations;
         ++realization) {
        record.completed_realization_ids.push_back(
            static_cast<std::size_t>(realization));
    }
    record.namespace_digest = noise_assignment_namespace_digest(context);
    record.partition_digest = noise_assignment_partition_digest(context);
    record.reconstruction_digest =
        noise_assignment_record_reconstruction_digest(record);
    return record;
}

inline auto noise_assignment_record_sort_key(
    const NoiseAssignmentRecord &record) {
    return std::tie(
        record.observation_id, record.conditioning_iteration,
        record.pass_ordinal, record.pass_id, record.namespace_digest,
        record.partition_digest);
}

inline std::string noise_assignment_records_digest(
    const std::vector<NoiseAssignmentRecord> &records) {
    std::vector<std::string> digests;
    digests.reserve(records.size());
    for (const auto &record : records) {
        digests.push_back(record.reconstruction_digest);
    }
    std::sort(digests.begin(), digests.end());
    std::string canonical;
    append_noise_string_field(
        canonical, "key_policy_version",
        noise_realization_key_policy_version);
    for (const auto &digest : digests) {
        append_noise_string_field(canonical, "assignment_digest", digest);
    }
    return citlali::utils::sha256(canonical);
}

struct NoiseRealizationProductIdentity {
    std::string key_policy_version =
        noise_realization_key_policy_version;
    std::string ensemble_mode =
        noise_ensemble_mode_source_imprinted_current;
    std::string product_scope;
    std::size_t realization_id = 0;
    std::string assignment_digest;
    std::string product_digest_join;
};

inline NoiseRealizationProductIdentity noise_realization_product_identity(
    const std::vector<NoiseAssignmentRecord> &records,
    const std::string &observation_id, bool coadd,
    std::size_t realization_id) {
    std::map<std::string, const NoiseAssignmentRecord *> latest;
    for (const auto &record : records) {
        if (!coadd && record.observation_id != observation_id) {
            continue;
        }
        if (!record.compact() ||
            record.reconstruction_digest !=
                noise_assignment_record_reconstruction_digest(record)) {
            throw std::logic_error(
                "noise realization product assignment is not reconstructible");
        }
        if (std::find(
                record.completed_realization_ids.begin(),
                record.completed_realization_ids.end(), realization_id) ==
            record.completed_realization_ids.end()) {
            continue;
        }
        auto &selected = latest[record.observation_id];
        if (selected == nullptr ||
            std::tie(record.conditioning_iteration, record.pass_ordinal,
                     record.pass_id) >
                std::tie(selected->conditioning_iteration,
                         selected->pass_ordinal,
                         selected->pass_id)) {
            selected = &record;
        }
    }
    if (latest.empty() || (!coadd && latest.count(observation_id) != 1)) {
        throw std::logic_error(
            "noise realization product lacks a completed assignment join");
    }

    std::vector<NoiseAssignmentRecord> selected_records;
    selected_records.reserve(latest.size());
    for (const auto &[id, record] : latest) {
        (void)id;
        selected_records.push_back(*record);
    }
    const std::string assignment_digest = selected_records.size() == 1
        ? selected_records.front().reconstruction_digest
        : noise_assignment_records_digest(selected_records);
    const std::string product_scope = coadd
        ? std::string{"coadd"}
        : std::string{"observation:"} + observation_id;
    std::string canonical;
    append_noise_string_field(
        canonical, "key_policy_version",
        noise_realization_key_policy_version);
    append_noise_string_field(
        canonical, "ensemble_mode",
        noise_ensemble_mode_source_imprinted_current);
    append_noise_string_field(canonical, "product_scope", product_scope);
    append_noise_integer_field(
        canonical, "realization_id", realization_id);
    append_noise_string_field(
        canonical, "assignment_digest", assignment_digest);
    return NoiseRealizationProductIdentity{
        noise_realization_key_policy_version,
        noise_ensemble_mode_source_imprinted_current,
        product_scope,
        realization_id,
        assignment_digest,
        citlali::utils::sha256(canonical),
    };
}

}  // namespace citlali::pipeline

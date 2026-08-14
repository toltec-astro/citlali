#pragma once

// Beammap APT table output helpers. The canonical producer path is kept here
// as a standalone, directly testable boundary; Beammap glue only supplies
// authoritative observation and runtime context.

#include <citlali/core/engine/calib.h>
#include <citlali/core/engine/detail/beammap_apt_keys.h>
#include <citlali/core/pipeline/canonical_artifact_publication.h>
#include <citlali/core/pipeline/canonical_apt_ecsv.h>

#include <Eigen/Core>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace beammap_apt_table_output_helpers {

namespace apt = citlali::pipeline::canonical_apt_v1;
namespace publication =
    citlali::pipeline::canonical_artifact_publication;

template <class Calib, class Flag2>
Eigen::MatrixXd apt_table(Calib &calib, const Flag2 &flag2) {
    Eigen::MatrixXd table(calib.n_dets, calib.apt_header_keys.size());

    Eigen::Index col = 0;
    for (const auto &key : calib.apt_header_keys) {
        if (!beammap_apt_keys::is_flag2(key)) {
            table.col(col) = calib.apt[key];
        } else {
            table.col(col) = flag2.template cast<double>();
        }
        ++col;
    }
    return table;
}

struct CanonicalAptDocumentContext {
    std::string occurrence;
    std::string event_reference;
    std::string software_revision;
    std::string configuration_reference;
    std::string event_time_utc;
    std::string project_id;
    std::string source_name;
    std::string observation_time_utc;
    std::string coordinate_frame;
};

inline void inject_issuance_context(
    CanonicalAptDocumentContext &context,
    const engine::CanonicalAptProducerState &producer) {
    publication::OpaqueIssuance issuance;
    try {
        issuance = publication::issue_opaque(producer.issuance_factory);
    } catch (const publication::PublicationError &error) {
        throw apt::ContractError(
            "canonical Beammap APT occurrence/event issuance failed: " +
            std::string(error.what()));
    }
    context.occurrence = issuance.occurrence;
    context.event_reference = issuance.event_reference;
}

inline std::int64_t exact_legacy_int64(double value,
                                       std::string_view field) {
    // INT64_MAX rounds to 2^63 in binary64, so a comparison against
    // static_cast<double>(INT64_MAX) would admit 2^63 and make the cast below
    // implementation-defined.  Use the exact binary64 interval boundaries.
    constexpr double int64_lower = -0x1p63;
    constexpr double int64_upper_exclusive = 0x1p63;
    if (!std::isfinite(value) || std::trunc(value) != value ||
        value < int64_lower || value >= int64_upper_exclusive) {
        throw apt::ContractError(
            "legacy Beammap field is not an exact int64: " +
            std::string(field));
    }
    const auto result = static_cast<std::int64_t>(value);
    if (static_cast<double>(result) != value) {
        throw apt::ContractError(
            "legacy Beammap field loses integer precision: " +
            std::string(field));
    }
    return result;
}

inline std::optional<apt::RegisteredField> canonical_registered_field(
    std::string_view name) {
    const auto &registry = apt::canonical_field_registry_v1();
    for (const auto &field : registry.required_baseline_fields) {
        if (field.name == name) {
            return field;
        }
    }
    for (const auto &field : registry.optional_extensions) {
        if (field.name == name) {
            return field;
        }
    }
    return std::nullopt;
}

inline std::string normalized_kids_model_name(std::string_view name) {
    if (beammap_apt_keys::is_flag(name)) {
        return std::string(beammap_apt_keys::kids_flag());
    }
    return std::string(name);
}

inline apt::Value typed_legacy_value(double value,
                                     const apt::RegisteredField &field) {
    switch (field.type) {
    case apt::ValueType::int64:
        return exact_legacy_int64(value, field.name);
    case apt::ValueType::float64:
        if ((field.nonfinite == apt::NonFinitePolicy::reject &&
             !std::isfinite(value)) ||
            (field.nonfinite == apt::NonFinitePolicy::nan_token &&
             std::isinf(value))) {
            throw apt::ContractError(
                "legacy Beammap float violates the canonical nonfinite policy: " +
                field.name);
        }
        return value;
    case apt::ValueType::boolean:
    case apt::ValueType::string:
        throw apt::ContractError(
            "all-double Beammap storage cannot supply this registered type: " +
            field.name);
    }
    throw apt::ContractError("unsupported canonical Beammap field type");
}

template <class FitReports>
std::vector<std::string> preflight_atomic_kids_fit_reports(
    const FitReports &reports, const apt::RawManifest &raw_manifest) {
    if (reports.size() != raw_manifest.inputs.size() || reports.empty()) {
        throw apt::ContractError(
            "atomic KIDs fit-report count disagrees with raw manifest");
    }
    const auto &header = reports.front().header;
    if (header.empty()) {
        throw apt::ContractError("atomic KIDs fit-report header is empty");
    }
    std::vector<std::string> names;
    names.reserve(header.size());
    std::set<std::string> unique;
    std::vector<apt::RegisteredField> contracts;
    contracts.reserve(header.size());
    for (const auto &source_name : header) {
        // The accepted legacy source field is `flag`; the canonical artifact
        // name `kids_flag` is producer-owned and is not itself an admitted
        // input spelling. This also keeps `flag` and `flag2` collision-free.
        if (source_name == beammap_apt_keys::kids_flag() ||
            beammap_apt_keys::is_flag2(source_name)) {
            throw apt::ContractError(
                "literal source kids_flag/flag2 is not an admitted KIDs fit-report field");
        }
        const auto name = normalized_kids_model_name(source_name);
        const auto contract = canonical_registered_field(name);
        if (!contract || apt::detail::protected_contract_name(name) ||
            !unique.insert(name).second) {
            throw apt::ContractError(
                "KIDs fit-report field is protected, duplicate, or absent from the canonical v1 registry: " +
                name);
        }
        names.push_back(name);
        contracts.push_back(*contract);
    }
    for (std::size_t input_index = 0; input_index < reports.size();
         ++input_index) {
        const auto &report = reports[input_index];
        const auto &input = raw_manifest.inputs[input_index];
        if (report.network != input.network ||
            report.observation != raw_manifest.observation.observation ||
            report.source.empty()) {
            throw apt::ContractError(
                "KIDs fit-report network/obsid/source metadata disagrees with raw input");
        }
        if (report.header != header) {
            throw apt::ContractError(
                "KIDs fit-report field names/order differ across raw network inputs");
        }
        const auto &model = report.model;
        if (model.rows() != raw_manifest.inputs[input_index].channel_count ||
            model.cols() !=
                static_cast<Eigen::Index>(report.header.size())) {
            throw apt::ContractError(
                "KIDs fit-report rows/columns disagree with its bound raw input/header");
        }
        for (Eigen::Index row = 0; row < model.rows(); ++row) {
            for (Eigen::Index column = 0; column < model.cols(); ++column) {
                (void)typed_legacy_value(
                    model(row, column),
                    contracts[static_cast<std::size_t>(column)]);
            }
        }
    }
    return names;
}

template <class Calib, class FitReports>
void apply_atomic_kids_fit_report_overlay(
    Calib &calib, const FitReports &reports,
    const apt::RawManifest &raw_manifest) {
    const auto names =
        preflight_atomic_kids_fit_reports(reports, raw_manifest);

    Eigen::Index total_rows = 0;
    for (const auto &report : reports) {
        const auto &model = report.model;
        if (model.rows() < 0 ||
            total_rows > std::numeric_limits<Eigen::Index>::max() -
                model.rows()) {
            throw apt::ContractError(
                "KIDs fit-report row cardinality is not representable");
        }
        total_rows += model.rows();
    }
    if (calib.n_dets < 0 || total_rows != calib.n_dets) {
        throw apt::ContractError(
            "KIDs fit-report overlay disagrees with detector cardinality");
    }

    // Construct every typed/preflighted column before mutating Calib. Shape,
    // name, registry, protected-field, and conversion failures therefore
    // leave the current scientific state untouched.
    std::vector<std::pair<std::string, Eigen::VectorXd>> columns;
    columns.reserve(names.size());
    for (Eigen::Index column = 0;
         column < static_cast<Eigen::Index>(names.size()); ++column) {
        Eigen::VectorXd values(calib.n_dets);
        Eigen::Index offset = 0;
        for (const auto &report : reports) {
            const auto &model = report.model;
            values.segment(offset, model.rows()) = model.col(column);
            offset += model.rows();
        }
        columns.emplace_back(names[static_cast<std::size_t>(column)],
                             std::move(values));
    }

    for (auto &[name, values] : columns) {
        calib.apt[name] = std::move(values);
        if (std::find(calib.apt_header_keys.begin(),
                      calib.apt_header_keys.end(), name) ==
            calib.apt_header_keys.end()) {
            calib.apt_header_keys.push_back(name);
        }
        const auto contract = canonical_registered_field(name);
        calib.apt_header_units[name] = contract->unit;
        calib.apt_meta[name].push_back("units: " + contract->unit);
        calib.apt_meta[name].push_back(contract->description);
    }
}

template <class Calib>
const Eigen::VectorXd &required_legacy_vector(const Calib &calib,
                                              const std::string &name) {
    const auto found = calib.apt.find(name);
    if (found == calib.apt.end() || found->second.size() != calib.n_dets) {
        throw apt::ContractError(
            "Beammap APT field is absent or has wrong detector cardinality: " +
            name);
    }
    return found->second;
}

inline std::int64_t exact_nonnegative_decimal(std::string_view value,
                                               std::string_view label) {
    std::int64_t result = -1;
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), result);
    if (value.empty() || error != std::errc{} ||
        end != value.data() + value.size() || result < 0) {
        throw apt::ContractError(
            "canonical Beammap APT has invalid " + std::string(label));
    }
    return result;
}

template <class TelescopeHeader>
apt::ObservationIdentity telescope_observation_identity(
    const TelescopeHeader &header, bool simulation) {
    const std::string prefix = simulation ? "Header.TelescopeBackend."
                                          : "Header.Dcs.";
    const auto read = [&](std::string_view suffix) {
        const auto name = prefix + std::string(suffix);
        const auto found = header.find(name);
        if (found == header.end() || found->second.size() != 1) {
            throw apt::ContractError(
                "canonical Beammap APT is missing scalar telescope identity header " +
                name);
        }
        const auto value = exact_legacy_int64(found->second(0), name);
        if (value < 0) {
            throw apt::ContractError(
                "canonical Beammap APT telescope identity is negative: " +
                name);
        }
        return value;
    };
    return {read("ObsNum"), read("SubObsNum"), read("ScanNum")};
}

template <class TelescopeData>
double telescope_observation_unix_time(const TelescopeData &data) {
    const auto found = data.find("TelTime");
    if (found == data.end() || found->second.size() == 0 ||
        !std::isfinite(found->second(0))) {
        throw apt::ContractError(
            "canonical Beammap APT requires authoritative first telescope TelTime");
    }
    return found->second(0);
}

template <class Calib>
void validate_current_raw_binding(const Calib &calib) {
    const auto &producer = calib.canonical_apt_producer;
    if (!producer.raw_inventory_ready || calib.n_dets <= 0 ||
        producer.rows.size() != static_cast<std::size_t>(calib.n_dets)) {
        throw apt::ContractError(
            "canonical Beammap APT has no complete retained raw binding");
    }
    const auto &uid_values = required_legacy_vector(calib, "uid");
    const auto &tone_values = required_legacy_vector(calib, "tone_freq");
    const auto &array_values = required_legacy_vector(calib, "array");
    const auto &network_values = required_legacy_vector(calib, "nw");
    const auto &channel_values = required_legacy_vector(calib, "kids_tone");
    for (Eigen::Index index = 0; index < calib.n_dets; ++index) {
        const auto &expected = producer.rows[static_cast<std::size_t>(index)];
        if (exact_legacy_int64(uid_values(index), "uid") != expected.uid ||
            exact_legacy_int64(array_values(index), "array") !=
                expected.array ||
            exact_legacy_int64(network_values(index), "nw") !=
                expected.network ||
            exact_legacy_int64(channel_values(index), "kids_tone") !=
                expected.channel ||
            std::bit_cast<std::uint64_t>(tone_values(index)) !=
                std::bit_cast<std::uint64_t>(
                    expected.tone_frequency_hz)) {
            throw apt::ContractError(
                "Beammap APT structure/ToneFreq drifted from retained raw authority");
        }
    }
}

template <class Calib, class Flag2>
apt::Document make_canonical_document(
    const Calib &calib, const Flag2 &flag2,
    const CanonicalAptDocumentContext &context) {
    validate_current_raw_binding(calib);
    const auto &producer = calib.canonical_apt_producer;
    if (!producer.raw_inventory_ready || calib.n_dets <= 0 ||
        producer.rows.size() != static_cast<std::size_t>(calib.n_dets) ||
        flag2.size() != calib.n_dets) {
        throw apt::ContractError(
            "canonical Beammap APT requires a retained complete raw inventory and exact detector/flag2 cardinality");
    }

    const std::set<std::string> core_names{
        "uid", "tone_freq", "array", "nw", "kids_tone"};
    std::set<std::string> header_names;
    std::map<std::string, apt::RegisteredField> registered;
    for (const auto &name : calib.apt_header_keys) {
        if (!header_names.insert(name).second) {
            throw apt::ContractError(
                "duplicate Beammap APT output header: " + name);
        }
        if (core_names.contains(name)) {
            continue;
        }
        const auto contract = canonical_registered_field(name);
        if (!contract || apt::detail::protected_contract_name(name)) {
            throw apt::ContractError(
                "Beammap APT output field is outside the canonical v1 registry: " +
                name);
        }
        registered.emplace(name, *contract);
    }
    for (const auto &name : core_names) {
        if (!header_names.contains(name)) {
            throw apt::ContractError(
                "Beammap APT output omits required structural field: " + name);
        }
    }
    for (const auto &field : apt::required_baseline_fields_v1()) {
        if (!registered.contains(field.name)) {
            throw apt::ContractError(
                "Beammap APT output omits required canonical field: " +
                field.name);
        }
    }

    apt::Document document;
    document.envelope = {
        context.occurrence,
        context.event_reference,
        std::string(apt::baseline_output_role_v1),
        "citlali",
        context.software_revision,
        context.configuration_reference,
        context.event_time_utc,
    };
    document.context = {
        context.project_id,
        context.source_name,
        context.observation_time_utc,
        context.coordinate_frame,
    };
    document.raw_manifest = producer.raw_manifest;
    for (const auto &[name, field] : registered) {
        (void)name;
        document.registered_fields.push_back(field);
    }

    const auto &uid_values = required_legacy_vector(calib, "uid");
    const auto &tone_values = required_legacy_vector(calib, "tone_freq");
    const auto &array_values = required_legacy_vector(calib, "array");
    const auto &network_values = required_legacy_vector(calib, "nw");
    const auto &channel_values = required_legacy_vector(calib, "kids_tone");
    document.rows.reserve(static_cast<std::size_t>(calib.n_dets));
    for (Eigen::Index index = 0; index < calib.n_dets; ++index) {
        const auto &expected =
            producer.rows[static_cast<std::size_t>(index)];
        const auto uid = exact_legacy_int64(uid_values(index), "uid");
        const auto array = exact_legacy_int64(array_values(index), "array");
        const auto network = exact_legacy_int64(network_values(index), "nw");
        const auto channel =
            exact_legacy_int64(channel_values(index), "kids_tone");
        const double tone = tone_values(index);
        if (uid != expected.uid || array != expected.array ||
            network != expected.network || channel != expected.channel ||
            std::bit_cast<std::uint64_t>(tone) !=
                std::bit_cast<std::uint64_t>(expected.tone_frequency_hz)) {
            throw apt::ContractError(
                "Beammap APT structure/ToneFreq drifted from the retained raw channel relation");
        }

        apt::Row row{uid, tone, array, network, channel, {}};
        for (const auto &[name, contract] : registered) {
            if (name == "flag2") {
                row.fields[name] = static_cast<std::int64_t>(flag2(index));
            } else {
                const auto &values = required_legacy_vector(calib, name);
                row.fields[name] = typed_legacy_value(values(index), contract);
            }
        }
        document.rows.push_back(std::move(row));
    }
    apt::validate(document);
    return document;
}

inline std::string utc_timestamp_from_time_t(std::time_t seconds) {
    std::tm utc{};
#if defined(_WIN32)
    if (gmtime_s(&utc, &seconds) != 0) {
#else
    if (gmtime_r(&seconds, &utc) == nullptr) {
#endif
        throw apt::ContractError(
            "canonical Beammap APT could not format UTC source time");
    }
    char buffer[32]{};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc) ==
        0) {
        throw apt::ContractError(
            "canonical Beammap APT UTC timestamp formatting failed");
    }
    const std::string result{buffer};
    if (!apt::detail::is_utc_timestamp(result)) {
        throw apt::ContractError(
            "canonical Beammap APT UTC source time is outside the v1 calendar range");
    }
    return result;
}

inline std::string utc_timestamp_from_unix_seconds(double unix_seconds) {
    const auto floored = std::floor(unix_seconds);
    static_assert(std::numeric_limits<std::time_t>::is_integer);
    constexpr int time_digits = std::numeric_limits<std::time_t>::digits;
    const double upper_exclusive = std::ldexp(1.0, time_digits);
    const double lower = std::numeric_limits<std::time_t>::is_signed
        ? -upper_exclusive
        : 0.0;
    if (!std::isfinite(unix_seconds) || floored < lower ||
        floored >= upper_exclusive) {
        throw apt::ContractError(
            "canonical Beammap APT UTC source time is not representable");
    }
    return utc_timestamp_from_time_t(static_cast<std::time_t>(floored));
}

inline std::string current_utc_timestamp() {
    return utc_timestamp_from_time_t(
        std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now()));
}

inline std::string canonical_apt_receipt_bytes(
    const apt::ByteTransportHash &transport) {
    if (transport.scope != apt::byte_transport_scope_v1 ||
        !apt::is_sha256_reference(transport.envelope_sha256) ||
        !apt::is_sha256_reference(transport.sha256)) {
        throw apt::ContractError(
            "canonical APT publication receipt has invalid scope or digest");
    }
    return publication::canonical_receipt_bytes(
        {std::string(publication::receipt_schema_v1), transport.scope,
         transport.envelope_sha256, transport.sha256,
         transport.byte_count});
}

inline apt::ByteTransportHash parse_canonical_apt_receipt(
    std::string_view bytes) {
    try {
        const auto receipt = publication::parse_canonical_receipt(
            bytes, publication::receipt_schema_v1,
            apt::byte_transport_scope_v1);
        return {receipt.scope, receipt.envelope_sha256,
                receipt.byte_sha256, receipt.byte_count};
    } catch (const publication::PublicationError &error) {
        throw apt::ContractError(
            "canonical APT receipt validation failed: " +
            std::string(error.what()));
    }
}

using PublicationStage = publication::PublicationStage;
using CanonicalAptPublicationHooks = publication::PublicationHooks;

struct CanonicalAptPublicationResult {
    std::filesystem::path ecsv_path;
    std::filesystem::path receipt_path;
    apt::Digests digests;
    apt::ByteTransportHash transport;
};

inline void validate_canonical_apt_bytes_and_receipt(
    std::string_view bytes, const apt::ByteTransportHash &transport) {
    (void)apt::parse_ecsv_with_transport(bytes, transport);
}

inline void validate_canonical_apt_bytes_and_receipt(
    std::string_view bytes, std::string_view receipt_bytes) {
    const auto transport = parse_canonical_apt_receipt(receipt_bytes);
    validate_canonical_apt_bytes_and_receipt(bytes, transport);
}

inline CanonicalAptPublicationResult validate_published_canonical_apt(
    const std::filesystem::path &ecsv_path,
    const std::filesystem::path &receipt_path) {
    std::optional<apt::ParsedEcsv> parsed;
    std::optional<apt::ByteTransportHash> transport;
    publication::validate_published_canonical_artifact(
        ecsv_path, receipt_path,
        [&](std::string_view bytes, std::string_view receipt_bytes) {
            transport = parse_canonical_apt_receipt(receipt_bytes);
            parsed = apt::parse_ecsv_with_transport(bytes, *transport);
        });
    return {ecsv_path, receipt_path, parsed->declared_digests, *transport};
}

inline CanonicalAptPublicationResult publish_canonical_apt(
    const apt::Document &document, const std::filesystem::path &ecsv_path,
    const CanonicalAptPublicationHooks &hooks = {}) {
    if (ecsv_path.extension() != ".ecsv") {
        throw std::runtime_error(
            "canonical APT publication target must end in .ecsv");
    }
    const auto receipt_path =
        std::filesystem::path(ecsv_path.string() + ".sha256");
    if (std::filesystem::exists(ecsv_path) ||
        std::filesystem::exists(receipt_path)) {
        throw std::runtime_error(
            "canonical APT refuses to overwrite an existing artifact or receipt");
    }

    const auto intended = apt::serialize_ecsv(document);
    CanonicalAptPublicationResult result{
        ecsv_path, receipt_path, intended.digests, intended.transport};
    const auto receipt_bytes =
        canonical_apt_receipt_bytes(intended.transport);
    publication::PublicationPlan plan{
        ecsv_path, receipt_path, intended.bytes, receipt_bytes,
        [&](std::string_view bytes, std::string_view receipt) {
            const auto transport = parse_canonical_apt_receipt(receipt);
            const auto parsed =
                apt::parse_ecsv_with_transport(bytes, transport);
            if (parsed.declared_digests.semantic_sha256 !=
                    intended.digests.semantic_sha256 ||
                parsed.declared_digests.envelope_sha256 !=
                    intended.digests.envelope_sha256 ||
                parsed.computed_transport.scope != intended.transport.scope ||
                parsed.computed_transport.envelope_sha256 !=
                    intended.transport.envelope_sha256 ||
                parsed.computed_transport.sha256 != intended.transport.sha256 ||
                parsed.computed_transport.byte_count !=
                    intended.transport.byte_count) {
                throw std::runtime_error(
                    "canonical APT reread disagrees with intended integrity values");
            }
        }};
    (void)publication::publish_canonical_artifact(plan, hooks);
    return result;
}

}  // namespace beammap_apt_table_output_helpers

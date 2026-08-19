#pragma once

#include <citlali/core/pipeline/canonical_apt_ecsv.h>
#include <citlali/core/pipeline/canonical_apt_v2.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace citlali::pipeline::canonical_apt_v2 {

namespace ecsv_v1 = citlali::pipeline::canonical_apt_v1::ecsv_detail;

namespace detail {

enum class FlatValueType { int64, uint64, float64, boolean, string };

inline std::string_view flat_value_type_token(FlatValueType type) {
    switch (type) {
    case FlatValueType::int64: return "int64";
    case FlatValueType::uint64: return "uint64";
    case FlatValueType::float64: return "float64";
    case FlatValueType::boolean: return "bool";
    case FlatValueType::string: return "string";
    }
    throw ContractError("unsupported canonical APT v2 flat value type");
}

inline FlatValueType parse_flat_value_type(std::string_view token) {
    if (token == "int64") return FlatValueType::int64;
    if (token == "uint64") return FlatValueType::uint64;
    if (token == "float64") return FlatValueType::float64;
    if (token == "bool") return FlatValueType::boolean;
    if (token == "string") return FlatValueType::string;
    throw ContractError("unsupported canonical APT v2 flat value type token: " +
                        std::string(token));
}

struct UInt64Value {
    std::uint64_t value = 0;
    friend bool operator==(const UInt64Value &, const UInt64Value &) = default;
};

using FlatValue =
    std::variant<NullValue, std::int64_t, UInt64Value, double, bool, std::string>;

struct FlatColumn {
    std::string name;
    FlatValueType datatype = FlatValueType::string;
    std::string unit{"N/A"};
    bool nullable = false;

    friend bool operator==(const FlatColumn &, const FlatColumn &) = default;
};

struct FlatComponent {
    std::string role;
    std::string schema;
    BundleKind kind = BundleKind::baseline;
    IssuanceContext issuance;
    ObservationIdentity observation;
    std::vector<std::pair<std::string, std::string>> metadata;
    std::vector<FlatColumn> columns;
    std::vector<std::vector<FlatValue>> rows;

    friend bool operator==(const FlatComponent &, const FlatComponent &) =
        default;
};

struct ParsedFlatComponent {
    FlatComponent document;
    ComponentDigests digests;
};

inline std::string_view document_semantic_scope(
    const FlatComponent &document) {
    if (document.role != "manifest") return semantic_scope_v2;
    return document.kind == BundleKind::baseline
        ? baseline_bundle_semantic_scope_v2
        : matched_bundle_semantic_scope_v2;
}

inline std::string_view document_envelope_scope(
    const FlatComponent &document) {
    if (document.role != "manifest") return envelope_scope_v2;
    return document.kind == BundleKind::baseline
        ? baseline_bundle_envelope_scope_v2
        : matched_bundle_envelope_scope_v2;
}

inline void require_exact_columns(const FlatComponent &document,
                                  const std::vector<FlatColumn> &expected) {
    if (document.columns != expected) {
        throw ContractError("canonical APT v2 component columns are not exact");
    }
}

inline std::string value_payload(const FlatValue &value, FlatValueType type) {
    if (std::holds_alternative<NullValue>(value)) return "";
    if (const auto integer = std::get_if<std::int64_t>(&value)) {
        if (type != FlatValueType::int64) {
            throw ContractError("canonical APT v2 ECSV cell type mismatch");
        }
        return std::to_string(*integer);
    }
    if (const auto integer = std::get_if<UInt64Value>(&value)) {
        if (type != FlatValueType::uint64) {
            throw ContractError("canonical APT v2 ECSV cell type mismatch");
        }
        return std::to_string(integer->value);
    }
    if (const auto number = std::get_if<double>(&value)) {
        if (type != FlatValueType::float64) {
            throw ContractError("canonical APT v2 ECSV cell type mismatch");
        }
        return ecsv_v1::format_float64(*number);
    }
    if (const auto boolean = std::get_if<bool>(&value)) {
        if (type != FlatValueType::boolean) {
            throw ContractError("canonical APT v2 ECSV cell type mismatch");
        }
        return *boolean ? "true" : "false";
    }
    if (type != FlatValueType::string) {
        throw ContractError("canonical APT v2 ECSV cell type mismatch");
    }
    return std::get<std::string>(value);
}

inline void add_flat_value(std::string &result, std::string label,
                           const FlatValue &value, FlatValueType type) {
    if (std::holds_alternative<NullValue>(value)) {
        result += canonical_frame(label,
                                  "null-" + std::string(flat_value_type_token(type)),
                                  "null");
        return;
    }
    std::visit(
        [&](const auto &typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, std::int64_t>) {
                add_int64(result, label, typed);
            } else if constexpr (std::is_same_v<T, UInt64Value>) {
                add_uint64(result, label, typed.value);
            } else if constexpr (std::is_same_v<T, double>) {
                add_float64(result, label, typed);
            } else if constexpr (std::is_same_v<T, bool>) {
                add_bool(result, label, typed);
            } else if constexpr (std::is_same_v<T, std::string>) {
                add_text(result, label, typed);
            }
        }, value);
}

inline std::string semantic_preimage(const FlatComponent &document) {
    validate(document.issuance);
    require_text(document.role, "component role");
    require_text(document.schema, "component schema");
    if (document.columns.empty()) {
        throw ContractError("canonical APT v2 component has no columns");
    }
    std::string result;
    add_text(result, "scope", document_semantic_scope(document));
    add_text(result, "schema", document.schema);
    add_text(result, "role", document.role);
    add_text(result, "product-kind", product_kind_token(document.kind));
    validate_observation(document.observation);
    add_int64(result, "observation.obsnum", document.observation.observation);
    add_int64(result, "observation.subobsnum",
              document.observation.subobservation);
    add_int64(result, "observation.scannum", document.observation.scan);
    auto metadata = document.metadata;
    std::sort(metadata.begin(), metadata.end());
    if (std::adjacent_find(metadata.begin(), metadata.end(),
                           [](const auto &lhs, const auto &rhs) {
                               return lhs.first == rhs.first;
                           }) != metadata.end()) {
        throw ContractError("canonical APT v2 component metadata key is duplicate");
    }
    add_uint64(result, "metadata.count", metadata.size());
    for (std::size_t index = 0; index < metadata.size(); ++index) {
        require_text(metadata[index].first, "metadata key");
        if (!v1::detail::canonical_text(metadata[index].second, true)) {
            throw ContractError("canonical APT v2 metadata value is invalid");
        }
        const auto prefix = "metadata." + std::to_string(index);
        add_text(result, prefix + ".key", metadata[index].first);
        add_text(result, prefix + ".value", metadata[index].second);
    }
    add_uint64(result, "column.count", document.columns.size());
    std::set<std::string> column_names;
    for (std::size_t index = 0; index < document.columns.size(); ++index) {
        const auto &column = document.columns[index];
        if (!column_names.insert(column.name).second) {
            throw ContractError("canonical APT v2 component column is duplicate");
        }
        require_text(column.name, "column name");
        require_text(column.unit, "column unit");
        const auto prefix = "column." + std::to_string(index);
        add_text(result, prefix + ".name", column.name);
        add_text(result, prefix + ".datatype",
                 flat_value_type_token(column.datatype));
        add_text(result, prefix + ".unit", column.unit);
        add_bool(result, prefix + ".nullable", column.nullable);
    }
    add_uint64(result, "row.count", document.rows.size());
    for (std::size_t row_index = 0; row_index < document.rows.size(); ++row_index) {
        const auto &row = document.rows[row_index];
        if (row.size() != document.columns.size()) {
            throw ContractError("canonical APT v2 component row width is invalid");
        }
        for (std::size_t column_index = 0;
             column_index < document.columns.size(); ++column_index) {
            const auto &column = document.columns[column_index];
            const auto &value = row[column_index];
            if (std::holds_alternative<NullValue>(value)) {
                if (!column.nullable) {
                    throw ContractError("canonical APT v2 nonnullable cell is null");
                }
            }
            (void)value_payload(value, column.datatype);
            add_flat_value(
                result, "row." + std::to_string(row_index) + "." + column.name,
                value, column.datatype);
        }
    }
    return result;
}

inline std::string ecsv_datatype(FlatValueType type) {
    return std::string(flat_value_type_token(type));
}

inline void emit_yaml_value(std::ostringstream &stream,
                            std::string_view prefix,
                            std::string_view value) {
    stream << prefix << ecsv_v1::yaml_quote(value) << '\n';
}

inline std::string serialize_flat_bytes(const FlatComponent &document,
                                        const ComponentDigests &digests) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "# %ECSV 1.0\n# ---\n# datatype:\n";
    for (const auto &column : document.columns) {
        stream << "# - name: " << ecsv_v1::yaml_quote(column.name) << '\n'
               << "#   datatype: "
               << ecsv_v1::yaml_quote(ecsv_datatype(column.datatype)) << '\n'
               << "#   unit: " << ecsv_v1::yaml_quote(column.unit) << '\n'
               << "#   nullable: " << (column.nullable ? "true" : "false")
               << '\n';
    }
    stream << "# meta:\n#   canonical_apt_v2:\n";
    emit_yaml_value(stream, "#     schema_version: ", document.schema);
    emit_yaml_value(stream, "#     component_role: ", document.role);
    emit_yaml_value(stream, "#     product_kind: ",
                    product_kind_token(document.kind));
    emit_yaml_value(stream, "#     occurrence: ", document.issuance.occurrence);
    emit_yaml_value(stream, "#     event_reference: ",
                    document.issuance.event_reference);
    emit_yaml_value(stream, "#     producer: ", document.issuance.producer);
    emit_yaml_value(stream, "#     software_revision: ",
                    document.issuance.software_revision);
    emit_yaml_value(stream, "#     configuration_reference: ",
                    document.issuance.configuration_reference);
    emit_yaml_value(stream, "#     event_time_utc: ",
                    document.issuance.event_time_utc);
    stream << "#     observation:\n"
           << "#       obsnum: " << document.observation.observation << '\n'
           << "#       subobsnum: " << document.observation.subobservation << '\n'
           << "#       scannum: " << document.observation.scan << '\n';
    emit_yaml_value(stream, "#     semantic_scope: ",
                    document_semantic_scope(document));
    emit_yaml_value(stream, "#     semantic_sha256: ", digests.semantic_sha256);
    emit_yaml_value(stream, "#     envelope_scope: ",
                    document_envelope_scope(document));
    emit_yaml_value(stream, "#     envelope_sha256: ", digests.envelope_sha256);
    stream << "#     row_count: " << document.rows.size() << '\n';
    auto metadata = document.metadata;
    std::sort(metadata.begin(), metadata.end());
    stream << "#     role_metadata_count: " << metadata.size() << '\n'
           << "#     role_metadata:\n";
    for (const auto &[key, value] : metadata) {
        stream << "#       - key: " << ecsv_v1::yaml_quote(key) << '\n';
        emit_yaml_value(stream, "#         value: ", value);
    }
    stream << "# delimiter: ','\n# schema: astropy-2.0\n";
    for (std::size_t index = 0; index < document.columns.size(); ++index) {
        if (index) stream << ',';
        stream << ecsv_v1::csv_quote(document.columns[index].name);
    }
    stream << '\n';
    for (const auto &row : document.rows) {
        for (std::size_t index = 0; index < row.size(); ++index) {
            if (index) stream << ',';
            if (std::holds_alternative<NullValue>(row[index])) {
                continue;
            }
            const auto payload =
                value_payload(row[index], document.columns[index].datatype);
            if (document.columns[index].datatype == FlatValueType::string) {
                stream << ecsv_v1::csv_quote(payload);
            } else {
                stream << payload;
            }
        }
        stream << '\n';
    }
    return stream.str();
}

inline SerializedComponent serialize_component(FlatComponent document) {
    const auto preimage = semantic_preimage(document);
    auto digests = make_component_digests(
        preimage, document.issuance, {}, document_envelope_scope(document));
    auto bytes = serialize_flat_bytes(document, digests);
    digests.transport_sha256 =
        "sha256:" + citlali::utils::sha256(bytes);
    digests.byte_count = static_cast<std::uint64_t>(bytes.size());
    return {std::move(document.role), std::move(document.schema),
            std::move(bytes), std::move(digests),
            static_cast<std::uint64_t>(document.rows.size())};
}

class EcsvLineReader {
public:
    explicit EcsvLineReader(std::string_view bytes) : bytes_(bytes) {
        if (bytes.empty() || bytes.back() != '\n' ||
            bytes.find('\r') != bytes.npos || bytes.find('\0') != bytes.npos) {
            throw ContractError("canonical APT v2 ECSV requires exact LF UTF-8 framing");
        }
    }

    std::string_view line() {
        if (offset_ >= bytes_.size()) {
            throw ContractError("canonical APT v2 ECSV ended unexpectedly");
        }
        const auto end = bytes_.find('\n', offset_);
        const auto result = bytes_.substr(offset_, end - offset_);
        offset_ = end + 1;
        return result;
    }

    std::string quoted(std::string_view prefix) {
        const auto current = line();
        if (!current.starts_with(prefix)) {
            throw ContractError("canonical APT v2 ECSV metadata is missing or reordered");
        }
        return ecsv_v1::yaml_unquote(current.substr(prefix.size()));
    }

    std::uint64_t uint64(std::string_view prefix) {
        const auto current = line();
        if (!current.starts_with(prefix)) {
            throw ContractError("canonical APT v2 ECSV count is missing or reordered");
        }
        std::uint64_t result = 0;
        const auto value = current.substr(prefix.size());
        const auto [end, error] = std::from_chars(
            value.data(), value.data() + value.size(), result, 10);
        if (error != std::errc{} || end != value.data() + value.size()) {
            throw ContractError("canonical APT v2 ECSV count is invalid");
        }
        return result;
    }

    std::string_view remaining() const { return bytes_.substr(offset_); }

private:
    std::string_view bytes_;
    std::size_t offset_ = 0;
};

inline std::uint64_t exact_uint64(std::string_view value,
                                  std::string_view label);

inline FlatValue parse_cell(std::string_view value, const FlatColumn &column) {
    if (value.empty()) {
        if (!column.nullable) {
            throw ContractError("canonical APT v2 ECSV nonnullable cell is empty");
        }
        return NullValue{};
    }
    switch (column.datatype) {
    case FlatValueType::int64:
        return ecsv_v1::parse_int64(value, column.name);
    case FlatValueType::uint64:
        return UInt64Value{exact_uint64(value, column.name)};
    case FlatValueType::float64:
        return ecsv_v1::parse_float64(value, column.name);
    case FlatValueType::boolean:
        if (value == "true") return true;
        if (value == "false") return false;
        throw ContractError("canonical APT v2 ECSV bool cell is invalid");
    case FlatValueType::string:
        if (!v1::detail::canonical_text(value, true)) {
            throw ContractError("canonical APT v2 ECSV string cell is invalid");
        }
        return std::string(value);
    }
    throw ContractError("canonical APT v2 ECSV datatype is unsupported");
}

inline ParsedFlatComponent parse_component(std::string_view bytes) {
    EcsvLineReader reader(bytes);
    if (reader.line() != "# %ECSV 1.0" || reader.line() != "# ---" ||
        reader.line() != "# datatype:") {
        throw ContractError("canonical APT v2 ECSV preamble is invalid");
    }
    std::vector<FlatColumn> columns;
    std::string_view pending = reader.line();
    while (pending.starts_with("# - name: ")) {
        FlatColumn column;
        column.name = ecsv_v1::yaml_unquote(pending.substr(10));
        column.datatype =
            parse_flat_value_type(reader.quoted("#   datatype: "));
        column.unit = reader.quoted("#   unit: ");
        const auto nullable = reader.line();
        if (nullable == "#   nullable: true") {
            column.nullable = true;
        } else if (nullable == "#   nullable: false") {
            column.nullable = false;
        } else {
            throw ContractError("canonical APT v2 ECSV nullable token is invalid");
        }
        columns.push_back(std::move(column));
        pending = reader.line();
    }
    if (pending != "# meta:" || reader.line() != "#   canonical_apt_v2:") {
        throw ContractError("canonical APT v2 ECSV metadata root is invalid");
    }
    FlatComponent document;
    document.columns = std::move(columns);
    document.schema = reader.quoted("#     schema_version: ");
    document.role = reader.quoted("#     component_role: ");
    document.kind =
        parse_product_kind(reader.quoted("#     product_kind: "));
    document.issuance.occurrence = reader.quoted("#     occurrence: ");
    document.issuance.event_reference =
        reader.quoted("#     event_reference: ");
    document.issuance.producer = reader.quoted("#     producer: ");
    document.issuance.software_revision =
        reader.quoted("#     software_revision: ");
    document.issuance.configuration_reference =
        reader.quoted("#     configuration_reference: ");
    document.issuance.event_time_utc =
        reader.quoted("#     event_time_utc: ");
    if (reader.line() != "#     observation:") {
        throw ContractError("canonical APT v2 observation metadata is missing");
    }
    document.observation.observation = static_cast<std::int64_t>(
        reader.uint64("#       obsnum: "));
    document.observation.subobservation = static_cast<std::int64_t>(
        reader.uint64("#       subobsnum: "));
    document.observation.scan = static_cast<std::int64_t>(
        reader.uint64("#       scannum: "));
    const auto declared_semantic_scope =
        reader.quoted("#     semantic_scope: ");
    if (declared_semantic_scope != document_semantic_scope(document)) {
        throw ContractError("canonical APT v2 semantic scope is invalid");
    }
    const auto declared_semantic = reader.quoted("#     semantic_sha256: ");
    const auto declared_envelope_scope =
        reader.quoted("#     envelope_scope: ");
    if (declared_envelope_scope != document_envelope_scope(document)) {
        throw ContractError("canonical APT v2 envelope scope is invalid");
    }
    const auto declared_envelope = reader.quoted("#     envelope_sha256: ");
    const auto row_count = reader.uint64("#     row_count: ");
    const auto metadata_count =
        reader.uint64("#     role_metadata_count: ");
    if (reader.line() != "#     role_metadata:") {
        throw ContractError("canonical APT v2 ECSV metadata list is invalid");
    }
    for (std::uint64_t index = 0; index < metadata_count; ++index) {
        auto key = reader.quoted("#       - key: ");
        auto value = reader.quoted("#         value: ");
        document.metadata.emplace_back(std::move(key), std::move(value));
    }
    if (reader.line() != "# delimiter: ','" ||
        reader.line() != "# schema: astropy-2.0") {
        throw ContractError("canonical APT v2 ECSV trailer metadata is invalid");
    }
    const auto header = ecsv_v1::parse_csv_line(reader.line());
    if (header.size() != document.columns.size()) {
        throw ContractError("canonical APT v2 ECSV header width is invalid");
    }
    for (std::size_t index = 0; index < header.size(); ++index) {
        if (header[index].value != document.columns[index].name ||
            !header[index].quoted) {
            throw ContractError("canonical APT v2 ECSV header is noncanonical");
        }
    }
    auto remaining = reader.remaining();
    std::size_t start = 0;
    while (start < remaining.size()) {
        const auto end = remaining.find('\n', start);
        const auto line = remaining.substr(start, end - start);
        if (line.empty()) {
            throw ContractError("canonical APT v2 ECSV contains an empty row");
        }
        const auto cells = ecsv_v1::parse_csv_line(line);
        if (cells.size() != document.columns.size()) {
            throw ContractError("canonical APT v2 ECSV row width is invalid");
        }
        std::vector<FlatValue> row;
        row.reserve(cells.size());
        for (std::size_t index = 0; index < cells.size(); ++index) {
            const bool is_null = cells[index].value.empty();
            const bool string_column =
                document.columns[index].datatype == FlatValueType::string;
            if ((is_null && cells[index].quoted) ||
                (!is_null && string_column != cells[index].quoted)) {
                throw ContractError("canonical APT v2 ECSV cell quoting is invalid");
            }
            row.push_back(parse_cell(cells[index].value,
                                     document.columns[index]));
        }
        document.rows.push_back(std::move(row));
        start = end + 1;
    }
    if (document.rows.size() != row_count) {
        throw ContractError("canonical APT v2 ECSV row count is inconsistent");
    }
    const auto preimage = semantic_preimage(document);
    auto digests = make_component_digests(
        preimage, document.issuance, {}, document_envelope_scope(document));
    if (digests.semantic_sha256 != declared_semantic ||
        digests.envelope_sha256 != declared_envelope) {
        throw ContractError("canonical APT v2 ECSV semantic/envelope digest mismatch");
    }
    digests.transport_sha256 =
        "sha256:" + citlali::utils::sha256(bytes);
    digests.byte_count = static_cast<std::uint64_t>(bytes.size());
    if (serialize_flat_bytes(document, digests) != bytes) {
        throw ContractError("canonical APT v2 ECSV bytes are not canonical");
    }
    return {std::move(document), std::move(digests)};
}

inline std::map<std::string, std::string> metadata_map(
    const FlatComponent &document) {
    std::map<std::string, std::string> result;
    for (const auto &[key, value] : document.metadata) {
        if (!result.emplace(key, value).second) {
            throw ContractError("canonical APT v2 metadata key is duplicate");
        }
    }
    return result;
}

inline void require_exact_metadata_keys(
    const std::map<std::string, std::string> &metadata,
    const std::set<std::string> &expected, std::string_view role) {
    std::set<std::string> actual;
    for (const auto &[key, unused] : metadata) {
        (void)unused;
        actual.insert(key);
    }
    if (actual != expected) {
        throw ContractError("canonical APT v2 " + std::string(role) +
                            " metadata keys are not exact");
    }
}

inline std::int64_t exact_int64(std::string_view value,
                                std::string_view label) {
    std::int64_t result = 0;
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), result, 10);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw ContractError("canonical APT v2 " + std::string(label) +
                            " is not exact int64");
    }
    return result;
}

inline std::uint64_t exact_uint64(std::string_view value,
                                  std::string_view label) {
    std::uint64_t result = 0;
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), result, 10);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw ContractError("canonical APT v2 " + std::string(label) +
                            " is not exact uint64");
    }
    return result;
}

inline double binary64_from_payload(std::string_view value) {
    if (value.size() != 16) {
        throw ContractError("canonical APT v2 binary64 token length is invalid");
    }
    std::uint64_t bits = 0;
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), bits, 16);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw ContractError("canonical APT v2 binary64 token is invalid");
    }
    return std::bit_cast<double>(bits);
}

inline std::string metadata_required(
    const std::map<std::string, std::string> &metadata,
    const std::string &key) {
    const auto found = metadata.find(key);
    if (found == metadata.end()) {
        throw ContractError("canonical APT v2 required metadata is absent: " + key);
    }
    return found->second;
}

inline std::vector<FlatColumn> field_columns() {
    return {
        {"field_uid", FlatValueType::int64, "N/A", false},
        {"name", FlatValueType::string, "N/A", false},
        {"datatype", FlatValueType::string, "N/A", false},
        {"unit", FlatValueType::string, "N/A", false},
        {"nullable", FlatValueType::boolean, "N/A", false},
        {"authority", FlatValueType::string, "N/A", false},
        {"authority_reference", FlatValueType::string, "N/A", true},
        {"identity_role", FlatValueType::string, "N/A", false},
        {"rule", FlatValueType::string, "N/A", false},
        {"source_field", FlatValueType::string, "N/A", true},
        {"missing_policy", FlatValueType::string, "N/A", false},
        {"description", FlatValueType::string, "N/A", false},
    };
}

inline SerializedComponent serialize_fields_component(
    const IssuanceContext &issuance, BundleKind kind,
    const ObservationIdentity &observation, std::vector<FieldRule> fields,
    std::string role = "fields") {
    if (role != "fields") {
        throw ContractError("canonical APT v2 fields role is not closed");
    }
    std::sort(fields.begin(), fields.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.name < rhs.name;
              });
    FlatComponent document;
    document.role = std::move(role);
    document.schema = std::string(field_table_schema_v2);
    document.kind = kind;
    document.issuance = issuance;
    document.observation = observation;
    document.columns = field_columns();
    for (const auto &field : fields) {
        validate(field);
        document.rows.push_back(
            {field.field_uid, field.name,
             std::string(v1::value_type_token(field.datatype)), field.unit,
             field.nullable, field.authority,
             field.authority_reference
                 ? FlatValue{*field.authority_reference}
                 : FlatValue{NullValue{}},
             field.identity_role,
             std::string(field_operation_token(field.operation)),
             field.source_field ? FlatValue{*field.source_field}
                                : FlatValue{NullValue{}},
             field.missing_policy, field.description});
    }
    return serialize_component(std::move(document));
}

inline std::vector<FieldRule> parse_fields_component(
    const ParsedFlatComponent &parsed) {
    if (parsed.document.schema != field_table_schema_v2 ||
        parsed.document.role != "fields") {
        throw ContractError("canonical APT v2 fields component schema is invalid");
    }
    require_exact_columns(parsed.document, field_columns());
    require_exact_metadata_keys(metadata_map(parsed.document), {}, "fields");
    std::vector<FieldRule> result;
    std::optional<std::string> prior_name;
    std::set<std::int64_t> field_uids;
    for (const auto &row : parsed.document.rows) {
        FieldRule field;
        field.field_uid = std::get<std::int64_t>(row[0]);
        field.name = std::get<std::string>(row[1]);
        field.datatype = v1::parse_value_type_token(std::get<std::string>(row[2]));
        field.unit = std::get<std::string>(row[3]);
        field.nullable = std::get<bool>(row[4]);
        field.authority = std::get<std::string>(row[5]);
        if (!std::holds_alternative<NullValue>(row[6])) {
            field.authority_reference = std::get<std::string>(row[6]);
        }
        field.identity_role = std::get<std::string>(row[7]);
        field.operation = parse_field_operation(std::get<std::string>(row[8]));
        if (!std::holds_alternative<NullValue>(row[9])) {
            field.source_field = std::get<std::string>(row[9]);
        }
        field.missing_policy = std::get<std::string>(row[10]);
        field.description = std::get<std::string>(row[11]);
        validate(field);
        if (!field_uids.insert(field.field_uid).second ||
            (prior_name && field.name <= *prior_name)) {
            throw ContractError("canonical APT v2 field rows are not name ordered");
        }
        prior_name = field.name;
        result.push_back(std::move(field));
    }
    return result;
}

inline std::vector<FlatColumn> source_columns() {
    return {
        {"source_uid", FlatValueType::int64, "N/A", false},
        {"role", FlatValueType::string, "N/A", false},
        {"content_sha256", FlatValueType::string, "N/A", false},
        {"byte_count", FlatValueType::uint64, "byte", false},
        {"obsnum", FlatValueType::int64, "N/A", false},
        {"subobsnum", FlatValueType::int64, "N/A", false},
        {"scannum", FlatValueType::int64, "N/A", false},
        {"nw", FlatValueType::int64, "N/A", false},
        {"interface", FlatValueType::string, "N/A", false},
        {"channel_count", FlatValueType::uint64, "channel", false},
    };
}

inline SerializedComponent serialize_sources_component(
    const IssuanceContext &issuance, BundleKind kind,
    const ObservationIdentity &observation, std::vector<SourceRecord> sources,
    std::string role = "sources") {
    if (role != "sources") {
        throw ContractError("canonical APT v2 sources role is not closed");
    }
    std::sort(sources.begin(), sources.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.source_uid < rhs.source_uid;
              });
    FlatComponent document;
    document.role = std::move(role);
    document.schema = std::string(source_table_schema_v2);
    document.kind = kind;
    document.issuance = issuance;
    document.observation = observation;
    document.columns = source_columns();
    for (const auto &source : sources) {
        validate(source);
        document.rows.push_back(
            {source.source_uid, std::string(source_role_token(source.role)),
             source.content_sha256, UInt64Value{source.byte_count},
             source.header_observation.observation,
             source.header_observation.subobservation,
             source.header_observation.scan, source.network,
             source.interface_name,
             UInt64Value{static_cast<std::uint64_t>(source.channel_count)}});
    }
    return serialize_component(std::move(document));
}

inline std::vector<SourceRecord> parse_sources_component(
    const ParsedFlatComponent &parsed) {
    if (parsed.document.schema != source_table_schema_v2 ||
        parsed.document.role != "sources") {
        throw ContractError("canonical APT v2 sources component schema is invalid");
    }
    require_exact_columns(parsed.document, source_columns());
    require_exact_metadata_keys(metadata_map(parsed.document), {}, "sources");
    std::vector<SourceRecord> result;
    std::optional<std::int64_t> prior_uid;
    for (const auto &row : parsed.document.rows) {
        SourceRecord source;
        source.source_uid = std::get<std::int64_t>(row[0]);
        source.role = parse_source_role(std::get<std::string>(row[1]));
        source.content_sha256 = std::get<std::string>(row[2]);
        source.byte_count = std::get<UInt64Value>(row[3]).value;
        source.header_observation = {std::get<std::int64_t>(row[4]),
                                     std::get<std::int64_t>(row[5]),
                                     std::get<std::int64_t>(row[6])};
        source.network = std::get<std::int64_t>(row[7]);
        source.interface_name = std::get<std::string>(row[8]);
        const auto channel_count = std::get<UInt64Value>(row[9]).value;
        if (channel_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw ContractError("canonical APT v2 channel count exceeds int64");
        }
        source.channel_count = static_cast<std::int64_t>(channel_count);
        validate(source);
        if (prior_uid && source.source_uid <= *prior_uid) {
            throw ContractError("canonical APT v2 sources are not UID ordered");
        }
        prior_uid = source.source_uid;
        result.push_back(std::move(source));
    }
    return result;
}

inline FlatValueType flat_type(ValueType type) {
    switch (type) {
    case ValueType::int64: return FlatValueType::int64;
    case ValueType::float64: return FlatValueType::float64;
    case ValueType::boolean: return FlatValueType::boolean;
    case ValueType::string: return FlatValueType::string;
    }
    throw ContractError("canonical APT v2 field type is unsupported");
}

inline FlatValue flat_value(const Value &value) {
    return std::visit(
        [](const auto &typed) -> FlatValue { return typed; }, value);
}

inline Value apt_value(const FlatValue &value) {
    return std::visit(
        [](const auto &typed) -> Value {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, UInt64Value>) {
                throw ContractError(
                    "canonical APT v2 scientific cell cannot be uint64");
            } else {
                return typed;
            }
        }, value);
}

inline bool structural_name(std::string_view name) {
    return name == "uid" || name == "tone_freq" || name == "array" ||
        name == "nw" || name == "kids_tone";
}

inline std::vector<FlatColumn> apt_columns(
    const std::vector<FieldRule> &fields) {
    std::vector<FlatColumn> result{
        {"uid", FlatValueType::int64, "N/A", false},
        {"tone_freq", FlatValueType::float64, "Hz", false},
        {"array", FlatValueType::int64, "N/A", false},
        {"nw", FlatValueType::int64, "N/A", false},
        {"kids_tone", FlatValueType::int64, "N/A", false},
    };
    std::vector<FieldRule> dynamic;
    for (const auto &field : fields) {
        if (!structural_name(field.name)) dynamic.push_back(field);
    }
    std::sort(dynamic.begin(), dynamic.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.name < rhs.name;
              });
    for (const auto &field : dynamic) {
        result.push_back(
            {field.name, flat_type(field.datatype), field.unit, field.nullable});
    }
    return result;
}

inline SerializedComponent serialize_apt_component(
    AptTable table, std::string role = "apt") {
    if (role != "apt") {
        throw ContractError("canonical APT v2 APT role is not closed");
    }
    validate(table);
    std::sort(table.rows.begin(), table.rows.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.presentation_rank < rhs.presentation_rank;
              });
    FlatComponent document;
    document.role = std::move(role);
    document.schema = table.kind == BundleKind::baseline
        ? std::string(baseline_apt_schema_v2)
        : std::string(matched_apt_schema_v2);
    document.kind = table.kind;
    document.issuance = table.issuance;
    document.observation = table.observation;
    document.columns = apt_columns(table.field_rules);
    for (const auto &row : table.rows) {
        std::vector<FlatValue> values{
            row.uid, row.tone_frequency_hz, row.array, row.network, row.channel};
        for (std::size_t index = 5; index < document.columns.size(); ++index) {
            values.push_back(flat_value(
                row.fields.at(document.columns[index].name)));
        }
        document.rows.push_back(std::move(values));
    }
    return serialize_component(std::move(document));
}

inline AptTable parse_apt_component(const ParsedFlatComponent &parsed,
                                    std::vector<FieldRule> fields) {
    const auto expected_schema = parsed.document.kind == BundleKind::baseline
        ? baseline_apt_schema_v2
        : matched_apt_schema_v2;
    if (parsed.document.schema != expected_schema ||
        parsed.document.role != "apt") {
        throw ContractError("canonical APT v2 APT component identity is invalid");
    }
    require_exact_columns(parsed.document, apt_columns(fields));
    require_exact_metadata_keys(metadata_map(parsed.document), {}, "apt");
    AptTable table;
    table.kind = parsed.document.kind;
    table.issuance = parsed.document.issuance;
    table.observation = parsed.document.observation;
    table.field_rules = std::move(fields);
    std::set<std::int64_t> uids;
    for (std::size_t row_index = 0;
         row_index < parsed.document.rows.size(); ++row_index) {
        const auto &values = parsed.document.rows[row_index];
        AptRow row;
        row.uid = std::get<std::int64_t>(values[0]);
        row.tone_frequency_hz = std::get<double>(values[1]);
        row.array = std::get<std::int64_t>(values[2]);
        row.network = std::get<std::int64_t>(values[3]);
        row.channel = std::get<std::int64_t>(values[4]);
        row.presentation_rank = row_index;
        if (!uids.insert(row.uid).second) {
            throw ContractError("canonical APT v2 APT UID is duplicate");
        }
        for (std::size_t index = 5; index < values.size(); ++index) {
            row.fields.emplace(parsed.document.columns[index].name,
                               apt_value(values[index]));
        }
        table.rows.push_back(std::move(row));
    }
    validate(table);
    return table;
}

inline std::vector<FlatColumn> relation_columns() {
    return {
        {"relation_uid", FlatValueType::int64, "N/A", false},
        {"output_uid", FlatValueType::int64, "N/A", false},
        {"target_occurrence", FlatValueType::string, "N/A", false},
        {"target_uid", FlatValueType::int64, "N/A", false},
        {"target_input_uid", FlatValueType::int64, "N/A", false},
        {"raw_source_uid", FlatValueType::int64, "N/A", false},
        {"kmp_source_uid", FlatValueType::int64, "N/A", false},
        {"kmp_row_index", FlatValueType::int64, "N/A", false},
        {"source_rank", FlatValueType::int64, "N/A", false},
        {"application_rank", FlatValueType::int64, "N/A", false},
        {"presentation_rank", FlatValueType::int64, "N/A", false},
        {"disposition", FlatValueType::string, "N/A", false},
        {"seed_occurrence", FlatValueType::string, "N/A", true},
        {"seed_uid", FlatValueType::int64, "N/A", true},
        {"pair_uid", FlatValueType::int64, "N/A", true},
        {"separation_hz", FlatValueType::float64, "Hz", true},
        {"is_good_match", FlatValueType::boolean, "N/A", true},
        {"network_evidence_uid", FlatValueType::int64, "N/A", false},
        {"reason", FlatValueType::string, "N/A", false},
    };
}

inline void append_identity_metadata(
    std::vector<std::pair<std::string, std::string>> &metadata,
    std::string_view prefix, const ComponentIdentity &identity) {
    validate(identity);
    metadata.emplace_back(std::string(prefix) + ".schema", identity.schema);
    metadata.emplace_back(std::string(prefix) + ".occurrence",
                          identity.occurrence);
    metadata.emplace_back(std::string(prefix) + ".semantic_sha256",
                          identity.semantic_sha256);
    metadata.emplace_back(std::string(prefix) + ".envelope_sha256",
                          identity.envelope_sha256);
}

inline ComponentIdentity identity_from_metadata(
    const std::map<std::string, std::string> &metadata,
    std::string_view prefix) {
    ComponentIdentity result{
        metadata_required(metadata, std::string(prefix) + ".schema"),
        metadata_required(metadata, std::string(prefix) + ".occurrence"),
        metadata_required(metadata, std::string(prefix) + ".semantic_sha256"),
        metadata_required(metadata, std::string(prefix) + ".envelope_sha256")};
    validate(result);
    return result;
}

inline SerializedComponent serialize_relation_component(RelationTable table) {
    validate(table);
    std::sort(table.rows.begin(), table.rows.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.relation_uid < rhs.relation_uid;
              });
    std::sort(table.network_evidence.begin(), table.network_evidence.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.network < rhs.network;
              });
    FlatComponent document;
    document.role = "relation";
    document.schema = std::string(relation_table_schema_v2);
    document.kind = BundleKind::matched;
    document.issuance = table.issuance;
    document.observation = table.observation;
    document.columns = relation_columns();
    append_identity_metadata(document.metadata, "baseline_parent",
                             table.baseline_parent);
    append_identity_metadata(document.metadata, "target_parent",
                             table.target_parent);
    document.metadata.emplace_back("target_issuance.occurrence",
                                   table.target_issuance.occurrence);
    document.metadata.emplace_back("target_issuance.event_reference",
                                   table.target_issuance.event_reference);
    document.metadata.emplace_back("target_issuance.producer",
                                   table.target_issuance.producer);
    document.metadata.emplace_back("target_issuance.software_revision",
                                   table.target_issuance.software_revision);
    document.metadata.emplace_back(
        "target_issuance.configuration_reference",
        table.target_issuance.configuration_reference);
    document.metadata.emplace_back("target_issuance.event_time_utc",
                                   table.target_issuance.event_time_utc);
    document.metadata.emplace_back("matcher.occurrence",
                                   table.matcher.matcher_run_occurrence);
    document.metadata.emplace_back("matcher.implementation_sha256",
                                   table.matcher.implementation_sha256);
    document.metadata.emplace_back("matcher.configuration_sha256",
                                   table.matcher.configuration_sha256);
    document.metadata.emplace_back("matcher.method", table.matcher.method);
    document.metadata.emplace_back("matcher.backend", table.matcher.backend);
    document.metadata.emplace_back("network_evidence.count",
                                   std::to_string(table.network_evidence.size()));
    for (std::size_t index = 0; index < table.network_evidence.size(); ++index) {
        const auto prefix = "network_evidence." + std::to_string(index);
        const auto &evidence = table.network_evidence[index];
        document.metadata.emplace_back(prefix + ".uid",
                                       std::to_string(evidence.evidence_uid));
        document.metadata.emplace_back(prefix + ".network",
                                       std::to_string(evidence.network));
        document.metadata.emplace_back(
            prefix + ".status",
            std::string(network_evidence_status_token(evidence.status)));
        document.metadata.emplace_back(prefix + ".frequency_shift_bits",
            evidence.frequency_shift_hz
                ? canonical_binary64(*evidence.frequency_shift_hz)
                : "null");
        document.metadata.emplace_back(prefix + ".gate_bits",
            evidence.gate_hz ? canonical_binary64(*evidence.gate_hz)
                             : "null");
        document.metadata.emplace_back(prefix + ".quality_factor_bits",
            evidence.quality_factor
                ? canonical_binary64(*evidence.quality_factor)
                : "null");
    }
    for (const auto &row : table.rows) {
        document.rows.push_back({
            row.relation_uid, row.output_uid, row.target.artifact.occurrence,
            row.target.local_uid, row.target_input_uid,
            row.target_raw_source_uid, row.target_kmp_source_uid,
            row.target_kmp_row_index,
            static_cast<std::int64_t>(row.source_rank),
            static_cast<std::int64_t>(row.application_rank),
            static_cast<std::int64_t>(row.presentation_rank),
            std::string(relation_disposition_token(row.disposition)),
            row.selected_seed ? FlatValue{row.selected_seed->artifact.occurrence}
                              : FlatValue{NullValue{}},
            row.selected_seed ? FlatValue{row.selected_seed->local_uid}
                              : FlatValue{NullValue{}},
            row.selected_pair_uid ? FlatValue{*row.selected_pair_uid}
                                  : FlatValue{NullValue{}},
            row.separation_hz ? FlatValue{*row.separation_hz}
                              : FlatValue{NullValue{}},
            row.is_good_match ? FlatValue{*row.is_good_match}
                              : FlatValue{NullValue{}},
            row.network_evidence_uid, row.reason});
    }
    return serialize_component(std::move(document));
}

inline RelationTable parse_relation_component(
    const ParsedFlatComponent &parsed,
    const ObservationIdentity &observation) {
    if (parsed.document.schema != relation_table_schema_v2 ||
        parsed.document.role != "relation" ||
        parsed.document.kind != BundleKind::matched) {
        throw ContractError("canonical APT v2 relation component identity is invalid");
    }
    require_exact_columns(parsed.document, relation_columns());
    const auto metadata = metadata_map(parsed.document);
    RelationTable result;
    result.issuance = parsed.document.issuance;
    result.observation = parsed.document.observation;
    result.baseline_parent = identity_from_metadata(metadata, "baseline_parent");
    result.target_parent = identity_from_metadata(metadata, "target_parent");
    result.target_issuance = {
        metadata_required(metadata, "target_issuance.occurrence"),
        metadata_required(metadata, "target_issuance.event_reference"),
        metadata_required(metadata, "target_issuance.producer"),
        metadata_required(metadata, "target_issuance.software_revision"),
        metadata_required(metadata,
                          "target_issuance.configuration_reference"),
        metadata_required(metadata, "target_issuance.event_time_utc")};
    result.matcher = {
        metadata_required(metadata, "matcher.occurrence"),
        metadata_required(metadata, "matcher.implementation_sha256"),
        metadata_required(metadata, "matcher.configuration_sha256"),
        metadata_required(metadata, "matcher.method"),
        metadata_required(metadata, "matcher.backend")};
    const auto evidence_count = exact_uint64(
        metadata_required(metadata, "network_evidence.count"),
        "network evidence count");
    std::set<std::string> expected_metadata{
        "baseline_parent.schema", "baseline_parent.occurrence",
        "baseline_parent.semantic_sha256", "baseline_parent.envelope_sha256",
        "target_parent.schema", "target_parent.occurrence",
        "target_parent.semantic_sha256", "target_parent.envelope_sha256",
        "target_issuance.occurrence", "target_issuance.event_reference",
        "target_issuance.producer", "target_issuance.software_revision",
        "target_issuance.configuration_reference",
        "target_issuance.event_time_utc",
        "matcher.occurrence", "matcher.implementation_sha256",
        "matcher.configuration_sha256", "matcher.method", "matcher.backend",
        "network_evidence.count"};
    for (std::uint64_t index = 0; index < evidence_count; ++index) {
        const auto prefix = "network_evidence." + std::to_string(index);
        for (const auto suffix : {".uid", ".network", ".status",
                                  ".frequency_shift_bits", ".gate_bits",
                                  ".quality_factor_bits"}) {
            expected_metadata.insert(prefix + suffix);
        }
        const auto optional_binary64 = [&](const std::string &key)
            -> std::optional<double> {
            const auto token = metadata_required(metadata, key);
            if (token == "null") return std::nullopt;
            return binary64_from_payload(token);
        };
        result.network_evidence.push_back({
            exact_int64(metadata_required(metadata, prefix + ".uid"),
                        "network evidence UID"),
            exact_int64(metadata_required(metadata, prefix + ".network"),
                        "network evidence network"),
            parse_network_evidence_status(
                metadata_required(metadata, prefix + ".status")),
            optional_binary64(prefix + ".frequency_shift_bits"),
            optional_binary64(prefix + ".gate_bits"),
            optional_binary64(prefix + ".quality_factor_bits")});
    }
    require_exact_metadata_keys(metadata, expected_metadata, "relation");
    for (const auto &values : parsed.document.rows) {
        RelationRecord row;
        row.relation_uid = std::get<std::int64_t>(values[0]);
        row.output_uid = std::get<std::int64_t>(values[1]);
        row.target = {result.target_parent, std::get<std::int64_t>(values[3])};
        if (std::get<std::string>(values[2]) !=
            result.target_parent.occurrence) {
            throw ContractError("canonical APT v2 relation target occurrence is foreign");
        }
        row.target_input_uid = std::get<std::int64_t>(values[4]);
        row.target_raw_source_uid = std::get<std::int64_t>(values[5]);
        row.target_kmp_source_uid = std::get<std::int64_t>(values[6]);
        row.target_kmp_row_index = std::get<std::int64_t>(values[7]);
        for (std::size_t index : {8U, 9U, 10U}) {
            if (std::get<std::int64_t>(values[index]) < 0) {
                throw ContractError("canonical APT v2 relation rank is negative");
            }
        }
        row.source_rank =
            static_cast<std::uint64_t>(std::get<std::int64_t>(values[8]));
        row.application_rank =
            static_cast<std::uint64_t>(std::get<std::int64_t>(values[9]));
        row.presentation_rank =
            static_cast<std::uint64_t>(std::get<std::int64_t>(values[10]));
        row.disposition =
            parse_relation_disposition(std::get<std::string>(values[11]));
        if (!std::holds_alternative<NullValue>(values[12])) {
            ScopedRowReference seed{
                result.baseline_parent, std::get<std::int64_t>(values[13])};
            if (std::get<std::string>(values[12]) !=
                result.baseline_parent.occurrence) {
                throw ContractError("canonical APT v2 relation seed occurrence is foreign");
            }
            row.selected_seed = std::move(seed);
            row.selected_pair_uid = std::get<std::int64_t>(values[14]);
            row.separation_hz = std::get<double>(values[15]);
            row.is_good_match = std::get<bool>(values[16]);
        } else if (!std::holds_alternative<NullValue>(values[13]) ||
                   !std::holds_alternative<NullValue>(values[14]) ||
                   !std::holds_alternative<NullValue>(values[15]) ||
                   !std::holds_alternative<NullValue>(values[16])) {
            throw ContractError("canonical APT v2 relation nullable selection is partial");
        }
        row.network_evidence_uid = std::get<std::int64_t>(values[17]);
        row.reason = std::get<std::string>(values[18]);
        result.rows.push_back(std::move(row));
    }
    if (parsed.document.observation != observation) {
        throw ContractError("canonical APT v2 relation observation is foreign");
    }
    validate(result);
    return result;
}

inline std::vector<FlatColumn> exception_columns() {
    return {
        {"exception_uid", FlatValueType::int64, "N/A", false},
        {"kind", FlatValueType::string, "N/A", false},
        {"target_uid", FlatValueType::int64, "N/A", true},
        {"field_name", FlatValueType::string, "N/A", true},
        {"candidate_seed_occurrence", FlatValueType::string, "N/A", true},
        {"candidate_seed_uid", FlatValueType::int64, "N/A", true},
        {"separation_hz", FlatValueType::float64, "Hz", true},
        {"is_good_match", FlatValueType::boolean, "N/A", true},
        {"operation", FlatValueType::string, "N/A", true},
        {"before_datatype", FlatValueType::string, "N/A", true},
        {"before_value", FlatValueType::string, "N/A", true},
        {"after_datatype", FlatValueType::string, "N/A", true},
        {"after_value", FlatValueType::string, "N/A", true},
        {"authority_reference", FlatValueType::string, "N/A", true},
        {"reason", FlatValueType::string, "N/A", false},
    };
}

inline std::string exact_value_token(const Value &value, ValueType type) {
    if (std::holds_alternative<NullValue>(value)) return "null";
    if (!v1::detail::value_matches_type(value, type)) {
        throw ContractError("canonical APT v2 exception value is untyped");
    }
    if (const auto integer = std::get_if<std::int64_t>(&value)) {
        return std::to_string(*integer);
    }
    if (const auto number = std::get_if<double>(&value)) {
        return canonical_binary64(*number);
    }
    if (const auto boolean = std::get_if<bool>(&value)) {
        return *boolean ? "true" : "false";
    }
    return std::get<std::string>(value);
}

inline Value parse_exact_value_token(std::string_view token, ValueType type) {
    if (token == "null") return NullValue{};
    switch (type) {
    case ValueType::int64: return exact_int64(token, "exception int64 value");
    case ValueType::float64: return binary64_from_payload(token);
    case ValueType::boolean:
        if (token == "true") return true;
        if (token == "false") return false;
        throw ContractError("canonical APT v2 exception bool token is invalid");
    case ValueType::string:
        if (!v1::detail::canonical_text(token, true)) {
            throw ContractError("canonical APT v2 exception string is invalid");
        }
        return std::string(token);
    }
    throw ContractError("canonical APT v2 exception datatype is invalid");
}

inline SerializedComponent serialize_exceptions_component(
    const IssuanceContext &issuance,
    const ObservationIdentity &observation,
    const ComponentIdentity &baseline_parent,
    std::vector<ExceptionRecord> exceptions) {
    validate(baseline_parent);
    std::sort(exceptions.begin(), exceptions.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.exception_uid < rhs.exception_uid;
              });
    FlatComponent document;
    document.role = "exceptions";
    document.schema = std::string(exception_table_schema_v2);
    document.kind = BundleKind::matched;
    document.issuance = issuance;
    document.observation = observation;
    document.columns = exception_columns();
    append_identity_metadata(document.metadata, "baseline_parent",
                             baseline_parent);
    for (const auto &exception : exceptions) {
        validate(exception);
        const auto field_type = exception.value_type
            ? FlatValue{std::string(v1::value_type_token(*exception.value_type))}
            : FlatValue{NullValue{}};
        const auto before = exception.before && exception.value_type
            ? FlatValue{exact_value_token(*exception.before,
                                          *exception.value_type)}
            : FlatValue{NullValue{}};
        const auto after = exception.after && exception.value_type
            ? FlatValue{exact_value_token(*exception.after,
                                          *exception.value_type)}
            : FlatValue{NullValue{}};
        document.rows.push_back({
            exception.exception_uid,
            std::string(exception_kind_token(exception.kind)),
            exception.target_uid ? FlatValue{*exception.target_uid}
                                 : FlatValue{NullValue{}},
            exception.field_name ? FlatValue{*exception.field_name}
                                 : FlatValue{NullValue{}},
            exception.seed ? FlatValue{exception.seed->artifact.occurrence}
                           : FlatValue{NullValue{}},
            exception.seed ? FlatValue{exception.seed->local_uid}
                           : FlatValue{NullValue{}},
            exception.separation_hz ? FlatValue{*exception.separation_hz}
                                    : FlatValue{NullValue{}},
            exception.is_good_match ? FlatValue{*exception.is_good_match}
                                    : FlatValue{NullValue{}},
            exception.operation
                ? FlatValue{std::string(
                      field_operation_token(*exception.operation))}
                : FlatValue{NullValue{}},
            field_type, before, field_type, after,
            exception.authority_reference
                ? FlatValue{*exception.authority_reference}
                : FlatValue{NullValue{}},
            exception.reason});
    }
    return serialize_component(std::move(document));
}

inline std::vector<ExceptionRecord> parse_exceptions_component(
    const ParsedFlatComponent &parsed,
    const ComponentIdentity &baseline_parent,
    const ObservationIdentity &observation) {
    if (parsed.document.schema != exception_table_schema_v2 ||
        parsed.document.role != "exceptions" ||
        parsed.document.kind != BundleKind::matched ||
        parsed.document.observation != observation) {
        throw ContractError("canonical APT v2 exception component identity is invalid");
    }
    require_exact_columns(parsed.document, exception_columns());
    const auto metadata = metadata_map(parsed.document);
    require_exact_metadata_keys(
        metadata,
        {"baseline_parent.schema", "baseline_parent.occurrence",
         "baseline_parent.semantic_sha256",
         "baseline_parent.envelope_sha256"},
        "exceptions");
    if (identity_from_metadata(metadata, "baseline_parent") != baseline_parent) {
        throw ContractError("canonical APT v2 exception baseline metadata is invalid");
    }
    std::vector<ExceptionRecord> result;
    std::optional<std::int64_t> prior_uid;
    for (const auto &values : parsed.document.rows) {
        ExceptionRecord exception;
        exception.exception_uid = std::get<std::int64_t>(values[0]);
        if (prior_uid && exception.exception_uid <= *prior_uid) {
            throw ContractError("canonical APT v2 exceptions are not UID ordered");
        }
        prior_uid = exception.exception_uid;
        exception.kind = parse_exception_kind(std::get<std::string>(values[1]));
        if (!std::holds_alternative<NullValue>(values[2])) {
            exception.target_uid = std::get<std::int64_t>(values[2]);
        }
        if (!std::holds_alternative<NullValue>(values[3])) {
            exception.field_name = std::get<std::string>(values[3]);
        }
        if (!std::holds_alternative<NullValue>(values[4])) {
            if (std::holds_alternative<NullValue>(values[5]) ||
                std::get<std::string>(values[4]) !=
                    baseline_parent.occurrence) {
                throw ContractError("canonical APT v2 exception seed is partial/foreign");
            }
            exception.seed = ScopedRowReference{
                baseline_parent, std::get<std::int64_t>(values[5])};
        } else if (!std::holds_alternative<NullValue>(values[5])) {
            throw ContractError("canonical APT v2 exception seed is partial");
        }
        if (!std::holds_alternative<NullValue>(values[6])) {
            exception.separation_hz = std::get<double>(values[6]);
        }
        if (!std::holds_alternative<NullValue>(values[7])) {
            exception.is_good_match = std::get<bool>(values[7]);
        }
        if (!std::holds_alternative<NullValue>(values[8])) {
            exception.operation = parse_field_operation(
                std::get<std::string>(values[8]));
        }
        if (!std::holds_alternative<NullValue>(values[9])) {
            const auto type = v1::parse_value_type_token(
                std::get<std::string>(values[9]));
            if (std::holds_alternative<NullValue>(values[10]) ||
                std::holds_alternative<NullValue>(values[11]) ||
                std::holds_alternative<NullValue>(values[12]) ||
                std::get<std::string>(values[9]) !=
                    std::get<std::string>(values[11])) {
                throw ContractError("canonical APT v2 exception typed values are partial");
            }
            exception.value_type = type;
            exception.before = parse_exact_value_token(
                std::get<std::string>(values[10]), type);
            exception.after = parse_exact_value_token(
                std::get<std::string>(values[12]), type);
        }
        if (!std::holds_alternative<NullValue>(values[13])) {
            exception.authority_reference = std::get<std::string>(values[13]);
        }
        exception.reason = std::get<std::string>(values[14]);
        validate(exception);
        result.push_back(std::move(exception));
    }
    return result;
}

inline std::vector<FlatColumn> manifest_columns() {
    return {
        {"role", FlatValueType::string, "N/A", false},
        {"relative_path", FlatValueType::string, "N/A", false},
        {"schema", FlatValueType::string, "N/A", false},
        {"semantic_sha256", FlatValueType::string, "N/A", false},
        {"envelope_sha256", FlatValueType::string, "N/A", false},
        {"transport_sha256", FlatValueType::string, "N/A", false},
        {"byte_count", FlatValueType::uint64, "byte", false},
        {"row_count", FlatValueType::uint64, "row", false},
    };
}

inline SerializedComponent serialize_manifest_component(
    BundleManifest manifest) {
    validate(manifest);
    std::sort(manifest.components.begin(), manifest.components.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.role < rhs.role;
              });
    FlatComponent document;
    document.role = "manifest";
    document.schema = std::string(manifest_schema_v2);
    document.kind = manifest.kind;
    document.issuance = manifest.issuance;
    document.observation = manifest.observation;
    document.columns = manifest_columns();
    document.metadata.emplace_back(
        "bundle_schema", manifest.kind == BundleKind::baseline
            ? std::string(baseline_bundle_schema_v2)
            : std::string(matched_bundle_schema_v2));
    document.metadata.emplace_back("contract_authority",
                                   std::string(contract_authority_v2));
    document.metadata.emplace_back("profile", manifest.profile);
    document.metadata.emplace_back("issuance_class", manifest.issuance_class);
    if (manifest.baseline_parent) {
        append_identity_metadata(document.metadata, "baseline_parent",
                                 *manifest.baseline_parent);
    }
    if (manifest.target_parent) {
        append_identity_metadata(document.metadata, "target_parent",
                                 *manifest.target_parent);
    }
    for (const auto &[key, value] :
         std::vector<std::pair<std::string, std::string>>{
             {"target_manifest_sha256", manifest.target_manifest_sha256},
             {"relation_sha256", manifest.relation_sha256},
             {"field_rules_sha256", manifest.field_rules_sha256},
             {"exceptions_sha256", manifest.exceptions_sha256}}) {
        if (!value.empty()) document.metadata.emplace_back(key, value);
    }
    for (const auto &component : manifest.components) {
        document.rows.push_back(
            {component.role, component.relative_path, component.schema,
             component.semantic_sha256, component.envelope_sha256,
             component.transport_sha256, UInt64Value{component.byte_count},
             UInt64Value{component.row_count}});
    }
    return serialize_component(std::move(document));
}

inline BundleManifest parse_manifest_component(
    const ParsedFlatComponent &parsed) {
    if (parsed.document.schema != manifest_schema_v2 ||
        parsed.document.role != "manifest") {
        throw ContractError("canonical APT v2 root manifest identity is invalid");
    }
    require_exact_columns(parsed.document, manifest_columns());
    BundleManifest result;
    result.schema = std::string(manifest_schema_v2);
    result.kind = parsed.document.kind;
    result.issuance = parsed.document.issuance;
    result.observation = parsed.document.observation;
    const auto metadata = metadata_map(parsed.document);
    const auto expected_bundle_schema = result.kind == BundleKind::baseline
        ? baseline_bundle_schema_v2
        : matched_bundle_schema_v2;
    if (metadata_required(metadata, "bundle_schema") !=
            expected_bundle_schema ||
        metadata_required(metadata, "contract_authority") !=
            contract_authority_v2) {
        throw ContractError("canonical APT v2 root contract identity is invalid");
    }
    result.profile = metadata_required(metadata, "profile");
    result.issuance_class = metadata_required(metadata, "issuance_class");
    std::set<std::string> expected_metadata{
        "bundle_schema", "contract_authority", "profile", "issuance_class"};
    if (result.kind == BundleKind::matched) {
        result.baseline_parent =
            identity_from_metadata(metadata, "baseline_parent");
        result.target_parent = identity_from_metadata(metadata, "target_parent");
        result.target_manifest_sha256 =
            metadata_required(metadata, "target_manifest_sha256");
        result.relation_sha256 = metadata_required(metadata, "relation_sha256");
        result.field_rules_sha256 =
            metadata_required(metadata, "field_rules_sha256");
        result.exceptions_sha256 =
            metadata_required(metadata, "exceptions_sha256");
        for (const auto prefix : {"baseline_parent", "target_parent"}) {
            for (const auto suffix : {".schema", ".occurrence",
                                      ".semantic_sha256", ".envelope_sha256"}) {
                expected_metadata.insert(std::string(prefix) + suffix);
            }
        }
        expected_metadata.insert("target_manifest_sha256");
        expected_metadata.insert("relation_sha256");
        expected_metadata.insert("field_rules_sha256");
        expected_metadata.insert("exceptions_sha256");
    }
    require_exact_metadata_keys(metadata, expected_metadata, "manifest");
    std::optional<std::string> prior_role;
    for (const auto &values : parsed.document.rows) {
        ComponentDescriptor component{
            std::get<std::string>(values[0]),
            std::get<std::string>(values[1]),
            std::get<std::string>(values[2]),
            std::get<std::string>(values[3]),
            std::get<std::string>(values[4]),
            std::get<std::string>(values[5]),
            std::get<UInt64Value>(values[6]).value,
            std::get<UInt64Value>(values[7]).value};
        if (prior_role && component.role <= *prior_role) {
            throw ContractError("canonical APT v2 manifest is not role ordered");
        }
        prior_role = component.role;
        result.components.push_back(std::move(component));
    }
    validate(result);
    return result;
}

}  // namespace detail

template <class Document>
struct VerifiedComponent {
    Document document;
    ComponentDigests digests;
    std::string role;
    std::string schema;
    BundleKind kind = BundleKind::baseline;
    IssuanceContext issuance;
    ObservationIdentity observation;
    std::uint64_t row_count = 0;
};

template <class Document>
inline VerifiedComponent<Document> make_verified_component(
    Document document, detail::ParsedFlatComponent parsed) {
    const auto row_count =
        static_cast<std::uint64_t>(parsed.document.rows.size());
    return {std::move(document), std::move(parsed.digests),
            std::move(parsed.document.role), std::move(parsed.document.schema),
            parsed.document.kind, std::move(parsed.document.issuance),
            parsed.document.observation, row_count};
}

inline SerializedComponent serialize_fields_component(
    const IssuanceContext &issuance, BundleKind kind,
    const ObservationIdentity &observation, std::vector<FieldRule> fields,
    std::string role = "fields") {
    return detail::serialize_fields_component(
        issuance, kind, observation, std::move(fields), std::move(role));
}

inline VerifiedComponent<std::vector<FieldRule>> verify_fields_component(
    std::string_view bytes) {
    auto parsed = detail::parse_component(bytes);
    auto document = detail::parse_fields_component(parsed);
    return make_verified_component(std::move(document), std::move(parsed));
}

inline SerializedComponent serialize_sources_component(
    const IssuanceContext &issuance, BundleKind kind,
    const ObservationIdentity &observation, std::vector<SourceRecord> sources,
    std::string role = "sources") {
    return detail::serialize_sources_component(
        issuance, kind, observation, std::move(sources), std::move(role));
}

inline VerifiedComponent<std::vector<SourceRecord>> verify_sources_component(
    std::string_view bytes) {
    auto parsed = detail::parse_component(bytes);
    auto document = detail::parse_sources_component(parsed);
    return make_verified_component(std::move(document), std::move(parsed));
}

inline SerializedComponent serialize_apt_component(
    AptTable table, std::string role = "apt") {
    return detail::serialize_apt_component(std::move(table), std::move(role));
}

inline VerifiedComponent<AptTable> verify_apt_component(
    std::string_view bytes, std::vector<FieldRule> fields) {
    auto parsed = detail::parse_component(bytes);
    auto document = detail::parse_apt_component(parsed, std::move(fields));
    return make_verified_component(std::move(document), std::move(parsed));
}

inline SerializedComponent serialize_relation_component(RelationTable table) {
    return detail::serialize_relation_component(std::move(table));
}

inline VerifiedComponent<RelationTable> verify_relation_component(
    std::string_view bytes, const ObservationIdentity &observation) {
    auto parsed = detail::parse_component(bytes);
    auto document = detail::parse_relation_component(parsed, observation);
    return make_verified_component(std::move(document), std::move(parsed));
}

inline SerializedComponent serialize_exceptions_component(
    const IssuanceContext &issuance,
    const ObservationIdentity &observation,
    const ComponentIdentity &baseline_parent,
    std::vector<ExceptionRecord> exceptions) {
    return detail::serialize_exceptions_component(
        issuance, observation, baseline_parent, std::move(exceptions));
}

inline VerifiedComponent<std::vector<ExceptionRecord>>
verify_exceptions_component(std::string_view bytes,
                            const ComponentIdentity &baseline_parent,
                            const ObservationIdentity &observation) {
    auto parsed = detail::parse_component(bytes);
    auto document = detail::parse_exceptions_component(
        parsed, baseline_parent, observation);
    return make_verified_component(std::move(document), std::move(parsed));
}

inline SerializedComponent serialize_manifest_component(
    BundleManifest manifest) {
    return detail::serialize_manifest_component(std::move(manifest));
}

inline VerifiedComponent<BundleManifest> verify_manifest_component(
    std::string_view bytes) {
    auto parsed = detail::parse_component(bytes);
    auto document = detail::parse_manifest_component(parsed);
    return make_verified_component(std::move(document), std::move(parsed));
}

}  // namespace citlali::pipeline::canonical_apt_v2

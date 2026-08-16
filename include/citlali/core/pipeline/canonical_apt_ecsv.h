#pragma once

#include <citlali/core/pipeline/canonical_apt_v1.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace citlali::pipeline::canonical_apt_v1 {

struct SerializedEcsv {
    std::string bytes;
    Digests digests;
    ByteTransportHash transport;
};

struct ParsedEcsv {
    Document document;
    Digests declared_digests;
    ByteTransportHash computed_transport;
};

inline ByteTransportHash make_byte_transport_hash(
    std::string_view bytes, std::string_view envelope_sha256_value) {
    if (!is_sha256_reference(envelope_sha256_value)) {
        throw ContractError(
            "canonical APT byte transport requires a valid envelope SHA-256 reference");
    }
    return {std::string(byte_transport_scope_v1),
            std::string(envelope_sha256_value),
            "sha256:" + citlali::utils::sha256(bytes),
            static_cast<std::uint64_t>(bytes.size())};
}

inline void validate_byte_transport(std::string_view bytes,
                                    std::string_view envelope_sha256_value,
                                    const ByteTransportHash &transport) {
    const auto expected =
        make_byte_transport_hash(bytes, envelope_sha256_value);
    if (transport.scope != expected.scope ||
        transport.envelope_sha256 != expected.envelope_sha256 ||
        transport.sha256 != expected.sha256 ||
        transport.byte_count != expected.byte_count) {
        throw ContractError(
            "canonical APT byte transport SHA-256 or envelope binding mismatch");
    }
}

namespace ecsv_detail {

inline bool starts_with(std::string_view value, std::string_view prefix) {
    return value.starts_with(prefix);
}

inline std::string yaml_quote(std::string_view value) {
    if (!detail::canonical_text(value, true)) {
        throw ContractError(
            "canonical APT YAML value is not valid single-line UTF-8 text");
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string result{"\""};
    for (const unsigned char ch : value) {
        switch (ch) {
        case '\\':
            result += "\\\\";
            break;
        case '"':
            result += "\\\"";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (ch < 0x20) {
                result += "\\u00";
                result += hex[(ch >> 4) & 0x0f];
                result += hex[ch & 0x0f];
            } else {
                result.push_back(static_cast<char>(ch));
            }
        }
    }
    result += '"';
    return result;
}

inline unsigned parse_hex_digit(char ch) {
    if (ch >= '0' && ch <= '9') {
        return static_cast<unsigned>(ch - '0');
    }
    if (ch >= 'a' && ch <= 'f') {
        return 10U + static_cast<unsigned>(ch - 'a');
    }
    if (ch >= 'A' && ch <= 'F') {
        return 10U + static_cast<unsigned>(ch - 'A');
    }
    throw ContractError("invalid hexadecimal YAML escape in canonical APT");
}

inline std::string yaml_unquote(std::string_view value) {
    if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
        throw ContractError(
            "canonical APT ECSV metadata string is not double quoted");
    }
    std::string result;
    for (std::size_t index = 1; index + 1 < value.size(); ++index) {
        const char ch = value[index];
        if (ch != '\\') {
            result.push_back(ch);
            continue;
        }
        if (++index + 1 >= value.size()) {
            throw ContractError(
                "truncated YAML escape in canonical APT ECSV metadata");
        }
        switch (value[index]) {
        case '\\':
            result.push_back('\\');
            break;
        case '"':
            result.push_back('"');
            break;
        case 'n':
            result.push_back('\n');
            break;
        case 'r':
            result.push_back('\r');
            break;
        case 't':
            result.push_back('\t');
            break;
        case 'u': {
            if (index + 4 >= value.size() || value[index + 1] != '0' ||
                value[index + 2] != '0') {
                throw ContractError(
                    "canonical APT supports only byte-sized YAML unicode escapes");
            }
            const unsigned byte =
                (parse_hex_digit(value[index + 3]) << 4U) |
                parse_hex_digit(value[index + 4]);
            result.push_back(static_cast<char>(byte));
            index += 4;
            break;
        }
        default:
            throw ContractError(
                "unsupported YAML escape in canonical APT ECSV metadata");
        }
    }
    return result;
}

inline std::string csv_quote(std::string_view value) {
    if (!detail::canonical_text(value, true)) {
        throw ContractError(
            "canonical APT CSV value is not valid single-line UTF-8 text");
    }
    std::string result{"\""};
    for (const char ch : value) {
        if (ch == '"') {
            result += "\"\"";
        } else {
            result.push_back(ch);
        }
    }
    result += '"';
    return result;
}

struct CsvCell {
    std::string value;
    bool quoted = false;
};

inline std::vector<CsvCell> parse_csv_line(std::string_view line) {
    std::vector<CsvCell> cells;
    std::size_t index = 0;
    while (true) {
        CsvCell cell;
        if (index < line.size() && line[index] == '"') {
            cell.quoted = true;
            ++index;
            bool closed = false;
            while (index < line.size()) {
                if (line[index] != '"') {
                    cell.value.push_back(line[index++]);
                    continue;
                }
                if (index + 1 < line.size() && line[index + 1] == '"') {
                    cell.value.push_back('"');
                    index += 2;
                    continue;
                }
                ++index;
                closed = true;
                break;
            }
            if (!closed) {
                throw ContractError(
                    "unterminated quoted cell in canonical APT ECSV");
            }
            if (index < line.size() && line[index] != ',') {
                throw ContractError(
                    "characters follow quoted cell in canonical APT ECSV");
            }
        } else {
            while (index < line.size() && line[index] != ',') {
                if (line[index] == '"') {
                    throw ContractError(
                        "quote appears inside unquoted canonical APT ECSV cell");
                }
                cell.value.push_back(line[index++]);
            }
        }
        cells.push_back(std::move(cell));
        if (index == line.size()) {
            break;
        }
        ++index;
        if (index == line.size()) {
            cells.push_back(CsvCell{});
            break;
        }
    }
    return cells;
}

inline std::int64_t parse_int64(std::string_view value,
                                std::string_view label) {
    if (value.empty()) {
        throw ContractError("missing exact int64 canonical APT cell: " +
                            std::string(label));
    }
    std::int64_t result = 0;
    const auto parsed =
        std::from_chars(value.data(), value.data() + value.size(), result, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw ContractError("invalid exact int64 canonical APT cell: " +
                            std::string(label));
    }
    return result;
}

inline double parse_float64(std::string_view value,
                            std::string_view label) {
    if (value == "nan") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (value == "inf" || value == "+inf") {
        return std::numeric_limits<double>::infinity();
    }
    if (value == "-inf") {
        return -std::numeric_limits<double>::infinity();
    }
    if (value.empty()) {
        throw ContractError("missing float64 canonical APT cell: " +
                            std::string(label));
    }
    double result = 0.0;
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(), result,
        std::chars_format::general);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != value.data() + value.size()) {
        throw ContractError("invalid float64 canonical APT cell: " +
                            std::string(label));
    }
    return result;
}

inline std::string format_float64(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return std::signbit(value) ? "-inf" : "inf";
    }
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10)
           << std::defaultfloat << value;
    return stream.str();
}

inline std::string ecsv_datatype(ValueType type) {
    return std::string(value_type_token(type));
}

inline bool physical_unit(std::string_view unit) {
    return !unit.empty() && unit != "N/A";
}

struct ColumnDeclaration {
    std::string name;
    std::string datatype;
    std::string unit;
    std::string description;
};

inline void emit_column(std::ostringstream &stream,
                        const ColumnDeclaration &column) {
    stream << "# - name: " << yaml_quote(column.name) << "\n";
    stream << "#   datatype: " << yaml_quote(column.datatype) << "\n";
    if (physical_unit(column.unit)) {
        stream << "#   unit: " << yaml_quote(column.unit) << "\n";
    }
    stream << "#   description: " << yaml_quote(column.description) << "\n";
}

inline std::vector<ColumnDeclaration> expected_columns(
    const Document &document) {
    std::vector<ColumnDeclaration> columns{
        {"uid", "int64", "N/A",
         "exact nonnegative artifact-local row key; never persistent detector identity"},
        {"tone_freq", "float64", "Hz",
         "raw readout tone-frequency attribute; not identity"},
        {"array", "int64", "N/A",
         "canonical TolTEC array enum; not row identity"},
        {"nw", "int64", "N/A", "raw-manifest network key"},
        {"kids_tone", "int64", "N/A",
         "zero-based raw channel key within network"},
    };
    for (const auto &field : detail::sorted_registered_fields(document)) {
        columns.push_back({field.name, ecsv_datatype(field.type), field.unit,
                           field.description});
    }
    return columns;
}

inline std::string value_to_csv(const Value &value) {
    return std::visit(
        [](const auto &typed) -> std::string {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, NullValue>) {
                return {};
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return std::to_string(typed);
            } else if constexpr (std::is_same_v<T, double>) {
                return format_float64(typed);
            } else if constexpr (std::is_same_v<T, bool>) {
                return typed ? "True" : "False";
            } else if constexpr (std::is_same_v<T, std::string>) {
                if (typed.empty()) {
                    throw ContractError(
                        "canonical APT v1 forbids an empty non-null ECSV string");
                }
                if (typed.find('\n') != std::string::npos ||
                    typed.find('\r') != std::string::npos) {
                    throw ContractError(
                        "canonical APT v1 row strings cannot contain line breaks");
                }
                return csv_quote(typed);
            }
        },
        value);
}

inline std::string metadata_value(std::string_view line,
                                  std::string_view prefix) {
    if (!starts_with(line, prefix)) {
        throw ContractError("internal canonical APT metadata prefix mismatch");
    }
    return yaml_unquote(line.substr(prefix.size()));
}

inline bool parse_metadata_bool(std::string_view value) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw ContractError("invalid canonical APT metadata boolean");
}

}  // namespace ecsv_detail

inline SerializedEcsv serialize_ecsv(
    const Document &document,
    const FieldRegistry &field_registry = canonical_field_registry_v1()) {
    validate(document, field_registry);
    const auto digests = compute_digests(document, field_registry);
    const auto fields = detail::sorted_registered_fields(document);
    const auto inputs = detail::sorted_raw_inputs(document);
    const auto columns = ecsv_detail::expected_columns(document);

    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "# %ECSV 1.0\n";
    stream << "# ---\n";
    stream << "# datatype:\n";
    for (const auto &column : columns) {
        ecsv_detail::emit_column(stream, column);
    }
    stream << "# meta:\n";
    stream << "#   canonical_apt_v1:\n";
    stream << "#     schema_version: "
           << ecsv_detail::yaml_quote(schema_version_v1) << "\n";
    stream << "#     profile: "
           << ecsv_detail::yaml_quote(document.profile) << "\n";
    stream << "#     field_registry: "
           << ecsv_detail::yaml_quote(document.field_registry) << "\n";
    stream << "#     framing_encoding: "
           << ecsv_detail::yaml_quote(framing_encoding_v1) << "\n";
    stream << "#     semantic_scope: "
           << ecsv_detail::yaml_quote(semantic_scope_v1) << "\n";
    stream << "#     semantic_sha256: "
           << ecsv_detail::yaml_quote(digests.semantic_sha256) << "\n";
    stream << "#     envelope_scope: "
           << ecsv_detail::yaml_quote(envelope_scope_v1) << "\n";
    stream << "#     envelope_sha256: "
           << ecsv_detail::yaml_quote(digests.envelope_sha256) << "\n";
    stream << "#     byte_transport_scope: "
           << ecsv_detail::yaml_quote(byte_transport_scope_v1) << "\n";
    stream << "#     occurrence: "
           << ecsv_detail::yaml_quote(document.envelope.occurrence) << "\n";
    stream << "#     event_reference: "
           << ecsv_detail::yaml_quote(document.envelope.event_reference)
           << "\n";
    stream << "#     output_role: "
           << ecsv_detail::yaml_quote(document.envelope.output_role) << "\n";
    stream << "#     producer: "
           << ecsv_detail::yaml_quote(document.envelope.producer) << "\n";
    stream << "#     software_revision: "
           << ecsv_detail::yaml_quote(document.envelope.software_revision)
           << "\n";
    stream << "#     configuration_reference: "
           << ecsv_detail::yaml_quote(
                  document.envelope.configuration_reference)
           << "\n";
    stream << "#     event_time_utc: "
           << ecsv_detail::yaml_quote(document.envelope.event_time_utc)
           << "\n";
    stream << "#     scientific_context:\n";
    stream << "#       project_id: "
           << ecsv_detail::yaml_quote(document.context.project_id) << "\n";
    stream << "#       source_name: "
           << ecsv_detail::yaml_quote(document.context.source_name) << "\n";
    stream << "#       observation_time_utc: "
           << ecsv_detail::yaml_quote(document.context.observation_time_utc)
           << "\n";
    stream << "#       coordinate_frame: "
           << ecsv_detail::yaml_quote(document.context.coordinate_frame)
           << "\n";
    stream << "#     observation:\n";
    stream << "#       observation: "
           << document.raw_manifest.observation.observation << "\n";
    stream << "#       subobservation: "
           << document.raw_manifest.observation.subobservation << "\n";
    stream << "#       scan: " << document.raw_manifest.observation.scan
           << "\n";
    stream << "#     raw_manifest:\n";
    for (const auto &input : inputs) {
        stream << "#       - network: " << input.network << "\n";
        stream << "#         interface: "
               << ecsv_detail::yaml_quote(input.interface_name) << "\n";
        stream << "#         channel_count: " << input.channel_count << "\n";
    }
    stream << "#     registered_fields:\n";
    for (const auto &field : fields) {
        stream << "#       - name: " << ecsv_detail::yaml_quote(field.name)
               << "\n";
        stream << "#         datatype: "
               << ecsv_detail::yaml_quote(value_type_token(field.type))
               << "\n";
        stream << "#         unit: " << ecsv_detail::yaml_quote(field.unit)
               << "\n";
        stream << "#         nullable: "
               << (field.nullable ? "true" : "false") << "\n";
        stream << "#         authority: "
               << ecsv_detail::yaml_quote(
                      field_authority_token(field.authority))
               << "\n";
        stream << "#         authority_reference: "
               << ecsv_detail::yaml_quote(field.authority_reference) << "\n";
        stream << "#         nonfinite: "
               << ecsv_detail::yaml_quote(
                      nonfinite_policy_token(field.nonfinite))
               << "\n";
        stream << "#         registry: "
               << ecsv_detail::yaml_quote(field.registry) << "\n";
        stream << "#         description: "
               << ecsv_detail::yaml_quote(field.description) << "\n";
        stream << "#         identity_role: \"nonidentity\"\n";
    }
    stream << "#     null_cell: \"unquoted-empty-v1\"\n";
    stream << "#     string_cell: \"quoted-utf8-single-line-v1\"\n";
    stream << "# delimiter: \",\"\n";
    stream << "# schema: \"astropy-2.0\"\n";

    for (std::size_t index = 0; index < columns.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << columns[index].name;
    }
    stream << '\n';
    for (const auto &row : document.rows) {
        stream << row.uid << ','
               << ecsv_detail::format_float64(row.tone_frequency_hz) << ','
               << row.array << ',' << row.network << ',' << row.channel;
        for (const auto &field : fields) {
            stream << ','
                   << ecsv_detail::value_to_csv(row.fields.at(field.name));
        }
        stream << '\n';
    }

    SerializedEcsv result;
    result.bytes = stream.str();
    result.digests = digests;
    result.transport =
        make_byte_transport_hash(result.bytes, digests.envelope_sha256);
    return result;
}

inline ParsedEcsv parse_ecsv(
    std::string_view bytes,
    const FieldRegistry &field_registry = canonical_field_registry_v1()) {
    using namespace ecsv_detail;
    if (bytes.empty() || bytes.back() != '\n') {
        throw ContractError(
            "canonical APT ECSV requires nonempty LF-terminated bytes");
    }
    if (bytes.find('\r') != std::string_view::npos) {
        throw ContractError("canonical APT ECSV rejects CR/CRLF bytes");
    }
    if (!detail::valid_utf8(bytes)) {
        throw ContractError("canonical APT ECSV bytes are not valid UTF-8");
    }

    std::vector<std::string_view> lines;
    std::size_t start = 0;
    while (start < bytes.size()) {
        const auto end = bytes.find('\n', start);
        lines.push_back(bytes.substr(start, end - start));
        start = end + 1;
    }
    if (lines.empty() || lines.front() != "# %ECSV 1.0") {
        throw ContractError("canonical APT ECSV requires exact ECSV 1.0");
    }

    Document document;
    document.profile.clear();
    document.field_registry.clear();
    Digests declared;
    std::string declared_schema;
    std::string declared_framing;
    std::string declared_semantic_scope;
    std::string declared_envelope_scope;
    std::string declared_transport_scope;
    bool found_meta_root = false;
    bool found_delimiter = false;
    bool found_astropy_schema = false;
    bool found_null_cell = false;
    bool found_string_cell = false;
    std::vector<ColumnDeclaration> declared_columns;
    ColumnDeclaration *current_column = nullptr;
    RawInput *current_input = nullptr;
    RegisteredField *current_field = nullptr;
    std::optional<std::size_t> csv_header_index;

    for (std::size_t index = 1; index < lines.size(); ++index) {
        const auto line = lines[index];
        if (!starts_with(line, "#")) {
            csv_header_index = index;
            break;
        }
        if (starts_with(line, "# - name: ")) {
            declared_columns.push_back({});
            current_column = &declared_columns.back();
            current_column->name =
                metadata_value(line, "# - name: ");
        } else if (starts_with(line, "#   datatype: ") && current_column) {
            current_column->datatype =
                metadata_value(line, "#   datatype: ");
        } else if (starts_with(line, "#   unit: ") && current_column) {
            current_column->unit = metadata_value(line, "#   unit: ");
        } else if (starts_with(line, "#   description: ") && current_column) {
            current_column->description =
                metadata_value(line, "#   description: ");
        } else if (line == "#   canonical_apt_v1:") {
            found_meta_root = true;
            current_column = nullptr;
        } else if (starts_with(line, "#     schema_version: ")) {
            declared_schema =
                metadata_value(line, "#     schema_version: ");
        } else if (starts_with(line, "#     profile: ")) {
            document.profile = metadata_value(line, "#     profile: ");
        } else if (starts_with(line, "#     field_registry: ")) {
            document.field_registry =
                metadata_value(line, "#     field_registry: ");
        } else if (starts_with(line, "#     framing_encoding: ")) {
            declared_framing =
                metadata_value(line, "#     framing_encoding: ");
        } else if (starts_with(line, "#     semantic_scope: ")) {
            declared_semantic_scope =
                metadata_value(line, "#     semantic_scope: ");
        } else if (starts_with(line, "#     semantic_sha256: ")) {
            declared.semantic_sha256 =
                metadata_value(line, "#     semantic_sha256: ");
        } else if (starts_with(line, "#     envelope_scope: ")) {
            declared_envelope_scope =
                metadata_value(line, "#     envelope_scope: ");
        } else if (starts_with(line, "#     envelope_sha256: ")) {
            declared.envelope_sha256 =
                metadata_value(line, "#     envelope_sha256: ");
        } else if (starts_with(line, "#     byte_transport_scope: ")) {
            declared_transport_scope =
                metadata_value(line, "#     byte_transport_scope: ");
        } else if (starts_with(line, "#     occurrence: ")) {
            document.envelope.occurrence =
                metadata_value(line, "#     occurrence: ");
        } else if (starts_with(line, "#     event_reference: ")) {
            document.envelope.event_reference =
                metadata_value(line, "#     event_reference: ");
        } else if (starts_with(line, "#     output_role: ")) {
            document.envelope.output_role =
                metadata_value(line, "#     output_role: ");
        } else if (starts_with(line, "#     producer: ")) {
            document.envelope.producer =
                metadata_value(line, "#     producer: ");
        } else if (starts_with(line, "#     software_revision: ")) {
            document.envelope.software_revision =
                metadata_value(line, "#     software_revision: ");
        } else if (starts_with(line, "#     configuration_reference: ")) {
            document.envelope.configuration_reference =
                metadata_value(line, "#     configuration_reference: ");
        } else if (starts_with(line, "#     event_time_utc: ")) {
            document.envelope.event_time_utc =
                metadata_value(line, "#     event_time_utc: ");
        } else if (starts_with(line, "#       project_id: ")) {
            document.context.project_id =
                metadata_value(line, "#       project_id: ");
        } else if (starts_with(line, "#       source_name: ")) {
            document.context.source_name =
                metadata_value(line, "#       source_name: ");
        } else if (starts_with(line, "#       observation_time_utc: ")) {
            document.context.observation_time_utc =
                metadata_value(line, "#       observation_time_utc: ");
        } else if (starts_with(line, "#       coordinate_frame: ")) {
            document.context.coordinate_frame =
                metadata_value(line, "#       coordinate_frame: ");
        } else if (starts_with(line, "#       observation: ")) {
            document.raw_manifest.observation.observation = parse_int64(
                line.substr(std::string_view("#       observation: ").size()),
                "observation");
        } else if (starts_with(line, "#       subobservation: ")) {
            document.raw_manifest.observation.subobservation = parse_int64(
                line.substr(
                    std::string_view("#       subobservation: ").size()),
                "subobservation");
        } else if (starts_with(line, "#       scan: ")) {
            document.raw_manifest.observation.scan = parse_int64(
                line.substr(std::string_view("#       scan: ").size()),
                "scan");
        } else if (starts_with(line, "#       - network: ")) {
            document.raw_manifest.inputs.push_back({});
            current_input = &document.raw_manifest.inputs.back();
            current_input->network = parse_int64(
                line.substr(std::string_view("#       - network: ").size()),
                "raw network");
            current_field = nullptr;
        } else if (starts_with(line, "#         interface: ") &&
                   current_input) {
            current_input->interface_name =
                metadata_value(line, "#         interface: ");
        } else if (starts_with(line, "#         channel_count: ") &&
                   current_input) {
            current_input->channel_count = parse_int64(
                line.substr(
                    std::string_view("#         channel_count: ").size()),
                "raw channel_count");
        } else if (starts_with(line, "#       - name: ")) {
            document.registered_fields.push_back({});
            current_field = &document.registered_fields.back();
            current_field->name = metadata_value(line, "#       - name: ");
            current_input = nullptr;
        } else if (starts_with(line, "#         datatype: ") &&
                   current_field) {
            current_field->type = parse_value_type_token(
                metadata_value(line, "#         datatype: "));
        } else if (starts_with(line, "#         unit: ") && current_field) {
            current_field->unit = metadata_value(line, "#         unit: ");
        } else if (starts_with(line, "#         nullable: ") &&
                   current_field) {
            current_field->nullable = parse_metadata_bool(
                line.substr(std::string_view("#         nullable: ").size()));
        } else if (starts_with(line, "#         authority: ") &&
                   current_field) {
            current_field->authority = parse_field_authority_token(
                metadata_value(line, "#         authority: "));
        } else if (starts_with(line, "#         authority_reference: ") &&
                   current_field) {
            current_field->authority_reference =
                metadata_value(line, "#         authority_reference: ");
        } else if (starts_with(line, "#         nonfinite: ") &&
                   current_field) {
            current_field->nonfinite = parse_nonfinite_policy_token(
                metadata_value(line, "#         nonfinite: "));
        } else if (starts_with(line, "#         registry: ") &&
                   current_field) {
            current_field->registry =
                metadata_value(line, "#         registry: ");
        } else if (starts_with(line, "#         description: ") &&
                   current_field) {
            current_field->description =
                metadata_value(line, "#         description: ");
        } else if (starts_with(line, "#         identity_role: ") &&
                   current_field) {
            if (metadata_value(line, "#         identity_role: ") !=
                "nonidentity") {
                throw ContractError(
                    "canonical APT registered field cannot participate in row identity");
            }
        } else if (line == "#     null_cell: \"unquoted-empty-v1\"") {
            found_null_cell = true;
        } else if (line ==
                   "#     string_cell: \"quoted-utf8-single-line-v1\"") {
            found_string_cell = true;
        } else if (line == "# delimiter: \",\"") {
            found_delimiter = true;
            current_field = nullptr;
            current_input = nullptr;
        } else if (line == "# schema: \"astropy-2.0\"") {
            found_astropy_schema = true;
        }
    }

    if (!csv_header_index || !found_meta_root || !found_delimiter ||
        !found_astropy_schema || !found_null_cell || !found_string_cell ||
        declared_schema != schema_version_v1 ||
        declared_framing != framing_encoding_v1 ||
        declared_semantic_scope != semantic_scope_v1 ||
        declared_envelope_scope != envelope_scope_v1 ||
        declared_transport_scope != byte_transport_scope_v1) {
        throw ContractError(
            "canonical APT ECSV header/profile metadata is incomplete or unsupported");
    }

    const auto expected = expected_columns(document);
    if (declared_columns.size() != expected.size()) {
        throw ContractError(
            "canonical APT ECSV declared column count is incorrect");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (declared_columns[index].name != expected[index].name ||
            declared_columns[index].datatype != expected[index].datatype ||
            declared_columns[index].unit !=
                (physical_unit(expected[index].unit)
                     ? expected[index].unit
                     : std::string{}) ||
            declared_columns[index].description !=
                expected[index].description) {
            throw ContractError(
                "canonical APT ECSV column name/type/unit/description contract mismatch");
        }
    }
    const auto header_cells = parse_csv_line(lines[*csv_header_index]);
    if (header_cells.size() != expected.size()) {
        throw ContractError("canonical APT ECSV CSV header count mismatch");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (header_cells[index].quoted ||
            header_cells[index].value != expected[index].name) {
            throw ContractError(
                "canonical APT ECSV CSV header order/name is noncanonical");
        }
    }

    const auto fields = detail::sorted_registered_fields(document);
    for (std::size_t line_index = *csv_header_index + 1;
         line_index < lines.size(); ++line_index) {
        if (lines[line_index].empty()) {
            throw ContractError("blank data row in canonical APT ECSV");
        }
        const auto cells = parse_csv_line(lines[line_index]);
        if (cells.size() != expected.size()) {
            throw ContractError(
                "canonical APT ECSV row has wrong field cardinality");
        }
        Row row;
        row.uid = parse_int64(cells[0].value, "uid");
        row.tone_frequency_hz = parse_float64(cells[1].value, "tone_freq");
        row.array = parse_int64(cells[2].value, "array");
        row.network = parse_int64(cells[3].value, "nw");
        row.channel = parse_int64(cells[4].value, "kids_tone");
        for (std::size_t field_index = 0; field_index < fields.size();
             ++field_index) {
            const auto &field = fields[field_index];
            const auto &cell = cells[field_index + 5];
            if (cell.value.empty() && !cell.quoted) {
                row.fields[field.name] = NullValue{};
                continue;
            }
            switch (field.type) {
            case ValueType::int64:
                row.fields[field.name] =
                    parse_int64(cell.value, field.name);
                break;
            case ValueType::float64:
                row.fields[field.name] =
                    parse_float64(cell.value, field.name);
                break;
            case ValueType::boolean:
                if (cell.value == "True" && !cell.quoted) {
                    row.fields[field.name] = true;
                } else if (cell.value == "False" && !cell.quoted) {
                    row.fields[field.name] = false;
                } else {
                    throw ContractError(
                        "invalid canonical APT ECSV boolean cell: " +
                        field.name);
                }
                break;
            case ValueType::string:
                if (!cell.quoted || cell.value.empty()) {
                    throw ContractError(
                        "canonical APT ECSV non-null string must be nonempty and quoted: " +
                        field.name);
                }
                row.fields[field.name] = cell.value;
                break;
            }
        }
        document.rows.push_back(std::move(row));
    }

    validate(document, field_registry);
    const auto computed = compute_digests(document, field_registry);
    if (!is_sha256_reference(declared.semantic_sha256) ||
        !is_sha256_reference(declared.envelope_sha256) ||
        computed.semantic_sha256 != declared.semantic_sha256 ||
        computed.envelope_sha256 != declared.envelope_sha256) {
        throw ContractError(
            "canonical APT ECSV embedded semantic/envelope SHA-256 mismatch");
    }
    const auto canonical = serialize_ecsv(document, field_registry);
    if (canonical.bytes != bytes) {
        throw ContractError(
            "canonical APT ECSV bytes are not the canonical v1 serialization");
    }
    return {std::move(document), declared, canonical.transport};
}

inline ParsedEcsv parse_ecsv_with_transport(
    std::string_view bytes, const ByteTransportHash &declared_transport,
    const FieldRegistry &field_registry = canonical_field_registry_v1()) {
    auto parsed = parse_ecsv(bytes, field_registry);
    validate_byte_transport(bytes, parsed.declared_digests.envelope_sha256,
                            declared_transport);
    return parsed;
}

}  // namespace citlali::pipeline::canonical_apt_v1

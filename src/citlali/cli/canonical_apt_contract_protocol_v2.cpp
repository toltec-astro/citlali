#include <citlali/core/cli/canonical_apt_contract_protocol_v2.h>

#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>
#include <citlali/core/pipeline/canonical_apt_v1.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <istream>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::cli::canonical_apt_contract_protocol_v2 {

namespace {

namespace apt = citlali::pipeline::canonical_apt_v2;
namespace publication =
    citlali::pipeline::canonical_artifact_publication;

class ProtocolError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Json {
    using Object = std::map<std::string, Json, std::less<>>;
    std::variant<std::string, Object> value;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    Json parse() {
        if (!citlali::pipeline::canonical_apt_v1::detail::valid_utf8(input_)) {
            throw ProtocolError("request is not valid UTF-8");
        }
        skip_space();
        auto result = parse_value(0);
        skip_space();
        if (position_ != input_.size()) {
            throw ProtocolError("trailing input follows the JSON request");
        }
        return result;
    }

private:
    char take() {
        if (position_ == input_.size()) {
            throw ProtocolError("unexpected end of JSON request");
        }
        return input_[position_++];
    }

    void skip_space() {
        while (position_ < input_.size() &&
               (input_[position_] == ' ' || input_[position_] == '\t' ||
                input_[position_] == '\r' || input_[position_] == '\n')) {
            ++position_;
        }
    }

    void append_utf8(std::string &output, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            output.push_back(static_cast<char>(
                0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0x10ffffU) {
            output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            output.push_back(static_cast<char>(
                0x80U | ((codepoint >> 12U) & 0x3fU)));
            output.push_back(static_cast<char>(
                0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            throw ProtocolError("JSON unicode escape is outside Unicode");
        }
    }

    std::uint32_t hex_quad() {
        std::uint32_t result = 0;
        for (int index = 0; index < 4; ++index) {
            const char ch = take();
            result <<= 4U;
            if (ch >= '0' && ch <= '9') {
                result |= static_cast<std::uint32_t>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                result |= 10U + static_cast<std::uint32_t>(ch - 'a');
            } else if (ch >= 'A' && ch <= 'F') {
                result |= 10U + static_cast<std::uint32_t>(ch - 'A');
            } else {
                throw ProtocolError("invalid JSON unicode escape");
            }
        }
        return result;
    }

    std::string parse_string() {
        if (take() != '"') throw ProtocolError("expected JSON string");
        std::string result;
        while (position_ < input_.size()) {
            const unsigned char raw =
                static_cast<unsigned char>(input_[position_++]);
            if (raw == '"') return result;
            if (raw < 0x20U) {
                throw ProtocolError("unescaped control byte in JSON string");
            }
            if (raw != '\\') {
                result.push_back(static_cast<char>(raw));
                continue;
            }
            const char escaped = take();
            switch (escaped) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u': {
                auto codepoint = hex_quad();
                if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                    if (take() != '\\' || take() != 'u') {
                        throw ProtocolError(
                            "JSON high surrogate lacks low surrogate");
                    }
                    const auto low = hex_quad();
                    if (low < 0xdc00U || low > 0xdfffU) {
                        throw ProtocolError("invalid JSON surrogate pair");
                    }
                    codepoint = 0x10000U +
                        ((codepoint - 0xd800U) << 10U) +
                        (low - 0xdc00U);
                } else if (codepoint >= 0xdc00U &&
                           codepoint <= 0xdfffU) {
                    throw ProtocolError("unpaired JSON low surrogate");
                }
                append_utf8(result, codepoint);
                break;
            }
            default: throw ProtocolError("unsupported JSON escape");
            }
        }
        throw ProtocolError("unterminated JSON string");
    }

    Json parse_object(std::size_t depth) {
        if (take() != '{') throw ProtocolError("expected JSON object");
        Json::Object result;
        skip_space();
        if (position_ < input_.size() && input_[position_] == '}') {
            ++position_;
            return Json{std::move(result)};
        }
        while (true) {
            skip_space();
            if (position_ == input_.size() || input_[position_] != '"') {
                throw ProtocolError("JSON object key is not a string");
            }
            auto key = parse_string();
            skip_space();
            if (take() != ':') throw ProtocolError("JSON member lacks colon");
            skip_space();
            if (!result.emplace(std::move(key), parse_value(depth + 1)).second) {
                throw ProtocolError("duplicate JSON object member");
            }
            skip_space();
            const char delimiter = take();
            if (delimiter == '}') break;
            if (delimiter != ',') {
                throw ProtocolError("invalid JSON object delimiter");
            }
        }
        return Json{std::move(result)};
    }

    Json parse_value(std::size_t depth) {
        if (depth > 16U || position_ == input_.size()) {
            throw ProtocolError("JSON request is too deep or truncated");
        }
        if (input_[position_] == '{') return parse_object(depth);
        if (input_[position_] == '"') return Json{parse_string()};
        throw ProtocolError(
            "v2 request accepts only closed JSON objects and strings");
    }

    std::string_view input_;
    std::size_t position_ = 0;
};

const Json::Object &object(const Json &value, std::string_view context) {
    const auto result = std::get_if<Json::Object>(&value.value);
    if (!result) {
        throw ProtocolError(std::string(context) + " must be an object");
    }
    return *result;
}

const std::string &string_value(const Json &value,
                                std::string_view context) {
    const auto result = std::get_if<std::string>(&value.value);
    if (!result || result->empty()) {
        throw ProtocolError(std::string(context) +
                            " must be a nonempty string");
    }
    return *result;
}

const Json &member(const Json::Object &value, std::string_view name,
                   std::string_view context) {
    const auto found = value.find(name);
    if (found == value.end()) {
        throw ProtocolError(std::string(context) + " lacks member " +
                            std::string(name));
    }
    return found->second;
}

void exact_members(const Json::Object &value,
                   std::initializer_list<std::string_view> names,
                   std::string_view context) {
    std::set<std::string, std::less<>> expected;
    for (const auto name : names) expected.emplace(name);
    std::set<std::string, std::less<>> actual;
    for (const auto &[name, item] : value) {
        (void)item;
        actual.emplace(name);
    }
    if (actual != expected) {
        throw ProtocolError(std::string(context) +
                            " has missing or unknown members");
    }
}

std::string quote(std::string_view input) {
    std::string result{"\""};
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char ch : input) {
        switch (ch) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (ch < 0x20U) {
                result += "\\u00";
                result.push_back(hex[ch >> 4U]);
                result.push_back(hex[ch & 0xfU]);
            } else {
                result.push_back(static_cast<char>(ch));
            }
        }
    }
    result.push_back('"');
    return result;
}

class ObjectBuilder {
public:
    void add(std::string_view key, std::string value) {
        members_.emplace_back(quote(key), std::move(value));
    }
    std::string finish() const {
        std::string result{"{"};
        for (std::size_t index = 0; index < members_.size(); ++index) {
            if (index) result.push_back(',');
            result += members_[index].first;
            result.push_back(':');
            result += members_[index].second;
        }
        result.push_back('}');
        return result;
    }
private:
    std::vector<std::pair<std::string, std::string>> members_;
};

std::string array(const std::vector<std::string> &values) {
    std::string result{"["};
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) result.push_back(',');
        result += values[index];
    }
    result.push_back(']');
    return result;
}

std::string decimal(std::int64_t value) { return quote(std::to_string(value)); }
std::string udecimal(std::uint64_t value) { return quote(std::to_string(value)); }
std::string nullable_text(const std::optional<std::string> &value) {
    return value ? quote(*value) : "null";
}
std::string nullable_int(const std::optional<std::int64_t> &value) {
    return value ? decimal(*value) : "null";
}
std::string nullable_bool(const std::optional<bool> &value) {
    return value ? (*value ? "true" : "false") : "null";
}
std::string nullable_bits(const std::optional<double> &value) {
    return value ? quote(apt::canonical_binary64(*value)) : "null";
}

std::string identity_json(const apt::ComponentIdentity &identity) {
    ObjectBuilder result;
    result.add("artifact_schema", quote(identity.schema));
    result.add("occurrence", quote(identity.occurrence));
    result.add("semantic_sha256", quote(identity.semantic_sha256));
    result.add("envelope_sha256", quote(identity.envelope_sha256));
    return result.finish();
}

std::string reference_json(const apt::ScopedRowReference &reference) {
    ObjectBuilder result;
    result.add("artifact_schema", quote(reference.artifact.schema));
    result.add("occurrence", quote(reference.artifact.occurrence));
    result.add("semantic_sha256", quote(reference.artifact.semantic_sha256));
    result.add("envelope_sha256", quote(reference.artifact.envelope_sha256));
    result.add("local_key", decimal(reference.local_uid));
    return result.finish();
}

std::string observation_json(const apt::ObservationIdentity &observation) {
    ObjectBuilder result;
    result.add("obsnum", decimal(observation.observation));
    result.add("subobsnum", decimal(observation.subobservation));
    result.add("scannum", decimal(observation.scan));
    return result.finish();
}

std::string typed_value_json(const apt::Value &value,
                             apt::ValueType datatype) {
    ObjectBuilder result;
    result.add("datatype", quote(apt::v1::value_type_token(datatype)));
    if (std::holds_alternative<apt::NullValue>(value)) {
        result.add("value", "null");
    } else if (const auto integer = std::get_if<std::int64_t>(&value)) {
        result.add("value", decimal(*integer));
    } else if (const auto number = std::get_if<double>(&value)) {
        result.add("value", quote(apt::canonical_binary64(*number)));
    } else if (const auto boolean = std::get_if<bool>(&value)) {
        result.add("value", *boolean ? "true" : "false");
    } else {
        result.add("value", quote(std::get<std::string>(value)));
    }
    return result.finish();
}

std::string verified_result_json(const apt::VerifiedBundle &verified) {
    const auto root = verified.manifest_path.parent_path();
    const auto descriptors = [&] {
        auto value = verified.manifest.components;
        std::sort(value.begin(), value.end(), [](const auto &lhs,
                                                const auto &rhs) {
            return lhs.role < rhs.role;
        });
        return value;
    }();
    std::vector<std::string> components;
    for (const auto &component : descriptors) {
        ObjectBuilder item;
        item.add("role", quote(component.role));
        item.add("path", quote((root / component.relative_path).string()));
        item.add("schema", quote(component.schema));
        item.add("semantic_sha256", quote(component.semantic_sha256));
        item.add("envelope_sha256", quote(component.envelope_sha256));
        item.add("transport_sha256", quote(component.transport_sha256));
        item.add("byte_count", udecimal(component.byte_count));
        item.add("row_count", udecimal(component.row_count));
        components.push_back(item.finish());
    }
    auto fields = verified.fields;
    std::sort(fields.begin(), fields.end(), [](const auto &lhs,
                                              const auto &rhs) {
        return lhs.field_uid < rhs.field_uid;
    });
    std::vector<std::string> field_values;
    std::map<std::string, const apt::FieldRule *> rule_by_name;
    for (const auto &field : fields) {
        rule_by_name.emplace(field.name, &field);
        ObjectBuilder item;
        item.add("field_uid", decimal(field.field_uid));
        item.add("name", quote(field.name));
        item.add("datatype", quote(apt::v1::value_type_token(field.datatype)));
        item.add("unit", quote(field.unit));
        item.add("nullable", field.nullable ? "true" : "false");
        item.add("authority", quote(field.authority));
        item.add("authority_reference",
                 nullable_text(field.authority_reference));
        item.add("identity_role", quote(field.identity_role));
        item.add("rule", quote(apt::field_operation_token(field.operation)));
        item.add("source_field", nullable_text(field.source_field));
        item.add("missing_policy", quote(field.missing_policy));
        item.add("description", quote(field.description));
        field_values.push_back(item.finish());
    }
    auto sources = verified.sources;
    std::sort(sources.begin(), sources.end(), [](const auto &lhs,
                                                const auto &rhs) {
        return lhs.source_uid < rhs.source_uid;
    });
    std::vector<std::string> source_values;
    std::map<std::int64_t, std::int64_t> raw_source_by_network;
    for (const auto &source : sources) {
        if (source.role == apt::SourceRole::raw) {
            raw_source_by_network.emplace(source.network, source.source_uid);
        }
        ObjectBuilder item;
        item.add("source_uid", decimal(source.source_uid));
        item.add("role", quote(apt::source_role_token(source.role)));
        item.add("content_sha256", quote(source.content_sha256));
        item.add("byte_count", udecimal(source.byte_count));
        item.add("header_observation",
                 observation_json(source.header_observation));
        item.add("nw", decimal(source.network));
        item.add("interface", quote(source.interface_name));
        item.add("channel_count", decimal(source.channel_count));
        source_values.push_back(item.finish());
    }
    std::map<std::int64_t, const apt::RelationRecord *> relation_by_output;
    if (verified.relation) {
        for (const auto &record : verified.relation->rows) {
            relation_by_output.emplace(record.output_uid, &record);
        }
    }
    auto rows = verified.apt.rows;
    std::sort(rows.begin(), rows.end(), [](const auto &lhs,
                                          const auto &rhs) {
        return lhs.uid < rhs.uid;
    });
    std::vector<std::string> row_values;
    for (const auto &row : rows) {
        const auto relation = relation_by_output.find(row.uid);
        const auto source_uid = relation == relation_by_output.end()
            ? raw_source_by_network.at(row.network)
            : relation->second->target_raw_source_uid;
        const auto source_rank = relation == relation_by_output.end()
            ? row.presentation_rank
            : relation->second->source_rank;
        ObjectBuilder dynamic;
        for (const auto &[name, value] : row.fields) {
            dynamic.add(name,
                        typed_value_json(value, rule_by_name.at(name)->datatype));
        }
        ObjectBuilder item;
        item.add("local_key", decimal(row.uid));
        item.add("source_uid", decimal(source_uid));
        item.add("source_rank", udecimal(source_rank));
        item.add("presentation_rank", udecimal(row.presentation_rank));
        item.add("tone_freq_bits", quote(apt::canonical_binary64(
                                      row.tone_frequency_hz)));
        item.add("array", decimal(row.array));
        item.add("nw", decimal(row.network));
        item.add("kids_tone", decimal(row.channel));
        item.add("fields", dynamic.finish());
        row_values.push_back(item.finish());
    }

    ObjectBuilder transport;
    transport.add("scope", quote(apt::bundle_transport_scope_v2));
    transport.add("manifest_sha256", quote(
        "sha256:" + citlali::utils::sha256(
            verified.payload.root_manifest_bytes)));
    transport.add("manifest_byte_count", udecimal(
        verified.payload.root_manifest_bytes.size()));
    transport.add("receipt_sha256", quote(
        "sha256:" + citlali::utils::sha256(
            verified.payload.root_receipt_bytes)));
    transport.add("receipt_byte_count", udecimal(
        verified.payload.root_receipt_bytes.size()));
    transport.add("total_byte_count", udecimal(verified.total_byte_count));
    transport.add("component_count", udecimal(descriptors.size()));

    ObjectBuilder result;
    result.add("product_kind", quote(apt::product_kind_token(
                                  verified.manifest.kind)));
    result.add("profile", quote(verified.manifest.profile));
    result.add("artifact", identity_json(verified.identity));
    result.add("observation", observation_json(verified.manifest.observation));
    result.add("manifest_path", quote(verified.manifest_path.string()));
    result.add("receipt_path", quote(verified.receipt_path.string()));
    const auto apt_descriptor = std::find_if(
        descriptors.begin(), descriptors.end(), [](const auto &item) {
            return item.role == "apt";
        });
    result.add("apt_path", quote((root / apt_descriptor->relative_path).string()));
    result.add("transport", transport.finish());
    result.add("components", array(components));
    result.add("fields", array(field_values));
    result.add("sources", array(source_values));
    result.add("rows", array(row_values));
    result.add("parser_count", udecimal(verified.parser_count));
    result.add("issuance_class", quote(verified.manifest.issuance_class));

    if (verified.relation) {
        const auto &relation = *verified.relation;
        ObjectBuilder matcher;
        matcher.add("matcher_run_occurrence",
                    quote(relation.matcher.matcher_run_occurrence));
        matcher.add("implementation_sha256",
                    quote(relation.matcher.implementation_sha256));
        matcher.add("configuration_sha256",
                    quote(relation.matcher.configuration_sha256));
        matcher.add("method", quote(relation.matcher.method));
        matcher.add("backend", quote(relation.matcher.backend));
        std::vector<std::string> evidence_values;
        for (const auto &evidence : relation.network_evidence) {
            ObjectBuilder item;
            item.add("network_evidence_uid", decimal(evidence.evidence_uid));
            item.add("nw", decimal(evidence.network));
            item.add("status", quote(apt::network_evidence_status_token(
                                  evidence.status)));
            item.add("frequency_shift_hz_bits",
                     nullable_bits(evidence.frequency_shift_hz));
            item.add("gate_hz_bits", nullable_bits(evidence.gate_hz));
            item.add("quality_factor_bits",
                     nullable_bits(evidence.quality_factor));
            evidence_values.push_back(item.finish());
        }
        std::vector<std::string> record_values;
        std::uint64_t matched = 0, unmatched = 0, ambiguous = 0;
        for (const auto &record : relation.rows) {
            matched += record.disposition == apt::RelationDisposition::matched;
            unmatched += record.disposition == apt::RelationDisposition::unmatched;
            ambiguous += record.disposition == apt::RelationDisposition::ambiguous;
            ObjectBuilder item;
            item.add("relation_uid", decimal(record.relation_uid));
            item.add("output_uid", decimal(record.output_uid));
            item.add("target", reference_json(record.target));
            item.add("target_input_uid", decimal(record.target_input_uid));
            item.add("raw_source_uid", decimal(record.target_raw_source_uid));
            item.add("kmp_source_uid", decimal(record.target_kmp_source_uid));
            item.add("kmp_row_index", decimal(record.target_kmp_row_index));
            item.add("source_rank", udecimal(record.source_rank));
            item.add("application_rank", udecimal(record.application_rank));
            item.add("presentation_rank", udecimal(record.presentation_rank));
            item.add("disposition", quote(
                apt::relation_disposition_token(record.disposition)));
            item.add("selected_seed", record.selected_seed
                ? reference_json(*record.selected_seed) : "null");
            item.add("pair_uid", nullable_int(record.selected_pair_uid));
            item.add("separation_hz_bits", nullable_bits(record.separation_hz));
            item.add("is_good_match", nullable_bool(record.is_good_match));
            item.add("network_evidence_uid",
                     decimal(record.network_evidence_uid));
            item.add("reason", quote(record.reason));
            record_values.push_back(item.finish());
        }
        std::vector<std::string> seed_values;
        std::uint64_t seed_matched = 0, seed_unused = 0;
        for (const auto &disposition : verified.seed_dispositions) {
            seed_matched += disposition.disposition == "matched";
            seed_unused += disposition.disposition == "unused";
            ObjectBuilder item;
            item.add("seed", reference_json(disposition.seed));
            item.add("disposition", quote(disposition.disposition));
            item.add("target", disposition.target
                ? reference_json(*disposition.target) : "null");
            item.add("pair_uid", nullable_int(disposition.pair_uid));
            seed_values.push_back(item.finish());
        }
        ObjectBuilder relation_result;
        relation_result.add("baseline_artifact",
                            identity_json(relation.baseline_parent));
        relation_result.add("target_artifact",
                            identity_json(relation.target_parent));
        relation_result.add("matcher", matcher.finish());
        relation_result.add("network_evidence", array(evidence_values));
        relation_result.add("records", array(record_values));
        relation_result.add("seed_dispositions", array(seed_values));
        relation_result.add("target_count", udecimal(relation.rows.size()));
        relation_result.add("matched_count", udecimal(matched));
        relation_result.add("unmatched_count", udecimal(unmatched));
        relation_result.add("ambiguous_count", udecimal(ambiguous));
        relation_result.add("seed_count",
                            udecimal(verified.seed_dispositions.size()));
        relation_result.add("seed_matched_count", udecimal(seed_matched));
        relation_result.add("seed_unused_count", udecimal(seed_unused));
        result.add("relation", relation_result.finish());
    }
    return result.finish();
}

std::string response(std::string_view request_id, std::string_view operation,
                     std::string result) {
    ObjectBuilder value;
    value.add("protocol", quote(protocol_v2));
    value.add("request_id", quote(request_id));
    value.add("operation", quote(operation));
    value.add("status", quote("ok"));
    value.add("result", std::move(result));
    return value.finish();
}

std::string error_response(const std::optional<std::string> &request_id,
                           std::string_view kind, std::string_view code,
                           std::string_view message) {
    ObjectBuilder error;
    error.add("kind", quote(kind));
    error.add("code", quote(code));
    error.add("path", quote("$"));
    error.add("message", quote(message));
    ObjectBuilder value;
    value.add("protocol", quote(protocol_v2));
    value.add("request_id", request_id ? quote(*request_id) : "null");
    value.add("status", quote("error"));
    value.add("error", error.finish());
    return value.finish();
}

std::optional<std::string> recover_request_id(const Json &request) {
    const auto outer = std::get_if<Json::Object>(&request.value);
    if (!outer) return std::nullopt;
    const auto found = outer->find("request_id");
    if (found == outer->end()) return std::nullopt;
    const auto value = std::get_if<std::string>(&found->second.value);
    return value && !value->empty() ? std::optional<std::string>{*value}
                                    : std::nullopt;
}

std::string process_valid_request(const Json &request,
                                  std::string &request_id,
                                  std::string &operation) {
    const auto &outer = object(request, "request");
    exact_members(outer, {"protocol", "request_id", "operation", "payload"},
                  "request");
    if (string_value(member(outer, "protocol", "request"),
                     "request.protocol") != protocol_v2) {
        throw ProtocolError("request.protocol is not supported v2");
    }
    request_id = string_value(member(outer, "request_id", "request"),
                              "request.request_id");
    operation = string_value(member(outer, "operation", "request"),
                             "request.operation");
    const auto &payload = object(member(outer, "payload", "request"),
                                 "request.payload");
    if (operation == validate_bundle_operation_v2 ||
        operation == describe_baseline_operation_v2) {
        exact_members(payload, {"root_manifest"}, "request.payload");
        const auto manifest = std::filesystem::absolute(string_value(
            member(payload, "root_manifest", "request.payload"),
            "request.payload.root_manifest"));
        auto verified = apt::verify_bundle_filesystem(manifest, true);
        if (operation == describe_baseline_operation_v2 &&
            verified.manifest.kind != apt::BundleKind::baseline) {
            throw apt::ContractError(
                "describe-baseline-v2 requires a fresh baseline bundle");
        }
        return response(request_id, operation,
                        verified_result_json(verified));
    }
    if (operation == canonicalize_target_operation_v2 ||
        operation == issue_observation_apt_operation_v2 ||
        operation == migrate_v1_to_v2_operation) {
        throw apt::ContractError(
            "operation is defined but deliberately unavailable until the TolAPT/TolProj compact-v2 boundary is integrated");
    }
    throw ProtocolError("request.operation is not supported");
}

}  // namespace

ProtocolResult process_request_line(
    std::string_view request_json,
    const ProtocolDependencies &dependencies) {
    (void)dependencies;
    std::optional<std::string> request_id;
    try {
        auto request = JsonParser(request_json).parse();
        request_id = recover_request_id(request);
        std::string exact_request_id;
        std::string operation;
        return {success_exit_code,
                process_valid_request(request, exact_request_id, operation)};
    } catch (const ProtocolError &error) {
        return {protocol_error_exit_code,
                error_response(request_id, "protocol", "invalid-request",
                               error.what())};
    } catch (const apt::ContractError &error) {
        return {contract_rejection_exit_code,
                error_response(request_id, "contract", "contract-rejection",
                               error.what())};
    } catch (const publication::PublicationError &error) {
        return {contract_rejection_exit_code,
                error_response(request_id, "contract",
                               "publication-rejection", error.what())};
    } catch (const std::exception &error) {
        return {protocol_error_exit_code,
                error_response(request_id, "internal", "internal-error",
                               error.what())};
    }
}

ProtocolDependencies production_dependencies() {
    ProtocolDependencies result;
    result.issuance_factory = [] {
        return publication::make_entropy_issuance(
            "apt-v2-occurrence:entropy/", "apt-v2-event:entropy/");
    };
    return result;
}

std::optional<int> dispatch_if_requested(
    int argc, char *argv[], std::istream &input, std::ostream &output,
    const ProtocolDependencies &dependencies) {
    bool requested = false;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == cli_option_v2) requested = true;
    }
    if (!requested) return std::nullopt;
    if (argc != 2 || std::string_view(argv[1]) != cli_option_v2) {
        const auto result = error_response(
            std::nullopt, "protocol", "invalid-invocation",
            "v2 protocol mode accepts no ordinary CLI arguments");
        output << result << '\n';
        return protocol_error_exit_code;
    }
    constexpr std::size_t maximum_request_bytes = 1024U * 1024U;
    std::string line;
    line.reserve(4096);
    char ch = 0;
    bool have_lf = false;
    while (input.get(ch)) {
        if (ch == '\n') {
            have_lf = true;
            break;
        }
        if (line.size() >= maximum_request_bytes) {
            const auto result = error_response(
                std::nullopt, "protocol", "request-too-large",
                "v2 request exceeds the 1 MiB framing bound");
            output << result << '\n';
            return protocol_error_exit_code;
        }
        line.push_back(ch);
    }
    if (!have_lf || input.peek() != std::char_traits<char>::eof()) {
        const auto result = error_response(
            std::nullopt, "protocol", "invalid-request-framing",
            "v2 protocol requires exactly one LF-terminated request");
        output << result << '\n';
        return protocol_error_exit_code;
    }
    const auto result = process_request_line(line, dependencies);
    output << result.response_json << '\n';
    output.flush();
    return result.exit_code;
}

}  // namespace citlali::cli::canonical_apt_contract_protocol_v2

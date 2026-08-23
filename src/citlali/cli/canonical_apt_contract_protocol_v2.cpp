#include <citlali/core/cli/canonical_apt_contract_protocol_v2.h>

#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>
#include <citlali/core/pipeline/canonical_apt_v1.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <istream>
#include <iterator>
#include <limits>
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
    using Array = std::vector<Json>;
    std::variant<std::string, Object, Array> value;
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
                throw ProtocolError(
                    "invalid JSON object delimiter at byte " +
                    std::to_string(position_ - 1U) + ": " + delimiter);
            }
        }
        return Json{std::move(result)};
    }

    Json parse_array(std::size_t depth) {
        if (take() != '[') throw ProtocolError("expected JSON array");
        Json::Array result;
        skip_space();
        if (position_ < input_.size() && input_[position_] == ']') {
            ++position_;
            return Json{std::move(result)};
        }
        while (true) {
            skip_space();
            result.push_back(parse_value(depth + 1));
            skip_space();
            const char delimiter = take();
            if (delimiter == ']') break;
            if (delimiter != ',') {
                throw ProtocolError(
                    "invalid JSON array delimiter at byte " +
                    std::to_string(position_ - 1U));
            }
        }
        return Json{std::move(result)};
    }

    Json parse_value(std::size_t depth) {
        if (depth > 16U || position_ == input_.size()) {
            throw ProtocolError("JSON request is too deep or truncated");
        }
        if (input_[position_] == '{') return parse_object(depth);
        if (input_[position_] == '[') return parse_array(depth);
        if (input_[position_] == '"') return Json{parse_string()};
        throw ProtocolError(
            "v2 request accepts only closed JSON objects, arrays, and strings");
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

const Json::Array &array_value(const Json &value,
                               std::string_view context) {
    const auto result = std::get_if<Json::Array>(&value.value);
    if (!result) {
        throw ProtocolError(std::string(context) + " must be an array");
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

std::int64_t int64_value(const Json &value, std::string_view context) {
    const auto &text = string_value(value, context);
    std::int64_t result = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        std::to_string(result) != text) {
        throw ProtocolError(std::string(context) +
                            " must be canonical int64 text");
    }
    return result;
}

std::uint64_t uint64_value(const Json &value, std::string_view context) {
    const auto &text = string_value(value, context);
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        std::to_string(result) != text) {
        throw ProtocolError(std::string(context) +
                            " must be canonical uint64 text");
    }
    return result;
}

double binary64_value(const Json &value, std::string_view context) {
    const auto &text = string_value(value, context);
    if (text.size() != 16U) {
        throw ProtocolError(std::string(context) +
                            " must be 16 lowercase binary64 hex digits");
    }
    std::uint64_t bits = 0;
    for (const char ch : text) {
        bits <<= 4U;
        if (ch >= '0' && ch <= '9') {
            bits |= static_cast<std::uint64_t>(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            bits |= 10U + static_cast<std::uint64_t>(ch - 'a');
        } else {
            throw ProtocolError(std::string(context) +
                                " must be lowercase binary64 hex");
        }
    }
    const double result = std::bit_cast<double>(bits);
    if (!std::isfinite(result) || apt::canonical_binary64(result) != text) {
        throw ProtocolError(std::string(context) +
                            " must encode one finite canonical binary64");
    }
    return result;
}

bool boolean_value(const Json &value, std::string_view context) {
    const auto &text = string_value(value, context);
    if (text == "true") return true;
    if (text == "false") return false;
    throw ProtocolError(std::string(context) +
                        " must be the string true or false");
}

apt::ObservationIdentity observation_value(const Json &value,
                                            std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record, {"obsnum", "subobsnum", "scannum"}, context);
    return {
        int64_value(member(record, "obsnum", context),
                    std::string(context) + ".obsnum"),
        int64_value(member(record, "subobsnum", context),
                    std::string(context) + ".subobsnum"),
        int64_value(member(record, "scannum", context),
                    std::string(context) + ".scannum"),
    };
}

std::string read_bounded_file(const std::filesystem::path &path,
                              std::uint64_t maximum_bytes,
                              std::string_view label) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size > maximum_bytes) {
        throw apt::ContractError(std::string(label) +
                                 " is absent, empty, or too large");
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw apt::ContractError("failed to open " + std::string(label));
    }
    std::string bytes{std::istreambuf_iterator<char>(stream),
                      std::istreambuf_iterator<char>()};
    if (stream.bad() || bytes.size() != size) {
        throw apt::ContractError("failed to read exact " +
                                 std::string(label) + " bytes");
    }
    return bytes;
}

void require_bound_source_bytes(const std::filesystem::path &locator,
                                const apt::SourceRecord &source) {
    constexpr std::uint64_t maximum_source_bytes = 4ULL * 1024ULL * 1024ULL *
        1024ULL;
    const auto bytes = read_bounded_file(
        locator, maximum_source_bytes, "canonical APT v2 bound source");
    if (bytes.size() != source.byte_count ||
        "sha256:" + citlali::utils::sha256(bytes) != source.content_sha256) {
        throw apt::ContractError(
            "canonical APT v2 bound source bytes disagree with the request");
    }
}

struct RequestedMatch {
    std::int64_t target_uid = 0;
    std::int64_t seed_uid = 0;
    double separation_hz = 0.0;
    bool is_good_match = false;
};

struct ProducerMatchRequest {
    std::string producer;
    std::string software_revision;
    std::string configuration_sha256;
    std::string event_time_utc;
    apt::ObservationIdentity observation;
    std::vector<apt::SourceRecord> sources;
    apt::TargetManifest target;
    apt::MatcherEvidence matcher;
    std::vector<apt::NetworkEvidence> network_evidence;
    std::vector<RequestedMatch> matches;
};

ProducerMatchRequest parse_producer_match_request(
    const Json &value, const publication::IssuanceFactory &issuance_factory) {
    const auto &document = object(value, "match request");
    exact_members(document,
                  {"schema", "producer", "software_revision",
                   "configuration_sha256", "event_time_utc", "observation",
                   "sources", "rows", "matcher", "network_evidence",
                   "matches"},
                  "match request");
    if (string_value(member(document, "schema", "match request"),
                     "match request.schema") !=
        "tolproj-canonical-apt-match-request-v1") {
        throw ProtocolError("match request schema is unsupported");
    }

    ProducerMatchRequest result;
    result.producer = string_value(
        member(document, "producer", "match request"),
        "match request.producer");
    if (result.producer != "tolproj") {
        throw ProtocolError(
            "match request producer must be tolproj");
    }
    result.software_revision = string_value(
        member(document, "software_revision", "match request"),
        "match request.software_revision");
    result.configuration_sha256 = string_value(
        member(document, "configuration_sha256", "match request"),
        "match request.configuration_sha256");
    apt::require_sha256(result.configuration_sha256,
                        "match request configuration digest");
    result.event_time_utc = string_value(
        member(document, "event_time_utc", "match request"),
        "match request.event_time_utc");
    result.observation = observation_value(
        member(document, "observation", "match request"),
        "match request.observation");

    for (const auto &item : array_value(
             member(document, "sources", "match request"),
             "match request.sources")) {
        const auto &record = object(item, "match request source");
        exact_members(record,
                      {"source_uid", "role", "locator", "content_sha256",
                       "byte_count", "header_observation", "nw", "interface",
                       "channel_count"},
                      "match request source");
        apt::SourceRecord source;
        source.source_uid = int64_value(
            member(record, "source_uid", "match request source"),
            "match request source.source_uid");
        source.role = apt::parse_source_role(string_value(
            member(record, "role", "match request source"),
            "match request source.role"));
        source.content_sha256 = string_value(
            member(record, "content_sha256", "match request source"),
            "match request source.content_sha256");
        source.byte_count = uint64_value(
            member(record, "byte_count", "match request source"),
            "match request source.byte_count");
        source.header_observation = observation_value(
            member(record, "header_observation", "match request source"),
            "match request source.header_observation");
        source.network = int64_value(
            member(record, "nw", "match request source"),
            "match request source.nw");
        source.interface_name = string_value(
            member(record, "interface", "match request source"),
            "match request source.interface");
        source.channel_count = int64_value(
            member(record, "channel_count", "match request source"),
            "match request source.channel_count");
        apt::validate(source);
        const std::filesystem::path locator{string_value(
            member(record, "locator", "match request source"),
            "match request source.locator")};
        if (!locator.is_absolute() || locator.lexically_normal() != locator) {
            throw ProtocolError(
                "match request source.locator must be an absolute normalized path");
        }
        require_bound_source_bytes(locator, source);
        result.sources.push_back(std::move(source));
    }

    const auto opaque_target = publication::issue_opaque(issuance_factory);
    apt::TargetManifest target;
    target.issuance = {opaque_target.occurrence,
                       opaque_target.event_reference,
                       result.producer,
                       result.software_revision,
                       result.configuration_sha256,
                       result.event_time_utc};
    target.observation = result.observation;
    target.sources = result.sources;
    for (const auto &item : array_value(
             member(document, "rows", "match request"),
             "match request.rows")) {
        const auto &record = object(item, "match request row");
        const std::set<std::string> without_flag{
            "uid", "input_uid", "raw_source_uid", "kmp_source_uid",
            "kmp_row_index", "source_rank", "application_rank",
            "tone_freq_bits", "array", "nw", "kids_tone", "kids_fr_bits",
            "kids_f_out_bits", "kids_Qr_bits"};
        std::set<std::string> actual;
        for (const auto &[name, field] : record) {
            (void)field;
            actual.insert(name);
        }
        const bool has_flag = actual.contains("kids_flag");
        auto expected = without_flag;
        if (has_flag) expected.insert("kids_flag");
        if (actual != expected) {
            throw ProtocolError(
                "match request row has missing or unknown members");
        }
        apt::TargetRow row;
        row.uid = int64_value(member(record, "uid", "match request row"),
                              "match request row.uid");
        row.input_uid = int64_value(
            member(record, "input_uid", "match request row"),
            "match request row.input_uid");
        row.raw_source_uid = int64_value(
            member(record, "raw_source_uid", "match request row"),
            "match request row.raw_source_uid");
        row.kmp_source_uid = int64_value(
            member(record, "kmp_source_uid", "match request row"),
            "match request row.kmp_source_uid");
        row.kmp_row_index = int64_value(
            member(record, "kmp_row_index", "match request row"),
            "match request row.kmp_row_index");
        row.source_rank = uint64_value(
            member(record, "source_rank", "match request row"),
            "match request row.source_rank");
        row.application_rank = uint64_value(
            member(record, "application_rank", "match request row"),
            "match request row.application_rank");
        row.tone_frequency_hz = binary64_value(
            member(record, "tone_freq_bits", "match request row"),
            "match request row.tone_freq_bits");
        row.array = int64_value(member(record, "array", "match request row"),
                                "match request row.array");
        row.network = int64_value(member(record, "nw", "match request row"),
                                  "match request row.nw");
        row.channel = int64_value(
            member(record, "kids_tone", "match request row"),
            "match request row.kids_tone");
        row.fields = {
            {"kids_fr", binary64_value(
                            member(record, "kids_fr_bits", "match request row"),
                            "match request row.kids_fr_bits")},
            {"kids_f_out", binary64_value(
                               member(record, "kids_f_out_bits", "match request row"),
                               "match request row.kids_f_out_bits")},
            {"kids_Qr", binary64_value(
                            member(record, "kids_Qr_bits", "match request row"),
                            "match request row.kids_Qr_bits")},
        };
        if (has_flag) {
            row.fields.emplace(
                "kids_flag",
                int64_value(member(record, "kids_flag", "match request row"),
                            "match request row.kids_flag"));
        }
        target.rows.push_back(std::move(row));
    }
    apt::validate(target);
    result.target = std::move(target);

    const auto &matcher = object(
        member(document, "matcher", "match request"),
        "match request.matcher");
    exact_members(matcher,
                  {"implementation_sha256", "configuration_sha256",
                   "method", "backend"},
                  "match request.matcher");
    const auto opaque_matcher = publication::issue_opaque(issuance_factory);
    result.matcher = {
        opaque_matcher.occurrence,
        string_value(member(matcher, "implementation_sha256",
                            "match request.matcher"),
                     "match request.matcher.implementation_sha256"),
        string_value(member(matcher, "configuration_sha256",
                            "match request.matcher"),
                     "match request.matcher.configuration_sha256"),
        string_value(member(matcher, "method", "match request.matcher"),
                     "match request.matcher.method"),
        string_value(member(matcher, "backend", "match request.matcher"),
                     "match request.matcher.backend"),
    };
    apt::require_sha256(result.matcher.implementation_sha256,
                        "matcher implementation digest");
    apt::require_sha256(result.matcher.configuration_sha256,
                        "matcher configuration digest");
    if (result.matcher.method != "tolproj-legacy-tone-match" ||
        (result.matcher.backend != "astropy" &&
         result.matcher.backend != "stilts")) {
        throw ProtocolError(
            "match request matcher identity is unsupported");
    }

    for (const auto &item : array_value(
             member(document, "network_evidence", "match request"),
             "match request.network_evidence")) {
        const auto &record = object(item, "match request network evidence");
        exact_members(record,
                      {"evidence_uid", "nw", "frequency_shift_bits",
                       "gate_bits", "quality_factor_bits"},
                      "match request network evidence");
        result.network_evidence.push_back({
            int64_value(member(record, "evidence_uid",
                               "match request network evidence"),
                        "match request network evidence.evidence_uid"),
            int64_value(member(record, "nw",
                               "match request network evidence"),
                        "match request network evidence.nw"),
            apt::NetworkEvidenceStatus::matched_capable,
            binary64_value(member(record, "frequency_shift_bits",
                                  "match request network evidence"),
                           "match request network evidence.frequency_shift_bits"),
            binary64_value(member(record, "gate_bits",
                                  "match request network evidence"),
                           "match request network evidence.gate_bits"),
            binary64_value(member(record, "quality_factor_bits",
                                  "match request network evidence"),
                           "match request network evidence.quality_factor_bits"),
        });
    }

    for (const auto &item : array_value(
             member(document, "matches", "match request"),
             "match request.matches")) {
        const auto &record = object(item, "match request match");
        exact_members(record,
                      {"target_uid", "seed_uid", "separation_bits",
                       "is_good_match"},
                      "match request match");
        result.matches.push_back({
            int64_value(member(record, "target_uid", "match request match"),
                        "match request match.target_uid"),
            int64_value(member(record, "seed_uid", "match request match"),
                        "match request match.seed_uid"),
            binary64_value(member(record, "separation_bits",
                                  "match request match"),
                           "match request match.separation_bits"),
            boolean_value(member(record, "is_good_match",
                                 "match request match"),
                          "match request match.is_good_match"),
        });
    }
    return result;
}

apt::VerifiedBundle issue_observation_matched_bundle(
    const apt::VerifiedBundle &baseline, ProducerMatchRequest request,
    const std::filesystem::path &publication_manifest,
    const ProtocolDependencies &dependencies) {
    if (baseline.manifest.kind != apt::BundleKind::baseline ||
        baseline.manifest.issuance_class != "fresh") {
        throw apt::ContractError(
            "observation issuance requires one fresh Beammap baseline v2 bundle");
    }
    apt::validate(request.target);
    if (request.target.observation != request.observation ||
        request.target.sources != request.sources) {
        throw apt::ContractError(
            "observation issuance target facts are internally inconsistent");
    }

    std::map<std::int64_t, const apt::AptRow *> seed_by_uid;
    for (const auto &seed : baseline.apt.rows) {
        if (!seed_by_uid.emplace(seed.uid, &seed).second) {
            throw apt::ContractError(
                "observation issuance baseline repeats a seed UID");
        }
    }
    std::map<std::int64_t, const RequestedMatch *> match_by_target;
    std::set<std::int64_t> selected_seeds;
    for (const auto &match : request.matches) {
        if (!match_by_target.emplace(match.target_uid, &match).second ||
            !selected_seeds.insert(match.seed_uid).second ||
            !seed_by_uid.contains(match.seed_uid)) {
            throw apt::ContractError(
                "observation issuance match relation is duplicate or foreign");
        }
    }
    std::map<std::int64_t, const apt::NetworkEvidence *> evidence_by_network;
    for (const auto &evidence : request.network_evidence) {
        if (!evidence_by_network.emplace(evidence.network, &evidence).second) {
            throw apt::ContractError(
                "observation issuance repeats network evidence");
        }
    }

    const auto opaque_output =
        publication::issue_opaque(dependencies.issuance_factory);
    const apt::IssuanceContext output_issuance{
        opaque_output.occurrence,
        opaque_output.event_reference,
        request.producer,
        request.software_revision,
        request.configuration_sha256,
        request.event_time_utc,
    };

    apt::AptTable output;
    output.kind = apt::BundleKind::matched;
    output.issuance = output_issuance;
    output.observation = request.observation;
    output.field_rules = apt::canonical_structural_field_rules_v2();
    std::int64_t next_field_uid =
        static_cast<std::int64_t>(output.field_rules.size());
    std::vector<const apt::FieldRule *> copied_baseline_fields;
    for (const auto &field : baseline.apt.field_rules) {
        if (field.name == "uid" || field.name == "tone_freq" ||
            field.name == "array" || field.name == "nw" ||
            field.name == "kids_tone" ||
            apt::is_authorized_kmp_field(field.name)) {
            continue;
        }
        auto copied = field;
        copied.field_uid = next_field_uid++;
        copied.nullable = true;
        copied.operation = apt::FieldOperation::copy_seed_or_null;
        copied.missing_policy = "typed-null";
        output.field_rules.push_back(std::move(copied));
        copied_baseline_fields.push_back(&field);
    }
    const bool include_kids_flag =
        request.target.rows.front().fields.contains("kids_flag");
    auto kmp_rules = apt::canonical_kmp_field_rules_v2(include_kids_flag);
    for (auto &field : kmp_rules) {
        field.field_uid = next_field_uid++;
        output.field_rules.push_back(std::move(field));
    }

    std::vector<const apt::TargetRow *> ordered_targets;
    ordered_targets.reserve(request.target.rows.size());
    for (const auto &row : request.target.rows) ordered_targets.push_back(&row);
    std::sort(ordered_targets.begin(), ordered_targets.end(),
              [](const auto *lhs, const auto *rhs) {
                  return lhs->application_rank < rhs->application_rank;
              });

    apt::RelationTable relation;
    relation.issuance = output_issuance;
    relation.observation = request.observation;
    relation.target_parent = apt::target_identity(request.target);
    relation.target_issuance = request.target.issuance;
    relation.baseline_parent = baseline.identity;
    relation.matcher = std::move(request.matcher);
    relation.network_evidence = request.network_evidence;

    std::int64_t next_pair_uid = 0;
    for (const auto *target : ordered_targets) {
        const auto evidence = evidence_by_network.find(target->network);
        if (evidence == evidence_by_network.end()) {
            throw apt::ContractError(
                "observation issuance target lacks network evidence");
        }
        const auto selected = match_by_target.find(target->uid);
        const apt::AptRow *seed = nullptr;
        if (selected != match_by_target.end()) {
            seed = seed_by_uid.at(selected->second->seed_uid);
            if (seed->network != target->network) {
                throw apt::ContractError(
                    "observation issuance selected a cross-network seed");
            }
            const auto target_frequency = std::get_if<double>(
                &target->fields.at("kids_fr"));
            if (target_frequency == nullptr ||
                !evidence->second->frequency_shift_hz) {
                throw apt::ContractError(
                    "observation issuance selected match lacks finite "
                    "frequency evidence");
            }
            const auto seed_frequency_field = seed->fields.find("kids_fr");
            const double seed_frequency =
                seed_frequency_field == seed->fields.end()
                ? seed->tone_frequency_hz
                : std::get<double>(seed_frequency_field->second);
            const auto expected_separation = std::abs(
                *target_frequency + *evidence->second->frequency_shift_hz -
                seed_frequency);
            const auto flag = seed->fields.find("flag");
            if (apt::canonical_binary64(expected_separation) !=
                    apt::canonical_binary64(
                        selected->second->separation_hz) ||
                flag == seed->fields.end() ||
                !std::holds_alternative<std::int64_t>(flag->second) ||
                (std::get<std::int64_t>(flag->second) == 0) !=
                    selected->second->is_good_match) {
                throw apt::ContractError(
                    "observation issuance selected match evidence disagrees "
                    "with target and baseline facts");
            }
        }

        apt::AptRow row;
        row.uid = target->uid;
        row.presentation_rank = target->application_rank;
        row.tone_frequency_hz = target->tone_frequency_hz;
        row.array = target->array;
        row.network = target->network;
        row.channel = target->channel;
        for (const auto *field : copied_baseline_fields) {
            row.fields.emplace(
                field->name,
                seed ? apt::copied_seed_value_or_null(
                           seed->fields.at(field->name))
                     : apt::Value{apt::NullValue{}});
        }
        for (const auto name :
             {"kids_fr", "kids_f_out", "kids_Qr", "kids_flag"}) {
            const auto value = target->fields.find(name);
            if (value != target->fields.end()) {
                row.fields.emplace(name, value->second);
            }
        }
        output.rows.push_back(std::move(row));

        apt::RelationRecord record;
        record.relation_uid =
            static_cast<std::int64_t>(target->application_rank);
        record.output_uid = target->uid;
        record.target = {relation.target_parent, target->uid};
        record.target_input_uid = target->input_uid;
        record.target_raw_source_uid = target->raw_source_uid;
        record.target_kmp_source_uid = target->kmp_source_uid;
        record.target_kmp_row_index = target->kmp_row_index;
        record.source_rank = target->source_rank;
        record.application_rank = target->application_rank;
        record.presentation_rank = target->application_rank;
        record.network_evidence_uid = evidence->second->evidence_uid;
        if (selected == match_by_target.end()) {
            record.disposition = apt::RelationDisposition::unmatched;
            record.reason = "tolproj-realized-unmatched";
        } else {
            record.disposition = apt::RelationDisposition::matched;
            record.selected_pair_uid = next_pair_uid++;
            record.selected_seed = apt::ScopedRowReference{
                baseline.identity, selected->second->seed_uid};
            record.separation_hz = selected->second->separation_hz;
            record.is_good_match = selected->second->is_good_match;
            record.reason = "tolproj-realized-selected-seed";
        }
        relation.rows.push_back(std::move(record));
    }
    if (match_by_target.size() != static_cast<std::size_t>(next_pair_uid)) {
        throw apt::ContractError(
            "observation issuance contains a match for an absent target");
    }

    auto prepared = apt::prepare_matched_bundle(
        std::move(output), std::move(relation), request.sources, {}, baseline);
    (void)apt::publish_prepared_bundle(
        publication_manifest, prepared, dependencies.publication_hooks);
    auto verified = apt::verify_bundle_filesystem(publication_manifest, true);
    if (verified.identity != prepared.identity ||
        verified.total_byte_count != prepared.total_byte_count ||
        verified.manifest.kind != apt::BundleKind::matched ||
        verified.manifest.issuance_class != "fresh" ||
        verified.manifest.baseline_parent != baseline.identity ||
        verified.manifest.observation != request.observation) {
        throw apt::ContractError(
            "published observation APT v2 disagrees with intended issuance");
    }
    return verified;
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
                                  std::string &operation,
                                  const ProtocolDependencies &dependencies) {
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
    if (operation == issue_observation_apt_operation_v2) {
        exact_members(payload,
                      {"baseline_root_manifest", "match_request",
                       "match_request_sha256", "publication_root_manifest"},
                      "request.payload");
        const auto baseline_manifest = std::filesystem::absolute(string_value(
            member(payload, "baseline_root_manifest", "request.payload"),
            "request.payload.baseline_root_manifest"));
        const auto match_request_path = std::filesystem::absolute(string_value(
            member(payload, "match_request", "request.payload"),
            "request.payload.match_request"));
        const auto expected_match_sha256 = string_value(
            member(payload, "match_request_sha256", "request.payload"),
            "request.payload.match_request_sha256");
        apt::require_sha256(expected_match_sha256,
                            "match request transport digest");
        const auto publication_manifest =
            std::filesystem::absolute(string_value(
                member(payload, "publication_root_manifest", "request.payload"),
                "request.payload.publication_root_manifest"));
        constexpr std::uint64_t maximum_match_request_bytes =
            64ULL * 1024ULL * 1024ULL;
        const auto match_bytes = read_bounded_file(
            match_request_path, maximum_match_request_bytes,
            "canonical APT v2 TolProj match request");
        if ("sha256:" + citlali::utils::sha256(match_bytes) !=
            expected_match_sha256) {
            throw apt::ContractError(
                "canonical APT v2 TolProj match request digest disagrees");
        }
        auto baseline = apt::verify_bundle_filesystem(baseline_manifest, true);
        auto producer_request = parse_producer_match_request(
            JsonParser(match_bytes).parse(), dependencies.issuance_factory);
        auto verified = issue_observation_matched_bundle(
            baseline, std::move(producer_request), publication_manifest,
            dependencies);
        return response(request_id, operation,
                        verified_result_json(verified));
    }
    if (operation == canonicalize_target_operation_v2 ||
        operation == migrate_v1_to_v2_operation) {
        throw apt::ContractError(
            "operation is defined but deliberately unavailable at this checkpoint");
    }
    throw ProtocolError("request.operation is not supported");
}

}  // namespace

ProtocolResult process_request_line(
    std::string_view request_json,
    const ProtocolDependencies &dependencies) {
    std::optional<std::string> request_id;
    try {
        auto request = JsonParser(request_json).parse();
        request_id = recover_request_id(request);
        std::string exact_request_id;
        std::string operation;
        return {success_exit_code,
                process_valid_request(request, exact_request_id, operation,
                                      dependencies)};
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

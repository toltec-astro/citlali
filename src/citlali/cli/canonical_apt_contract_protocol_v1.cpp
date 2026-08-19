#include <citlali/core/cli/canonical_apt_contract_protocol_v1.h>

#include <citlali/core/pipeline/canonical_apt_observation_v1.h>
#include <citlali_config/gitversion.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
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
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace citlali::cli::canonical_apt_contract_protocol_v1 {

namespace {

namespace observation =
    citlali::pipeline::canonical_apt_observation_v1;
namespace baseline = citlali::pipeline::canonical_apt_v1;
namespace artifact_publication =
    citlali::pipeline::canonical_artifact_publication;

inline constexpr std::string_view baseline_contract_id_v1 =
    "apt-prod-001-canonical-baseline-apt-v1";
inline constexpr std::string_view baseline_contract_sha256_v1 =
    "eb343ced3d4c8f303095b53f3fdca087bb478bd53d675b12958b47df244173b9";
inline constexpr std::string_view publication_receipt_schema_v1 =
    "citlali-canonical-apt-publication-receipt-v1";

class ProtocolError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct JsonNumber {
    std::string token;
};

struct Json {
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json, std::less<>>;
    using Value = std::variant<std::nullptr_t, bool, std::string, JsonNumber,
                               Array, Object>;

    Value value{nullptr};

    Json() = default;
    Json(std::nullptr_t) : value(nullptr) {}
    Json(bool input) : value(input) {}
    Json(std::string input) : value(std::move(input)) {}
    Json(std::string_view input) : value(std::string(input)) {}
    Json(const char *input) : value(std::string(input)) {}
    Json(JsonNumber input) : value(std::move(input)) {}
    Json(Array input) : value(std::move(input)) {}
    Json(Object input) : value(std::move(input)) {}
};

bool valid_utf8(std::string_view value) {
    return baseline::detail::valid_utf8(value);
}

void append_utf8(std::string &output, std::uint32_t codepoint) {
    if (codepoint <= 0x7fU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
        output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
        output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
        output.push_back(
            static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0x10ffffU) {
        output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
        output.push_back(
            static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
        output.push_back(
            static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
        throw ProtocolError("JSON unicode escape is outside Unicode range");
    }
}

class JsonParser {
public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    Json parse() {
        if (!valid_utf8(input_)) {
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
    static bool is_digit(char ch) { return ch >= '0' && ch <= '9'; }

    void skip_space() {
        while (position_ < input_.size() &&
               (input_[position_] == ' ' || input_[position_] == '\t' ||
                input_[position_] == '\n' || input_[position_] == '\r')) {
            ++position_;
        }
    }

    char take() {
        if (position_ == input_.size()) {
            throw ProtocolError("unexpected end of JSON request");
        }
        return input_[position_++];
    }

    void expect(char expected) {
        if (take() != expected) {
            throw ProtocolError("unexpected JSON token");
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
        expect('"');
        std::string result;
        while (position_ < input_.size()) {
            const unsigned char raw =
                static_cast<unsigned char>(input_[position_++]);
            if (raw == '"') {
                if (!valid_utf8(result)) {
                    throw ProtocolError("JSON string is not valid UTF-8");
                }
                return result;
            }
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
                std::uint32_t codepoint = hex_quad();
                if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                    if (take() != '\\' || take() != 'u') {
                        throw ProtocolError(
                            "JSON high surrogate lacks a low surrogate");
                    }
                    const auto low = hex_quad();
                    if (low < 0xdc00U || low > 0xdfffU) {
                        throw ProtocolError("invalid JSON surrogate pair");
                    }
                    codepoint = 0x10000U +
                        ((codepoint - 0xd800U) << 10U) +
                        (low - 0xdc00U);
                } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
                    throw ProtocolError("unpaired JSON low surrogate");
                }
                append_utf8(result, codepoint);
                break;
            }
            default:
                throw ProtocolError("unsupported JSON escape");
            }
        }
        throw ProtocolError("unterminated JSON string");
    }

    Json parse_number() {
        const auto begin = position_;
        if (input_[position_] == '-') {
            ++position_;
            if (position_ == input_.size()) {
                throw ProtocolError("truncated JSON number");
            }
        }
        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() && is_digit(input_[position_])) {
                throw ProtocolError("JSON number has a leading zero");
            }
        } else if (input_[position_] >= '1' && input_[position_] <= '9') {
            while (position_ < input_.size() && is_digit(input_[position_])) {
                ++position_;
            }
        } else {
            throw ProtocolError("invalid JSON number");
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const auto digits = position_;
            while (position_ < input_.size() && is_digit(input_[position_])) {
                ++position_;
            }
            if (position_ == digits) {
                throw ProtocolError("JSON fraction has no digits");
            }
        }
        if (position_ < input_.size() &&
            (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() &&
                (input_[position_] == '+' || input_[position_] == '-')) {
                ++position_;
            }
            const auto digits = position_;
            while (position_ < input_.size() && is_digit(input_[position_])) {
                ++position_;
            }
            if (position_ == digits) {
                throw ProtocolError("JSON exponent has no digits");
            }
        }
        return Json{JsonNumber{std::string(input_.substr(
            begin, position_ - begin))}};
    }

    Json parse_array(std::size_t depth) {
        expect('[');
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
            if (delimiter == ']') {
                return Json{std::move(result)};
            }
            if (delimiter != ',') {
                throw ProtocolError("JSON array lacks a comma");
            }
        }
    }

    Json parse_object(std::size_t depth) {
        expect('{');
        Json::Object result;
        skip_space();
        if (position_ < input_.size() && input_[position_] == '}') {
            ++position_;
            return Json{std::move(result)};
        }
        while (true) {
            skip_space();
            if (position_ == input_.size() || input_[position_] != '"') {
                throw ProtocolError("JSON object key must be a string");
            }
            auto key = parse_string();
            skip_space();
            expect(':');
            skip_space();
            auto value = parse_value(depth + 1);
            if (!result.emplace(std::move(key), std::move(value)).second) {
                throw ProtocolError("duplicate JSON object member");
            }
            skip_space();
            const char delimiter = take();
            if (delimiter == '}') {
                return Json{std::move(result)};
            }
            if (delimiter != ',') {
                throw ProtocolError("JSON object lacks a comma");
            }
        }
    }

    Json parse_value(std::size_t depth) {
        if (depth > 128U) {
            throw ProtocolError("JSON nesting exceeds protocol limit");
        }
        if (position_ == input_.size()) {
            throw ProtocolError("missing JSON value");
        }
        const char ch = input_[position_];
        if (ch == '{') return parse_object(depth);
        if (ch == '[') return parse_array(depth);
        if (ch == '"') return Json{parse_string()};
        if (ch == '-' || is_digit(ch)) return parse_number();
        const auto literal = [&](std::string_view token, Json value) {
            if (input_.substr(position_, token.size()) != token) {
                throw ProtocolError("invalid JSON literal");
            }
            position_ += token.size();
            return value;
        };
        if (ch == 't') return literal("true", Json{true});
        if (ch == 'f') return literal("false", Json{false});
        if (ch == 'n') return literal("null", Json{nullptr});
        throw ProtocolError("invalid JSON value");
    }

    std::string_view input_;
    std::size_t position_ = 0;
};

std::string json_escape(std::string_view input) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string output{"\""};
    for (const unsigned char ch : input) {
        switch (ch) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (ch < 0x20U) {
                output += "\\u00";
                output.push_back(hex[(ch >> 4U) & 0x0fU]);
                output.push_back(hex[ch & 0x0fU]);
            } else {
                output.push_back(static_cast<char>(ch));
            }
        }
    }
    output.push_back('"');
    return output;
}

std::string serialize_json(const Json &value) {
    return std::visit(
        [&](const auto &typed) -> std::string {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                return "null";
            } else if constexpr (std::is_same_v<T, bool>) {
                return typed ? "true" : "false";
            } else if constexpr (std::is_same_v<T, std::string>) {
                return json_escape(typed);
            } else if constexpr (std::is_same_v<T, JsonNumber>) {
                return typed.token;
            } else if constexpr (std::is_same_v<T, Json::Array>) {
                std::string result{"["};
                for (std::size_t index = 0; index < typed.size(); ++index) {
                    if (index != 0) result.push_back(',');
                    result += serialize_json(typed[index]);
                }
                result.push_back(']');
                return result;
            } else {
                std::string result{"{"};
                bool first = true;
                for (const auto &[key, item] : typed) {
                    if (!first) result.push_back(',');
                    first = false;
                    result += json_escape(key);
                    result.push_back(':');
                    result += serialize_json(item);
                }
                result.push_back('}');
                return result;
            }
        },
        value.value);
}

const Json::Object &object(const Json &value, std::string_view context) {
    const auto result = std::get_if<Json::Object>(&value.value);
    if (!result) {
        throw ProtocolError(std::string(context) + " must be an object");
    }
    return *result;
}

const Json::Array &array(const Json &value, std::string_view context) {
    const auto result = std::get_if<Json::Array>(&value.value);
    if (!result) {
        throw ProtocolError(std::string(context) + " must be an array");
    }
    return *result;
}

const std::string &string_value(const Json &value,
                                std::string_view context,
                                bool allow_empty = false) {
    const auto result = std::get_if<std::string>(&value.value);
    if (!result || (!allow_empty && result->empty()) ||
        !baseline::detail::canonical_text(*result, true)) {
        throw ProtocolError(std::string(context) +
                            " must be nonempty single-line UTF-8 text");
    }
    return *result;
}

bool bool_value(const Json &value, std::string_view context) {
    const auto result = std::get_if<bool>(&value.value);
    if (!result) {
        throw ProtocolError(std::string(context) + " must be a JSON boolean");
    }
    return *result;
}

const Json &member(const Json::Object &value, std::string_view key,
                   std::string_view context) {
    const auto found = value.find(key);
    if (found == value.end()) {
        throw ProtocolError(std::string(context) + " lacks required member " +
                            std::string(key));
    }
    return found->second;
}

const Json *optional_member(const Json::Object &value, std::string_view key) {
    const auto found = value.find(key);
    return found == value.end() ? nullptr : &found->second;
}

void exact_members(const Json::Object &value,
                   std::initializer_list<std::string_view> required,
                   std::initializer_list<std::string_view> optional,
                   std::string_view context) {
    std::set<std::string_view> allowed;
    for (const auto key : required) {
        allowed.insert(key);
        if (!value.contains(key)) {
            throw ProtocolError(std::string(context) +
                                " lacks required member " + std::string(key));
        }
    }
    allowed.insert(optional.begin(), optional.end());
    for (const auto &[key, unused] : value) {
        (void)unused;
        if (!allowed.contains(key)) {
            throw ProtocolError(std::string(context) +
                                " has unknown member " + key);
        }
    }
}

bool canonical_decimal(std::string_view value, bool allow_negative) {
    std::size_t position = 0;
    bool negative = false;
    if (allow_negative && !value.empty() && value.front() == '-') {
        position = 1;
        negative = true;
    }
    if (position == value.size()) return false;
    if (value[position] == '0') {
        return !negative && position + 1 == value.size();
    }
    if (value[position] < '1' || value[position] > '9') return false;
    for (++position; position < value.size(); ++position) {
        if (value[position] < '0' || value[position] > '9') return false;
    }
    return true;
}

std::int64_t int64_value(const Json &value, std::string_view context) {
    const auto &token = string_value(value, context);
    if (!canonical_decimal(token, true)) {
        throw ProtocolError(std::string(context) +
                            " must be a canonical decimal int64 string");
    }
    std::int64_t result = 0;
    const auto parsed = std::from_chars(token.data(),
                                        token.data() + token.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) {
        throw ProtocolError(std::string(context) + " is outside int64 range");
    }
    return result;
}

std::uint64_t uint64_value(const Json &value, std::string_view context) {
    const auto &token = string_value(value, context);
    if (!canonical_decimal(token, false)) {
        throw ProtocolError(std::string(context) +
                            " must be a canonical decimal uint64 string");
    }
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(token.data(),
                                        token.data() + token.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) {
        throw ProtocolError(std::string(context) + " is outside uint64 range");
    }
    return result;
}

double float64_value(const Json &value, std::string_view context) {
    const auto &token = string_value(value, context);
    if (token.size() != 16U ||
        !std::all_of(token.begin(), token.end(), [](char ch) {
            return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        })) {
        throw ProtocolError(std::string(context) +
                            " must be 16 lowercase IEEE-754 hex digits");
    }
    std::uint64_t bits = 0;
    const auto parsed = std::from_chars(token.data(),
                                        token.data() + token.size(), bits, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) {
        throw ProtocolError(std::string(context) +
                            " has invalid IEEE-754 bits");
    }
    if (((bits >> 52U) & 0x7ffU) == 0x7ffU) {
        throw ProtocolError(std::string(context) +
                            " must be finite; nonfinite JSON facts are rejected");
    }
    return std::bit_cast<double>(bits);
}

void require_sha256(std::string_view value, std::string_view context) {
    if (!baseline::is_sha256_reference(value)) {
        throw ProtocolError(std::string(context) +
                            " must be a lowercase sha256: reference");
    }
}

std::uint64_t regular_file_size(const std::filesystem::path &path,
                                std::string_view context) {
    std::error_code error;
    const auto status = std::filesystem::status(path, error);
    if (error || !std::filesystem::is_regular_file(status)) {
        throw baseline::ContractError(
            std::string(context) + " must be an existing regular file: " +
            path.string());
    }
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > std::numeric_limits<std::uint64_t>::max()) {
        throw baseline::ContractError(
            std::string(context) + " has no exact uint64 byte size: " +
            path.string());
    }
    return static_cast<std::uint64_t>(size);
}

std::string read_regular_file_exact(const std::filesystem::path &path,
                                    std::uint64_t expected_byte_count,
                                    std::string_view context) {
    if (regular_file_size(path, context) != expected_byte_count ||
        expected_byte_count > std::numeric_limits<std::size_t>::max() ||
        expected_byte_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max())) {
        throw baseline::ContractError(
            std::string(context) + " byte count disagrees with its receipt");
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw baseline::ContractError("cannot open " + std::string(context) +
                                      ": " + path.string());
    }
    std::string result(static_cast<std::size_t>(expected_byte_count), '\0');
    stream.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (stream.gcount() != static_cast<std::streamsize>(result.size())) {
        throw baseline::ContractError("cannot read exact " +
                                      std::string(context) + " bytes: " +
                                      path.string());
    }
    char extra = '\0';
    if (stream.get(extra) || !stream.eof()) {
        throw baseline::ContractError(
            std::string(context) + " changed while it was being read");
    }
    return result;
}

std::string read_small_receipt(const std::filesystem::path &path,
                               std::string_view context) {
    constexpr std::uint64_t maximum_receipt_bytes = 4096U;
    const auto size = regular_file_size(path, context);
    if (size == 0 || size > maximum_receipt_bytes) {
        throw baseline::ContractError(
            std::string(context) + " exceeds canonical receipt bounds");
    }
    return read_regular_file_exact(path, size, context);
}

struct StreamedFileBinding {
    std::string content_sha256;
    std::uint64_t byte_count = 0;
};

StreamedFileBinding stream_regular_file_binding(
    const std::filesystem::path &path, std::uint64_t expected_byte_count,
    std::string_view context) {
    if (regular_file_size(path, context) != expected_byte_count) {
        throw baseline::ContractError(
            std::string(context) + " byte count disagrees with its manifest");
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw baseline::ContractError("cannot open " + std::string(context) +
                                      ": " + path.string());
    }
    citlali::utils::Sha256 digest;
    std::array<char, 8192> buffer{};
    std::uint64_t count = 0;
    while (count < expected_byte_count) {
        const auto remaining = expected_byte_count - count;
        const auto requested = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, buffer.size()));
        stream.read(buffer.data(), static_cast<std::streamsize>(requested));
        const auto chunk = stream.gcount();
        if (chunk != static_cast<std::streamsize>(requested)) {
            throw baseline::ContractError(
                std::string(context) + " changed while it was being read");
        }
        count += static_cast<std::uint64_t>(chunk);
        digest.update(
            reinterpret_cast<const std::uint8_t *>(buffer.data()),
            static_cast<std::size_t>(chunk));
    }
    char extra = '\0';
    if (stream.get(extra) || !stream.eof()) {
        throw baseline::ContractError(
            std::string(context) + " changed while it was being read");
    }
    return {"sha256:" + digest.finish(), count};
}

std::filesystem::path receipt_path(const std::filesystem::path &artifact) {
    return std::filesystem::path(artifact.string() + ".sha256");
}

baseline::ObservationIdentity parse_observation(
    const Json &value, std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record, {"observation", "subobservation", "scan"}, {},
                  context);
    return {int64_value(member(record, "observation", context),
                        std::string(context) + ".observation"),
            int64_value(member(record, "subobservation", context),
                        std::string(context) + ".subobservation"),
            int64_value(member(record, "scan", context),
                        std::string(context) + ".scan")};
}

observation::IssuanceEnvelope parse_envelope(
    const Json &value, std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record,
                  {"occurrence", "event_reference", "software_revision",
                   "configuration_reference", "event_time_utc"},
                  {}, context);
    return {string_value(member(record, "occurrence", context),
                         std::string(context) + ".occurrence"),
            string_value(member(record, "event_reference", context),
                         std::string(context) + ".event_reference"),
            string_value(member(record, "software_revision", context),
                         std::string(context) + ".software_revision"),
            string_value(member(record, "configuration_reference", context),
                         std::string(context) + ".configuration_reference"),
            string_value(member(record, "event_time_utc", context),
                         std::string(context) + ".event_time_utc")};
}

observation::ArtifactIdentity parse_artifact_identity(
    const Json &value, std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record,
                  {"schema", "occurrence", "semantic_sha256",
                   "envelope_sha256"},
                  {}, context);
    observation::ArtifactIdentity result{
        string_value(member(record, "schema", context),
                     std::string(context) + ".schema"),
        string_value(member(record, "occurrence", context),
                     std::string(context) + ".occurrence"),
        string_value(member(record, "semantic_sha256", context),
                     std::string(context) + ".semantic_sha256"),
        string_value(member(record, "envelope_sha256", context),
                     std::string(context) + ".envelope_sha256")};
    require_sha256(result.semantic_sha256,
                   std::string(context) + ".semantic_sha256");
    require_sha256(result.envelope_sha256,
                   std::string(context) + ".envelope_sha256");
    return result;
}

observation::VerifiedBaselineReference parse_baseline_reference(
    const Json &value, std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record,
                  {"artifact", "profile", "descriptor_sha256",
                   "transport_scope", "transport_sha256", "byte_count",
                   "receipt_sha256", "receipt_byte_count"},
                  {}, context);
    observation::VerifiedBaselineReference result;
    result.artifact = parse_artifact_identity(
        member(record, "artifact", context),
        std::string(context) + ".artifact");
    result.profile = string_value(member(record, "profile", context),
                                  std::string(context) + ".profile");
    result.descriptor_sha256 = string_value(
        member(record, "descriptor_sha256", context),
        std::string(context) + ".descriptor_sha256");
    result.transport_scope = string_value(
        member(record, "transport_scope", context),
        std::string(context) + ".transport_scope");
    result.transport_sha256 = string_value(
        member(record, "transport_sha256", context),
        std::string(context) + ".transport_sha256");
    result.byte_count = uint64_value(member(record, "byte_count", context),
                                     std::string(context) + ".byte_count");
    result.receipt_sha256 = string_value(
        member(record, "receipt_sha256", context),
        std::string(context) + ".receipt_sha256");
    result.receipt_byte_count = uint64_value(
        member(record, "receipt_byte_count", context),
        std::string(context) + ".receipt_byte_count");
    for (const auto &[digest, label] :
         std::array<std::pair<std::string_view, std::string_view>, 3>{
             {{result.descriptor_sha256, "descriptor_sha256"},
              {result.transport_sha256, "transport_sha256"},
              {result.receipt_sha256, "receipt_sha256"}}}) {
        require_sha256(digest, std::string(context) + "." +
                                   std::string(label));
    }
    return result;
}

observation::SourceArtifact parse_source_artifact(
    const Json &value, std::string_view context,
    std::string_view expected_role) {
    const auto &record = object(value, context);
    exact_members(record,
                  {"source_key", "network", "interface",
                   "channel_count", "diagnostic_locator", "content_sha256",
                   "byte_count", "header_observation"},
                  {}, context);
    observation::SourceArtifact result;
    result.source_key = int64_value(member(record, "source_key", context),
                                    std::string(context) + ".source_key");
    result.role = std::string(expected_role);
    result.network = int64_value(member(record, "network", context),
                                 std::string(context) + ".network");
    result.interface_name = string_value(
        member(record, "interface", context),
        std::string(context) + ".interface");
    result.channel_count = int64_value(
        member(record, "channel_count", context),
        std::string(context) + ".channel_count");
    result.diagnostic_locator = string_value(
        member(record, "diagnostic_locator", context),
        std::string(context) + ".diagnostic_locator");
    result.content_sha256 = string_value(
        member(record, "content_sha256", context),
        std::string(context) + ".content_sha256");
    require_sha256(result.content_sha256,
                   std::string(context) + ".content_sha256");
    result.byte_count = uint64_value(member(record, "byte_count", context),
                                     std::string(context) + ".byte_count");
    result.header_observation = parse_observation(
        member(record, "header_observation", context),
        std::string(context) + ".header_observation");
    return result;
}

observation::TargetInput parse_target_input(
    const Json &value, std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record,
                  {"input_key", "network", "interface", "channel_count",
                   "raw_source", "kmp_source"},
                  {}, context);
    observation::TargetInput result;
    result.input_key = int64_value(member(record, "input_key", context),
                                   std::string(context) + ".input_key");
    result.network = int64_value(member(record, "network", context),
                                 std::string(context) + ".network");
    result.interface_name = string_value(
        member(record, "interface", context),
        std::string(context) + ".interface");
    result.channel_count = int64_value(
        member(record, "channel_count", context),
        std::string(context) + ".channel_count");
    result.raw_source = parse_source_artifact(
        member(record, "raw_source", context),
        std::string(context) + ".raw_source", "raw");
    result.kmp_source = parse_source_artifact(
        member(record, "kmp_source", context),
        std::string(context) + ".kmp_source", "kmp");
    return result;
}

observation::TargetManifest parse_target_facts(
    const Json &value, std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record,
                  {"envelope", "observation", "inputs", "rows",
                   "target_source_sequence", "target_application_sequence"},
                  {}, context);
    observation::TargetManifest result;
    result.envelope = parse_envelope(member(record, "envelope", context),
                                     std::string(context) + ".envelope");
    result.observation = parse_observation(
        member(record, "observation", context),
        std::string(context) + ".observation");
    const auto &inputs = array(member(record, "inputs", context),
                               std::string(context) + ".inputs");
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        result.inputs.push_back(parse_target_input(
            inputs[index], std::string(context) + ".inputs[" +
                               std::to_string(index) + "]"));
    }
    const auto &rows = array(member(record, "rows", context),
                             std::string(context) + ".rows");
    std::optional<bool> includes_kids_flag;
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto row_context = std::string(context) + ".rows[" +
            std::to_string(index) + "]";
        const auto &row_record = object(rows[index], row_context);
        exact_members(row_record,
                      {"row_key", "input_key", "kmp_source_key",
                       "kmp_row_index", "array", "network", "channel",
                       "fields"},
                      {}, row_context);
        observation::TargetRow row;
        row.row_key = int64_value(member(row_record, "row_key", row_context),
                                  row_context + ".row_key");
        row.input_key = int64_value(
            member(row_record, "input_key", row_context),
            row_context + ".input_key");
        row.kmp_source_key = int64_value(
            member(row_record, "kmp_source_key", row_context),
            row_context + ".kmp_source_key");
        row.kmp_row_index = int64_value(
            member(row_record, "kmp_row_index", row_context),
            row_context + ".kmp_row_index");
        row.array = int64_value(member(row_record, "array", row_context),
                                row_context + ".array");
        row.network = int64_value(
            member(row_record, "network", row_context),
            row_context + ".network");
        row.channel = int64_value(
            member(row_record, "channel", row_context),
            row_context + ".channel");
        const auto &row_fields = object(member(row_record, "fields", row_context),
                                        row_context + ".fields");
        exact_members(row_fields,
                      {"kids_fr", "kids_f_out", "kids_Qr"},
                      {"kids_flag"}, row_context + ".fields");
        const bool row_has_kids_flag = row_fields.contains("kids_flag");
        if (includes_kids_flag &&
            *includes_kids_flag != row_has_kids_flag) {
            throw ProtocolError(
                std::string(context) +
                ".rows must use one artifact-level kids_flag shape");
        }
        includes_kids_flag = row_has_kids_flag;
        row.fields.emplace(
            "kids_fr", float64_value(
                member(row_fields, "kids_fr", row_context + ".fields"),
                row_context + ".fields.kids_fr"));
        row.fields.emplace(
            "kids_f_out", float64_value(
                member(row_fields, "kids_f_out", row_context + ".fields"),
                row_context + ".fields.kids_f_out"));
        row.fields.emplace(
            "kids_Qr", float64_value(
                member(row_fields, "kids_Qr", row_context + ".fields"),
                row_context + ".fields.kids_Qr"));
        if (row_has_kids_flag) {
            row.fields.emplace(
                "kids_flag", int64_value(
                    member(row_fields, "kids_flag",
                           row_context + ".fields"),
                    row_context + ".fields.kids_flag"));
        }
        row.matching_frequency_hz =
            std::get<double>(row.fields.at("kids_fr"));
        row.output_tone_frequency_hz =
            std::get<double>(row.fields.at("kids_f_out"));
        result.rows.push_back(std::move(row));
    }
    result.registered_fields =
        observation::canonical_required_target_fields_v1();
    if (includes_kids_flag.value_or(false)) {
        result.registered_fields.push_back(
            observation::canonical_target_fields_v1().back());
    }
    const auto parse_sequence = [&](std::string_view key,
                                    std::vector<std::int64_t> &output) {
        const auto &items = array(member(record, key, context),
                                  std::string(context) + "." +
                                      std::string(key));
        for (std::size_t index = 0; index < items.size(); ++index) {
            output.push_back(int64_value(
                items[index], std::string(context) + "." +
                                  std::string(key) + "[" +
                                  std::to_string(index) + "]"));
        }
    };
    parse_sequence("target_source_sequence", result.target_source_sequence);
    parse_sequence("target_application_sequence",
                   result.target_application_sequence);
    return result;
}

observation::MatcherEvidence parse_matcher_evidence(
    const Json &value, std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record,
                  {"matcher_run_occurrence", "implementation_revision",
                   "configuration_reference", "method", "backend"},
                  {}, context);
    observation::MatcherEvidence result;
    result.matcher_run_occurrence = string_value(
        member(record, "matcher_run_occurrence", context),
        std::string(context) + ".matcher_run_occurrence");
    result.implementation_revision = string_value(
        member(record, "implementation_revision", context),
        std::string(context) + ".implementation_revision");
    result.configuration_reference = string_value(
        member(record, "configuration_reference", context),
        std::string(context) + ".configuration_reference");
    result.method = string_value(member(record, "method", context),
                                 std::string(context) + ".method");
    result.backend = string_value(member(record, "backend", context),
                                  std::string(context) + ".backend");
    return result;
}

struct ScopedLocalFact {
    std::string occurrence;
    std::int64_t local_key = 0;
};

ScopedLocalFact parse_scoped_local_fact(
    const Json &value, std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record, {"occurrence", "local_key"}, {}, context);
    return {
        string_value(member(record, "occurrence", context),
                     std::string(context) + ".occurrence"),
        int64_value(member(record, "local_key", context),
                    std::string(context) + ".local_key")};
}

observation::RowReference materialize_row_reference(
    const ScopedLocalFact &fact,
    const observation::ArtifactIdentity &identity,
    std::string_view context) {
    if (fact.occurrence != identity.occurrence) {
        throw baseline::ContractError(
            std::string(context) +
            " occurrence does not name its independently verified parent");
    }
    return observation::row_reference(identity, fact.local_key);
}

observation::MatchRelation parse_relation_facts(
    const Json &value, std::string_view context,
    const observation::VerifiedBaselineDescriptor &baseline_descriptor,
    const observation::TargetManifest &target) {
    const auto &record = object(value, context);
    exact_members(record,
                  {"envelope", "matcher", "network_evidence", "pairs",
                   "target_dispositions", "seed_dispositions",
                   "seed_source_sequence"},
                  {}, context);
    observation::MatchRelation result;
    result.envelope = parse_envelope(member(record, "envelope", context),
                                     std::string(context) + ".envelope");
    const auto target_identity = observation::artifact_identity(target);
    const auto baseline_identity =
        observation::artifact_identity(baseline_descriptor);
    result.baseline_parent =
        observation::baseline_reference(baseline_descriptor);
    result.target_parent = target_identity;
    result.matcher = parse_matcher_evidence(
        member(record, "matcher", context),
        std::string(context) + ".matcher");

    const auto &network_evidence = array(
        member(record, "network_evidence", context),
        std::string(context) + ".network_evidence");
    for (std::size_t index = 0; index < network_evidence.size(); ++index) {
        const auto item_context = std::string(context) +
            ".network_evidence[" + std::to_string(index) + "]";
        const auto &item = object(network_evidence[index], item_context);
        exact_members(item,
                      {"network", "frequency_shift_hz", "gate_hz",
                       "quality_factor"},
                      {}, item_context);
        observation::NetworkMatchEvidence evidence;
        evidence.network = int64_value(member(item, "network", item_context),
                                       item_context + ".network");
        evidence.frequency_shift_hz = float64_value(
            member(item, "frequency_shift_hz", item_context),
            item_context + ".frequency_shift_hz");
        evidence.gate_hz = float64_value(
            member(item, "gate_hz", item_context), item_context + ".gate_hz");
        evidence.quality_factor = float64_value(
            member(item, "quality_factor", item_context),
            item_context + ".quality_factor");
        result.network_evidence.push_back(std::move(evidence));
    }

    const auto &pairs = array(member(record, "pairs", context),
                              std::string(context) + ".pairs");
    for (std::size_t index = 0; index < pairs.size(); ++index) {
        const auto item_context = std::string(context) + ".pairs[" +
            std::to_string(index) + "]";
        const auto &item = object(pairs[index], item_context);
        exact_members(item,
                      {"pair_key", "target", "seed", "separation_hz",
                       "is_good_match"},
                      {}, item_context);
        const auto target_fact = parse_scoped_local_fact(
            member(item, "target", item_context), item_context + ".target");
        const auto seed_fact = parse_scoped_local_fact(
            member(item, "seed", item_context), item_context + ".seed");
        result.pairs.push_back({
            int64_value(member(item, "pair_key", item_context),
                        item_context + ".pair_key"),
            materialize_row_reference(target_fact, target_identity,
                                      item_context + ".target"),
            materialize_row_reference(seed_fact, baseline_identity,
                                      item_context + ".seed"),
            float64_value(member(item, "separation_hz", item_context),
                          item_context + ".separation_hz"),
            bool_value(member(item, "is_good_match", item_context),
                       item_context + ".is_good_match")});
    }

    const auto parse_dispositions = [&](std::string_view key,
                                        bool target_side) {
        std::vector<observation::EndpointDisposition> output;
        const auto &items = array(member(record, key, context),
                                  std::string(context) + "." +
                                      std::string(key));
        for (std::size_t index = 0; index < items.size(); ++index) {
            const auto item_context = std::string(context) + "." +
                std::string(key) + "[" + std::to_string(index) + "]";
            const auto &item = object(items[index], item_context);
            exact_members(item,
                          {"disposition_key", "endpoint", "state",
                           "pair_keys", "reason"},
                          {}, item_context);
            const auto &state = string_value(
                member(item, "state", item_context), item_context + ".state");
            observation::EndpointDispositionState parsed_state;
            if (state == "matched") {
                parsed_state = observation::EndpointDispositionState::matched;
            } else if (state == "unmatched" && target_side) {
                parsed_state = observation::EndpointDispositionState::unmatched;
            } else if (state == "unused" && !target_side) {
                parsed_state = observation::EndpointDispositionState::unused;
            } else {
                throw ProtocolError(item_context +
                                    ".state is invalid for endpoint side");
            }
            observation::EndpointDisposition disposition;
            disposition.disposition_key = int64_value(
                member(item, "disposition_key", item_context),
                item_context + ".disposition_key");
            const auto endpoint = parse_scoped_local_fact(
                member(item, "endpoint", item_context),
                item_context + ".endpoint");
            disposition.endpoint = materialize_row_reference(
                endpoint, target_side ? target_identity : baseline_identity,
                item_context + ".endpoint");
            disposition.state = parsed_state;
            const auto &pair_keys = array(member(item, "pair_keys", item_context),
                                          item_context + ".pair_keys");
            for (std::size_t pair_index = 0; pair_index < pair_keys.size();
                 ++pair_index) {
                disposition.pair_keys.push_back(int64_value(
                    pair_keys[pair_index], item_context + ".pair_keys[" +
                                               std::to_string(pair_index) +
                                               "]"));
            }
            disposition.reason = string_value(
                member(item, "reason", item_context), item_context + ".reason");
            output.push_back(std::move(disposition));
        }
        return output;
    };
    result.target_dispositions = parse_dispositions("target_dispositions", true);
    result.seed_dispositions = parse_dispositions("seed_dispositions", false);
    const auto &sequence = array(member(record, "seed_source_sequence", context),
                                 std::string(context) +
                                     ".seed_source_sequence");
    for (std::size_t index = 0; index < sequence.size(); ++index) {
        result.seed_source_sequence.push_back(int64_value(
            sequence[index], std::string(context) +
                                 ".seed_source_sequence[" +
                                 std::to_string(index) + "]"));
    }
    return result;
}

std::vector<observation::MatchedOutputFieldSource>
parse_field_source_selections(
    const Json &value, std::string_view context,
    const observation::VerifiedBaselineDescriptor &baseline_descriptor,
    const observation::TargetManifest &target,
    const observation::MatchRelation &relation) {
    std::vector<observation::MatchedOutputFieldSource> result;
    const auto output_fields = observation::canonical_output_field_contracts_v1(
        baseline_descriptor, target);
    std::set<std::string> selectable_fields;
    for (const auto &contract : output_fields) {
        if (contract.authorized_operation ==
            observation::TransformationOperation::
                copy_baseline_when_matched_null_when_unmatched) {
            selectable_fields.insert(contract.field.name);
        }
    }
    std::map<std::int64_t, std::set<std::int64_t>> pairs_for_target;
    for (const auto &disposition : relation.target_dispositions) {
        pairs_for_target.emplace(
            disposition.endpoint.local_key,
            std::set<std::int64_t>(disposition.pair_keys.begin(),
                                   disposition.pair_keys.end()));
    }
    std::set<std::int64_t> target_keys;
    for (const auto &row : target.rows) {
        target_keys.insert(row.row_key);
    }
    std::set<std::int64_t> seen_targets;
    const auto &items = array(value, context);
    for (std::size_t index = 0; index < items.size(); ++index) {
        const auto item_context = std::string(context) + "[" +
            std::to_string(index) + "]";
        const auto &item = object(items[index], item_context);
        exact_members(item,
                      {"target", "field_overrides"},
                      {"default_source_pair"}, item_context);
        const auto target_fact = parse_scoped_local_fact(
            member(item, "target", item_context), item_context + ".target");
        if (target_fact.occurrence != target.envelope.occurrence ||
            !target_keys.contains(target_fact.local_key) ||
            !seen_targets.insert(target_fact.local_key).second) {
            throw baseline::ContractError(
                "field-source target fact is foreign, unknown, or duplicate");
        }
        const auto target_pairs = pairs_for_target.find(target_fact.local_key);
        if (target_pairs == pairs_for_target.end()) {
            throw baseline::ContractError(
                "field-source target lacks a validated disposition");
        }
        std::optional<std::int64_t> default_pair;
        if (const auto default_value =
                optional_member(item, "default_source_pair")) {
            if (!std::holds_alternative<std::nullptr_t>(default_value->value)) {
                const auto fact = parse_scoped_local_fact(
                    *default_value,
                    item_context + ".default_source_pair");
                if (fact.occurrence != relation.envelope.occurrence) {
                    throw baseline::ContractError(
                        "default field-source pair has a foreign relation occurrence");
                }
                default_pair = fact.local_key;
            }
        }
        const bool matched = !target_pairs->second.empty();
        if ((matched &&
             (!default_pair ||
              !target_pairs->second.contains(*default_pair))) ||
            (!matched && default_pair)) {
            throw baseline::ContractError(
                "default field-source pair does not match the target disposition");
        }

        std::map<std::string, std::int64_t> overrides;
        const auto &override_items = array(
            member(item, "field_overrides", item_context),
            item_context + ".field_overrides");
        for (std::size_t override_index = 0;
             override_index < override_items.size(); ++override_index) {
            const auto override_context = item_context +
                ".field_overrides[" + std::to_string(override_index) + "]";
            const auto &override_record = object(
                override_items[override_index], override_context);
            exact_members(override_record,
                          {"field_name", "source_pair"}, {},
                          override_context);
            const auto &field_name = string_value(
                member(override_record, "field_name", override_context),
                override_context + ".field_name");
            const auto pair_fact = parse_scoped_local_fact(
                member(override_record, "source_pair", override_context),
                override_context + ".source_pair");
            if (!selectable_fields.contains(field_name) ||
                pair_fact.occurrence != relation.envelope.occurrence ||
                !target_pairs->second.contains(pair_fact.local_key) ||
                !overrides.emplace(field_name, pair_fact.local_key).second) {
                throw baseline::ContractError(
                    "field-source override is unknown, foreign, nonmember, or duplicate");
            }
        }
        if (!matched && !overrides.empty()) {
            throw baseline::ContractError(
                "unmatched field-source target cannot carry overrides");
        }
        for (const auto &field_name : selectable_fields) {
            const auto override_pair = overrides.find(field_name);
            result.push_back({
                target_fact.local_key, field_name,
                override_pair == overrides.end()
                    ? default_pair
                    : std::optional<std::int64_t>{override_pair->second}});
        }
    }
    if (seen_targets != target_keys) {
        throw baseline::ContractError(
            "field-source selections do not cover every target occurrence row");
    }
    return result;
}

struct ExpectedTransport {
    std::string scope;
    std::string envelope_sha256;
    std::string byte_sha256;
    std::uint64_t byte_count = 0;
};

ExpectedTransport parse_expected_transport(const Json &value,
                                           std::string_view context) {
    const auto &record = object(value, context);
    exact_members(record,
                  {"scope", "envelope_sha256", "byte_sha256", "byte_count"},
                  {}, context);
    ExpectedTransport result{
        string_value(member(record, "scope", context),
                     std::string(context) + ".scope"),
        string_value(member(record, "envelope_sha256", context),
                     std::string(context) + ".envelope_sha256"),
        string_value(member(record, "byte_sha256", context),
                     std::string(context) + ".byte_sha256"),
        uint64_value(member(record, "byte_count", context),
                     std::string(context) + ".byte_count")};
    require_sha256(result.envelope_sha256,
                   std::string(context) + ".envelope_sha256");
    require_sha256(result.byte_sha256,
                   std::string(context) + ".byte_sha256");
    return result;
}

Json artifact_identity_json(const observation::ArtifactIdentity &identity) {
    return Json::Object{{"schema", identity.schema},
                        {"occurrence", identity.occurrence},
                        {"semantic_sha256", identity.semantic_sha256},
                        {"envelope_sha256", identity.envelope_sha256}};
}

Json baseline_reference_json(
    const observation::VerifiedBaselineReference &reference) {
    return Json::Object{
        {"artifact", artifact_identity_json(reference.artifact)},
        {"profile", reference.profile},
        {"descriptor_sha256", reference.descriptor_sha256},
        {"transport_scope", reference.transport_scope},
        {"transport_sha256", reference.transport_sha256},
        {"byte_count", std::to_string(reference.byte_count)},
        {"receipt_sha256", reference.receipt_sha256},
        {"receipt_byte_count", std::to_string(reference.receipt_byte_count)}};
}

Json observation_identity_json(
    const baseline::ObservationIdentity &identity) {
    return Json::Object{
        {"observation", std::to_string(identity.observation)},
        {"subobservation", std::to_string(identity.subobservation)},
        {"scan", std::to_string(identity.scan)}};
}

Json issuance_envelope_json(
    const observation::IssuanceEnvelope &envelope) {
    return Json::Object{
        {"occurrence", envelope.occurrence},
        {"event_reference", envelope.event_reference},
        {"software_revision", envelope.software_revision},
        {"configuration_reference", envelope.configuration_reference},
        {"event_time_utc", envelope.event_time_utc}};
}

Json row_reference_json(const observation::RowReference &reference) {
    return Json::Object{
        {"artifact_schema", reference.artifact_schema},
        {"occurrence", reference.occurrence},
        {"envelope_sha256", reference.envelope_sha256},
        {"local_key", std::to_string(reference.local_key)}};
}

Json typed_field_json(const observation::TypedField &field) {
    return Json::Object{
        {"name", field.name},
        {"source_column",
         field.source_column ? Json{*field.source_column} : Json{nullptr}},
        {"datatype", std::string(baseline::value_type_token(field.type))},
        {"unit", field.unit},
        {"nullable", field.nullable},
        {"nonfinite",
         std::string(baseline::nonfinite_policy_token(field.nonfinite))},
        {"authority", field.authority},
        {"authority_reference", field.authority_reference},
        {"registry", field.registry},
        {"description", field.description},
        {"identity_role", field.identity_role}};
}

Json exact_value_json(const baseline::Value &value) {
    return std::visit(
        [](const auto &typed) -> Json {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, baseline::NullValue>) {
                return Json{nullptr};
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return Json{std::to_string(typed)};
            } else if constexpr (std::is_same_v<T, double>) {
                return Json{observation::canonical_binary64_payload(typed)};
            } else {
                return Json{typed};
            }
        },
        value);
}

Json int64_sequence_json(const std::vector<std::int64_t> &sequence) {
    Json::Array result;
    result.reserve(sequence.size());
    for (const auto value : sequence) {
        result.emplace_back(std::to_string(value));
    }
    return result;
}

Json source_artifact_json(const observation::SourceArtifact &source) {
    return Json::Object{
        {"source_key", std::to_string(source.source_key)},
        {"role", source.role},
        {"network", std::to_string(source.network)},
        {"interface", source.interface_name},
        {"channel_count", std::to_string(source.channel_count)},
        {"diagnostic_locator", source.diagnostic_locator},
        {"content_sha256", source.content_sha256},
        {"byte_count", std::to_string(source.byte_count)},
        {"header_observation",
         observation_identity_json(source.header_observation)}};
}

Json target_manifest_json(const observation::TargetManifest &target) {
    observation::validate(target);
    Json::Array inputs;
    for (const auto &input : target.inputs) {
        inputs.emplace_back(Json::Object{
            {"input_key", std::to_string(input.input_key)},
            {"network", std::to_string(input.network)},
            {"interface", input.interface_name},
            {"channel_count", std::to_string(input.channel_count)},
            {"raw_source", source_artifact_json(input.raw_source)},
            {"kmp_source", source_artifact_json(input.kmp_source)}});
    }
    Json::Array fields;
    std::map<std::string, baseline::ValueType> field_types;
    for (const auto &field : target.registered_fields) {
        fields.emplace_back(typed_field_json(field));
        field_types.emplace(field.name, field.type);
    }
    Json::Array rows;
    for (const auto &row : target.rows) {
        Json::Object values;
        for (const auto &[name, value] : row.fields) {
            (void)field_types.at(name);
            values.emplace(name, exact_value_json(value));
        }
        rows.emplace_back(Json::Object{
            {"row_key", std::to_string(row.row_key)},
            {"input_key", std::to_string(row.input_key)},
            {"kmp_source_key", std::to_string(row.kmp_source_key)},
            {"kmp_row_index", std::to_string(row.kmp_row_index)},
            {"matching_frequency_hz",
             observation::canonical_binary64_payload(
                 row.matching_frequency_hz)},
            {"output_tone_frequency_hz",
             observation::canonical_binary64_payload(
                 row.output_tone_frequency_hz)},
            {"array", std::to_string(row.array)},
            {"network", std::to_string(row.network)},
            {"channel", std::to_string(row.channel)},
            {"fields", std::move(values)}});
    }
    return Json::Object{
        {"schema_version", target.schema},
        {"contract_authority", target.contract_authority},
        {"observation_value_issuer", target.observation_value_issuer},
        {"envelope", issuance_envelope_json(target.envelope)},
        {"observation", observation_identity_json(target.observation)},
        {"inputs", std::move(inputs)},
        {"registered_fields", std::move(fields)},
        {"rows", std::move(rows)},
        {"target_source_sequence",
         int64_sequence_json(target.target_source_sequence)},
        {"target_application_sequence",
         int64_sequence_json(target.target_application_sequence)}};
}

Json matcher_evidence_json(
    const observation::MatcherEvidence &matcher) {
    return Json::Object{
        {"matcher_run_occurrence", matcher.matcher_run_occurrence},
        {"implementation_revision", matcher.implementation_revision},
        {"configuration_reference", matcher.configuration_reference},
        {"target_frequency_field", matcher.target_frequency_field},
        {"target_quality_factor_field",
         matcher.target_quality_factor_field},
        {"method", matcher.method},
        {"backend", matcher.backend}};
}

Json disposition_json(
    const observation::EndpointDisposition &disposition) {
    return Json::Object{
        {"disposition_key", std::to_string(disposition.disposition_key)},
        {"endpoint", row_reference_json(disposition.endpoint)},
        {"state", std::string(
                      observation::endpoint_disposition_token(
                          disposition.state))},
        {"pair_keys", int64_sequence_json(disposition.pair_keys)},
        {"reason", disposition.reason}};
}

Json match_relation_json(
    const observation::MatchRelation &relation,
    const observation::VerifiedBaselineDescriptor &baseline_descriptor,
    const observation::TargetManifest &target) {
    observation::validate(relation, baseline_descriptor, target);
    Json::Array network_evidence;
    for (const auto &evidence : relation.network_evidence) {
        network_evidence.emplace_back(Json::Object{
            {"network", std::to_string(evidence.network)},
            {"frequency_shift_hz",
             observation::canonical_binary64_payload(
                 evidence.frequency_shift_hz)},
            {"gate_hz",
             observation::canonical_binary64_payload(evidence.gate_hz)},
            {"quality_factor",
             observation::canonical_binary64_payload(
                 evidence.quality_factor)},
            {"quality_factor_field", evidence.quality_factor_field},
            {"quality_factor_authority_reference",
             evidence.quality_factor_authority_reference}});
    }
    Json::Array pairs;
    for (const auto &pair : relation.pairs) {
        pairs.emplace_back(Json::Object{
            {"pair_key", std::to_string(pair.pair_key)},
            {"target", row_reference_json(pair.target)},
            {"seed", row_reference_json(pair.seed)},
            {"separation_hz",
             observation::canonical_binary64_payload(pair.separation_hz)},
            {"is_good_match", pair.is_good_match}});
    }
    Json::Array target_dispositions;
    for (const auto &disposition : relation.target_dispositions) {
        target_dispositions.emplace_back(disposition_json(disposition));
    }
    Json::Array seed_dispositions;
    for (const auto &disposition : relation.seed_dispositions) {
        seed_dispositions.emplace_back(disposition_json(disposition));
    }
    return Json::Object{
        {"schema_version", relation.schema},
        {"contract_authority", relation.contract_authority},
        {"observation_value_issuer", relation.observation_value_issuer},
        {"mapping_domain", relation.mapping_domain},
        {"envelope", issuance_envelope_json(relation.envelope)},
        {"baseline_parent", baseline_reference_json(relation.baseline_parent)},
        {"target_parent", artifact_identity_json(relation.target_parent)},
        {"matcher", matcher_evidence_json(relation.matcher)},
        {"network_evidence", std::move(network_evidence)},
        {"pairs", std::move(pairs)},
        {"target_dispositions", std::move(target_dispositions)},
        {"seed_dispositions", std::move(seed_dispositions)},
        {"seed_source_sequence",
         int64_sequence_json(relation.seed_source_sequence)}};
}

Json matched_output_json(
    const observation::MatchedOutput &output,
    const observation::VerifiedBaselineDescriptor &baseline_descriptor,
    const observation::TargetManifest &target,
    const observation::MatchRelation &relation) {
    observation::validate(output, baseline_descriptor, target, relation);
    Json::Array registered_fields;
    std::map<std::string, baseline::ValueType> field_types;
    for (const auto &contract : output.registered_fields) {
        field_types.emplace(contract.field.name, contract.field.type);
        registered_fields.emplace_back(Json::Object{
            {"field", typed_field_json(contract.field)},
            {"authorized_operation",
             std::string(observation::transformation_operation_token(
                 contract.authorized_operation))},
            {"issuer_authority_reference",
             contract.issuer_authority_reference}});
    }
    Json::Array rows;
    for (const auto &row : output.rows) {
        Json::Object values;
        for (const auto &[name, value] : row.fields) {
            (void)field_types.at(name);
            values.emplace(name, exact_value_json(value));
        }
        Json::Array transformations;
        for (const auto &change : row.transformations) {
            transformations.emplace_back(Json::Object{
                {"field_name", change.field_name},
                {"operation",
                 std::string(observation::transformation_operation_token(
                     change.operation))},
                {"before", exact_value_json(change.before)},
                {"after", exact_value_json(change.after)},
                {"value_source",
                 std::string(observation::transformation_value_source_token(
                     change.value_source))},
                {"source_pair_key",
                 change.source_pair_key
                     ? Json{std::to_string(*change.source_pair_key)}
                     : Json{nullptr}},
                {"source_row",
                 change.source_row
                     ? row_reference_json(*change.source_row)
                     : Json{nullptr}},
                {"authority_reference", change.authority_reference},
                {"provenance_reference", change.provenance_reference}});
        }
        rows.emplace_back(Json::Object{
            {"uid", std::to_string(row.uid)},
            {"target", row_reference_json(row.target)},
            {"target_input_key", std::to_string(row.target_input_key)},
            {"tone_frequency_hz",
             observation::canonical_binary64_payload(
                 row.tone_frequency_hz)},
            {"array", std::to_string(row.array)},
            {"network", std::to_string(row.network)},
            {"channel", std::to_string(row.channel)},
            {"relation_pair_keys",
             int64_sequence_json(row.relation_pair_keys)},
            {"fields", std::move(values)},
            {"transformations", std::move(transformations)}});
    }
    return Json::Object{
        {"schema_version", output.schema},
        {"contract_authority", output.contract_authority},
        {"observation_value_issuer", output.observation_value_issuer},
        {"transformation_registry", output.transformation_registry},
        {"envelope", issuance_envelope_json(output.envelope)},
        {"baseline_parent", baseline_reference_json(output.baseline_parent)},
        {"target_parent", artifact_identity_json(output.target_parent)},
        {"relation_parent", artifact_identity_json(output.relation_parent)},
        {"registered_fields", std::move(registered_fields)},
        {"rows", std::move(rows)},
        {"output_presentation_sequence",
         int64_sequence_json(output.output_presentation_sequence)}};
}

Json typed_baseline_value_json(baseline::ValueType type,
                               const baseline::Value &value) {
    std::string datatype;
    switch (type) {
    case baseline::ValueType::int64: datatype = "int64"; break;
    case baseline::ValueType::float64: datatype = "float64-ieee754"; break;
    case baseline::ValueType::boolean: datatype = "bool"; break;
    case baseline::ValueType::string: datatype = "utf8"; break;
    }
    Json encoded = std::visit(
        [](const auto &typed) -> Json {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, baseline::NullValue>) {
                return Json{nullptr};
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return Json{std::to_string(typed)};
            } else if constexpr (std::is_same_v<T, double>) {
                return Json{observation::canonical_binary64_payload(typed)};
            } else {
                return Json{typed};
            }
        },
        value);
    return Json::Object{{"datatype", datatype}, {"value", std::move(encoded)}};
}

Json baseline_descriptor_json(
    const observation::VerifiedBaselineDescriptor &descriptor) {
    // Force exact-byte reconstruction again at the response boundary.
    (void)observation::baseline_descriptor_preimage(descriptor);
    const auto &document = descriptor.document();
    Json::Object envelope{
        {"occurrence", document.envelope.occurrence},
        {"event_reference", document.envelope.event_reference},
        {"output_role", document.envelope.output_role},
        {"producer", document.envelope.producer},
        {"software_revision", document.envelope.software_revision},
        {"configuration_reference",
         document.envelope.configuration_reference},
        {"event_time_utc", document.envelope.event_time_utc}};
    Json::Object context{
        {"project_id", document.context.project_id},
        {"source_name", document.context.source_name},
        {"observation_time_utc", document.context.observation_time_utc},
        {"coordinate_frame", document.context.coordinate_frame}};
    Json::Object observation_identity{
        {"observation",
         std::to_string(document.raw_manifest.observation.observation)},
        {"subobservation",
         std::to_string(document.raw_manifest.observation.subobservation)},
        {"scan", std::to_string(document.raw_manifest.observation.scan)}};
    Json::Array raw_manifest;
    for (const auto &input : document.raw_manifest.inputs) {
        raw_manifest.emplace_back(Json::Object{
            {"network", std::to_string(input.network)},
            {"interface", input.interface_name},
            {"channel_count", std::to_string(input.channel_count)}});
    }
    Json::Array fields;
    std::map<std::string, baseline::RegisteredField> field_by_name;
    for (const auto &field : document.registered_fields) {
        field_by_name.emplace(field.name, field);
        fields.emplace_back(Json::Object{
            {"name", field.name},
            {"datatype", std::string(baseline::value_type_token(field.type))},
            {"unit", field.unit},
            {"nullable", field.nullable},
            {"authority",
             std::string(baseline::field_authority_token(field.authority))},
            {"authority_reference", field.authority_reference},
            {"nonfinite",
             std::string(baseline::nonfinite_policy_token(field.nonfinite))},
            {"registry", field.registry},
            {"description", field.description},
            {"identity_role", "nonidentity"}});
    }
    Json::Array rows;
    Json::Array presentation;
    for (const auto &row : document.rows) {
        Json::Object row_fields;
        for (const auto &[name, value] : row.fields) {
            row_fields.emplace(
                name, typed_baseline_value_json(field_by_name.at(name).type,
                                                value));
        }
        rows.emplace_back(Json::Object{
            {"uid", std::to_string(row.uid)},
            {"tone_freq", typed_baseline_value_json(
                              baseline::ValueType::float64,
                              baseline::Value{row.tone_frequency_hz})},
            {"array", std::to_string(row.array)},
            {"network", std::to_string(row.network)},
            {"channel", std::to_string(row.channel)},
            {"fields", std::move(row_fields)}});
        presentation.emplace_back(std::to_string(row.uid));
    }
    return Json::Object{
        {"schema_version", descriptor.schema()},
        {"contract_authority", descriptor.contract_authority()},
        {"baseline_value_issuer", descriptor.baseline_value_issuer()},
        {"artifact_contract_id", std::string(baseline_contract_id_v1)},
        {"artifact_contract_sha256",
         std::string(baseline_contract_sha256_v1)},
        {"baseline_schema_version", std::string(baseline::schema_version_v1)},
        {"profile", document.profile},
        {"field_registry", document.field_registry},
        {"occurrence", document.envelope.occurrence},
        {"event_reference", document.envelope.event_reference},
        {"envelope", std::move(envelope)},
        {"scientific_context", std::move(context)},
        {"observation", std::move(observation_identity)},
        {"raw_manifest", std::move(raw_manifest)},
        {"registered_fields", std::move(fields)},
        {"rows", std::move(rows)},
        {"wire_presentation_sequence", std::move(presentation)},
        {"semantic_sha256", descriptor.digests().semantic_sha256},
        {"envelope_sha256", descriptor.digests().envelope_sha256},
        {"byte_transport_scope", descriptor.transport().scope},
        {"byte_sha256", descriptor.transport().sha256},
        {"byte_count", std::to_string(descriptor.transport().byte_count)},
        {"receipt_sha256", descriptor.receipt_sha256()},
        {"receipt_byte_count",
         std::to_string(descriptor.receipt_byte_count())}};
}

Json transport_json(std::string_view scope, std::string_view envelope_sha256,
                    std::string_view byte_sha256,
                    std::uint64_t byte_count) {
    return Json::Object{{"scope", std::string(scope)},
                        {"envelope_sha256", std::string(envelope_sha256)},
                        {"byte_sha256", std::string(byte_sha256)},
                        {"byte_count", std::to_string(byte_count)}};
}

void check_expected_baseline(
    const Json::Object &payload,
    const observation::VerifiedBaselineDescriptor &descriptor) {
    if (const auto expected = optional_member(payload, "expected_baseline")) {
        const auto parsed = parse_baseline_reference(*expected,
                                                     "expected_baseline");
        if (parsed != observation::baseline_reference(descriptor)) {
            throw baseline::ContractError(
                "expected_baseline does not match independently verified bytes");
        }
    }
}

std::filesystem::path request_path(const Json &value,
                                   std::string_view context,
                                   std::string_view required_suffix) {
    const auto &text = string_value(value, context);
    if (text.find('\0') != std::string::npos ||
        (required_suffix.size() > text.size()) ||
        !std::string_view(text).ends_with(required_suffix)) {
        throw ProtocolError(std::string(context) + " must end in " +
                            std::string(required_suffix));
    }
    return std::filesystem::path(text);
}

observation::VerifiedBaselineDescriptor load_baseline(
    const std::filesystem::path &artifact_path) {
    const auto receipt_bytes = read_small_receipt(
        receipt_path(artifact_path), "canonical baseline receipt");
    const auto receipt =
        observation::parse_canonical_baseline_receipt(receipt_bytes);
    const auto artifact_bytes = read_regular_file_exact(
        artifact_path, receipt.byte_count, "canonical baseline ECSV");
    return observation::verify_baseline_descriptor(artifact_bytes,
                                                   receipt_bytes);
}

Json success_response(std::string_view request_id,
                      std::string_view operation, Json result) {
    return Json::Object{{"protocol", std::string(protocol_v1)},
                        {"request_id", std::string(request_id)},
                        {"status", "ok"},
                        {"operation", std::string(operation)},
                        {"result", std::move(result)}};
}

Json error_response(const std::optional<std::string> &request_id,
                    std::string_view category, std::string_view code,
                    std::string_view message) {
    return Json::Object{
        {"protocol", std::string(protocol_v1)},
        {"request_id", request_id ? Json{*request_id} : Json{nullptr}},
        {"status", "error"},
        {"error", Json::Object{{"category", std::string(category)},
                                {"code", std::string(code)},
                                {"message", std::string(message)}}}};
}

void verify_bound_source_bytes(const observation::TargetManifest &target) {
    // The manifest has already established unique artifact-local source keys
    // and complete raw/KMP bindings. Locators remain presentation-only local
    // access handles: independently read each declared source once and trust
    // only its exact byte binding, never a locator-derived identity or an
    // inferred header/value.
    std::set<std::pair<std::int64_t, std::string>> verified;
    const auto verify = [&](const observation::SourceArtifact &source) {
        const auto key = std::make_pair(source.source_key,
                                        source.diagnostic_locator);
        if (!verified.insert(key).second) {
            return;
        }
        const auto binding = stream_regular_file_binding(
            std::filesystem::path(source.diagnostic_locator),
            source.byte_count,
            "bound observation raw/KMP source");
        if (binding.byte_count != source.byte_count ||
            binding.content_sha256 != source.content_sha256) {
            throw baseline::ContractError(
                "bound observation source bytes disagree with the target manifest");
        }
    };
    for (const auto &input : target.inputs) {
        verify(input.raw_source);
        verify(input.kmp_source);
    }
}

struct VerifiedObservationApt {
    observation::ParsedMatchedObservationEcsv parsed;
    artifact_publication::ReceiptBinding receipt;
};

VerifiedObservationApt verify_observation_apt_bytes(
    std::string_view artifact_bytes, std::string_view receipt_bytes,
    const observation::VerifiedBaselineDescriptor &baseline_descriptor) {
    auto receipt = artifact_publication::parse_canonical_receipt(
        receipt_bytes, artifact_publication::receipt_schema_v1,
        observation::matched_output_byte_transport_scope_v1);
    artifact_publication::validate_receipt_binding(artifact_bytes, receipt);
    auto parsed = observation::parse_matched_observation_ecsv(
        artifact_bytes, baseline_descriptor);
    if (parsed.declared_digests.envelope_sha256 != receipt.envelope_sha256 ||
        parsed.computed_transport.scope != receipt.scope ||
        parsed.computed_transport.sha256 != receipt.byte_sha256 ||
        parsed.computed_transport.byte_count != receipt.byte_count) {
        throw baseline::ContractError(
            "matched observation APT receipt does not bind its canonical reread");
    }
    return {std::move(parsed), std::move(receipt)};
}

void check_expected_artifact(
    const Json::Object &payload,
    const observation::ArtifactIdentity &actual) {
    if (const auto expected = optional_member(payload, "expected_artifact")) {
        if (parse_artifact_identity(*expected, "expected_artifact") != actual) {
            throw baseline::ContractError(
                "expected_artifact does not match independently verified bytes");
        }
    }
}

void check_expected_transport(
    const Json::Object &payload,
    const artifact_publication::ReceiptBinding &actual) {
    if (const auto expected = optional_member(payload, "expected_transport")) {
        const auto parsed = parse_expected_transport(*expected,
                                                     "expected_transport");
        if (parsed.scope != actual.scope ||
            parsed.envelope_sha256 != actual.envelope_sha256 ||
            parsed.byte_sha256 != actual.byte_sha256 ||
            parsed.byte_count != actual.byte_count) {
            throw baseline::ContractError(
                "expected_transport does not match independently verified bytes");
        }
    }
}

Json observation_apt_result_json(
    const observation::VerifiedBaselineDescriptor &baseline_descriptor,
    const observation::ParsedMatchedObservationEcsv &parsed,
    const artifact_publication::ReceiptBinding &receipt,
    const std::filesystem::path &artifact_path) {
    const auto target_identity = observation::artifact_identity(parsed.target);
    const auto relation_identity = observation::artifact_identity(
        parsed.relation, baseline_descriptor, parsed.target);
    const auto output_identity = observation::artifact_identity(
        parsed.output, baseline_descriptor, parsed.target, parsed.relation);
    return Json::Object{
        {"baseline", baseline_descriptor_json(baseline_descriptor)},
        {"target", target_manifest_json(parsed.target)},
        {"target_identity", artifact_identity_json(target_identity)},
        {"relation", match_relation_json(parsed.relation,
                                          baseline_descriptor,
                                          parsed.target)},
        {"relation_identity", artifact_identity_json(relation_identity)},
        {"output", matched_output_json(parsed.output,
                                       baseline_descriptor,
                                       parsed.target,
                                       parsed.relation)},
        {"artifact", artifact_identity_json(output_identity)},
        {"transport", transport_json(receipt.scope,
                                      receipt.envelope_sha256,
                                      receipt.byte_sha256,
                                      receipt.byte_count)},
        {"observation_apt_ecsv", artifact_path.string()},
        {"receipt", receipt_path(artifact_path).string()}};
}

Json describe_baseline(const Json::Object &payload) {
    exact_members(payload, {"baseline_ecsv"}, {"expected_baseline"},
                  "payload");
    const auto artifact_path = request_path(
        member(payload, "baseline_ecsv", "payload"),
        "payload.baseline_ecsv", ".ecsv");
    auto descriptor = load_baseline(artifact_path);
    check_expected_baseline(payload, descriptor);
    return Json::Object{
        {"baseline", baseline_descriptor_json(descriptor)},
        {"baseline_reference",
         baseline_reference_json(observation::baseline_reference(descriptor))},
        {"baseline_ecsv", artifact_path.string()},
        {"receipt", receipt_path(artifact_path).string()}};
}

std::string issue_observation_apt(
    const Json::Object &payload,
    const ProtocolDependencies &dependencies,
    std::string_view request_id, std::string_view operation) {
    exact_members(payload,
                  {"baseline_ecsv", "target", "relation",
                   "field_source_selections", "publication"},
                  {"expected_baseline"}, "payload");
    const auto baseline_path = request_path(
        member(payload, "baseline_ecsv", "payload"),
        "payload.baseline_ecsv", ".ecsv");
    auto baseline_descriptor = load_baseline(baseline_path);
    check_expected_baseline(payload, baseline_descriptor);

    auto target = parse_target_facts(member(payload, "target", "payload"),
                                     "payload.target");
    observation::validate(target);
    verify_bound_source_bytes(target);
    auto relation = parse_relation_facts(
        member(payload, "relation", "payload"), "payload.relation",
        baseline_descriptor, target);
    observation::validate(relation, baseline_descriptor, target);
    const auto field_sources = parse_field_source_selections(
        member(payload, "field_source_selections", "payload"),
        "payload.field_source_selections", baseline_descriptor, target,
        relation);

    const auto &publication = object(
        member(payload, "publication", "payload"), "payload.publication");
    exact_members(publication,
                  {"output_ecsv", "configuration_reference",
                   "event_time_utc"},
                  {}, "payload.publication");
    auto output_path = request_path(
        member(publication, "output_ecsv", "payload.publication"),
        "payload.publication.output_ecsv", ".apt.ecsv");
    if (output_path.parent_path().empty()) {
        output_path = std::filesystem::path(".") / output_path;
    }
    const auto &configuration_reference = string_value(
        member(publication, "configuration_reference", "payload.publication"),
        "payload.publication.configuration_reference");
    const auto &event_time_utc = string_value(
        member(publication, "event_time_utc", "payload.publication"),
        "payload.publication.event_time_utc");

    const auto issuance =
        artifact_publication::issue_opaque(dependencies.issuance_factory);
    observation::IssuanceEnvelope output_envelope{
        issuance.occurrence, issuance.event_reference,
        std::string(CITLALI_GIT_VERSION), configuration_reference,
        event_time_utc};
    auto output = observation::make_matched_observation_output_v1(
        std::move(output_envelope), baseline_descriptor, target, relation,
        field_sources);
    const auto serialized = observation::serialize_matched_observation_ecsv(
        output, baseline_descriptor, target, relation);
    auto receipt = artifact_publication::make_receipt_binding(
        std::string(artifact_publication::receipt_schema_v1),
        std::string(observation::matched_output_byte_transport_scope_v1),
        serialized.digests.envelope_sha256, serialized.bytes);
    if (serialized.transport.scope != receipt.scope ||
        serialized.transport.envelope_sha256 != receipt.envelope_sha256 ||
        serialized.transport.sha256 != receipt.byte_sha256 ||
        serialized.transport.byte_count != receipt.byte_count) {
        throw baseline::ContractError(
            "matched observation serializer transport disagrees with publication authority");
    }
    auto receipt_bytes =
        artifact_publication::canonical_receipt_bytes(receipt);
    auto verified = verify_observation_apt_bytes(
        serialized.bytes, receipt_bytes, baseline_descriptor);
    auto response_material = observation_apt_result_json(
        baseline_descriptor, verified.parsed, verified.receipt, output_path);
    // Finish every potentially allocating response construction step before
    // entering the no-replace publisher. Once its completion receipt becomes
    // visible, only noexcept string moves and the caller's stream write remain.
    auto response_json = serialize_json(success_response(
        request_id, operation, std::move(response_material)));

    artifact_publication::PublicationPlan plan;
    plan.artifact_path = output_path;
    plan.receipt_path = receipt_path(output_path);
    plan.artifact_bytes = serialized.bytes;
    plan.receipt_bytes = std::move(receipt_bytes);
    plan.validate = [&baseline_descriptor](std::string_view artifact_bytes,
                                           std::string_view receipt_bytes) {
        (void)verify_observation_apt_bytes(
            artifact_bytes, receipt_bytes, baseline_descriptor);
    };
    artifact_publication::publish_canonical_artifact(
        plan, dependencies.publication_hooks);
    return response_json;
}

Json validate_observation_apt(const Json::Object &payload) {
    exact_members(payload,
                  {"baseline_ecsv", "observation_apt_ecsv"},
                  {"expected_baseline", "expected_artifact",
                   "expected_transport"},
                  "payload");
    const auto baseline_path = request_path(
        member(payload, "baseline_ecsv", "payload"),
        "payload.baseline_ecsv", ".ecsv");
    auto baseline_descriptor = load_baseline(baseline_path);
    check_expected_baseline(payload, baseline_descriptor);
    const auto artifact_path = request_path(
        member(payload, "observation_apt_ecsv", "payload"),
        "payload.observation_apt_ecsv", ".apt.ecsv");
    // The sibling receipt is the sole completion marker. Reject its absence
    // or malformed framing before opening the larger scientific artifact.
    const auto receipt_bytes = read_small_receipt(
        receipt_path(artifact_path), "matched observation APT receipt");
    const auto receipt = artifact_publication::parse_canonical_receipt(
        receipt_bytes, artifact_publication::receipt_schema_v1,
        observation::matched_output_byte_transport_scope_v1);
    const auto artifact_bytes = read_regular_file_exact(
        artifact_path, receipt.byte_count, "matched observation APT ECSV");
    auto verified = verify_observation_apt_bytes(
        artifact_bytes, receipt_bytes, baseline_descriptor);
    const auto output_identity = observation::artifact_identity(
        verified.parsed.output, baseline_descriptor, verified.parsed.target,
        verified.parsed.relation);
    check_expected_artifact(payload, output_identity);
    check_expected_transport(payload, verified.receipt);
    return observation_apt_result_json(
        baseline_descriptor, verified.parsed, verified.receipt,
        artifact_path);
}

std::string process_valid_request(
    const Json &request, const ProtocolDependencies &dependencies,
    std::string &request_id, std::string &operation) {
    const auto &outer = object(request, "request");
    exact_members(outer, {"protocol", "request_id", "operation", "payload"},
                  {}, "request");
    const auto &protocol = string_value(member(outer, "protocol", "request"),
                                        "request.protocol");
    if (protocol != protocol_v1) {
        throw ProtocolError("request.protocol is not the supported version");
    }
    request_id = string_value(member(outer, "request_id", "request"),
                              "request.request_id");
    operation = string_value(member(outer, "operation", "request"),
                             "request.operation");
    const auto &payload = object(member(outer, "payload", "request"),
                                 "request.payload");
    if (operation == describe_baseline_operation_v1) {
        return serialize_json(success_response(
            request_id, operation, describe_baseline(payload)));
    } else if (operation == issue_observation_apt_operation_v1) {
        (void)dependencies;
        throw baseline::ContractError(
            "new canonical APT v1 issuance is disabled; v1 is read-only historical evidence");
    } else if (operation == validate_observation_apt_operation_v1) {
        return serialize_json(success_response(
            request_id, operation, validate_observation_apt(payload)));
    } else {
        throw ProtocolError("request.operation is not supported");
    }
}

std::optional<std::string> recover_request_id(const Json &request) {
    const auto outer = std::get_if<Json::Object>(&request.value);
    if (!outer) return std::nullopt;
    const auto found = outer->find("request_id");
    if (found == outer->end()) return std::nullopt;
    const auto text = std::get_if<std::string>(&found->second.value);
    if (!text || text->empty() ||
        !baseline::detail::canonical_text(*text, true)) {
        return std::nullopt;
    }
    return *text;
}

ProtocolResult framing_error(std::string_view message) {
    return {protocol_error_exit_code,
            serialize_json(error_response(std::nullopt, "protocol",
                                          "invalid-request-framing",
                                          message))};
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
        auto response_json = process_valid_request(
            request, dependencies, exact_request_id, operation);
        return {success_exit_code, std::move(response_json)};
    } catch (const ProtocolError &error) {
        return {protocol_error_exit_code,
                serialize_json(error_response(request_id, "protocol",
                                              "invalid-request",
                                              error.what()))};
    } catch (const baseline::ContractError &error) {
        return {contract_rejection_exit_code,
                serialize_json(error_response(request_id, "contract",
                                              "contract-rejection",
                                              error.what()))};
    } catch (const artifact_publication::PublicationError &error) {
        return {contract_rejection_exit_code,
                serialize_json(error_response(request_id, "contract",
                                              "publication-rejection",
                                              error.what()))};
    } catch (const std::exception &error) {
        return {protocol_error_exit_code,
                serialize_json(error_response(request_id, "internal",
                                              "internal-error",
                                              error.what()))};
    } catch (...) {
        return {protocol_error_exit_code,
                serialize_json(error_response(request_id, "internal",
                                              "internal-error",
                                              "unknown internal error"))};
    }
}

ProtocolDependencies production_dependencies() {
    ProtocolDependencies result;
    result.issuance_factory = [] {
        return artifact_publication::make_entropy_issuance(
            "occurrence:citlali/observation-matched-apt-v1#",
            "event:citlali/observation-matched-apt-v1#");
    };
    return result;
}

std::optional<int> dispatch_if_requested(
    int argc, char *argv[], std::istream &input, std::ostream &output,
    const ProtocolDependencies &dependencies) {
    bool requested = false;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == cli_option_v1) {
            requested = true;
        }
    }
    if (!requested) {
        return std::nullopt;
    }

    ProtocolResult result;
    if (argc != 2 || std::string_view(argv[1]) != cli_option_v1) {
        result = framing_error(
            "the canonical APT protocol option must be the only argument");
    } else {
        try {
            const std::string input_bytes{
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>()};
            if (input.bad()) {
                result = framing_error("failed to read the protocol request");
            } else if (input_bytes.empty() || input_bytes.back() != '\n' ||
                       input_bytes.find('\r') != std::string::npos ||
                       input_bytes.find('\n') != input_bytes.size() - 1U) {
                result = framing_error(
                    "protocol input must be exactly one LF-terminated JSON line");
            } else {
                result = process_request_line(
                    std::string_view(input_bytes).substr(
                        0, input_bytes.size() - 1U), dependencies);
            }
        } catch (const std::exception &) {
            result = framing_error(
                "protocol input collection failed before request dispatch");
        } catch (...) {
            result = framing_error(
                "protocol input collection failed before request dispatch");
        }
    }
    output << result.response_json << '\n';
    output.flush();
    if (!output) {
        return protocol_error_exit_code;
    }
    return result.exit_code;
}

}  // namespace citlali::cli::canonical_apt_contract_protocol_v1

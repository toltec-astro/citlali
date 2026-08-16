#pragma once

#include <citlali/core/utils/sha256.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace citlali::pipeline::canonical_apt_v1 {

inline constexpr std::string_view schema_version_v1 =
    "citlali-canonical-apt-v1";
inline constexpr std::string_view framing_encoding_v1 =
    "citlali-labelled-type-length-v1";
inline constexpr std::string_view semantic_scope_v1 =
    "citlali-canonical-apt-semantic-sha256-v1";
inline constexpr std::string_view envelope_scope_v1 =
    "citlali-canonical-apt-envelope-sha256-v1";
inline constexpr std::string_view byte_transport_scope_v1 =
    "citlali-canonical-apt-byte-transport-sha256-v1";
inline constexpr std::string_view baseline_profile_v1 =
    "citlali-beammap-baseline-apt-v1";
inline constexpr std::string_view baseline_field_registry_v1 =
    "citlali-canonical-apt-baseline-fields-v1";
inline constexpr std::string_view extension_registry_v1 =
    "citlali-canonical-apt-extension-registry-v1";
inline constexpr std::string_view field_registry_version_v1 =
    "citlali-canonical-apt-field-registry-v1";
inline constexpr std::string_view baseline_output_role_v1 =
    "beammap-baseline-apt";
inline constexpr std::int64_t uid_v1_max = 9007199254740991LL;

class ContractError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct NullValue {
    friend constexpr bool operator==(NullValue, NullValue) noexcept = default;
};

enum class ValueType {
    int64,
    float64,
    boolean,
    string,
};

enum class FieldAuthority {
    producer,
    copied_declared,
    unavailable,
};

enum class NonFinitePolicy {
    reject,
    nan_token,
    canonical_token,
};

using Value =
    std::variant<NullValue, std::int64_t, double, bool, std::string>;

struct RegisteredField {
    std::string name;
    ValueType type = ValueType::float64;
    std::string unit;
    bool nullable = false;
    FieldAuthority authority = FieldAuthority::unavailable;
    std::string authority_reference;
    NonFinitePolicy nonfinite = NonFinitePolicy::reject;
    std::string registry;
    std::string description;

    friend bool operator==(const RegisteredField &,
                           const RegisteredField &) = default;
};

struct ObservationIdentity {
    std::int64_t observation = 0;
    std::int64_t subobservation = 0;
    std::int64_t scan = 0;

    friend bool operator==(const ObservationIdentity &,
                           const ObservationIdentity &) = default;
};

struct RawInput {
    std::int64_t network = 0;
    std::string interface_name;
    std::int64_t channel_count = 0;

    friend bool operator==(const RawInput &, const RawInput &) = default;
};

struct RawManifest {
    ObservationIdentity observation;
    std::vector<RawInput> inputs;
};

struct ScientificContext {
    std::string project_id;
    std::string source_name;
    std::string observation_time_utc;
    std::string coordinate_frame;
};

struct Row {
    std::int64_t uid = 0;
    double tone_frequency_hz = 0.0;
    std::int64_t array = 0;
    std::int64_t network = 0;
    std::int64_t channel = 0;
    std::map<std::string, Value> fields;
};

struct Envelope {
    // Opaque issuer-provided occurrence reference. It is intentionally not
    // parsed as a UUID and is never derived from semantic content.
    std::string occurrence;
    std::string event_reference;
    std::string output_role;
    std::string producer;
    std::string software_revision;
    std::string configuration_reference;
    std::string event_time_utc;
};

struct Document {
    std::string profile{baseline_profile_v1};
    std::string field_registry{field_registry_version_v1};
    Envelope envelope;
    ScientificContext context;
    RawManifest raw_manifest;
    std::vector<RegisteredField> registered_fields;
    std::vector<Row> rows;
};

struct Digests {
    std::string semantic_sha256;
    std::string envelope_sha256;
};

struct ByteTransportHash {
    std::string scope;
    std::string envelope_sha256;
    std::string sha256;
    std::uint64_t byte_count = 0;
};

struct FieldRegistry {
    std::string version;
    std::vector<RegisteredField> required_baseline_fields;
    std::vector<RegisteredField> optional_extensions;
};

inline std::string_view value_type_token(ValueType type) {
    switch (type) {
    case ValueType::int64:
        return "int64";
    case ValueType::float64:
        return "float64";
    case ValueType::boolean:
        return "bool";
    case ValueType::string:
        return "string";
    }
    throw ContractError("unsupported canonical APT value type");
}

inline ValueType parse_value_type_token(std::string_view token) {
    if (token == "int64") {
        return ValueType::int64;
    }
    if (token == "float64") {
        return ValueType::float64;
    }
    if (token == "bool") {
        return ValueType::boolean;
    }
    if (token == "string") {
        return ValueType::string;
    }
    throw ContractError("unsupported canonical APT value type token: " +
                        std::string(token));
}

inline std::string_view field_authority_token(FieldAuthority authority) {
    switch (authority) {
    case FieldAuthority::producer:
        return "producer";
    case FieldAuthority::copied_declared:
        return "copied-declared";
    case FieldAuthority::unavailable:
        return "unavailable";
    }
    throw ContractError("unsupported canonical APT field authority");
}

inline FieldAuthority parse_field_authority_token(std::string_view token) {
    if (token == "producer") {
        return FieldAuthority::producer;
    }
    if (token == "copied-declared") {
        return FieldAuthority::copied_declared;
    }
    if (token == "unavailable") {
        return FieldAuthority::unavailable;
    }
    throw ContractError("unsupported canonical APT field authority token: " +
                        std::string(token));
}

inline std::string_view nonfinite_policy_token(NonFinitePolicy policy) {
    switch (policy) {
    case NonFinitePolicy::reject:
        return "reject";
    case NonFinitePolicy::nan_token:
        return "nan-token";
    case NonFinitePolicy::canonical_token:
        return "canonical-token";
    }
    throw ContractError("unsupported canonical APT nonfinite policy");
}

inline NonFinitePolicy parse_nonfinite_policy_token(std::string_view token) {
    if (token == "reject") {
        return NonFinitePolicy::reject;
    }
    if (token == "nan-token") {
        return NonFinitePolicy::nan_token;
    }
    if (token == "canonical-token") {
        return NonFinitePolicy::canonical_token;
    }
    throw ContractError(
        "unsupported canonical APT nonfinite policy token: " +
        std::string(token));
}

inline RegisteredField registered_field_spec(
    std::string name, ValueType type, std::string unit, bool nullable,
    FieldAuthority authority, std::string authority_reference,
    NonFinitePolicy nonfinite, std::string registry,
    std::string description) {
    return {std::move(name), type, std::move(unit), nullable, authority,
            std::move(authority_reference), nonfinite, std::move(registry),
            std::move(description)};
}

inline std::vector<RegisteredField> required_baseline_fields_v1() {
    const auto baseline = std::string(baseline_field_registry_v1);
    const auto producer = FieldAuthority::producer;
    const auto unavailable = FieldAuthority::unavailable;
    const auto nan_or_null = NonFinitePolicy::nan_token;
    const auto reject = NonFinitePolicy::reject;
    const auto fit = "citlali:beammap-fit-v1";
    const auto geometry = "citlali:beammap-geometry-v1";
    const auto calibration = "citlali:beammap-calibration-v1";
    const auto quality = "citlali:beammap-quality-v1";
    const auto unresolved = "authority-unresolved-v1";
    return {
        registered_field_spec("fg", ValueType::int64, "N/A", true,
                              unavailable, unresolved, reject, baseline,
                              "frequency group; authority unresolved and nonidentity"),
        registered_field_spec("pg", ValueType::int64, "N/A", true,
                              unavailable, unresolved, reject, baseline,
                              "polarization group; authority unresolved and nonidentity"),
        registered_field_spec("ori", ValueType::int64, "N/A", true,
                              unavailable, unresolved, reject, baseline,
                              "orientation; authority unresolved and nonidentity"),
        registered_field_spec("loc", ValueType::int64, "N/A", true,
                              unavailable, unresolved, reject, baseline,
                              "location; authority unresolved and nonidentity"),
        registered_field_spec("responsivity", ValueType::float64, "N/A", true,
                              unavailable, unresolved, nan_or_null, baseline,
                              "responsivity; physical authority unresolved"),
        registered_field_spec("flxscale", ValueType::float64,
                              "mJy/beam/xs", true, producer, calibration,
                              nan_or_null, baseline,
                              "flux conversion scale"),
        registered_field_spec("sens", ValueType::float64,
                              "mJy/beam x s^0.5", true, producer,
                              calibration, nan_or_null, baseline,
                              "sensitivity"),
        registered_field_spec("derot_elev", ValueType::float64, "rad", true,
                              producer, geometry, nan_or_null, baseline,
                              "derotation elevation angle"),
        registered_field_spec("amp", ValueType::float64, "xs", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted amplitude"),
        registered_field_spec("amp_err", ValueType::float64, "xs", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted amplitude error"),
        registered_field_spec("x_t", ValueType::float64, "arcsec", true,
                              producer, geometry, nan_or_null, baseline,
                              "fitted azimuthal offset"),
        registered_field_spec("x_t_err", ValueType::float64, "arcsec", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted azimuthal offset error"),
        registered_field_spec("y_t", ValueType::float64, "arcsec", true,
                              producer, geometry, nan_or_null, baseline,
                              "fitted altitude offset"),
        registered_field_spec("y_t_err", ValueType::float64, "arcsec", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted altitude offset error"),
        registered_field_spec("a_fwhm", ValueType::float64, "arcsec", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted azimuthal FWHM"),
        registered_field_spec("a_fwhm_err", ValueType::float64, "arcsec", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted azimuthal FWHM error"),
        registered_field_spec("b_fwhm", ValueType::float64, "arcsec", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted altitude FWHM"),
        registered_field_spec("b_fwhm_err", ValueType::float64, "arcsec", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted altitude FWHM error"),
        registered_field_spec("angle", ValueType::float64, "rad", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted rotation angle"),
        registered_field_spec("angle_err", ValueType::float64, "rad", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted angle uncertainty"),
        registered_field_spec("converge_iter", ValueType::int64, "N/A", false,
                              producer, fit, reject, baseline,
                              "beammap convergence iteration"),
        registered_field_spec("flag", ValueType::int64, "N/A", false,
                              producer, quality, reject, baseline,
                              "bad detector flag; closed values {0,1}"),
        registered_field_spec("sig2noise", ValueType::float64, "N/A", true,
                              producer, fit, nan_or_null, baseline,
                              "fitted signal to noise"),
        registered_field_spec("x_t_raw", ValueType::float64, "arcsec", true,
                              producer, geometry, nan_or_null, baseline,
                              "raw azimuthal offset"),
        registered_field_spec("y_t_raw", ValueType::float64, "arcsec", true,
                              producer, geometry, nan_or_null, baseline,
                              "raw altitude offset"),
        registered_field_spec("x_t_derot", ValueType::float64, "arcsec", true,
                              producer, geometry, nan_or_null, baseline,
                              "derotated azimuthal offset"),
        registered_field_spec("y_t_derot", ValueType::float64, "arcsec", true,
                              producer, geometry, nan_or_null, baseline,
                              "derotated altitude offset"),
    };
}

inline std::vector<RegisteredField> optional_extension_fields_v1() {
    const auto registry = std::string(extension_registry_v1);
    const auto producer = FieldAuthority::producer;
    const auto copied = FieldAuthority::copied_declared;
    const auto reject = NonFinitePolicy::reject;
    const auto nan_or_null = NonFinitePolicy::nan_token;
    const auto quality = "citlali:beammap-quality-v1";
    const auto mask = "citlali:beammap-mask-diagnostics-v1";
    const auto prior = "citlali:beammap-soft-prior-v1";
    const auto calibration = "citlali:beammap-empirical-calibration-v1";
    return {
        registered_field_spec("flag2", ValueType::int64, "N/A", false,
                              producer, quality, reject, registry,
                              "bitwise Beammap quality flag; allowed mask 0xff"),
        registered_field_spec(
            "kids_flag", ValueType::int64, "N/A", false, copied,
            "kids:fit-report-v1", reject, registry,
            "imported KIDs model-fit flag; exact integral values, nonidentity"),
        registered_field_spec("rfi_masked_samples", ValueType::int64,
                              "samples", false, producer, mask, reject,
                              registry, "number of samples masked by rfi_mask"),
        registered_field_spec("rfi_masked_scans", ValueType::int64,
                              "scans", false, producer, mask, reject,
                              registry, "number of scans masked by rfi_mask"),
        registered_field_spec("scan_band_masked_samples", ValueType::int64,
                              "samples", false, producer, mask, reject,
                              registry, "number of samples masked by scan_band_mask"),
        registered_field_spec("scan_band_masked_rows", ValueType::int64,
                              "rows", false, producer, mask, reject, registry,
                              "number of detector-map edge rows masked"),
        registered_field_spec("scan_band_masked_edge", ValueType::int64,
                              "N/A", false, producer, mask, reject, registry,
                              "scan-band edge code {0,1,2,3}"),
        registered_field_spec("scan_band_mask_rejected", ValueType::int64,
                              "N/A", false, producer, mask, reject, registry,
                              "scan-band mask rejection code {0,1}"),
        registered_field_spec("final_prior_slot_index", ValueType::int64,
                              "N/A", true, producer, prior, reject, registry,
                              "nearest soft-prior slot; nonidentity"),
        registered_field_spec("final_prior_d2", ValueType::float64, "N/A",
                              true, producer, prior, nan_or_null, registry,
                              "nearest soft-prior Mahalanobis distance squared"),
        registered_field_spec("cal_amp", ValueType::float64, "xs", true,
                              producer, calibration, nan_or_null, registry,
                              "beammap calibration amplitude"),
        registered_field_spec("cal_amp_method", ValueType::int64, "N/A",
                              false, producer, calibration, reject, registry,
                              "calibration amplitude method code {0,1}"),
        registered_field_spec("template_amp", ValueType::float64, "xs", true,
                              producer, calibration, nan_or_null, registry,
                              "empirical template matched amplitude"),
        registered_field_spec("template_offset", ValueType::float64, "xs",
                              true, producer, calibration, nan_or_null,
                              registry, "empirical template fitted offset"),
        registered_field_spec("template_resid_rms", ValueType::float64, "xs",
                              true, producer, calibration, nan_or_null,
                              registry, "empirical template residual RMS"),
        registered_field_spec("template_npix", ValueType::int64, "pix", false,
                              producer, calibration, reject, registry,
                              "empirical template fitted pixel count"),
        registered_field_spec("template_amp_over_fit_amp", ValueType::float64,
                              "N/A", true, producer, calibration,
                              nan_or_null, registry,
                              "template amplitude divided by fit amplitude"),
        registered_field_spec("cal_amp_over_fit_amp", ValueType::float64,
                              "N/A", true, producer, calibration,
                              nan_or_null, registry,
                              "calibration amplitude divided by fit amplitude"),
        registered_field_spec("map_peak_amp", ValueType::float64, "xs", true,
                              producer, calibration, nan_or_null, registry,
                              "baseline-subtracted local map peak"),
        registered_field_spec("map_peak_amp_over_fit_amp", ValueType::float64,
                              "N/A", true, producer, calibration,
                              nan_or_null, registry,
                              "map peak divided by fit amplitude"),
    };
}

inline const FieldRegistry &canonical_field_registry_v1() {
    static const FieldRegistry registry{
        std::string(field_registry_version_v1), required_baseline_fields_v1(),
        optional_extension_fields_v1()};
    return registry;
}

inline bool is_sha256_reference(std::string_view value) {
    constexpr std::string_view prefix = "sha256:";
    if (!value.starts_with(prefix) || value.size() != prefix.size() + 64) {
        return false;
    }
    return std::all_of(value.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
                       value.end(), [](unsigned char ch) {
                           return (ch >= '0' && ch <= '9') ||
                               (ch >= 'a' && ch <= 'f');
                       });
}

inline std::string canonical_frame(std::string_view label,
                                   std::string_view type,
                                   std::string_view payload) {
    std::string result;
    result.reserve(label.size() + type.size() + payload.size() + 48);
    result += "F";
    result += std::to_string(label.size());
    result += ":";
    result.append(label);
    result += "T";
    result += std::to_string(type.size());
    result += ":";
    result.append(type);
    result += "V";
    result += std::to_string(payload.size());
    result += ":";
    result.append(payload);
    result += ";";
    return result;
}

inline std::string canonical_float64_payload(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "canonical APT v1 requires IEEE-754 binary64 double");
    static_assert(std::numeric_limits<double>::is_iec559,
                  "canonical APT v1 requires IEC 559 floating point");
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return std::signbit(value) ? "-inf" : "+inf";
    }
    const auto bits = std::bit_cast<std::uint64_t>(value);
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::hex << std::nouppercase << std::setfill('0')
           << std::setw(16) << bits;
    return stream.str();
}

namespace detail {

inline bool valid_utf8(std::string_view value) {
    std::size_t index = 0;
    while (index < value.size()) {
        const auto lead = static_cast<unsigned char>(value[index]);
        if (lead <= 0x7f) {
            ++index;
            continue;
        }
        std::uint32_t code_point = 0;
        std::size_t continuation_count = 0;
        if (lead >= 0xc2 && lead <= 0xdf) {
            code_point = lead & 0x1fU;
            continuation_count = 1;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            code_point = lead & 0x0fU;
            continuation_count = 2;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            code_point = lead & 0x07U;
            continuation_count = 3;
        } else {
            return false;
        }
        if (index + continuation_count >= value.size()) {
            return false;
        }
        for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
            const auto next = static_cast<unsigned char>(value[index + offset]);
            if ((next & 0xc0U) != 0x80U) {
                return false;
            }
            code_point = (code_point << 6U) | (next & 0x3fU);
        }
        const bool overlong =
            (continuation_count == 1 && code_point < 0x80U) ||
            (continuation_count == 2 && code_point < 0x800U) ||
            (continuation_count == 3 && code_point < 0x10000U);
        if (overlong || (code_point >= 0xd800U && code_point <= 0xdfffU) ||
            code_point > 0x10ffffU) {
            return false;
        }
        index += continuation_count + 1;
    }
    return true;
}

inline bool canonical_text(std::string_view value, bool single_line) {
    if (!valid_utf8(value)) {
        return false;
    }
    for (std::size_t index = 0; index < value.size();) {
        const auto lead = static_cast<unsigned char>(value[index]);
        std::uint32_t code_point = lead;
        std::size_t count = 1;
        if (lead >= 0xc2 && lead <= 0xdf) {
            code_point = lead & 0x1fU;
            count = 2;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            code_point = lead & 0x0fU;
            count = 3;
        } else if (lead >= 0xf0) {
            code_point = lead & 0x07U;
            count = 4;
        }
        for (std::size_t offset = 1; offset < count; ++offset) {
            code_point = (code_point << 6U) |
                (static_cast<unsigned char>(value[index + offset]) & 0x3fU);
        }
        if (code_point == 0 || code_point == 0x7fU ||
            (code_point >= 0x80U && code_point <= 0x9fU) ||
            code_point == 0x85U || code_point == 0x2028U ||
            code_point == 0x2029U ||
            (code_point >= 0xfdd0U && code_point <= 0xfdefU) ||
            (code_point & 0xffffU) == 0xfffeU ||
            (code_point & 0xffffU) == 0xffffU ||
            (code_point < 0x20U && code_point != '\t') ||
            (single_line && (code_point == '\n' || code_point == '\r'))) {
            return false;
        }
        index += count;
    }
    return true;
}

inline void require_text(std::string_view label, const std::string &value,
                         bool allow_empty = false,
                         bool single_line = true) {
    if ((!allow_empty && value.empty()) ||
        !canonical_text(value, single_line)) {
        throw ContractError("canonical APT requires valid UTF-8 text for " +
                            std::string(label));
    }
}

inline bool valid_registered_name(std::string_view name) {
    if (name.empty() ||
        !((name.front() >= 'A' && name.front() <= 'Z') ||
          (name.front() >= 'a' && name.front() <= 'z'))) {
        return false;
    }
    return std::all_of(name.begin() + 1, name.end(), [](unsigned char ch) {
        return (ch >= 'A' && ch <= 'Z') ||
            (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') ||
            ch == '_';
    });
}

inline bool protected_structural_name(std::string_view name) {
    constexpr std::array<std::string_view, 5> names{
        "uid", "tone_freq", "array", "nw", "kids_tone"};
    return std::find(names.begin(), names.end(), name) != names.end();
}

inline bool protected_contract_name(std::string_view name) {
    if (protected_structural_name(name)) {
        return true;
    }
    constexpr std::array<std::string_view, 29> names{
        "schema_version", "profile", "field_registry", "framing_encoding",
        "semantic_scope", "semantic_sha256", "envelope_scope",
        "envelope_sha256", "byte_transport_scope", "occurrence",
        "event_reference", "output_role", "producer", "software_revision",
        "configuration_reference", "event_time_utc", "observation",
        "subobservation", "scan", "raw_manifest", "network", "channel",
        "interface", "channel_count", "scientific_context", "project_id",
        "source_name", "observation_time_utc", "coordinate_frame"};
    return std::find(names.begin(), names.end(), name) != names.end();
}

inline bool unresolved_design_name(std::string_view name) {
    constexpr std::array<std::string_view, 4> names{"fg", "pg", "ori",
                                                    "loc"};
    return std::find(names.begin(), names.end(), name) != names.end();
}

inline std::vector<RegisteredField> sorted_registered_fields(
    const Document &document) {
    auto result = document.registered_fields;
    std::sort(result.begin(), result.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return lhs.name < rhs.name;
    });
    return result;
}

inline std::vector<RawInput> sorted_raw_inputs(const Document &document) {
    auto result = document.raw_manifest.inputs;
    std::sort(result.begin(), result.end(), [](const auto &lhs,
                                               const auto &rhs) {
        return std::tie(lhs.network, lhs.interface_name) <
            std::tie(rhs.network, rhs.interface_name);
    });
    return result;
}

inline std::int64_t expected_array_for_network(std::int64_t network) {
    if (network >= 0 && network <= 6) {
        return 0;
    }
    if (network >= 7 && network <= 10) {
        return 1;
    }
    if (network >= 11 && network <= 12) {
        return 2;
    }
    throw ContractError("canonical APT v1 network is outside TolTEC enum {0..12}");
}

inline bool is_canonical_toltec_interface(std::string_view interface_name,
                                          std::int64_t network) {
    return interface_name == "toltec" + std::to_string(network);
}

inline bool is_utc_timestamp(std::string_view value) {
    if (value.size() < 20 || value.back() != 'Z' || value[4] != '-' ||
        value[7] != '-' || value[10] != 'T' || value[13] != ':' ||
        value[16] != ':') {
        return false;
    }
    const auto decimal = [](char ch) { return ch >= '0' && ch <= '9'; };
    for (const auto index : {0U, 1U, 2U, 3U, 5U, 6U, 8U, 9U, 11U, 12U,
                             14U, 15U, 17U, 18U}) {
        if (!decimal(value[index])) {
            return false;
        }
    }
    if (value.size() != 20) {
        if (value[19] != '.' || value.size() == 21) {
            return false;
        }
        for (std::size_t index = 20; index + 1 < value.size(); ++index) {
            if (!decimal(value[index])) {
                return false;
            }
        }
    }
    const auto number = [&](std::size_t offset, std::size_t count) {
        int result = 0;
        for (std::size_t index = 0; index < count; ++index) {
            result = result * 10 + (value[offset + index] - '0');
        }
        return result;
    };
    const int year = number(0, 4);
    const int month = number(5, 2);
    const int day = number(8, 2);
    const int hour = number(11, 2);
    const int minute = number(14, 2);
    const int second = number(17, 2);
    if (year == 0 || month < 1 || month > 12 || hour > 23 || minute > 59 ||
        second > 59) {
        return false;
    }
    constexpr std::array<int, 12> days_per_month{
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int days = days_per_month[static_cast<std::size_t>(month - 1)];
    const bool leap = (year % 4 == 0 && year % 100 != 0) ||
        (year % 400 == 0);
    if (month == 2 && leap) {
        ++days;
    }
    return day >= 1 && day <= days;
}

inline std::vector<Row> sorted_rows(const Document &document) {
    auto result = document.rows;
    std::sort(result.begin(), result.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.uid < rhs.uid;
              });
    return result;
}

inline bool value_matches_type(const Value &value, ValueType type) {
    if (std::holds_alternative<NullValue>(value)) {
        return true;
    }
    switch (type) {
    case ValueType::int64:
        return std::holds_alternative<std::int64_t>(value);
    case ValueType::float64:
        return std::holds_alternative<double>(value);
    case ValueType::boolean:
        return std::holds_alternative<bool>(value);
    case ValueType::string:
        return std::holds_alternative<std::string>(value);
    }
    return false;
}

inline void add_frame(std::string &preimage, std::string label,
                      std::string_view type, std::string payload) {
    preimage += canonical_frame(label, type, payload);
}

inline void add_string(std::string &preimage, std::string label,
                       std::string_view value) {
    add_frame(preimage, std::move(label), "utf8", std::string(value));
}

inline void add_int64(std::string &preimage, std::string label,
                      std::int64_t value) {
    add_frame(preimage, std::move(label), "int64", std::to_string(value));
}

inline void add_count(std::string &preimage, std::string label,
                      std::size_t value) {
    add_frame(preimage, std::move(label), "uint64", std::to_string(value));
}

inline void add_bool(std::string &preimage, std::string label, bool value) {
    add_frame(preimage, std::move(label), "bool", value ? "true" : "false");
}

inline void add_float64(std::string &preimage, std::string label,
                        double value) {
    add_frame(preimage, std::move(label), "float64-ieee754",
              canonical_float64_payload(value));
}

inline void add_value(std::string &preimage, std::string label,
                      const Value &value, ValueType declared_type) {
    std::visit(
        [&](const auto &typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, NullValue>) {
                add_frame(preimage, std::move(label),
                          "null-" + std::string(value_type_token(declared_type)),
                          "null");
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                add_int64(preimage, std::move(label), typed);
            } else if constexpr (std::is_same_v<T, double>) {
                add_float64(preimage, std::move(label), typed);
            } else if constexpr (std::is_same_v<T, bool>) {
                add_bool(preimage, std::move(label), typed);
            } else if constexpr (std::is_same_v<T, std::string>) {
                add_string(preimage, std::move(label), typed);
            }
        },
        value);
}

}  // namespace detail

inline void validate(
    const Document &document,
    const FieldRegistry &field_registry = canonical_field_registry_v1()) {
    detail::require_text("profile", document.profile);
    detail::require_text("field registry", document.field_registry);
    detail::require_text("authoritative registry version",
                         field_registry.version);
    if (document.profile != baseline_profile_v1 ||
        document.field_registry != field_registry.version) {
        throw ContractError(
            "canonical APT profile or authoritative field registry mismatch");
    }

    // A caller-supplied registry is a trust root only for a strict extension
    // of the immutable v1 catalog. It cannot weaken or counterfeit the
    // canonical baseline/profile contract.
    const auto field_map = [](const std::vector<RegisteredField> &fields,
                              std::string_view label) {
        std::map<std::string, RegisteredField> result;
        for (const auto &field : fields) {
            if (!result.emplace(field.name, field).second) {
                throw ContractError("duplicate field in " +
                                    std::string(label) + ": " + field.name);
            }
        }
        return result;
    };
    const auto &built_in_registry = canonical_field_registry_v1();
    const auto built_in_required = field_map(
        built_in_registry.required_baseline_fields,
        "canonical v1 required registry");
    const auto supplied_required = field_map(
        field_registry.required_baseline_fields,
        "supplied required registry");
    if (supplied_required != built_in_required) {
        throw ContractError(
            "supplied field registry does not preserve the exact canonical v1 required catalog");
    }
    const auto built_in_optional = field_map(
        built_in_registry.optional_extensions,
        "canonical v1 optional registry");
    const auto supplied_optional = field_map(
        field_registry.optional_extensions,
        "supplied optional registry");
    for (const auto &[name, expected] : built_in_optional) {
        const auto supplied = supplied_optional.find(name);
        if (supplied == supplied_optional.end() ||
            !(supplied->second == expected)) {
            throw ContractError(
                "supplied field registry does not preserve canonical v1 optional field: " +
                name);
        }
    }
    const bool is_built_in_version =
        field_registry.version == field_registry_version_v1;
    if ((is_built_in_version &&
         supplied_optional.size() != built_in_optional.size()) ||
        (!is_built_in_version &&
         supplied_optional.size() <= built_in_optional.size())) {
        throw ContractError(
            "field registry version is not the built-in catalog or a strict extension");
    }
    for (const auto &[name, extension] : supplied_optional) {
        if (!built_in_optional.contains(name) &&
            extension.registry != field_registry.version) {
            throw ContractError(
                "added field is not owned by the supplied extension registry: " +
                name);
        }
    }

    detail::require_text("opaque occurrence", document.envelope.occurrence);
    detail::require_text("event reference", document.envelope.event_reference);
    detail::require_text("output role", document.envelope.output_role);
    detail::require_text("producer", document.envelope.producer);
    detail::require_text("software revision",
                         document.envelope.software_revision);
    detail::require_text("configuration reference",
                         document.envelope.configuration_reference);
    detail::require_text("UTC event time", document.envelope.event_time_utc);
    if (document.envelope.output_role != baseline_output_role_v1 ||
        document.envelope.producer != "citlali" ||
        !detail::is_utc_timestamp(document.envelope.event_time_utc)) {
        throw ContractError(
            "canonical baseline APT envelope role/producer/UTC time is invalid");
    }

    detail::require_text("project id", document.context.project_id);
    detail::require_text("source name", document.context.source_name);
    detail::require_text("observation UTC time",
                         document.context.observation_time_utc);
    detail::require_text("coordinate frame",
                         document.context.coordinate_frame);
    if (!detail::is_utc_timestamp(document.context.observation_time_utc) ||
        document.context.coordinate_frame != "altaz") {
        throw ContractError(
            "canonical baseline APT scientific context time/frame is invalid");
    }

    const auto &observation = document.raw_manifest.observation;
    if (observation.observation < 0 || observation.subobservation < 0 ||
        observation.scan < 0) {
        throw ContractError(
            "canonical APT observation/subobservation/scan must be nonnegative");
    }
    if (document.rows.empty() || document.raw_manifest.inputs.empty()) {
        throw ContractError(
            "canonical APT requires at least one row and raw input");
    }

    std::map<std::string, RegisteredField> authorized;
    std::set<std::string> required_names;
    const auto register_authorized = [&](const RegisteredField &field,
                                         bool required) {
        if (!detail::valid_registered_name(field.name) ||
            detail::protected_contract_name(field.name) ||
            !authorized.emplace(field.name, field).second) {
            throw ContractError(
                "canonical APT authoritative registry has an invalid, protected, or duplicate field");
        }
        detail::require_text("authoritative field unit", field.unit);
        detail::require_text("authoritative field reference",
                             field.authority_reference);
        detail::require_text("authoritative field registry", field.registry);
        detail::require_text("authoritative field description",
                             field.description);
        if ((field.authority == FieldAuthority::unavailable &&
             !field.nullable) ||
            (field.type != ValueType::float64 &&
             field.nonfinite != NonFinitePolicy::reject) ||
            (detail::unresolved_design_name(field.name) &&
             (field.authority != FieldAuthority::unavailable ||
              !field.nullable))) {
            throw ContractError(
                "canonical APT authoritative registry has an inconsistent field contract");
        }
        if (required) {
            required_names.insert(field.name);
        }
    };
    for (const auto &field : field_registry.required_baseline_fields) {
        register_authorized(field, true);
    }
    for (const auto &field : field_registry.optional_extensions) {
        register_authorized(field, false);
    }

    std::map<std::string, RegisteredField> registered;
    for (const auto &field : document.registered_fields) {
        if (!detail::valid_registered_name(field.name) ||
            detail::protected_contract_name(field.name)) {
            throw ContractError(
                "registered field collides with protected canonical APT structure: " +
                field.name);
        }
        detail::require_text("field unit", field.unit);
        detail::require_text("field authority reference",
                             field.authority_reference);
        detail::require_text("field registry", field.registry);
        detail::require_text("field description", field.description);
        if (!registered.emplace(field.name, field).second) {
            throw ContractError("duplicate canonical APT registered field: " +
                                field.name);
        }
        const auto expected = authorized.find(field.name);
        if (expected == authorized.end() || !(field == expected->second)) {
            throw ContractError(
                "canonical APT field is not an exact member of the authoritative registry: " +
                field.name);
        }
        if (field.authority == FieldAuthority::unavailable &&
            !field.nullable) {
            throw ContractError(
                "unavailable canonical APT field must be nullable: " +
                field.name);
        }
        if (field.type != ValueType::float64 &&
            field.nonfinite != NonFinitePolicy::reject) {
            throw ContractError(
                "nonfinite token policy is valid only for float64 field: " +
                field.name);
        }
        if (detail::unresolved_design_name(field.name) &&
            (field.authority != FieldAuthority::unavailable ||
             !field.nullable)) {
            throw ContractError(
                "fg/pg/ori/loc require explicit nullable unavailable authority in v1");
        }
    }
    for (const auto &name : required_names) {
        if (!registered.contains(name)) {
            throw ContractError(
                "canonical baseline APT is missing required field: " + name);
        }
    }

    std::map<std::int64_t, RawInput> raw_inputs;
    std::set<std::string> raw_interfaces;
    std::uint64_t expected_row_count = 0;
    for (const auto &input : document.raw_manifest.inputs) {
        detail::require_text("raw interface", input.interface_name);
        if (input.network < 0 || input.channel_count <= 0 ||
            input.channel_count > uid_v1_max + 1 ||
            !detail::is_canonical_toltec_interface(input.interface_name,
                                                   input.network) ||
            !raw_inputs.emplace(input.network, input).second ||
            !raw_interfaces.insert(input.interface_name).second) {
            throw ContractError(
                "canonical APT raw manifest requires unique canonical TolTEC network/interface inputs with positive channel counts");
        }
        const auto count = static_cast<std::uint64_t>(input.channel_count);
        if (expected_row_count >
            static_cast<std::uint64_t>(uid_v1_max) + 1U - count) {
            throw ContractError(
                "canonical APT raw manifest channel count exceeds v1 capacity");
        }
        expected_row_count += count;
    }
    if (expected_row_count != document.rows.size()) {
        throw ContractError(
            "canonical APT raw manifest channel counts do not cover every row");
    }

    using Relation = std::pair<std::int64_t, std::int64_t>;
    std::set<std::int64_t> uids;
    std::set<Relation> row_relations;
    std::map<std::int64_t, std::uint64_t> rows_per_network;
    for (const auto &row : document.rows) {
        if (row.uid < 0 || row.uid > uid_v1_max) {
            throw ContractError(
                "canonical APT uid is outside exact v1 range [0, 2^53-1]");
        }
        if (!uids.insert(row.uid).second) {
            throw ContractError("duplicate canonical APT artifact-local uid");
        }
        const auto input = raw_inputs.find(row.network);
        if (input == raw_inputs.end() || row.channel < 0 ||
            row.channel >= input->second.channel_count) {
            throw ContractError(
                "canonical APT row references an absent or out-of-range raw channel");
        }
        if (row.array != detail::expected_array_for_network(row.network)) {
            throw ContractError(
                "canonical APT row array disagrees with the v1 network map");
        }
        if (!std::isfinite(row.tone_frequency_hz)) {
            throw ContractError(
                "canonical APT row tone_freq must be finite float64 Hz");
        }
        if (!row_relations.emplace(row.network, row.channel).second) {
            throw ContractError(
                "duplicate canonical APT row network/channel relation");
        }
        ++rows_per_network[row.network];
        if (row.fields.size() != registered.size()) {
            throw ContractError(
                "canonical APT row does not contain exactly the registered fields");
        }
        for (const auto &[name, contract] : registered) {
            const auto value_it = row.fields.find(name);
            if (value_it == row.fields.end()) {
                throw ContractError(
                    "canonical APT row is missing registered field: " + name);
            }
            const auto &value = value_it->second;
            if (!detail::value_matches_type(value, contract.type)) {
                throw ContractError(
                    "canonical APT registered field has wrong exact type: " +
                    name);
            }
            if (std::holds_alternative<NullValue>(value)) {
                if (!contract.nullable) {
                    throw ContractError(
                        "canonical APT nonnullable field is null: " + name);
                }
                continue;
            }
            // Unavailable authority describes the value's trust, not its
            // storage. This preserves current fg/pg/ori/loc values without
            // promoting them to identity or silently inventing provenance.
            if (contract.type == ValueType::float64) {
                const double typed = std::get<double>(value);
                if ((!std::isfinite(typed) &&
                     contract.nonfinite == NonFinitePolicy::reject) ||
                    (std::isinf(typed) &&
                     contract.nonfinite == NonFinitePolicy::nan_token)) {
                    throw ContractError(
                        "canonical APT float64 field rejects nonfinite value: " +
                        name);
                }
            }
            if (contract.type == ValueType::string) {
                const auto &typed = std::get<std::string>(value);
                detail::require_text("non-null row string " + name, typed);
            }
            if (contract.type == ValueType::int64) {
                const auto typed = std::get<std::int64_t>(value);
                if ((name == "flag" && typed != 0 && typed != 1) ||
                    (name == "flag2" && (typed < 0 || typed > 0xff)) ||
                    (name == "converge_iter" && typed < 0) ||
                    ((name == "rfi_masked_samples" ||
                      name == "rfi_masked_scans" ||
                      name == "scan_band_masked_samples" ||
                      name == "scan_band_masked_rows" ||
                      name == "template_npix") &&
                     typed < 0) ||
                    (name == "scan_band_masked_edge" &&
                     (typed < 0 || typed > 3)) ||
                    ((name == "scan_band_mask_rejected" ||
                      name == "cal_amp_method") &&
                     typed != 0 && typed != 1)) {
                    throw ContractError(
                        "canonical APT closed integer field has an invalid value: " +
                        name);
                }
            }
        }
        for (const auto &[name, value] : row.fields) {
            (void)value;
            if (!registered.contains(name)) {
                throw ContractError(
                    "canonical APT row contains unregistered field: " + name);
            }
        }
    }
    for (const auto &[network, input] : raw_inputs) {
        if (rows_per_network[network] !=
            static_cast<std::uint64_t>(input.channel_count)) {
            throw ContractError(
                "canonical APT uid-to-network/channel relation is not a complete manifest bijection");
        }
    }
}

inline std::string semantic_preimage(
    const Document &document,
    const FieldRegistry &field_registry = canonical_field_registry_v1()) {
    validate(document, field_registry);
    std::string preimage;
    detail::add_string(preimage, "encoding", framing_encoding_v1);
    detail::add_string(preimage, "scope", semantic_scope_v1);
    detail::add_string(preimage, "schema", schema_version_v1);
    detail::add_string(preimage, "profile", document.profile);
    detail::add_string(preimage, "field-registry",
                       document.field_registry);

    struct CoreContract {
        std::string_view name;
        std::string_view type;
        std::string_view unit;
        std::string_view authority;
        std::string_view role;
    };
    constexpr std::array<CoreContract, 5> core_contracts{{
        {"uid", "int64", "N/A", "canonical-issuer",
         "artifact-local-row-key"},
        {"tone_freq", "float64", "Hz", "raw-readout",
         "nonidentity-attribute"},
        {"array", "int64", "N/A", "network-map",
         "nonidentity-attribute"},
        {"nw", "int64", "N/A", "raw-manifest",
         "raw-channel-relation"},
        {"kids_tone", "int64", "N/A", "raw-manifest",
         "raw-channel-relation"},
    }};
    detail::add_count(preimage, "core.count", core_contracts.size());
    for (std::size_t index = 0; index < core_contracts.size(); ++index) {
        const auto prefix = "core." + std::to_string(index) + ".";
        const auto &contract = core_contracts[index];
        detail::add_string(preimage, prefix + "name", contract.name);
        detail::add_string(preimage, prefix + "type", contract.type);
        detail::add_string(preimage, prefix + "unit", contract.unit);
        detail::add_bool(preimage, prefix + "nullable", false);
        detail::add_string(preimage, prefix + "authority",
                           contract.authority);
        detail::add_string(preimage, prefix + "identity-role",
                           contract.role);
    }

    const auto fields = detail::sorted_registered_fields(document);
    detail::add_count(preimage, "registered.count", fields.size());
    for (std::size_t index = 0; index < fields.size(); ++index) {
        const auto prefix = "registered." + std::to_string(index) + ".";
        const auto &field = fields[index];
        detail::add_string(preimage, prefix + "name", field.name);
        detail::add_string(preimage, prefix + "type",
                           value_type_token(field.type));
        detail::add_string(preimage, prefix + "unit", field.unit);
        detail::add_bool(preimage, prefix + "nullable", field.nullable);
        detail::add_string(preimage, prefix + "authority",
                           field_authority_token(field.authority));
        detail::add_string(preimage, prefix + "authority-reference",
                           field.authority_reference);
        detail::add_string(preimage, prefix + "nonfinite",
                           nonfinite_policy_token(field.nonfinite));
        detail::add_string(preimage, prefix + "registry", field.registry);
        detail::add_string(preimage, prefix + "description",
                           field.description);
        detail::add_string(preimage, prefix + "identity-role",
                           "nonidentity");
    }

    const auto &observation = document.raw_manifest.observation;
    detail::add_int64(preimage, "observation.observation",
                      observation.observation);
    detail::add_int64(preimage, "observation.subobservation",
                      observation.subobservation);
    detail::add_int64(preimage, "observation.scan", observation.scan);

    detail::add_string(preimage, "context.project-id",
                       document.context.project_id);
    detail::add_string(preimage, "context.source-name",
                       document.context.source_name);
    detail::add_string(preimage, "context.observation-time-utc",
                       document.context.observation_time_utc);
    detail::add_string(preimage, "context.coordinate-frame",
                       document.context.coordinate_frame);

    const auto inputs = detail::sorted_raw_inputs(document);
    detail::add_count(preimage, "raw-input.count", inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto prefix = "raw-input." + std::to_string(index) + ".";
        detail::add_int64(preimage, prefix + "network",
                          inputs[index].network);
        detail::add_string(preimage, prefix + "interface",
                           inputs[index].interface_name);
        detail::add_int64(preimage, prefix + "channel-count",
                          inputs[index].channel_count);
    }
    // Expand the authoritative per-input counts to the exact canonical raw
    // channel inventory. This is sorted by (network, channel), independent of
    // input and table presentation order.
    detail::add_count(preimage, "raw-channel.count", document.rows.size());
    std::size_t raw_index = 0;
    for (const auto &input : inputs) {
        for (std::int64_t channel = 0; channel < input.channel_count;
             ++channel, ++raw_index) {
            const auto prefix =
                "raw-channel." + std::to_string(raw_index) + ".";
            detail::add_int64(preimage, prefix + "network", input.network);
            detail::add_int64(preimage, prefix + "channel", channel);
            detail::add_string(preimage, prefix + "interface",
                               input.interface_name);
        }
    }

    const auto rows = detail::sorted_rows(document);
    detail::add_count(preimage, "row.count", rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto prefix = "row." + std::to_string(index) + ".";
        const auto &row = rows[index];
        detail::add_int64(preimage, prefix + "uid", row.uid);
        detail::add_float64(preimage, prefix + "tone_freq",
                            row.tone_frequency_hz);
        detail::add_int64(preimage, prefix + "array", row.array);
        detail::add_int64(preimage, prefix + "nw", row.network);
        detail::add_int64(preimage, prefix + "kids_tone", row.channel);
        for (const auto &field : fields) {
            detail::add_value(preimage, prefix + "field." + field.name,
                              row.fields.at(field.name), field.type);
        }
    }
    return preimage;
}

inline std::string semantic_sha256(
    const Document &document,
    const FieldRegistry &field_registry = canonical_field_registry_v1()) {
    return "sha256:" +
        citlali::utils::sha256(semantic_preimage(document, field_registry));
}

inline std::string envelope_preimage(
    const Document &document,
    const FieldRegistry &field_registry = canonical_field_registry_v1()) {
    validate(document, field_registry);
    std::string preimage;
    detail::add_string(preimage, "encoding", framing_encoding_v1);
    detail::add_string(preimage, "scope", envelope_scope_v1);
    detail::add_string(preimage, "schema", schema_version_v1);
    detail::add_string(preimage, "profile", document.profile);
    detail::add_string(preimage, "field-registry",
                       document.field_registry);
    detail::add_string(preimage, "semantic-sha256",
                       semantic_sha256(document, field_registry));
    detail::add_string(preimage, "occurrence",
                       document.envelope.occurrence);
    detail::add_string(preimage, "event-reference",
                       document.envelope.event_reference);
    detail::add_string(preimage, "output-role",
                       document.envelope.output_role);
    detail::add_string(preimage, "producer", document.envelope.producer);
    detail::add_string(preimage, "software-revision",
                       document.envelope.software_revision);
    detail::add_string(preimage, "configuration-reference",
                       document.envelope.configuration_reference);
    detail::add_string(preimage, "event-time-utc",
                       document.envelope.event_time_utc);
    return preimage;
}

inline std::string envelope_sha256(
    const Document &document,
    const FieldRegistry &field_registry = canonical_field_registry_v1()) {
    return "sha256:" +
        citlali::utils::sha256(envelope_preimage(document, field_registry));
}

inline Digests compute_digests(
    const Document &document,
    const FieldRegistry &field_registry = canonical_field_registry_v1()) {
    const auto semantic = semantic_sha256(document, field_registry);
    return {semantic, envelope_sha256(document, field_registry)};
}

}  // namespace citlali::pipeline::canonical_apt_v1

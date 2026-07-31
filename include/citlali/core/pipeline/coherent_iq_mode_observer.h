#pragma once

#include <Eigen/Core>
#include <Eigen/QR>
#include <citlali/core/utils/sha256.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *coherent_iq_template_schema_version =
    "citlali-coherent-iq-mode-template-v1";
inline constexpr const char *coherent_iq_diagnostic_schema_version =
    "citlali-coherent-iq-mode-diagnostic-v1";

struct CoherentIqModeTone {
    int uid = -1;
    int tone_slot_zero_based = -1;
    double tone_offset_frequency_hz =
        std::numeric_limits<double>::quiet_NaN();
    std::vector<double> loadings;
};

struct CoherentIqModeTemplate {
    std::string template_id;
    std::string template_version;
    std::string created_at_utc;
    std::string lifecycle_state;
    int network = -1;
    std::string readout_id;
    std::string readout_coordinate_system;
    std::vector<std::string> mode_ids;
    std::vector<CoherentIqModeTone> tones;
    double tone_offset_tolerance_hz = 0.0;
    double minimum_compatible_tone_fraction = 1.0;
    std::map<std::string, std::string> required_metadata;
    std::vector<std::string> unresolved_metadata;
    std::string source_path;
    std::string source_sha256;
};

struct CoherentIqModeScore {
    std::string status = "unavailable";
    std::string template_id;
    std::string template_version;
    int network = -1;
    std::string primary_mode_id;
    double projection_amplitude_mrad =
        std::numeric_limits<double>::quiet_NaN();
    int sign = 0;
    double cosine_similarity = std::numeric_limits<double>::quiet_NaN();
    double absolute_cosine_similarity =
        std::numeric_limits<double>::quiet_NaN();
    double explained_energy_fraction =
        std::numeric_limits<double>::quiet_NaN();
    double residual_energy_mrad2 =
        std::numeric_limits<double>::quiet_NaN();
    double total_energy_mrad2 =
        std::numeric_limits<double>::quiet_NaN();
    double multi_mode_explained_energy_fraction =
        std::numeric_limits<double>::quiet_NaN();
    double common_phase_explained_energy_fraction =
        std::numeric_limits<double>::quiet_NaN();
    double delay_slope_explained_energy_fraction =
        std::numeric_limits<double>::quiet_NaN();
    int compatible_tone_count = 0;
    int template_tone_count = 0;
    double compatible_tone_fraction = 0.0;
    int rejected_tone_count = 0;
    std::string compatibility_note;
};

inline double coherent_iq_zero_baseline_r2(
    const Eigen::VectorXd &values, const Eigen::VectorXd &prediction) {
    const double denominator = values.squaredNorm();
    if (!(std::isfinite(denominator) && denominator > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0 - (values - prediction).squaredNorm() / denominator;
}

inline CoherentIqModeTemplate load_coherent_iq_mode_template(
    const std::string &path) {
    const YAML::Node root = YAML::LoadFile(path);
    if (!root.IsMap() ||
        root["schema_version"].as<std::string>("") !=
            coherent_iq_template_schema_version) {
        throw std::invalid_argument(
            "unsupported coherent-IQ mode-template schema in " + path);
    }

    CoherentIqModeTemplate result;
    result.source_path = path;
    result.source_sha256 = citlali::utils::sha256_file(path);
    result.template_id = root["template_id"].as<std::string>("");
    result.template_version =
        root["template_version"].as<std::string>("");
    result.created_at_utc =
        root["created_at_utc"].as<std::string>("");
    result.lifecycle_state =
        root["lifecycle_state"].as<std::string>("");
    if (result.template_id.empty() || result.template_version.empty() ||
        result.created_at_utc.empty()) {
        throw std::invalid_argument(
            "coherent-IQ template identity is incomplete in " + path);
    }
    if (result.lifecycle_state != "observe_only" &&
        result.lifecycle_state != "experimental") {
        throw std::invalid_argument(
            "coherent-IQ template lifecycle is not observe-only or "
            "experimental in " +
            path);
    }

    const auto identity = root["identity"];
    if (!identity.IsMap() || !identity["network"]) {
        throw std::invalid_argument(
            "coherent-IQ template has no network identity in " + path);
    }
    result.network = identity["network"].as<int>();
    result.readout_id =
        identity["readout_id"].as<std::string>("");
    result.readout_coordinate_system =
        identity["readout_coordinate_system"].as<std::string>("");
    if (result.network < 0 || result.readout_id.empty() ||
        result.readout_coordinate_system !=
            "apt_uid+signed_digital_tone_offset_hz") {
        throw std::invalid_argument(
            "coherent-IQ template readout identity is unsupported in " +
            path);
    }
    for (const auto *section :
         {"training", "validation", "provenance"}) {
        if (!root[section].IsMap()) {
            throw std::invalid_argument(
                "coherent-IQ template is missing required provenance "
                "section " +
                std::string(section) + " in " + path);
        }
    }

    const auto modes = root["modes"];
    if (!modes.IsSequence() || modes.size() == 0) {
        throw std::invalid_argument(
            "coherent-IQ template has no modes in " + path);
    }
    std::set<std::string> mode_ids;
    for (const auto &mode : modes) {
        const auto mode_id = mode["mode_id"].as<std::string>("");
        if (mode_id.empty() || !mode_ids.insert(mode_id).second) {
            throw std::invalid_argument(
                "coherent-IQ template mode IDs are empty or duplicated in " +
                path);
        }
        result.mode_ids.push_back(mode_id);
    }

    const auto tone_coordinate = root["tone_coordinate"];
    if (tone_coordinate["identity_field"].as<std::string>("") != "uid" ||
        tone_coordinate["ordering"].as<std::string>("") !=
            "uid_ascending" ||
        tone_coordinate["frequency_field"].as<std::string>("") !=
            "tone_offset_frequency_hz" ||
        tone_coordinate["frequency_meaning"].as<std::string>("").empty()) {
        throw std::invalid_argument(
            "coherent-IQ template tone-coordinate contract is unsupported in " +
            path);
    }
    const auto tones = tone_coordinate["tones"];
    if (!tones.IsSequence() || tones.size() == 0) {
        throw std::invalid_argument(
            "coherent-IQ template has no tones in " + path);
    }
    std::set<int> uids;
    std::set<int> tone_slots;
    int previous_uid = std::numeric_limits<int>::min();
    for (const auto &tone : tones) {
        CoherentIqModeTone row;
        row.uid = tone["uid"].as<int>();
        row.tone_slot_zero_based =
            tone["tone_slot_zero_based"].as<int>();
        row.tone_offset_frequency_hz =
            tone["tone_offset_frequency_hz"].as<double>();
        if (row.uid <= previous_uid || row.tone_slot_zero_based < 0 ||
            !std::isfinite(row.tone_offset_frequency_hz) ||
            !uids.insert(row.uid).second ||
            !tone_slots.insert(row.tone_slot_zero_based).second) {
            throw std::invalid_argument(
                "coherent-IQ template tone identity/order is invalid in " +
                path);
        }
        previous_uid = row.uid;
        const auto loadings = tone["loadings"];
        for (const auto &mode_id : result.mode_ids) {
            if (!loadings[mode_id]) {
                throw std::invalid_argument(
                    "coherent-IQ template tone is missing a mode loading in " +
                    path);
            }
            const double value = loadings[mode_id].as<double>();
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "coherent-IQ template contains a non-finite loading in " +
                    path);
            }
            row.loadings.push_back(value);
        }
        result.tones.push_back(std::move(row));
    }

    const auto normalization = root["normalization"];
    if (normalization["kind"].as<std::string>("") !=
            "rms_unity_over_template_tones" ||
        normalization["sign_rule"].as<std::string>("") !=
            "largest_absolute_loading_is_positive" ||
        normalization["projection_amplitude_unit"].as<std::string>("") !=
            "mrad RMS phase change") {
        throw std::invalid_argument(
            "coherent-IQ template normalization is unsupported in " + path);
    }
    for (std::size_t mode = 0; mode < result.mode_ids.size(); ++mode) {
        double sum_sq = 0.0;
        double largest_abs = -1.0;
        double anchor_value = 0.0;
        for (const auto &tone : result.tones) {
            const double value = tone.loadings[mode];
            sum_sq += value * value;
            if (std::abs(value) > largest_abs) {
                largest_abs = std::abs(value);
                anchor_value = value;
            }
        }
        const double rms =
            std::sqrt(sum_sq / static_cast<double>(result.tones.size()));
        if (std::abs(rms - 1.0) > 1.0e-6 || anchor_value < 0.0) {
            throw std::invalid_argument(
                "coherent-IQ template violates normalization/sign contract in " +
                path);
        }
    }

    const auto compatibility = root["compatibility"];
    if (!compatibility.IsMap() ||
        compatibility["partial_match_policy"].as<std::string>("") !=
            "explicit_coverage_or_fail_closed") {
        throw std::invalid_argument(
            "coherent-IQ template partial-match policy is unsupported in " +
            path);
    }
    result.tone_offset_tolerance_hz =
        compatibility["tone_offset_tolerance_hz"].as<double>();
    result.minimum_compatible_tone_fraction =
        compatibility["minimum_compatible_tone_fraction"].as<double>();
    if (!std::isfinite(result.tone_offset_tolerance_hz) ||
        result.tone_offset_tolerance_hz < 0.0 ||
        !std::isfinite(result.minimum_compatible_tone_fraction) ||
        result.minimum_compatible_tone_fraction <= 0.0 ||
        result.minimum_compatible_tone_fraction > 1.0) {
        throw std::invalid_argument(
            "coherent-IQ template compatibility bounds are invalid in " +
            path);
    }
    const auto required = compatibility["required_metadata"];
    if (required && required.IsMap()) {
        for (const auto &item : required) {
            result.required_metadata.emplace(
                item.first.as<std::string>(),
                item.second.as<std::string>());
        }
    }
    else if (!required || !required.IsMap()) {
        throw std::invalid_argument(
            "coherent-IQ template required_metadata must be a map in " +
            path);
    }
    const auto unresolved = compatibility["unresolved_metadata"];
    if (!unresolved.IsSequence()) {
        throw std::invalid_argument(
            "coherent-IQ template unresolved_metadata must be a sequence in " +
            path);
    }
    for (const auto &item : unresolved) {
        result.unresolved_metadata.push_back(item.as<std::string>());
    }
    return result;
}

inline CoherentIqModeScore score_coherent_iq_mode_event(
    const CoherentIqModeTemplate &mode_template, int network,
    const std::vector<int> &uids,
    const std::vector<double> &tone_offsets_hz,
    const std::vector<double> &phase_change_mrad,
    const std::map<std::string, std::string> &metadata = {}) {
    CoherentIqModeScore result;
    result.template_id = mode_template.template_id;
    result.template_version = mode_template.template_version;
    result.network = network;
    result.template_tone_count =
        static_cast<int>(mode_template.tones.size());

    if (network != mode_template.network) {
        result.status = "wrong_network";
        result.compatibility_note = "candidate network differs from template";
        return result;
    }
    for (const auto &[key, expected] :
         mode_template.required_metadata) {
        const auto actual = metadata.find(key);
        if (actual == metadata.end() || actual->second != expected) {
            result.status = "incompatible_metadata";
            result.compatibility_note =
                "required metadata mismatch: " + key;
            return result;
        }
    }
    if (uids.size() != tone_offsets_hz.size() ||
        uids.size() != phase_change_mrad.size()) {
        throw std::invalid_argument(
            "coherent-IQ candidate arrays must have equal length");
    }
    std::map<int, std::size_t> candidate_by_uid;
    for (std::size_t index = 0; index < uids.size(); ++index) {
        // Runtime APTs retain unmatched/flagged raw-tone rows so their row
        // ordering continues to match the raw I/Q columns. Those rows have
        // no usable phase measurement and may share a placeholder UID (zero
        // in current matched APTs). Exclude them before enforcing uniqueness;
        // two usable rows with the same UID must still fail closed.
        if (!std::isfinite(tone_offsets_hz[index]) ||
            !std::isfinite(phase_change_mrad[index])) {
            continue;
        }
        if (!candidate_by_uid.emplace(uids[index], index).second) {
            result.status = "incompatible_tone_map";
            result.compatibility_note =
                "usable candidate UIDs are not unique";
            return result;
        }
    }

    std::vector<double> observed;
    std::vector<double> offsets;
    std::vector<std::vector<double>> loading_rows;
    for (const auto &tone : mode_template.tones) {
        const auto found = candidate_by_uid.find(tone.uid);
        if (found == candidate_by_uid.end()) {
            ++result.rejected_tone_count;
            continue;
        }
        const auto index = found->second;
        const double offset = tone_offsets_hz[index];
        const double phase = phase_change_mrad[index];
        if (!std::isfinite(offset) || !std::isfinite(phase) ||
            std::abs(offset - tone.tone_offset_frequency_hz) >
                mode_template.tone_offset_tolerance_hz) {
            ++result.rejected_tone_count;
            continue;
        }
        observed.push_back(phase);
        offsets.push_back(offset);
        loading_rows.push_back(tone.loadings);
    }

    result.compatible_tone_count = static_cast<int>(observed.size());
    result.compatible_tone_fraction =
        mode_template.tones.empty()
            ? 0.0
            : static_cast<double>(observed.size()) /
                  static_cast<double>(mode_template.tones.size());
    const auto minimum_count =
        std::max<std::size_t>(3, mode_template.mode_ids.size() + 1);
    if (observed.size() < minimum_count ||
        result.compatible_tone_fraction <
            mode_template.minimum_compatible_tone_fraction) {
        result.status = "insufficient_compatible_tones";
        result.compatibility_note =
            "compatible tone coverage is below template requirement";
        return result;
    }

    const Eigen::Index n_tones =
        static_cast<Eigen::Index>(observed.size());
    const Eigen::Index n_modes =
        static_cast<Eigen::Index>(mode_template.mode_ids.size());
    Eigen::VectorXd y(n_tones);
    Eigen::VectorXd frequency(n_tones);
    Eigen::MatrixXd mode_matrix(n_tones, n_modes);
    for (Eigen::Index row = 0; row < n_tones; ++row) {
        y(row) = observed[static_cast<std::size_t>(row)];
        frequency(row) = offsets[static_cast<std::size_t>(row)];
        for (Eigen::Index mode = 0; mode < n_modes; ++mode) {
            mode_matrix(row, mode) =
                loading_rows[static_cast<std::size_t>(row)]
                            [static_cast<std::size_t>(mode)];
        }
    }
    result.total_energy_mrad2 = y.squaredNorm();
    if (!(std::isfinite(result.total_energy_mrad2) &&
          result.total_energy_mrad2 > 0.0)) {
        result.status = "zero_event_energy";
        return result;
    }

    Eigen::Index primary_mode = 0;
    Eigen::VectorXd primary_prediction =
        Eigen::VectorXd::Zero(n_tones);
    double best_explained = -1.0;
    for (Eigen::Index mode = 0; mode < n_modes; ++mode) {
        const auto loading = mode_matrix.col(mode);
        const double denominator = loading.squaredNorm();
        const double amplitude = y.dot(loading) / denominator;
        const double cosine =
            y.dot(loading) /
            std::sqrt(result.total_energy_mrad2 * denominator);
        const double explained = cosine * cosine;
        if (explained > best_explained) {
            best_explained = explained;
            primary_mode = mode;
            result.projection_amplitude_mrad = amplitude;
            result.cosine_similarity = cosine;
            result.absolute_cosine_similarity = std::abs(cosine);
            primary_prediction = amplitude * loading;
        }
    }
    result.primary_mode_id =
        mode_template.mode_ids[static_cast<std::size_t>(primary_mode)];
    result.sign = result.projection_amplitude_mrad > 0.0
                      ? 1
                      : (result.projection_amplitude_mrad < 0.0 ? -1 : 0);
    result.explained_energy_fraction = best_explained;
    result.residual_energy_mrad2 =
        (y - primary_prediction).squaredNorm();

    const Eigen::VectorXd coefficients =
        mode_matrix.colPivHouseholderQr().solve(y);
    result.multi_mode_explained_energy_fraction =
        coherent_iq_zero_baseline_r2(y, mode_matrix * coefficients);
    result.common_phase_explained_energy_fraction =
        coherent_iq_zero_baseline_r2(
            y, Eigen::VectorXd::Constant(n_tones, y.mean()));

    const double mean_offset = frequency.mean();
    const double offset_scale =
        std::sqrt((frequency.array() - mean_offset).square().mean());
    if (std::isfinite(offset_scale) && offset_scale > 0.0) {
        Eigen::MatrixXd design(n_tones, 2);
        design.col(0).setOnes();
        design.col(1) =
            (frequency.array() - mean_offset) / offset_scale;
        const Eigen::VectorXd delay_coefficients =
            design.colPivHouseholderQr().solve(y);
        result.delay_slope_explained_energy_fraction =
            coherent_iq_zero_baseline_r2(
                y, design * delay_coefficients);
    }
    result.status = "scored";
    return result;
}

}  // namespace citlali::pipeline

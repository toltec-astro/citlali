#include <citlali/core/pipeline/coherent_iq_mode_observer.h>
#include <citlali/core/pipeline/coherent_iq_mode_sidecar.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

citlali::pipeline::CoherentIqModeTemplate make_template(
    int network = 8, int tone_count = 8, int mode_count = 1) {
    citlali::pipeline::CoherentIqModeTemplate result;
    result.template_id = "synthetic-nw" + std::to_string(network);
    result.template_version = "test-v1";
    result.lifecycle_state = "observe_only";
    result.network = network;
    result.tone_offset_tolerance_hz = 1.0;
    result.minimum_compatible_tone_fraction = 0.75;
    for (int mode = 0; mode < mode_count; ++mode) {
        result.mode_ids.push_back("mode_" + std::to_string(mode + 1));
    }
    for (int tone = 0; tone < tone_count; ++tone) {
        citlali::pipeline::CoherentIqModeTone row;
        row.uid = 100 + tone;
        row.tone_slot_zero_based = tone;
        row.tone_offset_frequency_hz =
            -3.5e6 + 1.0e6 * static_cast<double>(tone);
        const double alternating = tone % 2 == 0 ? 1.0 : -1.0;
        row.loadings.push_back(alternating);
        if (mode_count > 1) {
            row.loadings.push_back(
                tone < tone_count / 2 ? 1.0 : -1.0);
        }
        result.tones.push_back(row);
    }
    return result;
}

template <class Template>
std::vector<int> uids(const Template &mode_template) {
    std::vector<int> result;
    for (const auto &tone : mode_template.tones) {
        result.push_back(tone.uid);
    }
    return result;
}

template <class Template>
std::vector<double> offsets(const Template &mode_template) {
    std::vector<double> result;
    for (const auto &tone : mode_template.tones) {
        result.push_back(tone.tone_offset_frequency_hz);
    }
    return result;
}

template <class Template>
std::vector<double> mode_vector(
    const Template &mode_template, std::size_t mode, double amplitude) {
    std::vector<double> result;
    for (const auto &tone : mode_template.tones) {
        result.push_back(amplitude * tone.loadings[mode]);
    }
    return result;
}

void write_template_fixture(
    const std::filesystem::path &path, bool valid_normalization) {
    YAML::Node root;
    root["schema_version"] =
        citlali::pipeline::coherent_iq_template_schema_version;
    root["template_id"] = "fixture-nw8";
    root["template_version"] = "fixture-v1";
    root["created_at_utc"] = "2026-07-30T12:00:00+00:00";
    root["lifecycle_state"] = "observe_only";
    root["identity"]["network"] = 8;
    root["identity"]["readout_id"] = "toltec-nw8";
    root["identity"]["readout_coordinate_system"] =
        "apt_uid+signed_digital_tone_offset_hz";
    root["tone_coordinate"]["identity_field"] = "uid";
    root["tone_coordinate"]["ordering"] = "uid_ascending";
    root["tone_coordinate"]["frequency_field"] =
        "tone_offset_frequency_hz";
    root["tone_coordinate"]["frequency_meaning"] =
        "signed digital tone offset from network LO";
    root["normalization"]["kind"] =
        "rms_unity_over_template_tones";
    root["normalization"]["sign_rule"] =
        "largest_absolute_loading_is_positive";
    root["normalization"]["projection_amplitude_unit"] =
        "mrad RMS phase change";
    root["modes"][0]["mode_id"] = "phase_mode_1";
    root["modes"][0]["rank"] = 1;
    root["modes"][0]["anchor_uid"] = 100;
    for (int tone = 0; tone < 8; ++tone) {
        auto row = root["tone_coordinate"]["tones"][tone];
        row["uid"] = 100 + tone;
        row["tone_slot_zero_based"] = tone;
        row["tone_offset_frequency_hz"] =
            -3.5e6 + 1.0e6 * static_cast<double>(tone);
        row["probe_frequency_hz"] =
            700.0e6 + 1.0e6 * static_cast<double>(tone);
        row["loadings"]["phase_mode_1"] =
            tone == 0 && !valid_normalization
                ? 2.0
                : (tone % 2 == 0 ? 1.0 : -1.0);
    }
    root["training"] = YAML::Node(YAML::NodeType::Map);
    root["compatibility"]["tone_offset_tolerance_hz"] = 1.0;
    root["compatibility"]["minimum_compatible_tone_fraction"] = 0.75;
    root["compatibility"]["required_metadata"] =
        YAML::Node(YAML::NodeType::Map);
    root["compatibility"]["partial_match_policy"] =
        "explicit_coverage_or_fail_closed";
    root["compatibility"]["unresolved_metadata"].push_back(
        "firmware_state");
    root["validation"] = YAML::Node(YAML::NodeType::Map);
    root["provenance"] = YAML::Node(YAML::NodeType::Map);
    std::ofstream output(path);
    output << root;
}

}  // namespace

TEST(coherent_iq_mode_observer, exact_rank_one_amplitude_and_sign) {
    const auto mode_template = make_template();
    for (const double amplitude : {7.0, -3.5}) {
        const auto score =
            citlali::pipeline::score_coherent_iq_mode_event(
                mode_template, 8, uids(mode_template),
                offsets(mode_template),
                mode_vector(mode_template, 0, amplitude));
        EXPECT_EQ(score.status, "scored");
        EXPECT_NEAR(score.projection_amplitude_mrad, amplitude, 1.0e-12);
        EXPECT_EQ(score.sign, amplitude > 0.0 ? 1 : -1);
        EXPECT_NEAR(score.absolute_cosine_similarity, 1.0, 1.0e-12);
        EXPECT_NEAR(score.explained_energy_fraction, 1.0, 1.0e-12);
        EXPECT_NEAR(score.residual_energy_mrad2, 0.0, 1.0e-12);
    }
}

TEST(coherent_iq_mode_observer, partial_tones_obey_coverage_contract) {
    const auto mode_template = make_template();
    auto candidate_uids = uids(mode_template);
    auto candidate_offsets = offsets(mode_template);
    auto phase = mode_vector(mode_template, 0, 4.0);
    candidate_uids.resize(6);
    candidate_offsets.resize(6);
    phase.resize(6);
    EXPECT_EQ(
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, candidate_uids, candidate_offsets, phase)
            .status,
        "scored");
    candidate_uids.resize(5);
    candidate_offsets.resize(5);
    phase.resize(5);
    EXPECT_EQ(
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, candidate_uids, candidate_offsets, phase)
            .status,
        "insufficient_compatible_tones");
}

TEST(coherent_iq_mode_observer, incompatible_tone_map_fails_closed) {
    const auto mode_template = make_template();
    auto candidate_uids = uids(mode_template);
    auto candidate_offsets = offsets(mode_template);
    auto phase = mode_vector(mode_template, 0, 4.0);
    candidate_uids[1] = candidate_uids[0];
    EXPECT_EQ(
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, candidate_uids, candidate_offsets, phase)
            .status,
        "incompatible_tone_map");

    candidate_uids = uids(mode_template);
    candidate_offsets = offsets(mode_template);
    auto reordered_phase = phase;
    std::reverse(candidate_uids.begin(), candidate_uids.end());
    std::reverse(candidate_offsets.begin(), candidate_offsets.end());
    std::reverse(reordered_phase.begin(), reordered_phase.end());
    const auto reordered =
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template,
            8,
            candidate_uids,
            candidate_offsets,
            reordered_phase);
    EXPECT_EQ(reordered.status, "scored");
    EXPECT_NEAR(reordered.projection_amplitude_mrad, 4.0, 1.0e-12);

    candidate_uids = uids(mode_template);
    candidate_offsets = offsets(mode_template);
    candidate_offsets[0] += 2.0;
    candidate_offsets[1] += 2.0;
    candidate_offsets[2] += 2.0;
    const auto incompatible =
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, candidate_uids, candidate_offsets, phase);
    EXPECT_EQ(incompatible.status, "insufficient_compatible_tones");
}

TEST(coherent_iq_mode_observer, distinguishes_local_common_and_delay_models) {
    const auto mode_template = make_template();
    auto local = std::vector<double>(8, 0.0);
    local[0] = 20.0;
    const auto local_score =
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, uids(mode_template),
            offsets(mode_template), local);
    EXPECT_EQ(local_score.status, "scored");
    EXPECT_LT(local_score.explained_energy_fraction, 0.2);

    const auto common = std::vector<double>(8, 5.0);
    const auto common_score =
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, uids(mode_template),
            offsets(mode_template), common);
    EXPECT_NEAR(
        common_score.common_phase_explained_energy_fraction, 1.0,
        1.0e-12);
    EXPECT_NEAR(common_score.explained_energy_fraction, 0.0, 1.0e-12);

    std::vector<double> delay;
    for (const auto offset : offsets(mode_template)) {
        delay.push_back(2.0 + offset / 1.0e6);
    }
    const auto delay_score =
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, uids(mode_template),
            offsets(mode_template), delay);
    EXPECT_NEAR(
        delay_score.delay_slope_explained_energy_fraction, 1.0,
        1.0e-12);
}

TEST(coherent_iq_mode_observer, multi_mode_metric_explains_two_mode_event) {
    const auto mode_template = make_template(8, 8, 2);
    auto phase = mode_vector(mode_template, 0, 2.0);
    const auto secondary = mode_vector(mode_template, 1, 3.0);
    for (std::size_t i = 0; i < phase.size(); ++i) {
        phase[i] += secondary[i];
    }
    const auto score =
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, uids(mode_template),
            offsets(mode_template), phase);
    EXPECT_EQ(score.status, "scored");
    EXPECT_NEAR(
        score.multi_mode_explained_energy_fraction, 1.0, 1.0e-12);
    EXPECT_LT(score.explained_energy_fraction, 1.0);
}

TEST(coherent_iq_mode_observer, null_wrong_network_and_metadata_fail_closed) {
    auto mode_template = make_template();
    const auto null_score =
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, uids(mode_template),
            offsets(mode_template), std::vector<double>(8, 0.0));
    EXPECT_EQ(null_score.status, "zero_event_energy");
    EXPECT_EQ(
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 7, uids(mode_template),
            offsets(mode_template),
            mode_vector(mode_template, 0, 1.0))
            .status,
        "wrong_network");

    mode_template.required_metadata["firmware"] = "known";
    EXPECT_EQ(
        citlali::pipeline::score_coherent_iq_mode_event(
            mode_template, 8, uids(mode_template),
            offsets(mode_template),
            mode_vector(mode_template, 0, 1.0))
            .status,
        "incompatible_metadata");
}

TEST(coherent_iq_mode_observer, scoring_does_not_mutate_candidate_data) {
    const auto mode_template = make_template();
    const auto candidate_uids = uids(mode_template);
    const auto candidate_offsets = offsets(mode_template);
    const auto phase = mode_vector(mode_template, 0, 4.0);
    const auto original_uids = candidate_uids;
    const auto original_offsets = candidate_offsets;
    const auto original_phase = phase;
    (void)citlali::pipeline::score_coherent_iq_mode_event(
        mode_template, 8, candidate_uids, candidate_offsets, phase);
    EXPECT_EQ(candidate_uids, original_uids);
    EXPECT_EQ(candidate_offsets, original_offsets);
    EXPECT_EQ(phase, original_phase);
}

TEST(coherent_iq_mode_observer, candidate_records_are_compact_and_deterministic) {
    using citlali::pipeline::CoherentIqCandidatePoint;
    std::vector<CoherentIqCandidatePoint> points{
        {10.00, 5.0, "step"},
        {10.02, 7.0, "impulsive"},
        {11.00, 6.0, "step"},
        {12.00, 4.0, "step"},
    };
    const auto first =
        citlali::pipeline::cluster_coherent_iq_candidates(
            1, 8, points, 0.05, 2);
    std::reverse(points.begin(), points.end());
    const auto second =
        citlali::pipeline::cluster_coherent_iq_candidates(
            1, 8, points, 0.05, 2);
    ASSERT_EQ(first.size(), 2U);
    ASSERT_EQ(second.size(), first.size());
    EXPECT_DOUBLE_EQ(first[0].time_unix_sec, 10.01);
    EXPECT_EQ(first[0].supporting_detector_events, 2);
    EXPECT_EQ(first[0].candidate_kinds, "impulsive+step");
    for (std::size_t index = 0; index < first.size(); ++index) {
        EXPECT_DOUBLE_EQ(
            first[index].time_unix_sec, second[index].time_unix_sec);
        EXPECT_EQ(
            first[index].supporting_detector_events,
            second[index].supporting_detector_events);
    }
}

TEST(coherent_iq_mode_observer, cross_network_candidates_seed_one_shared_event) {
    using citlali::pipeline::CoherentIqCandidate;
    std::vector<CoherentIqCandidate> candidates{
        {2, 1, 100.00, 14, 8.0, "step", 1, "1"},
        {2, 2, 100.04, 22, 12.0, "impulsive", 1, "2"},
        {2, 8, 100.02, 18, 9.0, "step", 1, "8"},
        {3, 2, 101.00, 10, 6.0, "step", 1, "2"},
    };
    const auto events =
        citlali::pipeline::
            cluster_coherent_iq_candidates_across_networks(
                candidates, 0.1);

    ASSERT_EQ(events.size(), 2U);
    EXPECT_EQ(events[0].scan_one_based, 2);
    EXPECT_DOUBLE_EQ(events[0].time_unix_sec, 100.02);
    EXPECT_EQ(events[0].supporting_detector_events, 54);
    EXPECT_DOUBLE_EQ(events[0].maximum_rtc_score, 12.0);
    EXPECT_EQ(events[0].candidate_kinds, "impulsive+step");
    EXPECT_EQ(events[0].seed_network_count, 3);
    EXPECT_EQ(events[0].seed_networks, "1 2 8");
    EXPECT_EQ(events[1].scan_one_based, 3);
}

TEST(coherent_iq_mode_observer, template_loader_enforces_versioned_contract) {
    const auto root = std::filesystem::temp_directory_path() /
                      "citlali-coherent-iq-template-test";
    std::filesystem::create_directories(root);
    const auto valid_path = root / "valid.yaml";
    const auto invalid_path = root / "invalid.yaml";
    write_template_fixture(valid_path, true);
    write_template_fixture(invalid_path, false);

    const auto loaded =
        citlali::pipeline::load_coherent_iq_mode_template(
            valid_path.string());
    EXPECT_EQ(loaded.network, 8);
    EXPECT_EQ(loaded.readout_id, "toltec-nw8");
    EXPECT_EQ(loaded.tones.size(), 8U);
    EXPECT_EQ(loaded.unresolved_metadata.size(), 1U);
    EXPECT_EQ(loaded.source_sha256.size(), 64U);
    EXPECT_THROW(
        citlali::pipeline::load_coherent_iq_mode_template(
            invalid_path.string()),
        std::invalid_argument);

    std::filesystem::remove_all(root);
}

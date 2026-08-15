#include <citlali/core/mapmaking/oof_transfer_reference.h>
#include <citlali/core/timestream/rtc/response_manifest.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using timestream::RTCArtifactIdentity;
using timestream::RTCManifestObjectState;
using timestream::RTCResponseManifest;
using timestream::RTCResponseRole;
using timestream::RTCResponseStage;
using timestream::RTCResponseStageSlot;
using timestream::RTCResponseStageState;
using timestream::RTCResponseState;
using timestream::RTCUnavailableReason;
using timestream::RTCUnavailableReasonCode;

std::string digest(char digit = 'a') {
    return "sha256:" + std::string(64, digit);
}

RTCArtifactIdentity artifact(const std::string &id, char digit = 'a') {
    return RTCArtifactIdentity{id, digest(digit), std::uint64_t{17}};
}

RTCResponseStage make_stage(RTCResponseStageSlot slot, std::size_t order) {
    RTCResponseStage stage;
    stage.slot = slot;
    stage.stage_id = timestream::rtc_response_stage_slot_name(slot);
    stage.stage_version = "v1";
    stage.enabled = true;
    stage.applicable = true;
    stage.order = order;
    stage.response_role = RTCResponseRole::signal_changing;
    stage.input_parents = {artifact("stage-input", '1')};
    stage.output_parent = artifact("stage-output", '2');
    stage.operator_identity = "operator:v1";
    stage.normalization_state = "normalization:unchanged";
    stage.grid_identity = artifact("grid", '3');
    stage.wcs_identity = artifact("wcs", '4');
    stage.support_identity = artifact("support", '5');
    stage.validity_identity = artifact("validity", '6');
    stage.response_state = RTCResponseStageState::complete;
    stage.artifact = artifact("stage-artifact", '7');
    return stage;
}

RTCResponseManifest make_manifest() {
    RTCResponseManifest manifest;
    manifest.input_observation = artifact("observation", '0');
    manifest.input_reduction = artifact("reduction", '1');
    manifest.detector = artifact("detectors", '2');
    manifest.cohort = artifact("cohort", '3');
    manifest.scan = artifact("scan", '4');
    manifest.sample_origin = artifact("sample-origin", '5');
    manifest.native_time_origin = artifact("native-time-origin", '6');
    manifest.grid = artifact("grid", '7');
    manifest.wcs = artifact("wcs", '8');
    manifest.support = artifact("support", '9');
    manifest.validity = artifact("validity", 'a');
    manifest.input_parents = {artifact("input-parent", 'b')};
    manifest.requested_state = "requested:v1";
    manifest.effective_state = "effective:v1";
    manifest.observation_resolved_state = "observation-resolved:v1";
    manifest.realized_state = "realized:v1";
    manifest.output_stage_identity = "rtc_conditioned_inner_terminal";
    manifest.response_state = RTCResponseState::unavailable;
    manifest.unavailable_reasons = {
        {RTCUnavailableReasonCode::omitted_enabled_response_stage,
         "donor_replacement", "46ad2388:rtcproc.h:946-974"},
        {RTCUnavailableReasonCode::omitted_enabled_response_stage,
         "altaz_projection", "46ad2388:rtcproc.h:1316"},
        {RTCUnavailableReasonCode::incomplete_response_provenance,
         "support_eligibility_causes_and_realized_response_identity",
         "46ad2388:rtcproc.h:897-1322"},
    };
    manifest.normalization_at_rtc_terminal_ptc_boundary =
        "normalization:rtc-terminal";
    manifest.normalization_before_jinc_consumption =
        "normalization:before-jinc";
    manifest.final_manifest_digest = digest('c');
    manifest.final_manifest_byte_count = std::uint64_t{4096};
    manifest.final_manifest_bundle_identities = {"bundle:rtc-response:v1"};
    manifest.final_manifest_member_identities = {"member:manifest:v1"};
    manifest.object_state = RTCManifestObjectState::complete;
    for (std::size_t i = 0; i < timestream::rtc_response_stage_slots.size();
         ++i) {
        manifest.stages.push_back(make_stage(
            timestream::rtc_response_stage_slots[i], i + 1));
    }
    manifest.stages[3].response_state = RTCResponseStageState::partial;
    manifest.stages[12].response_state = RTCResponseStageState::partial;
    return manifest;
}

TEST(SciMap003RtcResponseManifest, FrozenStageOrderHasFifteenSlots) {
    ASSERT_EQ(timestream::rtc_response_stage_slots.size(), 15U);
    EXPECT_EQ(timestream::rtc_response_stage_slots.front(),
              RTCResponseStageSlot::enabled_pre_calibration_response_changes);
    EXPECT_EQ(timestream::rtc_response_stage_slots[13],
              RTCResponseStageSlot::rtc_conditioned_inner_terminal);
    EXPECT_EQ(timestream::rtc_response_stage_slots.back(),
              RTCResponseStageSlot::ptc_consumer_boundary_link);
}

TEST(SciMap003RtcResponseManifest,
     GoverningBaseRemainsUnavailableForExactControlledReasons) {
    const auto manifest = make_manifest();
    const auto result =
        timestream::validate_rtc_response_manifest_at_sci_map_003_governing_base(
            manifest);
    EXPECT_TRUE(result.valid) << result.summary();
    ASSERT_EQ(manifest.unavailable_reasons.size(), 3U);
    EXPECT_EQ(manifest.unavailable_reasons[0].affected_object,
              "donor_replacement");
    EXPECT_EQ(manifest.unavailable_reasons[1].affected_object,
              "altaz_projection");
    EXPECT_EQ(manifest.unavailable_reasons[2].code,
              RTCUnavailableReasonCode::incomplete_response_provenance);
}

TEST(SciMap003RtcResponseManifest,
     GoverningBaseCannotPromoteACompleteResponse) {
    auto manifest = make_manifest();
    manifest.response_state = RTCResponseState::complete;
    manifest.unavailable_reasons.clear();
    for (auto &stage : manifest.stages) {
        stage.response_state = RTCResponseStageState::complete;
    }
    const auto generic =
        timestream::validate_rtc_response_manifest(manifest);
    EXPECT_TRUE(generic.valid) << generic.summary();
    const auto governing =
        timestream::validate_rtc_response_manifest_at_sci_map_003_governing_base(
            manifest);
    EXPECT_FALSE(governing.valid);
    EXPECT_TRUE(governing.contains(
        "governing application RTC response must remain unavailable"));
}

TEST(SciMap003RtcResponseManifest, CompleteResponseRejectsPartialStages) {
    auto manifest = make_manifest();
    manifest.response_state = RTCResponseState::complete;
    manifest.unavailable_reasons.clear();
    const auto result = timestream::validate_rtc_response_manifest(manifest);
    EXPECT_FALSE(result.valid);
    EXPECT_TRUE(result.contains("complete response contains a non-complete stage"));
}

TEST(SciMap003RtcResponseManifest, InvalidInputsFailClosed) {
    using Mutation = std::pair<std::string,
                               std::function<void(RTCResponseManifest &)>>;
    const std::vector<Mutation> mutations{
        {"observation identity", [](auto &m) { m.input_observation.id.clear(); }},
        {"observation byte count",
         [](auto &m) { m.input_observation.byte_count.reset(); }},
        {"reduction digest", [](auto &m) { m.input_reduction.digest = "bad"; }},
        {"detector identity", [](auto &m) { m.detector.id.clear(); }},
        {"cohort identity", [](auto &m) { m.cohort.id.clear(); }},
        {"scan identity", [](auto &m) { m.scan.id.clear(); }},
        {"sample origin", [](auto &m) { m.sample_origin.id.clear(); }},
        {"native-time origin", [](auto &m) { m.native_time_origin.id.clear(); }},
        {"grid identity", [](auto &m) { m.grid.id.clear(); }},
        {"WCS identity", [](auto &m) { m.wcs.id.clear(); }},
        {"support identity", [](auto &m) { m.support.id.clear(); }},
        {"validity identity", [](auto &m) { m.validity.id.clear(); }},
        {"input parents", [](auto &m) { m.input_parents.clear(); }},
        {"input parent digest",
         [](auto &m) { m.input_parents.front().digest = "bad"; }},
        {"requested state", [](auto &m) { m.requested_state.clear(); }},
        {"effective state", [](auto &m) { m.effective_state.clear(); }},
        {"observation-resolved state",
         [](auto &m) { m.observation_resolved_state.clear(); }},
        {"realized state", [](auto &m) { m.realized_state.clear(); }},
        {"terminal stage",
         [](auto &m) { m.output_stage_identity = "generic_rtc"; }},
        {"unavailable reasons", [](auto &m) { m.unavailable_reasons.clear(); }},
        {"reason evidence",
         [](auto &m) { m.unavailable_reasons.front().evidence.clear(); }},
        {"reason affected object",
         [](auto &m) { m.unavailable_reasons.front().affected_object.clear(); }},
        {"complete response with unavailable reasons",
         [](auto &m) {
             m.response_state = RTCResponseState::complete;
             for (auto &stage : m.stages) {
                 stage.response_state = RTCResponseStageState::complete;
             }
         }},
        {"RTC normalization",
         [](auto &m) {
             m.normalization_at_rtc_terminal_ptc_boundary.clear();
         }},
        {"JINC normalization",
         [](auto &m) { m.normalization_before_jinc_consumption.clear(); }},
        {"manifest digest", [](auto &m) { m.final_manifest_digest = "bad"; }},
        {"manifest byte count",
         [](auto &m) { m.final_manifest_byte_count.reset(); }},
        {"bundle identities",
         [](auto &m) { m.final_manifest_bundle_identities.clear(); }},
        {"member identities",
         [](auto &m) { m.final_manifest_member_identities.clear(); }},
        {"failed object",
         [](auto &m) { m.object_state = RTCManifestObjectState::failed; }},
        {"stage count", [](auto &m) { m.stages.pop_back(); }},
        {"stage order", [](auto &m) { std::swap(m.stages[0], m.stages[1]); }},
        {"stage ordinal", [](auto &m) { m.stages[0].order = 0; }},
        {"stage id", [](auto &m) { m.stages[0].stage_id.clear(); }},
        {"stage version", [](auto &m) { m.stages[0].stage_version.clear(); }},
        {"stage input", [](auto &m) { m.stages[0].input_parents.clear(); }},
        {"stage output parent",
         [](auto &m) { m.stages[0].output_parent.id.clear(); }},
        {"stage operator", [](auto &m) { m.stages[0].operator_identity.clear(); }},
        {"stage normalization",
         [](auto &m) { m.stages[0].normalization_state.clear(); }},
        {"stage grid", [](auto &m) { m.stages[0].grid_identity.id.clear(); }},
        {"stage WCS", [](auto &m) { m.stages[0].wcs_identity.id.clear(); }},
        {"stage support",
         [](auto &m) { m.stages[0].support_identity.id.clear(); }},
        {"stage validity",
         [](auto &m) { m.stages[0].validity_identity.id.clear(); }},
        {"stage artifact", [](auto &m) { m.stages[0].artifact.digest = "bad"; }},
    };
    for (const auto &[label, mutate] : mutations) {
        SCOPED_TRACE(label);
        auto manifest = make_manifest();
        mutate(manifest);
        const auto result = timestream::validate_rtc_response_manifest(manifest);
        EXPECT_FALSE(result.valid);
    }
}

mapmaking::DiscreteGInput make_g_input() {
    mapmaking::DiscreteGInput input;
    for (const auto group : mapmaking::discrete_g_field_groups) {
        input.field_groups.push_back(
            {group,
             std::string(mapmaking::discrete_g_field_group_name(group)),
             digest('d')});
    }
    input.grid.rows = 7;
    input.grid.cols = 9;
    input.grid.wcs_linear_arcsec_per_pixel << -0.8, 0.25, -0.15, 1.1;
    input.grid.crpix1_fits = 4.25;
    input.grid.crpix2_fits = 3.75;
    input.grid.crval1 = 0.0;
    input.grid.crval2 = 0.0;
    input.grid.frame = mapmaking::DiscreteGFrame::altaz_tangent_plane;
    input.grid.pixelization = mapmaking::DiscreteGPixelization::pixel_center_sampled;
    input.grid.signal_unit = mapmaking::DiscreteGSignalUnit::mjy_per_beam;
    input.grid.axis1 = "AZOFFSET";
    input.grid.axis2 = "ELOFFSET";
    input.grid.angular_unit = "arcsec";
    input.grid.epoch = "observation-resolved";
    input.denominator = Eigen::MatrixXd::Constant(7, 9, 3.5);

    mapmaking::DiscreteGDetectorComponent first;
    first.apt_row_identity = "apt-row:101";
    first.detector_identity = "uid:4101";
    first.fwhm_arcsec = 2.4;
    first.weights = Eigen::MatrixXd::Constant(7, 9, 1.25);
    mapmaking::DiscreteGDetectorComponent second;
    second.apt_row_identity = "apt-row:203";
    second.detector_identity = "uid:4102";
    second.fwhm_arcsec = 3.1;
    second.weights = Eigen::MatrixXd::Constant(7, 9, 0.75);
    for (Eigen::Index row = 0; row < second.weights.rows(); ++row) {
        for (Eigen::Index col = 0; col < second.weights.cols(); ++col) {
            second.weights(row, col) +=
                0.01 * static_cast<double>(row + 2 * col);
        }
    }
    input.detectors = {std::move(first), std::move(second)};
    return input;
}

Eigen::MatrixXd independent_g_reconstruction(
    const mapmaking::DiscreteGInput &input) {
    Eigen::MatrixXd result(input.grid.rows, input.grid.cols);
    const long double fwhm_to_sigma =
        1.0L / (2.0L * std::sqrt(2.0L * std::log(2.0L)));
    for (Eigen::Index row = 0; row < input.grid.rows; ++row) {
        for (Eigen::Index col = 0; col < input.grid.cols; ++col) {
            const long double dc =
                static_cast<long double>(col) -
                (static_cast<long double>(input.grid.crpix1_fits) - 1.0L);
            const long double dr =
                static_cast<long double>(row) -
                (static_cast<long double>(input.grid.crpix2_fits) - 1.0L);
            const long double x =
                static_cast<long double>(input.grid.wcs_linear_arcsec_per_pixel(0, 0)) * dc +
                static_cast<long double>(input.grid.wcs_linear_arcsec_per_pixel(0, 1)) * dr;
            const long double y =
                static_cast<long double>(input.grid.wcs_linear_arcsec_per_pixel(1, 0)) * dc +
                static_cast<long double>(input.grid.wcs_linear_arcsec_per_pixel(1, 1)) * dr;
            const long double rho2 = x * x + y * y;
            long double numerator = 0.0L;
            for (const auto &detector : input.detectors) {
                const long double sigma =
                    static_cast<long double>(detector.fwhm_arcsec) * fwhm_to_sigma;
                const long double limit2 = 9.0L * sigma * sigma;
                const long double phi =
                    rho2 <= limit2
                        ? std::exp(-rho2 / (2.0L * sigma * sigma))
                        : 0.0L;
                numerator +=
                    static_cast<long double>(detector.weights(row, col)) * phi;
            }
            result(row, col) = static_cast<double>(
                numerator / static_cast<long double>(input.denominator(row, col)));
        }
    }
    return result;
}

TEST(SciMap003DiscreteG, IndependentReconstructionMeetsPreregisteredBound) {
    const auto input = make_g_input();
    const auto expected = independent_g_reconstruction(input);
    const auto result = mapmaking::render_discrete_g(input);
    ASSERT_EQ(result.state, mapmaking::DiscreteGState::available);
    ASSERT_TRUE(result.unavailable_reasons.empty());
    ASSERT_EQ(result.plane.rows(), expected.rows());
    ASSERT_EQ(result.plane.cols(), expected.cols());
    const double scale = std::max(1.0, expected.cwiseAbs().maxCoeff());
    const double bound =
        256.0 * std::numeric_limits<double>::epsilon() *
        std::max<std::size_t>(1, input.detectors.size()) * scale;
    const double max_abs_error =
        (result.plane - expected).cwiseAbs().maxCoeff();
    std::ostringstream error_text;
    error_text.precision(17);
    error_text << max_abs_error;
    std::ostringstream bound_text;
    bound_text.precision(17);
    bound_text << bound;
    RecordProperty("independent_reconstruction_max_abs_error",
                   error_text.str());
    RecordProperty("preregistered_binary64_bound", bound_text.str());
    EXPECT_LE(max_abs_error, bound);
}

TEST(SciMap003DiscreteG,
     CenterAmplitudeInclusiveThreeSigmaAndDivideOnceAreExact) {
    auto input = make_g_input();
    input.grid.wcs_linear_arcsec_per_pixel.setIdentity();
    input.grid.crpix1_fits = 5.0;
    input.grid.crpix2_fits = 4.0;
    input.denominator.setConstant(4.0);
    input.detectors.resize(1);
    auto &detector = input.detectors.front();
    detector.fwhm_arcsec = 2.0 * std::sqrt(2.0 * std::log(2.0));
    detector.weights.setConstant(2.0);
    const auto result = mapmaking::render_discrete_g(input);
    ASSERT_EQ(result.state, mapmaking::DiscreteGState::available);
    EXPECT_DOUBLE_EQ(result.plane(3, 4), 0.5);
    EXPECT_DOUBLE_EQ(result.plane(3, 7), 0.5 * std::exp(-4.5));
    EXPECT_DOUBLE_EQ(result.plane(3, 8), 0.0);
}

TEST(SciMap003DiscreteG, MissingAnyFrozenFieldGroupFailsClosed) {
    for (std::size_t missing = 0;
         missing < mapmaking::discrete_g_field_groups.size(); ++missing) {
        SCOPED_TRACE(missing);
        auto input = make_g_input();
        input.field_groups.erase(input.field_groups.begin() +
                                 static_cast<std::ptrdiff_t>(missing));
        const auto result = mapmaking::render_discrete_g(input);
        EXPECT_EQ(result.state, mapmaking::DiscreteGState::unavailable);
        EXPECT_EQ(result.plane.size(), 0);
        EXPECT_FALSE(result.unavailable_reasons.empty());
    }
}

TEST(SciMap003DiscreteG, InvalidInputsFailClosedWithoutAPlane) {
    using Mutation =
        std::pair<std::string, std::function<void(mapmaking::DiscreteGInput &)>>;
    const std::vector<Mutation> mutations{
        {"duplicate field group", [](auto &x) { x.field_groups.push_back(x.field_groups.front()); }},
        {"missing field identity", [](auto &x) { x.field_groups.front().identity.clear(); }},
        {"bad field digest", [](auto &x) { x.field_groups.front().digest = "bad"; }},
        {"zero rows", [](auto &x) { x.grid.rows = 0; }},
        {"even rows", [](auto &x) { x.grid.rows = 6; }},
        {"even columns", [](auto &x) { x.grid.cols = 8; }},
        {"singular WCS", [](auto &x) { x.grid.wcs_linear_arcsec_per_pixel.setZero(); }},
        {"nonfinite WCS", [](auto &x) {
             x.grid.wcs_linear_arcsec_per_pixel(0, 0) =
                 std::numeric_limits<double>::quiet_NaN();
         }},
        {"nonfinite CRPIX", [](auto &x) {
             x.grid.crpix1_fits = std::numeric_limits<double>::infinity();
         }},
        {"nonfinite CRVAL", [](auto &x) {
             x.grid.crval2 = std::numeric_limits<double>::quiet_NaN();
         }},
        {"wrong axis", [](auto &x) { x.grid.axis1 = "RA---TAN"; }},
        {"wrong second axis", [](auto &x) { x.grid.axis2 = "DEC--TAN"; }},
        {"wrong angular unit", [](auto &x) { x.grid.angular_unit = "deg"; }},
        {"missing epoch", [](auto &x) { x.grid.epoch.clear(); }},
        {"denominator shape", [](auto &x) { x.denominator.conservativeResize(6, 9); }},
        {"zero denominator", [](auto &x) { x.denominator(0, 0) = 0.0; }},
        {"negative denominator", [](auto &x) { x.denominator(0, 0) = -1.0; }},
        {"nonfinite denominator", [](auto &x) {
             x.denominator(0, 0) = std::numeric_limits<double>::quiet_NaN();
         }},
        {"no detectors", [](auto &x) { x.detectors.clear(); }},
        {"missing APT row", [](auto &x) { x.detectors[0].apt_row_identity.clear(); }},
        {"missing detector", [](auto &x) { x.detectors[0].detector_identity.clear(); }},
        {"duplicate APT row", [](auto &x) {
             x.detectors[1].apt_row_identity = x.detectors[0].apt_row_identity;
         }},
        {"duplicate detector", [](auto &x) {
             x.detectors[1].detector_identity = x.detectors[0].detector_identity;
         }},
        {"bad FWHM", [](auto &x) { x.detectors[0].fwhm_arcsec = 0.0; }},
        {"nonfinite FWHM", [](auto &x) {
             x.detectors[0].fwhm_arcsec =
                 std::numeric_limits<double>::quiet_NaN();
         }},
        {"weight shape", [](auto &x) { x.detectors[0].weights.conservativeResize(7, 8); }},
        {"nonfinite weight", [](auto &x) {
             x.detectors[0].weights(0, 0) =
                 std::numeric_limits<double>::quiet_NaN();
         }},
    };
    for (const auto &[label, mutate] : mutations) {
        SCOPED_TRACE(label);
        auto input = make_g_input();
        mutate(input);
        const auto result = mapmaking::render_discrete_g(input);
        EXPECT_EQ(result.state, mapmaking::DiscreteGState::unavailable);
        EXPECT_EQ(result.plane.size(), 0);
        ASSERT_FALSE(result.unavailable_reasons.empty());
        for (const auto &reason : result.unavailable_reasons) {
            EXPECT_FALSE(reason.affected_object.empty());
            EXPECT_FALSE(reason.evidence.empty());
        }
    }
}

}  // namespace

#include <citlali/core/pipeline/ast_scan_motion.h>
#include <citlali/core/pipeline/ast_scan_motion_alignment.h>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;

constexpr double arcsec_to_radians =
    std::numbers::pi_v<double> / (180.0 * 3600.0);
constexpr pipeline::AstScanMotionIdentityBinding binding{11, 12, 13, 14};

Eigen::VectorXd vector(const std::vector<double> &values) {
  Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
  for (std::size_t index = 0; index < values.size(); ++index) {
    result(static_cast<Eigen::Index>(index)) = values[index];
  }
  return result;
}

Eigen::VectorXd vector(std::initializer_list<double> values) {
  return vector(std::vector<double>(values));
}

Eigen::VectorXi integer_vector(const std::vector<int> &values) {
  Eigen::VectorXi result(static_cast<Eigen::Index>(values.size()));
  for (std::size_t index = 0; index < values.size(); ++index) {
    result(static_cast<Eigen::Index>(index)) = values[index];
  }
  return result;
}

pipeline::AstScanMotionSourceMetadata valid_metadata() {
  return {pipeline::AstScanMotionProducerKind::real_toltec,
          "Science",
          "Lissajous",
          1,
          2000.0,
          0,
          50.0,
          pipeline::AstScanMotionFieldRegistry::
              source_ra_act_source_dec_act_j2000_radians,
          "tel_toltec:test:sha256:0123456789abcdef"};
}

std::shared_ptr<const pipeline::AstScanMotionSource>
source(Eigen::VectorXd times, Eigen::VectorXd ra, Eigen::VectorXd dec,
       pipeline::AstScanMotionSourceMetadata metadata = valid_metadata(),
       pipeline::NativeObservationScope telescope_scope =
           pipeline::NativeObservationScope{152390, 0, 2},
       pipeline::NativeObservationScope detector_scope =
           pipeline::NativeObservationScope{152390, 0, 2}) {
  return pipeline::AstScanMotionSource::admit(
      telescope_scope, detector_scope, 100, std::move(metadata),
      std::move(times), std::move(ra), std::move(dec));
}

pipeline::AstScanMotionSourceMetadata beammap_metadata() {
  auto result = valid_metadata();
  result.dcs_observation_goal = "BeamMap";
  result.dcs_observation_program = "Map";
  result.beammap_profile =
      pipeline::AstBeammapProfileMetadata{0,
                                          "Az",
                                          "Continuous",
                                          "Rectilinear",
                                          0,
                                          0.0,
                                          0.0,
                                          20.0 * arcsec_to_radians,
                                          20.0 * arcsec_to_radians,
                                          0.0};
  return result;
}

std::shared_ptr<const pipeline::AstScanMotionSource> beammap_source(
    std::size_t count, const std::vector<double> &x_offset_arcsec,
    const std::vector<int> &hold_state,
    pipeline::AstScanMotionSourceMetadata metadata = beammap_metadata()) {
  if (x_offset_arcsec.size() != count || hold_state.size() != count) {
    throw std::invalid_argument("Beammap test source shape mismatch");
  }
  std::vector<double> times(count);
  std::vector<double> ra(count);
  std::vector<double> zero(count, 0.0);
  std::vector<double> telescope_azimuth(count);
  for (std::size_t index = 0; index < count; ++index) {
    const double elapsed = 0.020 * static_cast<double>(index);
    times[index] = 1000.0 + elapsed;
    ra[index] = 25.0 * elapsed * arcsec_to_radians;
    telescope_azimuth[index] = x_offset_arcsec[index] * arcsec_to_radians;
  }
  pipeline::AstBeammapMembershipSource membership{integer_vector(hold_state),
                                                  vector(telescope_azimuth),
                                                  vector(zero),
                                                  vector(zero),
                                                  vector(zero),
                                                  vector(zero),
                                                  vector(zero)};
  return pipeline::AstScanMotionSource::admit(
      pipeline::NativeObservationScope{148670, 0, 2},
      pipeline::NativeObservationScope{148670, 0, 2}, 100, std::move(metadata),
      vector(times), vector(ra), vector(zero), std::move(membership));
}

std::shared_ptr<const pipeline::AstScanMotionSource>
motion_source(std::size_t count, double speed_arcsec_per_sec,
              double acceleration_arcsec_per_sec2 = 0.0,
              double start_time_sec = 1000.0, double cadence_sec = 0.020,
              double initial_ra_rad = 0.0) {
  std::vector<double> times(count);
  std::vector<double> ra(count);
  std::vector<double> dec(count, 0.0);
  for (std::size_t index = 0; index < count; ++index) {
    const double elapsed = static_cast<double>(index) * cadence_sec;
    times[index] = start_time_sec + elapsed;
    ra[index] = initial_ra_rad +
                (speed_arcsec_per_sec * elapsed +
                 0.5 * acceleration_arcsec_per_sec2 * elapsed * elapsed) *
                    arcsec_to_radians;
  }
  return source(vector(times), vector(ra), vector(dec));
}

std::shared_ptr<const pipeline::AstScanMotionProduct>
product(std::shared_ptr<const pipeline::AstScanMotionSource> input,
        std::span<const pipeline::AstScanMotionProcessingSpan> schedule = {}) {
  return pipeline::build_ast_scan_motion_product(std::move(input), binding,
                                                 schedule);
}

std::shared_ptr<const pipeline::NativeNetworkAlignment>
network(pipeline::TimestreamNetworkId network_id,
        pipeline::TimestreamNativeRow first_row,
        std::initializer_list<double> times,
        std::vector<pipeline::TimestreamPacketCounter> counters) {
  return std::make_shared<const pipeline::NativeNetworkAlignment>(
      network_id, first_row, vector(times), std::move(counters));
}

TEST(ast_scan_motion,
     exact_field_registry_source_identity_lifecycle_and_memory_are_compact) {
  const auto input = motion_source(41, 10.0);
  const auto result = product(input);

  EXPECT_EQ(pipeline::ast_scan_motion_policy_id, "wp7-ast-scan-motion-v2");
  EXPECT_EQ(pipeline::ast_scan_motion_product_role,
            "SCI-AST:scan_motion_planning@2");
  EXPECT_EQ(pipeline::ast_scan_motion_time_field,
            "Data.TelescopeBackend.TelTime");
  EXPECT_EQ(pipeline::ast_scan_motion_ra_field,
            "Data.TelescopeBackend.SourceRaAct");
  EXPECT_EQ(pipeline::ast_scan_motion_dec_field,
            "Data.TelescopeBackend.SourceDecAct");
  EXPECT_EQ(pipeline::ast_scan_motion_hold_field, "Data.TelescopeBackend.Hold");
  EXPECT_EQ(result->source_handle(), input);
  EXPECT_EQ(result->scope(), (pipeline::NativeObservationScope{152390, 0, 2}));
  EXPECT_EQ(result->identity_binding(), binding);
  EXPECT_EQ(input->identity(100),
            (pipeline::AstTelescopeRecordIdentity{
                pipeline::NativeObservationScope{152390, 0, 2}, 100}));
  EXPECT_EQ(input->first_record(), 100);
  EXPECT_EQ(input->past_last_record(), 141);
  EXPECT_EQ(result->record_count(), 41U);
  EXPECT_EQ(result->route_profile(),
            pipeline::AstScanMotionRouteProfile::science_lissajous);
  ASSERT_TRUE(result->physical_segment_identity(120));
  EXPECT_EQ(*result->physical_segment_identity(120),
            (pipeline::AstScanMotionPhysicalSegmentIdentity{
                pipeline::NativeObservationScope{152390, 0, 2},
                pipeline::AstScanMotionRouteProfile::science_lissajous, 100}));
  EXPECT_EQ(result->memory_evidence().derived_record_bytes,
            41U * sizeof(pipeline::AstScanMotionDerivedRecord));
  EXPECT_EQ(result->memory_evidence().referenced_source_axis_count, 1U);
  EXPECT_EQ(result->memory_evidence().referenced_source_direction_plane_count,
            2U);
  EXPECT_EQ(result->memory_evidence().referenced_beammap_membership_plane_count,
            0U);
  EXPECT_EQ(result->memory_evidence().physical_segment_directory_bytes,
            sizeof(pipeline::AstTelescopeRecord));

}

TEST(ast_scan_motion,
     exact_continuity_defect_and_speed_boundaries_are_inclusive_or_strict) {
  EXPECT_TRUE(pipeline::ast_scan_motion_continuous_interval(
      std::nextafter(0.030, 0.0)));
  EXPECT_TRUE(pipeline::ast_scan_motion_continuous_interval(0.030));
  EXPECT_FALSE(pipeline::ast_scan_motion_continuous_interval(
      std::nextafter(0.030, 1.0)));
  EXPECT_FALSE(pipeline::ast_scan_motion_continuous_interval(0.0));

  EXPECT_FALSE(
      pipeline::ast_scan_motion_telemetry_defect(std::nextafter(2.0, 0.0)));
  EXPECT_FALSE(pipeline::ast_scan_motion_telemetry_defect(2.0));
  EXPECT_TRUE(
      pipeline::ast_scan_motion_telemetry_defect(std::nextafter(2.0, 3.0)));

  EXPECT_FALSE(
      pipeline::ast_scan_motion_speed_admitted(std::nextafter(1.0, 0.0)));
  EXPECT_TRUE(pipeline::ast_scan_motion_speed_admitted(1.0));
  EXPECT_TRUE(
      pipeline::ast_scan_motion_speed_admitted(std::nextafter(1.0, 2.0)));
}

TEST(ast_scan_motion,
     quadratic_j2000_derivative_preserves_wrap_acceleration_and_support) {
  const auto result = product(motion_source(41, 10.0, 5.0, 1000.0, 0.020,
                                            2.0 * std::numbers::pi_v<double> -
                                                0.5 * arcsec_to_radians));

  const auto &center = result->record(120);
  ASSERT_TRUE(center.telemetry_quality_classified());
  ASSERT_FALSE(center.telemetry_defect());
  ASSERT_TRUE(center.derivative_valid());
  EXPECT_NEAR(center.east_velocity_arcsec_per_sec(), 12.0, 1.0e-7);
  EXPECT_NEAR(center.north_velocity_arcsec_per_sec(), 0.0, 1.0e-9);
  EXPECT_NEAR(center.scalar_speed_arcsec_per_sec(), 12.0, 1.0e-7);
  EXPECT_LT(center.telemetry_residual_arcsec(), 0.02);

  const auto quality_support = result->telemetry_support(105);
  ASSERT_TRUE(quality_support);
  EXPECT_EQ(quality_support->first_record, 100);
  EXPECT_EQ(quality_support->past_last_record, 111);
  const auto derivative_support = result->derivative_support(120);
  ASSERT_TRUE(derivative_support);
  EXPECT_EQ(derivative_support->first_record, 115);
  EXPECT_EQ(derivative_support->past_last_record, 126);
  EXPECT_DOUBLE_EQ(derivative_support->first_time_unix_sec, 1000.300);
  EXPECT_DOUBLE_EQ(derivative_support->last_time_unix_sec, 1000.500);

  const auto &summary = result->scan_summary();
  ASSERT_TRUE(summary.maximum_available);
  EXPECT_EQ(summary.maximizing_record, 130);
  EXPECT_NEAR(summary.maximum_speed_arcsec_per_sec, 13.0, 1.0e-7);
  EXPECT_EQ(summary.derivative_valid_record_count, 21U);
  EXPECT_EQ(summary.continuity_run_count, 1U);
}

TEST(ast_scan_motion,
     unsupported_family_facts_are_typed_without_inventing_motion) {
  auto metadata = valid_metadata();
  metadata.producer_kind = pipeline::AstScanMotionProducerKind::simulation;
  metadata.dcs_observation_goal = "Pointing";
  metadata.dcs_observation_program = "Map";
  metadata.scan_file_valid = 0;
  metadata.source_epoch = 1950.0;
  metadata.source_coordinate_system = 1;
  metadata.nominal_producer_cadence_hz = 100.0;
  metadata.field_registry = pipeline::AstScanMotionFieldRegistry::unsupported;
  const auto input = source(vector({1.0, 1.02}), vector({0.0, 0.0}),
                            vector({0.0, 0.0}), std::move(metadata),
                            pipeline::NativeObservationScope{152390, 0, 2},
                            pipeline::NativeObservationScope{152390, 0, 3});
  const auto result = product(input);
  const auto causes = result->record(100).causes();

  EXPECT_TRUE(pipeline::has_cause(
      causes, pipeline::AstScanMotionCause::unsupported_producer_kind));
  EXPECT_TRUE(pipeline::has_cause(
      causes, pipeline::AstScanMotionCause::unsupported_observation_program));
  EXPECT_TRUE(pipeline::has_cause(
      causes, pipeline::AstScanMotionCause::invalid_scan_file));
  EXPECT_TRUE(pipeline::has_cause(
      causes, pipeline::AstScanMotionCause::unsupported_source_frame));
  EXPECT_TRUE(pipeline::has_cause(
      causes, pipeline::AstScanMotionCause::unsupported_producer_cadence));
  EXPECT_TRUE(pipeline::has_cause(
      causes, pipeline::AstScanMotionCause::unsupported_field_registry));
  EXPECT_TRUE(pipeline::has_cause(
      causes, pipeline::AstScanMotionCause::observation_scope_mismatch));
  EXPECT_FALSE(result->record(100).derivative_valid());
  EXPECT_FALSE(result->scan_summary().maximum_available);
  EXPECT_TRUE(pipeline::has_cause(
      result->scan_summary().causes,
      pipeline::AstScanMotionCause::scan_maximum_incomplete));
}

TEST(ast_scan_motion,
     science_oof_and_pointing_lissajous_share_only_the_v1_numerical_operator) {
  const std::vector<std::pair<std::string, pipeline::AstScanMotionRouteProfile>>
      profiles{
          {"Science", pipeline::AstScanMotionRouteProfile::science_lissajous},
          {"Oof", pipeline::AstScanMotionRouteProfile::oof_lissajous},
          {"Pointing",
           pipeline::AstScanMotionRouteProfile::pointing_lissajous}};
  for (const auto &[goal, expected] : profiles) {
    auto metadata = valid_metadata();
    metadata.dcs_observation_goal = goal;
    const auto input = source(
        motion_source(41, 25.0)->producer_times_unix_sec(),
        motion_source(41, 25.0)->source_ra_act_rad(),
        motion_source(41, 25.0)->source_dec_act_rad(), std::move(metadata));
    const auto result = product(input);
    EXPECT_EQ(result->route_profile(), expected);
    EXPECT_EQ(result->scan_summary().physical_scan_member_count, 41U);
    EXPECT_EQ(result->scan_summary().physical_segment_count, 1U);
    ASSERT_TRUE(result->record(120).derivative_valid());
    EXPECT_NEAR(result->record(120).scalar_speed_arcsec_per_sec(), 25.0,
                1.0e-7);
  }
}

TEST(ast_scan_motion,
     beammap_membership_is_nonhold_and_inclusive_at_the_realized_footprint) {
  std::vector<double> x_offset(61, 0.0);
  std::vector<int> hold(61, 0);
  x_offset[10] = 9.999999999;
  x_offset[20] = 10.0;
  x_offset[30] = 10.000000001;
  x_offset[40] = std::numeric_limits<double>::quiet_NaN();
  x_offset[50] = std::numeric_limits<double>::quiet_NaN();
  hold[50] = 2;
  const auto result = product(beammap_source(61, x_offset, hold));

  EXPECT_EQ(
      result->route_profile(),
      pipeline::AstScanMotionRouteProfile::rectilinear_continuous_beammap);
  EXPECT_TRUE(result->record(110).physical_scan_member());
  EXPECT_TRUE(result->record(120).physical_scan_member());
  EXPECT_FALSE(result->record(130).physical_scan_member());
  EXPECT_TRUE(pipeline::has_cause(
      result->record(130).causes(),
      pipeline::AstScanMotionCause::outside_scan_footprint));
  EXPECT_FALSE(result->physical_segment_identity(130));
  EXPECT_FALSE(result->record(140).physical_scan_member());
  EXPECT_TRUE(pipeline::has_cause(
      result->record(140).causes(),
      pipeline::AstScanMotionCause::nonfinite_membership_field));
  EXPECT_FALSE(result->record(150).physical_scan_member());
  EXPECT_TRUE(
      pipeline::has_cause(result->record(150).causes(),
                          pipeline::AstScanMotionCause::producer_hold_active));
  EXPECT_FALSE(pipeline::has_cause(
      result->record(150).causes(),
      pipeline::AstScanMotionCause::nonfinite_membership_field));
  EXPECT_TRUE(result->record(130).raw_direction_structurally_valid());
  EXPECT_FALSE(result->record(130).realized_direction_valid());
  EXPECT_FALSE(result->scan_summary().maximum_available);
  EXPECT_TRUE(pipeline::has_cause(
      result->scan_summary().causes,
      pipeline::AstScanMotionCause::scan_maximum_incomplete));
  EXPECT_EQ(result->memory_evidence().referenced_beammap_membership_plane_count,
            7U);
}

TEST(ast_scan_motion,
     beammap_nonfinite_profile_membership_fields_preserve_hold_precedence) {
  using ProfileField = double pipeline::AstBeammapProfileMetadata::*;
  constexpr std::array<ProfileField, 5> fields{
      &pipeline::AstBeammapProfileMetadata::x_offset_map_lengths,
      &pipeline::AstBeammapProfileMetadata::y_offset_map_lengths,
      &pipeline::AstBeammapProfileMetadata::x_length_rad,
      &pipeline::AstBeammapProfileMetadata::y_length_rad,
      &pipeline::AstBeammapProfileMetadata::scan_angle_rad};
  const std::array<double, 2> nonfinite_values{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()};

  std::vector<double> x_offset(21, 0.0);
  std::vector<int> hold(21, 0);
  hold[10] = 4;
  for (const auto field : fields) {
    for (const double nonfinite : nonfinite_values) {
      auto metadata = beammap_metadata();
      (*metadata.beammap_profile).*field = nonfinite;
      const auto result =
          product(beammap_source(21, x_offset, hold, std::move(metadata)));

      EXPECT_TRUE(pipeline::has_cause(
          result->record(100).causes(),
          pipeline::AstScanMotionCause::nonfinite_membership_field));
      EXPECT_FALSE(pipeline::has_cause(
          result->record(100).causes(),
          pipeline::AstScanMotionCause::unsupported_beammap_profile));
      EXPECT_TRUE(pipeline::has_cause(
          result->record(110).causes(),
          pipeline::AstScanMotionCause::producer_hold_active));
      EXPECT_FALSE(pipeline::has_cause(
          result->record(110).causes(),
          pipeline::AstScanMotionCause::nonfinite_membership_field));
      EXPECT_FALSE(result->scan_summary().maximum_available);
      EXPECT_TRUE(pipeline::has_cause(
          result->scan_summary().causes,
          pipeline::AstScanMotionCause::scan_maximum_incomplete));
    }
  }
}

TEST(ast_scan_motion,
     beammap_segments_retain_short_physical_facts_and_isolate_support) {
  std::vector<double> x_offset(75, 0.0);
  std::vector<int> hold(75, 0);
  for (std::size_t index = 31; index < 35; ++index) {
    x_offset[index] = 20.0;
  }
  for (std::size_t index = 66; index < 70; ++index) {
    hold[index] = 8;
  }
  const auto input = beammap_source(75, x_offset, hold);
  const auto result = product(input);

  const auto &summary = result->scan_summary();
  EXPECT_EQ(summary.physical_scan_member_count, 67U);
  EXPECT_EQ(summary.physical_segment_count, 3U);
  EXPECT_EQ(summary.continuity_run_count, 3U);
  EXPECT_EQ(summary.derivative_valid_record_count, 22U);
  EXPECT_EQ(summary.admitted_candidate_count, 22U);
  ASSERT_TRUE(summary.maximum_available);
  EXPECT_NEAR(summary.maximum_speed_arcsec_per_sec, 25.0, 1.0e-7);

  ASSERT_TRUE(result->physical_segment_identity(100));
  EXPECT_EQ(result->physical_segment_identity(100)->first_record, 100);
  ASSERT_TRUE(result->physical_segment_identity(135));
  EXPECT_EQ(result->physical_segment_identity(135)->first_record, 135);
  ASSERT_TRUE(result->physical_segment_identity(170));
  EXPECT_EQ(result->physical_segment_identity(170)->first_record, 170);
  EXPECT_TRUE(result->record(170).physical_scan_member());
  EXPECT_FALSE(result->record(170).derivative_valid());
  EXPECT_TRUE(pipeline::has_cause(
      result->record(170).causes(),
      pipeline::AstScanMotionCause::telemetry_quality_support_unavailable));
  EXPECT_TRUE(pipeline::has_cause(
      result->record(131).causes(),
      pipeline::AstScanMotionCause::outside_scan_footprint));
  EXPECT_TRUE(
      pipeline::has_cause(result->record(166).causes(),
                          pipeline::AstScanMotionCause::producer_hold_active));
  const auto support = result->derivative_support(145);
  ASSERT_TRUE(support);
  EXPECT_EQ(support->first_record, 140);
  EXPECT_EQ(support->past_last_record, 151);

  const std::vector<pipeline::AstScanMotionProcessingSpan> schedule{
      {150, 175}, {100, 125}, {125, 150}};
  const auto partitioned = product(input, schedule);
  ASSERT_EQ(result->record_count(), partitioned->record_count());
  for (std::size_t index = 0; index < result->record_count(); ++index) {
    const auto record = result->record_identity(index);
    const auto &left = result->record_at_local(index);
    const auto &right = partitioned->record_at_local(index);
    EXPECT_EQ(left.causes(), right.causes());
    EXPECT_EQ(left.physical_scan_member(), right.physical_scan_member());
    EXPECT_EQ(left.physical_segment_index(), right.physical_segment_index());
    EXPECT_EQ(left.continuity_run(), right.continuity_run());
    EXPECT_EQ(left.derivative_valid(), right.derivative_valid());
    EXPECT_EQ(
        std::bit_cast<std::uint64_t>(left.scalar_speed_arcsec_per_sec()),
        std::bit_cast<std::uint64_t>(right.scalar_speed_arcsec_per_sec()));
    EXPECT_EQ(result->physical_segment_identity(record),
              partitioned->physical_segment_identity(record));
    EXPECT_EQ(result->derivative_support(record),
              partitioned->derivative_support(record));
  }
  EXPECT_EQ(result->scan_summary().physical_scan_member_count,
            partitioned->scan_summary().physical_scan_member_count);
  EXPECT_EQ(result->scan_summary().physical_segment_count,
            partitioned->scan_summary().physical_segment_count);
}

TEST(ast_scan_motion,
     unsupported_beammap_profiles_and_missing_membership_fields_fail_closed) {
  std::vector<double> x_offset(21, 0.0);
  std::vector<int> hold(21, 0);
  auto unsupported = beammap_metadata();
  unsupported.beammap_profile->hold_during_turns = 1;
  const auto unsupported_result =
      product(beammap_source(21, x_offset, hold, std::move(unsupported)));
  EXPECT_TRUE(pipeline::has_cause(
      unsupported_result->record(100).causes(),
      pipeline::AstScanMotionCause::unsupported_beammap_profile));
  EXPECT_EQ(unsupported_result->scan_summary().physical_scan_member_count, 0U);

  auto invalid_footprint = beammap_metadata();
  invalid_footprint.beammap_profile->x_length_rad = 0.0;
  const auto invalid_footprint_result = product(
      beammap_source(21, x_offset, hold, std::move(invalid_footprint)));
  EXPECT_TRUE(pipeline::has_cause(
      invalid_footprint_result->record(100).causes(),
      pipeline::AstScanMotionCause::unsupported_beammap_profile));
  EXPECT_EQ(
      invalid_footprint_result->scan_summary().physical_scan_member_count, 0U);

  auto metadata = beammap_metadata();
  const auto nominal = motion_source(21, 25.0);
  const auto missing_fields = product(pipeline::AstScanMotionSource::admit(
      pipeline::NativeObservationScope{148670, 0, 2},
      pipeline::NativeObservationScope{148670, 0, 2}, 100, std::move(metadata),
      nominal->producer_times_unix_sec(), nominal->source_ra_act_rad(),
      nominal->source_dec_act_rad()));
  EXPECT_TRUE(pipeline::has_cause(
      missing_fields->record(100).causes(),
      pipeline::AstScanMotionCause::unsupported_field_registry));
  EXPECT_FALSE(missing_fields->scan_summary().maximum_available);
}

TEST(ast_scan_motion,
     constant_below_exact_and_above_threshold_motion_are_distinct) {
  const auto constant = product(motion_source(41, 0.0));
  ASSERT_TRUE(constant->record(120).derivative_valid());
  EXPECT_DOUBLE_EQ(constant->record(120).scalar_speed_arcsec_per_sec(), 0.0);
  EXPECT_FALSE(constant->scan_summary().maximum_available);
  EXPECT_EQ(constant->scan_summary().admitted_candidate_count, 0U);
  EXPECT_TRUE(pipeline::has_cause(
      constant->scan_summary().causes,
      pipeline::AstScanMotionCause::no_admitted_scan_motion));

  const auto slow = product(motion_source(41, 0.5));
  ASSERT_TRUE(slow->record(120).derivative_valid());
  EXPECT_NEAR(slow->record(120).scalar_speed_arcsec_per_sec(), 0.5, 1.0e-8);
  EXPECT_FALSE(slow->scan_summary().maximum_available);
  EXPECT_TRUE(pipeline::has_cause(
      slow->scan_summary().causes,
      pipeline::AstScanMotionCause::no_admitted_scan_motion));

  const auto exact = product(motion_source(41, 1.0));
  ASSERT_TRUE(exact->record(120).derivative_valid());
  EXPECT_NEAR(exact->record(120).scalar_speed_arcsec_per_sec(), 1.0, 1.0e-8);
  EXPECT_TRUE(pipeline::ast_scan_motion_speed_admitted(1.0));

  const auto above = product(motion_source(41, 1.5));
  ASSERT_TRUE(above->record(120).derivative_valid());
  EXPECT_NEAR(above->record(120).scalar_speed_arcsec_per_sec(), 1.5, 1.0e-8);
  EXPECT_TRUE(above->scan_summary().maximum_available);
}

TEST(ast_scan_motion,
     endpoints_gap_nonfinite_topology_and_rank_failure_remain_distinct) {
  const auto nominal = product(motion_source(41, 10.0));
  EXPECT_TRUE(pipeline::has_cause(
      nominal->record(100).causes(),
      pipeline::AstScanMotionCause::telemetry_quality_support_unavailable));
  EXPECT_FALSE(nominal->record(100).telemetry_quality_classified());
  EXPECT_TRUE(nominal->record(105).telemetry_quality_classified());
  EXPECT_FALSE(nominal->record(105).derivative_valid());
  EXPECT_TRUE(pipeline::has_cause(
      nominal->record(105).causes(),
      pipeline::AstScanMotionCause::derivative_support_intersects_invalidity));
  EXPECT_TRUE(nominal->record(110).derivative_valid());

  auto gap_source = motion_source(41, 10.0);
  Eigen::VectorXd gap_times = gap_source->producer_times_unix_sec();
  for (Eigen::Index index = 20; index < gap_times.size(); ++index) {
    gap_times(index) += 0.020;
  }
  const auto gap =
      product(source(std::move(gap_times), gap_source->source_ra_act_rad(),
                     gap_source->source_dec_act_rad()));
  EXPECT_TRUE(pipeline::has_cause(gap->record(120).causes(),
                                  pipeline::AstScanMotionCause::telescope_gap));
  EXPECT_EQ(gap->scan_summary().continuity_run_count, 2U);
  EXPECT_TRUE(pipeline::has_cause(
      gap->scan_summary().causes,
      pipeline::AstScanMotionCause::scan_maximum_incomplete));

  auto nonfinite_source = motion_source(41, 10.0);
  Eigen::VectorXd nonfinite_ra = nonfinite_source->source_ra_act_rad();
  nonfinite_ra(20) = std::numeric_limits<double>::quiet_NaN();
  const auto nonfinite = product(
      source(nonfinite_source->producer_times_unix_sec(),
             std::move(nonfinite_ra), nonfinite_source->source_dec_act_rad()));
  EXPECT_TRUE(pipeline::has_cause(
      nonfinite->record(120).causes(),
      pipeline::AstScanMotionCause::nonfinite_or_unnormalizable_direction));

  Eigen::VectorXd invalid_times = nonfinite_source->producer_times_unix_sec();
  invalid_times(10) = std::numeric_limits<double>::quiet_NaN();
  invalid_times(20) = invalid_times(19);
  const auto time_invalid = product(
      source(std::move(invalid_times), nonfinite_source->source_ra_act_rad(),
             nonfinite_source->source_dec_act_rad()));
  EXPECT_TRUE(pipeline::has_cause(
      time_invalid->record(110).causes(),
      pipeline::AstScanMotionCause::nonfinite_telescope_time));
  EXPECT_TRUE(pipeline::has_cause(
      time_invalid->record(120).causes(),
      pipeline::AstScanMotionCause::nonmonotonic_telescope_time));
  EXPECT_FALSE(time_invalid->source_time_axis_mapping_eligible());

  Eigen::VectorXd topology_times(41);
  Eigen::VectorXd topology_ra = Eigen::VectorXd::Zero(41);
  Eigen::VectorXd topology_dec = Eigen::VectorXd::Zero(41);
  for (Eigen::Index index = 0; index < topology_times.size(); ++index) {
    topology_times(index) = 1000.0 + 0.020 * static_cast<double>(index);
  }
  topology_ra(20) = std::numbers::pi_v<double>;
  const auto topology =
      product(source(std::move(topology_times), std::move(topology_ra),
                     std::move(topology_dec)));
  EXPECT_TRUE(pipeline::has_cause(
      topology->record(120).causes(),
      pipeline::AstScanMotionCause::spherical_topology_unavailable));

  const double tiny = std::numeric_limits<double>::denorm_min() * 4.0;
  std::vector<double> tiny_times(31);
  for (std::size_t index = 0; index < tiny_times.size(); ++index) {
    tiny_times[index] = tiny * static_cast<double>(index);
  }
  const auto rank =
      product(source(vector(tiny_times), Eigen::VectorXd::Zero(31),
                     Eigen::VectorXd::Zero(31)));
  EXPECT_TRUE(pipeline::has_cause(
      rank->record(115).causes(),
      pipeline::AstScanMotionCause::rank_deficient_derivative_fit));
  EXPECT_FALSE(rank->record(115).derivative_valid());
}

TEST(ast_scan_motion,
     isolated_position_spike_is_defect_but_sustained_fast_motion_sets_maximum) {
  const auto fast_source = motion_source(61, 220.0);
  Eigen::VectorXd spiked_ra = fast_source->source_ra_act_rad();
  spiked_ra(30) += 28.0 * arcsec_to_radians;
  const auto result =
      product(source(fast_source->producer_times_unix_sec(),
                     std::move(spiked_ra), fast_source->source_dec_act_rad()));

  const auto &spike = result->record(130);
  EXPECT_TRUE(spike.telemetry_quality_classified());
  EXPECT_TRUE(spike.telemetry_defect());
  EXPECT_TRUE(pipeline::has_cause(
      spike.causes(), pipeline::AstScanMotionCause::telemetry_defect));
  EXPECT_GT(spike.telemetry_residual_arcsec(), 27.9);
  EXPECT_FALSE(spike.derivative_valid());

  const auto &retained = result->record(115);
  EXPECT_FALSE(retained.telemetry_defect());
  ASSERT_TRUE(retained.derivative_valid());
  EXPECT_NEAR(retained.scalar_speed_arcsec_per_sec(), 220.0, 1.0e-6);
  ASSERT_TRUE(result->scan_summary().maximum_available);
  EXPECT_NEAR(result->scan_summary().maximum_speed_arcsec_per_sec, 220.0,
              1.0e-6);
  EXPECT_EQ(result->scan_summary().telemetry_defect_count, 1U);
}

TEST(ast_scan_motion,
     engineering_partition_order_does_not_change_scientific_results) {
  const auto input = motion_source(41, 25.0, 3.0);
  const auto whole = product(input);
  const std::vector<pipeline::AstScanMotionProcessingSpan> shuffled{
      {120, 141}, {100, 110}, {110, 120}};
  const auto partitioned = product(input, shuffled);

  EXPECT_EQ(whole->source_handle(), partitioned->source_handle());
  EXPECT_EQ(whole->identity_binding(), partitioned->identity_binding());
  ASSERT_EQ(whole->record_count(), partitioned->record_count());
  for (std::size_t index = 0; index < whole->record_count(); ++index) {
    const auto &lhs = whole->record_at_local(index);
    const auto &rhs = partitioned->record_at_local(index);
    EXPECT_EQ(whole->record_identity(index),
              partitioned->record_identity(index));
    EXPECT_EQ(lhs.causes(), rhs.causes());
    EXPECT_EQ(lhs.physical_scan_member(), rhs.physical_scan_member());
    EXPECT_EQ(lhs.physical_segment_index(), rhs.physical_segment_index());
    EXPECT_EQ(lhs.continuity_run(), rhs.continuity_run());
    EXPECT_EQ(lhs.raw_direction_structurally_valid(),
              rhs.raw_direction_structurally_valid());
    EXPECT_EQ(lhs.telemetry_quality_classified(),
              rhs.telemetry_quality_classified());
    EXPECT_EQ(lhs.telemetry_defect(), rhs.telemetry_defect());
    EXPECT_EQ(lhs.realized_direction_valid(), rhs.realized_direction_valid());
    EXPECT_EQ(lhs.derivative_valid(), rhs.derivative_valid());
    EXPECT_EQ(std::bit_cast<std::uint64_t>(lhs.telemetry_residual_arcsec()),
              std::bit_cast<std::uint64_t>(rhs.telemetry_residual_arcsec()));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(lhs.east_velocity_arcsec_per_sec()),
              std::bit_cast<std::uint64_t>(rhs.east_velocity_arcsec_per_sec()));
    EXPECT_EQ(
        std::bit_cast<std::uint64_t>(lhs.north_velocity_arcsec_per_sec()),
        std::bit_cast<std::uint64_t>(rhs.north_velocity_arcsec_per_sec()));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(lhs.scalar_speed_arcsec_per_sec()),
              std::bit_cast<std::uint64_t>(rhs.scalar_speed_arcsec_per_sec()));
    const auto identity = whole->record_identity(index);
    EXPECT_EQ(whole->physical_segment_identity(identity),
              partitioned->physical_segment_identity(identity));
    EXPECT_EQ(whole->telemetry_support(identity),
              partitioned->telemetry_support(identity));
    EXPECT_EQ(whole->derivative_support(identity),
              partitioned->derivative_support(identity));
  }
  EXPECT_EQ(whole->scan_summary().maximum_available,
            partitioned->scan_summary().maximum_available);
  EXPECT_EQ(whole->scan_summary().causes, partitioned->scan_summary().causes);
  EXPECT_EQ(whole->scan_summary().maximizing_record,
            partitioned->scan_summary().maximizing_record);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(
                whole->scan_summary().maximum_speed_arcsec_per_sec),
            std::bit_cast<std::uint64_t>(
                partitioned->scan_summary().maximum_speed_arcsec_per_sec));
  EXPECT_EQ(whole->scan_summary().record_count,
            partitioned->scan_summary().record_count);
  EXPECT_EQ(whole->scan_summary().physical_scan_member_count,
            partitioned->scan_summary().physical_scan_member_count);
  EXPECT_EQ(whole->scan_summary().physical_segment_count,
            partitioned->scan_summary().physical_segment_count);
  EXPECT_EQ(whole->scan_summary().continuity_run_count,
            partitioned->scan_summary().continuity_run_count);
  EXPECT_EQ(whole->scan_summary().derivative_valid_record_count,
            partitioned->scan_summary().derivative_valid_record_count);
  EXPECT_EQ(whole->scan_summary().admitted_candidate_count,
            partitioned->scan_summary().admitted_candidate_count);
  EXPECT_EQ(whole->scan_summary().telemetry_defect_count,
            partitioned->scan_summary().telemetry_defect_count);

  const std::vector<pipeline::AstScanMotionProcessingSpan> incomplete{
      {100, 120}, {121, 141}};
  EXPECT_THROW(product(input, incomplete), std::invalid_argument);
}

TEST(ast_scan_motion_alignment,
     two_networks_preserve_distinct_times_and_exact_source_support) {
  const auto raw = product(motion_source(41, 10.0, 5.0));
  const auto nw0 = network(0, 10, {1000.2400, 1000.2600}, {100, 101});
  const auto nw7 = network(7, 70, {1000.2425, 1000.2625}, {700, 701});
  const auto views = pipeline::AstScanMotionNetworkViews::admit(
      pipeline::NativeObservationScope{152390, 0, 2}, raw, {nw7, nw0});

  EXPECT_EQ(views->raw_product_handle(), raw);
  EXPECT_EQ(views->participant_network_ids()[0], 0);
  EXPECT_EQ(views->participant_network_ids()[1], 7);
  const auto &view0 = views->network(0);
  const auto &view7 = views->network(7);
  EXPECT_EQ(view0.network_timing_handle(), nw0);
  EXPECT_EQ(view7.network_timing_handle(), nw7);
  EXPECT_DOUBLE_EQ(view0.identity(10).reconstructed_time_unix_sec(), 1000.2400);
  EXPECT_DOUBLE_EQ(view7.identity(70).reconstructed_time_unix_sec(), 1000.2425);
  EXPECT_NE(view0.identity(10).reconstructed_time_unix_sec(),
            view7.identity(70).reconstructed_time_unix_sec());
  ASSERT_TRUE(view0.scalar_speed_arcsec_per_sec(10));
  ASSERT_TRUE(view7.scalar_speed_arcsec_per_sec(70));
  EXPECT_NEAR(*view0.scalar_speed_arcsec_per_sec(10), 11.2, 1.0e-7);
  EXPECT_NEAR(*view7.scalar_speed_arcsec_per_sec(70), 11.2125, 1.0e-7);

  const auto support = view7.support(70);
  ASSERT_TRUE(support);
  EXPECT_EQ(support->network_occurrence, view7.identity(70));
  EXPECT_EQ(support->lower_source_record.record, 112);
  EXPECT_EQ(support->upper_source_record.record, 113);
  EXPECT_DOUBLE_EQ(support->lower_source_time_unix_sec, 1000.240);
  EXPECT_DOUBLE_EQ(support->upper_source_time_unix_sec, 1000.260);
  EXPECT_NEAR(support->lower_weight, 0.875, 5.0e-12);
  EXPECT_NEAR(support->upper_weight, 0.125, 5.0e-12);
  EXPECT_EQ(view7.memory_evidence().referenced_raw_product_count, 1U);
  EXPECT_EQ(view7.memory_evidence().referenced_network_time_axis_count, 1U);
  EXPECT_EQ(view7.memory_evidence().derived_mapping_record_bytes,
            2U * sizeof(pipeline::AstScanMotionMappedRecord));
}

TEST(ast_scan_motion_alignment,
     one_network_gap_does_not_manufacture_an_occurrence_in_another) {
  const auto raw = product(motion_source(41, 20.0));
  const auto nw0 =
      network(0, 10, {1000.2400, 1000.2600, 1000.2800}, {100, 101, 102});
  const auto nw7 = network(7, 70, {1000.2425, 1000.2825}, {700, 702});
  const auto views = pipeline::AstScanMotionNetworkViews::admit(
      pipeline::NativeObservationScope{152390, 0, 2}, raw, {nw0, nw7});

  EXPECT_EQ(views->network(0).occurrence_count(), 3U);
  EXPECT_EQ(views->network(7).occurrence_count(), 2U);
  EXPECT_EQ(views->network(0).first_native_row(), 10);
  EXPECT_EQ(views->network(0).past_last_native_row(), 13);
  EXPECT_EQ(views->network(7).first_native_row(), 70);
  EXPECT_EQ(views->network(7).past_last_native_row(), 72);
  EXPECT_TRUE(views->network(0).record(10).available());
  EXPECT_TRUE(views->network(0).record(11).available());
  EXPECT_TRUE(views->network(0).record(12).available());
  EXPECT_TRUE(views->network(7).record(70).available());
  EXPECT_TRUE(views->network(7).record(71).available());
  EXPECT_THROW(views->network(7).record(72), std::out_of_range);
}

TEST(ast_scan_motion_alignment,
     invalid_source_support_is_not_upgraded_by_network_mapping) {
  const auto input = motion_source(61, 20.0);
  Eigen::VectorXd spiked_ra = input->source_ra_act_rad();
  spiked_ra(30) += 28.0 * arcsec_to_radians;
  const auto raw =
      product(source(input->producer_times_unix_sec(), std::move(spiked_ra),
                     input->source_dec_act_rad()));
  const auto view = pipeline::AstScanMotionNetworkView::admit(
      raw, network(0, 10, {1000.6000}, {100}));

  EXPECT_FALSE(view->record(10).available());
  EXPECT_FALSE(view->scalar_speed_arcsec_per_sec(10));
  EXPECT_FALSE(view->support(10));
  EXPECT_TRUE(pipeline::has_cause(
      view->record(10).causes(),
      pipeline::AstScanMotionCause::network_mapping_support_unavailable));
  EXPECT_TRUE(
      pipeline::has_cause(view->record(10).causes(),
                          pipeline::AstScanMotionCause::telemetry_defect));
}

TEST(
    ast_scan_motion_alignment,
    ordinary_motion_implementation_excludes_common_analysis_grid_dependencies) {
  namespace fs = std::filesystem;
  const auto repository = fs::path{__FILE__}.parent_path().parent_path();
  const std::vector<fs::path> headers{
      repository / "include/citlali/core/pipeline/ast_scan_motion.h",
      repository / "include/citlali/core/pipeline/ast_scan_motion_alignment.h",
      repository / "src/citlali/core/pipeline/ast_scan_motion.cpp",
      repository / "src/citlali/core/pipeline/ast_scan_motion_alignment.cpp"};
  const std::vector<std::string> forbidden{
      "CommonAnalysisGrid", "common_analysis_grid_paired_readout.h",
      "NativeAlignmentPlan", "common_slot"};

  for (const auto &header : headers) {
    std::ifstream stream(header);
    ASSERT_TRUE(stream) << header;
    std::ostringstream content;
    content << stream.rdbuf();
    for (const auto &token : forbidden) {
      EXPECT_EQ(content.str().find(token), std::string::npos)
          << header << " contains forbidden ordinary AST dependency " << token;
    }
  }
}

} // namespace

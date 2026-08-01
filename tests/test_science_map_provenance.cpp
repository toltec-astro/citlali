#include <citlali/core/pipeline/coadd_provenance.h>
#include <citlali/core/pipeline/coadd_provenance_lifecycle.h>
#include <citlali/core/pipeline/mapmaking_provenance.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/science_map_provenance_serialization.h>

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

struct FakeScienceMapEngine {
    struct {
        mapmaking::ScienceMapProducts science_products;
    } omb;
};

struct FakeCoaddScienceMapEngine {
    citlali::pipeline::CoaddExecutionPlan coadd_plan;
    struct {
        mapmaking::ScienceMapProducts science_products;
    } cmb;
};

mapmaking::ScienceMapBundleIdentity make_bundle_identity(
    std::size_t map_count = 1) {
    mapmaking::ScienceMapBundleIdentity identity;
    identity.grouping = "array";
    identity.signal_unit = "mJy/beam";
    identity.estimator_identity = "ordinary-naive-map-v1";
    identity.response_identity = "kernel:gaussian-v1";
    identity.required_companions = {
        "weight_I", "kernel_I", "retained_exposure_I"};
    identity.wcs.coordinate_frame = "equatorial-j2000";
    identity.wcs.projection = "TAN";
    identity.wcs.axis_types = {"RA---TAN", "DEC--TAN"};
    identity.wcs.axis_units = {"deg", "deg"};
    identity.wcs.pixel_scale = {
        -1.0 / 3600.0, 1.0 / 3600.0};
    identity.wcs.reference_world = {
        123.45678901234567, -23.45678901234567};
    identity.wcs.reference_pixel = {3.0, 2.0};
    identity.wcs.source_epoch = 2000.0;
    identity.wcs.orientation_rad =
        std::nextafter(0.0, 1.0);
    identity.rows = 5;
    identity.cols = 7;
    for (std::size_t index = 0; index < map_count; ++index) {
        mapmaking::ScienceMapSlotIdentity slot;
        slot.ordered_slot = index;
        slot.grouping = "array";
        slot.group_identity = "array:" + std::to_string(index);
        slot.array_identity = static_cast<long long>(index);
        slot.stokes_identity = 0;
        slot.frequency_hz =
            2.0e11 + static_cast<double>(index) * 1.0e9;
        identity.ordered_slots.push_back(slot);
    }
    return identity;
}

mapmaking::ScienceMapRealizedMap make_realized_map(
    const mapmaking::ScienceMapBundleIdentity &identity) {
    mapmaking::ScienceMapRealizedMap realized;
    realized.initialized = true;
    realized.normalization.support_algorithm =
        mapmaking::science_map_normalization_support_version;
    realized.normalization.coefficient_stage =
        mapmaking::science_map_observation_normalization_coefficient_stage;
    realized.normalization.requested_cut = 0.1;
    realized.normalization.realized_cut = 0.01;
    realized.normalization.realized_threshold =
        0.12345678901234566;
    realized.normalization.selected_positive_value =
        1.2345678901234567;
    realized.normalization.positive_value_count = 11;
    realized.normalization.selected_zero_based_index = 8;
    realized.normalization.selected_index_available = true;
    realized.science_policy = realized.normalization;
    realized.science_policy.support_algorithm =
        mapmaking::science_map_policy_support_version;
    realized.science_policy.realized_cut = 0.1;
    realized.science_policy.realized_threshold =
        1.2345678901234567;
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(
                     mapmaking::ScienceMapProduct::count);
         ++index) {
        realized.product_available.at(index) = true;
        realized.product_nonzero_count.at(index) = index + 1;
        realized.product_value_sum.at(index) =
            index == static_cast<std::size_t>(
                         mapmaking::ScienceMapProduct::retained_exposure)
                ? mapmaking::science_map_double_hex(12.75)
                : std::to_string(100 + index);
    }
    realized.required_companions = identity.required_companions;
    realized.admitted_bundle_identity =
        mapmaking::science_map_bundle_identity_digest(identity);
    realized.raw_parent_digest = "sha256:raw-parent";
    return realized;
}

mapmaking::ScienceMapCoaddAdmission make_coadd_admission(
    const mapmaking::ScienceMapBundleIdentity &identity) {
    mapmaking::ScienceMapCoaddAdmission admission;
    admission.observation_id = "152390";
    admission.delta_row = 2;
    admission.delta_col = 3;
    admission.observation_rows = 1;
    admission.observation_cols = 1;
    admission.coadd_rows = 5;
    admission.coadd_cols = 7;
    admission.ordered_map_count = identity.ordered_slots.size();
    auto observation_identity = identity;
    observation_identity.rows = admission.observation_rows;
    observation_identity.cols = admission.observation_cols;
    observation_identity.wcs.reference_pixel[0] -= admission.delta_col;
    observation_identity.wcs.reference_pixel[1] -= admission.delta_row;
    admission.admitted_bundle_identity =
        mapmaking::science_map_bundle_identity_digest(observation_identity);
    admission.response_identity = identity.response_identity;
    admission.coefficient_stage =
        mapmaking::science_map_observation_unscaled_coefficient_stage;
    admission.normalization_support_policy =
        mapmaking::science_map_normalization_support_version;
    admission.science_policy_support_policy =
        mapmaking::science_map_policy_support_version;
    admission.validity_policy = mapmaking::science_map_validity_version;
    admission.nonfinite_policy =
        mapmaking::science_map_nonfinite_policy_version;
    admission.observation_exposure_seconds =
        12.345678901234567;
    admission.numerically_contributing_pixel_count.assign(
        identity.ordered_slots.size(), 3);
    admission.observation_raw_parent_digests.assign(
        identity.ordered_slots.size(), "sha256:obs-map-0");
    return admission;
}

TEST(science_map_provenance, exact_double_round_trip_distinguishes_float_aliases) {
    const double first = 123.45678901234567;
    const double second = std::nextafter(first, 124.0);
    ASSERT_EQ(static_cast<float>(first), static_cast<float>(second));

    const auto first_stored =
        YAML::Load(YAML::Dump(
            citlali::pipeline::science_map_exact_double_node(first)));
    const auto second_stored =
        YAML::Load(YAML::Dump(
            citlali::pipeline::science_map_exact_double_node(second)));

    const double first_recovered =
        citlali::pipeline::science_map_exact_double_value(first_stored);
    const double second_recovered =
        citlali::pipeline::science_map_exact_double_value(second_stored);
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        first, first_recovered));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        second, second_recovered));
    EXPECT_NE(first_stored["hex"].as<std::string>(),
              second_stored["hex"].as<std::string>());

    const double subnormal = std::nextafter(0.0, 1.0);
    const auto subnormal_stored = YAML::Load(YAML::Dump(
        citlali::pipeline::science_map_exact_double_node(subnormal)));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        subnormal,
        citlali::pipeline::science_map_exact_double_value(
            subnormal_stored)));

    auto inconsistent = first_stored;
    inconsistent["hex"] = second_stored["hex"];
    EXPECT_THROW(
        citlali::pipeline::science_map_exact_double_value(inconsistent),
        std::logic_error);
}

TEST(science_map_provenance, serializes_full_identity_and_eight_product_facts) {
    const auto identity = make_bundle_identity();
    const auto realized = make_realized_map(identity);

    const auto identity_node =
        citlali::pipeline::science_map_bundle_identity_node(identity);
    const auto realized_node =
        citlali::pipeline::science_map_realized_map_node(realized, 0);

    EXPECT_EQ(identity_node["identity_digest"].as<std::string>(),
              mapmaking::science_map_bundle_identity_digest(identity));
    EXPECT_EQ(identity_node["parallel_equivalence_policy"].as<std::string>(),
              mapmaking::science_map_parallel_equivalence_policy);
    EXPECT_EQ(identity_node["policies"]["normalization_support"]
                  .as<std::string>(),
              mapmaking::science_map_normalization_support_version);
    EXPECT_EQ(identity_node["policies"]["science_policy_support"]
                  .as<std::string>(),
              mapmaking::science_map_policy_support_version);
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        identity.wcs.reference_world.at(0),
        citlali::pipeline::science_map_exact_double_value(
            identity_node["wcs"]["reference_world"][0])));
    ASSERT_EQ(realized_node["products"].size(), 8U);
    for (std::size_t index = 0; index < 8; ++index) {
        EXPECT_TRUE(realized_node["products"][index]["available"]
                        .as<bool>());
        EXPECT_EQ(realized_node["products"][index]["nonzero_count"]
                      .as<std::size_t>(),
                  index + 1);
        EXPECT_FALSE(realized_node["products"][index]["value_sum"]
                         .as<std::string>()
                         .empty());
    }
    EXPECT_EQ(realized_node["products"][0]["identity"].as<std::string>(),
              "geometric_hits_I");
    EXPECT_EQ(realized_node["products"][4]["unit"].as<std::string>(),
              "detector s");
    EXPECT_EQ(realized_node["products"][4]["value_sum_encoding"]
                  .as<std::string>(),
              "binary64-c99-hexfloat");
    ASSERT_EQ(realized_node["required_companions"].size(),
              identity.required_companions.size());
    EXPECT_EQ(realized_node["thresholds"]["normalization"]
                  ["coefficient_stage"]
                      .as<std::string>(),
              mapmaking::science_map_observation_normalization_coefficient_stage);
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        realized.normalization.realized_cut,
        citlali::pipeline::science_map_exact_double_value(
            realized_node["thresholds"]["normalization"]
                         ["realized_cut"])));
    EXPECT_EQ(realized_node["raw_parent_digest"].as<std::string>(),
              "sha256:raw-parent");

    auto unavailable = realized;
    const auto validity_index = static_cast<std::size_t>(
        mapmaking::ScienceMapProduct::science_valid);
    unavailable.product_available.at(validity_index) = false;
    unavailable.product_absence_reason.at(validity_index) =
        "required companion unavailable";
    const auto unavailable_node =
        citlali::pipeline::science_map_realized_map_node(unavailable, 0);
    EXPECT_FALSE(unavailable_node["products"][validity_index]
                     ["available"]
                         .as<bool>());
    EXPECT_EQ(unavailable_node["products"][validity_index]
                  ["absence_reason"]
                      .as<std::string>(),
              "required companion unavailable");
}

TEST(science_map_provenance, mapmaking_v3_owns_observation_science_records) {
    citlali::config::MapmakingConfig request;
    request.coverage_cut = 0.12345678901234566;
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        request, citlali::config::ReductionType::science);
    plan.begin_iteration();
    plan.begin_observation(0, "152390", 1, 4.84813681109536e-6, 1);
    const auto identity = make_bundle_identity();
    plan.record_observation_science_state(
        identity, {make_realized_map(identity)});

    const auto node = citlali::pipeline::mapmaking_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-mapmaking-provenance-v3");
    EXPECT_TRUE(node["observations"][0]["science_state"]["available"]
                    .as<bool>());
    EXPECT_EQ(node["observations"][0]["science_state"]
                  ["bundle_identity"]["value"]["identity_digest"]
                      .as<std::string>(),
              mapmaking::science_map_bundle_identity_digest(identity));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        request.coverage_cut,
        citlali::pipeline::science_map_exact_double_value(
            node["science_contract"]["cuts"]["requested"])));
    EXPECT_EQ(node["science_contract"]["coefficient"]
                  ["precision_status"]
                      .as<std::string>(),
              citlali::pipeline::science_map_precision_status);
    EXPECT_EQ(node["science_contract"]["coefficient"]
                  ["covariance_status"]
                      .as<std::string>(),
              citlali::pipeline::science_map_covariance_status);
}

TEST(science_map_provenance, rejects_mapmaking_science_cardinality_mismatch) {
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    plan.begin_iteration();
    plan.begin_observation(0, "152390", 2, 4.84813681109536e-6, 2);

    EXPECT_THROW(
        plan.record_observation_science_state(
            make_bundle_identity(),
            {make_realized_map(make_bundle_identity())}),
        std::logic_error);
}

TEST(science_map_provenance, captures_observation_state_one_way_at_completion) {
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    plan.begin_iteration();
    plan.begin_observation(0, "152390", 1, 4.84813681109536e-6, 1);
    FakeScienceMapEngine engine;
    const auto identity = make_bundle_identity();
    engine.omb.science_products.initialized = true;
    engine.omb.science_products.bundle_identity = identity;
    engine.omb.science_products.realized = {make_realized_map(identity)};

    citlali::pipeline::record_mapmaking_observation_science_state_if_available(
        engine, plan);

    ASSERT_TRUE(plan.observations.back().bundle_identity.has_value());
    EXPECT_EQ(
        mapmaking::science_map_bundle_identity_digest(
            *plan.observations.back().bundle_identity),
        mapmaking::science_map_bundle_identity_digest(identity));
    ASSERT_EQ(plan.observations.back().realized_maps.size(), 1U);
    EXPECT_EQ(plan.observations.back().realized_maps.front().raw_parent_digest,
              "sha256:raw-parent");
}

TEST(science_map_provenance,
     persists_per_product_absence_for_unavailable_observation_profile) {
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    plan.begin_iteration();
    plan.begin_observation(0, "152390", 1, 4.84813681109536e-6, 1);
    FakeScienceMapEngine engine;
    engine.omb.science_products.allocate(
        1, 5, 7, false, false, false,
        "method-specific contribution predicate unavailable");

    citlali::pipeline::record_mapmaking_observation_science_state_if_available(
        engine, plan);

    const auto node = citlali::pipeline::mapmaking_provenance_node(plan);
    EXPECT_FALSE(node["observations"][0]["science_state"]["available"]
                     .as<bool>());
    ASSERT_EQ(node["observations"][0]["science_state"]["realized_maps"]
                  .size(),
              1U);
    const auto inventory = node["observations"][0]["science_state"]
                               ["realized_maps"][0]["products"];
    ASSERT_EQ(inventory.size(), static_cast<std::size_t>(
                                    mapmaking::ScienceMapProduct::count));
    for (const auto &product : inventory) {
        EXPECT_FALSE(product["available"].as<bool>());
        EXPECT_EQ(product["absence_reason"].as<std::string>(),
                  "method-specific contribution predicate unavailable");
    }
}

TEST(science_map_provenance,
     persists_per_product_absence_for_unavailable_coadd_profile) {
    citlali::config::CoaddConfig request;
    request.enabled = true;
    FakeCoaddScienceMapEngine engine;
    engine.coadd_plan.reset_from_request(request, true);
    engine.cmb.science_products.allocate(
        1, 9, 11, true, false, false,
        "polarization science-map product profile is unavailable");

    citlali::pipeline::record_coadd_realized_maps_if_available(engine);

    const auto node =
        citlali::pipeline::coadd_provenance_node(engine.coadd_plan);
    EXPECT_FALSE(node["observation_resolved"]["available"].as<bool>());
    ASSERT_EQ(node["observation_resolved"]["realized_maps"].size(), 1U);
    const auto inventory = node["observation_resolved"]["realized_maps"][0]
                               ["products"];
    ASSERT_EQ(inventory.size(), static_cast<std::size_t>(
                                    mapmaking::ScienceMapProduct::count));
    for (const auto &product : inventory) {
        EXPECT_FALSE(product["available"].as<bool>());
        EXPECT_EQ(product["absence_reason"].as<std::string>(),
                  "polarization science-map product profile is unavailable");
    }
}

TEST(science_map_provenance, coadd_v2_preserves_membership_at_completion) {
    citlali::config::MapmakingConfig mapmaking_request;
    mapmaking_request.coverage_cut = 0.12345678901234566;
    citlali::pipeline::MapmakingExecutionPlan mapmaking_plan;
    mapmaking_plan.reset_from_request(
        mapmaking_request, citlali::config::ReductionType::science);
    mapmaking_plan.begin_iteration();
    mapmaking_plan.begin_observation(
        0, "152390", 1, 4.84813681109536e-6, 1);
    citlali::pipeline::complete_mapmaking_observation(mapmaking_plan);
    mapmaking_plan.begin_coadd(1, 1);
    citlali::pipeline::complete_mapmaking_coadd(mapmaking_plan);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking_plan);

    citlali::config::CoaddConfig request;
    request.enabled = true;
    citlali::pipeline::CoaddExecutionPlan plan;
    plan.reset_from_request(request, true);
    const auto identity = make_bundle_identity();
    plan.record_science_state(identity, {make_realized_map(identity)});
    const auto admission = make_coadd_admission(identity);
    plan.record_admission(admission);

    citlali::pipeline::record_coadd_run_completed(plan, mapmaking_plan);
    ASSERT_EQ(plan.science.admissions.size(), 1U);
    const auto node = citlali::pipeline::coadd_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-coadd-provenance-v2");
    EXPECT_TRUE(node["observation_resolved"]["available"].as<bool>());
    EXPECT_EQ(node["observation_resolved"]["admitted_observation_count"]
                  .as<std::size_t>(),
              1U);
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["observation_id"]
                      .as<std::string>(),
              "152390");
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["embedding"]["delta_row"]
                      .as<Eigen::Index>(),
              2);
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["observation_shape"]["rows"]
                      .as<Eigen::Index>(),
              1);
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["coadd_shape"]["cols"]
                      .as<Eigen::Index>(),
              7);
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["ordered_map_count"]
                      .as<std::size_t>(),
              1U);
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["embedding"]["registration_identity"]
                      .as<std::string>(),
              "centered-integer-common-grid-embedding-v1");
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["embedding"]["centering_identity"]
                      .as<std::string>(),
              "L-identity-v1");
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["policies"]["normalization_support"]
                      .as<std::string>(),
              mapmaking::science_map_normalization_support_version);
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["numerically_contributing_pixel_count"][0]
                      .as<std::size_t>(),
              3U);
    EXPECT_EQ(node["observation_resolved"]["admissions"][0]
                  ["observation_raw_parent_digests"][0]
                      .as<std::string>(),
              "sha256:obs-map-0");
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        admission.observation_exposure_seconds,
        citlali::pipeline::science_map_exact_double_value(
            node["observation_resolved"]["admissions"][0]
                ["observation_exposure_seconds"])));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        mapmaking_request.coverage_cut,
        citlali::pipeline::science_map_exact_double_value(
            node["science_contract"]["cuts"]["effective"]["value"])));
}

TEST(science_map_provenance, resets_coadd_admission_state_per_iteration) {
    citlali::config::CoaddConfig request;
    request.enabled = true;
    citlali::pipeline::CoaddExecutionPlan plan;
    plan.reset_from_request(request, true);
    const auto identity = make_bundle_identity();
    plan.record_science_state(identity, {make_realized_map(identity)});
    plan.record_admission(make_coadd_admission(identity));
    plan.science.requested_coverage_cut = 0.1;
    plan.science.effective_coverage_cut = 0.1;
    plan.realized.reduction_completed = true;

    plan.begin_iteration();

    EXPECT_TRUE(plan.initialized);
    EXPECT_TRUE(plan.requested.enabled);
    EXPECT_TRUE(plan.effective.enabled);
    EXPECT_FALSE(plan.science.common_identity.has_value());
    EXPECT_TRUE(plan.science.realized_maps.empty());
    EXPECT_TRUE(plan.science.admissions.empty());
    EXPECT_FALSE(plan.science.requested_coverage_cut.has_value());
    EXPECT_FALSE(plan.science.effective_coverage_cut.has_value());
    EXPECT_FALSE(plan.realized.reduction_completed);
}

TEST(science_map_provenance, captures_coadd_realized_maps_without_membership_mutation) {
    citlali::config::CoaddConfig request;
    request.enabled = true;
    FakeCoaddScienceMapEngine engine;
    engine.coadd_plan.reset_from_request(request, true);
    const auto identity = make_bundle_identity();
    const auto admission = make_coadd_admission(identity);
    engine.coadd_plan.resolve_common_identity(identity);
    engine.coadd_plan.record_admission(admission);
    engine.cmb.science_products.initialized = true;
    engine.cmb.science_products.identity_admitted = true;
    engine.cmb.science_products.bundle_identity = identity;
    engine.cmb.science_products.realized = {make_realized_map(identity)};
    engine.cmb.science_products.coadd_admissions = {admission};

    citlali::pipeline::record_coadd_realized_maps_if_available(engine);

    ASSERT_EQ(engine.coadd_plan.science.realized_maps.size(), 1U);
    EXPECT_EQ(engine.coadd_plan.science.admissions.size(), 1U);
    EXPECT_TRUE(citlali::pipeline::science_map_coadd_admission_provenance_equal(
        engine.coadd_plan.science.admissions.front(), admission));
}

TEST(science_map_provenance,
     rejects_realized_coadd_membership_and_admission_provenance_drift) {
    using Admission = mapmaking::ScienceMapCoaddAdmission;
    using Mutator = std::function<void(Admission &)>;
    const std::vector<std::pair<std::string, Mutator>> tampering = {
        {"observation-membership", [](auto &admission) {
             admission.observation_id = "tampered-observation";
         }},
        {"centered-offset", [](auto &admission) {
             admission.delta_col += 1;
         }},
        {"admitted-bundle-identity", [](auto &admission) {
             admission.admitted_bundle_identity =
                 "tampered-admitted-identity";
         }},
        {"response-identity", [](auto &admission) {
             admission.response_identity = "tampered-response";
         }},
        {"coefficient-stage", [](auto &admission) {
             admission.coefficient_stage = "tampered-coefficient-stage";
         }},
        {"raw-parent-digest", [](auto &admission) {
             admission.observation_raw_parent_digests.front() =
                 "tampered-raw-parent";
         }},
    };

    for (const auto &[label, mutate] : tampering) {
        SCOPED_TRACE(label);
        citlali::config::CoaddConfig request;
        request.enabled = true;
        FakeCoaddScienceMapEngine engine;
        engine.coadd_plan.reset_from_request(request, true);
        const auto identity = make_bundle_identity();
        const auto admission = make_coadd_admission(identity);
        engine.coadd_plan.resolve_common_identity(identity);
        engine.coadd_plan.record_admission(admission);
        engine.cmb.science_products.initialized = true;
        engine.cmb.science_products.identity_admitted = true;
        engine.cmb.science_products.bundle_identity = identity;
        engine.cmb.science_products.realized = {
            make_realized_map(identity)};
        auto drifted_admission = admission;
        mutate(drifted_admission);
        engine.cmb.science_products.coadd_admissions = {
            drifted_admission};

        EXPECT_THROW(
            citlali::pipeline::record_coadd_realized_maps_if_available(
                engine),
            std::logic_error);
        EXPECT_TRUE(engine.coadd_plan.science.realized_maps.empty());
        ASSERT_EQ(engine.coadd_plan.science.admissions.size(), 1U);
        EXPECT_TRUE(
            citlali::pipeline::science_map_coadd_admission_provenance_equal(
                engine.coadd_plan.science.admissions.front(), admission));
    }
}

TEST(science_map_provenance, serializes_explicit_absence_reasons) {
    citlali::pipeline::MapmakingExecutionPlan mapmaking_plan;
    mapmaking_plan.reset_from_request(
        citlali::config::MapmakingConfig{},
        citlali::config::ReductionType::science);
    mapmaking_plan.begin_iteration();
    mapmaking_plan.begin_observation(
        0, "152390", 1, 4.84813681109536e-6, 1);
    const auto mapmaking_node =
        citlali::pipeline::mapmaking_provenance_node(mapmaking_plan);
    EXPECT_FALSE(mapmaking_node["observations"][0]["science_state"]
                     ["available"]
                         .as<bool>());
    EXPECT_FALSE(mapmaking_node["observations"][0]["science_state"]
                     ["absence_reason"]
                         .as<std::string>()
                         .empty());

    citlali::pipeline::CoaddExecutionPlan coadd_plan;
    coadd_plan.reset_from_request(citlali::config::CoaddConfig{}, true);
    coadd_plan.realized.reduction_completed = true;
    const auto coadd_node =
        citlali::pipeline::coadd_provenance_node(coadd_plan);
    EXPECT_FALSE(coadd_node["observation_resolved"]["available"]
                     .as<bool>());
    EXPECT_FALSE(coadd_node["observation_resolved"]["absence_reason"]
                     .as<std::string>()
                     .empty());
}

}  // namespace

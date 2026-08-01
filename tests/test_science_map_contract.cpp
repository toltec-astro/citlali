#include <citlali/core/mapmaking/map.h>
#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/pipeline/observation_coadd_accumulation.h>
#include <citlali/core/utils/utils.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

mapmaking::ScienceMapBundleIdentity make_identity(Eigen::Index rows,
                                                  Eigen::Index cols,
                                                  Eigen::Index map_count,
                                                  bool kernel = false,
                                                  Eigen::Index n_noise = 0) {
    mapmaking::ScienceMapBundleIdentity identity;
    identity.grouping = "array";
    identity.signal_unit = "mJy/beam";
    identity.estimator_identity =
        "ordinary-naive-normalized-gridding-v1";
    identity.response_identity = kernel ? "response:gaussian" :
                                          "response:identity";
    if (kernel) {
        identity.required_companions.push_back("kernel_I");
    }
    for (Eigen::Index realization = 0; realization < n_noise;
         ++realization) {
        identity.required_companions.push_back(
            "noise_realization_" + std::to_string(realization) + "_I");
    }
    identity.wcs.coordinate_frame = "radec";
    identity.wcs.projection = "TAN";
    identity.wcs.axis_types = {"RA---TAN", "DEC--TAN"};
    identity.wcs.axis_units = {"deg", "deg"};
    identity.wcs.pixel_scale = {-1.0 / 3600.0, 1.0 / 3600.0};
    identity.wcs.reference_world = {123.45678901234567,
                                    -23.45678901234567};
    identity.wcs.reference_pixel = {
        static_cast<double>(cols - 1) / 2.0,
        static_cast<double>(rows - 1) / 2.0};
    identity.wcs.source_epoch = 2000.0;
    identity.wcs.orientation_rad = 0.0;
    identity.rows = rows;
    identity.cols = cols;
    for (Eigen::Index slot = 0; slot < map_count; ++slot) {
        mapmaking::ScienceMapSlotIdentity map_slot;
        map_slot.ordered_slot = static_cast<std::size_t>(slot);
        map_slot.grouping = "array";
        map_slot.group_identity = "array:" + std::to_string(slot);
        map_slot.array_identity = slot;
        map_slot.stokes_identity = 0;
        map_slot.frequency_hz = 2.0e11 + static_cast<double>(slot) * 1.0e9;
        identity.ordered_slots.push_back(map_slot);
    }
    return identity;
}

mapmaking::MapBuffer make_buffer(const std::string &name, Eigen::Index rows,
                                 Eigen::Index cols, Eigen::Index map_count,
                                 bool kernel = false,
                                 Eigen::Index n_noise = 0) {
    mapmaking::MapBuffer buffer{name};
    buffer.n_rows = rows;
    buffer.n_cols = cols;
    buffer.pixel_size_rad = 1.0 / 3600.0 * DEG_TO_RAD;
    buffer.sig_unit = "mJy/beam";
    buffer.map_grouping = name == "cmb" ? "" : "array";
    buffer.parallel_policy = "seq";
    buffer.cov_cut = 0.0;
    buffer.n_noise = n_noise;
    buffer.wcs.naxis = {static_cast<int>(cols), static_cast<int>(rows), 1, 1};
    buffer.wcs.crpix = {static_cast<float>(cols - 1) / 2.0F,
                        static_cast<float>(rows - 1) / 2.0F, 0.0F, 0.0F};
    buffer.wcs.crval = {0.0F, 0.0F, 0.0F, 0.0F};
    buffer.wcs.cdelt = {-1.0F / 3600.0F, 1.0F / 3600.0F, 1.0F, 1.0F};
    buffer.wcs.ctype = {"RA---TAN", "DEC--TAN", "FREQ", "STOKES"};
    buffer.wcs.cunit = {"deg", "deg", "Hz", ""};
    citlali::pipeline::allocate_map_matrices(
        buffer, map_count, false, kernel, true, true);
    if (n_noise > 0) {
        for (Eigen::Index slot = 0; slot < map_count; ++slot) {
            buffer.noise.emplace_back(rows, cols, n_noise);
            buffer.noise.back().setZero();
        }
    }
    return buffer;
}

void refresh_realized_identity(mapmaking::MapBuffer &observation) {
    for (std::size_t slot = 0;
         slot < observation.science_products.realized.size(); ++slot) {
        mapmaking::science_map_finalize_realized_product_facts(
            observation, slot);
    }
}

mapmaking::MapBuffer make_normalized_observation(
    Eigen::Index rows, Eigen::Index cols, Eigen::Index map_count,
    const std::vector<double> &signal_values,
    const std::vector<double> &coefficient_values,
    const std::vector<std::uint8_t> &normalization_values,
    const std::vector<std::uint8_t> &policy_values,
    bool kernel = false, Eigen::Index n_noise = 0) {
    auto observation =
        make_buffer("omb", rows, cols, map_count, kernel, n_noise);
    observation.science_products.bundle_identity =
        make_identity(rows, cols, map_count, kernel, n_noise);
    observation.science_products.identity_admitted = true;
    const Eigen::Index pixel_count = rows * cols;
    if (static_cast<Eigen::Index>(signal_values.size()) != pixel_count ||
        static_cast<Eigen::Index>(coefficient_values.size()) != pixel_count ||
        static_cast<Eigen::Index>(normalization_values.size()) != pixel_count ||
        static_cast<Eigen::Index>(policy_values.size()) != pixel_count) {
        throw std::logic_error("bad science-map test fixture cardinality");
    }
    for (Eigen::Index slot = 0; slot < map_count; ++slot) {
        for (Eigen::Index index = 0; index < pixel_count; ++index) {
            const Eigen::Index row = index % rows;
            const Eigen::Index col = index / rows;
            observation.signal[slot](row, col) = signal_values[index];
            observation.weight[slot](row, col) = coefficient_values[index];
            observation.science_products.normalization_support[slot](row,
                                                                      col) =
                normalization_values[index];
            observation.science_products.science_policy_support[slot](row,
                                                                       col) =
                policy_values[index];
            observation.science_products.science_valid[slot](row, col) =
                normalization_values[index] && policy_values[index] ? 1 : 0;
            observation.science_products.geometric_hits[slot](row, col) =
                10 + index;
            observation.science_products.contributing_hits[slot](row, col) =
                normalization_values[index] ? 2 + index : 0;
            observation.science_products.upstream_eligible_exposure[slot](row,
                                                                           col) =
                0.25 * static_cast<double>(index + 1);
            observation.science_products.retained_exposure[slot](row, col) =
                normalization_values[index]
                    ? 0.5 * static_cast<double>(index + 1)
                    : 0.0;
            observation.coverage[slot](row, col) =
                observation.science_products.retained_exposure[slot](row,
                                                                      col);
            if (kernel) {
                observation.kernel[slot](row, col) =
                    1.5 + static_cast<double>(index);
            }
            for (Eigen::Index realization = 0; realization < n_noise;
                 ++realization) {
                observation.noise[slot](row, col, realization) =
                    signal_values[index] + static_cast<double>(realization);
            }
        }
        auto &realized = observation.science_products.realized[slot];
        realized.initialized = true;
        realized.product_available.fill(true);
        realized.product_available[static_cast<std::size_t>(
            mapmaking::ScienceMapProduct::coadd_observation_count)] = false;
        realized.raw_parent_digest =
            "raw-parent:" + std::to_string(slot);
        realized.normalization.support_algorithm =
            mapmaking::science_map_normalization_support_version;
        realized.science_policy.support_algorithm =
            mapmaking::science_map_policy_support_version;
        realized.normalization.coefficient_stage =
            mapmaking::science_map_observation_normalization_coefficient_stage;
        realized.science_policy.coefficient_stage =
            observation.science_products.coefficient_stage;
    }
    refresh_realized_identity(observation);
    return observation;
}

TEST(science_map_contract,
     coadd_dispatch_uses_explicit_profile_authority_and_fails_closed) {
    constexpr Eigen::Index n_maps = 2;
    auto supported_observation = make_buffer("omb", 2, 2, n_maps);
    auto supported_coadd = make_buffer("cmb", 2, 2, n_maps);
    EXPECT_TRUE(citlali::pipeline::science_map_v1_coadd_profile_enabled(
        supported_coadd, supported_observation, n_maps));

    supported_observation.science_products.geometric_hits.pop_back();
    EXPECT_THROW(
        citlali::pipeline::science_map_v1_coadd_profile_enabled(
            supported_coadd, supported_observation, n_maps),
        std::runtime_error);

    const auto make_unavailable = [=](const std::string &name) {
        mapmaking::MapBuffer buffer{name};
        buffer.n_rows = 2;
        buffer.n_cols = 2;
        citlali::pipeline::allocate_map_matrices(
            buffer, n_maps, false, false, true, false,
            "non-array map-grouping science-map product profile is unavailable");
        return buffer;
    };
    auto unavailable_observation = make_unavailable("omb");
    auto unavailable_coadd = make_unavailable("cmb");
    EXPECT_FALSE(citlali::pipeline::science_map_v1_coadd_profile_enabled(
        unavailable_coadd, unavailable_observation, n_maps));

    unavailable_observation.science_products.realized[1]
        .product_absence_reason[0].clear();
    EXPECT_THROW(
        citlali::pipeline::science_map_v1_coadd_profile_enabled(
            unavailable_coadd, unavailable_observation, n_maps),
        std::runtime_error);
}

std::string coadd_state_digest(const mapmaking::MapBuffer &coadd) {
    mapmaking::ScienceMapCanonicalDigest digest;
    digest.add_string(coadd.map_grouping);
    digest.add_string(coadd.sig_unit);
    digest.add_double(coadd.exposure_time);
    digest.add_integer(coadd.obsnums.size());
    for (const auto &obsnum : coadd.obsnums) {
        digest.add_string(obsnum);
    }
    digest.add_integer(coadd.science_products.coadd_admissions.size());
    for (const auto &admission :
         coadd.science_products.coadd_admissions) {
        digest.add_string(admission.observation_id);
        digest.add_integer(admission.delta_row);
        digest.add_integer(admission.delta_col);
        digest.add_integer(admission.observation_rows);
        digest.add_integer(admission.observation_cols);
        digest.add_integer(admission.coadd_rows);
        digest.add_integer(admission.coadd_cols);
        digest.add_integer(admission.ordered_map_count);
        digest.add_string(admission.admitted_bundle_identity);
        digest.add_string(admission.response_identity);
        digest.add_string(admission.registration_identity);
        digest.add_string(admission.centering_identity);
        digest.add_string(admission.coefficient_stage);
        digest.add_string(admission.normalization_support_policy);
        digest.add_string(admission.science_policy_support_policy);
        digest.add_string(admission.validity_policy);
        digest.add_string(admission.nonfinite_policy);
        digest.add_double(admission.observation_exposure_seconds);
        for (const auto count :
             admission.numerically_contributing_pixel_count) {
            digest.add_integer(count);
        }
        for (const auto &raw_parent :
             admission.observation_raw_parent_digests) {
            digest.add_string(raw_parent);
        }
    }
    digest.add_integer(coadd.science_products.identity_admitted ? 1 : 0);
    digest.add_string(coadd.science_products.coefficient_stage);
    digest.add_string(coadd.science_products.bundle_identity
                          ? mapmaking::science_map_bundle_identity_digest(
                                *coadd.science_products.bundle_identity)
                          : "none");
    const auto hash_planes = [&](const auto &planes) {
        digest.add_integer(planes.size());
        for (const auto &plane : planes) {
            mapmaking::science_map_hash_matrix(digest, plane);
        }
    };
    hash_planes(coadd.signal);
    hash_planes(coadd.weight);
    hash_planes(coadd.kernel);
    hash_planes(coadd.coverage);
    hash_planes(coadd.science_products.geometric_hits);
    hash_planes(coadd.science_products.contributing_hits);
    hash_planes(coadd.science_products.coadd_observation_count);
    hash_planes(coadd.science_products.upstream_eligible_exposure);
    hash_planes(coadd.science_products.retained_exposure);
    hash_planes(coadd.science_products.normalization_support);
    hash_planes(coadd.science_products.science_policy_support);
    hash_planes(coadd.science_products.science_valid);
    for (const auto &cube : coadd.noise) {
        digest.add_integer(cube.dimension(0));
        digest.add_integer(cube.dimension(1));
        digest.add_integer(cube.dimension(2));
        for (Eigen::Index realization = 0; realization < cube.dimension(2);
             ++realization) {
            for (Eigen::Index col = 0; col < cube.dimension(1); ++col) {
                for (Eigen::Index row = 0; row < cube.dimension(0); ++row) {
                    digest.add_double(cube(row, col, realization));
                }
            }
        }
    }
    return digest.finish();
}

TEST(science_map_contract, centered_offsets_reject_odd_negative_and_zero_shapes) {
    EXPECT_EQ(citlali::pipeline::centered_coadd_offsets(5, 7, 3, 3),
              std::make_tuple(Eigen::Index{1}, Eigen::Index{2}));
    EXPECT_THROW(citlali::pipeline::centered_coadd_offsets(4, 7, 3, 3),
                 std::runtime_error);
    EXPECT_THROW(citlali::pipeline::centered_coadd_offsets(3, 3, 5, 3),
                 std::runtime_error);
    EXPECT_THROW(citlali::pipeline::centered_coadd_offsets(0, 3, 1, 1),
                 std::runtime_error);
}

TEST(science_map_contract, unequal_coefficient_coadd_preserves_L_identity) {
    auto coadd = make_buffer("cmb", 3, 5, 1);
    auto first = make_normalized_observation(
        1, 3, 1, {5.0, 10.0, 20.0}, {1.0, 2.0, 4.0}, {1, 1, 1},
        {1, 0, 1});
    auto second = make_normalized_observation(
        1, 3, 1, {7.0, 4.0, -2.0}, {3.0, 1.0, 2.0}, {1, 1, 1},
        {1, 1, 1});
    auto third = make_normalized_observation(
        1, 3, 1, {9.0, 6.0, 8.0}, {2.0, 5.0, 1.0}, {1, 0, 1},
        {1, 0, 1});

    citlali::pipeline::accumulate_observation_into_coadd(
        coadd, first, 1, false, "1", 10.0);
    citlali::pipeline::accumulate_observation_into_coadd(
        coadd, second, 1, false, "2", 20.0);
    citlali::pipeline::accumulate_observation_into_coadd(
        coadd, third, 1, false, "3", 30.0);

    const Eigen::Index row = 1;
    EXPECT_DOUBLE_EQ(coadd.weight[0](row, 1), 6.0);
    EXPECT_DOUBLE_EQ(coadd.weight[0](row, 2), 3.0);
    EXPECT_DOUBLE_EQ(coadd.weight[0](row, 3), 7.0);
    EXPECT_DOUBLE_EQ(coadd.signal[0](row, 1), 44.0);
    EXPECT_DOUBLE_EQ(coadd.signal[0](row, 2), 24.0);
    EXPECT_DOUBLE_EQ(coadd.signal[0](row, 3), 84.0);
    EXPECT_EQ(coadd.science_products.coadd_observation_count[0](row, 1), 3);
    EXPECT_EQ(coadd.science_products.coadd_observation_count[0](row, 2), 2);
    EXPECT_EQ(coadd.science_products.coadd_observation_count[0](row, 3), 3);
    EXPECT_DOUBLE_EQ(coadd.science_products.retained_exposure[0](row, 1),
                     1.5);
    EXPECT_DOUBLE_EQ(coadd.science_products.retained_exposure[0](row, 2),
                     2.0);
    EXPECT_DOUBLE_EQ(coadd.science_products.retained_exposure[0](row, 3),
                     4.5);
    EXPECT_EQ(coadd.obsnums,
              (std::vector<std::string>{"1", "2", "3"}));
    EXPECT_DOUBLE_EQ(coadd.exposure_time, 60.0);
    ASSERT_EQ(coadd.science_products.coadd_admissions.size(), 3U);
    EXPECT_EQ(coadd.science_products.coadd_admissions[2]
                  .numerically_contributing_pixel_count[0],
              2U);

    coadd.normalize_maps();
    EXPECT_DOUBLE_EQ(coadd.signal[0](row, 1), 44.0 / 6.0);
    EXPECT_DOUBLE_EQ(coadd.signal[0](row, 2), 8.0);
    EXPECT_DOUBLE_EQ(coadd.signal[0](row, 3), 12.0);
    EXPECT_EQ(coadd.science_products.normalization_support[0](row, 1), 1);
    EXPECT_EQ(coadd.science_products.science_policy_support[0](row, 1), 1);
    EXPECT_EQ(coadd.science_products.science_valid[0](row, 1), 1);

    // A nonzero constant is not mean-subtracted or source-recentered.
    auto constant_coadd = make_buffer("cmb", 3, 5, 1);
    auto constant = make_normalized_observation(
        1, 3, 1, {11.0, 11.0, 11.0}, {1.0, 1.0, 1.0}, {1, 1, 1},
        {1, 1, 1});
    citlali::pipeline::accumulate_observation_into_coadd(
        constant_coadd, constant, 1, false, "constant", 1.0);
    constant_coadd.normalize_maps();
    EXPECT_DOUBLE_EQ(constant_coadd.signal[0](1, 1), 11.0);
    EXPECT_DOUBLE_EQ(constant_coadd.signal[0](1, 2), 11.0);
    EXPECT_DOUBLE_EQ(constant_coadd.signal[0](1, 3), 11.0);
}

TEST(science_map_contract, bundle_mismatches_fail_without_any_coadd_mutation) {
    auto coadd = make_buffer("cmb", 3, 5, 2, true);
    auto canonical = make_normalized_observation(
        1, 3, 2, {1.0, 2.0, 3.0}, {1.0, 1.0, 1.0}, {1, 1, 1},
        {1, 1, 1}, true);
    citlali::pipeline::accumulate_observation_into_coadd(
        coadd, canonical, 2, true, "canonical", 10.0);

    using Mutator = std::function<void(mapmaking::ScienceMapBundleIdentity &)>;
    const std::vector<Mutator> mismatches = {
        [](auto &identity) { identity.ordered_slots.back().group_identity = "array:99"; },
        [](auto &identity) { identity.signal_unit = "uK"; },
        [](auto &identity) { identity.response_identity = "response:other"; },
        [](auto &identity) {
            identity.wcs.reference_world[0] =
                std::nextafter(identity.wcs.reference_world[0], 124.0);
        },
        [](auto &identity) { identity.wcs.coordinate_frame = "galactic"; },
        [](auto &identity) { identity.wcs.projection = "SIN"; },
        [](auto &identity) {
            identity.wcs.pixel_scale[0] =
                std::nextafter(identity.wcs.pixel_scale[0], 0.0);
        },
        [](auto &identity) {
            identity.wcs.orientation_rad =
                std::nextafter(identity.wcs.orientation_rad, 1.0);
        },
        [](auto &identity) {
            identity.wcs.reference_pixel[0] += 1.0;
        },
    };

    for (const auto &mutate : mismatches) {
        auto incompatible = canonical;
        mutate(*incompatible.science_products.bundle_identity);
        incompatible.sig_unit =
            incompatible.science_products.bundle_identity->signal_unit;
        refresh_realized_identity(incompatible);
        const auto before = coadd_state_digest(coadd);
        EXPECT_THROW(citlali::pipeline::accumulate_observation_into_coadd(
                         coadd, incompatible, 2, true, "rejected", 99.0),
                     std::runtime_error);
        EXPECT_EQ(coadd_state_digest(coadd), before);
    }
}

TEST(science_map_contract,
     realized_provenance_tampering_rejects_the_whole_bundle_atomically) {
    auto coadd = make_buffer("cmb", 3, 3, 2);
    auto canonical = make_normalized_observation(
        1, 1, 2, {2.0}, {3.0}, {1}, {1});
    citlali::pipeline::accumulate_observation_into_coadd(
        coadd, canonical, 2, false, "canonical-provenance", 1.0);

    using Mutator = std::function<void(mapmaking::MapBuffer &)>;
    const std::vector<std::pair<std::string, Mutator>> tampering = {
        {"order-statistic-algorithm-version", [](auto &observation) {
             observation.science_products.realized.back()
                 .normalization.order_statistic_algorithm =
                 "tampered-order-statistic-version";
         }},
        {"support-algorithm-version", [](auto &observation) {
             observation.science_products.realized.back()
                 .science_policy.support_algorithm =
                 "tampered-support-version";
         }},
        {"coefficient-stage", [](auto &observation) {
             observation.science_products.realized.back()
                 .normalization.coefficient_stage =
                 "tampered-coefficient-stage";
         }},
        {"requested-cut", [](auto &observation) {
             auto &cut = observation.science_products.realized.back()
                             .normalization.requested_cut;
             cut = std::nextafter(
                 cut, std::numeric_limits<double>::infinity());
         }},
        {"realized-cut", [](auto &observation) {
             auto &cut = observation.science_products.realized.back()
                             .science_policy.realized_cut;
             cut = std::nextafter(
                 cut, std::numeric_limits<double>::infinity());
         }},
        {"full-precision-normalization-threshold", [](auto &observation) {
             auto &threshold = observation.science_products.realized.back()
                                   .normalization.realized_threshold;
             threshold = std::nextafter(
                 threshold, std::numeric_limits<double>::infinity());
         }},
        {"full-precision-policy-threshold", [](auto &observation) {
             auto &threshold = observation.science_products.realized.back()
                                   .science_policy.realized_threshold;
             threshold = std::nextafter(
                 threshold, std::numeric_limits<double>::infinity());
         }},
        {"comparison-convention", [](auto &observation) {
             observation.science_products.realized.back()
                 .normalization.comparison_convention = ">";
         }},
        {"admitted-bundle-identity", [](auto &observation) {
             observation.science_products.realized.back()
                 .admitted_bundle_identity = "tampered-bundle-identity";
         }},
        {"raw-parent-digest", [](auto &observation) {
             observation.science_products.realized.back()
                 .raw_parent_digest = "tampered-raw-parent-digest";
         }},
    };

    for (const auto &[label, mutate] : tampering) {
        SCOPED_TRACE(label);
        auto candidate = canonical;
        mutate(candidate);
        const auto before = coadd_state_digest(coadd);
        EXPECT_THROW(
            citlali::pipeline::accumulate_observation_into_coadd(
                coadd, candidate, 2, false, "tampered-" + label, 99.0),
            std::runtime_error);
        EXPECT_EQ(coadd_state_digest(coadd), before);
    }
}

TEST(science_map_contract, invalid_payload_is_skipped_but_valid_nonfinite_fails_atomically) {
    auto coadd = make_buffer("cmb", 1, 1, 1, true, 1);
    auto skipped = make_normalized_observation(
        1, 1, 1, {std::numeric_limits<double>::quiet_NaN()},
        {std::numeric_limits<double>::infinity()}, {0}, {0}, true, 1);
    skipped.kernel[0](0, 0) = -std::numeric_limits<double>::infinity();
    skipped.noise[0](0, 0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    refresh_realized_identity(skipped);
    citlali::pipeline::accumulate_observation_into_coadd(
        coadd, skipped, 1, true, "skipped", 1.0);
    EXPECT_DOUBLE_EQ(coadd.signal[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(coadd.weight[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(coadd.kernel[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(coadd.noise[0](0, 0, 0), 0.0);
    EXPECT_EQ(coadd.science_products.coadd_observation_count[0](0, 0), 0);

    const std::vector<double> bad_values = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    for (const double bad : bad_values) {
        for (int target = 0; target < 4; ++target) {
            auto invalid = make_normalized_observation(
                1, 1, 1, {2.0}, {3.0}, {1}, {1}, true, 1);
            if (target == 0) {
                invalid.signal[0](0, 0) = bad;
            }
            else if (target == 1) {
                invalid.weight[0](0, 0) = bad;
            }
            else if (target == 2) {
                invalid.kernel[0](0, 0) = bad;
            }
            else {
                invalid.noise[0](0, 0, 0) = bad;
            }
            refresh_realized_identity(invalid);
            const auto before = coadd_state_digest(coadd);
            EXPECT_THROW(citlali::pipeline::accumulate_observation_into_coadd(
                             coadd, invalid, 1, true, "bad", 1.0),
                         std::runtime_error);
            EXPECT_EQ(coadd_state_digest(coadd), before);
        }
    }
}

TEST(science_map_contract, threshold_rule_is_finite_positive_and_boundary_exact) {
    Eigen::MatrixXd values(1, 10);
    const double selected = 4.0;
    values << 0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), 1.0, 2.0, 3.0,
        selected, 5.0;
    const auto result =
        engine_utils::find_weight_threshold_selection(values, 0.5);
    EXPECT_EQ(result.positive_value_count, 5U);
    EXPECT_EQ(result.selected_zero_based_index, 4U);
    EXPECT_DOUBLE_EQ(result.selected_positive_value, 5.0);
    EXPECT_DOUBLE_EQ(result.threshold, 2.5);

    Eigen::MatrixXd boundary(1, 7);
    boundary << std::nextafter(result.threshold, 0.0), result.threshold,
        std::nextafter(result.threshold,
                       std::numeric_limits<double>::infinity()),
        0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity();
    const auto support = boundary.array().isFinite() &&
                         (boundary.array() > 0.0) &&
                         (boundary.array() >= result.threshold);
    EXPECT_FALSE(support(0, 0));
    EXPECT_TRUE(support(0, 1));
    EXPECT_TRUE(support(0, 2));
    EXPECT_FALSE(support(0, 3));
    EXPECT_FALSE(support(0, 4));
    EXPECT_FALSE(support(0, 5));
    EXPECT_FALSE(support(0, 6));

    Eigen::MatrixXd empty = Eigen::MatrixXd::Zero(2, 2);
    const auto empty_result =
        engine_utils::find_weight_threshold_selection(empty, 1.0);
    EXPECT_EQ(empty_result.positive_value_count, 0U);
    EXPECT_FALSE(empty_result.selected_index_available);
    EXPECT_DOUBLE_EQ(empty_result.threshold, 0.0);

    Eigen::MatrixXd constant = Eigen::MatrixXd::Constant(2, 2, 7.0);
    const auto constant_result =
        engine_utils::find_weight_threshold_selection(constant, 1.0);
    EXPECT_EQ(constant_result.selected_zero_based_index, 3U);
    EXPECT_DOUBLE_EQ(constant_result.threshold, 7.0);
}

TEST(science_map_contract, normalization_persists_eight_distinct_pixel_facts) {
    auto map = make_buffer("omb", 1, 5, 1, false);
    map.cov_cut = 1.0;
    map.science_products.bundle_identity = make_identity(1, 5, 1);
    map.science_products.identity_admitted = true;
    map.signal[0] << 0.0, 0.0, 0.1, 2.0, 16.0;
    map.weight[0] << 0.0, 0.0, 0.1, 1.0, 4.0;
    map.science_products.geometric_hits[0] << 1, 1, 1, 1, 1;
    map.science_products.contributing_hits[0] << 0, 0, 1, 1, 1;
    map.science_products.upstream_eligible_exposure[0]
        << 0.0, 1.0, 1.0, 1.0, 1.0;
    map.science_products.retained_exposure[0]
        << 0.0, 0.0, 1.0, 1.0, 1.0;
    map.coverage[0] = map.science_products.retained_exposure[0];

    map.normalize_maps();

    const auto &products = map.science_products;
    EXPECT_EQ(products.geometric_hits[0](0, 0), 1);
    EXPECT_DOUBLE_EQ(products.upstream_eligible_exposure[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(products.upstream_eligible_exposure[0](0, 1), 1.0);
    EXPECT_EQ(products.contributing_hits[0](0, 1), 0);
    EXPECT_EQ(products.contributing_hits[0](0, 2), 1);
    EXPECT_EQ(products.normalization_support[0](0, 2), 0);
    EXPECT_EQ(products.science_policy_support[0](0, 2), 0);
    EXPECT_DOUBLE_EQ(products.retained_exposure[0](0, 2), 0.0);
    EXPECT_EQ(products.science_valid[0](0, 2), 0);
    EXPECT_EQ(products.normalization_support[0](0, 3), 1);
    EXPECT_EQ(products.science_policy_support[0](0, 3), 0);
    EXPECT_DOUBLE_EQ(products.retained_exposure[0](0, 3), 1.0);
    EXPECT_EQ(products.science_valid[0](0, 3), 0);
    EXPECT_EQ(products.science_valid[0](0, 4), 1);
    EXPECT_TRUE(citlali::pipeline::science_map_double_matrix_exact_equal(
        map.coverage[0], products.retained_exposure[0]));
    ASSERT_TRUE(products.realized[0].initialized);
    EXPECT_FALSE(products.realized[0]
                     .product_available[static_cast<std::size_t>(
                         mapmaking::ScienceMapProduct::coadd_observation_count)]);
    EXPECT_EQ(products.realized[0].normalization.comparison_convention, ">=");
    EXPECT_EQ(products.realized[0].science_policy.comparison_convention, ">=");
    EXPECT_FALSE(products.realized[0].raw_parent_digest.empty());
}

TEST(science_map_contract,
     all_valid_coadd_preserves_historical_arithmetic_order_bitwise) {
    auto coadd = make_buffer("cmb", 3, 3, 1, true, 1);
    const std::vector<double> signals = {3.25, -7.5, 0.125};
    const std::vector<double> coefficients = {0.1, 10.0, 0.3};
    const std::vector<double> kernels = {2.5, -0.75, 11.0};
    const std::vector<double> noise_values = {-4.0, 6.5, 0.25};

    double expected_q = 0.0;
    double expected_n = 0.0;
    double expected_k = 0.0;
    double expected_noise = 0.0;
    double expected_retained_exposure = 0.0;
    std::int64_t expected_geometric_hits = 0;
    std::int64_t expected_contributing_hits = 0;
    for (std::size_t observation_index = 0;
         observation_index < signals.size(); ++observation_index) {
        auto observation = make_normalized_observation(
            1, 1, 1, {signals[observation_index]},
            {coefficients[observation_index]}, {1}, {1}, true, 1);
        observation.kernel[0](0, 0) = kernels[observation_index];
        observation.noise[0](0, 0, 0) = noise_values[observation_index];
        refresh_realized_identity(observation);

        // Independent unchanged-compatible reference: retain observation
        // order and the historical Q += u, N += u*m, K += u*k operations.
        expected_q += coefficients[observation_index];
        expected_n += coefficients[observation_index] *
                      signals[observation_index];
        expected_k += coefficients[observation_index] *
                      kernels[observation_index];
        expected_noise += coefficients[observation_index] *
                          noise_values[observation_index];
        expected_retained_exposure +=
            observation.science_products.retained_exposure[0](0, 0);
        expected_geometric_hits +=
            observation.science_products.geometric_hits[0](0, 0);
        expected_contributing_hits +=
            observation.science_products.contributing_hits[0](0, 0);

        citlali::pipeline::accumulate_observation_into_coadd(
            coadd, observation, 1, true,
            "obs-" + std::to_string(observation_index), 1.0);
    }

    constexpr Eigen::Index center = 1;
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        coadd.weight[0](center, center), expected_q));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        coadd.signal[0](center, center), expected_n));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        coadd.kernel[0](center, center), expected_k));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        coadd.noise[0](center, center, 0), expected_noise));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        coadd.science_products.retained_exposure[0](center, center),
        expected_retained_exposure));
    EXPECT_EQ(coadd.science_products.geometric_hits[0](center, center),
              expected_geometric_hits);
    EXPECT_EQ(coadd.science_products.contributing_hits[0](center, center),
              expected_contributing_hits);
    EXPECT_EQ(coadd.science_products.coadd_observation_count[0](center,
                                                                 center),
              3);
    ASSERT_EQ(coadd.science_products.coadd_admissions.size(), 3U);
    EXPECT_EQ(coadd.science_products.coadd_admissions[0].observation_id,
              "obs-0");
    EXPECT_EQ(coadd.science_products.coadd_admissions[1].observation_id,
              "obs-1");
    EXPECT_EQ(coadd.science_products.coadd_admissions[2].observation_id,
              "obs-2");
    for (const auto &admission : coadd.science_products.coadd_admissions) {
        EXPECT_EQ(admission.delta_row, center);
        EXPECT_EQ(admission.delta_col, center);
        EXPECT_EQ(admission.centering_identity, "L-identity-v1");
    }

    coadd.normalize_maps();
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        coadd.signal[0](center, center), expected_n / expected_q));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        coadd.kernel[0](center, center), expected_k / expected_q));
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        coadd.noise[0](center, center, 0), expected_noise / expected_q));
}

TEST(science_map_contract,
     unsupported_profile_legacy_lane_preserves_arithmetic_without_science_claims) {
    constexpr Eigen::Index n_maps = 2;
    constexpr Eigen::Index n_noise = 2;

    const auto make_legacy_buffer = [=](const std::string &name,
                                        Eigen::Index rows,
                                        Eigen::Index cols) {
        mapmaking::MapBuffer buffer{name};
        buffer.n_rows = rows;
        buffer.n_cols = cols;
        buffer.n_noise = n_noise;
        buffer.science_products.ordinary_contribution_predicate_available =
            false;
        for (Eigen::Index slot = 0; slot < n_maps; ++slot) {
            buffer.signal.emplace_back(
                Eigen::MatrixXd::Zero(rows, cols));
            buffer.weight.emplace_back(
                Eigen::MatrixXd::Zero(rows, cols));
            buffer.kernel.emplace_back(
                Eigen::MatrixXd::Zero(rows, cols));
            buffer.coverage.emplace_back(
                Eigen::MatrixXd::Zero(rows, cols));
            buffer.noise.emplace_back(rows, cols, n_noise);
            buffer.noise.back().setZero();
        }
        return buffer;
    };

    auto coadd = make_legacy_buffer("cmb", 3, 4);
    coadd.obsnums = {"seed"};
    coadd.exposure_time = 0.25;
    for (Eigen::Index slot = 0; slot < n_maps; ++slot) {
        for (Eigen::Index col = 0; col < coadd.n_cols; ++col) {
            for (Eigen::Index row = 0; row < coadd.n_rows; ++row) {
                const double index = static_cast<double>(
                    1 + row + coadd.n_rows * col +
                    coadd.n_rows * coadd.n_cols * slot);
                coadd.weight[slot](row, col) = 0.01 * index;
                coadd.signal[slot](row, col) = -0.125 * index;
                coadd.kernel[slot](row, col) = 0.5 * index;
                coadd.coverage[slot](row, col) = 0.75 * index;
                for (Eigen::Index realization = 0;
                     realization < n_noise; ++realization) {
                    coadd.noise[slot](row, col, realization) =
                        (realization == 0 ? -0.25 : 0.375) * index;
                }
            }
        }
    }

    auto first = make_legacy_buffer("omb", 1, 2);
    first.weight[0] << 0.1, 10.0;
    first.signal[0] << 3.25, -7.5;
    first.kernel[0] << 2.5, -0.75;
    first.coverage[0] << 1.25, 2.5;
    first.weight[1] << 0.3, 4.0;
    first.signal[1] << 0.125, 11.0;
    first.kernel[1] << -2.0, 0.0625;
    first.coverage[1] << 3.0, 0.5;
    first.noise[0](0, 0, 0) = -4.0;
    first.noise[0](0, 1, 0) = 6.5;
    first.noise[0](0, 0, 1) = 0.25;
    first.noise[0](0, 1, 1) = -9.0;
    first.noise[1](0, 0, 0) = 12.0;
    first.noise[1](0, 1, 0) = -0.5;
    first.noise[1](0, 0, 1) = 7.0;
    first.noise[1](0, 1, 1) = 0.125;

    auto second = make_legacy_buffer("omb", 1, 2);
    second.weight[0] << 3.0, 0.2;
    second.signal[0] << -1.25, 8.0;
    second.kernel[0] << 0.5, 16.0;
    second.coverage[0] << 0.75, 4.5;
    second.weight[1] << 2.0, 0.7;
    second.signal[1] << 5.5, -3.0;
    second.kernel[1] << 1.25, -8.0;
    second.coverage[1] << 2.25, 1.5;
    second.noise[0](0, 0, 0) = 1.5;
    second.noise[0](0, 1, 0) = -2.0;
    second.noise[0](0, 0, 1) = 10.0;
    second.noise[0](0, 1, 1) = 0.75;
    second.noise[1](0, 0, 0) = -6.0;
    second.noise[1](0, 1, 0) = 2.5;
    second.noise[1](0, 0, 1) = -0.0625;
    second.noise[1](0, 1, 1) = 4.0;

    auto expected_weight = coadd.weight;
    auto expected_signal = coadd.signal;
    auto expected_kernel = coadd.kernel;
    auto expected_coverage = coadd.coverage;
    auto expected_noise = coadd.noise;
    auto expected_obsnums = coadd.obsnums;
    double expected_exposure = coadd.exposure_time;

    const auto reference_then_accumulate =
        [&](const mapmaking::MapBuffer &observation,
            const std::string &observation_id, double exposure_seconds) {
            // Independent statement of the historical compatibility
            // arithmetic, retaining observation and operation order exactly.
            constexpr Eigen::Index delta_row = 1;
            constexpr Eigen::Index delta_col = 1;
            for (Eigen::Index slot = 0; slot < n_maps; ++slot) {
                for (Eigen::Index col = 0; col < observation.n_cols; ++col) {
                    for (Eigen::Index row = 0; row < observation.n_rows;
                         ++row) {
                        const Eigen::Index coadd_row = row + delta_row;
                        const Eigen::Index coadd_col = col + delta_col;
                        const double coefficient =
                            observation.weight[slot](row, col);
                        expected_weight[slot](coadd_row, coadd_col) +=
                            coefficient;
                        expected_signal[slot](coadd_row, coadd_col) +=
                            observation.signal[slot](row, col) * coefficient;
                        expected_kernel[slot](coadd_row, coadd_col) +=
                            observation.kernel[slot](row, col) * coefficient;
                        expected_coverage[slot](coadd_row, coadd_col) +=
                            observation.coverage[slot](row, col);
                        for (Eigen::Index realization = 0;
                             realization < n_noise; ++realization) {
                            expected_noise[slot](coadd_row, coadd_col,
                                                 realization) +=
                                observation.noise[slot](row, col,
                                                        realization) *
                                coefficient;
                        }
                    }
                }
            }
            // The production legacy lifecycle records membership/exposure
            // before arithmetic. The numerical compatibility primitive must
            // not record them a second time.
            expected_exposure += exposure_seconds;
            expected_obsnums.push_back(observation_id);
            coadd.exposure_time += exposure_seconds;
            coadd.obsnums.push_back(observation_id);
            citlali::pipeline::accumulate_legacy_observation_into_coadd(
                coadd, observation, n_maps, true);
        };

    reference_then_accumulate(first, "legacy-1", 0.1);
    reference_then_accumulate(second, "legacy-2", 0.2);

    for (Eigen::Index slot = 0; slot < n_maps; ++slot) {
        EXPECT_TRUE(citlali::pipeline::science_map_double_matrix_exact_equal(
            coadd.weight[slot], expected_weight[slot]));
        EXPECT_TRUE(citlali::pipeline::science_map_double_matrix_exact_equal(
            coadd.signal[slot], expected_signal[slot]));
        EXPECT_TRUE(citlali::pipeline::science_map_double_matrix_exact_equal(
            coadd.kernel[slot], expected_kernel[slot]));
        EXPECT_TRUE(citlali::pipeline::science_map_double_matrix_exact_equal(
            coadd.coverage[slot], expected_coverage[slot]));
        for (Eigen::Index realization = 0; realization < n_noise;
             ++realization) {
            for (Eigen::Index col = 0; col < coadd.n_cols; ++col) {
                for (Eigen::Index row = 0; row < coadd.n_rows; ++row) {
                    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
                        coadd.noise[slot](row, col, realization),
                        expected_noise[slot](row, col, realization)));
                }
            }
        }
    }
    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
        coadd.exposure_time, expected_exposure));
    EXPECT_EQ(coadd.obsnums, expected_obsnums);

    // The compatibility lane is explicitly outside F009 admission and F010
    // product exposure. It must not manufacture either claim.
    EXPECT_FALSE(coadd.science_products.initialized);
    EXPECT_FALSE(coadd.science_products.ordinary_contribution_predicate_available);
    EXPECT_FALSE(coadd.science_products.identity_admitted);
    EXPECT_FALSE(coadd.science_products.bundle_identity.has_value());
    EXPECT_TRUE(coadd.science_products.geometric_hits.empty());
    EXPECT_TRUE(coadd.science_products.contributing_hits.empty());
    EXPECT_TRUE(coadd.science_products.coadd_observation_count.empty());
    EXPECT_TRUE(coadd.science_products.upstream_eligible_exposure.empty());
    EXPECT_TRUE(coadd.science_products.retained_exposure.empty());
    EXPECT_TRUE(coadd.science_products.normalization_support.empty());
    EXPECT_TRUE(coadd.science_products.science_policy_support.empty());
    EXPECT_TRUE(coadd.science_products.science_valid.empty());
    EXPECT_TRUE(coadd.science_products.realized.empty());
    EXPECT_TRUE(coadd.science_products.coadd_admissions.empty());
}

}  // namespace

#include <citlali/core/engine/detail/sci_align_telescope_alias_contract.h>

#include <gtest/gtest.h>

#include <map>
#include <string>

namespace {

Eigen::VectorXd values(std::initializer_list<double> input) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(input.size()));
    Eigen::Index index = 0;
    for (const double value : input) {
        result(index++) = value;
    }
    return result;
}

TEST(sci_align_telescope_boundary,
     resolves_only_complete_atomic_ra_dec_schema_pairs) {
    const std::map<std::string, Eigen::VectorXd> canonical{
        {"Data.TelescopeBackend.SourceRaAct", values({1.0, 2.0})},
        {"Data.TelescopeBackend.SourceDecAct", values({3.0, 4.0})},
    };
    const auto resolved =
        citlali::engine_detail::resolve_equatorial_aliases(canonical);
    EXPECT_EQ(resolved.source_schema, "SourceRaAct_SourceDecAct");
    EXPECT_EQ(resolved.right_ascension, values({1.0, 2.0}));
    EXPECT_EQ(resolved.declination, values({3.0, 4.0}));

    auto partial = canonical;
    partial.erase("Data.TelescopeBackend.SourceDecAct");
    partial.emplace("Data.TelescopeBackend.TelDecAct", values({3.0, 4.0}));
    EXPECT_THROW(
        citlali::engine_detail::resolve_equatorial_aliases(partial),
        std::runtime_error);
}

TEST(sci_align_telescope_boundary,
     duplicate_ra_dec_schema_pairs_must_be_bit_exact) {
    std::map<std::string, Eigen::VectorXd> aliases{
        {"Data.TelescopeBackend.SourceRaAct", values({1.0, 2.0})},
        {"Data.TelescopeBackend.SourceDecAct", values({3.0, 4.0})},
        {"Data.TelescopeBackend.TelRaAct", values({1.0, 2.0})},
        {"Data.TelescopeBackend.TelDecAct", values({3.0, 4.0})},
    };
    EXPECT_NO_THROW(
        citlali::engine_detail::resolve_equatorial_aliases(aliases));
    aliases.at("Data.TelescopeBackend.TelDecAct")(1) = 4.5;
    EXPECT_THROW(
        citlali::engine_detail::resolve_equatorial_aliases(aliases),
        std::runtime_error);
}

TEST(sci_align_telescope_boundary,
     simulation_epoch_snapshot_preserves_governing_legacy_shape_and_value) {
    const auto snapshot = citlali::engine_detail::
        simulation_j2000_compatibility_header_snapshot();
    EXPECT_EQ(snapshot.type,
              citlali::pipeline::sci_align::
                  TelescopeHeaderNumericType::float64);
    ASSERT_EQ(snapshot.dimensions.size(), 1U);
    EXPECT_EQ(snapshot.dimensions.front().name, "tel_header_n_pts");
    EXPECT_EQ(snapshot.dimensions.front().size, 1U);
    const auto &values = std::get<std::vector<double>>(snapshot.values);
    ASSERT_EQ(values.size(), 1U);
    EXPECT_DOUBLE_EQ(values.front(), 2000.0);
}

}  // namespace

#pragma once

#include <citlali/core/pipeline/sci_align_contract.h>

#include <array>
#include <optional>
#include <string_view>

namespace citlali::pipeline::sci_align {

inline constexpr std::string_view active_field_registry_version =
    "sci-align-active-field-registry-v2";
inline constexpr std::string_view active_field_registry_authority =
    "ALIGN-P0-D004-plus-SCI-ALIGN-001-HOLD-PRODUCER-AUTHORITY-2026-08-02";
inline constexpr std::string_view active_hold_native_semantics_authority =
    "SCI-ALIGN-001-HOLD-PRODUCER-AUTHORITY-2026-08-02;sha256=d6edb175c3aa62ccf92d9644675ece9c8db572a90146370a9c201c296f211c7e";

enum class FieldOperator {
    native_coordinate,
    bracketed_linear,
    bracketed_shortest_arc,
    legacy_whole_word_linear_any_nonzero,
    exact_diagnostic,
};

struct ActiveFieldRegistryEntry {
    std::string_view field_id;
    std::string_view raw_name;
    std::string_view canonical_name;
    std::string_view scientific_identity;
    std::string_view unit;
    std::string_view frame;
    FieldTopology topology;
    FieldOperator permitted_operator;
    bool required_for_admitted_intensity_profile;
    std::string_view output_identity;
};

struct ActiveFieldAliasEntry {
    std::string_view canonical_field_id;
    std::string_view raw_alias;
    std::string_view availability;
};

inline constexpr std::array<ActiveFieldRegistryEntry, 20>
    active_field_registry{{
        {"lmt.tel_time", "Data.TelescopeBackend.TelTime", "TelTime",
         "legacy telescope bracketing coordinate", "s",
         "unproved legacy telescope clock", FieldTopology::exact_only,
         FieldOperator::native_coordinate, true, "common_time compatibility alias"},
        {"lmt.act_gal_ang", "Data.TelescopeBackend.ActGalAng", "ActGalAng",
         "actual galactic angle", "rad", "native telescope realization",
         FieldTopology::circular, FieldOperator::bracketed_shortest_arc, true,
         "ActGalAng"},
        {"lmt.act_par_ang", "Data.TelescopeBackend.ActParAng", "ActParAng",
         "actual parallactic angle", "rad", "native telescope realization",
         FieldTopology::circular, FieldOperator::bracketed_shortest_arc, true,
         "ActParAng"},
        {"lmt.source_az", "Data.TelescopeBackend.SourceAz", "SourceAz",
         "source azimuth", "rad", "horizontal",
         FieldTopology::circular, FieldOperator::bracketed_shortest_arc, true,
         "SourceAz"},
        {"lmt.source_l", "Data.TelescopeBackend.SourceLAct", "TelL",
         "actual source galactic longitude", "rad", "galactic",
         FieldTopology::circular, FieldOperator::bracketed_shortest_arc, true,
         "TelL"},
        {"lmt.source_ra", "Data.TelescopeBackend.SourceRaAct", "TelRa",
         "actual source right ascension", "rad", "equatorial native",
         FieldTopology::circular, FieldOperator::bracketed_shortest_arc, true,
         "TelRa"},
        {"lmt.tel_az_act", "Data.TelescopeBackend.TelAzAct", "TelAzAct",
         "actual telescope azimuth", "rad", "horizontal",
         FieldTopology::circular, FieldOperator::bracketed_shortest_arc, true,
         "TelAzAct"},
        {"lmt.tel_az_des", "Data.TelescopeBackend.TelAzDes", "TelAzDes",
         "desired telescope azimuth", "rad", "horizontal",
         FieldTopology::circular, FieldOperator::bracketed_shortest_arc, true,
         "TelAzDes"},
        {"lmt.source_b", "Data.TelescopeBackend.SourceBAct", "TelB",
         "actual source galactic latitude", "rad", "galactic",
         FieldTopology::continuous_scalar, FieldOperator::bracketed_linear,
         true, "TelB"},
        {"lmt.source_dec", "Data.TelescopeBackend.SourceDecAct", "TelDec",
         "actual source declination", "rad", "equatorial native",
         FieldTopology::continuous_scalar, FieldOperator::bracketed_linear,
         true, "TelDec"},
        {"lmt.source_el", "Data.TelescopeBackend.SourceEl", "SourceEl",
         "source elevation", "rad", "horizontal",
         FieldTopology::continuous_scalar, FieldOperator::bracketed_linear,
         true, "SourceEl"},
        {"lmt.tel_az_cor", "Data.TelescopeBackend.TelAzCor", "TelAzCor",
         "telescope azimuth correction", "rad", "horizontal signed",
         FieldTopology::continuous_scalar, FieldOperator::bracketed_linear,
         true, "TelAzCor"},
        {"lmt.tel_az_map", "Data.TelescopeBackend.TelAzMap", "TelAzMap",
         "telescope azimuth map coordinate", "rad", "map signed",
         FieldTopology::continuous_scalar, FieldOperator::bracketed_linear,
         true, "TelAzMap"},
        {"lmt.tel_el_act", "Data.TelescopeBackend.TelElAct", "TelElAct",
         "actual telescope elevation", "rad", "horizontal bounded",
         FieldTopology::continuous_scalar, FieldOperator::bracketed_linear,
         true, "TelElAct"},
        {"lmt.tel_el_cor", "Data.TelescopeBackend.TelElCor", "TelElCor",
         "telescope elevation correction", "rad", "horizontal signed",
         FieldTopology::continuous_scalar, FieldOperator::bracketed_linear,
         true, "TelElCor"},
        {"lmt.tel_el_des", "Data.TelescopeBackend.TelElDes", "TelElDes",
         "desired telescope elevation", "rad", "horizontal bounded",
         FieldTopology::continuous_scalar, FieldOperator::bracketed_linear,
         true, "TelElDes"},
        {"lmt.tel_el_map", "Data.TelescopeBackend.TelElMap", "TelElMap",
         "telescope elevation map coordinate", "rad", "map signed",
         FieldTopology::continuous_scalar, FieldOperator::bracketed_linear,
         true, "TelElMap"},
        {"lmt.hold_raw_word", "Data.TelescopeBackend.Hold", "Hold",
         "exact native producer-defined Hold reason bitmask plus named legacy "
         "compatibility view", "1",
         "state word; zero only science-valid; unknown bits fail closed; "
         "transition side unresolved", FieldTopology::exact_only,
         FieldOperator::legacy_whole_word_linear_any_nonzero, true,
         "Hold: post-nonzero 0/1 compatibility alias; exact raw word "
         "retained internally; no routine exporter"},
        {"lmt.tel_utc", "Data.TelescopeBackend.TelUtc", "TelUTC",
         "native telescope UTC diagnostic", "s", "unproved native clock",
         FieldTopology::exact_only, FieldOperator::exact_diagnostic, false,
         "common_time compatibility alias"},
        {"lmt.pps_time", "Data.TelescopeBackend.PpsTime", "PpsTime",
         "native telescope PPS diagnostic", "s", "unproved native clock",
         FieldTopology::exact_only, FieldOperator::exact_diagnostic, false,
         "not aligned standard"},
    }};

// These are schema-versioned alternatives for the same canonical RA/Dec
// identities, not additional scientific fields. If both canonical and alias
// forms are present their arrays must agree exactly.
inline constexpr std::array<ActiveFieldAliasEntry, 2>
    active_field_aliases{{
        {"lmt.source_ra", "Data.TelescopeBackend.TelRaAct",
         "legacy_alias_if_exactly_equal"},
        {"lmt.source_dec", "Data.TelescopeBackend.TelDecAct",
         "legacy_alias_if_exactly_equal"},
    }};

inline constexpr std::string_view field_topology_name(
    FieldTopology topology) noexcept {
    switch (topology) {
        case FieldTopology::continuous_scalar:
            return "continuous_scalar";
        case FieldTopology::circular:
            return "circular_angle";
        case FieldTopology::declared_half_open_step:
            return "declared_half_open_step_state";
        case FieldTopology::exact_only:
            return "exact_only";
    }
    return "exact_only";
}

inline constexpr std::string_view field_operator_name(
    FieldOperator operation) noexcept {
    switch (operation) {
        case FieldOperator::native_coordinate:
            return "native_coordinate";
        case FieldOperator::bracketed_linear:
            return "bracketed_linear";
        case FieldOperator::bracketed_shortest_arc:
            return "bracketed_shortest_arc_period_2pi";
        case FieldOperator::legacy_whole_word_linear_any_nonzero:
            return "legacy_4x_linear_any_nonzero";
        case FieldOperator::exact_diagnostic:
            return "exact_diagnostic_only";
    }
    return "exact_diagnostic_only";
}

inline constexpr std::string_view active_field_availability(
    const ActiveFieldRegistryEntry &entry) noexcept {
    return entry.required_for_admitted_intensity_profile
               ? "required_for_admitted_intensity_profile"
               : "optional_diagnostic";
}

inline constexpr std::string_view active_field_source_dtype(
    const ActiveFieldRegistryEntry &) noexcept {
    return "float64";
}

inline constexpr std::string_view active_field_source_shape(
    const ActiveFieldRegistryEntry &) noexcept {
    return "time";
}

inline constexpr std::string_view active_field_raw_unit(
    const ActiveFieldRegistryEntry &entry) noexcept {
    if (entry.canonical_name == "Hold") {
        return "boolean_raw_attribute_conflicts_with_observed_multi_bit_word";
    }
    if (entry.unit == "s") {
        return "sec";
    }
    return entry.unit;
}

inline constexpr std::string_view active_field_source_authority(
    const ActiveFieldRegistryEntry &entry) noexcept {
    if (entry.canonical_name == "Hold") {
        return active_hold_native_semantics_authority;
    }
    return "D004_owner_decision_plus_bound_local_raw_schema";
}

inline constexpr std::string_view active_field_validity_policy(
    const ActiveFieldRegistryEntry &entry) noexcept {
    if (entry.canonical_name == "Hold") {
        return "finite_nonnegative_integral_lossless_raw_word;native_science_valid_iff_zero;unknown_bits_fail_closed;legacy_whole_word_linear_any_nonzero_transition_side_unresolved";
    }
    switch (entry.permitted_operator) {
        case FieldOperator::native_coordinate:
            return "finite_nonempty_strictly_increasing";
        case FieldOperator::bracketed_linear:
        case FieldOperator::bracketed_shortest_arc:
            return "finite_shape_matches_TelTime_adjacent_finite_bracket_no_extrapolation_producer_gap_authority_unavailable";
        case FieldOperator::legacy_whole_word_linear_any_nonzero:
            return "finite_nonnegative_integral_lossless_raw_word_then_named_legacy_view";
        case FieldOperator::exact_diagnostic:
            return "native_exact_identity_only_no_generic_interpolation";
    }
    return "unavailable";
}

inline constexpr std::string_view active_field_support_rule(
    const ActiveFieldRegistryEntry &entry) noexcept {
    switch (entry.permitted_operator) {
        case FieldOperator::native_coordinate:
        case FieldOperator::exact_diagnostic:
            return "native_exact_support_only";
        case FieldOperator::bracketed_linear:
        case FieldOperator::bracketed_shortest_arc:
        case FieldOperator::legacy_whole_word_linear_any_nonzero:
            return "at_most_one_adjacent_finite_native_interval_no_general_numeric_runtime_ceiling_producer_gap_authority_unavailable";
    }
    return "unavailable";
}

inline constexpr std::optional<double> active_field_runtime_maximum_support_span_sec(
    const ActiveFieldRegistryEntry &) noexcept {
    // D005's 0.021130561828613281 s value is an inclusive fixed-cohort
    // validation envelope, not producer authority or a runtime limit.
    return std::nullopt;
}

inline const ActiveFieldRegistryEntry *active_field_entry(
    std::string_view canonical_name) noexcept {
    for (const auto &entry : active_field_registry) {
        if (entry.canonical_name == canonical_name) {
            return &entry;
        }
    }
    return nullptr;
}

inline std::string_view aligned_telescope_output_unit(
    std::string_view name) noexcept {
    if (const auto *entry = active_field_entry(name); entry != nullptr) {
        if (name == "TelTime" || name == "TelUTC") {
            return "s";
        }
        return entry->unit;
    }
    // Tangent-plane and derived pointing fields are existing AST outputs in
    // radians. Unknown fields are not silently labeled as angles.
    if (name == "ra_phys" || name == "dec_phys" || name == "az_phys" ||
        name == "alt_phys" || name == "l_phys" || name == "b_phys" ||
        name == "lat_phys" || name == "lon_phys") {
        return "rad";
    }
    return "";
}

}  // namespace citlali::pipeline::sci_align

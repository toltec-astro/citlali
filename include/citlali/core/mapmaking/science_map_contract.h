#pragma once

#include <Eigen/Core>

#include <citlali/core/utils/sha256.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace mapmaking {

inline constexpr const char *science_map_contract_version =
    "citlali-science-map-contract-v1";
inline constexpr const char *science_map_order_statistic_version =
    "positive-order-statistic-floor-075-midpoint-v1";
inline constexpr const char *science_map_normalization_support_version =
    "finite-positive-ge-threshold-coverage-cut-div10-v1";
inline constexpr const char *science_map_policy_support_version =
    "finite-positive-ge-threshold-coverage-cut-v1";
inline constexpr const char *science_map_validity_version =
    "normalization-and-policy-and-finite-companions-and-identity-v1";
inline constexpr const char *science_map_ordinary_contribution_version =
    "ordinary-naive-finite-positive-coefficient-v1";
inline constexpr const char *science_map_coadd_estimator_version =
    "centered-integer-normalized-weighted-mean-L-identity-v1";
inline constexpr const char *science_map_nonfinite_policy_version =
    "explicit-invalid-skip-valid-nonfinite-fail-v1";
inline constexpr const char *science_map_coefficient_policy_version =
    "nonprecision-normalization-coefficient-v1";
inline constexpr const char *science_map_parallel_equivalence_policy =
    "within-scan-exact-scan-farm-2gamma-n-sumabs-v1";
inline constexpr const char *science_map_observation_normalization_coefficient_stage =
    "pre-observation-normalization-accumulated-coefficient";
inline constexpr const char *science_map_coadd_normalization_coefficient_stage =
    "pre-coadd-normalization-sum-of-admitted-observation-coefficients";
inline constexpr const char *science_map_observation_unscaled_coefficient_stage =
    "post-observation-normalization-no-empirical-rescale";
inline constexpr const char *science_map_observation_empirical_coefficient_stage =
    "post-observation-normalization-global-empirical-rescale-applied";
inline constexpr const char *science_map_coadd_unscaled_coefficient_stage =
    "post-coadd-normalization-no-empirical-rescale";
inline constexpr const char *science_map_coadd_empirical_coefficient_stage =
    "post-coadd-normalization-global-empirical-rescale-applied";
inline constexpr const char *science_map_digest_version =
    "canonical-hexfloat-sha256-v1";

enum class ScienceMapProduct : std::size_t {
    geometric_hits = 0,
    contributing_hits,
    coadd_observation_count,
    upstream_eligible_exposure,
    retained_exposure,
    normalization_support,
    science_policy_support,
    science_valid,
    count,
};

inline constexpr std::array<const char *, 8> science_map_product_names = {
    "geometric_hits_I",
    "contributing_hits_I",
    "coadd_observation_count_I",
    "upstream_eligible_exposure_I",
    "retained_exposure_I",
    "normalization_support_I",
    "science_policy_support_I",
    "science_valid_I",
};

inline constexpr std::array<const char *, 8> science_map_product_units = {
    "count", "count", "count", "detector s", "detector s", "1", "1", "1",
};

inline const char *science_map_product_name(ScienceMapProduct product) {
    return science_map_product_names.at(static_cast<std::size_t>(product));
}

inline const char *science_map_product_unit(ScienceMapProduct product) {
    return science_map_product_units.at(static_cast<std::size_t>(product));
}

inline bool science_map_exact_double_equal(double lhs, double rhs) {
    std::uint64_t lhs_bits = 0;
    std::uint64_t rhs_bits = 0;
    static_assert(sizeof(lhs_bits) == sizeof(lhs),
                  "science-map identity assumes binary64 double");
    std::memcpy(&lhs_bits, &lhs, sizeof(lhs));
    std::memcpy(&rhs_bits, &rhs, sizeof(rhs));
    return lhs_bits == rhs_bits;
}

inline std::string science_map_double_hex(double value) {
    std::ostringstream stream;
    stream << std::hexfloat << value;
    return stream.str();
}

struct ScienceMapWcsIdentity {
    std::string coordinate_frame;
    std::string projection;
    std::vector<std::string> axis_types;
    std::vector<std::string> axis_units;
    std::vector<double> pixel_scale;
    std::vector<double> reference_world;
    std::vector<double> reference_pixel;
    double source_epoch = std::numeric_limits<double>::quiet_NaN();
    double orientation_rad = 0.0;
};

struct ScienceMapSlotIdentity {
    std::size_t ordered_slot = 0;
    std::string grouping;
    std::string group_identity;
    long long array_identity = -1;
    long long stokes_identity = 0;
    double frequency_hz = std::numeric_limits<double>::quiet_NaN();
};

struct ScienceMapBundleIdentity {
    std::string contract_version = science_map_contract_version;
    std::string grouping;
    std::string signal_unit;
    std::string estimator_identity;
    std::string response_identity;
    std::string parallel_equivalence_policy =
        science_map_parallel_equivalence_policy;
    std::vector<std::string> required_companions;
    std::string validity_policy = science_map_validity_version;
    std::string coefficient_policy = science_map_coefficient_policy_version;
    std::string normalization_support_policy =
        science_map_normalization_support_version;
    std::string science_policy_support_policy =
        science_map_policy_support_version;
    std::string nonfinite_policy = science_map_nonfinite_policy_version;
    ScienceMapWcsIdentity wcs;
    std::vector<ScienceMapSlotIdentity> ordered_slots;
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;
};

struct ScienceMapThresholdRealization {
    std::string order_statistic_algorithm =
        science_map_order_statistic_version;
    std::string support_algorithm;
    std::string coefficient_product = "weight_I";
    std::string coefficient_stage;
    double requested_cut = 0.0;
    double realized_cut = 0.0;
    double realized_threshold = 0.0;
    double selected_positive_value = 0.0;
    std::size_t positive_value_count = 0;
    std::size_t selected_zero_based_index = 0;
    bool selected_index_available = false;
    std::string finite_convention = "coefficient must be finite";
    std::string positivity_convention = "coefficient > 0";
    std::string comparison_convention = ">=";
};

struct ScienceMapRealizedMap {
    bool initialized = false;
    std::array<bool, static_cast<std::size_t>(ScienceMapProduct::count)>
        product_available{};
    std::array<std::string,
               static_cast<std::size_t>(ScienceMapProduct::count)>
        product_absence_reason{};
    ScienceMapThresholdRealization normalization;
    ScienceMapThresholdRealization science_policy;
    std::array<std::size_t,
               static_cast<std::size_t>(ScienceMapProduct::count)>
        product_nonzero_count{};
    // Decimal strings for integer/mask sums and canonical hexfloat strings for
    // exposure sums; paired with product identity, this is lossless and avoids
    // forcing heterogeneous facts through one floating-point count type.
    std::array<std::string,
               static_cast<std::size_t>(ScienceMapProduct::count)>
        product_value_sum{};
    std::vector<std::string> required_companions;
    std::string admitted_bundle_identity;
    std::string raw_parent_digest;
};

inline bool science_map_realized_map_has_explicit_product_absence(
    const ScienceMapRealizedMap &record) {
    for (std::size_t product = 0;
         product < static_cast<std::size_t>(ScienceMapProduct::count);
         ++product) {
        if (record.product_available[product] ||
            record.product_absence_reason[product].empty()) {
            return false;
        }
    }
    return true;
}

struct ScienceMapCoaddAdmission {
    std::string observation_id;
    Eigen::Index delta_row = 0;
    Eigen::Index delta_col = 0;
    Eigen::Index observation_rows = 0;
    Eigen::Index observation_cols = 0;
    Eigen::Index coadd_rows = 0;
    Eigen::Index coadd_cols = 0;
    std::size_t ordered_map_count = 0;
    std::string admitted_bundle_identity;
    std::string response_identity;
    std::string registration_identity =
        "centered-integer-common-grid-embedding-v1";
    std::string centering_identity = "L-identity-v1";
    std::string coefficient_stage;
    std::string normalization_support_policy;
    std::string science_policy_support_policy;
    std::string validity_policy;
    std::string nonfinite_policy;
    double observation_exposure_seconds = 0.0;
    std::vector<std::size_t> numerically_contributing_pixel_count;
    std::vector<std::string> observation_raw_parent_digests;
};

using ScienceMapCountPlane =
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>;
using ScienceMapMaskPlane =
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

struct ScienceMapProducts {
    bool initialized = false;
    bool is_coadd = false;
    bool ordinary_contribution_predicate_available = false;
    bool identity_admitted = false;
    std::string coefficient_stage =
        science_map_observation_unscaled_coefficient_stage;
    std::optional<ScienceMapBundleIdentity> bundle_identity;

    std::vector<ScienceMapCountPlane> geometric_hits;
    std::vector<ScienceMapCountPlane> contributing_hits;
    std::vector<ScienceMapCountPlane> coadd_observation_count;
    std::vector<Eigen::MatrixXd> upstream_eligible_exposure;
    std::vector<Eigen::MatrixXd> retained_exposure;
    std::vector<ScienceMapMaskPlane> normalization_support;
    std::vector<ScienceMapMaskPlane> science_policy_support;
    std::vector<ScienceMapMaskPlane> science_valid;

    std::vector<ScienceMapRealizedMap> realized;
    std::vector<ScienceMapCoaddAdmission> coadd_admissions;

    void clear() {
        initialized = false;
        is_coadd = false;
        ordinary_contribution_predicate_available = false;
        identity_admitted = false;
        coefficient_stage = science_map_observation_unscaled_coefficient_stage;
        bundle_identity.reset();
        geometric_hits.clear();
        contributing_hits.clear();
        coadd_observation_count.clear();
        upstream_eligible_exposure.clear();
        retained_exposure.clear();
        normalization_support.clear();
        science_policy_support.clear();
        science_valid.clear();
        realized.clear();
        coadd_admissions.clear();
    }

    void allocate(Eigen::Index map_count, Eigen::Index rows, Eigen::Index cols,
                  bool coadd, bool ordinary_predicate_available,
                  bool allocate_product_planes = true,
                  std::string product_absence_reason = {}) {
        clear();
        initialized = true;
        is_coadd = coadd;
        coefficient_stage = coadd
            ? science_map_coadd_unscaled_coefficient_stage
            : science_map_observation_unscaled_coefficient_stage;
        ordinary_contribution_predicate_available =
            ordinary_predicate_available;
        const auto count = static_cast<std::size_t>(map_count);
        realized.assign(count, ScienceMapRealizedMap{});
        if (!allocate_product_planes) {
            if (product_absence_reason.empty()) {
                product_absence_reason = ordinary_predicate_available
                    ? "science-map product profile is unavailable"
                    : "method-specific contribution predicate unavailable";
            }
            for (auto &record : realized) {
                record.product_absence_reason.fill(product_absence_reason);
            }
            return;
        }
        const ScienceMapCountPlane count_zeros =
            ScienceMapCountPlane::Zero(rows, cols);
        const Eigen::MatrixXd exposure_zeros =
            Eigen::MatrixXd::Zero(rows, cols);
        const ScienceMapMaskPlane mask_zeros =
            ScienceMapMaskPlane::Zero(rows, cols);
        geometric_hits.assign(count, count_zeros);
        contributing_hits.assign(count, count_zeros);
        coadd_observation_count.assign(count, count_zeros);
        upstream_eligible_exposure.assign(count, exposure_zeros);
        retained_exposure.assign(count, exposure_zeros);
        normalization_support.assign(count, mask_zeros);
        science_policy_support.assign(count, mask_zeros);
        science_valid.assign(count, mask_zeros);
    }
};

class ScienceMapCanonicalDigest {
public:
    void add_string(const std::string &value) {
        digest_.update(std::to_string(value.size()));
        digest_.update(":");
        digest_.update(value);
        digest_.update(";");
    }

    template <class Integer>
    void add_integer(Integer value) {
        add_string(std::to_string(value));
    }

    void add_double(double value) { add_string(science_map_double_hex(value)); }

    std::string finish() {
        return std::string(science_map_digest_version) + ":" + digest_.finish();
    }

private:
    citlali::utils::Sha256 digest_;
};

inline std::string science_map_bundle_identity_digest(
    const ScienceMapBundleIdentity &identity) {
    ScienceMapCanonicalDigest digest;
    digest.add_string(identity.contract_version);
    digest.add_string(identity.grouping);
    digest.add_string(identity.signal_unit);
    digest.add_string(identity.estimator_identity);
    digest.add_string(identity.response_identity);
    digest.add_string(identity.parallel_equivalence_policy);
    digest.add_integer(identity.required_companions.size());
    for (const auto &companion : identity.required_companions) {
        digest.add_string(companion);
    }
    digest.add_string(identity.validity_policy);
    digest.add_string(identity.coefficient_policy);
    digest.add_string(identity.normalization_support_policy);
    digest.add_string(identity.science_policy_support_policy);
    digest.add_string(identity.nonfinite_policy);
    digest.add_string(identity.wcs.coordinate_frame);
    digest.add_string(identity.wcs.projection);
    digest.add_integer(identity.wcs.axis_types.size());
    for (const auto &value : identity.wcs.axis_types) {
        digest.add_string(value);
    }
    digest.add_integer(identity.wcs.axis_units.size());
    for (const auto &value : identity.wcs.axis_units) {
        digest.add_string(value);
    }
    digest.add_integer(identity.wcs.pixel_scale.size());
    for (const auto value : identity.wcs.pixel_scale) {
        digest.add_double(value);
    }
    digest.add_integer(identity.wcs.reference_world.size());
    for (const auto value : identity.wcs.reference_world) {
        digest.add_double(value);
    }
    digest.add_integer(identity.wcs.reference_pixel.size());
    for (const auto value : identity.wcs.reference_pixel) {
        digest.add_double(value);
    }
    digest.add_double(identity.wcs.source_epoch);
    digest.add_double(identity.wcs.orientation_rad);
    digest.add_integer(identity.rows);
    digest.add_integer(identity.cols);
    digest.add_integer(identity.ordered_slots.size());
    for (const auto &slot : identity.ordered_slots) {
        digest.add_integer(slot.ordered_slot);
        digest.add_string(slot.grouping);
        digest.add_string(slot.group_identity);
        digest.add_integer(slot.array_identity);
        digest.add_integer(slot.stokes_identity);
        digest.add_double(slot.frequency_hz);
    }
    return digest.finish();
}

template <class Matrix>
inline void science_map_hash_matrix(ScienceMapCanonicalDigest &digest,
                                    const Matrix &matrix) {
    const auto rows = matrix.rows();
    const auto cols = matrix.cols();
    digest.add_integer(rows);
    digest.add_integer(cols);
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            using Scalar = std::remove_cv_t<typename Matrix::Scalar>;
            if constexpr (std::is_integral_v<Scalar>) {
                digest.add_integer(matrix(row, col));
            }
            else {
                digest.add_double(static_cast<double>(matrix(row, col)));
            }
        }
    }
}

// Canonical digest of the immutable numerical/product bundle consumed by a
// downstream science-map operator. The bundle identity is included by digest,
// and every typed plane is hashed without converting integer facts through
// binary64.
template <class MapBuffer>
inline std::string science_map_raw_parent_digest(const MapBuffer &buffer,
                                                 std::size_t slot) {
    const auto &products = buffer.science_products;
    if (!products.bundle_identity || slot >= buffer.signal.size() ||
        slot >= buffer.weight.size() ||
        slot >= products.geometric_hits.size() ||
        slot >= products.contributing_hits.size() ||
        slot >= products.coadd_observation_count.size() ||
        slot >= products.upstream_eligible_exposure.size() ||
        slot >= products.retained_exposure.size() ||
        slot >= products.normalization_support.size() ||
        slot >= products.science_policy_support.size() ||
        slot >= products.science_valid.size() ||
        slot >= products.realized.size()) {
        throw std::logic_error(
            "science-map raw-parent digest requires a complete typed bundle");
    }

    ScienceMapCanonicalDigest digest;
    digest.add_string(science_map_bundle_identity_digest(
        *products.bundle_identity));
    if (products.is_coadd) {
        digest.add_integer(products.coadd_admissions.size());
        for (const auto &admission : products.coadd_admissions) {
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
            digest.add_integer(
                admission.numerically_contributing_pixel_count.size());
            for (const auto value :
                 admission.numerically_contributing_pixel_count) {
                digest.add_integer(value);
            }
            digest.add_integer(
                admission.observation_raw_parent_digests.size());
            for (const auto &value :
                 admission.observation_raw_parent_digests) {
                digest.add_string(value);
            }
        }
    }
    science_map_hash_matrix(digest, buffer.signal[slot]);
    science_map_hash_matrix(digest, buffer.weight[slot]);
    if (!buffer.kernel.empty()) {
        if (slot >= buffer.kernel.size()) {
            throw std::logic_error(
                "science-map raw-parent digest kernel inventory mismatch");
        }
        science_map_hash_matrix(digest, buffer.kernel[slot]);
    }
    if (!buffer.noise.empty()) {
        if (slot >= buffer.noise.size() || buffer.n_noise < 0 ||
            buffer.noise[slot].dimension(0) != buffer.n_rows ||
            buffer.noise[slot].dimension(1) != buffer.n_cols ||
            buffer.noise[slot].dimension(2) != buffer.n_noise) {
            throw std::logic_error(
                "science-map raw-parent digest realization inventory mismatch");
        }
        digest.add_integer(buffer.n_noise);
        for (Eigen::Index realization = 0; realization < buffer.n_noise;
             ++realization) {
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                           Eigen::Dynamic>> noise_matrix(
                buffer.noise[slot].data() +
                    realization * buffer.n_rows * buffer.n_cols,
                buffer.n_rows, buffer.n_cols);
            science_map_hash_matrix(digest, noise_matrix);
        }
    }
    science_map_hash_matrix(digest, products.geometric_hits[slot]);
    science_map_hash_matrix(digest, products.contributing_hits[slot]);
    science_map_hash_matrix(digest, products.coadd_observation_count[slot]);
    science_map_hash_matrix(digest, products.upstream_eligible_exposure[slot]);
    science_map_hash_matrix(digest, products.retained_exposure[slot]);
    science_map_hash_matrix(digest, products.normalization_support[slot]);
    science_map_hash_matrix(digest, products.science_policy_support[slot]);
    science_map_hash_matrix(digest, products.science_valid[slot]);

    // Bind realized lifecycle/provenance facts to the product bytes. The
    // digest field itself is deliberately excluded so this value can be
    // recomputed at admission and detect a one-field provenance mutation.
    const auto &record = products.realized[slot];
    digest.add_integer(record.initialized);
    for (std::size_t product = 0;
         product < static_cast<std::size_t>(ScienceMapProduct::count);
         ++product) {
        digest.add_integer(record.product_available[product]);
        digest.add_string(record.product_absence_reason[product]);
        digest.add_integer(record.product_nonzero_count[product]);
        digest.add_string(record.product_value_sum[product]);
    }
    const auto hash_threshold = [&](const ScienceMapThresholdRealization &value) {
        digest.add_string(value.order_statistic_algorithm);
        digest.add_string(value.support_algorithm);
        digest.add_string(value.coefficient_product);
        digest.add_string(value.coefficient_stage);
        digest.add_double(value.requested_cut);
        digest.add_double(value.realized_cut);
        digest.add_double(value.realized_threshold);
        digest.add_double(value.selected_positive_value);
        digest.add_integer(value.positive_value_count);
        digest.add_integer(value.selected_zero_based_index);
        digest.add_integer(value.selected_index_available);
        digest.add_string(value.finite_convention);
        digest.add_string(value.positivity_convention);
        digest.add_string(value.comparison_convention);
    };
    hash_threshold(record.normalization);
    hash_threshold(record.science_policy);
    digest.add_integer(record.required_companions.size());
    for (const auto &companion : record.required_companions) {
        digest.add_string(companion);
    }
    digest.add_string(record.admitted_bundle_identity);
    return digest.finish();
}

template <class Plane>
inline std::size_t science_map_nonzero_count(const Plane &plane) {
    return static_cast<std::size_t>((plane.array() != 0).count());
}

template <class Plane>
inline std::string science_map_integer_plane_sum(const Plane &plane) {
    static_assert(std::is_integral_v<typename Plane::Scalar>);
    std::int64_t sum = 0;
    for (Eigen::Index col = 0; col < plane.cols(); ++col) {
        for (Eigen::Index row = 0; row < plane.rows(); ++row) {
            const auto value = static_cast<std::int64_t>(plane(row, col));
            if (value < 0 ||
                sum > std::numeric_limits<std::int64_t>::max() - value) {
                throw std::overflow_error(
                    "science-map count plane is negative or overflows its sum");
            }
            sum += value;
        }
    }
    return std::to_string(sum);
}

template <class Plane>
inline std::string science_map_exposure_plane_sum(const Plane &plane) {
    const double sum = plane.sum();
    if (!std::isfinite(sum) || (plane.array() < 0.0).any() ||
        !plane.array().isFinite().all()) {
        throw std::overflow_error(
            "science-map exposure plane is negative or non-finite");
    }
    return science_map_double_hex(sum);
}

template <class MapBuffer>
inline void science_map_finalize_realized_product_facts(MapBuffer &buffer,
                                                        std::size_t slot) {
    auto &products = buffer.science_products;
    if (!products.bundle_identity || slot >= products.realized.size()) {
        throw std::logic_error(
            "science-map product finalization requires an admitted identity");
    }
    auto &record = products.realized[slot];
    record.initialized = true;
    record.product_available.fill(true);
    record.product_absence_reason.fill("");
    if (!products.is_coadd) {
        const auto index = static_cast<std::size_t>(
            ScienceMapProduct::coadd_observation_count);
        record.product_available[index] = false;
        record.product_absence_reason[index] =
            "not applicable to observation maps";
    }
    record.required_companions = products.bundle_identity->required_companions;
    record.admitted_bundle_identity =
        science_map_bundle_identity_digest(*products.bundle_identity);

    const auto set_integer = [&](ScienceMapProduct product,
                                 const auto &plane) {
        const auto index = static_cast<std::size_t>(product);
        record.product_nonzero_count[index] =
            science_map_nonzero_count(plane);
        record.product_value_sum[index] =
            science_map_integer_plane_sum(plane);
    };
    const auto set_exposure = [&](ScienceMapProduct product,
                                  const auto &plane) {
        const auto index = static_cast<std::size_t>(product);
        record.product_nonzero_count[index] =
            science_map_nonzero_count(plane);
        record.product_value_sum[index] =
            science_map_exposure_plane_sum(plane);
    };
    set_integer(ScienceMapProduct::geometric_hits,
                products.geometric_hits.at(slot));
    set_integer(ScienceMapProduct::contributing_hits,
                products.contributing_hits.at(slot));
    set_integer(ScienceMapProduct::coadd_observation_count,
                products.coadd_observation_count.at(slot));
    set_exposure(ScienceMapProduct::upstream_eligible_exposure,
                 products.upstream_eligible_exposure.at(slot));
    set_exposure(ScienceMapProduct::retained_exposure,
                 products.retained_exposure.at(slot));
    set_integer(ScienceMapProduct::normalization_support,
                products.normalization_support.at(slot));
    set_integer(ScienceMapProduct::science_policy_support,
                products.science_policy_support.at(slot));
    set_integer(ScienceMapProduct::science_valid,
                products.science_valid.at(slot));
    record.raw_parent_digest = science_map_raw_parent_digest(buffer, slot);
}

inline bool science_map_realized_product_facts_match(
    const ScienceMapProducts &products, std::size_t slot) {
    if (!products.bundle_identity || slot >= products.realized.size()) {
        return false;
    }
    const auto &record = products.realized[slot];
    if (!record.initialized ||
        record.required_companions !=
            products.bundle_identity->required_companions ||
        record.admitted_bundle_identity !=
            science_map_bundle_identity_digest(*products.bundle_identity)) {
        return false;
    }
    for (std::size_t product = 0;
         product < static_cast<std::size_t>(ScienceMapProduct::count);
         ++product) {
        const bool expected_available = products.is_coadd ||
            product != static_cast<std::size_t>(
                ScienceMapProduct::coadd_observation_count);
        if (record.product_available[product] != expected_available ||
            (expected_available &&
             !record.product_absence_reason[product].empty()) ||
            (!expected_available &&
             record.product_absence_reason[product] !=
                 "not applicable to observation maps")) {
            return false;
        }
    }
    const auto integer_matches = [&](ScienceMapProduct product,
                                     const auto &plane) {
        const auto index = static_cast<std::size_t>(product);
        return record.product_nonzero_count[index] ==
                   science_map_nonzero_count(plane) &&
               record.product_value_sum[index] ==
                   science_map_integer_plane_sum(plane);
    };
    const auto exposure_matches = [&](ScienceMapProduct product,
                                      const auto &plane) {
        const auto index = static_cast<std::size_t>(product);
        return record.product_nonzero_count[index] ==
                   science_map_nonzero_count(plane) &&
               record.product_value_sum[index] ==
                   science_map_exposure_plane_sum(plane);
    };
    return integer_matches(ScienceMapProduct::geometric_hits,
                           products.geometric_hits.at(slot)) &&
           integer_matches(ScienceMapProduct::contributing_hits,
                           products.contributing_hits.at(slot)) &&
           integer_matches(ScienceMapProduct::coadd_observation_count,
                           products.coadd_observation_count.at(slot)) &&
           exposure_matches(ScienceMapProduct::upstream_eligible_exposure,
                            products.upstream_eligible_exposure.at(slot)) &&
           exposure_matches(ScienceMapProduct::retained_exposure,
                            products.retained_exposure.at(slot)) &&
           integer_matches(ScienceMapProduct::normalization_support,
                           products.normalization_support.at(slot)) &&
           integer_matches(ScienceMapProduct::science_policy_support,
                           products.science_policy_support.at(slot)) &&
           integer_matches(ScienceMapProduct::science_valid,
                           products.science_valid.at(slot));
}

template <class MapBuffer>
inline bool science_map_realized_product_facts_match(
    const MapBuffer &buffer, std::size_t slot) {
    return science_map_realized_product_facts_match(
        buffer.science_products, slot);
}

inline bool science_map_product_available(const ScienceMapRealizedMap &record,
                                          ScienceMapProduct product) {
    return record.product_available.at(static_cast<std::size_t>(product));
}

}  // namespace mapmaking

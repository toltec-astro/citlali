#pragma once

#include <citlali/core/mapmaking/science_map_contract.h>
#include <citlali/core/mapmaking/jinc_contract.h>
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/noise_execution_plan.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <type_traits>

namespace citlali::pipeline {

template <class Hdu>
void add_noise_product_package_join(
    Hdu &hdu, const char *product_identity, const std::string &scope,
    const std::string &validity, const std::string &restriction) {
    add_noise_product_join_keys(
        hdu, noise_package_id, noise_provenance_join_id, product_identity,
        noise_product_contract_version,
        noise_product_semantic_digest(product_identity), scope, validity,
        restriction);
}

template <class ScienceProducts>
bool science_map_any_product_available(const ScienceProducts &products) {
    return products.initialized && std::any_of(
        products.realized.begin(), products.realized.end(),
        [](const auto &record) {
            return std::any_of(record.product_available.begin(),
                               record.product_available.end(),
                               [](bool available) { return available; });
        });
}

template <class ScienceProducts>
bool science_map_successor_coadd_product(
    const ScienceProducts &products) {
    return products.initialized && products.is_coadd &&
           products.ordinary_contribution_predicate_available;
}

// Output-family identity is deliberately weaker than successor science-map
// admission.  A method such as JINC can retain the established legacy coadd
// arithmetic while truthfully remaining outside the F009/F010 successor
// profile.  It is still a coadd product, however, and must not publish
// observation-only empirical-noise companions merely because that richer
// science-map profile is unavailable.  Legacy coadds retain legacy iteration
// control metadata such as MEDRMS; successor coadds replace that contract.
template <class ScienceProducts>
bool science_map_coadd_output_product(
    const ScienceProducts &products) {
    return products.initialized && products.is_coadd;
}

inline std::size_t science_map_product_index(
    mapmaking::ScienceMapProduct product) {
    return static_cast<std::size_t>(product);
}

template <class ScienceProducts>
bool science_map_product_available(
    const ScienceProducts &products, Eigen::Index map_index,
    mapmaking::ScienceMapProduct product) {
    return products.initialized && map_index >= 0 &&
           map_index < static_cast<Eigen::Index>(products.realized.size()) &&
           products.realized[static_cast<std::size_t>(map_index)]
               .product_available[science_map_product_index(product)];
}

template <class ScienceProducts>
bool science_map_available_product_planes_have_shape(
    const ScienceProducts &products, Eigen::Index map_index,
    Eigen::Index rows, Eigen::Index cols) {
    const auto available_shape = [&](mapmaking::ScienceMapProduct product,
                                     const auto &planes) {
        return !science_map_product_available(products, map_index, product) ||
               has_map_image_slot(planes, map_index, rows, cols);
    };
    return available_shape(mapmaking::ScienceMapProduct::geometric_hits,
                           products.geometric_hits) &&
           available_shape(mapmaking::ScienceMapProduct::contributing_hits,
                           products.contributing_hits) &&
           available_shape(
               mapmaking::ScienceMapProduct::coadd_observation_count,
               products.coadd_observation_count) &&
           available_shape(
               mapmaking::ScienceMapProduct::upstream_eligible_exposure,
               products.upstream_eligible_exposure) &&
           available_shape(mapmaking::ScienceMapProduct::retained_exposure,
                           products.retained_exposure) &&
           available_shape(mapmaking::ScienceMapProduct::normalization_support,
                           products.normalization_support) &&
           available_shape(mapmaking::ScienceMapProduct::science_policy_support,
                           products.science_policy_support) &&
           available_shape(mapmaking::ScienceMapProduct::science_valid,
                           products.science_valid);
}

template <class ScienceProducts>
bool science_map_supported_output_bundle_complete(
    const ScienceProducts &products, std::size_t map_count,
    Eigen::Index rows, Eigen::Index cols) {
    if (!products.initialized ||
        !products.ordinary_contribution_predicate_available ||
        !products.identity_admitted || !products.bundle_identity ||
        products.bundle_identity->ordered_slots.size() != map_count ||
        products.realized.size() != map_count ||
        products.geometric_hits.size() != map_count ||
        products.contributing_hits.size() != map_count ||
        products.coadd_observation_count.size() != map_count ||
        products.upstream_eligible_exposure.size() != map_count ||
        products.retained_exposure.size() != map_count ||
        products.normalization_support.size() != map_count ||
        products.science_policy_support.size() != map_count ||
        products.science_valid.size() != map_count) {
        return false;
    }
    try {
        for (std::size_t slot = 0; slot < map_count; ++slot) {
            if (products.realized[slot].raw_parent_digest.empty() ||
                !mapmaking::science_map_realized_product_facts_match(
                    products, slot) ||
                !science_map_available_product_planes_have_shape(
                    products, static_cast<Eigen::Index>(slot), rows, cols)) {
                return false;
            }
        }
    }
    catch (const std::exception &) {
        return false;
    }
    return true;
}

template <class ScienceProducts>
bool science_map_unavailable_output_bundle_complete(
    const ScienceProducts &products, std::size_t map_count) {
    if (!products.initialized ||
        products.ordinary_contribution_predicate_available ||
        products.identity_admitted || products.bundle_identity ||
        products.realized.size() != map_count ||
        !products.geometric_hits.empty() ||
        !products.contributing_hits.empty() ||
        !products.coadd_observation_count.empty() ||
        !products.upstream_eligible_exposure.empty() ||
        !products.retained_exposure.empty() ||
        !products.normalization_support.empty() ||
        !products.science_policy_support.empty() ||
        !products.science_valid.empty() ||
        !products.coadd_admissions.empty()) {
        return false;
    }
    return std::all_of(
        products.realized.begin(), products.realized.end(),
        [](const auto &record) {
            return mapmaking::
                science_map_realized_map_has_explicit_product_absence(record);
        });
}

template <class ScienceProducts, class Logger>
void require_science_map_output_profile_authority(
    const ScienceProducts &products, std::size_t map_count,
    Eigen::Index rows, Eigen::Index cols, const Logger &logger) {
    const bool complete = products.initialized &&
        (products.ordinary_contribution_predicate_available
             ? science_map_supported_output_bundle_complete(
                   products, map_count, rows, cols)
             : science_map_unavailable_output_bundle_complete(
                   products, map_count));
    if (!complete) {
        fail_required_output(
            logger,
            "science-map output profile authority or product inventory is incomplete");
    }
}

template <typename LeftDerived, typename RightDerived>
bool science_map_planes_bitwise_equal(
    const Eigen::DenseBase<LeftDerived> &left,
    const Eigen::DenseBase<RightDerived> &right) {
    using LeftScalar = typename LeftDerived::Scalar;
    using RightScalar = typename RightDerived::Scalar;
    static_assert(std::is_same_v<LeftScalar, RightScalar>,
                  "bitwise alias comparison requires identical scalar types");
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        return false;
    }
    for (Eigen::Index row = 0; row < left.rows(); ++row) {
        for (Eigen::Index col = 0; col < left.cols(); ++col) {
            if (std::memcmp(&left.derived()(row, col),
                            &right.derived()(row, col),
                            sizeof(LeftScalar)) != 0) {
                return false;
            }
        }
    }
    return true;
}

template <class FitsEntry, class MapBuffer, class Wcs, class Logger>
void add_science_map_product_image_hdus(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Wcs &wcs, double source_epoch, const Logger &logger,
    bool is_filtered_output = false) {
    const bool frozen_parent_available =
        static_cast<bool>(mb->raw_science_parent);
    if (is_filtered_output && !frozen_parent_available) {
        const auto &live = mb->science_products;
        const bool live_has_product =
            science_map_any_product_available(live);
        if (live_has_product) {
            fail_required_output(
                logger,
                "filtered science-map output lacks an immutable raw-parent snapshot");
        }
    }
    const auto &products =
        is_filtered_output && frozen_parent_available
            ? *mb->raw_science_parent
            : mb->science_products;
    if (!products.initialized || i < 0 ||
        i >= static_cast<Eigen::Index>(products.realized.size())) {
        return;
    }
    const auto slot = static_cast<std::size_t>(i);
    const auto &record = products.realized[slot];
    const bool any_product_available = std::any_of(
        record.product_available.begin(), record.product_available.end(),
        [](bool available) { return available; });
    if (!any_product_available) {
        return;
    }
    const bool facts_match =
        mapmaking::science_map_realized_product_facts_match(products, slot);
    const bool raw_parent_matches = is_filtered_output
        ? !record.raw_parent_digest.empty()
        : record.raw_parent_digest ==
              mapmaking::science_map_raw_parent_digest(*mb, slot);
    if (!products.identity_admitted || !products.bundle_identity ||
        !facts_match || !raw_parent_matches) {
        fail_required_output(
            logger,
            fmt::format(
                "science-map product state/provenance is stale or incomplete for map index {}",
                static_cast<long long>(i)));
    }

    auto preflight_product = [&](mapmaking::ScienceMapProduct product,
                                 const auto &planes,
                                 const std::string &hdu_name) {
        const bool available =
            science_map_product_available(products, i, product);
        if (available &&
            !has_map_image_slot(planes, i, mb->n_rows, mb->n_cols)) {
            fail_required_output(
                logger,
                fmt::format(
                    "science-map product {} is marked available but map index {} lacks the declared plane",
                    hdu_name, static_cast<long long>(i)));
        }
        return available;
    };

    preflight_product(
        mapmaking::ScienceMapProduct::geometric_hits,
        products.geometric_hits,
        geometric_hits_map_hdu_name(map_name, stokes_suffix));
    preflight_product(
        mapmaking::ScienceMapProduct::contributing_hits,
        products.contributing_hits,
        contributing_hits_map_hdu_name(map_name, stokes_suffix));
    preflight_product(
        mapmaking::ScienceMapProduct::coadd_observation_count,
        products.coadd_observation_count,
        coadd_observation_count_map_hdu_name(map_name, stokes_suffix));
    preflight_product(
        mapmaking::ScienceMapProduct::upstream_eligible_exposure,
        products.upstream_eligible_exposure,
        upstream_eligible_exposure_map_hdu_name(map_name, stokes_suffix));
    const bool retained_declared = preflight_product(
        mapmaking::ScienceMapProduct::retained_exposure,
        products.retained_exposure,
        retained_exposure_map_hdu_name(map_name, stokes_suffix));
    preflight_product(
        mapmaking::ScienceMapProduct::normalization_support,
        products.normalization_support,
        normalization_support_map_hdu_name(map_name, stokes_suffix));
    preflight_product(
        mapmaking::ScienceMapProduct::science_policy_support,
        products.science_policy_support,
        science_policy_support_map_hdu_name(map_name, stokes_suffix));
    preflight_product(
        mapmaking::ScienceMapProduct::science_valid,
        products.science_valid,
        science_valid_map_hdu_name(map_name, stokes_suffix));

    if (!is_filtered_output && retained_declared &&
        (!has_map_image_slot(mb->coverage, i, mb->n_rows, mb->n_cols) ||
         !science_map_planes_bitwise_equal(
             mb->coverage[static_cast<std::size_t>(i)],
             products.retained_exposure[static_cast<std::size_t>(i)]))) {
        fail_required_output(
            logger,
            fmt::format(
                "coverage compatibility alias differs from retained exposure for map index {}",
                static_cast<long long>(i)));
    }

    auto write_product = [&](mapmaking::ScienceMapProduct product,
                             auto &planes, const std::string &hdu_name,
                             const auto &add_metadata) {
        if (!science_map_product_available(products, i, product)) {
            return false;
        }
        add_map_hdu_with_wcs(
            fits_entry, hdu_name, planes[static_cast<std::size_t>(i)], wcs,
            source_epoch);
        add_metadata(*fits_entry.hdus.back());
        if (is_filtered_output) {
            add_raw_parent_identity_keys(
                *fits_entry.hdus.back(), record.raw_parent_digest);
        }
        return true;
    };

    write_product(
        mapmaking::ScienceMapProduct::geometric_hits,
        products.geometric_hits,
        geometric_hits_map_hdu_name(map_name, stokes_suffix),
        [](auto &hdu) { add_geometric_hits_map_metadata(hdu); });
    write_product(
        mapmaking::ScienceMapProduct::contributing_hits,
        products.contributing_hits,
        contributing_hits_map_hdu_name(map_name, stokes_suffix),
        [](auto &hdu) { add_contributing_hits_map_metadata(hdu); });
    write_product(
        mapmaking::ScienceMapProduct::coadd_observation_count,
        products.coadd_observation_count,
        coadd_observation_count_map_hdu_name(map_name, stokes_suffix),
        [](auto &hdu) { add_coadd_observation_count_map_metadata(hdu); });
    write_product(
        mapmaking::ScienceMapProduct::upstream_eligible_exposure,
        products.upstream_eligible_exposure,
        upstream_eligible_exposure_map_hdu_name(map_name, stokes_suffix),
        [](auto &hdu) { add_upstream_eligible_exposure_map_metadata(hdu); });

    const std::string retained_name =
        retained_exposure_map_hdu_name(map_name, stokes_suffix);
    const bool retained_available = write_product(
        mapmaking::ScienceMapProduct::retained_exposure,
        products.retained_exposure, retained_name,
        [](auto &hdu) { add_retained_exposure_map_metadata(hdu); });

    const bool normalization_available = write_product(
        mapmaking::ScienceMapProduct::normalization_support,
        products.normalization_support,
        normalization_support_map_hdu_name(map_name, stokes_suffix),
        [](auto &hdu) { add_normalization_support_map_metadata(hdu); });
    if (normalization_available) {
        const auto &realized =
            products.realized[static_cast<std::size_t>(i)].normalization;
        add_image_weight_threshold_key(
            *fits_entry.hdus.back(), realized.realized_threshold);
    }

    const std::string policy_name =
        science_policy_support_map_hdu_name(map_name, stokes_suffix);
    const bool policy_available = write_product(
        mapmaking::ScienceMapProduct::science_policy_support,
        products.science_policy_support, policy_name,
        [](auto &hdu) { add_science_policy_support_map_metadata(hdu); });
    if (policy_available) {
        const auto &realized =
            products.realized[static_cast<std::size_t>(i)].science_policy;
        add_image_weight_threshold_key(
            *fits_entry.hdus.back(), realized.realized_threshold);
    }

    write_product(
        mapmaking::ScienceMapProduct::science_valid,
        products.science_valid,
        science_valid_map_hdu_name(map_name, stokes_suffix),
        [](auto &hdu) { add_science_valid_map_metadata(hdu); });

    if (retained_available) {
        add_map_hdu_with_wcs(
            fits_entry, coverage_map_hdu_name(map_name, stokes_suffix),
            products.retained_exposure[static_cast<std::size_t>(i)], wcs,
            source_epoch);
        add_coverage_map_metadata(*fits_entry.hdus.back(), retained_name);
        if (is_filtered_output) {
            add_raw_parent_identity_keys(
                *fits_entry.hdus.back(), record.raw_parent_digest);
        }
    }

    if (policy_available) {
        add_map_hdu_with_wcs(
            fits_entry, coverage_mask_map_hdu_name(map_name, stokes_suffix),
            products.science_policy_support[static_cast<std::size_t>(i)], wcs,
            source_epoch);
        add_coverage_mask_map_metadata(*fits_entry.hdus.back(), policy_name);
        if (is_filtered_output) {
            add_raw_parent_identity_keys(
                *fits_entry.hdus.back(), record.raw_parent_digest);
        }
        const auto &realized =
            products.realized[static_cast<std::size_t>(i)].science_policy;
        add_image_weight_threshold_key(
            *fits_entry.hdus.back(), realized.realized_threshold);
    }
}

template <class FitsEntry, class MapBuffer, class Wcs, class Logger>
void add_primary_map_image_hdus(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Wcs &wcs, double source_epoch, bool empirical_weight_calibration,
    bool empirical_noise_products_expected, bool is_beammap,
    bool coadd_output_product, bool successor_coadd_product,
    const Logger &logger,
    bool is_filtered_output = false,
    const std::string &raw_parent_digest = {}) {
    const bool publish_empirical_product_companions =
        empirical_noise_products_expected && !coadd_output_product;
    const bool publish_legacy_fruit_loop_scale =
        coadd_output_product && !successor_coadd_product &&
        !is_filtered_output;
    const bool stack_scatter_valid =
        i >= 0 && i < mb->noise_stack_scatter_valid.size()
            ? mb->noise_stack_scatter_valid(i) != 0
            : mb->n_noise > 0;
    const bool scale_valid =
        i >= 0 && i < mb->noise_weight_scale_valid.size()
            ? mb->noise_weight_scale_valid(i) != 0
            : i >= 0 && i < mb->noise_weight_scale.size() &&
                  std::isfinite(mb->noise_weight_scale(i)) &&
                  mb->noise_weight_scale(i) > 0.0;
    const std::string map_pixel_scope =
        is_filtered_output ? "filtered_map_pixel" : "raw_map_pixel";
    if (publish_empirical_product_companions &&
        (!has_map_image_slot(mb->weight_formal, i, mb->n_rows, mb->n_cols) ||
         !has_map_image_slot(mb->noise_variance, i, mb->n_rows,
                             mb->n_cols))) {
        fail_required_output(
            logger,
            fmt::format(
                "empirical noise products were requested but map index {} lacks formal-coefficient or conditional-stack-scatter data",
                static_cast<long long>(i)));
    }
    if (!successor_coadd_product &&
        (i < 0 || i >= mb->median_err.size())) {
        fail_required_output(
            logger,
            fmt::format(
                "observation map index {} lacks the required legacy median coefficient-scale diagnostic",
                static_cast<long long>(i)));
    }
    add_map_hdu_with_wcs(
        fits_entry, signal_map_hdu_name(map_name, stokes_suffix),
        mb->signal[i], wcs, source_epoch);
    add_signal_map_metadata(*fits_entry.hdus.back(), mb->sig_unit);
    if (is_filtered_output && !raw_parent_digest.empty()) {
        add_raw_parent_identity_keys(
            *fits_entry.hdus.back(), raw_parent_digest);
    }

    add_map_hdu_with_wcs(
        fits_entry, weight_map_hdu_name(map_name, stokes_suffix),
        mb->weight[i], wcs, source_epoch);
    const std::string weight_unit = map_weight_unit(mb->sig_unit);
    if (mb->jinc_products.initialized) {
        add_jinc_weight_map_metadata(
            *fits_entry.hdus.back(), weight_unit,
            empirical_weight_calibration);
    }
    else {
        add_weight_map_metadata(
            *fits_entry.hdus.back(), weight_unit,
            empirical_weight_calibration);
    }
    if (empirical_weight_calibration) {
        add_noise_product_package_join(
            *fits_entry.hdus.back(), noise_scaled_coefficient_product_id,
            map_pixel_scope, scale_valid ? "available" : "unavailable",
            "existing_use_only_nonprecision_not_precision");
    }
    if (is_filtered_output && !raw_parent_digest.empty()) {
        add_raw_parent_identity_keys(
            *fits_entry.hdus.back(), raw_parent_digest);
    }
    if (publish_legacy_fruit_loop_scale && i >= 0 &&
        i < mb->median_rms.size() && std::isfinite(mb->median_rms(i)) &&
        mb->median_rms(i) > 0.0) {
        add_image_median_rms_key(
            *fits_entry.hdus.back(), mb->median_rms(i), mb->sig_unit);
    }
    if (publish_empirical_product_companions) {
        const Eigen::Index valid_pixels =
            i >= 0 && i < mb->noise_valid_pixels.size()
                ? static_cast<Eigen::Index>(mb->noise_valid_pixels(i))
                : 0;
        const double median_ratio =
            i >= 0 && i < mb->noise_weight_median_ratio.size()
                ? mb->noise_weight_median_ratio(i)
                : std::numeric_limits<double>::quiet_NaN();
        const double scale =
            i >= 0 && i < mb->noise_weight_scale.size()
                ? mb->noise_weight_scale(i)
                : std::numeric_limits<double>::quiet_NaN();
        add_global_nonprecision_scale_diagnostic_keys(
            *fits_entry.hdus.back(), scale_valid, valid_pixels,
            median_ratio, scale);
        if (scale_valid) {
            add_empirical_weight_scale_key(*fits_entry.hdus.back(), scale);
            add_weight_variance_median_key(
                *fits_entry.hdus.back(), median_ratio);
        }
    }

    if (!successor_coadd_product) {
        const double median_err_value = mb->median_err(i);
        const double median_err = map_median_error_or_zero_logged(
            median_err_value, is_beammap, map_name, fits_entry.filepath,
            logger);
        add_image_median_error_key(
            *fits_entry.hdus.back(), median_err, mb->sig_unit);
    }

    if (publish_empirical_product_companions) {
        add_map_hdu_with_wcs(
            fits_entry, formal_weight_map_hdu_name(map_name, stokes_suffix),
            mb->weight_formal[i], wcs, source_epoch);
        if (mb->jinc_products.initialized) {
            add_jinc_formal_weight_map_metadata(
                *fits_entry.hdus.back(), weight_unit);
        }
        else {
            add_formal_weight_map_metadata(
                *fits_entry.hdus.back(), weight_unit);
        }
        add_noise_product_package_join(
            *fits_entry.hdus.back(), noise_formal_coefficient_product_id,
            map_pixel_scope, "available",
            "nonprecision_snapshot_not_inverse_variance");

        const std::string variance_unit = map_variance_unit(mb->sig_unit);
        add_map_hdu_with_wcs(
            fits_entry, noise_variance_map_hdu_name(map_name, stokes_suffix),
            mb->noise_variance[i], wcs, source_epoch);
        add_noise_variance_map_metadata(
            *fits_entry.hdus.back(), variance_unit, "");
        add_noise_product_package_join(
            *fits_entry.hdus.back(),
            noise_conditional_stack_scatter_product_id, map_pixel_scope,
            stack_scatter_valid ? "conditional_descriptive" : "unavailable",
            is_filtered_output
                ? "retained_legacy_name_not_physical_noise_variance_strict_parity_pending_FLT"
                : "retained_legacy_name_not_physical_noise_variance_or_covariance");
        if (i < mb->median_rms.size() && std::isfinite(mb->median_rms(i))) {
            add_image_median_rms_key(
                *fits_entry.hdus.back(), mb->median_rms(i), mb->sig_unit);
        }
    }
}

template <class FitsEntry, class MapBuffer, class Kernel, class ArrayFwhm,
          class Wcs, class Logger>
void add_kernel_map_image_hdu(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Kernel &kernel, const ArrayFwhm &array_fwhm, const Wcs &wcs,
    double source_epoch, double rad_to_arcsec, const Logger &logger) {
    fits_entry.add_hdu(kernel_map_hdu_name(map_name, stokes_suffix),
                       mb->kernel[i]);
    add_image_type_key(
        *fits_entry.hdus.back(), kernel.type, kernel_type_comment());

    double fwhm = kernel_fwhm_arcsec(
        kernel.type, kernel.fwhm_rad, array_fwhm, rad_to_arcsec);
    fwhm = kernel_fwhm_or_invalid(
        fwhm, map_name, fits_entry.filepath, logger);
    add_kernel_fwhm_key(*fits_entry.hdus.back(), fwhm);
    fits_entry.add_wcs(fits_entry.hdus.back(), wcs, source_epoch);
    if (mb->jinc_products.initialized) {
        add_jinc_kernel_map_metadata(
            *fits_entry.hdus.back(), mb->sig_unit);
    }
    else {
        add_kernel_map_metadata(*fits_entry.hdus.back(), mb->sig_unit);
    }
}

struct FilteredScatterValidity {
    bool available = false;
    const char *status = "scatter_unavailable_or_nonfinite";
};

template <class ScatterPlane, class SupportPlane>
FilteredScatterValidity filtered_scatter_validity(
    Eigen::Index completed_realizations, const ScatterPlane &scatter,
    double response_normalization, const SupportPlane *support) {
    if (completed_realizations < 2) {
        return {false, "R_lt_2"};
    }
    if (scatter.size() == 0 || !scatter.array().isFinite().any()) {
        return {false, "scatter_unavailable_or_nonfinite"};
    }
    if (!std::isfinite(response_normalization) ||
        response_normalization == 0.0) {
        return {false, "response_invalid"};
    }
    if (support == nullptr || support->rows() != scatter.rows() ||
        support->cols() != scatter.cols()) {
        return {false, "support_invalid"};
    }
    bool supported_value = false;
    for (Eigen::Index row = 0; row < scatter.rows(); ++row) {
        for (Eigen::Index col = 0; col < scatter.cols(); ++col) {
            if ((*support)(row, col) != 0) {
                supported_value = true;
                if (std::isfinite(scatter(row, col))) {
                    return {
                        true, "available_where_finite_on_valid_support"};
                }
            }
        }
    }
    return supported_value
        ? FilteredScatterValidity{
              false, "scatter_unavailable_or_nonfinite"}
        : FilteredScatterValidity{false, "support_invalid"};
}

template <class FitsEntry, class MapBuffer, class Wcs, class Logger>
void add_coverage_support_image_hdus(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Wcs &wcs, double source_epoch, bool is_filtered_output,
    bool empirical_noise_products_expected, bool coadd_product,
    const Logger &logger) {
    if (mb->jinc_products.initialized) {
        if (!has_map_image_slot(mb->coverage, i, mb->n_rows, mb->n_cols) ||
            i < 0 ||
            i >= static_cast<Eigen::Index>(
                mb->jinc_products.formal_support.size()) ||
            mb->jinc_products.formal_support[static_cast<std::size_t>(i)]
                    .rows() != mb->n_rows ||
            mb->jinc_products.formal_support[static_cast<std::size_t>(i)]
                    .cols() != mb->n_cols) {
            fail_required_output(
                logger,
                "JINC output lacks coefficient-squared coverage or authoritative formal support");
        }
        add_map_hdu_with_wcs(
            fits_entry, coverage_map_hdu_name(map_name, stokes_suffix),
            mb->coverage[static_cast<std::size_t>(i)], wcs, source_epoch);
        add_jinc_coverage_map_metadata(*fits_entry.hdus.back());
        add_map_hdu_with_wcs(
            fits_entry,
            coverage_mask_map_hdu_name(map_name, stokes_suffix),
            mb->jinc_products.formal_support[static_cast<std::size_t>(i)],
            wcs, source_epoch);
        add_jinc_formal_support_map_metadata(*fits_entry.hdus.back());
    }
    if (!mb->coverage.empty()) {
        const bool empirical_snr_available = has_map_image_slot(
            mb->sig2noise_pixel, i, mb->n_rows, mb->n_cols);
        if (!coadd_product && empirical_noise_products_expected &&
            !empirical_snr_available) {
            fail_required_output(
                logger,
                fmt::format(
                    "empirical noise products were requested but map index {} lacks coefficient-standardized signal data",
                    static_cast<long long>(i)));
        }
        const bool point_source_products_available =
            has_map_image_slot(
                mb->point_source_uncertainty, i, mb->n_rows, mb->n_cols) &&
            has_map_image_slot(
                mb->sig2noise_point_source, i, mb->n_rows, mb->n_cols);
        if (!coadd_product && is_filtered_output &&
            empirical_noise_products_expected &&
            !point_source_products_available) {
            fail_required_output(
                logger,
                fmt::format(
                    "empirical noise products were requested for filtered map index {} but filtered-pixel stack-scatter products are absent",
                    static_cast<long long>(i)));
        }
    }

    // SCI-MAP-001 support and validity are lifecycle-owned products. The
    // writer serializes those exact planes and never reconstructs a mask from
    // weight, exposure, finite signal, or a compatibility alias.
    add_science_map_product_image_hdus(
        fits_entry, mb, i, map_name, stokes_suffix, wcs, source_epoch,
        logger, is_filtered_output);

    // Detector-grouped Beammap products historically have no coverage family.
    // Their explicit F010 absence authority is checked above, then the original
    // no-coverage output guard prevents any new standardized/S/N family.
    if (mb->coverage.empty()) {
        return;
    }

    // SCI-MAP-001 keeps the coadd coefficient nonprecision and covariance
    // unavailable pending SCI-PTC-001. Coadds therefore publish the raw F010
    // hierarchy but no formal-standardized, S/N, uncertainty, or point-source
    // significance/flux family. Observation behavior remains profile-driven.
    if (coadd_product) {
        return;
    }

    if (empirical_noise_products_expected) {
        Eigen::MatrixXd &coefficient_standardized_signal =
            mb->sig2noise_pixel[i];
        const bool scale_valid =
            i >= 0 && i < mb->noise_weight_scale_valid.size()
                ? mb->noise_weight_scale_valid(i) != 0
                : i >= 0 && i < mb->noise_weight_scale.size() &&
                      std::isfinite(mb->noise_weight_scale(i)) &&
                      mb->noise_weight_scale(i) > 0.0;
        const char *product_scope =
            is_filtered_output ? "filtered_map_pixel" : "raw_map_pixel";
        const char *product_validity =
            scale_valid ? "available_where_finite" : "unavailable";
        add_map_hdu_with_wcs(
            fits_entry, legacy_pixel_snr_map_hdu_name(map_name, stokes_suffix),
            coefficient_standardized_signal, wcs, source_epoch);
        add_legacy_pixel_snr_map_metadata(
            *fits_entry.hdus.back(), "");
        add_noise_product_package_join(
            *fits_entry.hdus.back(),
            noise_coefficient_standardized_signal_product_id,
            product_scope, product_validity,
            "retained_legacy_name_engineering_standardization_not_significance");
    }
    else {
        Eigen::MatrixXd formal_standardized_signal =
            standardized_signal_from_weight(mb->signal[i], mb->weight[i]);
        add_map_hdu_with_wcs(
            fits_entry,
            formal_standardized_signal_map_hdu_name(
                map_name, stokes_suffix),
            formal_standardized_signal, wcs, source_epoch);
        add_formal_standardized_signal_map_metadata(
            *fits_entry.hdus.back());
    }

    if (is_filtered_output && empirical_noise_products_expected) {
        const auto &support_authority = mb->raw_science_parent
            ? *mb->raw_science_parent
            : mb->science_products;
        const mapmaking::ScienceMapMaskPlane *support = nullptr;
        if (science_map_product_available(
                support_authority, i,
                mapmaking::ScienceMapProduct::science_policy_support) &&
            i >= 0 &&
            i < static_cast<Eigen::Index>(
                    support_authority.science_policy_support.size())) {
            support = &support_authority.science_policy_support[
                static_cast<std::size_t>(i)];
        }
        constexpr double response_normalization = 1.0;
        const auto validity = filtered_scatter_validity(
            mb->n_noise, mb->point_source_uncertainty[i],
            response_normalization, support);
        const char *scatter_validity = validity.status;
        const char *ratio_validity = validity.available
            ? "available_where_finite_positive_denominator_on_valid_support"
            : validity.status;
        Eigen::MatrixXd filtered_scatter =
            mb->point_source_uncertainty[i];
        Eigen::MatrixXd filtered_ratio = mb->sig2noise_point_source[i];
        const double unavailable =
            std::numeric_limits<double>::quiet_NaN();
        if (!validity.available) {
            filtered_scatter.setConstant(unavailable);
            filtered_ratio.setConstant(unavailable);
        }
        else {
            for (Eigen::Index row = 0; row < filtered_scatter.rows(); ++row) {
                for (Eigen::Index col = 0;
                     col < filtered_scatter.cols(); ++col) {
                    if ((*support)(row, col) == 0) {
                        filtered_scatter(row, col) = unavailable;
                        filtered_ratio(row, col) = unavailable;
                    }
                }
            }
        }
        const std::string signal_name =
            signal_map_hdu_name(map_name, stokes_suffix);
        add_map_hdu_with_wcs(
            fits_entry, point_source_flux_map_hdu_name(
                map_name, stokes_suffix),
            mb->signal[i], wcs, source_epoch);
        add_point_source_flux_map_metadata(
            *fits_entry.hdus.back(), mb->sig_unit, signal_name);
        add_point_source_response_norm_key(
            *fits_entry.hdus.back(), response_normalization);

        add_map_hdu_with_wcs(
            fits_entry, point_source_uncertainty_map_hdu_name(
                map_name, stokes_suffix),
            filtered_scatter, wcs, source_epoch);
        add_point_source_uncertainty_map_metadata(
            *fits_entry.hdus.back(), mb->sig_unit, "");
        add_noise_product_package_join(
            *fits_entry.hdus.back(),
            noise_filtered_pixel_stack_scatter_product_id,
            "filtered_map_pixel", scatter_validity,
            "retained_legacy_name_not_aperture_uncertainty_strict_parity_pending_FLT");

        add_map_hdu_with_wcs(
            fits_entry, point_source_snr_map_hdu_name(
                map_name, stokes_suffix),
            filtered_ratio, wcs, source_epoch);
        add_point_source_snr_map_metadata(
            *fits_entry.hdus.back(), "");
        add_noise_product_package_join(
            *fits_entry.hdus.back(), noise_stack_scatter_ratio_product_id,
            "filtered_map_pixel", ratio_validity,
            "retained_legacy_name_conditional_descriptive_ratio_not_significance");
    }
}

template <class FitsEntry, class MapBuffer, class Wcs>
void add_noise_realization_image_hdus(
    FitsEntry &fits_entry, MapBuffer &mb, Eigen::Index i,
    const std::string &map_name, const std::string &stokes_suffix,
    const Wcs &wcs, double source_epoch, double median_rms) {
    for (Eigen::Index n = 0; n < mb->n_noise; ++n) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
            noise_matrix(
                mb->noise[i].data() + n * mb->n_rows * mb->n_cols,
                mb->n_rows, mb->n_cols);

        add_map_hdu_with_wcs(
            fits_entry, noise_signal_map_hdu_name(map_name, n, stokes_suffix),
            noise_matrix, wcs, source_epoch);
        add_noise_image_summary_keys(
            *fits_entry.hdus.back(), mb->sig_unit, median_rms);
        add_noise_product_package_join(
            *fits_entry.hdus.back(), noise_realization_product_id,
            "realization_map_index_" + std::to_string(n),
            "conditional_design_member",
            "source_imprinted_current_not_physical_noise_repeat");
    }
}

}  // namespace citlali::pipeline

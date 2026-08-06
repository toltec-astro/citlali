#pragma once

#include <exception>

// Beammap implementation detail. Include only after Beammap is declared.

namespace citlali::engine_detail::beammap {

inline DirectionalProduct capture_product_state(
    const Beammap &beammap,
    citlali::config::BeammapDirectionMode mode) {
    DirectionalProduct state;
    state.mode = mode;
    state.calib = beammap.calib;
    state.params = beammap.params;
    state.perrors = beammap.perrors;
    state.p0 = beammap.p0;
    state.perror0 = beammap.perror0;
    state.converged = beammap.converged;
    state.converge_iter = beammap.converge_iter;
    state.good_fits = beammap.good_fits;
    state.flag2 = beammap.flag2;
    state.fit_diag_init_params = beammap.fit_diag_init_params;
    state.fit_diag_lower_limits = beammap.fit_diag_lower_limits;
    state.fit_diag_upper_limits = beammap.fit_diag_upper_limits;
    state.fit_diag_hit_lower = beammap.fit_diag_hit_lower;
    state.fit_diag_hit_upper = beammap.fit_diag_hit_upper;
    state.fit_diag_bound_code = beammap.fit_diag_bound_code;
    state.fit_diag_bound_nhit = beammap.fit_diag_bound_nhit;
    state.prior_diag_values = beammap.prior_diag_values;
    state.final_prior_d2_diag = beammap.final_prior_d2_diag;
    state.final_prior_slot_index_diag =
        beammap.final_prior_slot_index_diag;
    state.reference_detector = beammap.beammap_reference_det_found;
    state.priors_centered = beammap.beammap_soft_priors_are_centered;
    state.priors_derotated = beammap.beammap_soft_priors_are_derotated;
    state.prior_center_x_arcsec =
        beammap.beammap_prior_array_center_x_arcsec;
    state.prior_center_y_arcsec =
        beammap.beammap_prior_array_center_y_arcsec;
    state.prior_alignment = beammap.beammap_prior_array_alignment;
    state.source_flux_mjy_beam = beammap.source_flux_mJy_beam;
    state.source_flux_mjy_sr = beammap.source_flux_MJy_Sr;
    return state;
}

inline void restore_product_state(
    Beammap &beammap, const DirectionalProduct &state) {
    beammap.calib = state.calib;
    beammap.params = state.params;
    beammap.perrors = state.perrors;
    beammap.p0 = state.p0;
    beammap.perror0 = state.perror0;
    beammap.converged = state.converged;
    beammap.converge_iter = state.converge_iter;
    beammap.good_fits = state.good_fits;
    beammap.flag2 = state.flag2;
    beammap.fit_diag_init_params = state.fit_diag_init_params;
    beammap.fit_diag_lower_limits = state.fit_diag_lower_limits;
    beammap.fit_diag_upper_limits = state.fit_diag_upper_limits;
    beammap.fit_diag_hit_lower = state.fit_diag_hit_lower;
    beammap.fit_diag_hit_upper = state.fit_diag_hit_upper;
    beammap.fit_diag_bound_code = state.fit_diag_bound_code;
    beammap.fit_diag_bound_nhit = state.fit_diag_bound_nhit;
    beammap.prior_diag_values = state.prior_diag_values;
    beammap.final_prior_d2_diag = state.final_prior_d2_diag;
    beammap.final_prior_slot_index_diag =
        state.final_prior_slot_index_diag;
    beammap.beammap_reference_det_found = state.reference_detector;
    beammap.beammap_soft_priors_are_centered = state.priors_centered;
    beammap.beammap_soft_priors_are_derotated = state.priors_derotated;
    beammap.beammap_prior_array_center_x_arcsec =
        state.prior_center_x_arcsec;
    beammap.beammap_prior_array_center_y_arcsec =
        state.prior_center_y_arcsec;
    beammap.beammap_prior_array_alignment = state.prior_alignment;
    beammap.source_flux_mJy_beam = state.source_flux_mjy_beam;
    beammap.source_flux_MJy_Sr = state.source_flux_mjy_sr;
}

class ProductStateTransaction {
public:
    explicit ProductStateTransaction(Beammap &beammap)
        : beammap_{beammap}, saved_{capture_product_state(
                                  beammap,
                                  citlali::config::BeammapDirectionMode::standard)} {}

    ProductStateTransaction(const ProductStateTransaction &) = delete;
    ProductStateTransaction &operator=(const ProductStateTransaction &) = delete;

    ~ProductStateTransaction() noexcept {
        if (!restored_) {
            try {
                restore();
            }
            catch (...) {
                std::terminate();
            }
        }
    }

    const DirectionalProduct &saved() const noexcept { return saved_; }

    void restore() {
        if (!restored_) {
            restore_product_state(beammap_, saved_);
            restored_ = true;
        }
    }

private:
    Beammap &beammap_;
    DirectionalProduct saved_;
    bool restored_ = false;
};

class ObservationMapBufferTransaction {
public:
    ObservationMapBufferTransaction(
        mapmaking::MapBuffer &standard,
        mapmaking::MapBuffer &directional)
        : standard_{standard}, directional_{directional} {
        std::swap(standard_, directional_);
    }

    ObservationMapBufferTransaction(
        const ObservationMapBufferTransaction &) = delete;
    ObservationMapBufferTransaction &operator=(
        const ObservationMapBufferTransaction &) = delete;

    ~ObservationMapBufferTransaction() noexcept {
        std::swap(standard_, directional_);
    }

private:
    mapmaking::MapBuffer &standard_;
    mapmaking::MapBuffer &directional_;
};

}  // namespace citlali::engine_detail::beammap

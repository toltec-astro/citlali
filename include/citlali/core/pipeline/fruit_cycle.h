#pragma once
#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <citlali/core/mapmaking/forward_map.h>
#include <memory>
#include <string>
#include <vector>

namespace citlali::pipeline {
// Owner-approved POINT empirical direct route, 2026-09-23. Coordinates are
// FK5/J2000 tangent degrees; brightness is mJy/nominal-beam. No beam/filter or
// upstream-response correction. Grid centers are FITS 1-based. Outside the
// AST-centered 40-arcsec pixel disk is explicitly no feedback, not zero sky.
enum class FruitModelSupport : std::uint8_t { unavailable=0, outside=1, excluded=2, admitted=3 };
struct FruitOperand { double value; FruitModelSupport support; };
class FruitPointModel {
public:
    FruitPointModel(std::string identity, mapmaking::ForwardMapGrid grid,
        double crpix1,double crpix2,double cdelt1,double cdelt2,
        std::vector<double> brightness,std::vector<std::uint8_t> valid);
    FruitOperand at(double tangent_lon_deg,double tangent_lat_deg) const;
    const std::string &identity() const {return identity_;}
private:
    std::string identity_;mapmaking::ForwardMapGrid grid_;
    double cx_,cy_,dx_,dy_;std::vector<double> values_;std::vector<std::uint8_t> valid_;
};
// Bounded response experiment: add a signed projected brightness operand only
// on the immutable parent's eligible entries. Invalid entries remain bitwise
// untouched; unavailable projection on an eligible entry is an error.
PtcMatrix fruit_injected_parent(const PtcMatrix &original, const PtcMask &eligible,
    const PtcMatrix &injection, const PtcMask &projection_supported);
// Verify a frozen intervention against its current native projection, then use
// the saved parent verbatim. Same axes/units/support as fruit_injected_parent;
// numerical operand equality and bytewise parent equality include invalid data.
PtcMatrix fruit_verified_injected_parent(const PtcMatrix &original, const PtcMask &eligible,
    const PtcMatrix &injection, const PtcMask &projection_supported,
    const PtcMatrix &saved_injection, const PtcMatrix &saved_parent);
// Immutable residual owns the exact projected operand and input eligibility.
// Matrix axes: scheduled RTC slots x admitted detector columns, row-major.
// Missing model support excludes the affected input; never fills a sky value.
class FruitResidual {
public:
    static std::shared_ptr<const FruitResidual> subtract(std::string identity,
        const PtcMatrix &original,const PtcMask &eligible,PtcMatrix operand,PtcMask model_supported);
    const auto &identity() const {return identity_;}
    const auto &values() const {return values_;}
    const auto &operand() const {return operand_;}
    const auto &eligible() const {return eligible_;}
private:
    std::string identity_;PtcMatrix values_,operand_;PtcMask eligible_;
};
// Learned state is bound by ownership to exactly one current residual. The
// optional initializer never replaces current-residual learning or centering.
class FruitLearned {
public:
    static FruitLearned learn(std::shared_ptr<const FruitResidual>,PtcSolverRequest,
        const PtcWarmStart *initializer=nullptr,const std::string &expected_scope="");
    struct Result { PtcApplied cleaned; PtcMatrix total; };
    Result apply(const std::shared_ptr<const FruitResidual> &exact) const;
    const auto &prepared() const {return prepared_;}
    const auto &fit() const {return fit_;}
private:
    std::shared_ptr<const FruitResidual> residual_;PtcPrepared prepared_;PtcFit fit_;
};
// Owner-authorized POINT152389 offline control only. Imported basis provenance
// is explicit; no learned-state lineage or convergence claim is manufactured.
// Means and masked application coefficients are freshly computed for this exact
// residual. Values/causes retain the original PTC rules; total rejoins its operand.
struct FruitReferenceBasisResult {
    PtcPrepared prepared;
    PtcApplied cleaned;
    PtcMatrix total;
    std::string basis_identity;
};
FruitReferenceBasisResult fruit_apply_reference_basis(
    const std::shared_ptr<const FruitResidual> &,PtcSolverRequest,
    const Eigen::MatrixXd &basis,std::string authenticated_basis_identity);
} // namespace citlali::pipeline

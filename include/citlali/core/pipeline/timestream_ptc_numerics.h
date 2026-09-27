#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

namespace citlali::pipeline {
using PtcMatrix = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using PtcMask = Eigen::Array<std::uint8_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
enum class PtcMethod { observed_als, pairwise_covariance, masked_covariance_count };
struct PtcSolverRequest {
    PtcMethod method=PtcMethod::observed_als;
    int rank=0, iteration_limit=100;
    double relative_objective_tolerance=1e-5, relative_rank_tolerance=1e-10;
};
struct PtcPattern { std::vector<Eigen::Index> entries, occurrences; };
// Prepared once per exact group/segment. Zero storage outside the mask is an
// arithmetic sentinel, never an observed value or a contribution to the fit.
struct PtcPrepared {
    PtcMatrix centered;
    PtcMask eligible;
    Eigen::VectorXd mean;
    std::vector<PtcPattern> time_patterns, detector_patterns;
    std::size_t eligible_count=0, complete_times=0;
    double preparation_seconds=0;
    static PtcPrepared prepare(const PtcMatrix &values,const PtcMask &eligible);
};
struct PtcFit {
    PtcSolverRequest request;
    Eigen::MatrixXd basis;
    bool converged=false;
    std::string stopping_reason, initialization="pairwise-covariance-eigenspace-v1";
    std::vector<double> objective;
    // Experimental count-normalized pair-overlap covariance; signed, unrepaired.
    std::vector<double> signed_spectrum;
    std::vector<std::size_t> overlap_counts, overlap_frequencies;
    double eigen_relative_residual=0, eigen_residual_tolerance=0,
        negative_to_positive_power=0, most_negative_to_largest_positive=0;
    int iterations=0;
    double fit_seconds=0, initialization_seconds=0, covariance_seconds=0,
        decomposition_seconds=0, coefficient_seconds=0, basis_seconds=0, check_seconds=0;
    std::size_t coefficient_factorizations=0, coefficient_factor_reuses=0;
};
struct PtcApplied {
    PtcMatrix values;
    // 0 retained; 1 excluded input; 2 failed fit; 4 deficient application;
    // 8 nonfinite application; 16 no eligible segment input (pipeline publication).
    // Causes are operation-local, not CAL facts.
    PtcMask causes;
    std::size_t retained=0, failed_times=0, factorizations=0, factor_reuses=0;
    double seconds=0;
};
// An initializer only. Current means, coefficients, objective and stopping are
// re-evaluated. Scope binds trajectory/branch/method/rank/network/physical segment
// and detector ordering; exact binary eligibility is checked independently.
struct PtcWarmStart {
    Eigen::MatrixXd basis;
    PtcMask eligible;
    PtcSolverRequest request;
    std::string scope;
    static PtcWarmStart from_converged(const PtcFit &,const PtcMask &,std::string);
};
PtcFit ptc_learn(const PtcPrepared &,PtcSolverRequest,
    const PtcWarmStart *initializer=nullptr,const std::string &expected_scope="");
PtcApplied ptc_apply(const PtcPrepared &,const PtcFit &);
// Apply the same frozen local linear operator to an explicit CAL-grid response.
// No subtraction of the signal mean, no response-derived admission or relearning.
PtcApplied ptc_response(const PtcPrepared &,const PtcFit &,const PtcMatrix &response);
} // namespace citlali::pipeline

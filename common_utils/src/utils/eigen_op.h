#pragma once

#include <Eigen/Core>
#include "meta.h"


namespace eigen_op {

template<typename Derived, typename ... CArgs>
struct UnaryMatrixOp {
    using Context =     std::tuple<CArgs...>;
    const Context context{};
    UnaryMatrixOp() = default ;
    template<typename ... Args>
    UnaryMatrixOp(std::piecewise_construct_t, Args&&... args): context{FWD(args)...} {}

    template<typename MatA, typename MatB, typename ...Args>
    void run(const Eigen::DenseBase<MatA>& in,
             Eigen::DenseBase<MatB> const& out_,
             Args ... args
             ) {
         auto& out = const_cast<Eigen::DenseBase<MatB>&>(out_).derived();
         out = in.derived();
         run(out, FWD(args)...);
    }
    template<typename Mat, typename ... Args>
    void run(const Eigen::DenseBase<Mat>& data, Args...args) {
        auto& op = static_cast<Derived&>(*this);
        op.run_inplace(
                const_cast<Eigen::DenseBase<Mat>&>(data).derived(),
                   FWD(args)...);
    }

};

}  // namespace proc_utils
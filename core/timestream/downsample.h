#pragma once

#include <Eigen/Dense>

namespace timestream {

template<typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD, typename DerivedE>
void downsample(Eigen::DenseBase<DerivedA> &in_scans, Eigen::DenseBase<DerivedB> &in_flags, Eigen::DenseBase<DerivedC> &in_scanindex,
                Eigen::DenseBase<DerivedD> &out_scans, Eigen::DenseBase<DerivedE> &out_flags, Eigen::DenseBase<DerivedC> &out_scanindex, const int dsf){

    out_scans = Eigen::Map<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> > (in_scans.derived().data(), (in_scans.rows()+(dsf - 1))/dsf, in_scans.cols(),Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(in_scans.outerStride(),in_scans.innerStride()*dsf));
    out_flags = Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>,0,Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> > (in_flags.derived().data(), (in_flags.rows()+(dsf - 1))/dsf, in_flags.cols(),Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(in_flags.outerStride(),in_flags.innerStride()*dsf));

    //out_scanindex = (in_scanindex.derived().array()/dsf).floor();

    SPDLOG_INFO("OUT_SCANS {}",out_scans.derived());
    SPDLOG_INFO("in_scans {}",in_scans.derived());

}
} //namespace

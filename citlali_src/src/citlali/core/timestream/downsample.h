#pragma once

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::Stride;
using Eigen::InnerStride;
using Eigen::Dynamic;

namespace timestream {

auto downsample(TCData<LaliDataKind::RTC,MatrixXd>&in, TCData<LaliDataKind::PTC,MatrixXd>&out, const int dsf){

    auto rows = in.scans.data.rows();
    auto cols = in.scans.data.cols();

    out.scans.data = Map<MatrixXd,0,Stride<Dynamic,Dynamic>> (in.scans.data.data(), (rows+(dsf - 1))/dsf, cols, Stride<Dynamic,Dynamic>(in.scans.data.outerStride(),in.scans.data.innerStride()*dsf));
    out.flags.data = Map<Matrix<bool,Dynamic,Dynamic>,0,Stride<Dynamic,Dynamic>> (in.flags.data.data(), (rows+(dsf - 1))/dsf, cols,Stride<Dynamic,Dynamic>(in.flags.data.outerStride(),in.flags.data.innerStride()*dsf));

    out.telLat.data = Map<VectorXd,0,InnerStride<Dynamic>> (in.telLat.data.data(),(rows+(dsf - 1))/dsf,InnerStride<Dynamic>(in.telLat.data.innerStride()*dsf));
    out.telLon.data = Map<VectorXd,0,InnerStride<Dynamic>> (in.telLon.data.data(),(rows+(dsf - 1))/dsf,InnerStride<Dynamic>(in.telLon.data.innerStride()*dsf));
    out.telElDes.data = Map<VectorXd,0,InnerStride<Dynamic>> (in.telElDes.data.data(),(rows+(dsf - 1))/dsf,InnerStride<Dynamic>(in.telElDes.data.innerStride()*dsf));
    out.ParAng.data = Map<VectorXd,0,InnerStride<Dynamic>> (in.ParAng.data.data(),(rows+(dsf - 1))/dsf,InnerStride<Dynamic>(in.ParAng.data.innerStride()*dsf));

}
} // namespace timestream

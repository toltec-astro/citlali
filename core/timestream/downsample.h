#pragma once

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::Stride;
using Eigen::InnerStride;
using Eigen::Dynamic;

namespace timestream {

void downsample(TCData<LaliDataKind::RTC,MatrixXd>&in, TCData<LaliDataKind::PTC,MatrixXd>&out, const int dsf){


    out.scans.data = Map<MatrixXd,0,Stride<Dynamic,Dynamic> > (in.scans.data.data(), (in.scans.data.rows()+(dsf - 1))/dsf, in.scans.data.cols(),Stride<Dynamic,Dynamic>(in.scans.data.outerStride(),in.scans.data.innerStride()*dsf));
    out.flags.data = Map<Matrix<bool,Dynamic,Dynamic>,0,Stride<Dynamic,Dynamic> > (in.flags.data.data(), (in.flags.data.rows()+(dsf - 1))/dsf, in.flags.data.cols(),Stride<Dynamic,Dynamic>(in.flags.data.outerStride(),in.flags.data.innerStride()*dsf));

    out.telLat.data = Map<VectorXd,0,InnerStride<Dynamic>> (in.telLat.data.data(),(in.telLat.data.size()+(dsf - 1))/dsf,InnerStride<Dynamic>(in.telLat.data.innerStride()*dsf));
    out.telLon.data = Map<VectorXd,0,InnerStride<Dynamic>> (in.telLon.data.data(),(in.telLon.data.size()+(dsf - 1))/dsf,InnerStride<Dynamic>(in.telLon.data.innerStride()*dsf));
    out.telElDes.data = Map<VectorXd,0,InnerStride<Dynamic>> (in.telElDes.data.data(),(in.telElDes.data.size()+(dsf - 1))/dsf,InnerStride<Dynamic>(in.telElDes.data.innerStride()*dsf));
    out.ParAng.data = Map<VectorXd,0,InnerStride<Dynamic>> (in.ParAng.data.data(),(in.ParAng.data.size()+(dsf - 1))/dsf,InnerStride<Dynamic>(in.ParAng.data.innerStride()*dsf));

}


} //namespace

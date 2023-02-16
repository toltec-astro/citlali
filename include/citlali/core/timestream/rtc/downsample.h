#pragma once

namespace timestream {

class Downsampler {
public:
    double factor;

    template <typename DerivedA, typename DerivedB>
    void downsample(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &out) {
        // define to save space
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        using Eigen::InnerStride;
        using Eigen::Dynamic;

        // use eigen stride to skip over points
        using EigenStrideMap = Map<Matrix<typename DerivedA::Scalar, Dynamic, Dynamic>,0, Stride<Dynamic,Dynamic>>;

        // saving space
        auto rows = in.rows();
        auto cols = in.cols();

        out = EigenStrideMap (in.derived().data(), (rows+(factor-1))/factor, cols, Stride<Dynamic,
                                                                                         Dynamic>(in.outerStride(),in.innerStride()*factor));
    }
};

} // namespace timestream

#pragma once

namespace timestream {

class Downsampler {
public:
    int factor;
    double downsampled_freq_Hz;

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

    template <typename DerivedA, typename DerivedB>
    void downsample_flags(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &out) {
        auto rows = in.rows();
        auto cols = in.cols();
        auto out_rows = (rows + (factor - 1)) / factor;

        out.derived().resize(out_rows, cols);
        out.derived().setZero();

        for (Eigen::Index r = 0; r < out_rows; ++r) {
            Eigen::Index start = r * factor;
            Eigen::Index end = std::min<Eigen::Index>(start + factor, rows);
            for (Eigen::Index c = 0; c < cols; ++c) {
                bool any_flag = false;
                for (Eigen::Index rr = start; rr < end; ++rr) {
                    if (in(rr, c)) {
                        any_flag = true;
                        break;
                    }
                }
                out(r, c) = any_flag;
            }
        }
    }
};

} // namespace timestream

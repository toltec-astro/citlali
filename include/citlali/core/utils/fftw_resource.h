#pragma once

#include <fftw3.h>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace citlali::utils {

class FftwComplexBuffer {
public:
    using Allocate = void *(*)(std::size_t);
    using Release = void (*)(void *);
    FftwComplexBuffer(int rows, int columns, Allocate allocate = fftw_malloc,
                      Release release = fftw_free) : release_(release) {
        if (rows <= 0 || columns <= 0 ||
            static_cast<std::size_t>(rows) >
                std::numeric_limits<std::size_t>::max() / sizeof(fftw_complex) /
                    static_cast<std::size_t>(columns)) {
            throw std::invalid_argument("invalid FFTW buffer dimensions");
        }
        value_ = static_cast<fftw_complex *>(allocate(
            sizeof(fftw_complex) * static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(columns)));
        if (!value_) throw std::runtime_error("FFTW buffer allocation failed");
    }
    ~FftwComplexBuffer() noexcept { release_(value_); }
    FftwComplexBuffer(const FftwComplexBuffer &) = delete;
    FftwComplexBuffer &operator=(const FftwComplexBuffer &) = delete;
    FftwComplexBuffer(FftwComplexBuffer &&) = delete;
    FftwComplexBuffer &operator=(FftwComplexBuffer &&) = delete;
    fftw_complex *get() const noexcept { return value_; }
private:
    fftw_complex *value_ = nullptr;
    Release release_;
};

class FftwPlan {
public:
    using Create = fftw_plan (*)(int, int, fftw_complex *, fftw_complex *, int, unsigned);
    using Destroy = void (*)(fftw_plan);
    FftwPlan(int rows, int columns, fftw_complex *input, fftw_complex *output,
             int direction, unsigned flags, Create create = fftw_plan_dft_2d,
             Destroy destroy = fftw_destroy_plan) : destroy_(destroy) {
        if (!input || !output) throw std::invalid_argument("FFTW plan needs owned buffers");
        value_ = create(rows, columns, input, output, direction, flags);
        if (!value_) throw std::runtime_error("FFTW plan creation failed");
    }
    ~FftwPlan() noexcept { destroy_(value_); }
    FftwPlan(const FftwPlan &) = delete;
    FftwPlan &operator=(const FftwPlan &) = delete;
    FftwPlan(FftwPlan &&) = delete;
    FftwPlan &operator=(FftwPlan &&) = delete;
    fftw_plan get() const noexcept { return value_; }
private:
    fftw_plan value_ = nullptr;
    Destroy destroy_;
};

}  // namespace citlali::utils

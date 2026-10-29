#pragma once

#include <fitsio.h>
#include <stdexcept>
#include <utility>

namespace citlali::utils {

// CFITSIO ownership only; callers retain their data/header validation policy.
// Explicit close returns its independent status so required writes can fail.
class FitsHandle {
public:
    using Close = int (*)(fitsfile *, int *);
    explicit FitsHandle(Close close = fits_close_file) noexcept : close_(close) {}
    ~FitsHandle() noexcept { (void)close(); }
    FitsHandle(const FitsHandle &) = delete;
    FitsHandle &operator=(const FitsHandle &) = delete;
    FitsHandle(FitsHandle &&) = delete;
    FitsHandle &operator=(FitsHandle &&) = delete;

    fitsfile *get() const noexcept { return value_; }
    fitsfile **address() {
        if (value_) throw std::logic_error("FITS handle is already open");
        return &value_;
    }
    int close() noexcept {
        int status = 0;
        if (auto *value = std::exchange(value_, nullptr)) close_(value, &status);
        return status;
    }

private:
    fitsfile *value_ = nullptr;
    Close close_;
};

}  // namespace citlali::utils

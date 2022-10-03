#pragma once

#include <netcdf>

struct DataIOError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

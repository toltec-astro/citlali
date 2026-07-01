#pragma once

#include <functional>

namespace citlali::pipeline::detail {

template <class T>
const T &unwrap_reference_wrapper(const T &value) {
    return value;
}

template <class T>
const T &unwrap_reference_wrapper(const std::reference_wrapper<T> &value) {
    return value.get();
}

template <class T>
const T &unwrap_reference_wrapper(
    const std::reference_wrapper<const T> &value) {
    return value.get();
}

}  // namespace citlali::pipeline::detail

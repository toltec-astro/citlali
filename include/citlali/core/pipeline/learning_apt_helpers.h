#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include <Eigen/Core>

#include <citlali/core/timestream/timestream.h>

namespace citlali::pipeline {

template <class Apt>
Eigen::Index learning_find_det_by_uid(const Apt &apt, int uid) {
    if (uid == timestream::kTransientFillInt || uid < 0) {
        return -1;
    }
    const auto uid_it = apt.find("uid");
    if (uid_it == apt.end()) {
        return static_cast<Eigen::Index>(uid);
    }
    for (Eigen::Index i = 0; i < uid_it->second.size(); ++i) {
        if (std::isfinite(uid_it->second(i)) &&
            static_cast<int>(std::llround(uid_it->second(i))) == uid) {
            return i;
        }
    }
    return -1;
}

template <class Apt>
int learning_apt_int(const Apt &apt, const std::string &key,
                     Eigen::Index det, int fallback) {
    const auto it = apt.find(key);
    if (it == apt.end() || det < 0 || det >= it->second.size() ||
        !std::isfinite(it->second(det))) {
        return fallback;
    }
    return static_cast<int>(std::llround(it->second(det)));
}

template <class Apt>
int learning_array_for_nw(const Apt &apt, int nw, int fallback) {
    const auto nw_it = apt.find("nw");
    const auto array_it = apt.find("array");
    if (nw_it == apt.end() || array_it == apt.end()) {
        return fallback;
    }
    const Eigen::Index n =
        std::min<Eigen::Index>(nw_it->second.size(), array_it->second.size());
    for (Eigen::Index det = 0; det < n; ++det) {
        if (!std::isfinite(nw_it->second(det)) ||
            !std::isfinite(array_it->second(det))) {
            continue;
        }
        if (static_cast<int>(std::llround(nw_it->second(det))) == nw) {
            return static_cast<int>(std::llround(array_it->second(det)));
        }
    }
    return fallback;
}

}  // namespace citlali::pipeline

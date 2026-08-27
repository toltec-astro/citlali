#pragma once

#include <kids/core/kidsdata.h>

#if defined(CITLALI_KIDSCPP_V3)
#include <kids/toltec/timestream.h>
#else
#include <kids/toltec/toltec.h>
#endif

#include <tula/container.h>

#include <filesystem>
#include <string_view>

namespace citlali::compat::kidscpp {

#if defined(CITLALI_KIDSCPP_V3)
using RawTimeStream = kids::toltec::RawTimeStream;
using RawTimeStreamMeta = kids::toltec::RawTimeStreamMeta;

inline constexpr std::string_view data_spec =
    kids::toltec::raw_timestream_spec;

inline auto get_raw_timestream_meta(const std::filesystem::path &source)
    -> RawTimeStreamMeta {
    return kids::toltec::get_raw_timestream_meta(source);
}

inline auto read_raw_timestream_slice(
    const std::filesystem::path &source,
    const tula::container_utils::Slice<int> &slice) -> RawTimeStream {
    return kids::toltec::read_raw_timestream_slice(
        source,
        kids::toltec::SampleSlice{
            std::get<0>(slice), std::get<1>(slice), std::get<2>(slice)});
}
#else
using RawTimeStream = kids::KidsData<kids::KidsDataKind::RawTimeStream>;
using RawTimeStreamMeta = kids::KidsData<>::meta_t;

inline constexpr std::string_view data_spec = kids::toltec::name;

inline auto get_raw_timestream_meta(const std::filesystem::path &source)
    -> RawTimeStreamMeta {
    auto [kind, meta] = kids::toltec::get_meta<>(source);
    static_cast<void>(kind);
    return meta;
}

inline auto read_raw_timestream_slice(
    const std::filesystem::path &source,
    const tula::container_utils::Slice<int> &slice) -> RawTimeStream {
    return kids::toltec::read_data_slice<kids::KidsDataKind::RawTimeStream>(
        source, slice);
}
#endif

} // namespace citlali::compat::kidscpp

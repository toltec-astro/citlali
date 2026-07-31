#include <citlali/core/compat/kidscpp_raw_timestream.h>

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>

TEST(KidscppRawTimestreamCompat, ExposesTheSelectedDataSpecification) {
    using citlali::compat::kidscpp::data_spec;

    EXPECT_FALSE(data_spec.empty());
#if defined(CITLALI_KIDSCPP_V3)
    EXPECT_EQ(data_spec, std::string_view{kids::toltec::raw_timestream_spec});
    static_assert(std::is_same_v<
                  citlali::compat::kidscpp::RawTimeStream,
                  kids::toltec::RawTimeStream>);
    static_assert(std::is_same_v<
                  citlali::compat::kidscpp::RawTimeStreamMeta,
                  kids::toltec::RawTimeStreamMeta>);
#else
    EXPECT_EQ(data_spec, std::string_view{kids::toltec::name});
    static_assert(std::is_same_v<
                  citlali::compat::kidscpp::RawTimeStream,
                  kids::KidsData<kids::KidsDataKind::RawTimeStream>>);
#endif
}

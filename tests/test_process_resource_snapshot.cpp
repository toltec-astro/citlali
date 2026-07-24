#include <citlali/core/pipeline/cli_summary.h>
#include <citlali/core/utils/process_resource_snapshot.h>

#include <gtest/gtest.h>

TEST(ProcessResourceSnapshot, ParsesLinuxStatusValues) {
    EXPECT_EQ(citlali::utils::parse_proc_status_value(
                  "VmRSS:\t   12345 kB", "VmRSS:"),
              12345);
    EXPECT_EQ(citlali::utils::parse_proc_status_value(
                  "Threads:\t6", "Threads:"),
              6);
    EXPECT_EQ(citlali::utils::parse_proc_status_value(
                  "VmSize:\tinvalid", "VmSize:"),
              -1);
    EXPECT_EQ(citlali::utils::parse_proc_status_value(
                  "VmHWM:\t123 kB", "VmRSS:"),
              -1);
}

TEST(ProcessResourceSnapshot, ConvertsKibibytesToGibibytes) {
    EXPECT_DOUBLE_EQ(citlali::pipeline::physical_memory_gb(1024LL * 1024LL),
                     1.0);
}

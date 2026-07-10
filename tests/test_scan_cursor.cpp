#include <citlali/core/pipeline/timestream_scan_generation.h>

#include <gtest/gtest.h>

TEST(scan_cursor, enumerates_each_scan_once) {
    citlali::pipeline::ScanCursor cursor(3);

    EXPECT_EQ(cursor.next(), 0);
    EXPECT_EQ(cursor.next(), 1);
    EXPECT_EQ(cursor.next(), 2);
    EXPECT_EQ(cursor.next(), std::nullopt);
    EXPECT_EQ(cursor.next(), std::nullopt);
}

TEST(scan_cursor, new_run_starts_clean_after_interrupted_run) {
    citlali::pipeline::ScanCursor interrupted(3);
    EXPECT_EQ(interrupted.next(), 0);
    EXPECT_EQ(interrupted.next(), 1);

    citlali::pipeline::ScanCursor next_run(3);
    EXPECT_EQ(next_run.next(), 0);
    EXPECT_EQ(next_run.next(), 1);
    EXPECT_EQ(next_run.next(), 2);
    EXPECT_EQ(next_run.next(), std::nullopt);
}

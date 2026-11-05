#include <citlali/core/utils/parallel_work.h>
#include <gtest/gtest.h>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>

using citlali::utils::parallel_for;
using citlali::utils::ordered_parallel_batches;

TEST(ParallelWork, BudgetOverlapAndEveryIndexExactlyOnce) {
    std::vector<int> visits(17);
    std::mutex mutex;std::condition_variable ready;int arrived=0;
    const auto report=parallel_for(visits.size(),{3},[&](std::size_t i) {
        if(i<3) {
            std::unique_lock lock(mutex);++arrived;ready.notify_all();
            ready.wait_for(lock,std::chrono::seconds(5),[&]{return arrived==3;});
        }
        ++visits[i];
    });
    EXPECT_EQ(report.jobs,17);EXPECT_EQ(report.started_workers,3);
    EXPECT_EQ(report.peak_active_jobs,3);
    for(auto count:visits)EXPECT_EQ(count,1);
    const auto capped=parallel_for(2,{12},[](auto){});
    EXPECT_EQ(capped.started_workers,2);EXPECT_LE(capped.peak_active_jobs,2);
    EXPECT_EQ(parallel_for(0,{3},[](auto){}).started_workers,0);
    EXPECT_THROW(parallel_for(1,{0},[](auto){}),std::invalid_argument);
}

TEST(ParallelWork, FailuresJoinAllJobsAndRethrowFirstIndex) {
    std::atomic<int> finished{0};
    try {
        parallel_for(9,{3},[&](std::size_t i) {
            ++finished;
            if(i==1 || i==4)throw std::runtime_error(std::to_string(i));
        });
        FAIL()<<"worker failure was swallowed";
    } catch(const std::runtime_error &e) {EXPECT_EQ(std::string(e.what()),"1");}
    EXPECT_EQ(finished,9);
}

TEST(ParallelWork, BoundedMoveOnlyResultsPreserveFloatingPointOrder) {
    struct Result {
        double value;std::atomic<int> &live;
        Result(double v,std::atomic<int> &l):value(v),live(l){++live;}
        ~Result(){--live;}
    };
    const std::vector<double> values{1.e16,1.,-1.e16,3.,1.e16,-1.e16,2.};
    std::atomic<int> live{0},peak{0};std::vector<std::size_t> order;double sum=0;
    const auto report=ordered_parallel_batches(values.size(),{3},[&](std::size_t i) {
        auto result=std::make_unique<Result>(values[i],live);
        auto previous=peak.load();const auto count=live.load();
        while(previous<count && !peak.compare_exchange_weak(previous,count)){}
        return result;
    },[&](std::size_t i,std::unique_ptr<Result> result) {order.push_back(i);sum+=result->value;});
    EXPECT_EQ(sum,std::accumulate(values.begin(),values.end(),0.));
    EXPECT_EQ(order,(std::vector<std::size_t>{0,1,2,3,4,5,6}));
    EXPECT_EQ(live,0);EXPECT_LE(peak,3);EXPECT_EQ(report.jobs,values.size());
}

TEST(ParallelWork, FailedBatchCannotReachSerialConsumer) {
    std::vector<int> consumed;
    EXPECT_THROW(ordered_parallel_batches(8,{3},[](std::size_t i) {
        if(i==1)throw std::runtime_error("failed segment");return int(i);
    },[&](auto,int value){consumed.push_back(value);}),std::runtime_error);
    EXPECT_TRUE(consumed.empty());
}

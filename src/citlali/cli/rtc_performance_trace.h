// Application-owned, bounded measurement of this connected route. No sample
// identities or scientific policy; one line per stage/allocation category.
#include <sys/resource.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <malloc/malloc.h>
#endif
namespace {
class RtcPerformanceTrace {
public:
    explicit RtcPerformanceTrace(const fs::path &directory) {
        fs::create_directories(directory);out_.open(directory/"performance.jsonl");
        require(bool(out_),"performance evidence output failed");mark("start");
    }
    void mark(const char *stage) {
        struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
        out_<<std::setprecision(17)<<"{\"stage\":\""<<stage<<"\",\"seconds\":"
            <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count();
#if defined(__APPLE__)
        mach_task_basic_info_data_t info{};mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
        if(task_info(mach_task_self(),MACH_TASK_BASIC_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS)
            out_<<",\"rss_bytes\":"<<info.resident_size;
        malloc_statistics_t allocations{};malloc_zone_statistics(nullptr,&allocations);
        out_<<",\"allocator_live_bytes\":"<<allocations.size_in_use
            <<",\"allocator_live_blocks\":"<<allocations.blocks_in_use
            <<",\"peak_rss_bytes\":"<<usage.ru_maxrss;
#else
        out_<<",\"peak_rss_bytes\":"<<usage.ru_maxrss*1024;
#endif
        out_<<"}\n";out_.flush();require(bool(out_),"performance evidence output failed");
    }
    void array(const char *category,std::size_t count,std::size_t element_bytes,std::size_t allocations) {
        out_<<"{\"array\":\""<<category<<"\",\"elements\":"<<count<<",\"element_bytes\":"<<element_bytes
            <<",\"payload_bytes\":"<<count*element_bytes<<",\"payload_allocations\":"
            <<(allocations?std::to_string(allocations):"null")<<"}\n";
        out_.flush();require(bool(out_),"performance allocation evidence output failed");
    }
private:
    std::ofstream out_;
    std::chrono::steady_clock::time_point start_=std::chrono::steady_clock::now();
};
}

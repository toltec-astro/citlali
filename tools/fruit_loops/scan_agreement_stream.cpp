// EL-F13 isolated in-memory occurrence accumulator. No filesystem or Citlali calls.
// Wire input: little-endian int64[8], binary64[5], exactly 104 bytes.
// All maps are channel,row,column: N,C,Q,absN,absC,absQ,terms,unique,sum(f).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void require(bool good, const char* why) { if (!good) throw std::runtime_error(why); }
struct Map {
    size_t p; std::vector<double> data; std::vector<int64_t> last;
    explicit Map(size_t pixels): p(pixels), data(9*p), last(p,-1) {}
    void add(size_t i, int64_t uid, double n, double c, double q, double f) {
        if (f==0) return;
        n *= f; c *= f; q *= f*f;
        data[i]+=n; data[p+i]+=c; data[2*p+i]+=q;
        data[3*p+i]+=std::abs(n); data[4*p+i]+=std::abs(c); data[5*p+i]+=std::abs(q);
        data[6*p+i]+=1; data[8*p+i]+=f;
        if(last[i]!=uid) { data[7*p+i]+=1; last[i]=uid; }
    }
};
struct Kernel {int rows,cols; std::vector<double> k,q;};
struct Key {int64_t uid; int array,scan; std::array<std::unique_ptr<Map>,3> probes;
    std::unique_ptr<Map> target;};
struct State {
    int rows,cols,scan=-1; size_t p; uint64_t records=0,occurrences=0;
    int64_t previous_uid=-1,previous_sample=-1;
    std::set<int> seen_scans; std::vector<int> scan_order; std::set<std::pair<int,int64_t>> blocks;
    std::map<int64_t,int> uid_arrays;
    std::map<int64_t,std::vector<uint64_t>> uid_bits;
    std::array<std::vector<Kernel>,3> kernels;
    std::array<std::unique_ptr<Map>,36> scans,clean;
    std::array<std::unique_ptr<Map>,3> full;
    std::vector<Key> keys; std::array<std::set<int64_t>,3> excluded;
    State(int r,int c):rows(r),cols(c),p(size_t(r)*c) {
        require(r>0&&c>0&&p<=2000000,"invalid shape");
        for(auto &v:scans) v=std::make_unique<Map>(p);
        for(auto &v:full) v=std::make_unique<Map>(p);
    }
    void commit() {
        if(scan<0) return;
        for(int a=0;a<3;++a) for(size_t i=0;i<3*p;++i)
            full[a]->data[i]+=scans[scan*3+a]->data[i];
    }
    void feed(const char *bytes,size_t count) {
        require(count%104==0,"truncated record");
        for(size_t offset=0;offset<count;offset+=104) {
            std::array<int64_t,8> id; std::array<double,5> v;
            std::memcpy(id.data(),bytes+offset,64); std::memcpy(v.data(),bytes+offset+64,40);
            ++records;
            if(id[0]==0) {
                require(id[1]>=0&&id[1]<12,"invalid scan marker");
                require(seen_scans.insert(int(id[1])).second,"duplicate scan marker");
                scan_order.push_back(int(id[1]));
                for(int j=2;j<8;++j) require(id[j]==0,"nonzero marker identity");
                for(double x:v) require(x==0,"nonzero marker value");
                commit(); scan=int(id[1]); previous_uid=-1; previous_sample=-1; continue;
            }
            require(id[0]==1&&scan>=0&&id[1]==scan,"invalid sequence");
            require(id[2]>=0&&id[3]>=0&&id[3]<3&&id[4]>=0&&id[4]<rows&&
                    id[5]>=0&&id[5]<cols&&id[6]>=0&&id[7]>=0,"invalid identity");
            int a=int(id[3]); auto uid=id[2];
            require(size_t(id[6])<kernels[a].size(),"invalid kernel index");
            for(double x:v) require(std::isfinite(x),"nonfinite record");
            require(v[1]>0&&v[2]>0&&v[4]==1,"invalid ordinary coefficient");
            if(uid!=previous_uid) {
                require(blocks.emplace(scan,uid).second,"noncontiguous UID block");
                previous_uid=uid; previous_sample=-1;
            }
            require(id[7]>previous_sample,"nonmonotonic sample"); previous_sample=id[7];
            auto inserted=uid_arrays.emplace(uid,a);
            require(inserted.second||inserted.first->second==a,"UID changed array");
            auto bitit=uid_bits.find(uid);
            if(bitit==uid_bits.end()) bitit=uid_bits.emplace(uid,std::vector<uint64_t>((p+63)/64)).first;
            auto &bits=bitit->second;
            ++occurrences;
            const auto &ker=kernels[a][id[6]];
            int r0=int(id[4])-(ker.rows-1)/2,c0=int(id[5])-(ker.cols-1)/2;
            Map *ordinary=scans[scan*3+a].get(), *reference=clean[scan*3+a].get();
            if(excluded[a].count(uid)) reference=nullptr;
            std::vector<std::pair<Map*,double>> extras;
            for(auto &key:keys) if(key.array==a&&key.scan==scan) {
                bool target=key.uid==uid;
                for(int j=0;j<3;++j) extras.emplace_back(key.probes[j].get(),target?j*0.5:1.0);
                if(target) extras.emplace_back(key.target.get(),1.0);
            }
            auto &tot=full[a]->data;
            for(int r=std::max(0,r0);r<std::min(rows,r0+ker.rows);++r)
                for(int c=std::max(0,c0);c<std::min(cols,c0+ker.cols);++c) {
                    size_t i=size_t(r)*cols+c,j=size_t(r-r0)*ker.cols+c-c0;
                    double n=ker.k[j]*v[0],coeff=ker.k[j]*v[1],q=ker.q[j]*v[2];
                    ordinary->add(i,uid,n,coeff,q,1);
                    if(reference) reference->add(i,uid,n,coeff,q,1);
                    for(auto &extra:extras) extra.first->add(i,uid,n,coeff,q,extra.second);
                    tot[3*p+i]+=std::abs(n); tot[4*p+i]+=std::abs(coeff); tot[5*p+i]+=std::abs(q);
                    tot[6*p+i]+=1; tot[8*p+i]+=1;
                    uint64_t bit=uint64_t(1)<<(i%64);
                    if(!(bits[i/64]&bit)) {bits[i/64]|=bit;tot[7*p+i]+=1;}
                }
        }
    }
};
template<class F> int safe(F f) {try {f();return 0;} catch(const std::exception &e){error=e.what();return -1;}}
}
extern "C" {
const char* sa_error(){return error.c_str();}
int sa_scan_order(void*p,int i){return static_cast<State*>(p)->scan_order.at(i);}
void* sa_create(int r,int c) {try {uint16_t x=1;require(*reinterpret_cast<char*>(&x)==1,"requires little endian");return new State(r,c);}catch(const std::exception&e){error=e.what();return nullptr;}}
void sa_destroy(void *p){delete static_cast<State*>(p);}
int sa_kernel(void*p,int a,int k,int r,int c,const double*x,const double*q) {return safe([&]{
    auto&s=*static_cast<State*>(p);require(s.records==0&&a>=0&&a<3&&k==int(s.kernels[a].size())&&r>0&&c>0&&r%2&&c%2,"invalid kernel registration");
    Kernel v{r,c,{x,x+r*c},{q,q+r*c}};
    for(int i=0;i<r*c;++i)require(std::isfinite(x[i])&&std::isfinite(q[i])&&q[i]>=0&&q[i]==x[i]*x[i],"invalid squared bank");
    s.kernels[a].push_back(std::move(v));});}
int sa_key(void*p,int64_t uid,int a,int scan) {return safe([&]{
    auto&s=*static_cast<State*>(p);require(s.records==0&&s.keys.size()<16&&uid>=0&&a>=0&&a<3&&scan>=0&&scan<12,"invalid key");
    for(auto&k:s.keys)require(!(k.uid==uid&&k.array==a&&k.scan==scan),"duplicate key");
    if(s.excluded[a].empty())for(int j=0;j<12;++j)s.clean[j*3+a]=std::make_unique<Map>(s.p);
    s.excluded[a].insert(uid);Key k;k.uid=uid;k.array=a;k.scan=scan;
    for(auto&v:k.probes)v=std::make_unique<Map>(s.p);k.target=std::make_unique<Map>(s.p);s.keys.push_back(std::move(k));});}
int sa_feed(void*p,const char*bytes,size_t n){return safe([&]{static_cast<State*>(p)->feed(bytes,n);});}
int sa_finish(void*p,uint64_t records,uint64_t occurrences){return safe([&]{auto&s=*static_cast<State*>(p);
    require(s.records==records&&s.occurrences==occurrences&&s.seen_scans.size()==12,"record/scan closure failed");s.commit();});}
const double* sa_map(void*p,int kind,int index,int scan){try{auto&s=*static_cast<State*>(p);Map*m=nullptr;
    if(kind==0)m=s.scans.at(scan*3+index).get();
    else if(kind==1){m=s.clean.at(scan*3+index).get();if(!m)m=s.scans.at(scan*3+index).get();}
    else if(kind==2)m=s.full.at(index).get();
    else if(kind==3)m=s.keys.at(index).probes.at(scan).get();
    else if(kind==4)m=s.keys.at(index).target.get();
    require(m,"unknown map");return m->data.data();}catch(const std::exception&e){error=e.what();return nullptr;}}
}

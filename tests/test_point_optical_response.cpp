#include <citlali/core/pipeline/point_optical_response.h>
#include "../src/citlali/cli/point_optical_reports.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
using namespace citlali::pipeline;

namespace {
std::vector<PointOpticalNetwork> fixture() {
    std::vector<PointOpticalNetwork> out;
    for(int network=0;network<3;++network) {
        PointOpticalNetwork n;n.observation_identity="observation:fixture";n.cal_identity="cal:"+std::to_string(network);
        n.source_mask_identity="AST:source-free:"+std::to_string(network);n.array=0;n.network=network;
        constexpr int rows=1200,detectors=12;n.values.resize(rows,detectors);
        n.eligible=PtcMask::Ones(rows,detectors);n.source_free=PtcMask::Ones(rows,detectors);
        for(int d=0;d<detectors;++d)n.detector_ids.push_back("APT:fixture:N"+std::to_string(network)+":row="+std::to_string(d));
        for(int t=0;t<rows;++t) {
            n.times_unix_sec.push_back(1000.+.05*t);n.science_blocks.push_back(t/200);
            const double common=50*(std::sin(.016*t)+.6*std::cos(.043*t)+.2*std::sin(.003*t*t));
            for(int d=0;d<detectors;++d) {
                double gain=1+.01*d,noise=.03;
                if(network==0&&d==0)gain=.03;
                if(network==0&&d==1)noise=1.5;
                if(network==0&&d==5)gain=-1;
                if(network==0&&d==7)noise=.0005;
                n.values(t,d)=1000+5*d+gain*common+noise*(std::sin(.23*t+.71*d+network)+.3*std::cos(.009*t*t+.83*d+network));
            }
        }
        out.push_back(std::move(n));
    }
    return out;
}
PointOpticalRequest request(){PointOpticalRequest r;r.bootstrap_samples=400;return r;}

// Four blocks have within-block slope information; eight others identify the
// near-unit full response through their between-block offsets. Dropping the
// latter blocks would instead report a confidently weak response near 0.035.
std::vector<PointOpticalNetwork> sparse_block_fixture(bool constant_blocks=false) {
    std::vector<double> reference,target;std::vector<std::int64_t> blocks;
    const std::array<double,8> levels{-100,-70,70,100,-90,-60,60,90};
    std::size_t level=0;
    for(int block=0;block<12;++block) {
        if(block==0 || block==1 || block==6 || block==7) {
            for(int row=0;row<10;++row) {
                const double x=-1.+2.*row/9.;reference.push_back(x);
                target.push_back((.03+.01*(block%6))*x+.0001*std::sin(row));blocks.push_back(block);
            }
        } else {
            const double x=levels[level++];
            for(int row=0;row<(constant_blocks?4:1);++row) {
                reference.push_back(x);target.push_back(x);blocks.push_back(block);
            }
        }
    }
    std::vector<PointOpticalNetwork> networks;
    for(int network=0;network<3;++network) {
        PointOpticalNetwork n;n.observation_identity="observation:sparse-block-regression";
        n.cal_identity="cal:sparse:"+std::to_string(network);n.source_mask_identity="source-free:sparse:"+std::to_string(network);
        n.array=0;n.network=network;n.science_blocks=blocks;
        n.values.resize(reference.size(),12);n.eligible=PtcMask::Ones(n.values.rows(),12);n.source_free=n.eligible;
        for(int detector=0;detector<12;++detector)n.detector_ids.push_back("sparse:N"+std::to_string(network)+":D"+std::to_string(detector));
        for(std::size_t row=0;row<reference.size();++row) {
            n.times_unix_sec.push_back(1000.+row);
            // Slightly different, positively coupled peer networks make the
            // shared-reference interval identifiable without hiding the test.
            const double gain=1.+.0001*(network+1)*(blocks[row]+1);
            for(int detector=0;detector<12;++detector)
                n.values(row,detector)=network==0 && detector==0?target[row]:gain*reference[row]+detector;
        }
        networks.push_back(std::move(n));
    }
    return networks;
}
}

TEST(PointOpticalResponse, WeakHealthyQuietNoisyAndReversedHaveSeparateDecisions) {
    const auto data=fixture();const auto evidence=PointOpticalEvidence::learn(data,request());
    const auto plan=PointOpticalPlan::consider(evidence);const auto &decision=plan->for_network(data[0]);
    ASSERT_TRUE(evidence->networks()[0].reference_available)<<evidence->networks()[0].reference_reason;
    EXPECT_EQ(decision[0].state,PointOpticalState::weak_response)<<decision[0].reason;
    EXPECT_EQ(decision[1].state,PointOpticalState::retained)<<decision[1].reason;
    EXPECT_EQ(decision[5].state,PointOpticalState::sign_mismatch)<<decision[5].reason;
    EXPECT_EQ(decision[7].state,PointOpticalState::retained)<<decision[7].reason;
    EXPECT_LT(evidence->networks()[0].detectors[7].binned_residual_rms,evidence->networks()[0].detectors[1].binned_residual_rms);
    // Changing the owner-selected cut changes Consider, not the learned cohort.
    EXPECT_EQ(PointOpticalPlan::consider(evidence,.01)->for_network(data[0])[0].state,PointOpticalState::retained);
}

TEST(PointOpticalResponse, CompactReportMatchesFrozenDecisionsAndFullIntervals) {
    auto data=fixture();
    data.front().eligible.col(4).setZero();
    const auto plan=PointOpticalPlan::consider(PointOpticalEvidence::learn(data,request()));
    const auto full=citlali::cli::detail::point_health_record(*plan);
    const auto compact=citlali::cli::detail::point_decision_record(*plan);
    EXPECT_EQ(compact["schema"].as<std::string>(),"citlali-point-optical-decisions-v1");
    std::vector<std::string> keys;
    for(const auto &field:compact)keys.push_back(field.first.as<std::string>());
    EXPECT_EQ(keys,(std::vector<std::string>{"schema","method","min_relative_response","bootstrap_seed_policy",
                    "bootstrap_seed_identity","uncertainty_limitations","networks"}));
    for(const auto *key:{"method","min_relative_response","bootstrap_seed_policy","bootstrap_seed_identity","uncertainty_limitations"})
        EXPECT_EQ(YAML::Dump(compact[key]),YAML::Dump(full[key]));
    ASSERT_EQ(compact["networks"].size(),full["networks"].size());
    std::size_t retained=0,unavailable=0,exceptions=0;
    for(std::size_t i=0;i<full["networks"].size();++i) {
        const auto a=full["networks"][i], b=compact["networks"][i];
        EXPECT_EQ(a["network"].as<int>(),b["network"].as<int>());
        EXPECT_EQ(a["array"].as<int>(),b["array"].as<int>());
        ASSERT_EQ(a["detectors"].size(),b["detectors"].size());
        for(std::size_t j=0;j<a["detectors"].size();++j) {
            const auto x=a["detectors"][j], y=b["detectors"][j];
            EXPECT_EQ(x["detector_id"].as<std::string>(),y["detector_id"].as<std::string>());
            const auto state=x["state"].as<std::string>();
            EXPECT_EQ(state,y["state"].as<std::string>());
            if(state=="retained" || state=="prior_unavailable") {
                EXPECT_EQ(y.size(),2u);
                if(state=="retained")++retained;else ++unavailable;
            } else {
                EXPECT_EQ(y.size(),5u);
                for(const auto *key:{"reason","relative_response","interval"})
                    EXPECT_EQ(YAML::Dump(x[key]),YAML::Dump(y[key]));
                ++exceptions;
            }
        }
    }
    EXPECT_GT(retained,0u);EXPECT_GT(unavailable,0u);EXPECT_GT(exceptions,0u);
}

TEST(PointOpticalResponse, SourceProtectionExcludesSourceFromLearningOnly) {
    auto clean=fixture();for(int t=400;t<430;++t)clean[0].source_free(t,2)=0;
    auto injected=clean;for(int t=400;t<430;++t)injected[0].values(t,2)+=1e9;
    injected[0].cal_identity="different-receipt-for-protected-source";
    const auto a=PointOpticalEvidence::learn(clean,request()),b=PointOpticalEvidence::learn(injected,request());
    const auto &x=a->networks()[0].detectors[2],&y=b->networks()[0].detectors[2];
    EXPECT_DOUBLE_EQ(x.relative_response,y.relative_response);EXPECT_DOUBLE_EQ(x.interval.lower,y.interval.lower);
    EXPECT_EQ(a->seed_identity(),b->seed_identity());
    EXPECT_EQ(x.available_samples,1200);EXPECT_EQ(x.source_free_samples,1170);
    EXPECT_GT(injected[0].values(405,2),1e8); // owner never mutates calibrated input
}

TEST(PointOpticalResponse, ReceiptMetadataDoesNotChangeBootstrapDrawsOrDecisions) {
    const auto data=fixture();auto republished=data;
    for(auto &n:republished) {
        // The adapter's receipt digest changes with elapsed-time and output
        // metadata. These opaque publication bindings must still be retained.
        n.cal_identity+="-new-destination-and-Learn_seconds";
        n.source_mask_identity+="-new-receipt-binding";
    }
    const auto a=PointOpticalEvidence::learn(data,request());
    const auto b=PointOpticalEvidence::learn(republished,request());
    EXPECT_EQ(a->seed_identity(),b->seed_identity());
    const auto pa=PointOpticalPlan::consider(a),pb=PointOpticalPlan::consider(b);
    auto same_number=[](double x,double y){if(std::isnan(x))EXPECT_TRUE(std::isnan(y));else EXPECT_DOUBLE_EQ(x,y);};
    auto same_interval=[&](const PointOpticalInterval &x,const PointOpticalInterval &y) {
        EXPECT_EQ(x.seed,y.seed);EXPECT_EQ(x.available,y.available);EXPECT_EQ(x.reason,y.reason);
        EXPECT_EQ(x.informative_blocks,y.informative_blocks);EXPECT_EQ(x.resamples,y.resamples);
        same_number(x.lower,y.lower);same_number(x.upper,y.upper);
    };
    for(std::size_t n=0;n<data.size();++n) {
        EXPECT_EQ(b->networks()[n].cal_identity,republished[n].cal_identity);
        EXPECT_EQ(b->networks()[n].source_mask_identity,republished[n].source_mask_identity);
        // Stable science seeds are not interchangeable replay/provenance IDs.
        EXPECT_THROW(pa->for_network(republished[n]),std::invalid_argument);
        for(std::size_t d=0;d<data[n].detector_ids.size();++d) {
            const auto &x=a->networks()[n].detectors[d],&y=b->networks()[n].detectors[d];
            same_number(x.relative_response,y.relative_response);same_interval(x.interval,y.interval);
            for(std::size_t c=0;c<x.checks.size();++c)same_interval(x.checks[c],y.checks[c]);
            EXPECT_EQ(pa->for_network(data[n])[d].state,pb->for_network(republished[n])[d].state);
            EXPECT_EQ(pa->for_network(data[n])[d].reason,pb->for_network(republished[n])[d].reason);
        }
    }
}

TEST(PointOpticalResponse, LearningInputsAndCohortRemainSeedRelevant) {
    const auto data=fixture();const auto baseline=PointOpticalEvidence::learn(data,request());
    for(int change=0;change<8;++change) {
        SCOPED_TRACE(change);auto changed=data;
        switch(change) {
        case 0: changed[0].values(10,1)+=100;break;
        case 1: changed[2].values(10,1)+=100;break; // held-out peer content
        case 2: changed[0].eligible.block(0,1,20,1).setZero();break;
        case 3: changed[0].source_free.block(0,1,20,1).setZero();break;
        case 4: for(auto &n:changed)for(auto &block:n.science_blocks)block+=10;break;
        case 5: changed[2].detector_ids[1]+="-different-occurrence";break;
        case 6: changed[2].network=4;break;
        case 7: for(auto &n:changed)for(auto &t:n.times_unix_sec)t+=10.;break;
        }
        const auto actual=PointOpticalEvidence::learn(changed,request());
        EXPECT_NE(baseline->seed_identity(),actual->seed_identity());
    }
    // Request identity separates different binning/resampling policies, while
    // the Consider threshold must not alter learned random draws.
    auto r=request();r.bin_seconds=1.;EXPECT_NE(baseline->seed_identity(),PointOpticalEvidence::learn(data,r)->seed_identity());
    r=request();r.bootstrap_samples=800;EXPECT_NE(baseline->seed_identity(),PointOpticalEvidence::learn(data,r)->seed_identity());
    r=request();r.confidence_level=.98;EXPECT_NE(baseline->seed_identity(),PointOpticalEvidence::learn(data,r)->seed_identity());
    r=request();r.min_relative_response=.2;EXPECT_EQ(baseline->seed_identity(),PointOpticalEvidence::learn(data,r)->seed_identity());
}

TEST(PointOpticalResponse, CanonicalDomainOrderPreservesSeedIdentity) {
    const auto data=fixture();auto reordered=data;std::reverse(reordered.begin(),reordered.end());
    for(auto &n:reordered) {
        std::swap(n.detector_ids[0],n.detector_ids[1]);n.values.col(0).swap(n.values.col(1));
        n.eligible.col(0).swap(n.eligible.col(1));n.source_free.col(0).swap(n.source_free.col(1));
    }
    const auto a=PointOpticalEvidence::learn(data,request()),b=PointOpticalEvidence::learn(reordered,request());
    EXPECT_EQ(a->seed_identity(),b->seed_identity());
    for(const auto &n:a->networks()) {
        const auto other=std::find_if(b->networks().begin(),b->networks().end(),[&](const auto &x){return x.network==n.network;});
        ASSERT_NE(other,b->networks().end());
        for(const auto &d:n.detectors) {
            const auto peer=std::find_if(other->detectors.begin(),other->detectors.end(),[&](const auto &x){return x.detector_id==d.detector_id;});
            ASSERT_NE(peer,other->detectors.end());
            for(std::size_t c=0;c<d.checks.size();++c)EXPECT_EQ(d.checks[c].seed,peer->checks[c].seed);
        }
    }
}

TEST(PointOpticalResponse, RestorationRejectsMissingForeignOrTamperedSeedIdentity) {
    const auto e=PointOpticalEvidence::learn(fixture(),request());
    EXPECT_THROW(PointOpticalEvidence::restore_verified(e->request(),e->networks(),""),std::invalid_argument);
    EXPECT_THROW(PointOpticalEvidence::restore_verified(e->request(),e->networks(),"sha256:"+std::string(64,'0')),std::invalid_argument);
    auto changed=e->networks();changed[0].detectors[1].checks[0].seed^=1;
    EXPECT_THROW(PointOpticalEvidence::restore_verified(e->request(),changed,e->seed_identity()),std::invalid_argument);
    const auto restored=PointOpticalEvidence::restore_verified(e->request(),e->networks(),e->seed_identity());
    EXPECT_EQ(restored->seed_identity(),e->seed_identity());
}

TEST(PointOpticalResponse, MissingAndUnresolvedSupportAreNotConfidentWeakResponse) {
    auto data=fixture();data[0].eligible.col(4).setZero();data[0].values.col(4).setConstant(NAN);
    data[0].source_free.col(3).setZero();data[0].source_free.block(0,6,600,1).setZero();
    const auto evidence=PointOpticalEvidence::learn(data,request());const auto plan=PointOpticalPlan::consider(evidence);
    const auto &d=plan->for_network(data[0]);EXPECT_EQ(d[4].state,PointOpticalState::prior_unavailable);
    EXPECT_EQ(d[3].state,PointOpticalState::unresolved);EXPECT_EQ(d[6].state,PointOpticalState::unresolved);
    EXPECT_FALSE(d[3].retained());EXPECT_FALSE(d[6].retained());
}

TEST(PointOpticalResponse, OwnNetworkCannotBecomeItsReference) {
    auto data=fixture();data.resize(1);const auto evidence=PointOpticalEvidence::learn(data,request());
    EXPECT_FALSE(evidence->networks()[0].reference_available);
    const auto plan=PointOpticalPlan::consider(evidence);
    for(const auto &d:plan->for_network(data[0]))EXPECT_EQ(d.state,PointOpticalState::unresolved);
}

TEST(PointOpticalResponse, TwoNetworkArrayNeedsCrossArrayCorroboration) {
    auto data=fixture();auto n11=data[0],n12=data[1];
    for(auto *n:{&n11,&n12}) {
        n->network+=11;n->array=2;n->cal_identity+="-a2000";n->source_mask_identity+="-a2000";
        for(auto &id:n->detector_ids)id+="-a2000";
    }
    data.push_back(n11);data.push_back(n12);
    const auto good=PointOpticalEvidence::learn(data,request());
    EXPECT_TRUE(good->networks()[3].cross_array_required);EXPECT_TRUE(good->networks()[3].cross_array_corroborated);
    const auto alone=PointOpticalEvidence::learn({n11,n12},request());
    EXPECT_FALSE(alone->networks()[0].cross_array_corroborated);
    EXPECT_EQ(PointOpticalPlan::consider(alone)->for_network(n11)[1].state,PointOpticalState::unresolved);
}

TEST(PointOpticalResponse, VerifiedRestoreIsDeterministicAndMembershipRejectsForeignBindings) {
    const auto data=fixture();const auto a=PointOpticalEvidence::learn(data,request());const auto b=PointOpticalEvidence::learn(data,request());
    EXPECT_DOUBLE_EQ(a->networks()[0].detectors[0].interval.lower,b->networks()[0].detectors[0].interval.lower);
    EXPECT_EQ(a->networks()[0].detectors[0].checks[0].seed,b->networks()[0].detectors[0].checks[0].seed);
    const auto restored=PointOpticalEvidence::restore_verified(a->request(),a->networks(),a->seed_identity());const auto p=PointOpticalPlan::consider(restored);
    EXPECT_EQ(p->for_network(data[0])[0].state,PointOpticalState::weak_response);
    auto foreign=data[0];foreign.cal_identity="foreign";EXPECT_THROW(p->for_network(foreign),std::invalid_argument);
    foreign=data[0];std::swap(foreign.detector_ids[0],foreign.detector_ids[1]);EXPECT_THROW(p->for_network(foreign),std::invalid_argument);
    auto bad=a->networks();bad[0].detectors[0].interval.upper=bad[0].detectors[0].interval.lower-1;
    EXPECT_THROW(PointOpticalEvidence::restore_verified(a->request(),bad,a->seed_identity()),std::invalid_argument);
}

TEST(PointOpticalResponse, InvalidFiniteEligibilityShapesAndConfigurationFailBeforeLearning) {
    auto data=fixture();data[0].values(20,1)=NAN;EXPECT_THROW(PointOpticalEvidence::learn(data,request()),std::invalid_argument);
    data[0].eligible(20,1)=0;EXPECT_NO_THROW(PointOpticalEvidence::learn(data,request()));
    auto r=request();r.min_relative_response=0;EXPECT_THROW(PointOpticalEvidence::learn(data,r),std::invalid_argument);
    r=request();r.bootstrap_samples=10;EXPECT_THROW(PointOpticalEvidence::learn(data,r),std::invalid_argument);
    data=fixture();data[0].times_unix_sec[2]=data[0].times_unix_sec[1];EXPECT_THROW(PointOpticalEvidence::learn(data,request()),std::invalid_argument);
    data=fixture();data[1].detector_ids[0]=data[0].detector_ids[0];EXPECT_THROW(PointOpticalEvidence::learn(data,request()),std::invalid_argument);
    data=fixture();data[0].values.col(1).setConstant(std::numeric_limits<double>::max());
    EXPECT_THROW(PointOpticalEvidence::learn(data,request()),std::runtime_error);
}

TEST(PointOpticalResponse, NoisyEvidenceDoesNotUseCorrelationAsARejectionThreshold) {
    auto data=fixture();for(int t=0;t<data[0].values.rows();++t)data[0].values(t,8)+=100*std::sin(.37*t*t);
    const auto e=PointOpticalEvidence::learn(data,request());const auto p=PointOpticalPlan::consider(e);
    EXPECT_NE(p->for_network(data[0])[8].state,PointOpticalState::weak_response);
    EXPECT_GT(e->networks()[0].detectors[8].binned_residual_rms,e->networks()[0].detectors[7].binned_residual_rms);
}

TEST(PointOpticalResponse, BootstrapRetainsSparseAndConstantBlocksThatIdentifyTheFullResponse) {
    for(bool constant_blocks:{false,true}) {
        SCOPED_TRACE(constant_blocks?"constant blocks":"singleton blocks");
        const auto data=sparse_block_fixture(constant_blocks);
        const auto evidence=PointOpticalEvidence::learn(data,request());
        ASSERT_TRUE(evidence->networks()[0].reference_available)<<evidence->networks()[0].reference_reason;
        const auto &detector=evidence->networks()[0].detectors[0];
        EXPECT_GT(detector.relative_response,.9);
        EXPECT_LT(detector.relative_response,1.1);
        for(std::size_t check=0;check<detector.checks.size();++check)
            EXPECT_EQ(detector.checks[check].informative_blocks,check<2?12:6);
        const auto plan=PointOpticalPlan::consider(evidence);
        EXPECT_NE(plan->for_network(data[0])[0].state,PointOpticalState::weak_response);
        if(detector.interval.available)EXPECT_GT(detector.interval.upper,.9);
    }
}

TEST(PointOpticalResponse, RankDeficientBlockResampleIsUnavailableAndDetectorUnresolved) {
    auto data=sparse_block_fixture(true);
    for(Eigen::Index row=0;row<data[0].values.rows();++row) {
        const auto block=data[0].science_blocks[row];
        data[0].source_free(row,0)=block==2 || block==3;
    }
    const auto evidence=PointOpticalEvidence::learn(data,request());
    ASSERT_TRUE(evidence->networks()[0].reference_available)<<evidence->networks()[0].reference_reason;
    const auto &detector=evidence->networks()[0].detectors[0];
    ASSERT_TRUE(std::isfinite(detector.relative_response));
    EXPECT_GT(detector.relative_response,.9);
    for(int panel=0;panel<2;++panel) {
        const auto &check=detector.checks[panel];
        EXPECT_EQ(check.informative_blocks,2);EXPECT_FALSE(check.available);
        EXPECT_EQ(check.reason,"resample_rank_unavailable");
    }
    const auto plan=PointOpticalPlan::consider(evidence);
    EXPECT_EQ(plan->for_network(data[0])[0].state,PointOpticalState::unresolved);
}

TEST(PointOpticalResponse, ParallelWorkersPreserveEveryEvidenceValueSeedAndDecision) {
    auto data=fixture();
    data[0].eligible.col(4).setZero();data[0].values.col(4).setConstant(NAN);
    data[0].source_free.col(3).setZero();
    // Include the two-network cross-array reference branch, not just bootstrap.
    for(int i=0;i<2;++i) {
        auto n=data[i];n.network+=11;n.array=2;n.cal_identity+="-a2000";
        n.source_mask_identity+="-a2000";
        for(auto &id:n.detector_ids)id+="-a2000";
        data.push_back(std::move(n));
    }
    const auto serial=PointOpticalEvidence::learn(data,request(),{1});
    const auto serial_plan=PointOpticalPlan::consider(serial);
    auto number=[](double a,double b){EXPECT_TRUE((std::isnan(a)&&std::isnan(b)) || a==b);};
    auto interval=[&](const auto &a,const auto &b) {
        EXPECT_EQ(a.available,b.available);number(a.lower,b.lower);number(a.upper,b.upper);
        EXPECT_EQ(a.informative_blocks,b.informative_blocks);EXPECT_EQ(a.resamples,b.resamples);
        EXPECT_EQ(a.seed,b.seed);EXPECT_EQ(a.reason,b.reason);
    };
    for(unsigned workers:{3,12}) {
        SCOPED_TRACE(workers);PointOpticalExecutionReport report;
        const auto other=PointOpticalEvidence::learn(data,request(),{workers},&report);
        const auto plan=PointOpticalPlan::consider(other);
        EXPECT_EQ(serial->seed_identity(),other->seed_identity());
        EXPECT_EQ(report.aggregation.jobs,data.size());
        EXPECT_EQ(report.references.started_workers,std::min<std::size_t>(workers,data.size()));
        EXPECT_EQ(report.bootstrap.jobs,60);EXPECT_LE(report.bootstrap.peak_active_jobs,workers);
        ASSERT_EQ(serial->networks().size(),other->networks().size());
        for(std::size_t ni=0;ni<data.size();++ni) {
            const auto &a=serial->networks()[ni],&b=other->networks()[ni];
            EXPECT_EQ(a.observation_identity,b.observation_identity);EXPECT_EQ(a.cal_identity,b.cal_identity);
            EXPECT_EQ(a.source_mask_identity,b.source_mask_identity);EXPECT_EQ(a.array,b.array);EXPECT_EQ(a.network,b.network);
            EXPECT_EQ(a.detector_ids,b.detector_ids);EXPECT_EQ(a.reference_networks,b.reference_networks);
            EXPECT_EQ(a.reference_available,b.reference_available);EXPECT_EQ(a.cross_array_required,b.cross_array_required);
            EXPECT_EQ(a.cross_array_corroborated,b.cross_array_corroborated);EXPECT_EQ(a.reference_reason,b.reference_reason);
            number(a.response_normalization,b.response_normalization);
            ASSERT_EQ(a.detectors.size(),b.detectors.size());
            for(std::size_t di=0;di<a.detectors.size();++di) {
                const auto &x=a.detectors[di],&y=b.detectors[di];
                EXPECT_EQ(x.detector_id,y.detector_id);EXPECT_EQ(x.available_samples,y.available_samples);
                EXPECT_EQ(x.source_free_samples,y.source_free_samples);EXPECT_EQ(x.binned_samples,y.binned_samples);
                number(x.relative_response,y.relative_response);number(x.correlation,y.correlation);
                number(x.binned_residual_rms,y.binned_residual_rms);interval(x.interval,y.interval);
                for(std::size_t k=0;k<x.checks.size();++k)interval(x.checks[k],y.checks[k]);
                EXPECT_EQ(serial_plan->for_network(data[ni])[di].state,plan->for_network(data[ni])[di].state);
                EXPECT_EQ(serial_plan->for_network(data[ni])[di].reason,plan->for_network(data[ni])[di].reason);
            }
        }
    }
    EXPECT_THROW(PointOpticalEvidence::learn(data,request(),{0}),std::invalid_argument);
    data[0].values(20,1)=NAN;
    EXPECT_THROW(PointOpticalEvidence::learn(data,request(),{3}),std::invalid_argument);
}

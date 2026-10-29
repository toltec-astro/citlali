#include <citlali/core/pipeline/point_optical_response.h>
#include <gtest/gtest.h>
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

TEST(PointOpticalResponse, SourceProtectionExcludesSourceFromLearningOnly) {
    auto clean=fixture();for(int t=400;t<430;++t)clean[0].source_free(t,2)=0;
    auto injected=clean;for(int t=400;t<430;++t)injected[0].values(t,2)+=1e9;
    const auto a=PointOpticalEvidence::learn(clean,request()),b=PointOpticalEvidence::learn(injected,request());
    const auto &x=a->networks()[0].detectors[2],&y=b->networks()[0].detectors[2];
    EXPECT_DOUBLE_EQ(x.relative_response,y.relative_response);EXPECT_DOUBLE_EQ(x.interval.lower,y.interval.lower);
    EXPECT_EQ(x.available_samples,1200);EXPECT_EQ(x.source_free_samples,1170);
    EXPECT_GT(injected[0].values(405,2),1e8); // owner never mutates calibrated input
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
    const auto restored=PointOpticalEvidence::restore_verified(a->request(),a->networks());const auto p=PointOpticalPlan::consider(restored);
    EXPECT_EQ(p->for_network(data[0])[0].state,PointOpticalState::weak_response);
    auto foreign=data[0];foreign.cal_identity="foreign";EXPECT_THROW(p->for_network(foreign),std::invalid_argument);
    foreign=data[0];std::swap(foreign.detector_ids[0],foreign.detector_ids[1]);EXPECT_THROW(p->for_network(foreign),std::invalid_argument);
    auto bad=a->networks();bad[0].detectors[0].interval.upper=bad[0].detectors[0].interval.lower-1;
    EXPECT_THROW(PointOpticalEvidence::restore_verified(a->request(),bad),std::invalid_argument);
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

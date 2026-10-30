#include "../src/citlali/cli/point_weighted_bindings.h"
#include <gtest/gtest.h>

namespace {
using namespace citlali::cli::detail;
const std::string a(64,'a'),b(64,'b'),c(64,'c'),d(64,'d');

struct FeedbackFixture {
    PointFeedbackBinding expected{a,b,c,"obs:152418_0_2","sha256:"+d,"a1100",1};
    YAML::Node feedback,bundle;
    FeedbackFixture() {
        feedback["schema"]="configured-point-base-feedback-v1";feedback["status"]="ready";
        feedback["model_sha256"]=a;feedback["input_bundle_sha256"]=b;feedback["map_sha256"]=c;
        feedback["observation"]=expected.observation;feedback["frozen_health_identity"]=expected.health_identity;
        feedback["array"]=expected.array;feedback["replacement"]=true;feedback["gain"]=1.;
        feedback["feedback_domain_radius_arcsec"]=40.;feedback["frame"]="delta-source-AltAz";
        feedback["coordinate_units"]="arcsec";feedback["signal_units"]="mJy/nominal beam";
        bundle["schema"]="citlali-forward-map-bundle-v1";bundle["outcome"]="success";
        bundle["signal_unit"]="mJy/beam";bundle["observation"]=expected.observation;
        bundle["frozen_health_identity"]=expected.health_identity;bundle["array"]=expected.array;
        bundle["cycle_index"]=expected.previous_cycle;bundle["files"]["map.fits"]=c;
    }
    void validate() const {validate_point_feedback_provenance(feedback,bundle,expected);}
};

TEST(PointWeightedBindings, FeedbackRequiresExactObservationAndFrozenHealth) {
    FeedbackFixture valid;EXPECT_NO_THROW(valid.validate());
    for (const auto *key:{"observation","frozen_health_identity","array"}) {
        FeedbackFixture x;x.feedback[key]="foreign";EXPECT_THROW(x.validate(),std::invalid_argument);
        FeedbackFixture y;y.bundle[key]="foreign";EXPECT_THROW(y.validate(),std::invalid_argument);
    }
    FeedbackFixture both;both.feedback["observation"]="obs:152389_0_2";both.bundle["observation"]="obs:152389_0_2";
    EXPECT_THROW(both.validate(),std::invalid_argument);
}

TEST(PointWeightedBindings, FeedbackAuthenticatesPayloadBundleMapAndPrecedingCycle) {
    for (const auto *key:{"model_sha256","input_bundle_sha256","map_sha256"}) {
        FeedbackFixture x;x.feedback[key]=d;EXPECT_THROW(x.validate(),std::invalid_argument);
    }
    FeedbackFixture file;file.bundle["files"]["map.fits"]=d;EXPECT_THROW(file.validate(),std::invalid_argument);
    for (int cycle:{-1,0,2}) {FeedbackFixture x;x.bundle["cycle_index"]=cycle;EXPECT_THROW(x.validate(),std::invalid_argument);}
    FeedbackFixture digest;digest.expected.map_sha256="not-a-digest";EXPECT_THROW(digest.validate(),std::invalid_argument);
}

TEST(PointWeightedBindings, FeedbackRejectsStoppedIncompatibleOrMalformedPublication) {
    for (const auto *key:{"schema","status","frame","coordinate_units","signal_units"}) {
        FeedbackFixture x;x.feedback[key]="different";EXPECT_THROW(x.validate(),std::invalid_argument);
    }
    FeedbackFixture replacement;replacement.feedback["replacement"]=false;EXPECT_THROW(replacement.validate(),std::invalid_argument);
    FeedbackFixture gain;gain.feedback["gain"]=2.;EXPECT_THROW(gain.validate(),std::invalid_argument);
    FeedbackFixture radius;radius.feedback["feedback_domain_radius_arcsec"]=30.;EXPECT_THROW(radius.validate(),std::invalid_argument);
    FeedbackFixture absent;absent.feedback.remove("model_sha256");EXPECT_THROW(absent.validate(),std::invalid_argument);
    FeedbackFixture malformed;malformed.bundle["cycle_index"]="not-a-number";EXPECT_THROW(malformed.validate(),std::invalid_argument);
}

struct ProcessingFixture {
    YAML::Node producer,input,processing;
    ProcessingFixture() {
        producer["processing_scans_sha256"]=a;
        input["decision_apply"]["processing_request"]["sha256"]=b;
        input["decision_apply"]["timing_inputs"]=YAML::Load("[{network: 0, sha256: native-clock}]");
        processing["generation"]="native-processing-request:sha256:"+b;
        processing["generation_kind"]="fresh-realization-from-explicit-request";
        processing["timing_inputs"]=YAML::Clone(input["decision_apply"]["timing_inputs"]);
        processing["scans"]=YAML::Load("[{scan: 0, science_native: [[1, 10], [10, 15]]}, {scan: 1, science_native: [[20, 30]]}]");
    }
    std::vector<PointScienceInterval> validate() const {
        return validate_point_processing_binding(producer,input,processing,a,a,40);
    }
};

TEST(PointWeightedBindings, ProcessingBindsProducerDigestGenerationAndTiming) {
    ProcessingFixture valid;const auto intervals=valid.validate();ASSERT_EQ(intervals.size(),3);
    EXPECT_EQ(intervals[0].scan,0);EXPECT_EQ(intervals[0].first,1);EXPECT_EQ(intervals[2].past_last,30);
    EXPECT_THROW(validate_point_processing_binding(valid.producer,valid.input,valid.processing,a,b,40),std::invalid_argument);
    ProcessingFixture producer;producer.producer["processing_scans_sha256"]=c;EXPECT_THROW(producer.validate(),std::invalid_argument);
    ProcessingFixture generation;generation.processing["generation"]="native-processing-request:sha256:"+c;EXPECT_THROW(generation.validate(),std::invalid_argument);
    ProcessingFixture timing;timing.processing["timing_inputs"][0]["network"]=1;EXPECT_THROW(timing.validate(),std::invalid_argument);
    ProcessingFixture ambiguous;ambiguous.input["decision_apply"]["processing_provenance"]["sha256"]=c;EXPECT_THROW(ambiguous.validate(),std::invalid_argument);
}

TEST(PointWeightedBindings, ProcessingAllowsAuthenticatedExistingGenerationAndUnmappedScans) {
    ProcessingFixture x;x.input["decision_apply"].remove("processing_request");
    x.input["decision_apply"]["processing_provenance"]["sha256"]=b;
    x.processing["generation"]="sha256:"+b;x.processing["generation_kind"]="verified-existing-processing-generation";
    YAML::Node empty;empty["scan"]=2;x.processing["scans"].push_back(empty);
    EXPECT_EQ(x.validate().size(),3);
}

TEST(PointWeightedBindings, ProcessingRejectsDuplicateOverlappingOrReorderedNativeIntervals) {
    for (const auto *scans:{
        "[{scan: 0, science_native: [[1, 10]]}, {scan: 0, science_native: [[20, 30]]}]",
        "[{scan: 1, science_native: [[1, 10]]}, {scan: 0, science_native: [[20, 30]]}]",
        "[{scan: 0, science_native: [[1, 10], [9, 15]]}]",
        "[{scan: 0, science_native: [[1, 20]]}, {scan: 1, science_native: [[19, 30]]}]",
        "[{scan: 0, science_native: [[20, 30], [1, 10]]}]",
        "[{scan: 0, science_native: [[20, 30]]}, {scan: 1, science_native: [[1, 10]]}]"}) {
        ProcessingFixture x;x.processing["scans"]=YAML::Load(scans);EXPECT_THROW(x.validate(),std::invalid_argument)<<scans;
    }
}

TEST(PointWeightedBindings, ProcessingRejectsInvalidOrEmptyNativeDomain) {
    for (const auto *range:{"[-1, 10]","[10, 10]","[10, 41]","[10, 5]","[10]","[1.5, 10]"}) {
        ProcessingFixture x;x.processing["scans"][0]["science_native"][0]=YAML::Load(range);
        EXPECT_THROW(x.validate(),std::invalid_argument)<<range;
    }
    ProcessingFixture x;x.processing["scans"]=YAML::Load("[{scan: 0}]");EXPECT_THROW(x.validate(),std::invalid_argument);
}

TEST(PointWeightedBindings, HealthDecisionRequiresStateAndReason) {
    YAML::Node detector;detector["state"]="weak_response";detector["reason"]="confidently_weak_shared_optical_response";
    EXPECT_NO_THROW(validate_point_health_decision(detector,"weak_response","confidently_weak_shared_optical_response"));
    EXPECT_THROW(validate_point_health_decision(detector,"retained","confidently_weak_shared_optical_response"),std::invalid_argument);
    EXPECT_THROW(validate_point_health_decision(detector,"weak_response","different"),std::invalid_argument);
    detector.remove("reason");EXPECT_THROW(validate_point_health_decision(detector,"weak_response","reason"),std::invalid_argument);
}
} // namespace

#include "../src/citlali/cli/point_recovery_bindings.h"
TEST(PointWeightedBindings, RecoveryCaseLineageAndSeedLifetime) {
    YAML::Node n=YAML::Load("schema: citlali-point-recovery-v1\ncase: SEED\nobservation: obs:152418_0_2\nconfig_sha256: cfg\nhealth_sha256: health\narrays: {a1100: {}, a1400: {}, a2000: {}}\n");
    using citlali::cli::detail::validate_point_recovery;
    EXPECT_THROW(validate_point_recovery(n,"obs:152418_0_2","cfg","health",0),std::invalid_argument);
    EXPECT_NO_THROW(validate_point_recovery(n,"obs:152418_0_2","cfg","health",1));
    EXPECT_THROW(validate_point_recovery(n,"obs:152418_0_2","cfg","health",11),std::invalid_argument);
    EXPECT_THROW(validate_point_recovery(n,"other","cfg","health",1),std::invalid_argument);
    n["case"]="unbounded";
    EXPECT_THROW(validate_point_recovery(n,"obs:152418_0_2","cfg","health",1),std::invalid_argument);
    n["case"]="CORE-";
    EXPECT_NO_THROW(validate_point_recovery(n,"obs:152418_0_2","cfg","health",0));
    n["unexpected"]=true;
    EXPECT_THROW(validate_point_recovery(n,"obs:152418_0_2","cfg","health",0),std::invalid_argument);
}
TEST(PointWeightedBindings, RecoveryReferenceRequiresBothWeightedTruths) {
    YAML::Node n=YAML::Load("schema: citlali-point-recovery-v1\ncase: REF\nobservation: obs:152418_0_2\nconfig_sha256: cfg\nhealth_sha256: health\narrays: {a1100: {}, a1400: {}, a2000: {}}\n");
    using citlali::cli::detail::validate_point_recovery;
    EXPECT_THROW(validate_point_recovery(n,"obs:152418_0_2","cfg","health",0),std::invalid_argument);
    for(const auto *a:{"a1100","a1400","a2000"}) n["weighted_references"][a]=YAML::Load("[{}, {}]");
    EXPECT_NO_THROW(validate_point_recovery(n,"obs:152418_0_2","cfg","health",0));
    n["case"]="CORE+";
    EXPECT_THROW(validate_point_recovery(n,"obs:152418_0_2","cfg","health",0),std::invalid_argument);
}

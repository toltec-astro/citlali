#include "../src/citlali/cli/point_weighted_bindings.h"
#include "../src/citlali/cli/point_observation_pass.h"
#include "../src/citlali/cli/point_cal_handoff.h"
#include "../src/citlali/cli/map_file_ledger.h"
#include <chrono>
#include <gtest/gtest.h>

namespace {
using namespace citlali::cli::detail;
const std::string a(64,'a'),b(64,'b'),c(64,'c'),d(64,'d');

TEST(PointCalHandoff, ReceiptBindsChannelValuesSlotsCausesAndNetwork) {
    PointCalNetwork network{"products/network7","inputs/network7.json",a,b,{{17,{0.5,-3.0},{0,2},{0,1,0}}}};
    YAML::Node record;
    record["available"]=2;
    record["files"]=network.channels[0].declarations();
    const std::string old_values("\0\0\0\0\0\0\xe0\x3f\0\0\0\0\0\0\x08\xc0",16);
    const std::string old_slots("\0\0\0\0\0\0\0\0\x02\0\0\0\0\0\0\0",16);
    const std::string old_causes("\0\0\1\0\0\0",6);
    EXPECT_EQ(record["files"]["channel-17-value.f64"].as<std::string>(),citlali::utils::sha256(old_values));
    EXPECT_EQ(record["files"]["channel-17-slot.i64"].as<std::string>(),citlali::utils::sha256(old_slots));
    EXPECT_EQ(record["files"]["channel-17-causes.u16"].as<std::string>(),citlali::utils::sha256(old_causes));
    network.require_binding("products/network7","inputs/network7.json",a,b,1);
    EXPECT_NO_THROW(network.admit(0,17,3,record));
    EXPECT_THROW(network.require_binding("products/network8","inputs/network7.json",a,b,1),std::runtime_error);
    EXPECT_THROW(network.require_binding("products/network7","inputs/replacement.json",a,b,1),std::runtime_error);
    EXPECT_THROW(network.require_binding("products/network7","inputs/network7.json",c,b,1),std::runtime_error);
    EXPECT_THROW(network.require_binding("products/network7","inputs/network7.json",a,c,1),std::runtime_error);
    EXPECT_THROW(network.admit(0,18,3,record),std::runtime_error);
    EXPECT_THROW(network.admit(0,17,4,record),std::runtime_error);
    for (const auto &suffix:{"-value.f64","-slot.i64","-causes.u16"}) {
        auto changed=YAML::Clone(record);
        changed["files"][std::string("channel-17")+suffix]=c;
        EXPECT_THROW(network.admit(0,17,3,changed),std::runtime_error);
    }
    network.channels[0].values[0]=2.0;
    EXPECT_THROW(network.admit(0,17,3,record),std::runtime_error);
}

TEST(PointCalHandoff, LiveBindingRejectsWrongNetworkInputOrGeneration) {
    PointCalNetwork network;
    network.root="network7";network.input_path="inputs/7.json";network.input_sha=a;
    EXPECT_THROW(network.require_live_generation(network.binding()),std::runtime_error);
    network.generation_sha=b;
    EXPECT_NO_THROW(network.require_live_generation(network.binding()));
    for(const auto *key:{"root","transport","handoff_sha256"}) {
        auto request=network.binding();request[key]="foreign";
        EXPECT_THROW(network.require_live_generation(request),std::runtime_error);
    }
    for(const auto *key:{"path","sha256"}) {
        auto request=network.binding();request["input"][key]="foreign";
        EXPECT_THROW(network.require_live_generation(request),std::runtime_error);
    }
}

TEST(PointCalHandoff, ZeroAvailableCalPreservesRtcAndAllCauseCounts) {
    PointCalNetwork network;
    network.input_sha=a;network.receipt_sha=b;network.generation_sha=c;
    network.learning=YAML::Load("{observation: 152389, network: 7}");
    network.cal=YAML::Load("{available: 0, scheduled: 3, input_RTC_attempt: 1001, input_VAL_generation: 1,"
        " output_VAL_generation: 2, selected_child_APT: apt, unit: mJy/nominal-beam, cause_bits: {rtc_unavailable: 1},"
        " detectors: [{joint_cause_counts: {1: 1, 9: 1}}, {joint_cause_counts: {1: 1}}]}");
    const auto terminal=YAML::Load("{state: complete-logical-RTC-terminal, scheduled_slots: 3}");
    const auto report=network.unavailable_report(terminal);
    EXPECT_EQ(report["available"].as<int>(),0);
    EXPECT_EQ(report["RTC_terminal"]["scheduled_slots"].as<int>(),3);
    EXPECT_EQ(report["joint_cause_counts"]["1"].as<int>(),2);
    EXPECT_EQ(report["joint_cause_counts"]["9"].as<int>(),1);
    EXPECT_EQ(report["CAL_sha256"].as<std::string>(),b);
    network.cal["scheduled"]=4;
    EXPECT_THROW(network.unavailable_report(terminal),std::logic_error);
    network.cal["scheduled"]=3;network.cal["available"]=1;
    EXPECT_THROW(network.unavailable_report(terminal),std::logic_error);
}

TEST(PointCalHandoff, EmptySupportedDomainStillBindsEveryScheduledCause) {
    PointCalNetwork network{"root","input",a,b,{{2,{},{},{1,2}}}};
    YAML::Node record;
    record["available"]=0;record["files"]=network.channels[0].declarations();
    EXPECT_EQ(record["files"]["channel-2-value.f64"].as<std::string>(),citlali::utils::sha256(""));
    EXPECT_NO_THROW(network.admit(0,2,2,record));
    record["files"]["extra"]=a;
    EXPECT_THROW(network.admit(0,2,2,record),std::runtime_error);
}

TEST(PointCalHandoff, FileVerificationCountsActualChannelAndOtherBytes) {
    const auto root=std::filesystem::temp_directory_path()/("point-hash-io-"+std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {std::filesystem::path dir;~Cleanup(){std::error_code ec;std::filesystem::remove_all(dir,ec);}} cleanup{root};
    ASSERT_TRUE(std::filesystem::create_directory(root));
    std::ofstream(root/"channel",std::ios::binary)<<"xyz";
    std::ofstream(root/"other",std::ios::binary)<<"ab";
    MapFileLedger ledger;
    ledger.bind_channel(root/"channel",citlali::utils::sha256("xyz"));
    ledger.bind(root/"other");ledger.verify();
    EXPECT_EQ(ledger.io.files,4u);EXPECT_EQ(ledger.io.bytes,10u);
    EXPECT_EQ(ledger.channel_io.files,2u);EXPECT_EQ(ledger.channel_io.bytes,6u);
    std::ofstream(root/"other",std::ios::binary)<<"XY";
    EXPECT_THROW(ledger.verify(),std::runtime_error);
}

TEST(PointObservationPass, BindsExactNextDescriptorRequestAndOutput) {
    namespace fs=std::filesystem;
    const auto root=fs::temp_directory_path()/("point-pass-"+std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {fs::path root; ~Cleanup(){std::error_code ec;fs::remove_all(root,ec);}} cleanup{root};
    fs::create_directories(root/"health");
    fs::create_directories(root/"weighted-00");
    const auto descriptor=root/"health/development.yaml";
    YAML::Node config;
    config["schema"]="citlali-development-v1";
    config["input"]["path"]=(root/"health-request.json").string();
    config["input"]["sha256"]=a;
    config["output"]=(root/"health/products").string();
    config["terminal"]="map";
    config["network_workers"]=3;
    auto save=[&](const fs::path &path){std::ofstream(path)<<YAML::Dump(config);return citlali::utils::sha256_file(path);};
    auto sha=save(descriptor);
    EXPECT_EQ(read_point_observation_pass(root,0,3,sha,a).request_sha,a);
    EXPECT_THROW(read_point_observation_pass(root,0,3,sha,b),std::invalid_argument);
    EXPECT_THROW(read_point_observation_pass(root,0,2,sha,a),std::invalid_argument);
    EXPECT_THROW(read_point_observation_pass(root,0,3,c,a),std::runtime_error);
    config["input"]["path"]=(root/"weighted-00-request.json").string();
    config["output"]=(root/"weighted-00/products").string();
    sha=save(root/"weighted-00/development.yaml");
    EXPECT_EQ(read_point_observation_pass(root,1,3,sha,a).output,root/"weighted-00/products");
    config["output"]=(root/"health/products").string();
    sha=save(root/"weighted-00/development.yaml");
    EXPECT_THROW(read_point_observation_pass(root,1,3,sha,a),std::invalid_argument);
}

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
TEST(PointWeightedBindings, ActualNativeSeedSwitchExpiresAndSignedParentsAreFresh) {
    using namespace citlali::pipeline;
    using citlali::cli::detail::apply_point_recovery;
    PtcMatrix parent(2,2),ordinary(2,2),addition(2,2);
    parent<<10,20,30,40;ordinary<<1,2,3,4;addition<<.25,.5,.75,1.;
    const auto mask=PtcMask::Ones(2,2).eval();
    auto y=parent,m=ordinary;
    apply_point_recovery("SEED",1,y,m,mask,addition,mask);
    EXPECT_TRUE(y==parent);EXPECT_TRUE(m==ordinary+addition);
    PtcMatrix fresh_y=parent,replacement=ordinary*2.;
    apply_point_recovery("SEED",2,fresh_y,replacement,mask,addition,mask);
    EXPECT_TRUE(fresh_y==parent);EXPECT_TRUE(replacement==ordinary*2.);
    for(const auto sign:{-1.,1.}) {
        y=parent;m=ordinary;
        apply_point_recovery(sign<0?"CORE-":"CORE+",2,y,m,mask,sign*addition,mask);
        EXPECT_TRUE(y==parent+sign*addition);EXPECT_TRUE(m==ordinary);
        const auto residual=FruitResidual::subtract("fixture",y,mask,m,mask);
        EXPECT_TRUE(residual->values()==parent+sign*addition-ordinary);
        EXPECT_TRUE(residual->operand()==ordinary); // restore only what was subtracted
    }
    y=parent;m=ordinary;
    apply_point_recovery("REF",0,y,m,mask,addition,mask);
    EXPECT_TRUE(y==parent);EXPECT_TRUE(m==ordinary);
}

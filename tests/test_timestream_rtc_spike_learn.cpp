#include <citlali/core/pipeline/timestream_rtc_spike_optical_reference.h>
#include "timestream_successor_identity_test_support.h"
#include <gtest/gtest.h>
#include <bit>
#include <chrono>
#include <iostream>
#include <random>

namespace {
using namespace citlali::pipeline;
namespace support = citlali::test::timestream_successor;

struct Input {
    std::vector<double> times;
    std::vector<TimestreamPacketCounter> counters;
    NativePairedReadoutMatrix x, r;
    std::vector<NativeReadoutCoordinateState> xs, rs;
    explicit Input(std::size_t rows = 600, std::size_t detectors = 1, double cadence = 0.008192)
        : times(rows), counters(rows), x(rows, detectors), r(rows, detectors),
          xs(support::valid_states(rows * detectors)), rs(xs) {
        for (std::size_t i = 0; i < rows; ++i) {
            times[i] = 1000.0 + cadence * i;
            counters[i] = 2000 + i;
            for (std::size_t d = 0; d < detectors; ++d) {
                x(i,d) = std::array{0.0, 1.0, 0.0, -1.0, 0.0, 0.0}[i % 6];
                r(i,d) = 2 * x(i,d);
            }
        }
    }
    std::shared_ptr<const NativePairedReadoutObservation> freeze() const {
        std::vector<NativePairedReadoutNetwork> networks;
        networks.push_back(NativePairedReadoutNetwork::admit(
            support::occurrence_axis(0, 100, times, counters),
            support::detector_axis(0, x.cols()), support::mapping_authority(0, "spike-test"),
            x, r, xs, rs));
        return support::make_observation(std::move(networks), {0});
    }
};

struct Fixture {
    std::shared_ptr<const NativePairedReadoutObservation> parent;
    std::shared_ptr<const NativePairedReadoutView> view;
    std::shared_ptr<const ValSnapshot> val;
    std::shared_ptr<const RtcSpikeSourceProtection> protection;
    Fixture(const Input &input, RtcSpikeProtection state = RtcSpikeProtection::outside_source,
            std::vector<RtcSpikeProtectionRegion> regions = {}) : parent(input.freeze()),
        view(NativePairedReadoutView::full(parent)), val(ValSnapshot::initial(parent)),
        protection(RtcSpikeSourceProtection::admit(parent, "source-test-v1", state, std::move(regions))) {}
    auto learn() const { return learn_rtc_spike_candidates(view, val, protection, 71); }
};

TEST(rtc_spike_learn, original_coordinates_have_separate_noise_and_candidates) {
    Input input;
    input.x(300,0) += 40;
    input.r(400,0) -= 80;
    Fixture f(input);
    const auto e = f.learn();
    ASSERT_EQ(e->blocks().size(), 1U);
    EXPECT_DOUBLE_EQ(e->blocks()[0].coordinates[0].scale, 1.4826);
    EXPECT_DOUBLE_EQ(e->blocks()[0].coordinates[1].scale, 2 * 1.4826);
    ASSERT_EQ(e->candidates().size(), 4U);
    EXPECT_EQ(e->candidates()[0].coordinate, NativeReadoutCoordinate::x);
    EXPECT_EQ(e->candidates()[0].earlier_row, 399);
    EXPECT_EQ(e->candidates()[0].later_row, 400);
    EXPECT_GT(e->candidates()[0].centered_difference, 0);
    EXPECT_LT(e->candidates()[1].centered_difference, 0);
    EXPECT_EQ(e->candidates()[2].coordinate, NativeReadoutCoordinate::r);
    EXPECT_EQ(e->endpoint_address(2, true), f.val->address(0, 500, 0));
    EXPECT_EQ(e->input_handle(), f.view);
    EXPECT_EQ(e->val_snapshot_handle(), f.val);
    EXPECT_EQ(e->protection_handle(), f.protection);
    EXPECT_DOUBLE_EQ(f.parent->network(0).value(NativeReadoutCoordinate::x, 400, 0), 40);
    const auto decision = RtcSpikeLearningDecision::consider(e, f.val, 9);
    EXPECT_FALSE(decision->requires_pair_exclusion_from_mapmaking(0));
    EXPECT_EQ(decision->candidate_disposition(0), RtcSpikeCandidateDisposition::event_assessment_required);
}

TEST(rtc_spike_learn, threshold_is_inclusive_in_both_signs_without_tail_probability) {
    const double cutoff = 5 * 1.4826;
    for (double sign : {-1.0, 1.0}) {
        for (double value : {std::nextafter(cutoff, 0.0), cutoff, std::nextafter(cutoff, 100.0)}) {
            Input input(601);
            input.x(0,0) = 0;
            input.x(1,0) = sign * value;
            for (Eigen::Index i = 2; i < input.x.rows(); ++i)
                input.x(i,0) = input.x(i-1,0) + std::array{-1.0, 0.0, 1.0}[i % 3];
            Fixture f(input);
            const auto e = f.learn();
            ASSERT_TRUE(e->blocks()[0].coordinates[0].available());
            EXPECT_DOUBLE_EQ(e->blocks()[0].coordinates[0].scale, 1.4826);
            EXPECT_EQ(std::count_if(e->candidates().begin(), e->candidates().end(),
                [](const auto &c) { return c.coordinate == NativeReadoutCoordinate::x && c.later_row == 101; }),
                value >= cutoff ? 1 : 0);
        }
    }
}

TEST(rtc_spike_learn, median_difference_removes_linear_drift_without_changing_originals) {
    Input base;
    base.x(300,0) += 40;
    Input drifting = base;
    for (Eigen::Index i = 0; i < drifting.x.rows(); ++i) drifting.x(i,0) += 100 + 3*i;
    const auto a = Fixture(base).learn(), b = Fixture(drifting).learn();
    EXPECT_DOUBLE_EQ(b->blocks()[0].coordinates[0].center - a->blocks()[0].coordinates[0].center, 3);
    EXPECT_DOUBLE_EQ(b->blocks()[0].coordinates[0].scale, a->blocks()[0].coordinates[0].scale);
    ASSERT_EQ(a->candidates().size(), b->candidates().size());
    for (std::size_t i=0; i<a->candidates().size(); ++i) {
        EXPECT_EQ(a->candidates()[i].later_row, b->candidates()[i].later_row);
        EXPECT_DOUBLE_EQ(a->candidates()[i].absolute_score, b->candidates()[i].absolute_score);
    }
}

TEST(rtc_spike_learn, minimum_support_and_zero_scale_exclude_pair_with_local_causes) {
    for (std::size_t rows : {256U, 257U}) {
        Fixture f{Input(rows)};
        const auto e=f.learn();
        EXPECT_EQ(e->blocks()[0].coordinates[0].admitted_differences, rows-1);
        EXPECT_EQ(e->blocks()[0].pair_screening_available(), rows==257);
        EXPECT_EQ(RtcSpikeLearningDecision::consider(e,f.val,1)->requires_pair_exclusion_from_mapmaking(0), rows==256);
    }
    Input input(600,2);
    input.r.col(0).setZero();
    Fixture f(input);
    const auto e=f.learn();
    ASSERT_EQ(e->blocks().size(), 2U);
    EXPECT_TRUE(e->blocks()[0].coordinates[0].available());
    EXPECT_EQ(e->blocks()[0].coordinates[1].cause, RtcSpikeNoiseCause::zero_scale);
    EXPECT_TRUE(std::isnan(e->blocks()[0].coordinates[1].scale));
    const auto d=RtcSpikeLearningDecision::consider(e,f.val,1);
    EXPECT_TRUE(d->requires_pair_exclusion_from_mapmaking(0));
    EXPECT_FALSE(d->requires_pair_exclusion_from_mapmaking(1));
}

TEST(rtc_spike_learn, declared_invalid_is_excluded_before_nonfinite_payload_is_read) {
    Input input;
    input.x(300,0)=std::numeric_limits<double>::quiet_NaN();
    input.xs[300]=NativeReadoutCoordinateState::measured(true,false,true,false);
    const auto e=Fixture(input).learn();
    EXPECT_TRUE(e->blocks()[0].pair_screening_available());
    EXPECT_EQ(e->blocks()[0].coordinates[0].excluded_differences, 2U);
    EXPECT_EQ(e->blocks()[0].coordinates[0].admitted_differences, 597U);
    EXPECT_TRUE(e->candidates().empty());
}

TEST(rtc_spike_learn, quantized_zero_mad_with_egregious_jump_requires_exclusion_not_false_clearance) {
    Input input;
    for (Eigen::Index i=0;i<input.x.rows();++i)
        input.x(i,0)=std::array{0.0,1.0,0.0,-1.0}[i%4];
    input.x(300,0)+=4000;
    Fixture f(input);
    const auto e=f.learn();
    EXPECT_EQ(e->blocks()[0].coordinates[0].cause,RtcSpikeNoiseCause::zero_scale);
    EXPECT_TRUE(RtcSpikeLearningDecision::consider(e,f.val,1)->requires_pair_exclusion_from_mapmaking(0));
    EXPECT_TRUE(e->candidates().empty());
    EXPECT_DOUBLE_EQ(f.parent->network(0).value(NativeReadoutCoordinate::x,400,0),4000);
}

TEST(rtc_spike_learn, native_ingress_rejects_admitted_nonfinite_and_rtc_overflow_fails_without_coercion) {
    for (bool overflow : {false,true}) {
        Input input;
        input.x(300,0)=overflow ? std::numeric_limits<double>::max() : std::numeric_limits<double>::infinity();
        input.x(301,0)= -std::numeric_limits<double>::max();
        if (!overflow) {
            EXPECT_THROW(input.freeze(),std::invalid_argument);
            continue;
        }
        const auto e=Fixture(input).learn();
        EXPECT_FALSE(e->blocks()[0].coordinates[0].available());
        EXPECT_TRUE(e->blocks()[0].coordinates[1].available());
        EXPECT_TRUE(e->candidates().empty());
        EXPECT_TRUE(std::isnan(e->blocks()[0].coordinates[0].center));
        EXPECT_NE(static_cast<unsigned>(e->blocks()[0].coordinates[0].cause) &
            static_cast<unsigned>(overflow ? RtcSpikeNoiseCause::arithmetic_nonfinite : RtcSpikeNoiseCause::admitted_nonfinite),0U);
    }
}

TEST(rtc_spike_learn, time_blocks_cross_edges_but_native_gaps_restart_noise_support) {
    Input input(3000,1,0.0078125); // 128 Hz: exact block boundary at row 1280.
    for (Eigen::Index i=1280;i<input.x.rows();++i) input.x(i,0)+=50;
    Fixture f(input);
    const auto e=f.learn();
    ASSERT_EQ(e->blocks().size(),3U);
    EXPECT_EQ(e->blocks()[1].first,1380);
    EXPECT_EQ(e->blocks()[1].coordinates[0].admitted_differences,1280U);
    ASSERT_EQ(e->candidates().size(),1U);
    EXPECT_EQ(e->candidates()[0].later_row,1380);
    EXPECT_EQ(e->candidates()[0].noise_block_index,1U); // later endpoint owns edge
    EXPECT_EQ(RtcSpikeLearningDecision::consider(e,f.val,1)->candidate_disposition(0),
              RtcSpikeCandidateDisposition::event_assessment_required); // may be a level shift
    for (std::size_t i=1280;i<input.counters.size();++i) ++input.counters[i];
    const auto gap=Fixture(input).learn();
    EXPECT_TRUE(gap->candidates().empty());
    EXPECT_EQ(gap->blocks()[1].run_first,1380);
    EXPECT_EQ(gap->blocks()[1].time_block_index,0U);
    EXPECT_EQ(gap->blocks()[1].coordinates[0].admitted_differences,1279U);
}

TEST(rtc_spike_learn, partition_schedules_cannot_change_physical_populations_or_evidence) {
    Input input(3000);
    input.x(1220,0)+=40;
    Fixture f(input);
    const auto whole=f.learn();
    std::vector<std::shared_ptr<const NativePairedReadoutView>> parts;
    for (TimestreamNativeRow row=100;row<3100;row+=127)
        parts.push_back(NativePairedReadoutView::admit(f.parent, {{0,row,std::min(row+127,TimestreamNativeRow{3100})}}));
    const auto split=learn_rtc_spike_candidates_partitioned(f.view,parts,f.val,f.protection,71);
    EXPECT_EQ(std::vector(whole->candidates().begin(),whole->candidates().end()),
              std::vector(split->candidates().begin(),split->candidates().end()));
    ASSERT_EQ(whole->blocks().size(),split->blocks().size());
    for (std::size_t i=0;i<whole->blocks().size();++i) {
        EXPECT_EQ(whole->blocks()[i].first,split->blocks()[i].first);
        EXPECT_EQ(whole->blocks()[i].coordinates[0].admitted_differences,split->blocks()[i].coordinates[0].admitted_differences);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(whole->blocks()[i].coordinates[0].scale),
                  std::bit_cast<std::uint64_t>(split->blocks()[i].coordinates[0].scale));
    }
    EXPECT_LE(split->peak_scratch_differences(),1221U);
    EXPECT_LT(split->logical_owned_bytes(),3000*sizeof(double));
    EXPECT_THROW(learn_rtc_spike_candidates(parts[0],f.val,f.protection,1),std::invalid_argument);
    parts.pop_back();
    EXPECT_THROW(learn_rtc_spike_candidates_partitioned(f.view,parts,f.val,f.protection,1),std::invalid_argument);
}

TEST(rtc_spike_learn, source_protection_changes_required_assessment_only_and_unknown_is_not_outside) {
    Input input;
    input.x(300,0)+=40;
    for (auto state : {RtcSpikeProtection::outside_source,RtcSpikeProtection::protected_source,RtcSpikeProtection::unavailable}) {
        Fixture f(input,RtcSpikeProtection::outside_source,{{0,0,400,401,state}});
        const auto e=f.learn();
        ASSERT_EQ(e->candidates().size(),2U);
        EXPECT_DOUBLE_EQ(e->blocks()[0].coordinates[0].scale,1.4826);
        EXPECT_EQ(e->blocks()[0].coordinates[0].admitted_differences,599U);
        EXPECT_EQ(e->candidates()[0].protection,state);
        EXPECT_EQ(e->candidates()[1].protection,state);
        const auto d=RtcSpikeLearningDecision::consider(e,f.val,1);
        EXPECT_EQ(d->candidate_disposition(0), state==RtcSpikeProtection::outside_source ?
            RtcSpikeCandidateDisposition::event_assessment_required : state==RtcSpikeProtection::protected_source ?
            RtcSpikeCandidateDisposition::protected_optical_assessment_required : RtcSpikeCandidateDisposition::source_protection_unavailable);
    }
}

TEST(rtc_spike_learn, protection_and_snapshot_identity_fail_closed) {
    Fixture f{Input{}}, foreign{Input{}};
    EXPECT_THROW(learn_rtc_spike_candidates(f.view,foreign.val,f.protection,1),std::invalid_argument);
    EXPECT_THROW(learn_rtc_spike_candidates(f.view,f.val,foreign.protection,1),std::invalid_argument);
    EXPECT_THROW(RtcSpikeSourceProtection::admit(f.parent,"a",RtcSpikeProtection::outside_source,
        {{0,0,200,300,RtcSpikeProtection::protected_source},{0,0,250,400,RtcSpikeProtection::unavailable}}),std::invalid_argument);
    EXPECT_THROW(RtcSpikeSourceProtection::admit(f.parent,"",RtcSpikeProtection::outside_source),std::invalid_argument);
    const auto e=f.learn();
    EXPECT_THROW(RtcSpikeLearningDecision::consider(e,ValSnapshot::initial(f.parent),1),std::invalid_argument);
    ValDeltaBuilder b(f.val,{ValProducer::align,5});
    b.propose(f.val->address(0,100,0),ValFactCode{8},ValFactState{1},ValFactCause{1});
    const auto later=ValSnapshot::commit(b.freeze());
    EXPECT_THROW(learn_rtc_spike_candidates(f.view,later,f.protection,1),std::invalid_argument);
    EXPECT_THROW(RtcSpikeLearningDecision::consider(e,later,1),std::invalid_argument);
}

TEST(rtc_spike_learn, noise_failure_publication_uses_exact_original_coordinate_and_retains_snapshot) {
    Input input;
    input.r.setZero();
    Fixture f(input);
    const auto e=f.learn();
    const auto subject=ValNativeRealization::create(f.parent,{ValProducer::align,3},1,ValNativeProductRole::original_input,0);
    const std::array subjects{subject};
    auto delta=e->noise_failure_delta(f.val,subjects);
    ASSERT_EQ(delta.findings().size(),600U);
    const auto key=delta.findings()[0].key();
    ASSERT_NE(key.native_target(),nullptr);
    EXPECT_EQ(key.native_target()->coordinate(),NativeReadoutCoordinate::r);
    EXPECT_EQ(key.native_target()->realization_handle(),subject);
    EXPECT_EQ(key.product(),(ValProducerProductIdentity{ValProducer::rtc,71}));
    EXPECT_EQ(f.val->find(key),nullptr);
    const auto committed=ValSnapshot::commit(std::move(delta));
    EXPECT_NE(committed->find(key),nullptr);
    EXPECT_EQ(f.val->find(key),nullptr);
    EXPECT_THROW(e->noise_failure_delta(committed,subjects),std::invalid_argument);
    const std::array residual{ValNativeRealization::create(f.parent,{ValProducer::rtc,3},2,ValNativeProductRole::derived_residual,0)};
    EXPECT_THROW(e->noise_failure_delta(f.val,residual),std::invalid_argument);
}

TEST(rtc_spike_learn, full_view_of_a_clipped_occurrence_axis_cannot_reanchor_a_physical_run) {
    const auto original=Input{}.freeze();
    const auto &network=original->network(0);
    const auto &axis=network.occurrence_axis();
    const auto occurrences=axis.occurrences().subspan(10,500);
    const auto clipped=std::make_shared<const NativePairedReadoutOccurrenceAxis>(
        axis.native_timing_handle(),110,std::vector(occurrences.begin(),occurrences.end()));
    std::vector<NativePairedReadoutNetwork> networks;
    networks.push_back(NativePairedReadoutNetwork::admit(clipped,support::detector_axis(0,1),
        network.mapping_authority_handle(),network.values(NativeReadoutCoordinate::x).middleRows(10,500),
        network.values(NativeReadoutCoordinate::r).middleRows(10,500),support::valid_states(500),support::valid_states(500)));
    const auto parent=support::make_observation(std::move(networks),{0});
    const auto protection=RtcSpikeSourceProtection::admit(parent,"source",RtcSpikeProtection::outside_source);
    EXPECT_THROW(learn_rtc_spike_candidates(NativePairedReadoutView::full(parent),ValSnapshot::initial(parent),protection,1),std::invalid_argument);
}

TEST(rtc_spike_learn, injected_events_on_fluctuating_background_are_candidates_without_a_clipping_quota) {
    Input input(4000);
    std::mt19937_64 random(713901);
    // Use an explicit bounded noise generator so the population is identical
    // across standard-library normal_distribution implementations.
    for (Eigen::Index i=0;i<input.x.rows();++i) {
        const double noise=static_cast<double>(random()%2001)/1000.0-1;
        input.x(i,0)=noise+30*std::sin(2*std::numbers::pi*i*0.008192/40);
    }
    for (Eigen::Index i=50;i<3950;i+=25) input.x(i,0)+=30; // >3%, no quota
    Fixture f(input);
    const auto e=f.learn();
    for (Eigen::Index i=50;i<3950;i+=25) {
        EXPECT_TRUE(std::any_of(e->candidates().begin(),e->candidates().end(),
            [i](const auto &c) { return c.coordinate==NativeReadoutCoordinate::x && c.later_row==100+i; }));
    }
    EXPECT_GT(e->candidates().size(),0.03*input.x.rows());
}

TEST(rtc_spike_learn, compact_evidence_measurement_does_not_copy_heavy_parent_planes) {
    Input input(16384,16);
    Fixture f(input);
    const auto started=std::chrono::steady_clock::now();
    const auto e=f.learn();
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    EXPECT_EQ(e->input_handle()->parent_handle(),f.parent);
    EXPECT_EQ(e->blocks().size(),14U*16);
    EXPECT_TRUE(e->candidates().empty());
    EXPECT_LE(e->peak_scratch_differences(),1221U);
    EXPECT_LT(e->logical_owned_bytes(),16384U*16*sizeof(double)/10);
    std::cout << "RTC_SPIKE_LEARN_MEASUREMENT synthetic_rows=16384 detectors=16 seconds=" << seconds
              << " owned_bytes=" << e->logical_owned_bytes()
              << " peak_scratch_differences=" << e->peak_scratch_differences() << '\n';
}

std::shared_ptr<const AstScanMotionNetworkViews> motion(const Fixture &f, bool available=true) {
    Eigen::VectorXd t(1001),ra(1001),dec=Eigen::VectorXd::Zero(1001);
    for (Eigen::Index i=0;i<t.size();++i) {
        t(i)=999.0+0.020*i;
        ra(i)=25.0*(t(i)-999)*std::numbers::pi/(180*3600);
    }
    AstScanMotionSourceMetadata metadata{AstScanMotionProducerKind::real_toltec,"Science","Lissajous",1,
        2000.0,0,50.0,AstScanMotionFieldRegistry::source_ra_act_source_dec_act_j2000_radians,"test-motion-v1"};
    if (!available) metadata.scan_file_valid=0;
    const auto source=AstScanMotionSource::admit(f.parent->scope(),f.parent->scope(),0,metadata,t,ra,dec);
    const auto raw=build_ast_scan_motion_product(source,{11,12,13,14});
    return AstScanMotionNetworkViews::admit(f.parent->scope(),raw,{f.parent->network(0).occurrence_axis().native_timing_handle()});
}

TEST(rtc_spike_optical_reference, sampled_average_has_unit_dc_sinc_attenuation_and_free_phase) {
    Input input;
    input.x(300,0)+=40;
    Fixture f(input,RtcSpikeProtection::protected_source);
    const auto e=f.learn();
    const auto ref=RtcSpikeOpticalReference::bind(e,0,motion(f),RtcSpikeBeamArray::a1100,"detector-association:0");
    EXPECT_EQ(ref->evidence_handle(),e);
    EXPECT_EQ(ref->readout_model_id,"rtc-native-readout-uniform-average-assumption-v1");
    EXPECT_TRUE(ref->conditional_on_readout_assumption);
    EXPECT_EQ(ref->sampled_harmonic(true,0,0),std::complex<double>(1,0));
    EXPECT_NEAR(std::abs(ref->sampled_harmonic(true,0,std::numbers::pi/2)-std::complex<double>(0,1)),0,1e-15);
    const auto interval=f.parent->network(0).occurrence_axis().occurrence(400).integration_support;
    const double width=interval.end_unix_sec-interval.begin_unix_sec;
    EXPECT_NEAR(std::abs(ref->sampled_harmonic(true,1/width,0)),0,1e-15);
    EXPECT_NEAR(std::abs(ref->sampled_harmonic(true,0.5/width,0)),2/std::numbers::pi,1e-15);
    EXPECT_NEAR(std::abs(ref->sampled_harmonic(true,10,0.77)),std::abs(ref->sampled_harmonic(true,10,0)),1e-15);
    EXPECT_EQ(ref->sampled_harmonic(true,-10,-0.77),std::conj(ref->sampled_harmonic(true,10,0.77)));
    EXPECT_THROW(ref->sampled_harmonic(true,std::numeric_limits<double>::infinity(),0),std::invalid_argument);
    const auto scale=ref->local_scale(true);
    ASSERT_TRUE(scale);
    EXPECT_NEAR(scale->speed_arcsec_per_sec,25,1e-7);
    EXPECT_NEAR(scale->airy_fwhm_arcsec,4.6786413788,1e-10);
    ASSERT_TRUE(scale->crossing_fwhm_sec);
    EXPECT_NEAR(*scale->crossing_fwhm_sec,4.6786413788/25,1e-9);
    EXPECT_EQ(RtcSpikeLearningDecision::consider(e,f.val,1)->candidate_disposition(0),
              RtcSpikeCandidateDisposition::protected_optical_assessment_required);
}

TEST(rtc_spike_optical_reference, missing_or_foreign_motion_never_becomes_a_speed_bound) {
    Input input;
    input.x(300,0)+=40;
    Fixture f(input,RtcSpikeProtection::protected_source), foreign(input,RtcSpikeProtection::protected_source);
    const auto e=f.learn();
    EXPECT_THROW(RtcSpikeOpticalReference::bind(e,0,motion(foreign),RtcSpikeBeamArray::a1100,"detector-association:0"),std::invalid_argument);
    const auto ref=RtcSpikeOpticalReference::bind(e,0,motion(f,false),RtcSpikeBeamArray::a1100,"detector-association:0");
    EXPECT_FALSE(ref->local_scale(true));
    Fixture outside(input);
    EXPECT_THROW(RtcSpikeOpticalReference::bind(outside.learn(),0,motion(outside),RtcSpikeBeamArray::a1100,"detector-association:0"),std::invalid_argument);
}

} // namespace

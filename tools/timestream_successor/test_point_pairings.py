import numpy as np
import pytest
from point_pairings import balanced_partitions,footprints,normalized_sum,measured,compared,population_summary,population_comparison
from point_pairings_io import replay_scans,scheduled_native_rows,validate_processing_intervals


def test_native_science_intervals_are_projected_to_actual_RTC_lattice():
    np.testing.assert_array_equal(scheduled_native_rows([[33,39]],2,0),[34,36,38])
    np.testing.assert_array_equal(scheduled_native_rows([[32,38]],2,1),[33,35,37])
    np.testing.assert_array_equal(scheduled_native_rows([[32,35],[40,43]],2,0),[32,34,40,42])
    with pytest.raises(ValueError):scheduled_native_rows([[0,3]],0,0)


def test_processing_intervals_reject_off_lattice_endpoint_changes_and_ordering():
    segments=[dict(scan=0,native_interval=[32,38]),dict(scan=1,native_interval=[38,44])]
    p=dict(scans=[dict(scan=s['scan'],science_native=[s['native_interval'].copy()]) for s in segments])
    assert validate_processing_intervals(p,segments,50)==[0,1]
    p['scans'][0]['science_native'][0][1]=37
    with pytest.raises(ValueError,match='bound PTC'):validate_processing_intervals(p,segments,50)
    segments=[dict(scan=0,native_interval=[38,44]),dict(scan=1,native_interval=[32,38])]
    p=dict(scans=[dict(scan=s['scan'],science_native=[s['native_interval'].copy()]) for s in segments])
    with pytest.raises(ValueError,match='nonchronological'):validate_processing_intervals(p,segments,50)
from point_empirical import gaussian
from validate_point_empirical import fixture


@pytest.mark.parametrize('count,expected',[(4,3),(6,10)])
def test_all_balanced_partitions_are_unique_complete_complements(count,expected):
    scans=tuple(range(count));pairs=balanced_partitions(scans)
    assert len(pairs)==expected
    assert len({frozenset((frozenset(a),frozenset(b))) for a,b in pairs})==expected
    for a,b in pairs:
        assert len(a)==len(b)==count//2 and not set(a)&set(b)
        assert set(a)|set(b)==set(scans) and 0 in a
    assert (tuple(range(0,count,2)),tuple(range(1,count,2))) in pairs


@pytest.mark.parametrize('scans',[(0,1,2),(0,0),()])
def test_bad_partition_domain_is_rejected(scans):
    with pytest.raises(ValueError):balanced_partitions(scans)


def test_subset_map_sums_numerators_and_uses_own_support():
    q=np.array([[[1.,0.],[1.,1.]],[[9.,100.],[9.,9.]]]);n=q*np.array([2.,8.])[:,None,None]
    together=normalized_sum(n,q,[0,1],0.)
    assert together['signal'][0,0]==pytest.approx(7.4)
    assert together['signal'][0,0] != pytest.approx((2+8)/2)
    first=normalized_sum(n,q,[0],.5)
    assert not first['support'][0,1] and np.isnan(first['signal'][0,1])
    assert np.all(first['signal'][first['support']]==2)


def test_missing_fit_periphery_withholds_fit_without_changing_compact_measurement():
    h,x,y,p=fixture(81);regions,t,major,x,y=footprints(x.shape,h,p)
    signal=gaussian(p,x,y);q=np.ones(x.shape)*100
    a=normalized_sum(signal[None]*q,q[None],[0],.5)
    q2=q.copy();hole=tuple(np.argwhere(regions['fit_periphery'])[0]);q2[hole]=0
    b=normalized_sum(signal[None]*q2,q2[None],[0],.5)
    ma=measured(a,regions,t,h,6,p,major,x,y);mb=measured(b,regions,t,h,6,p,major,x,y)
    assert ma['fit']['status']=='success' and ma['fit']['fit_pixels']==1257
    assert mb['fit']['status']=='unavailable'
    assert ma['fixed_template']['amplitude']==pytest.approx(mb['fixed_template']['amplitude'])
    _,_,comparison=compared(a,b,ma,mb,regions,t)
    assert not comparison['fit_comparison_available']
    assert comparison['fixed_template_A_minus_B']['amplitude']==pytest.approx(0.)
    assert comparison['local_coverage']['fit']['common_supported_pixels']==1256


def test_source_hole_withholds_both_measurements_and_local_ratios_include_depth():
    h,x,y,p=fixture(81);regions,t,major,x,y=footprints(x.shape,h,p);signal=gaussian(p,x,y)
    q=np.ones(x.shape)*100;q2=q.copy()*2;hole=tuple(np.argwhere(regions['core'])[0]);q2[hole]=0
    a=normalized_sum(signal[None]*q,q[None],[0],.5);b=normalized_sum(signal[None]*q2,q2[None],[0],.5)
    ma=measured(a,regions,t,h,6,p,major,x,y);mb=measured(b,regions,t,h,6,p,major,x,y)
    _,_,c=compared(a,b,ma,mb,regions,t)
    assert c['fixed_template_A_minus_B'] is None and not c['fit_comparison_available']
    assert c['local_coverage']['fit']['ratio_QA_QB_positive_pixels']==[.5]*5
    assert mb['apertures'][0]['available'] is False


def test_saved_replay_preserves_slots_detector_occurrences_and_exclusions(tmp_path):
    import yaml,hashlib
    native=tmp_path/'native';ptc=native/'donor-continuity/ptc';ptc.mkdir(parents=True);cal=ptc.parent/'cal';cal.mkdir();root=tmp_path/'map';root.mkdir()
    times=np.arange(4,dtype='<f8')+1771482887;times.tofile(native/'native-time.f64')
    def save(p,obj):p.write_text(yaml.safe_dump(obj))
    save(native/'receipt.yaml',dict(native_time_sha256=hashlib.sha256((native/'native-time.f64').read_bytes()).hexdigest()))
    processing=dict(generation='test',absolute_epoch_uncertainty='unknown',native_outside_processing=[],scans=[dict(scan=i,science_slots=[2*i,2*i+2],science_native=[[2*i,2*i+2]]) for i in range(2)])
    save(native/'processing-scans.yaml',processing);save(cal/'receipt.yaml',dict(detectors=[dict(channel=7),dict(channel=9)]))
    save(ptc/'receipt.yaml',dict(segments=[dict(scan=i,converged=True,native_interval=[2*i,2*i+2]) for i in range(2)]))
    pixels=np.array([4,4,-1,4,5,5],dtype='<i4');pixels.tofile(root/'operator-pixel.i32');plan={'ordered_column_domain':[],'request_identity':'fixture'}
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    save(root/'bundle.yaml',dict(array='fixture'))
    save(root.parent/'input-manifest.yaml',{str(native/n):sha(native/n) for n in ['receipt.yaml','native-time.f64']})
    save(root.parent/'attempt.yaml',dict(request_identity='fixture',products={'fixture':sha(root/'bundle.yaml')},input_manifest_sha256=sha(root.parent/'input-manifest.yaml')))
    vals=[]
    for i in range(2):
        stem=f'segment-{i}';cause=np.array([[0,0],[1,0]],dtype='u1');v=np.array([[2,4],[99,6]],dtype='<f8')+i
        cause.tofile(ptc/(stem+'-causes.u8'));v.tofile(ptc/(stem+'-cleaned.f64'));np.arange(2*i,2*i+2,dtype='<i8').tofile(ptc/(stem+'-slots.i64'));vals.extend(v[cause==0])
        segment=dict(scan=i,network=0,scheduled_times=2,output_detectors=2,output_detector_indices=[0,1],files={stem+'-cleaned.f64':'fixture'})
        pub=dict(network=0,source_segment=segment,source_PTC_root=str(ptc),source_CAL_root=str(cal),processing='test',detector_occurrences=['occ7','occ9'],RTC_native_phase=0,RTC_factor=1,identity=f'pub{i}',rank=1)
        save(root/f'pub{i}.yaml',pub);plan['ordered_column_domain'].append(dict(publication_file=f'pub{i}.yaml',first_input_row=i*3,input_rows=3))
    counts=np.zeros(9,dtype='i8');numerator=np.zeros(9);use=pixels>=0;np.add.at(counts,pixels[use],1);np.add.at(numerator,pixels[use],np.array(vals)[use])
    full=dict(PRIMARY=np.zeros((3,3)),HITS=counts.reshape(3,3),NORMALIZATION=counts.reshape(3,3),NUMERATOR=numerator.reshape(3,3));regions={'core':np.ones((3,3),bool)}
    n,q,ids,audit=replay_scans(root,plan,full,regions,dict(processing_generation='test'))
    assert ids==[0,1];np.testing.assert_array_equal(q.sum(axis=0),full['HITS'])
    assert audit['members'][0]['region_counts']['core']==[1,1]
    assert audit['members'][0]['map_rejected_per_output_detector']==[0,1]
    assert audit['members'][0]['cause_code_counts_per_output_detector']['1']==[1,0]
    pop=population_summary(audit['members'],[0,1],regions)
    assert pop['core']['0']['contributions']==5 and pop['core']['0']['contributing_detectors']==2
    comparison=population_comparison(population_summary(audit['members'],[0],regions),population_summary(audit['members'],[1],regions))
    assert comparison['core']['detectors_both']==2
    assert comparison['core']['networks']['0']['contributions_A']==2
    assert comparison['core']['networks']['0']['contributions_B']==3
    np.array([0,2],dtype='<i8').tofile(ptc/'segment-0-slots.i64')
    with pytest.raises(ValueError,match='slots'):replay_scans(root,plan,full,regions,dict(processing_generation='test'))
    np.array([0,1],dtype='<i8').tofile(ptc/'segment-0-slots.i64')
    (times+100).tofile(native/'native-time.f64')
    save(native/'receipt.yaml',dict(native_time_sha256=sha(native/'native-time.f64')))
    with pytest.raises(ValueError,match='digest differs'):replay_scans(root,plan,full,regions,dict(processing_generation='test'))

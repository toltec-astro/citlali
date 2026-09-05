"""Fixed EL-F13 constructions, passed and byte-registered before real scoring."""
import copy
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest
from scipy.special import j1

import scan_agreement as sa

LIBRARY=Path(os.environ.get('EL_F13_LIBRARY',Path(__file__).with_name('scan_agreement_stream.dylib')))
RECORD=struct.Struct('<8q5d')
CASES={
    'retention_helpful':dict(complement=10.,target=0.,reference=(0.,0.),expected=['pass','pass'],errors=(10.,20/3,5.)),
    'deletion_helpful':dict(complement=0.,target=10.,reference=(0.,0.),expected=['insufficient_room']*2),
    'references_disagree':dict(complement=10.,target=0.,reference=(0.,10.),expected=['insufficient_room']*2),
    'zero_effect':dict(complement=10.,target=10.,reference=(0.,0.),expected=['no_gain']*2),
    'below_floor':dict(complement=.05,target=0.,reference=(0.,0.),expected=['insufficient_room']*2),
    'common_wrong_feature':dict(complement=0.,target=10.,reference=(10.,10.),expected=['pass','pass'],truth=0.,expected_limitation='agreement does not identify truth'),
}


@pytest.fixture(scope='session',autouse=True)
def compiled_stream_for_tests(tmp_path_factory):
    """Normal pytest works without leaving an untracked binary in the repo."""
    global LIBRARY
    if 'EL_F13_LIBRARY' not in os.environ:
        LIBRARY=tmp_path_factory.mktemp('el_f13_stream')/'scan_agreement_stream.dylib'
        subprocess.run(['clang++','-O2','-std=c++17','-ffp-contract=off','-fno-fast-math',
                        '-shared','-fPIC',str(Path(__file__).with_name('scan_agreement_stream.cpp')),
                        '-o',str(LIBRARY)],check=True)
    assert LIBRARY.is_file()


def marker(s):return RECORD.pack(0,s,0,0,0,0,0,0,0,0,0,0,0)


def record(s,uid,a,row,col,sample,signal,c=1.,kernel=0):
    return RECORD.pack(1,s,uid,a,row,col,kernel,sample,signal*c,c,c*c,signal,1.)


def key(uid=101,a=0,s=0):return dict(observation='123424',array=a,uid=uid,scan=s,disposition='eligible')


def constant_spool(case, renumber=0):
    rows=[]
    for s in range(12):
        rows.append(marker(s))
        for a in range(3):
            for which in range(2):
                uid=100+10*a+which+renumber
                signal=(case['target'] if which else case['complement']) if s==0 else (999. if which else case['reference'][s%2])
                for pixel in range(1024): rows.append(record(s,uid,a,pixel//32,pixel%32,pixel,signal))
    return b''.join(rows)


def run_stream(data,shape,kernels,keys):
    st=sa.Stream(LIBRARY,shape,kernels,keys)
    for i in range(0,len(data),104*317):st.feed(data[i:i+104*317])
    st.finish(len(data)//104,len(data)//104-12)
    return st


@pytest.mark.parametrize('name',list(CASES))
@pytest.mark.parametrize('geometry',['pixel','signed_jinc'])
def test_declared_constant_constructions(name,geometry):
    case=CASES[name]; data=constant_spool(case)
    bank=[[(np.ones((1,1)),np.ones((1,1)))]]*3
    if geometry=='signed_jinc':bank=geometry_fixture()[1]
    st=run_stream(data,(32,32),bank,[key(101+10*a,a) for a in range(3)])
    try:
        for a in range(3):
            models=[sa.finalize(st.plane(3,a,j),np.ones((32,32))) for j in range(3)]
            memberships=sa.reference_scans(0,st.scan_order());refs=[]
            for members in memberships:
                raw,counts=sa.combine([st.plane(1,a,s) for s in members]);refs.append(sa.finalize(raw,counts))
            footprint=models[0]['support']&models[0]['conditioned']
            out,domain,reasons=sa.assess(models,refs,footprint,memberships)
            assert [x['status'] for x in out['probes']]==case['expected']
            assert domain.sum()>=256 and not reasons.any()
            if name=='retention_helpful':
                for j,e in enumerate(case['errors']):
                    interval=sa.interval(models[j],refs[0],domain)
                    with localcontext() as ctx:
                        ctx.prec=60
                        expected=Decimal(10)/(1+Decimal(j)/2)
                    assert Decimal(interval['lower'])<=expected<=Decimal(interval['upper'])
    finally:st.close()


def constant_raw(n,c=1.,q=1.,shape=(32,32),terms=1):
    raw=np.zeros((9,*shape));raw[0]=n;raw[1]=c;raw[2]=q
    raw[3]=abs(n);raw[4]=abs(c);raw[5]=abs(q);raw[6]=terms;raw[7]=1;raw[8]=terms
    return raw


def simple_models():
    models=[sa.finalize(constant_raw(n,c,q),np.ones((32,32))) for n,c,q in ((10,1,1),(10,1.5,1.25),(10,2,2))]
    refs=[sa.finalize(constant_raw(0,4,4,terms=4),np.full((32,32),4)) for _ in range(2)]
    return models,refs


@pytest.mark.parametrize('change,expected',[
    ('missing_scan','unavailable_reference_scans'),('small_overlap','unavailable_overlap'),
    ('denominator','unavailable_overlap'),('lost_probe','unavailable_probe_support'),
    ('pixel_scan_count','unavailable_overlap'),('empty','no_contribution')])
def test_unavailable(change,expected):
    models,refs=simple_models();footprint=np.ones((32,32),bool);members=[[2,4],[1,3]]
    if change=='missing_scan':members[0]=[2]
    if change=='small_overlap':refs[0]['support'][:5]=False
    if change=='denominator':models[0]=sa.finalize(constant_raw(1,1e-9,1),np.ones((32,32)))
    if change=='lost_probe':models[1]['support'][0,0]=False
    if change=='pixel_scan_count':refs[0]['scan_count'][:5]=1
    if change=='empty':footprint[:]=False
    out,_,_=sa.assess(models,refs,footprint,members)
    assert out['probes'][0]['status']==expected


def test_nonfinite_gamma_and_threshold_boundaries():
    raw=constant_raw(1);raw[0,0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):sa.finalize(raw,np.ones((32,32)))
    with pytest.raises(ValueError,match='gamma'):sa.gamma(1/sa.U)
    models,refs=simple_models()
    # Exactly 10% nominal gain cannot pass conservative intervals at the boundary.
    models[1]=sa.finalize(constant_raw(9),np.ones((32,32)))
    out,_,_=sa.assess(models,refs,np.ones((32,32),bool),[[2,4],[1,3]])
    assert out['probes'][0]['status']=='no_gain'
    for baseline in (.1,np.nextafter(.1,0),np.nextafter(.1,np.inf)):
        model=sa.finalize(constant_raw(baseline),np.ones((32,32)))
        value=sa.interval(model,refs[0],np.ones((32,32),bool))
        assert value['lower']<=baseline<=value['upper']
    weights=np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.])
    assert sa.threshold(weights,.1)==.8


def geometry_fixture():
    banks=[]
    for a in range(3):
        bank=[]
        for phase in (0.,.375):
            y,x=np.mgrid[-2:3,-2:3];r=(2.3+.2*a)*np.hypot(x-phase,y+.125)
            k=2*j1(r)/r
            assert (k<0).any()
            bank.append((k,k*k))
        banks.append(bank)
    data=[]; decoded=[]
    for s in range(12):
        data.append(marker(s))
        for a in range(3):
            for uid in (100+10*a,101+10*a):
                for sample,(row,col) in enumerate(((0,0),(2,2),(4,4))):
                    signal=2+.125*row-.25*col+(uid%2)*.5
                    blob=record(s,uid,a,row,col,sample,signal,1.3,sample%2)
                    data.append(blob);decoded.append(RECORD.unpack(blob))
    return b''.join(data),banks,decoded


def test_signed_clipped_subpixel_jinc_noninteger_square_and_high_precision():
    data,banks,decoded=geometry_fixture()
    st=run_stream(data,(5,5),banks,[key(101+10*a,a) for a in range(3)])
    try:
        with localcontext() as ctx:
            ctx.prec=65
            for a in range(3):
                for probe,f in enumerate((0.,.5,1.)):
                    expected=[[[Decimal(0) for _ in range(5)] for _ in range(5)] for _ in range(3)]
                    for rec in decoded:
                        _,s,uid,arr,row,col,ki,_,ns,cs,qs,_,_=rec
                        if s!=0 or arr!=a:continue
                        factor=Decimal(f if uid==101+10*a else 1.)
                        for r in range(max(0,row-2),min(5,row+3)):
                            for c in range(max(0,col-2),min(5,col+3)):
                                kr,kc=r-row+2,c-col+2
                                for j,scale in enumerate((ns,cs,qs)):
                                    kval=banks[a][ki][j==2][kr,kc]
                                    expected[j][r][c]+=Decimal(float(kval))*Decimal(scale)*(factor**(2 if j==2 else 1))
                    raw=st.plane(3,a,probe);fin=sa.finalize(raw,np.ones((5,5)))
                    for j in range(3):
                        for r in range(5):
                            for c in range(5):
                                assert abs(Decimal(float(raw[j,r,c]))-expected[j][r][c])<=Decimal(float(fin['bounds'][j,r,c]))
                    for r,c in zip(*np.where(fin['conditioned'])):
                        exact=expected[0][r][c]/expected[1][r][c]
                        # The bound is for raw N/C; finalization deliberately zeros
                        # pixels below provisional coverage. Do not test those zeros
                        # against a raw-ratio expectation.
                        actual=float(raw[0,r,c]/raw[1,r,c])
                        assert abs(Decimal(actual)-exact)<=Decimal(float(fin['signal_error'][r,c]))
                # Half squares its multiplier in Q; full observation sums scan maps.
                assert np.allclose(st.plane(3,a,1)[2],st.plane(3,a,0)[2]+.25*st.plane(4,a)[2],rtol=2e-15,atol=1e-15)
                combined=np.zeros((3,5,5))
                for s in st.scan_order():combined+=st.plane(0,a,s)[:3]
                assert np.array_equal(combined,st.plane(2,a)[:3])
    finally:st.close()


def test_signed_cancellation_against_decimal():
    values=[1e16,1.,-1e16,3.]
    raw=constant_raw(sum(values),c=4,q=4,terms=4);raw[3]=sum(abs(x) for x in values)
    fin=sa.finalize(raw,np.ones((32,32)))
    exact=sum(Decimal(x) for x in values)/4
    assert abs(Decimal(float(fin['signal'][0,0]))-exact)<=Decimal(float(fin['signal_error'][0,0]))


@pytest.mark.parametrize('damage',['short','badkind','nonfinite','multiplier','duplicate','sample','array','kernel'])
def test_corrupt_spool(damage):
    bank=[[(np.ones((1,1)),np.ones((1,1)))]]*3
    st=sa.Stream(LIBRARY,(2,2),bank,[])
    good=record(0,100,0,0,0,0,1.)
    bad=list(RECORD.unpack(good))
    if damage=='short':data=marker(0)+good[:-1]
    elif damage=='duplicate':data=marker(0)+marker(0)
    elif damage=='sample':data=marker(0)+good+good
    else:
        if damage=='badkind':bad[0]=2
        if damage=='nonfinite':bad[8]=float('nan')
        if damage=='multiplier':bad[12]=.5
        if damage=='array':bad[3]=4
        if damage=='kernel':bad[6]=5
        data=marker(0)+RECORD.pack(*bad)
    try:
        with pytest.raises(ValueError):st.feed(data)
    finally:st.close()


def test_multiple_keys_uid_renumbering_and_full_reference_exclusion():
    case=CASES['retention_helpful'];bank=[[(np.ones((1,1)),np.ones((1,1)))]]*3
    outcomes=[]
    for offset in (0,70000):
        keys=[key(101+offset,0,0),key(100+offset,0,2)]
        st=run_stream(constant_spool(case,offset),(32,32),bank,keys)
        try:
            assert len(keys)==2
            for s in range(12):assert not st.plane(1,0,s).any()
            outcomes.append([st.plane(3,i,j).copy() for i in range(2) for j in range(3)])
        finally:st.close()
    assert all(np.array_equal(x,y) for x,y in zip(*outcomes))


def test_poison_unused_scores_and_boundary_access(tmp_path):
    text='SCI-FRUIT-EL-F12-STATE-R0.1+CAP-001\nH 1\n0\n1\n"123424" 2 101 0 eligible 1 1 1 .2 8 1024 1024 0 0\n0\n0\n'
    poisoned=text.replace('1 1 1 .2 8 1024 1024 0 0','FUTURE INJECTED ORACLE 999 NaN 0 -1 100 100')
    assert sa.decode_census(text,1)==sa.decode_census(poisoned,1)
    records=[dict(path=str(tmp_path/f'current_{i}')) for i in range(6)]
    for order in (records,list(reversed(records))):
        access=sa.BoundaryAccess(order)
        assert access.open_path(records[0]['path'],'test')==records[0]['path']
        with pytest.raises(ValueError,match='forbidden'):access.open_path(tmp_path/'future_injected','poison')


def test_exact_native_fits_representation():
    internal=np.arange(12,dtype=float).reshape(3,4)
    # Mirrors the explicit native writer loop, independent of the converter.
    written=np.array([[internal[r,internal.shape[1]-c-1] for c in range(4)] for r in range(3)])
    assert np.array_equal(sa.fits_orientation(internal),written)
    assert np.array_equal(sa.fits_orientation(written.reshape(1,1,3,4)),internal)


def register_fixtures(directory):
    directory=Path(directory);directory.mkdir()
    sa.write_json(directory/'EXPECTED_R0.1.json',CASES)
    for name,case in CASES.items():
        (directory/f'{name}.bin.gz').write_bytes(gzip.compress(constant_spool(case),mtime=0))
    data,banks,_=geometry_fixture();(directory/'signed_jinc.bin.gz').write_bytes(gzip.compress(data,mtime=0))
    sa.write_json(directory/'JINC_KERNELS_R0.1.json',[[[x.tolist() for x in pair] for pair in bank] for bank in banks])
    sa.write_json(directory/'FIXTURE_MANIFEST_R0.1.json',[sa.file_record(p) for p in sorted(directory.iterdir())])


if __name__=='__main__':
    import sys
    register_fixtures(sys.argv[1])

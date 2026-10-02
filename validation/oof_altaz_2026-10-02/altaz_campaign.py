"""Reuse the saved-OOF JINC runner; native AST supplies source-offset AltAz."""
import sys
from pathlib import Path
D=Path(__file__).parent;W=D.parents[1]
sys.path.insert(0,str(W/'validation/oof_jinc_2026-10-02'))
import campaign as c
from campaign import *
D=Path(__file__).parent
FK5=c.ROOT
ROOT=FK5.with_name('successor-oof-a2000-altaz-jinc-20261002')
c.ROOT=ROOT;c.D=D;c.PRIOR=[*c.PRIOR,FK5]
CLI=W/'build/bin/citlali'
FRAME='delta-source-AltAz'

def independent_ast_check(cfg,segment,xy):
    """Validation oracle only, never production coordinates or map deposition."""
    from netCDF4 import Dataset
    from astropy.table import Table
    manifest=Path(verified(cfg['manifest']));m=Table.read(manifest,format='ascii.ecsv')
    row=next(r for r in m if r['role']=='apt');aptfile=manifest.parent/row['relative_path']
    require(digest(aptfile)==str(row['transport_sha256']).removeprefix('sha256:'),'APT oracle binding')
    apt=Table.read(aptfile,format='ascii.ecsv');byuid={int(r['uid']):r for r in apt}
    nw=int(cfg['network']);role=PARENT/f"S-{cfg['observation']}";native=np.fromfile(role/f'upstream/products/network{nw}/native-time.f64',dtype='<f8')
    t=native[segment['native_phase']+np.array(segment['slots'])*segment['factor']]
    with Dataset(verified(cfg['telescope'])) as tel:
        tt=np.array(tel['Data.TelescopeBackend.TelTime'][:]);vals={}
        for k in ('TelAzAct','TelElAct','TelAzCor','TelElCor','SourceAz','SourceEl'):
            a=np.array(tel['Data.TelescopeBackend.'+k][:]);require(np.isfinite(a).all(),'oracle nonfinite telescope')
            if a.max()>1.99*np.pi and a.min()<np.pi:a=np.where(a<np.pi,a+2*np.pi,a)
            vals[k]=np.interp(t,tt,a)
    az=vals['TelAzAct']-vals['SourceAz'];az=np.where(az>.9*2*np.pi,az-2*np.pi,az)
    el=vals['TelElAct']-vals['TelElCor'];lon=np.cos(el)*az-vals['TelAzCor'];lat=vals['TelElAct']-vals['SourceEl']-vals['TelElCor']
    effective=yread(verified(cfg['effective_config']));entries=[x for x in effective['inputs'] if x['meta']['name']==f"{cfg['observation']}_0_1"]
    ast=next(x for x in entries[0]['cal_items'] if x['type']=='astrometry');offsets={x['axes_name']:x['value_arcsec'] for x in ast['pointing_offsets'] if 'axes_name' in x}
    require(len(offsets['az'])==len(offsets['alt'])==1,'oracle bounded constant pointing offsets')
    error=0.
    # All times, three detectors spanning each segment, independent ECSV join.
    for di in sorted({0,xy.shape[1]//2,xy.shape[1]-1}):
        uid=int(segment['detector_occurrences'][di].split(':row=')[-1].split(':')[0]);r=byuid[uid]
        gx=float(r['x_t']);gy=float(r['y_t'])
        if not np.isfinite([gx,gy]).all():require(np.isnan(xy[:,di,2:]).all(),'missing detector geometry');continue
        x=(lon+(np.cos(el)*gx-np.sin(el)*gy+offsets['az'][0])*np.pi/180/3600)*180/np.pi
        y=(lat+(np.cos(el)*gy+np.sin(el)*gx+offsets['alt'][0])*np.pi/180/3600)*180/np.pi
        error=max(error,float(np.max(np.abs(xy[:,di,2]-x))),float(np.max(np.abs(xy[:,di,3]-y))))
    require(error<1e-10,'independent AST convention mismatch')
    return dict(max_coordinate_error_arcsec=error*3600,checked_detectors=min(3,xy.shape[1]),scheduled_times=len(t),formula='historical delta-source AltAz; corrected elevation rotation; native exact times; no parallactic rotation')

def prepare():
    rev=revision();capacity();out=ROOT/'prepared'/rev;out.mkdir(parents=True,exist_ok=False)
    old=read(FK5/'PREPARED.json');closed=read(FK5/'CLOSEOUT.json');require(closed['state']=='ASSESSED_CLOSED','FK5 predecessor not closed')
    # Mutable build locator is replaced by the separately preserved executed binary.
    archived=FK5/'executed-jinc-binaries/citlali_jinc_diagnostic'
    require(digest(archived)==old['helper']['sha256'],'executed FK5 helper not preserved')
    refs=[bound(FK5/'PREPARED.json'),bound(FK5/'CLOSEOUT.json'),bound(archived),bound(D/'WORK_ORDER.md'),bound(CLI)]
    for b in old['bindings']:verified(b)
    refs.extend(old['bindings']);states=[]
    measurements=read(PARENT/'report/MEASUREMENTS.json')
    for obs,previous in zip(OBS,old['states']):
        require(obs==previous['observation'],'predecessor order');capacity()
        role=PARENT/f'S-{obs}';auth=SavedInputs(role);base=role/'M0/products/a2000';dest=out/str(obs);dest.mkdir()
        plan=auth.yaml(base/'plan.yaml');oldq=read(verified(previous['request']));oldgeom=np.fromfile(verified(oldq['geometry_payload']),dtype='<f8').reshape(-1,4)
        rr=next(x for x in measurements['observations'] if x['observation']==obs)
        with fits.open(verified(rr['reference'])) as f:
            h=f['signal_I'].header.copy()
        require((h['CTYPE1'],h['CTYPE2'],h['CUNIT1'],h['CUNIT2'],h['CDELT1'],h['CDELT2'],h['CRVAL1'],h['CRVAL2'])==('AZOFFSET','ELOFFSET','arcsec','arcsec',-1.,1.,0.,0.),'historical R grid differs')
        # Native export uses explicit degree coordinates; published WCS uses arcsec.
        wh={k:h[k] for k in ('CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2')};wh.update(CUNIT1='deg',CUNIT2='deg',CDELT1=-1/3600,CDELT2=1/3600)
        q=dict(schema='oof-a2000-altaz-ast-v1',observation=obs,array_index=2,WCS=wh,map_request=bound(auth.get(role/'inputs/map-request.json')),parent_plan=bound(auth.get(base/'plan.yaml')),archived_input_manifest=bound(auth.get(base/'input-manifest.yaml')))
        save_json(dest/'ast-request.json',q)
        dev=dict(schema='citlali-development-v1',terminal='map',input=bound(dest/'ast-request.json'),output=str(dest/'ast'))
        (dest/'development.yaml').write_text(yaml.safe_dump(dev,sort_keys=False))
        with (dest/'ast.log').open('w') as log:run=subprocess.run([str(CLI),str(dest/'development.yaml')],cwd=W,stdout=log,stderr=subprocess.STDOUT)
        require(run.returncode==0,'native AST export failed: '+str(dest/'ast.log'))
        require('[error]' not in (dest/'ast.log').read_text().lower(),'unexpected AST error output')
        ast=yread(dest/'ast/receipt.yaml');require(ast['frame']==FRAME and ast['observation']==f'obs:{obs}_0_1' and len(ast['segments'])==len(plan['ordered_column_domain']),'native AST scope')
        xyall=[];segchecks=[]
        for e,s in zip(plan['ordered_column_domain'],ast['segments']):
            pub=auth.yaml(base/e['publication_file'],e['publication_sha256']);source=pub['source_segment'];nt=source['scheduled_times'];nd=source['output_detectors'];stem=s['name'].split('network-')[-1].split('-segment-');nw=int(stem[0]);si=int(stem[1]);root=Path(pub['source_PTC_root'])
            require(s['source_segment']==source and s['detector_occurrences']==pub['detector_occurrences'] and nw==pub['network'],'native AST segment/detector join')
            slots=np.fromfile(auth.get(root/f'segment-{si}-slots.i64',source['files'][f'segment-{si}-slots.i64']),dtype='<i8')
            require(np.array_equal(slots,s['slots']) and s['factor']==pub['RTC_factor'] and s['native_phase']==pub['RTC_native_phase'],'native AST RTC schedule join')
            causes=np.fromfile(auth.get(root/f'segment-{si}-causes.u8',source['files'][f'segment-{si}-causes.u8']),dtype='u1').reshape(nt,nd)
            xy=np.fromfile(verified(s['coordinates']),dtype='<f8').reshape(nt,nd,4);cal=np.fromfile(verified(s['CAL_causes']),dtype='<u2').reshape(nt,nd)
            retained=causes==0;require((cal[retained]==0).all() and np.isfinite(xy[retained]).all() and int(retained.sum())==e['input_rows'],'new AST changes saved scientific admission')
            xyall.append(xy[retained,:2]);cfg=read(auth.get(role/f'inputs/network{nw}-input.json'));oracle=independent_ast_check(cfg,s,xy)
            segchecks.append(dict(name=s['name'],retained=int(retained.sum()),oracle=oracle,frequency_hz=s['frequency_hz']))
        xy=np.concatenate(xyall);require(len(xy)==len(oldgeom),'new AST occurrence count')
        # Exact historical grid is retained; fail rather than drop out-of-map centers.
        require(np.all((xy[:,0]>=.5)&(xy[:,0]<h['NAXIS1']+.5)&(xy[:,1]>=.5)&(xy[:,1]<h['NAXIS2']+.5)),'historical R grid excludes retained centers; bounded extent repair needed')
        geometry=oldgeom.copy();geometry[:,:2]=xy;geometry.astype('<f8').tofile(dest/'geometry.f64')
        require(np.array_equal(geometry[:,2:],oldgeom[:,2:]),'coefficients or frequencies changed')
        header={k:h[k] for k in ('NAXIS1','NAXIS2','CTYPE1','CTYPE2','CUNIT1','CUNIT2','CRVAL1','CRVAL2','CRPIX1','CRPIX2','CDELT1','CDELT2')}
        header.update(OBSNUM=obs,SUBOBS=0,SCAN=1,ASTFRAME=FRAME,MAPMAKER='JINC',BUNIT='mJy/beam',ARRAY='a2000',NOMBFWHM=10.)
        save_json(dest/'header.json',header)
        request=dict(oldq);request.update(schema='oof-a2000-altaz-jinc-diagnostic-v1',frame=FRAME,geometry_payload=bound(dest/'geometry.f64'),spec=dict(oldq['spec'],rows=h['NAXIS2'],columns=h['NAXIS1']))
        save_json(dest/'request.json',request)
        state=dict(previous,request=bound(dest/'request.json'),header=bound(dest/'header.json'),AST=bound(dest/'ast/receipt.yaml'),frame=FRAME,reference=rr['reference'],coordinate_checks=segchecks,unchanged_value_payload=oldq['value_payload'],same_samples=True,grid_policy='exact historical R WCS, all retained centers inside',native_AST_source=rev)
        states.append(state);refs.extend(auth.used.values());refs.append(rr['reference'])
        print(obs,'native AltAz/identity/oracle checked; exact R grid encloses all',len(xy),'samples',flush=True)
    refs={b['path']:b for b in refs}
    for p in out.rglob('*'):
        if p.is_file():refs[str(p)]=bound(p)
    for b in refs.values():verified(b)
    require(revision()==rev,'source changed during preparation')
    save_json(ROOT/'PREPARED.json',dict(source_revision=rev,helper=bound(HELPER),states=states,bindings=list(refs.values()),storage=capacity(),observational_accumulations=0,RTC_CAL_PTC_executions=0))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','freeze','check','run']);args=a.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='check':check();print('Frozen AltAz launch ready; no maps launched.')
    else:getattr(c,args.action)()

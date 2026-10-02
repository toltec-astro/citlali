"""Saved products only; run after manual resume of the three-map batch."""
from altaz_campaign import *
import assess_saved as common
import matplotlib.pyplot as plt

def main():
    p=read(ROOT/'PREPARED.json');s=read(ROOT/'STATUS.json');freeze=read(ROOT/'FREEZE.json')
    require(s['state']=='COMPLETE_PENDING_OWNER_REVIEW' and len(s['completed'])==3,'three finished maps required')
    for b in p['bindings']+freeze['bindings']:verified(b)
    verified(p['helper']);out=ROOT/'assessment';out.mkdir(exist_ok=False)
    rows=[];maps=[]
    for obs,state,done in zip(OBS,p['states'],s['completed']):
        b=read(verified(done['bundle']));q=read(verified(state['request']));r=yread(verified(b['receipt']))
        require(b['frame']==r['frame']==FRAME and b['parent']==state and r['request_sha256']==state['request']['sha256'],'map frame or parent mismatch')
        dest=ROOT/str(obs)
        for name,h in r['files'].items():require(digest(dest/name)==h,'changed native output')
        verified(b['diagnostics']);ref=state['reference']
        with fits.open(verified(b['map'])) as f:
            h=f[0].header;shape=f[0].data.shape;v=f['LOCAL_VALID'].data.astype(bool)
            require(h['ASTFRAME']==FRAME and h['CTYPE1']=='AZOFFSET' and h['CTYPE2']=='ELOFFSET' and h['CUNIT1']==h['CUNIT2']=='arcsec','FITS frame')
            m=np.fromfile(dest/'m.f64',dtype='<f8').reshape(shape);ca=np.fromfile(dest/'causes.u8',dtype='u1').reshape(shape)
            require(np.array_equal(v,ca==0) and np.array_equal(f[0].data[v],m[v]),'native signal/support fidelity')
            err=np.fromfile(dest/'error-4.f64',dtype='<f8').reshape(shape);tol=np.fromfile(dest/'tolerance-lower.f64',dtype='<f8').reshape(shape)
            require(np.all(err[v]<=tol[v]),'numerical certificate')
        R=common.load_map(ref,'R');S=common.load_map(b['map'],'JINC')
        require(R[0].shape==S[0].shape and np.array_equal(R[2],S[2]) and np.array_equal(R[3],S[3]),'R/S coordinate grid differs')
        row=dict(observation=obs,R=ref,S=b['map'],metrics={k:{str(span):common.metrics(*m,span/2) for span in (100,32)} for k,m in [('R',R),('S',S)]},frame=FRAME,numerical_error_max=float(err[v].max()),same_saved_samples=state['same_samples'])
        rows.append(row);maps.append((R,S))
    figures=[]
    for norm,half,name in [(False,50,'raw-100arcsec'),(True,50,'normalized-100arcsec'),(True,16,'normalized-32arcsec')]:
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        for j,(obs,pair,row) in enumerate(zip(OBS,maps,rows)):
            for i,(label,m) in enumerate(zip(('R','S'),pair)):
                common.panel(axes[i,j],m,f'{obs} | '+('Historical R' if label=='R' else 'Successor cold M0')+' | AltAz 1″ JINC',half,row['metrics'][label]['32']['peak'] if norm else None)
                axes[i,j].set_xlabel('source-offset azimuth (arcsec)');axes[i,j].set_ylabel('source-offset elevation (arcsec)')
        fig.suptitle('Same source-offset AltAz coordinates; no fitted registration or rescaling\n'+('Each image normalized by its own central32″ peak' if norm else 'Common brightness scale, mJy / nominal beam'))
        cb=plt.cm.ScalarMappable(norm=common.Normalize(-.3,1) if norm else common.Normalize(-100,560),cmap='RdBu_r');fig.colorbar(cb,ax=axes,label='peak fraction' if norm else 'mJy / nominal beam',shrink=.7)
        path=out/(name+'.png');fig.savefig(path,dpi=150);plt.close(fig);figures.append(bound(path))
    save_json(out/'MEASUREMENTS.json',dict(executed_source=p['source_revision'],analysis_source=revision(),observations=rows,figures=figures,limitations=['historical preprocessing/weights differ','r0.3 kernel29x29 versus historical27x27','signed-C cancellation at edges is not a coverage cut','no optical inference or absolute calibration qualification']))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()

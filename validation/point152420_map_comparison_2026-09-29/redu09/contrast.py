"""Owner-requested additional saved reduction; reuse the frozen comparison fit."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import compare as c
import numpy as np,yaml
from astropy.io import fits
RAW=Path('/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/pointings/reduced/redu09/152420/raw')
MAP=RAW/'toltec_commissioning_a1100_pointing_152420_citlali.fits'
PRIOR=HERE.parent/'results/metrics.json'

def main():
 out=HERE/'results';out.mkdir(exist_ok=False)
 inputs=[c.bound(p) for p in (MAP,RAW/'index.yaml',PRIOR)]
 previous=json.loads(PRIOR.read_text())
 for b in previous['inputs']+[previous['script'],previous['solver']]:assert c.bound(Path(b['path']))['sha256']==b['sha256']
 with fits.open(MAP) as f:
  assert f[0].header['OBSNUM0']=='152420' and f['signal_I'].header['BUNIT']=='mJy/beam'
  recipe={k:f[0].header[k] for k in ('FRUITLOOPS_ITER','CONFIG.FRUITLOOPS','CONFIG.CLEANED.NEIG','CONFIG.CLEANED.MODESEL','CONFIG.WEIGHT.TYPE','CONFIG.DOWNSAMPLED','GROUPING','JINC_R','JINC_A','JINC_B','JINC_C')}
 index=yaml.load((RAW/'index.yaml').read_text(),Loader=yaml.CSafeLoader)
 entries=[v for v in index.get('products',[]) if isinstance(v,dict) and v.get('name')==MAP.name]
 if entries and 'sha256' in entries[0]:assert entries[0]['sha256']==inputs[0]['sha256']
 d=c.load(MAP,True);native=c.fit(d);binned=c.fit(c.bin2(d));core=c.fit(d,15.,native['parameters'][1:3],native['parameters'])
 records=dict(redu09_native=native,redu09_bin2=binned,redu09_core15=core)
 comparison={k:previous['fits'][k] for k in ('old_native','old_bin2','new_native')};comparison.update(records)
 ratios={k:dict(redu09_over_amplitude=native['amplitude']/r['amplitude'],redu09_over_major=native['fwhm_major_minor_arcsec'][0]/r['fwhm_major_minor_arcsec'][0]) for k,r in previous['fits'].items() if k in ('old_native','old_bin2','new_native')}
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
 for col,(name,title,data) in enumerate([('old_native','Earlier V2 · no FRUIT · 1″',c.load(c.OLD,True)),('redu09_native','redu09 · FRUIT index 4 · 1″',d),('new_native','Current M0 · no FRUIT · 2″',c.load(c.NEW,False))]):
  r=comparison[name];p=np.array(r['parameters']);s,x,y=(data[k] for k in ('signal','x','y'));model=c.gaussian(p,x,y)
  dx=x[0,1]-x[0,0];dy=y[1,0]-y[0,0];extent=[x[0,0]-dx/2,x[0,-1]+dx/2,y[0,0]-dy/2,y[-1,0]+dy/2]
  for row,values in enumerate((s-p[6],s-model)):
   ax=axes[row,col];im=ax.imshow(np.where(data['support'],values,np.nan),origin='lower',extent=extent,cmap='RdBu_r',vmin=-150,vmax=700 if row==0 else 150)
   if row==0:ax.contour(x,y,model-p[6],levels=[p[0]/2],colors='black',linewidths=1.2)
   ax.plot(0,0,'+',color='gray');ax.plot(*p[1:3],'x',color='black',ms=6)
   ax.add_patch(plt.Circle((0,0),40,fill=False,color='gray',ls='--',lw=.8))
   ax.set(xlim=(40,-40),ylim=(-40,40),aspect='equal',xlabel='native x offset (arcsec)',ylabel='native y offset (arcsec)')
   fig.colorbar(im,ax=ax,shrink=.8,label='mJy / beam')
  a,b=r['fwhm_major_minor_arcsec'];axes[0,col].set_title(f'{title}\nA = {p[0]:.1f} mJy/beam; FWHM {a:.2f}″ × {b:.2f}″');axes[1,col].set_title('Data − Gaussian and constant')
 fig.suptitle('POINT 152420 · a1100 · common 40″ free-ellipse fit\nNative Alt-Az (left/middle) and FK5 tangent (right); no sky registration. Different processing stages.',fontsize=13)
 fig.savefig(out/'contrast.png',dpi=160);plt.close(fig)
 assert inputs==[c.bound(p) for p in (MAP,RAW/'index.yaml',PRIOR)]
 for b in previous['inputs']:assert c.bound(Path(b['path']))['sha256']==b['sha256']
 report=dict(schema='point152420-redu09-contrast-v1',inputs=inputs,script=c.bound(Path(__file__)),helper=c.bound(HERE.parent/'compare.py'),solver=previous['solver'],recipe=recipe,producer_index={k:v for k,v in index.items() if k not in ('products','files/dirs')},producer_index_sha256_binding_available=bool(entries and 'sha256' in entries[0]),map_header=d['header'],fits_checksums=d['checksums'],fits=records,ratios=ratios,inputs_unchanged=True,previous_fits_not_rerun=True,limitations=previous['limitations']+['redu09 is FRUIT-enabled iteration index4, unlike prior V2 and successor M0; not a controlled FRUIT or RTC/PTC recovery experiment.','Fitted amplitude is peak above the fitted constant, not integrated flux.'])
 (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(recipe=recipe,fits=records,ratios=ratios,index_sha_binding=report['producer_index_sha256_binding_available']),indent=2))
if __name__=='__main__':main()

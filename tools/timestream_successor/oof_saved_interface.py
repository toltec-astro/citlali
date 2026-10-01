"""Checked, lossless AltAz saved-map attachment to LMTOOF's existing DataGrid.

This is an interface fixture, not inversion admission. No recentering, interpolation,
flux normalization, source gate or noise estimate is performed. FK5 is rejected.
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from inspect_forward_map import require
from point_fruit_trajectory import bound


def embed_altaz(signal, support, header):
    signal=np.asarray(signal,float);support=np.asarray(support,bool)
    require(signal.ndim==2 and signal.shape==support.shape,'map/support shape mismatch')
    require(header['CTYPE1'] in ('AZOFFSET','AZ-OFFSET') and header['CTYPE2'] in ('ELOFFSET','EL-OFFSET'),
            'LMTOOF requires AltAz offsets; FK5 needs sample-level AST projection, not a fixed map rotation')
    require(header['BUNIT']=='mJy/beam','explicit physical brightness unit required')
    scales=[];axes=[]
    for i,n in enumerate((signal.shape[1],signal.shape[0]),1):
        unit=header[f'CUNIT{i}'];require(unit in ('deg','arcsec'),'unsupported angular unit')
        factor=3600. if unit=='deg' else 1.
        scale=header[f'CDELT{i}']*factor
        require(np.isfinite(scale) and scale!=0,'invalid angular spacing')
        scales.append(abs(scale))
        axes.append(((np.arange(n)+1-header[f'CRPIX{i}'])*header[f'CDELT{i}']+header[f'CRVAL{i}'])*factor)
    require(scales[0]==scales[1],'LMTOOF requires equal axis scales')
    # No skew/rotation/distortion is representable by the existing BaseGrid reader.
    for k in header:
        if k.startswith(('CD1_','CD2_','PC1_','PC2_')):
            require(float(header[k])==(1. if k[-3]==k[-1] and k.startswith('PC') else 0.),'rotated/skewed WCS unsupported')
    x,y=[a/scales[0] for a in axes]
    require(np.max(abs(x-np.rint(x)))<1e-10 and np.max(abs(y-np.rint(y)))<1e-10,
            'non-integer center would require scientific resampling')
    x,y=np.rint(x).astype(int),np.rint(y).astype(int)
    extent=max(abs(x).max(),abs(y).max());n=1
    while n//2<=extent:n*=2
    require(n<=4096,'unbounded FFT attachment')
    out=np.full((n,n),np.nan);valid=support&np.isfinite(signal)
    out[np.ix_(y+n//2,x+n//2)]=np.where(valid,signal,np.nan)
    require(np.isfinite(out).sum()==valid.sum(),'support changed during embedding')
    return out,scales[0],valid


def attach_historical(path, destination, *, requested_focus_mm, pupil_stop_m):
    """pupil_stop_m is a named MODEL operand, never inferred from absent OOF_RS."""
    path=Path(path);destination=Path(destination)
    require(not destination.exists(),'preserve previous attachment')
    with fits.open(path) as h:
        ph=h[0].header.copy();hdr=h['signal_I'].header.copy();z=np.squeeze(h['signal_I'].data)
        weight=np.squeeze(h['weight_I'].data);coverage=np.squeeze(h['coverage_bool_I'].data)
        require(z.ndim==2 and z.shape==weight.shape==coverage.shape,'historical support dimensions')
        require(np.isin(coverage,[0,1]).all(),'coverage mask is not binary')
        # Versioned historical operational coverage: not successor SCI_SUPPORT or precision.
        valid=(coverage==1)&np.isfinite(weight)&(weight>0)&np.isfinite(z)
        out,scale,valid=embed_altaz(z,valid,hdr)
        for k in ('OOF_W','OOF_ID','OOF_T','OOF_RO','OOF_RI','OOF_M2X','OOF_M2Y','OOF_M2Z','OOF_RMS'):
            require(k in ph and np.isfinite(ph[k]),'missing explicit OOF header: '+k)
        require(ph['OOF_ID']==1100 and ph['OOF_W']==.0011,'declared a1100 nominal wavelength differs')
        require(abs(ph['OOF_M2Z']-requested_focus_mm*1000)<1e-8,'focus unit/sign differs from requested physical M2 Z')
        require(np.isfinite(pupil_stop_m) and 0<pupil_stop_m<=ph['OOF_RO'],'explicit model pupil stop invalid')
        n=out.shape[0];target=fits.Header()
        for i,c in ((1,'AZ-OFFSET'),(2,'EL-OFFSET')):
            target[f'CTYPE{i}']=c;target[f'CUNIT{i}']='deg';target[f'CDELT{i}']=scale/3600.;target[f'CRPIX{i}']=n//2+1;target[f'CRVAL{i}']=0.
        for k in ('OOF_W','OOF_ID','OOF_T','OOF_RO','OOF_RI','OOF_M2X','OOF_M2Y','OOF_M2Z','OOF_RMS'):target[k]=ph[k]
        target['OOF_RS']=pupil_stop_m;target['BUNIT']='mJy/beam';target['ADAPMODE']='INTERFACE-TEST';target['NORMSCL']=1.
        target['HISTORY']='Lossless axis permutation/integer embedding; no source centering, cropping or brightness scaling.'
        destination.parent.mkdir(parents=True,exist_ok=True);fits.PrimaryHDU(out,target).writeto(destination)
        return dict(source=bound(path),attachment=bound(destination),input_shape=list(z.shape),shape=list(out.shape),supported_pixels=int(valid.sum()),input_finite_pixels=int(np.isfinite(z).sum()),angular_pixel_arcsec=scale,wavelength_m=ph['OOF_W'],wavelength_status='nominal recorded array wavelength; no measured bandpass inference',focus_microns=ph['OOF_M2Z'],focus_convention='absolute requested M2Z, mm*1000; no best-focus subtraction or sign flip',normalization=1.,pointing_shift_arcsec=[0.,0.],support='historical coverage_bool AND finite positive weight AND finite signal; NaN elsewhere; not covariance',OOF_RMS=ph['OOF_RMS'],uncertainty='historical diagnostic header only; not new empirical/independent noise or scalar-sigma admission',pupil_stop_m=pupil_stop_m,pupil_status='explicit interface-test model operand; Python frontend22.5 versus C++DishParams22.0 remains a model decision',solver_inversion_admitted=False)

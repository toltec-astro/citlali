"""Describe a central POINT source without modifying the science MAP.

An unweighted elliptical Gaussian plus constant is fitted to finite science
pixels within 40 arcsec of the AST center. This is a descriptive measured width,
not a recovered flux, detection significance, PSF, or transfer calibration.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares
from inspect_forward_map import read_yaml, digest, require


def gaussian(parameters, x, y):
    amp, cx, cy, sx, sy, angle, background = parameters
    dx, dy = x-cx, y-cy
    u = np.cos(angle)*dx + np.sin(angle)*dy
    v = -np.sin(angle)*dx + np.cos(angle)*dy
    return background + amp*np.exp(-.5*((u/sx)**2+(v/sy)**2))


def measure(signal, support, header, nominal):
    yy, xx = np.indices(signal.shape)
    x = (xx+1-header['CRPIX1'])*header['CDELT1']*3600
    y = (yy+1-header['CRPIX2'])*header['CDELT2']*3600
    radius = np.hypot(x,y)
    good = support & np.isfinite(signal)
    roi = good & (radius <= 40)
    central = good & (radius <= 20)
    if roi.sum() < 30 or central.sum() < 10:
        raise ValueError('insufficient central science support for POINT description')
    peak_index = np.argmax(np.where(central, signal, -np.inf))
    background = float(np.median(signal[roi]))
    peak = float(signal.flat[peak_index])
    if peak <= background:
        raise ValueError('no positive central peak above local median')
    initial = [peak-background, x.flat[peak_index], y.flat[peak_index],
               nominal/np.sqrt(8*np.log(2)), nominal/np.sqrt(8*np.log(2)), 0, background]
    lower = [0, -20, -20, .5, .5, -np.pi, -np.inf]
    upper = [np.inf, 20, 20, 20, 20, np.pi, np.inf]
    fit = least_squares(lambda p: gaussian(p,x[roi],y[roi])-signal[roi],initial,
                        bounds=(lower,upper),x_scale='jac',max_nfev=3000,
                        method='trf',jac='2-point',loss='linear',f_scale=1.,
                        tr_solver='exact',tr_options={},diff_step=None,verbose=0,
                        ftol=1e-12,xtol=1e-12,gtol=1e-12)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('POINT descriptive Gaussian fit did not converge')
    p=fit.x
    fwhm=np.sort(p[3:5]*np.sqrt(8*np.log(2)))[::-1]
    center_pixel=[p[1]/(header['CDELT1']*3600)+header['CRPIX1'],
                  p[2]/(header['CDELT2']*3600)+header['CRPIX2']]
    sky=WCS(header).all_pix2world([center_pixel],1)[0]
    result=dict(method='unweighted elliptical Gaussian plus constant; 40 arcsec radius around AST center',
        search_and_centroid_domain='20 arcsec around AST center; positive-amplitude descriptive fit',
        nominal_calibration_beam_fwhm_arcsec=nominal,
        central_pixel_peak_mJy_per_nominal_beam=peak,
        fitted_amplitude_above_constant_mJy_per_nominal_beam=float(p[0]),
        fitted_constant_mJy_per_nominal_beam=float(p[6]),
        centroid_tangent_east_north_arcsec=[float(p[1]),float(p[2])],
        centroid_FK5_J2000_ra_dec_deg=sky.tolist(),
        measured_fwhm_major_minor_arcsec=fwhm.tolist(),
        measured_fwhm_geometric_mean_arcsec=float(np.sqrt(np.prod(fwhm))),
        fit_pixels=int(roi.sum()),fit_residual_rms_mJy_per_nominal_beam=float(np.sqrt(np.mean(fit.fun**2))),
        fit_constraint_active=bool(np.any(fit.active_mask)),
        interpretation='first-pass source morphology only; no flux recovery, uncertainty, detection significance or full-chain transfer claim')
    return result,p,x,y


def inspect_point(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    records=[]
    for ax,array,nominal in zip(axes,('a1100','a1400','a2000'),(5.,7.5,10.)):
        bundle=read_yaml(root/array/'bundle.yaml')
        require(bundle['schema']=='citlali-forward-map-bundle-v1' and bundle['outcome']=='success' and
                bundle['observation']=='obs:152389_0_2' and bundle['array']==array and
                bundle['nominal_beam_fwhm_arcsec']==nominal,'POINT bundle observation/array/beam mismatch')
        require(digest(root/array/'map.fits')==bundle['files']['map.fits'],'POINT MAP content differs from bundle')
        with fits.open(root/array/'map.fits',memmap=False,checksum=True) as hdus:
            require(all(h.verify_checksum()==1 and h.verify_datasum()==1 for h in hdus),'POINT FITS checksum failure')
            signal=np.array(hdus[0].data);header=hdus[0].header.copy()
            support=hdus['SCI_SUPPORT'].data.astype(bool)
        if header['RADESYS']!='FK5' or header['EQUINOX']!=2000:
            raise ValueError('POINT description requires the selected FK5 J2000 frame')
        result,p,x,y=measure(signal,support,header,nominal)
        result.update(array=array,MAP_product_identity=bundle['product_identity'],map_sha256=bundle['files']['map.fits'])
        records.append(result)
        window=support & np.isfinite(signal) & (np.abs(x)<=45) & (np.abs(y)<=45)
        limit=float(np.max(np.abs(signal[window])))
        dx,dy=header['CDELT1']*3600,header['CDELT2']*3600
        extent=[x[0,0]-dx/2,x[0,-1]+dx/2,y[0,0]-dy/2,y[-1,0]+dy/2]
        im=ax.imshow(signal,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.contour(x,y,gaussian(p,x,y)-p[6],levels=[p[0]/2],colors='black',linewidths=.8)
        ax.plot(0,0,'+',color='gray',ms=11,mew=1.2,label='AST center')
        ax.plot(p[1],p[2],'x',color='black',ms=6,label='Gaussian centroid')
        ax.set(xlim=(45,-45),ylim=(-45,45),xlabel='East offset (arcsec)',ylabel='North offset (arcsec)')
        widths=result['measured_fwhm_major_minor_arcsec']
        ax.set_title(f'{array}\nMeasured FWHM {widths[0]:.1f} × {widths[1]:.1f} arcsec')
        fig.colorbar(im,ax=ax,shrink=.78,label='mJy / nominal beam')
    axes[0].legend(loc='upper left',fontsize=7)
    fig.suptitle('POINT 152389 · 1146+399 · first forward successor MAP\nFK5 J2000 · signed signal · no spatial mask · ALS rank 10',fontsize=12)
    fig.savefig(output/'central-source.png',dpi=180);plt.close(fig)
    report=dict(schema='citlali-point-source-description-v1',observation=152389,arrays=records,
                full_procedure_response='unavailable; actual source fit is not a calibrated injection-recovery test')
    (output/'source-measurements.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('maps',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(inspect_point(a.maps,a.output),indent=2))

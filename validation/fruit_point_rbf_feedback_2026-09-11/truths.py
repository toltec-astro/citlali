"""Independent pupil-diffraction PSFs; no RBF imports or fitted basis parameters."""
import numpy as np
from scipy import ndimage

PSF_CONFIG=dict(fft_size=2048,pupil_radius_pixels=64.,obscuration_radius_fraction=.30,
    fine_pixel_arcsec=.5,defocus_waves=.60,coma_waves=.30,main_translation_arcsec=[17.,-11.],sub_lattice_translation_arcsec=[2.,1.])

def optical_image(defocused=True):
    n=PSF_CONFIG['fft_size'];yy,xx=np.mgrid[-n//2:n//2,-n//2:n//2];u=xx/PSF_CONFIG['pupil_radius_pixels'];v=yy/PSF_CONFIG['pupil_radius_pixels'];r=np.hypot(u,v)
    mask=(r<=1)&(r>=PSF_CONFIG['obscuration_radius_fraction']);phase=np.zeros_like(r)
    if defocused:phase=2*np.pi*(PSF_CONFIG['defocus_waves']*(2*r*r-1)+PSF_CONFIG['coma_waves']*(3*r*r-2)*u)
    pupil=np.zeros(r.shape,complex);pupil[mask]=np.exp(1j*phase[mask]);wave=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    intensity=abs(wave)**2;intensity/=intensity.sum()*PSF_CONFIG['fine_pixel_arcsec']**2;return intensity

def sample_psf(image,x,y,translation=(17.,-11.)):
    out=np.zeros_like(x,dtype=float);n=image.shape[0];scale=PSF_CONFIG['fine_pixel_arcsec']
    # Integrate each 2-arcsec ordinary-grid pixel using sixteen fixed subpixels.
    for dx in [-.75,-.25,.25,.75]:
        for dy in [-.75,-.25,.25,.75]:
            coords=np.array([(y-translation[1]+dy)/scale+n//2,(x-translation[0]+dx)/scale+n//2])
            out+=ndimage.map_coordinates(image,coords,order=1,mode='constant',cval=0.,prefilter=False)/16
    return out

def map_truth(data,image,brightness=1.,translation=(17.,-11.)):
    z=sample_psf(image,data.x,data.ygrid,translation)
    # Equal integrated brightness to the old peak-100, FWHM 12x8 compact source.
    integral=100*2*np.pi*12*8/(8*np.log(2))
    return np.where(data.D,z[None,:]*integral*brightness,0.)

def concentration(z,D):
    F=float(np.sum(z[D])*4);power=float(np.dot(z[D],z[D])*4)
    return dict(integrated_brightness=F,effective_area_arcsec2=F*F/power if F>0 and power>0 else None)

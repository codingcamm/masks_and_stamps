import time
import numpy as np
from astropy.io import fits
import argparse

import healpy as hp
from pixell import enmap, utils, reproject, curvedsky, enplot
from maps import white_noise
from pixell import wcsutils, powspec, fft as enfft
from pixell.enmap import ndmap, fft, ifft
from astropy import wcs

# global Constants
TCMB    = 2.726 #Kelvin
TCMB_uK = 2.726e6 #micro-Kelvin
hplanck = 6.626068e-34 #MKS
kboltz  = 1.3806503e-23 #MKS
clight  = 299792458.0 #MKS

# path to websky data
path    = "/mnt/welch/USERS/cwhitaker/maps/websky/"

# generating coordinates
def arccos_arange(start, finish, steps):
    """
    Converts an arange array of angles (in degrees) to their arccosine values in
    radians.
    Returns:
    np.array of arccosine values in radians.
    """
    degrees_array = np.arange(start, finish, steps)
    radians_array = np.deg2rad(degrees_array)
    # (arccosine values will be between -1 and 1)
    cosine_values = np.cos(radians_array)
    y = np.arccos(cosine_values)
    return y

def generate_coords():
    """
    Generates an array of pair positions centered a step. Currently hard coded
    to be between
    (-69, 69) degrees in DEC to avoid pole distortions in maps and
    (-180, 180) degrees in RA
    Returns:
    Unique np.array of pair position coordinates.
    """

    # coordinates
    x = np.arange(0, 361 * utils.degree, 1.5 * utils.degree)
    y = arccos_arange(-69, 69, 1.5)

    # array of coordinates
    dec, ra = np.meshgrid(y, x)

    # flatten the meshgrid arrays and combine them into pairs of positions
    coords = np.array([dec.ravel(), ra.ravel()]).T
    coords = np.unique(coords, axis=0)
    return coords
    
# functions for map conversions
def dBnudT(nu_ghz):
    nu = 1.e9*np.asarray(nu_ghz)
    X = hplanck*nu/(kboltz*TCMB)
    return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK * 1e26

def ItoDeltaT(nu_ghz):
    return 1./dBnudT(nu_ghz)

# beam smoothing
def smooth_gauss(emap, sigma):
    """
    Smooth the map given as the first argument with a gaussian beam
    with the given standard deviation sigma in radians.
    """
    if np.all(sigma == 0): return emap.copy()
    f  = fft(emap)
    x2 = np.sum(emap.lmap()**2*sigma**2,0)
    if sigma >= 0: f *= np.exp(-0.5*x2)
    else:          f *= 1-np.exp(-0.5*x2)
    return enmap.enmap(ifft(f).real, emap.wcs)

def fwhm_to_sigma(fwhm):
    # Converts fwhm arcmins to sigma radians
    return fwhm / (2.*np.sqrt(2.*np.log(2.))) * (np.pi / (180.*60.))

shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(0.5 / 60))

# CMB
alm = hp.read_alm(path + 'lensed_alm.fits', hdu=(1, 2, 3))
cmb_map = curvedsky.alm2map(alm.astype(np.complex128)[0, :], enmap.empty(shape, wcs, dtype=np.float32))

# kSZ Effect
ksz_map        = hp.read_map(path + "ksz.fits")
npix           = ksz_map.size  # assuming single healpix map
nside          = hp.npix2nside(npix)
lmax           = 3 * nside
ksz_map        = reproject.healpix2map(ksz_map, shape=cmb_map.shape, wcs=cmb_map.wcs, lmax=lmax)

# tSZ Effect
tsz_map        = hp.read_map(path + "tsz_8192.fits")
npix           = tsz_map.size  # assuming single healpix map
nside          = hp.npix2nside(npix)
lmax           = 3 * nside
tsz_map        = reproject.healpix2map(tsz_map, shape=cmb_map.shape, wcs=cmb_map.wcs, lmax=lmax)

def create_websky_map(path:str, freq:str, noise=None, fwhm=None):
    """
    Generates a full-sky Websky map.
    
    Parameters
    ----------
    path : str
        Directory pointed to Websky map components
    freq: str
        Frequency label for the map ("090", "150", "220", etc.).
    noise: int, optional
        White noise level (in muK-arcmin) added to the map.
        ACT noise ~ 20 muK-arcmin
    fwhm: float, optional
        Beam full width at half maximum in arcminutes applied to the map.
        ACT fwhm @ 90 GHz = 2.2 arcmins
    """
    if 90 <= int(freq) <= 98:
        cib_map    = enmap.read_map(path + "cib_90.2_car.fits")
        cib_map    = enmap.resample(cib_map, cmb_map.shape)
        radio_map  = enmap.read_map(path + "/map_radio_0.5arcmin_f90.2.fits")
        radio_map  = enmap.resample(radio_map, cmb_map.shape)
        tsz_factor = -4.2840 * 1e6 # tSZ and CIB conversions for 90 GHz
        cib_factor = 2.5947 * 1e3
    elif 150 <= int(freq) <= 153:
        cib_map    = enmap.read_map(path + "cib_153_car.fits")
        cib_map    = enmap.resample(cib_map, cmb_map.shape)
        radio_map  = enmap.read_map(path + "/map_radio_0.5arcmin_f143.0.fits")
        radio_map  = enmap.resample(radio_map, cmb_map.shape)
        tsz_factor = -2.7685 * 1e6 # tSZ and CIB conversions for 150 GHz
        cib_factor = 4.6831 * 1e3
    elif 217 <= int(freq) <= 225:
        cib_map    = enmap.read_map(path + "cib_217_car.fits")
        cib_map    = enmap.resample(cib_map, cmb_map.shape)
        radio_map  = enmap.read_map(path + "/map_radio_0.5arcmin_f217.0.fits")
        radio_map  = enmap.resample(radio_map, cmb_map.shape)
        tsz_factor = 3.1517 * 1e5 # tSZ and CIB conversions for 220 GHz
        cib_factor = 2.0676 * 1e3
        #cib_225 = 2.0716 * 1e3
    else:
        raise ValueError(f"Frequency {int(freq)} GHz not supported. "
                         "Supported ranges: 90–98, 150–153, 217–225 GHz.")
        
    # adding optional noise
    if noise is None:
        noise_map = enmap.zeros(shape, wcs)
    elif noise == 0:
        raise ValueError(f"Noise must be greater than zero, omit noise parameter for noiseless map")
    else:
        noise_map = white_noise(shape, wcs, noise_muK_arcmin=noise)

    # combine maps
    websky_map = radio_map*ItoDeltaT(int(freq)) + cmb_map + ksz_map + tsz_map*tsz_factor + cib_map*cib_factor + noise_map
    
    if fwhm is not None:
        print(fwhm_to_sigma(fwhm))
        websky_map = smooth_gauss(websky_map, sigma=fwhm_to_sigma(fwhm))
    
    return websky_map

# running from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("freq", type=str)
    parser.add_argument("noise", type=int, nargs='?')
    parser.add_argument("fwhm", type=float, nargs='?')
    args = parser.parse_args()
    
    websky_map = create_websky_map(path=args.path, freq=args.freq, noise=args.noise, fwhm=args.fwhm)
    # creating fits file of generated map
    filename = [f"websky_f{args.freq}"]
    
    if args.noise is not None:
        filename.append(f"noise{args.noise}")
    
    if args.fwhm is not None:
        fwhm_str = f"{args.fwhm}".replace('.', '_')
        filename.append(f"fwhm_{fwhm_str}")
    
    filename = "_".join(filename) + ".fits"
    enmap.write_map(filename, websky_map)
    print(f"Saving file as {filename}")
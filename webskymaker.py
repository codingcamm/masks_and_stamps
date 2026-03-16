#General imports
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import time


#Imports for creating WebSky maps
import h5py
import healpy as hp
from pixell import enmap,utils, reproject, bunch, curvedsky, enplot
import os, sys


#Unit conversion
from astropy import units as u
from astropy.constants import k_B, h
from maps import *


# Creating stamps
from astropy import wcs
import random
from astLib import astWCS, astImages
from astropy.wcs import WCS
import astropy.io.fits as pyfits
from pixell import wcsutils, powspec, fft as enfft
from pixell.enmap import ndmap, samewcs, fft, ifft


# Global Constants
TCMB    = 2.726 #Kelvin
TCMB_uK = 2.726e6 #micro-Kelvin
hplanck = 6.626068e-34 #MKS
kboltz  = 1.3806503e-23 #MKS
clight  = 299792458.0 #MKS
path    = "/mnt/welch/USERS/cwhitaker/maps/websky/"

# Unit conversion
# Generating coordinates
def arccos_arange(start, finish, steps):
    """
    Converts an arange array of angles (in degrees) to their arccosine values in radians.
    Returns:
    Array of arccosine values in radians.
    """
    degrees_array = np.arange(start, finish, steps)
    radians_array = np.deg2rad(degrees_array)
    # (arccosine values will be between -1 and 1)
    cosine_values = np.cos(radians_array)
    y = np.arccos(cosine_values)
    return y

def generate_coords():
    # Arccosine helper function

    # Coordinates
    x = np.arange(0, 361 * utils.degree, 1.5 * utils.degree)
    y = arccos_arange(-69, 69, 1.5)

    # Array of coordinates
    dec, ra = np.meshgrid(y, x)

    # Flatten the meshgrid arrays and combine them into pairs of positions
    coords = np.array([dec.ravel(), ra.ravel()]).T
    coords = np.unique(coords, axis=0)
    return coords
    
def dBnudT(nu_ghz):
    nu = 1.e9*np.asarray(nu_ghz)
    X = hplanck*nu/(kboltz*TCMB)
    return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK * 1e26

def ItoDeltaT(nu_ghz):
    return 1./dBnudT(nu_ghz)
    
# Creating a websky map of some frequency
# shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(0.5 / 60))
def create_websky_map(path, freq, noise=None, sigma=None):
    # Shape and WCS of the maps
    # For FullSky
    shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(0.5 / 60))

    # Access to WebSky Data
    #path = "/mnt/welch/USERS/cwhitaker/maps/websky/"

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

    if 90 <= freq <= 90.2:
        cib_map    = enmap.read_map(path + "cib_90.2_car.fits")
        cib_map    = enmap.resample(cib_map, cmb_map.shape)
        radio_map  = enmap.read_map(path + "/map_radio_0.5arcmin_f90.2.fits")
        radio_map  = enmap.resample(radio_map, cmb_map.shape)
        tsz_factor = -4.2840 * 1e6 # tSZ conversion for 90 GHz
        cib_factor = 2.5947 * 1e3
    elif 150 <= freq <= 153:
        cib_map    = enmap.read_map(path + "cib_153_car.fits")
        cib_map    = enmap.resample(cib_map, cmb_map.shape)
        radio_map  = enmap.read_map(path + "/map_radio_0.5arcmin_f143.0.fits")
        radio_map  = enmap.resample(radio_map, cmb_map.shape)
        tsz_factor = -2.7685 * 1e6 # tSZ conversion for 150 GHz
        cib_factor = 4.6831 * 1e3
    # Conversion factors
    elif 217 <= freq <= 225:
        cib_map    = enmap.read_map(path + "cib_217_car.fits")
        cib_map    = enmap.resample(cib_map, cmb_map.shape)
        radio_map  = enmap.read_map(path + "/map_radio_0.5arcmin_f217.0.fits")
        radio_map  = enmap.resample(radio_map, cmb_map.shape)
        tsz_factor = 3.1517 * 1e5 # tSZ conversion for 220 GHz
        cib_factor = 2.0676 * 1e3
        #cib_225 = 2.0716 * 1e3
    else:
        raise ValueError(f"Frequency {freq} GHz not supported. "
                         "Supported ranges: 90–90.2, 150–153, 217–225 GHz.")

    # Noise control
    def white_noise(shape,wcs,noise_muK_arcmin=None,seed=None,ipsizemap=None,div=None):
        """
        Generate a non-band-limited white noise map.
        """
        if div is None: div = ivar(shape,wcs,noise_muK_arcmin,ipsizemap=ipsizemap)
        if seed is not None: np.random.seed(seed)
        return np.random.standard_normal(shape) / np.sqrt(div)

    # Smoothing
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

    if noise is None:
        noise_map = enmap.zeros(shape, wcs)
    elif noise == 0:
        raise ValueError(f"Noise must be greater than zero, omit noise parameter for noiseless map")
    else:
        # ACT Noise around noise_muK_arcmin=20
        noise_map = white_noise(shape, wcs, noise_muK_arcmin=noise)

    # Combine maps
    websky_map = radio_map*ItoDeltaT(freq) + cmb_map + ksz_map + tsz_map*tsz_factor + cib_map*cib_factor + noise_map
    
    if sigma is not None:
        websky_map = smooth_gauss(websky_map, sigma=sigma)
    
    return websky_map, wcs
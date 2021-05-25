import time

from astropy.io import fits
import numpy as np
import os
import pandas as pd

from galaxy_classification import isLRG, isELG, isQSO_cuts

hdulistBricksSouthSummary = fits.open('../bricks_data/survey-bricks-dr9-south.fits')
data_south = hdulistBricksSouthSummary[1].data
brickname_south = data_south.field('brickname')
brickid_south = data_south.field('brickid')
south_survey_is_south = data_south.field('survey_primary')

hdulistBricksNorthSummary = fits.open('../bricks_data/survey-bricks-dr9-north.fits')
data_north = hdulistBricksNorthSummary[1].data
brickname_north = data_north.field('brickname')
brickid_north = data_north.field('brickid')
survey_north = data_north.field('survey_primary')
north_survey_is_south = np.invert(survey_north)

# for brick in range(600):
#
#    randomint = random.randint(0, len(brickname_south))
#    brickname = brickname_south[randomint]
#    brickid = brickid_south[randomint]
#
#   folder = brickname[:3]
#   url = f'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/tractor/{folder}/tractor-{brickname}.fits'
#   wget.download(url, '/Volumes/Astrostick/bricks_data/south/')
#
#   if (brick % 2) == 0:
#       print("Brick south successfully downloaded: ", brickname,":", brickid)

# print()
# print("=============================== Download South Completed ==================================")
# print()
#
# for brick in range(300):
#   randomint = random.randint(0, len(brickname_north))
#   brickname = brickname_north[randomint]
#   brickid = brickid_north[randomint]
#
#   folder = brickname[:3]
#   url = f'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/north/tractor/{folder}/tractor-{brickname}.fits'
#   wget.download(url, '/Volumes/Astrostick/bricks_data/north/')
#
#   if (brick % 2) == 0:
#      print("Brick north successfully downloaded: ", brickname,":", brickid)
#
# print()
# print("=============================== Download North Completed ==================================")
# print()


print()
print("=============================== Classification South... ==================================")
print()

start = time.time()

bricknames_south_sample = []

for filename in os.listdir('/Volumes/Astrostick/bricks_data/south_test/'):
    brickn = filename.replace("tractor-", "")
    brickn = brickn.replace(".fits", "")
    bricknames_south_sample.append(brickn)

df = pd.DataFrame(columns=['BrickID', 'ObjectID', 'RA', 'DEC', 'South', 'Target_type'])


def get_extinction_corrected_fluxes(data):
    flux_g = data.field('flux_g')
    flux_r = data.field('flux_r')
    flux_z = data.field('flux_z')
    flux_w1 = data.field('flux_w1')
    flux_w2 = data.field('flux_w2')

    # getting predicted -band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing

    fiberflux_g = data.field('fiberflux_g')
    fiberflux_r = data.field('fiberflux_r')
    fiberflux_z = data.field('fiberflux_z')

    # correcting for extinction ---> divide by the transmission

    mw_transmission_g = data.field('mw_transmission_g')
    mw_transmission_r = data.field('mw_transmission_r')
    mw_transmission_z = data.field('mw_transmission_z')
    mw_transmission_w1 = data.field('mw_transmission_w1')
    mw_transmission_w2 = data.field('mw_transmission_w2')

    # correcting for extinction ---> divide by the transmission
    return flux_g / mw_transmission_g, \
           flux_r / mw_transmission_r, \
           flux_z / mw_transmission_z, \
           flux_w1 / mw_transmission_w1, \
           flux_w2 / mw_transmission_w2, \
           fiberflux_g / mw_transmission_g, \
           fiberflux_r / mw_transmission_r, \
           fiberflux_z / mw_transmission_z


def get_fluxes_not_extinction_corrected(data):
    # Retrurns Predicted -band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing
    # Returns Inverse variance of FLUXES
    return data.field('fibertotflux_g'), \
           data.field('fibertotflux_r'), \
           data.field('fibertotflux_z'), \
           data.field('flux_ivar_g'), \
           data.field('flux_ivar_r'), \
           data.field('flux_ivar_z'), \
           data.field('flux_ivar_w1'), \
           data.field('flux_ivar_w2')


class Brick:
    'Represents all attributes necessary to classify all galaxies in a brick into their categories'

    def __init__(self, data):
        self.data = data
        self.initialise_brick()

    def initialise_brick(self):
        self.flux_g = self.data.field('flux_g')
        self.flux_r = self.data.field('flux_r')
        self.flux_z = self.data.field('flux_z')
        self.flux_w1 = self.data.field('flux_w1')
        self.flux_w2 = self.data.field('flux_w2')

        # getting predicted -band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing

        self.fiberflux_g = self.data.field('fiberflux_g')
        self.fiberflux_r = self.data.field('fiberflux_r')
        self.fiberflux_z = self.data.field('fiberflux_z')

        # correcting for extinction ---> divide by the transmission

        self.mw_transmission_g = self.data.field('mw_transmission_g')
        self.mw_transmission_r = self.data.field('mw_transmission_r')
        self.mw_transmission_z = self.data.field('mw_transmission_z')
        self.mw_transmission_w1 = self.data.field('mw_transmission_w1')
        self.mw_transmission_w2 = self.data.field('mw_transmission_w2')

        # Retrurns Predicted -band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing
        # Returns Inverse variance of FLUXES

        self.fibertotflux_g = self.data.field('fibertotflux_g')
        self.fibertotflux_r = self.data.field('fibertotflux_r')
        self.fibertotflux_z = self.data.field('fibertotflux_z')

        self.flux_ivar_g = self.data.field('flux_ivar_g')
        self.flux_ivar_r = self.data.field('flux_ivar_r')
        self.flux_ivar_z = self.data.field('flux_ivar_z')
        self.flux_ivar_w1 = self.data.field('flux_ivar_w1')
        self.flux_ivar_w2 = self.data.field('flux_ivar_w2')



        self.extinction_correction()

    def extinction_correction(self):
        # correcting for extinction ---> divide by the transmission
        self.flux_g = self.flux_g / self.mw_transmission_g
        self.flux_r = self.flux_r / self.mw_transmission_r
        self.flux_z = self.flux_z / self.mw_transmission_z
        self.flux_w1 = self.flux_w1 / self.mw_transmission_w1
        self.flux_w2 = self.flux_w2 / self.mw_transmission_w2

        self.fiberflux_g = self.fiberflux_g / self.mw_transmission_g
        self.fiberflux_r = self.fiberflux_r / self.mw_transmission_r
        self.fiberflux_z = self.fiberflux_z / self.mw_transmission_z


for no, brickname in enumerate(bricknames_south_sample):
    # df = pd.read_csv('../bricks_data/galaxy_catalogue_sample.csv')

    brickid = brickid_south[np.where(brickname_south == brickname)]
    if len(brickid > 0):
        brickid = brickid[0]
    else:
        brickid = 0
        # Check how there can be a brickname without corresponding id

    hdulistSingleBrick = fits.open(f'/Volumes/Astrostick/bricks_data/south_test/tractor-{brickname}.fits')
    data = hdulistSingleBrick[1].data

    brick = Brick(data)

    # Obtaining the flux in nano-maggies of g, r, z, W1 and W2 bands.
    flux_g, flux_r, flux_z, flux_w1, flux_w2, fiberflux_g, fiberflux_r, fiberflux_z = get_extinction_corrected_fluxes(
        data)

    # This Flux is not corrected for extinction
    fibertotflux_g, fibertotflux_r, fibertotflux_z, flux_ivar_g, flux_ivar_r, flux_ivar_z, flux_ivar_w1, flux_ivar_w2 = get_fluxes_not_extinction_corrected(
        data)

    # Get the number of pixels contributed to the central pixels in the bands
    nobs_g = data.field('nobs_g')
    nobs_r = data.field('nobs_r')
    nobs_z = data.field('nobs_z')

    # get the Gaia-based g-, b- and r-band MAGNITUDES.
    gaia_g_mag = data.field('gaia_phot_g_mean_mag')
    gaia_b_mag = data.field('gaia_phot_bp_mean_mag')
    gaia_r_mag = data.field('gaia_phot_rp_mean_mag')

    # Get the Signal-to-noise in g, r, z, W1 and W2 defined as the flux per
    # band divided by sigma (flux x sqrt of the inverse variance).
    snr_g = flux_g * np.sqrt(flux_ivar_g)
    snr_r = flux_r * np.sqrt(flux_ivar_r)
    snr_z = flux_z * np.sqrt(flux_ivar_z)
    snr_w1 = flux_w1 * np.sqrt(flux_ivar_w1)
    snr_w2 = flux_w2 * np.sqrt(flux_ivar_w2)

    # Retrieving the maskbits for quasar detection and boolean for brick is in north
    maskbits = data.field('maskbits')

    # Extracting Positions, and Object IDs
    ids = data.field('brickid')
    ra = data.field('ra')
    dec = data.field('dec')
    objid = data.field('objid')

    south = south_survey_is_south[np.where(brickid_south == brickid)]
    south_array = np.full(shape=len(flux_g), fill_value=south)

    if len(south) > 0:
        south = south[0]
    else:
        south = True

    target_type = np.zeros(len(flux_g))

    is_LRG = isLRG(gflux=flux_g, rflux=flux_r, zflux=flux_z, w1flux=flux_w1, w2flux=flux_w2,
                   zfiberflux=fiberflux_z, rfluxivar=flux_ivar_r, zfluxivar=flux_ivar_z, w1fluxivar=flux_ivar_w1,
                   gaiagmag=gaia_g_mag, gnobs=nobs_g, rnobs=nobs_r, znobs=nobs_z, maskbits=maskbits,
                   zfibertotflux=fibertotflux_z, primary=None, south=south)

    target_type[np.where(is_LRG == True)] = 1

    is_ELG, is_ELGVLO = isELG(gflux=flux_g, rflux=flux_r, zflux=flux_z, w1flux=flux_w1, w2flux=flux_w2,
                              gfiberflux=fiberflux_g, gsnr=snr_g, rsnr=snr_r, zsnr=snr_z,
                              gnobs=nobs_g, rnobs=nobs_r, znobs=nobs_z,
                              maskbits=maskbits, south=south)

    target_type[np.where(is_ELG == True)] = 2
    target_type[np.where(is_ELGVLO == True)] = 2

    is_QSO = isQSO_cuts(gflux=flux_g, rflux=flux_r, zflux=flux_z, w1flux=flux_w1, w2flux=flux_w2,
                        w1snr=snr_w1, w2snr=snr_w2, maskbits=maskbits,
                        gnobs=nobs_g, rnobs=nobs_r, znobs=nobs_z,
                        objtype=None, primary=None, optical=False, south=south)

    target_type[np.where(is_QSO == True)] = 3

    stacked_array = np.stack((ids, objid, ra, dec, south_array, target_type), axis=1)
    support_df = pd.DataFrame(stacked_array, columns=['BrickID', 'ObjectID', 'RA', 'DEC', 'South', 'Target_type'])
    support_df.drop(support_df[support_df.Target_type == 0].index, inplace=True)

    df = df.append(support_df)

    # df = df.append({'BrickID': brickid, 'ObjectID': objid, 'RA': ra, 'DEC': dec,
    #                   'South': south, 'Target_type': 3}, ignore_index=True)

    # Do not forget to check this clause
    # ['BrickID', 'ObjectID','RA', 'DEC', 'South', 'Target_type']

    '''
    for i in range(len(flux_g)):

        if isLRG(gflux=flux_g[i], rflux=flux_r[i], zflux=flux_z[i], w1flux=flux_w1[i], w2flux=flux_w2[i],
                 zfiberflux=fiberflux_z[i], rfluxivar=flux_ivar_r[i], zfluxivar=flux_ivar_z[i], w1fluxivar=flux_ivar_w1[i],
                 gaiagmag=gaia_g_mag[i], gnobs=nobs_g[i], rnobs=nobs_r[i], znobs=nobs_z[i], maskbits=maskbits[i],
                 zfibertotflux=fibertotflux_z[i], primary=None, south=south):
            df = df.append({'BrickID': brickid, 'ObjectID': objid[i],'RA': ra[i], 'DEC': dec[i],
                             'South': south, 'Target_type': 1}, ignore_index=True)
            continue

        elg, elgvlo = isELG(gflux=flux_g[i], rflux=flux_r[i], zflux=flux_z[i], w1flux=flux_w1[i], w2flux=flux_w2[i],
                            gfiberflux=fiberflux_g[i], gsnr=snr_g[i], rsnr=snr_r[i], zsnr=snr_z[i],
                            gnobs=nobs_g[i], rnobs=nobs_r[i], znobs=nobs_z[i],
                            maskbits=maskbits[i],south=south)
        if elg or elgvlo:
            df = df.append({'BrickID': brickid, 'ObjectID': objid[i],'RA': ra[i], 'DEC': dec[i],
                             'South': south, 'Target_type': 2}, ignore_index=True)
            continue

        if isQSO_cuts(gflux=flux_g[i], rflux=flux_r[i], zflux=flux_z[i], w1flux=flux_w1[i], w2flux=flux_w2[i],
                      w1snr=snr_w1[i], w2snr=snr_w2[i], maskbits=maskbits[i],
                      gnobs=nobs_g[i], rnobs=nobs_r[i], znobs=nobs_z[i],
                      objtype=None, primary=None, optical=False, south=south):
            df = df.append({'BrickID': brickid, 'ObjectID': objid[i],'RA': ra[i], 'DEC': dec[i],
                             'South': south, 'Target_type': 3}, ignore_index=True)
            continue

    '''

    # df.to_csv('../bricks_data/galaxy_catalogue_sample_profiling.csv', index=False)

    # if no % 3 == 0:
    # print(no, " of ", len(bricknames_south_sample), "bricks processed")

    # print(" ===================== Brick", brickname, " complete=====================")

print()
print("=============================== Classification South Completed ==================================")
print()
df = df[df['Target_type'] > 0]
df.to_csv('../bricks_data/galaxy_catalogue_sample_profiling.csv', index=False)
print(df.shape)
print(df.head())

print("Time taken for 15 bricks: ", time.time() - start)

'''
bricknames_north_sample = []

for filename in os.listdir('/Volumes/Astrostick/bricks_data/north/'):
    brickn = filename.replace("tractor-", "")
    brickn = brickn.replace(".fits", "")
    bricknames_north_sample.append(brickn)

df = pd.DataFrame(columns=['BrickID', 'ObjectID','RA', 'DEC', 'South', 'Target_type'])


for no, brickname in enumerate(bricknames_north_sample):
    brickid = brickid_north[np.where(brickname_north == brickname)][0]
    hdulistSingleBrick = fits.open(f'/Volumes/Astrostick/bricks_data/north/tractor-{brickname}.fits')
    data = hdulistSingleBrick[1].data

    #Obtaining the flux in nano-maggies of g, r, z, W1 and W2 bands.
    flux_g = data.field('flux_g')
    flux_r = data.field('flux_r')
    flux_z = data.field('flux_z')
    flux_w1 = data.field('flux_w1')
    flux_w2 = data.field('flux_w2')


    mw_transmission_g = data.field('mw_transmission_g')
    mw_transmission_r = data.field('mw_transmission_r')
    mw_transmission_z = data.field('mw_transmission_z')
    mw_transmission_w1 = data.field('mw_transmission_w1')
    mw_transmission_w2 = data.field('mw_transmission_w2')

    #correcting for extinction ---> divide by the transmission
    flux_g = flux_g / mw_transmission_g
    flux_r = flux_r / mw_transmission_r
    flux_z = flux_z / mw_transmission_z
    flux_w1 = flux_w1 / mw_transmission_w1
    flux_w2 = flux_w2 / mw_transmission_w2

    # getting predicted -band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing
    fiberflux_g = data.field('fiberflux_g')
    fiberflux_r = data.field('fiberflux_r')
    fiberflux_z = data.field('fiberflux_z')

    #correcting for extinction ---> divide by the transmission
    fiberflux_g = fiberflux_g / mw_transmission_g
    fiberflux_r = fiberflux_r / mw_transmission_r
    fiberflux_z = fiberflux_z / mw_transmission_z

    # Predicted -band flux within a fiber of diameter 1.5 arcsec from all sources at this location in 1 arcsec Gaussian seeing
    # This Flux is not corrected for extinction
    fibertotflux_g = data.field('fibertotflux_g')
    fibertotflux_r = data.field('fibertotflux_r')
    fibertotflux_z = data.field('fibertotflux_z')

    # Get Inverse variance of FLUXES
    flux_ivar_g = data.field('flux_ivar_g')
    flux_ivar_r = data.field('flux_ivar_r')
    flux_ivar_z = data.field('flux_ivar_z')
    flux_ivar_w1 = data.field('flux_ivar_w1')
    flux_ivar_w2 = data.field('flux_ivar_w2')


    # Get the number of pixels contributed to the central pixels in the bands
    nobs_g = data.field('nobs_g')
    nobs_r = data.field('nobs_r')
    nobs_z = data.field('nobs_z')

    # get the Gaia-based g-, b- and r-band MAGNITUDES.
    gaia_g_mag = data.field('gaia_phot_g_mean_mag')
    gaia_b_mag = data.field('gaia_phot_bp_mean_mag')
    gaia_r_mag = data.field('gaia_phot_rp_mean_mag')

    # Get the Signal-to-noise in g, r, z, W1 and W2 defined as the flux per
    # band divided by sigma (flux x sqrt of the inverse variance).
    snr_g = flux_g * np.sqrt(flux_ivar_g)
    snr_r = flux_r * np.sqrt(flux_ivar_r)
    snr_z = flux_z * np.sqrt(flux_ivar_z)
    snr_w1 = flux_w1 * np.sqrt(flux_ivar_w1)
    snr_w2 = flux_w2 * np.sqrt(flux_ivar_w2)


    #Retrieving the maskbits for quasar detection and boolean for brick is in north
    maskbits = data.field('maskbits')

    #Extracting Positions, and Object IDs
    ra = data.field('ra')
    dec = data.field('dec')
    objid = data.field('objid')


    south = north_survey_is_south[np.where(brickid_north == brickid)][0]

    #['BrickID', 'ObjectID','RA', 'DEC', 'South', 'Target_type']

    for i in range(len(flux_g)):

        if isLRG(gflux=flux_g[i], rflux=flux_r[i], zflux=flux_z[i], w1flux=flux_w1[i], w2flux=flux_w2[i],
                 zfiberflux=fiberflux_z[i], rfluxivar=flux_ivar_r[i], zfluxivar=flux_ivar_z[i], w1fluxivar=flux_ivar_w1[i],
                 gaiagmag=gaia_g_mag[i], gnobs=nobs_g[i], rnobs=nobs_r[i], znobs=nobs_z[i], maskbits=maskbits[i],
                 zfibertotflux=fibertotflux_z[i], primary=None, south=south):
            df =  df.append({'BrickID': brickid, 'ObjectID': objid[i],'RA': ra[i], 'DEC': dec[i],
                             'South': south, 'Target_type': 1}, ignore_index=True)
            continue

        elg, elgvlo = isELG(gflux=flux_g[i], rflux=flux_r[i], zflux=flux_z[i], w1flux=flux_w1[i], w2flux=flux_w2[i],
                            gfiberflux=fiberflux_g[i], gsnr=snr_g[i], rsnr=snr_r[i], zsnr=snr_z[i],
                            gnobs=nobs_g[i], rnobs=nobs_r[i], znobs=nobs_z[i],
                            maskbits=maskbits[i],south=south)
        if elg or elgvlo:
            df = df.append({'BrickID': brickid, 'ObjectID': objid[i],'RA': ra[i], 'DEC': dec[i],
                             'South': south, 'Target_type': 2}, ignore_index=True)
            continue

        if isQSO_cuts(gflux=flux_g[i], rflux=flux_r[i], zflux=flux_z[i], w1flux=flux_w1[i], w2flux=flux_w2[i],
                      w1snr=snr_w1[i], w2snr=snr_w2[i], maskbits=maskbits[i],
                      gnobs=nobs_g[i], rnobs=nobs_r[i], znobs=nobs_z[i],
                      objtype=None, primary=None, optical=False, south=south):
            df = df.append({'BrickID': brickid, 'ObjectID': objid[i],'RA': ra[i], 'DEC': dec[i],
                             'South': south, 'Target_type': 3}, ignore_index=True)
            continue

    if no % 20 == 0:
        print(no, " of ", len(bricknames_south_sample), "bricks processed:", brickname)

df.to_csv('../bricks_data/galaxy_catalogue_sample.csv', index=False)

print()
print("=============================== Classification North Completed ==================================")
print()


print()
print("=============================== Removing South ... ==================================")
print()


for filename in os.listdir('/Volumes/Astrostick/bricks_data/south/'):
    print("Removing file:", filename)
    os.remove(f'/Volumes/Astrostick/bricks_data/south/{filename}')

print()
print("=============================== Removing South Completed ==================================")
print()

'''

'''
print()
print("=============================== Removing North ... ==================================")
print()

for filename in os.listdir('/Volumes/Astrostick/bricks_data/north/'):
    print("Removing file:", filename)
    os.remove(f'/Volumes/Astrostick/bricks_data/north/{filename}')

print()
print("=============================== Removing North Completed ==================================")
print()

'''

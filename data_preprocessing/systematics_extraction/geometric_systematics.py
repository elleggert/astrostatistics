import numpy as np
from astropy.io import fits
import healpy as hp
import pickle
from collections import defaultdict

decamCCD = fits.open('../../bricks_data/ccds-annotated-decam-dr9.fits')
mosaicCCD = fits.open('../../bricks_data/ccds-annotated-mosaic-dr9.fits')
bassCCD = fits.open('../../bricks_data/ccds-annotated-90prime-dr9.fits')

dataDecam = decamCCD[1].data
dataMosaic = mosaicCCD[1].data
dataBass = bassCCD[1].data


def raDec2thetaPhi(ra, dec):
    return (0.5 * np.pi - np.deg2rad(dec)), (np.deg2rad(ra))

NSIDE = 512
NSIDE_SUB = 2048
NPIX = hp.nside2npix(NSIDE)

# Extracting systematics
filter_colour = np.concatenate((dataDecam.field('filter'), dataMosaic.field('filter'), dataBass.field('filter')),
                               axis=0)

# camera = np.concatenate((dataDecam.field('camera'), dataMosaic.field('camera'), dataBass.field('camera')), axis=0)

exptime = np.concatenate((dataDecam.field('exptime'), dataMosaic.field('exptime'), dataBass.field('exptime')), axis=0)
airmass = np.concatenate((dataDecam.field('airmass'), dataMosaic.field('airmass'), dataBass.field('airmass')), axis=0)
fwhm = np.concatenate((dataDecam.field('fwhm'), dataMosaic.field('fwhm'), dataBass.field('fwhm')), axis=0)
seeing = fwhm * 0.262
ccdskysb = np.concatenate((dataDecam.field('ccdskysb'), dataMosaic.field('ccdskysb'), dataBass.field('ccdskysb')),
                          axis=0)
ccdskycounts = np.concatenate(
    (dataDecam.field('ccdskycounts'), dataMosaic.field('ccdskycounts'), dataBass.field('ccdskycounts')), axis=0)

# Use this cell to simply import an existing pixel2subpixel mapping

with open(f'../../bricks_data/pixel2subpixel_{NSIDE}_{NSIDE_SUB}.pickle', 'rb') as f:
    pixel2subpixel_dict = pickle.load(f)
    f.close()

# For DECAM, BASS, MzLS
with open(f'../../bricks_data/pixel2ccd_{NSIDE}.pickle', 'rb') as f:
    pixel2ccd_dict = pickle.load(f)
    f.close()

pixels_overall = pixel2ccd_dict.keys()

# For DECAM, BASS, MzLS
with open(f'../../bricks_data/pixel2ccd_{NSIDE_SUB}_non_inclusive.pickle', 'rb') as f:
    subpixel2ccd_dict = pickle.load(f)
    f.close()

pixel2systematics_dict = defaultdict(list)
for i, sample_pixel in enumerate(pixels_overall):

    subpixels_per_pixel = pixel2subpixel_dict[sample_pixel]

    airmass_mean = airmass_std = airmass_median = 0

    seeing_g_mean = seeing_g_std = seeing_g_median = 0
    seeing_r_mean = seeing_r_std = seeing_r_median = 0
    seeing_z_mean = seeing_z_std = seeing_z_median = 0

    ccdskysb_g_mean = ccdskysb_g_std = ccdskysb_g_median = 0
    ccdskysb_r_mean = ccdskysb_r_std = ccdskysb_r_median = 0
    ccdskysb_z_mean = ccdskysb_z_std = ccdskysb_z_median = 0

    ccdskycounts_g_mean = ccdskycounts_g_std = ccdskycounts_g_median = 0
    ccdskycounts_r_mean = ccdskycounts_r_std = ccdskycounts_r_median = 0
    ccdskycounts_z_mean = ccdskycounts_z_std = ccdskycounts_z_median = 0

    subpixels_covered = subpixels_covered_g = subpixels_covered_r = subpixels_covered_z = 0

    airmass_min = seeing_g_min = seeing_r_min = seeing_z_min = ccdskysb_g_min = ccdskysb_r_min = ccdskysb_z_min = ccdskycounts_g_min = ccdskycounts_r_min = ccdskycounts_z_min = math.inf

    airmass_max = seeing_g_max = seeing_r_max = seeing_z_max = ccdskysb_g_max = ccdskysb_r_max = ccdskysb_z_max = ccdskycounts_g_max = ccdskycounts_r_max = ccdskycounts_z_max = - math.inf

    # Go through all 16 subpixels in the sample pixel
    for subpixel in subpixels_per_pixel:
        # Condition needed in case a subpixel is not covered by CCDs
        if subpixel not in subpixel2ccd_dict.keys():
            continue
        subpixels_covered += 1

        ccds_per_subpixel = subpixel2ccd_dict[subpixel]
        # Get values for singular systematics
        airmass_mean += airmass[ccds_per_subpixel].mean()
        airmass_std += airmass[ccds_per_subpixel].std()
        airmass_median += np.median(airmass[ccds_per_subpixel])
        airmass_min = min(airmass[ccds_per_subpixel].min(), airmass_min)
        airmass_max = max(airmass[ccds_per_subpixel].max(), airmass_max)

        # Get values for per band systematics

        mask_g = (filter_colour[ccds_per_subpixel] == 'g')
        mask_r = (filter_colour[ccds_per_subpixel] == 'r')
        mask_z = (filter_colour[ccds_per_subpixel] == 'z')

        see = seeing[ccds_per_subpixel]
        seeing_g = see[mask_g]
        seeing_r = see[mask_r]
        seeing_z = see[mask_z]


        # Sky background
        sb = ccdskysb[ccds_per_subpixel]
        ccdskysb_g = sb[mask_g]
        ccdskysb_r = sb[mask_r]
        ccdskysb_z = sb[mask_z]

        # Sky Counts
        sc = ccdskycounts[ccds_per_subpixel]
        ccdskycounts_g = sc[mask_g]
        ccdskycounts_r = sc[mask_r]
        ccdskycounts_z = sc[mask_z]

        g_exp = mask_g.sum()
        if g_exp > 0:
            print("Vals in subpixel", seeing_g)
            print("Prev Max", seeing_g_max)
            subpixels_covered_g += 1
            seeing_g_mean += seeing_g.mean()
            seeing_g_std += seeing_g.std()
            seeing_g_median += np.median(seeing_g)
            seeing_g_min = min(seeing_g.min(), seeing_g_min)
            seeing_g_max = max(seeing_g.max() , seeing_g_max)
            print("Post Max", seeing_g_max)
            print()

            ccdskysb_g_mean += ccdskysb_g.mean()
            ccdskysb_g_std += ccdskysb_g.std()
            ccdskysb_g_median += np.median(ccdskysb_g)
            ccdskysb_g_min = min(ccdskysb_g.min(), ccdskysb_g_min)
            ccdskysb_g_max = max(ccdskysb_g.max() , ccdskysb_g_max)

            ccdskycounts_g_mean += ccdskycounts_g.mean()
            ccdskycounts_g_std += ccdskycounts_g.std()
            ccdskycounts_g_median += np.median(ccdskycounts_g)
            ccdskycounts_g_min = min(ccdskycounts_g.min(), ccdskycounts_g_min)
            ccdskycounts_g_max = max(ccdskycounts_g.max(), ccdskycounts_g_max)

        r_exp = mask_r.sum()
        if r_exp > 0:
            subpixels_covered_r += 1
            seeing_r_mean += seeing_r.mean()
            seeing_r_std += seeing_r.std()
            seeing_r_median += np.median(seeing_r)
            seeing_r_min = min(seeing_r.min(),seeing_r_min )
            seeing_r_max = max(seeing_r.max(), seeing_r_max)

            ccdskysb_r_mean += ccdskysb_r.mean()
            ccdskysb_r_std += ccdskysb_r.std()
            ccdskysb_r_median += np.median(ccdskysb_r)
            ccdskysb_r_min = min(ccdskysb_r.min(), ccdskysb_r_min)
            ccdskysb_r_max = max(ccdskysb_r.max(), ccdskysb_r_max)

            ccdskycounts_r_mean += ccdskycounts_r.mean()
            ccdskycounts_r_std += ccdskycounts_r.std()
            ccdskycounts_r_median += np.median(ccdskycounts_r)
            ccdskycounts_r_min = min(ccdskycounts_r.min(),ccdskycounts_r_min )
            ccdskycounts_r_max = max(ccdskycounts_r.max(),ccdskycounts_r_max )

        z_exp = mask_z.sum()
        if z_exp > 0:
            subpixels_covered_z += 1
            seeing_z_mean += seeing_z.mean()
            seeing_z_std += seeing_z.std()
            seeing_z_median += np.median(seeing_z)
            seeing_z_min = min(seeing_z.min(), seeing_z_min)
            seeing_z_max = max(seeing_z.max(), seeing_z_max)

            ccdskysb_z_mean += ccdskysb_z.mean()
            ccdskysb_z_std += ccdskysb_z.std()
            ccdskysb_z_median += np.median(ccdskysb_z)
            ccdskysb_z_min = min(ccdskysb_z.min(),ccdskysb_z_min )
            ccdskysb_z_max = max(ccdskysb_z.max(), ccdskysb_z_max)

            ccdskycounts_z_mean += ccdskycounts_z.mean()
            ccdskycounts_z_std += ccdskycounts_z.std()
            ccdskycounts_z_median += np.median(ccdskycounts_z)
            ccdskycounts_z_min = min(ccdskycounts_z.min(), ccdskycounts_z_min)
            ccdskycounts_z_max = max(ccdskycounts_z.max(), ccdskycounts_z_max)

    if i % 17843 == 0:
        print(int(i / 17843), '%')

    # Do not append to dictionary if none of the subpixels is covered in full
    if subpixels_covered == 0:
        continue

    systematics_per_pixel = []
    systematics_per_pixel.append(airmass_mean / subpixels_covered)
    systematics_per_pixel.append(airmass_std / subpixels_covered)
    systematics_per_pixel.append(airmass_median / subpixels_covered)
    systematics_per_pixel.append(airmass_min)
    systematics_per_pixel.append(airmass_max)

    systematics_per_pixel.append(ccdskysb_g_mean / subpixels_covered_g)
    systematics_per_pixel.append(ccdskysb_g_std / subpixels_covered_g)
    systematics_per_pixel.append(ccdskysb_g_median / subpixels_covered_g)
    systematics_per_pixel.append(ccdskysb_g_min)
    systematics_per_pixel.append(ccdskysb_g_max)

    systematics_per_pixel.append(ccdskysb_r_mean / subpixels_covered_r)
    systematics_per_pixel.append(ccdskysb_r_std / subpixels_covered_r)
    systematics_per_pixel.append(ccdskysb_r_median / subpixels_covered_r)
    systematics_per_pixel.append(ccdskysb_r_min)
    systematics_per_pixel.append(ccdskysb_r_max)

    systematics_per_pixel.append(ccdskysb_z_mean / subpixels_covered_z)
    systematics_per_pixel.append(ccdskysb_z_std / subpixels_covered_z)
    systematics_per_pixel.append(ccdskysb_z_median / subpixels_covered_z)
    systematics_per_pixel.append(ccdskysb_z_min)
    systematics_per_pixel.append(ccdskysb_z_max)

    systematics_per_pixel.append(ccdskycounts_g_mean / subpixels_covered_g)
    systematics_per_pixel.append(ccdskycounts_g_std / subpixels_covered_g)
    systematics_per_pixel.append(ccdskycounts_g_median / subpixels_covered_g)
    systematics_per_pixel.append(ccdskycounts_g_min)
    systematics_per_pixel.append(ccdskycounts_g_max)

    systematics_per_pixel.append(ccdskycounts_r_mean / subpixels_covered_r)
    systematics_per_pixel.append(ccdskycounts_r_std / subpixels_covered_r)
    systematics_per_pixel.append(ccdskycounts_r_median / subpixels_covered_r)
    systematics_per_pixel.append(ccdskycounts_r_min)
    systematics_per_pixel.append(ccdskycounts_r_max)

    systematics_per_pixel.append(ccdskycounts_z_mean / subpixels_covered_z)
    systematics_per_pixel.append(ccdskycounts_z_std / subpixels_covered_z)
    systematics_per_pixel.append(ccdskycounts_z_median / subpixels_covered_z)
    systematics_per_pixel.append(ccdskycounts_z_min)
    systematics_per_pixel.append(ccdskycounts_z_max)

    systematics_per_pixel.append(seeing_g_mean / subpixels_covered_g)
    systematics_per_pixel.append(seeing_g_std / subpixels_covered_g)
    systematics_per_pixel.append(seeing_g_median / subpixels_covered_g)
    systematics_per_pixel.append(seeing_g_min)
    systematics_per_pixel.append(seeing_g_max)

    systematics_per_pixel.append(seeing_r_mean / subpixels_covered_r)
    systematics_per_pixel.append(seeing_r_std / subpixels_covered_r)
    systematics_per_pixel.append(seeing_r_median / subpixels_covered_r)
    systematics_per_pixel.append(seeing_r_min)
    systematics_per_pixel.append(seeing_r_max)

    systematics_per_pixel.append(seeing_z_mean / subpixels_covered_z)
    systematics_per_pixel.append(seeing_z_std / subpixels_covered_z)
    systematics_per_pixel.append(seeing_z_median / subpixels_covered_z)
    systematics_per_pixel.append(seeing_z_min)
    systematics_per_pixel.append(seeing_z_max)

    # Also appending fraction of pixel covered to cut on it later
    systematics_per_pixel.append(subpixels_covered / 16)

    pixel2systematics_dict[sample_pixel] = systematics_per_pixel

with open(f'../../bricks_data/pixel2systematics_reporting', 'wb') as f:
    pickle.dump(pixel2systematics_dict, f)
    f.close()

system = ['airmass_mean',
          'airmass_std',
          'airmass_median',
          'airmass_min',
          'airmass_max',

          'ccdskysb_g_mean',
          'ccdskysb_g_std',
          'ccdskysb_g_median',
          'ccdskysb_g_min',
          'ccdskysb_g_max',

          'ccdskysb_r_mean',
          'ccdskysb_r_std',
          'ccdskysb_r_median',
          'ccdskysb_r_min',
          'ccdskysb_r_max',

          'ccdskysb_z_mean',
          'ccdskysb_z_std',
          'ccdskysb_z_median',
          'ccdskysb_z_min',
          'ccdskysb_z_max',

          'ccdskycounts_g_mean',
          'ccdskycounts_g_std',
          'ccdskycounts_g_median',
          'ccdskycounts_g_min',
          'ccdskycounts_g_max',

          'ccdskycounts_r_mean',
          'ccdskycounts_r_std',
          'ccdskycounts_r_median',
          'ccdskycounts_r_min',
          'ccdskycounts_r_max',

          'ccdskycounts_z_mean',
          'ccdskycounts_z_std',
          'ccdskycounts_z_median',
          'ccdskycounts_z_min',
          'ccdskycounts_z_max',

          'seeing_g_mean',
          'seeing_g_std',
          'seeing_g_median',
          'seeing_g_min',
          'seeing_g_max',

          'seeing_r_mean',
          'seeing_r_std',
          'seeing_r_median',
          'seeing_r_min',
          'seeing_r_max',

          'seeing_z_mean',
          'seeing_z_std',
          'seeing_z_median',
          'seeing_z_min',
          'seeing_z_max',

          'subpixels_covered']

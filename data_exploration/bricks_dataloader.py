from astropy.io import fits
import wget
import numpy as np
import random
import os
import pandas as pd



def get_imaging_maskbits(bitnamelist=None):
    """Return MASKBITS names and bits from the Legacy Surveys.

    Parameters
    ----------
    bitnamelist : :class:`list`, optional, defaults to ``None``
        If not ``None``, return the bit values corresponding to the
        passed names. Otherwise, return the full MASKBITS dictionary.

    Returns
    -------
    :class:`list` or `dict`
        A list of the MASKBITS values if `bitnamelist` is passed,
        otherwise the full MASKBITS dictionary of names-to-values.

    Notes
    -----
    - For the definitions of the mask bits, see, e.g.,
      https://www.legacysurvey.org/dr8/bitmasks/#maskbits
    """
    bitdict = {"BRIGHT": 1, "ALLMASK_G": 5, "ALLMASK_R": 6, "ALLMASK_Z": 7,
               "BAILOUT": 10, "MEDIUM": 11, "GALAXY": 12, "CLUSTER": 13}

    # ADM look up the bit value for each passed bit name.
    if bitnamelist is not None:
        return [bitdict[bitname] for bitname in bitnamelist]

    return bitdict

def get_default_maskbits(bgs=False, mws=False):
    """Return the names of the default MASKBITS for targets.

    Parameters
    ----------
    bgs : :class:`bool`, defaults to ``False``.
        If ``True`` load the "default" scheme for Bright Galaxy Survey
        targets. Otherwise, load the default for other target classes.
    mws : :class:`bool`, defaults to ``False``.
        If ``True`` load the "default" scheme for Milky Way Survey
        targets. Otherwise, load the default for other target classes.

    Returns
    -------
    :class:`list`
        A list of the default MASKBITS names for targets.

    Notes
    -----
    - Only one of `bgs` or `mws` can be ``True``.
    """
    if bgs and mws:
        msg = "Only one of bgs or mws can be passed as True"
        #Adapting to print messages
        print(msg)
        #log.critical(msg)
        raise ValueError(msg)
    if bgs:
        return ["BRIGHT", "CLUSTER"]
    if mws:
        return ["BRIGHT", "GALAXY"]

    return ["BRIGHT", "GALAXY", "CLUSTER"]

def imaging_mask(maskbits, bitnamelist=get_default_maskbits(),
                 bgsmask=False, mwsmask=False):
    """Apply the 'geometric' masks from the Legacy Surveys imaging.

    Parameters
    ----------
    maskbits : :class:`~numpy.ndarray` or ``None``
        General array of `Legacy Surveys mask`_ bits.
    bitnamelist : :class:`list`, defaults to func:`get_default_maskbits()`
        List of Legacy Surveys mask bits to set to ``False``.
    bgsmask : :class:`bool`, defaults to ``False``.
        Load the "default" scheme for Bright Galaxy Survey targets.
        Overrides `bitnamelist`.
    bgsmask : :class:`bool`, defaults to ``False``.
        Load the "default" scheme for Milky Way Survey targets.
        Overrides `bitnamelist`.

    Returns
    -------
    :class:`~numpy.ndarray`
        A boolean array that is the same length as `maskbits` that
        contains ``False`` where any bits in `bitnamelist` are set.

    Notes
    -----
    - Only one of `bgsmask` or `mwsmask` can be ``True``.
    """
    # ADM default for the BGS or MWS..
    if bgsmask or mwsmask:
        bitnamelist = get_default_maskbits(bgs=bgsmask, mws=mwsmask)

    # ADM get the bit values for the passed (or default) bit names.
    bits = get_imaging_maskbits(bitnamelist)

    # ADM Create array of True and set to False where a mask bit is set.
    mb = np.ones_like(maskbits, dtype='?')
    for bit in bits:
        mb &= ((maskbits & 2**bit) == 0)

    return mb

def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          zfiberflux=None, rfluxivar=None, zfluxivar=None, w1fluxivar=None,
          gaiagmag=None, gnobs=None, rnobs=None, znobs=None, maskbits=None,
          zfibertotflux=None, primary=None, south=True):
    """
    Parameters
    ----------
    south: boolean, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS)
        if ``south=False``, otherwise use cuts appropriate to the
        Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an LRG target.

    Notes
    -----
    - Current version (05/07/21) is version 260 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # ADM LRG targets.
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg_quality = primary.copy()

    # ADM basic quality cuts.
    lrg_quality &= notinLRG_mask(
        primary=primary, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
        rfluxivar=rfluxivar, zfluxivar=zfluxivar, w1fluxivar=w1fluxivar,
        gaiagmag=gaiagmag, maskbits=maskbits, zfibertotflux=zfibertotflux
    )

    # ADM color-based selection of LRGs.
    lrg = isLRG_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux, south=south, primary=primary
    )

    lrg &= lrg_quality

    return lrg

def notinLRG_mask(primary=None, rflux=None, zflux=None, w1flux=None,
                  zfiberflux=None, gnobs=None, rnobs=None, znobs=None,
                  rfluxivar=None, zfluxivar=None, w1fluxivar=None,
                  gaiagmag=None, maskbits=None, zfibertotflux=None):
    """See :func:`~desitarget.cuts.isLRG` for details.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is NOT masked for poor quality.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    # ADM to maintain backwards-compatibility with mocks.
    if zfiberflux is None:
        print('Setting zfiberflux to zflux!!!') ## Altered in my code to keep compatibilty
        #log.warning('Setting zfiberflux to zflux!!!')
        zfiberflux = zflux.copy()

    lrg &= (rfluxivar > 0) & (rflux > 0)   # ADM quality in r.
    lrg &= (zfluxivar > 0) & (zflux > 0) & (zfiberflux > 0)   # ADM quality in z.
    lrg &= (w1fluxivar > 0) & (w1flux > 0)  # ADM quality in W1.

    lrg &= (gaiagmag == 0) | (gaiagmag > 18)  # remove bright GAIA sources

    # ADM remove stars with zfibertot < 17.5 that are missing from GAIA.
    lrg &= zfibertotflux < 10**(-0.4*(17.5-22.5))

    # ADM observed in every band.
    lrg &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # ADM default mask bits from the Legacy Surveys not set.
    lrg &= imaging_mask(maskbits)

    return lrg

def isLRG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 zfiberflux=None, ggood=None,
                 w2flux=None, primary=None, south=True):
    """(see, e.g., :func:`~desitarget.cuts.isLRG`).

    Notes:
        - the `ggood` and `w2flux` inputs are an attempt to maintain
          backwards-compatibility with the mocks.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    # ADM to maintain backwards-compatibility with mocks.
    if zfiberflux is None:
        print('Setting zfiberflux to zflux!!!') ## Altered in my code to keep compatibilty
        #log.warning('Setting zfiberflux to zflux!!!')
        zfiberflux = zflux.copy()

    gmag = 22.5 - 2.5 * np.log10(gflux.clip(1e-7))
    # ADM safe as these fluxes are set to > 0 in notinLRG_mask.
    rmag = 22.5 - 2.5 * np.log10(rflux.clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10(zflux.clip(1e-7))
    w1mag = 22.5 - 2.5 * np.log10(w1flux.clip(1e-7))
    zfibermag = 22.5 - 2.5 * np.log10(zfiberflux.clip(1e-7))

    # Full SV3 selection
    if south:
        lrg &= zmag - w1mag > 0.8 * (rmag - zmag) - 0.6  # non-stellar cut
        lrg &= zfibermag < 21.6                   # faint limit
        lrg &= (gmag - w1mag > 2.9) | (rmag - w1mag > 1.8)  # low-z cuts
        lrg &= (
            ((rmag - w1mag > (w1mag - 17.14) * 1.8)
             & (rmag - w1mag > (w1mag - 16.33) * 1.))
            | (rmag - w1mag > 3.3)
        )  # double sliding cuts and high-z extension
    else:
        lrg &= zmag - w1mag > 0.8 * (rmag - zmag) - 0.6  # non-stellar cut
        lrg &= zfibermag < 21.61                   # faint limit
        lrg &= (gmag - w1mag > 2.97) | (rmag - w1mag > 1.8)  # low-z cuts
        lrg &= (
            ((rmag - w1mag > (w1mag - 17.13) * 1.83)
             & (rmag - w1mag > (w1mag - 16.31) * 1.))
            | (rmag - w1mag > 3.4)
        )  # double sliding cuts and high-z extension

    return lrg

def isELG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 w2flux=None, gfiberflux=None, south=True, primary=None):
    """Color cuts for ELG target selection classes
    (see, e.g., :func:`~desitarget.cuts.set_target_bits` for parameters).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    # ADM work in magnitudes instead of fluxes. NOTE THIS IS ONLY OK AS
    # ADM the snr masking in ALL OF g, r AND z ENSURES positive fluxes.
    g = 22.5 - 2.5*np.log10(gflux.clip(1e-16))
    r = 22.5 - 2.5*np.log10(rflux.clip(1e-16))
    z = 22.5 - 2.5*np.log10(zflux.clip(1e-16))
    gfib = 22.5 - 2.5*np.log10(gfiberflux.clip(1e-16))

    # ADM cuts shared by the northern and southern selections.
    elg &= g > 20                       # bright cut.
    elg &= r - z > 0.15                  # blue cut.
#    elg &= r - z < 1.6                  # red cut.

    # ADM cuts that are unique to the north or south. Identical for sv3
    # ADM but keep the north/south formalism in case we use it later.
    if south:
        elg &= gfib < 24.1  # faint cut.
        elg &= g - r < 0.5*(r - z) + 0.1  # remove stars, low-z galaxies.
    else:
        elg &= gfib < 24.1  # faint cut.
        elg &= g - r < 0.5*(r - z) + 0.1  # remove stars, low-z galaxies.

    # ADM separate a low-priority and a regular sample.
    elgvlo = elg.copy()

    # ADM low-priority OII flux cut.
    elgvlo &= g - r < -1.2*(r - z) + 1.6
    elgvlo &= g - r >= -1.2*(r - z) + 1.3

    # ADM high-priority OII flux cut.
    elg &= g - r < -1.2*(r - z) + 1.3

    return elgvlo, elg

def notinELG_mask(maskbits=None, gsnr=None, rsnr=None, zsnr=None,
                  gnobs=None, rnobs=None, znobs=None, primary=None):
    """Standard set of masking cuts used by all ELG target selection classes.
    (see :func:`~desitarget.cuts.set_target_bits` for parameters).
    """
    if primary is None:
        primary = np.ones_like(maskbits, dtype='?')
    elg = primary.copy()

    # ADM good signal-to-noise in all bands.
    elg &= (gsnr > 0) & (rsnr > 0) & (zsnr > 0)

    # ADM observed in every band.
    elg &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # ADM default mask bits from the Legacy Surveys not set.
    elg &= imaging_mask(maskbits)

    return elg

def isELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          gfiberflux=None, gsnr=None, rsnr=None, zsnr=None,
          gnobs=None, rnobs=None, znobs=None,
          maskbits=None, south=True, primary=None):
    """Definition of ELG target classes. Returns a boolean array.
    (see :func:`~desitarget.cuts.set_target_bits` for parameters).

    Notes:
    - Current version (03/27/21) is version 8 on `the SV3 wiki`_.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    nomask = notinELG_mask(
        maskbits=maskbits, gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
        gnobs=gnobs, rnobs=rnobs, znobs=znobs, primary=primary)

    elgvlo, elg = isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                               w1flux=w1flux, w2flux=w2flux,
                               gfiberflux=gfiberflux, south=south,
                               primary=primary)

    return elgvlo & nomask, elg & nomask


def _is_row(table):
    """Return True/False if this is a table row instead of a full table.

    supports numpy.ndarray, astropy.io.fits.FITS_rec, astropy.table.Table
    """
    import astropy.io.fits.fitsrec
    import astropy.table.row
    if isinstance(table, (astropy.io.fits.fitsrec.FITS_record,
                          astropy.table.row.Row)) or \
       np.isscalar(table):
        return True
    else:
        return False

def shift_photo_north(gflux=None, rflux=None, zflux=None):
    """Convert fluxes in the northern (BASS/MzLS) to the southern (DECaLS) system.

    Parameters
    ----------
    gflux, rflux, zflux : :class:`array_like` or `float`
        The flux in nano-maggies of g, r, z bands.

    Returns
    -------
    The equivalent fluxes shifted to the southern system.

    Notes
    -----
    - see also https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=3390;filename=Raichoor_DESI_05Dec2017.pdf;version=1
    - Update for DR9 https://desi.lbl.gov/trac/attachment/wiki/TargetSelectionWG/TargetSelection/North_vs_South_dr9.png
    """
    # ADM if floats were sent, treat them like arrays.
    flt = False
    if _is_row(gflux):
        flt = True
        gflux = np.atleast_1d(gflux)
        rflux = np.atleast_1d(rflux)
        zflux = np.atleast_1d(zflux)

    # ADM only use the g-band color shift when r and g are non-zero
    gshift = gflux * 10**(-0.4*0.004)
    w = np.where((gflux != 0) & (rflux != 0))
    gshift[w] = (gflux[w] * 10**(-0.4*0.004) * (gflux[w]/rflux[w])**complex(-0.059)).real

    # ADM only use the r-band color shift when r and z are non-zero
    # ADM and only use the z-band color shift when r and z are non-zero
    w = np.where((rflux != 0) & (zflux != 0))
    rshift = rflux * 10**(0.4*0.003)
    zshift = zflux * 10**(0.4*0.013)

    rshift[w] = (rflux[w] * 10**(0.4*0.003) * (rflux[w]/zflux[w])**complex(-0.024)).real
    zshift[w] = (zflux[w] * 10**(0.4*0.013) * (rflux[w]/zflux[w])**complex(+0.015)).real

    if flt:
        return gshift[0], rshift[0], zshift[0]

    return gshift, rshift, zshift


def _psflike(psftype):
    """ If the object is PSF """
    # ADM explicitly checking for NoneType. In the past we have had bugs
    # ADM where we forgot to pass objtype=objtype in, e.g., isSTD.
    if psftype is None:
        msg = "NoneType submitted to _psfflike function"
        print(msg)
        #log.critical(msg)
        raise ValueError(msg)

    psftype = np.asarray(psftype)
    # ADM in Python3 these string literals become byte-like
    # ADM so to retain Python2 compatibility we need to check
    # ADM against both bytes and unicode.
    # ADM Also 'PSF' for astropy.io.fits; 'PSF ' for fitsio (sigh).
    psflike = ((psftype == 'PSF') | (psftype == b'PSF') |
               (psftype == 'PSF ') | (psftype == b'PSF '))

    return psflike

def isQSO_cuts(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
               w1snr=None, w2snr=None, maskbits=None,
               gnobs=None, rnobs=None, znobs=None,
               objtype=None, primary=None, optical=False, south=True):
    """QSO targets from color cuts. Returns a boolean array.

    Parameters
    ----------
    optical : :class:`boolean`, defaults to ``False``
        Apply just optical color-cuts.
    south : :class:`boolean`, defaults to ``True``
        Use cuts for Northern imaging (BASS/MzLS) if ``south=False``,
        otherwise use cuts for Southern imaging (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` for objects passing quasar color/morphology/logic cuts.

    Notes
    -----
    - Current version (06/05/19) is version 176 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """

    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                       w1flux=w1flux, w2flux=w2flux,
                       optical=optical, south=south)

    if south:
        qso &= w1snr > 4
        qso &= w2snr > 2
    else:
        qso &= w1snr > 4
        qso &= w2snr > 3

    # ADM observed in every band.
    qso &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    if primary is not None:
        qso &= primary

    if objtype is not None: # Needed???
        qso &= _psflike(objtype)

    # ADM default mask bits from the Legacy Surveys not set.
    qso &= imaging_mask(maskbits)

    return qso

def isQSO_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 optical=False, south=True):
    """Tests if sources have quasar-like colors in a color box.
    (see, e.g., :func:`~desitarget.cuts.isQSO_cuts`).
    """
    # ----- Quasars
    # Create some composite fluxes.
    wflux = 0.75*w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3

    qso = np.ones_like(gflux, dtype='?')
    qso &= rflux < 10**((22.5-17.5)/2.5)    # r>17.5
    qso &= rflux > 10**((22.5-22.7)/2.5)    # r<22.7
    qso &= grzflux < 10**((22.5-17)/2.5)    # grz>17
    qso &= rflux < gflux * 10**(1.3/2.5)    # (g-r)<1.3
    qso &= zflux > rflux * 10**(-0.4/2.5)   # (r-z)>-0.4
    qso &= zflux < rflux * 10**(1.1/2.5)    # (r-z)<1.1

    if not optical:
        if south:
            qso &= w2flux > w1flux * 10**(-0.4/2.5)              # (W1-W2)>-0.4
        else:
            qso &= w2flux > w1flux * 10**(-0.3/2.5)              # (W1-W2)>-0.3
        # (grz-W)>(g-z)-1.0
        qso &= wflux * gflux > zflux * grzflux * 10**(-1.0/2.5)

    # Harder cut on stellar contamination
    mainseq = rflux > gflux * 10**(0.20/2.5)  # g-r>0.2

    # Clip to avoid warnings for -ve numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    mainseq &= rflux**(1+1.53) > gflux * zflux**1.53 * 10**((-0.100+0.20)/2.5)
    mainseq &= rflux**(1+1.53) < gflux * zflux**1.53 * 10**((+0.100+0.20)/2.5)
    if not optical:
        mainseq &= w2flux < w1flux * 10**(0.3/2.5)
    qso &= ~mainseq

    return qso




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

#for brick in range(600):
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

#print()
#print("=============================== Download South Completed ==================================")
#print()
#
#for brick in range(300):
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
#print()
#print("=============================== Download North Completed ==================================")
#print()


print()
print("=============================== Classification South... ==================================")
print()

bricknames_south_sample = []

for filename in os.listdir('/Volumes/Astrostick/bricks_data/south/'):
    brickn = filename.replace("tractor-", "")
    brickn = brickn.replace(".fits", "")
    bricknames_south_sample.append(brickn)


for no, brickname in enumerate(bricknames_south_sample):
    df = pd.read_csv('../bricks_data/galaxy_catalogue_sample.csv')

    brickid = brickid_south[np.where(brickname_south == brickname)]
    if len(brickid > 0):
        brickid = brickid[0]
    else:
        brickid = 0
        ##### Check tomorrow how there can be a brickname without corresponding id

    hdulistSingleBrick = fits.open(f'/Volumes/Astrostick/bricks_data/south/tractor-{brickname}.fits')
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


    south = south_survey_is_south[np.where(brickid_south == brickid)]
    if len(south) > 0:
        south = south[0]
    else:
        south = True

    # Do not forget to check this clause
    #['BrickID', 'ObjectID','RA', 'DEC', 'South', 'Target_type']

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

    if no % 20 == 0:
        print(no, " of ", len(bricknames_south_sample), "bricks processed")

    df.to_csv('../bricks_data/galaxy_catalogue_sample.csv', index=False)
    print(" ===================== Brick", brickname, " complete=====================")


print()
print("=============================== Classification South Completed ==================================")
print()
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
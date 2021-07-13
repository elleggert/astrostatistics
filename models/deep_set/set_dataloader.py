import numpy as np
from astropy.io import fits
from sklearn.preprocessing import LabelEncoder


# noinspection PyAttributeOutsideInit
class CCD:
    def __init__(self):

        decamCCD = fits.open('../../bricks_data/ccds-annotated-decam-dr9.fits')
        mosaicCCD = fits.open('../../bricks_data/ccds-annotated-mosaic-dr9.fits')
        bassCCD = fits.open('../../bricks_data/ccds-annotated-90prime-dr9.fits')
        self.dataDecam = decamCCD[1].data
        self.dataMosaic = mosaicCCD[1].data
        self.dataBass = bassCCD[1].data

        self.initialise_systematics()

        self.initalise_boundaries()

        self.encode_categoricals()

        self.stack_systematics()

    def initalise_boundaries(self):
        ra0 = np.concatenate((self.dataDecam.field('ra0'), self.dataMosaic.field('ra0'), self.dataBass.field('ra0')), axis=0)
        dec0 = np.concatenate((self.dataDecam.field('dec0'), self.dataMosaic.field('dec0'), self.dataBass.field('dec0')), axis=0)
        ra1 = np.concatenate((self.dataDecam.field('ra1'), self.dataMosaic.field('ra1'), self.dataBass.field('ra1')), axis=0)
        dec1 = np.concatenate((self.dataDecam.field('dec1'), self.dataMosaic.field('dec1'), self.dataBass.field('dec1')), axis=0)
        ra2 = np.concatenate((self.dataDecam.field('ra2'), self.dataMosaic.field('ra2'), self.dataBass.field('ra2')), axis=0)
        dec2 = np.concatenate((self.dataDecam.field('dec2'), self.dataMosaic.field('dec2'), self.dataBass.field('dec2')), axis=0)
        ra3 = np.concatenate((self.dataDecam.field('ra3'), self.dataMosaic.field('ra3'), self.dataBass.field('ra3')), axis=0)
        dec3 = np.concatenate((self.dataDecam.field('dec3'), self.dataMosaic.field('dec3'), self.dataBass.field('dec3')), axis=0)

    def initialise_systematics(self):
        # Extracting systematics
        self.filter_colour = np.concatenate(
            (self.dataDecam.field('filter'), self.dataMosaic.field('filter'), self.dataBass.field('filter')), axis=0)
        self.camera = np.concatenate((self.dataDecam.field('camera'), self.dataMosaic.field('camera'), self.dataBass.field('camera')),
                                     axis=0)
        self.exptime = np.concatenate(
            (self.dataDecam.field('exptime'), self.dataMosaic.field('exptime'), self.dataBass.field('exptime')), axis=0)
        self.airmass = np.concatenate(
            (self.dataDecam.field('airmass'), self.dataMosaic.field('airmass'), self.dataBass.field('airmass')), axis=0)
        self.fwhm = np.concatenate((self.dataDecam.field('fwhm'), self.dataMosaic.field('fwhm'), self.dataBass.field('fwhm')), axis=0)
        self.seeing = self.fwhm * 0.262
        # self.skyrms = np.concatenate((dataDecam.field('skyrms'), dataMosaic.field('skyrms'), dataBass.field('skyrms')), axis=0)
        # self.sig1 = np.concatenate((dataDecam.field('sig1'), dataMosaic.field('sig1'), dataBass.field('sig1')), axis = 0)
        # self.ccdskycounts = np.concatenate((dataDecam.field('ccdskycounts'), dataMosaic.field('ccdskycounts'), dataBass.field('ccdskycounts')), axis = 0)
        # self.ccdskysb = np.concatenate((dataDecam.field('ccdskysb'), dataMosaic.field('ccdskysb'), dataBass.field('ccdskysb')), axis = 0)
        # self.ccdphrms = np.concatenate((dataDecam.field('ccdphrms'), dataMosaic.field('ccdphrms'), dataBass.field('ccdphrms')), axis = 0)
        # self.phrms = np.concatenate((dataDecam.field('phrms'), dataMosaic.field('phrms'), dataBass.field('phrms')), axis = 0)
        # self.ccdnastrom = np.concatenate((dataDecam.field('ccdnastrom'), dataMosaic.field('ccdnastrom'), dataBass.field('ccdnastrom')), axis = 0)
        # self.ccdnphotom = np.concatenate((dataDecam.field('ccdnphotom'), dataMosaic.field('ccdnphotom'), dataBass.field('ccdnphotom')), axis = 0)
        # self.meansky = np.concatenate((dataDecam.field('meansky'), dataMosaic.field('meansky'), dataBass.field('meansky')), axis = 0)
        # self.stdsky = np.concatenate((dataDecam.field('stdsky'), dataMosaic.field('stdsky'), dataBass.field('stdsky')), axis = 0)
        # self.maxsky = np.concatenate((dataDecam.field('maxsky'), dataMosaic.field('maxsky'), dataBass.field('maxsky')), axis = 0)
        # self.minsky = np.concatenate((dataDecam.field('minsky'), dataMosaic.field('minsky'), dataBass.field('minsky')), axis = 0)
        # self.pixscale_mean = np.concatenate((dataDecam.field('pixscale_mean'), dataMosaic.field('pixscale_mean'), dataBass.field('pixscale_mean')), axis = 0)
        # self.pixscale_std = np.concatenate((dataDecam.field('pixscale_std'), dataMosaic.field('pixscale_std'), dataBass.field('pixscale_std')), axis = 0)
        # self.pixscale_max = np.concatenate((dataDecam.field('pixscale_max'), dataMosaic.field('pixscale_max'), dataBass.field('pixscale_max')), axis = 0)
        # self.pixscale_min = np.concatenate((dataDecam.field('pixscale_min'), dataMosaic.field('pixscale_min'), dataBass.field('pixscale_min')), axis = 0)
        # self.galnorm_mean = np.concatenate((dataDecam.field('galnorm_mean'), dataMosaic.field('galnorm_mean'), dataBass.field('galnorm_mean')), axis = 0)
        # self.galnorm_std = np.concatenate((dataDecam.field('galnorm_std'), dataMosaic.field('galnorm_std'), dataBass.field('galnorm_std')), axis = 0)
        # self.humidity = np.concatenate((dataDecam.field('humidity'), dataMosaic.field('humidity'), dataBass.field('humidity')), axis = 0)
        # self.outtemp = np.concatenate((dataDecam.field('outtemp'), dataMosaic.field('outtemp'), dataBass.field('outtemp')), axis = 0)
        # self.tileebv = np.concatenate((dataDecam.field('tileebv'), dataMosaic.field('tileebv'), dataBass.field('tileebv')), axis = 0)
        # self.ebv = np.concatenate((dataDecam.field('ebv'), dataMosaic.field('ebv'), dataBass.field('ebv')), axis = 0)
        # self.galdepth = np.concatenate((dataDecam.field('galdepth'), dataMosaic.field('galdepth'), dataBass.field('galdepth')), axis = 0)
        # self.gaussgaldepth = np.concatenate((dataDecam.field('gaussgaldepth'), dataMosaic.field('gaussgaldepth'), dataBass.field('gaussgaldepth')), axis = 0)

    def encode_categoricals(self):

        encoder = LabelEncoder()
        encoder.fit(np.unique(self.camera))
        self.camera = encoder.transform(self.camera)

        encoder.fit(self.filter_colour)
        self.filter_colour = encoder.transform(self.filter_colour)

    def stack_systematics(self):
        self.data = np.stack((self.filter_colour, self.camera, self.exptime, self.airmass, self.seeing), axis=1)

    def get_ccds(self, ids):
        return self.data[ids]



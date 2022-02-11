"""Interface to provide access to the annotated CCDs in the legacy surveys"""

import numpy as np
from astropy.io import fits
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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

    def initialise_for_deepset(self):
        # Do these two if you want to prepare a dataset that is already scaled and has filter colour encoded
        self.stack_scale_systematics()
        self.encode_add_categoricals()

    def initalise_boundaries(self):
        self.ra = self.concat_surveys('ra')
        self.dec = self.concat_surveys('dec')
        self.ra0 = self.concat_surveys('ra0')
        self.dec0 = self.concat_surveys('dec0')
        self.ra1 = self.concat_surveys('ra1')
        self.dec1 = self.concat_surveys('dec1')
        self.ra2 = self.concat_surveys('ra2')
        self.dec2 = self.concat_surveys('dec2')
        self.ra3 = self.concat_surveys('ra3')
        self.dec3 = self.concat_surveys('dec3')

    def concat_surveys(self, field):
        return np.concatenate((self.dataDecam.field(field),
                               self.dataMosaic.field(field),
                               self.dataBass.field(field)),
                              axis=0)

    def initialise_systematics(self):
        # Extracting systematics
        self.filter_colour = self.concat_surveys('filter')
        # self.camera = self.concat_surveys('camera')
        # self.exptime = self.concat_surveys('exptime')
        self.airmass = self.concat_surveys('airmass')
        self.seeing = self.concat_surveys('fwhm') * 0.262
        self.ccdskysb = self.concat_surveys('ccdskysb')
        # self.galdepth = self.concat_surveys('galdepth')
        # self.ebv = self.concat_surveys('ebv')
        # self.ccdnphotom = self.concat_surveys('ccdnphotom')
        # self.skyrms = self.concat_surveys('skyrms')
        self.ccdskycounts = self.concat_surveys('ccdskycounts')
        # self.meansky = self.concat_surveys('meansky')
        # self.pixscale_mean = self.concat_surveys('pixscale_mean')
        # self.ccdnastrom = self.concat_surveys('ccdnastrom')
        # self.mjd_obs = self.concat_surveys('mjd_obs')
        # self.sig1 = self.concat_surveys('sig1')
        # self.ccd_cuts = self.concat_surveys('ccd_cuts')

        self.sys_tuple = (self.airmass,
                          self.seeing,
                          self.ccdskysb,
                          self.ccdskycounts)
        self.no_ccds = len(self.filter_colour)
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

    def encode_add_categoricals(self):
        # Function to encode and add the categorical band metric to the image
        encoder = LabelEncoder()
        encoder.fit(self.filter_colour)
        self.filter_colour_encoded = encoder.transform(self.filter_colour)
        self.filter_colour_encoded = self.filter_colour_encoded[:, np.newaxis]

        # Add encoded categoricals
        self.data = np.concatenate((self.data, self.filter_colour_encoded), axis=1)
        self.num_features = self.data.shape[1]
        print(self.num_features)

    def stack_scale_systematics(self):
        # Do this one if you do want scaled inputs and only stack the important metrics

        self.data = np.stack(self.sys_tuple,
                             axis=1)
        self.scaler_in = MinMaxScaler()

        self.data = self.scaler_in.fit_transform(self.data)
        self.num_features = self.data.shape[1]

    def stack_systematics(self):
        # Do this one if you do not want scaled inputs and only stack the important metrics without filter colour
        self.data = np.stack(self.sys_tuple,
                             axis=1)
        # Add filter colour
        self.data = np.concatenate((self.data, self.filter_colour), axis=1)
        self.num_features = self.data.shape[1]

    def get_ccds(self, ids):
        return self.data[ids]

    def get_all_boundaries(self):
        # Function to return Pixel Boundaries
        return self.ra0, self.dec0, self.ra1, self.dec1, self.ra2, self.dec2, self.ra3, self.dec3

    def get_ccd_boundaries(self, ids):
        xs = [self.ra0[ids], self.ra1[ids], self.ra2[ids], self.ra3[ids], self.ra0[ids]]
        ys = [self.dec0[ids], self.dec1[ids], self.dec2[ids], self.dec3[ids], self.dec0[ids]]
        return xs, ys


    def get_filter_mask(self, colour):
        # Function to return only mask for images of a certain band (pass 'g', 'r' or 'z')
        m = (self.filter_colour == colour)
        return m




import numpy as np
import time
from desitarget.cuts import isLRG, isELG, isQSO_cuts, isQSO_randomforest, apply_cuts


class Brick:
    'Represents all attributes necessary to classify all galaxies in a brick into their categories'

    def __init__(self, data):
        self.data = data
        self.no_of_objects = 0
        """ The init method does not intialise any of the elements, will be initialised depending on the usage of the 
        brick e.g. galaxy classification / stellar density calculation to avoid unnecessary storage"""

        self.ra = None
        self.dec = None
        self.ids = None
        self.objid = None
        self.flux_g = None
        self.flux_r = None
        self.flux_z = None
        self.flux_w1 = None
        self.flux_w2 = None
        self.fiberflux_g = None
        self.fiberflux_r = None
        self.fiberflux_z = None
        self.fibertotflux_g = None
        self.fibertotflux_r = None
        self.fibertotflux_z = None
        self.flux_ivar_g = None
        self.flux_ivar_r = None
        self.flux_ivar_z = None
        self.flux_ivar_w1 = None
        self.flux_ivar_w2 = None
        self.mw_transmission_g = None
        self.mw_transmission_r = None
        self.mw_transmission_z = None
        self.mw_transmission_w1 = None
        self.mw_transmission_w2 = None
        self.gaia_g_mag = None
        self.gaia_b_mag = None
        self.gaia_r_mag = None
        self.nobs_g = None
        self.nobs_r = None
        self.nobs_z = None
        self.maskbits = None
        self.fitbits = None
        self.snr_g = None
        self.snr_r = None
        self.snr_z = None
        self.snr_w1 = None
        self.snr_w2 = None
        self.mag_g = None
        self.mag_r = None
        self.mag_z = None
        self.south = None
        self.type = None

    def initialise_brick_for_galaxy_classification(self, south=None):
        """Calls different Initialisation Methods"""
        self.initialise_position()
        self.initialise_ids()
        self.initialise_fluxes()
        self.initialise_no_observations_maskbits()
        self.initialise_gaia_magnitudes()
        self.extinction_correction()
        self.calculate_signal_to_noise()
        self.set_south(south)
        self.no_of_objects = len(self.flux_g)

    def initialise_ids(self):
        """Initialise the object parameters"""
        self.ids = self.data.field('brickid')
        self.objid = self.data.field('objid')

    def initialise_position(self):
        """Extracts the positions of all objects in the brick"""
        self.ra = self.data.field('ra')
        self.dec = self.data.field('dec')

    def initialise_brick_for_stellar_density(self):
        """Helper functions for stellar extraction are invoked"""
        self.initialise_position()
        self.initialise_fluxes()
        self.initialise_type()
        self.initialise_magnitudes_colours_uncorrected()
        self.no_of_objects = len(self.flux_g)

    def initialise_magnitudes_colours_uncorrected(self):
        """ Magnitudes are extracted"""
        self.mag_g = 22.5 - 2.5 * np.log10(self.flux_g.clip(1e-7))
        self.mag_r = 22.5 - 2.5 * np.log10(self.flux_r.clip(1e-7))
        self.mag_z = 22.5 - 2.5 * np.log10(self.flux_z.clip(1e-7))

    def calculate_signal_to_noise(self):
        """Get the Signal-to-noise in g, r, z, W1 and W2 defined as the flux per
        band divided by sigma (flux x sqrt of the inverse variance)."""
        self.snr_g = self.flux_g * np.sqrt(self.flux_ivar_g)
        self.snr_r = self.flux_r * np.sqrt(self.flux_ivar_r)
        self.snr_z = self.flux_z * np.sqrt(self.flux_ivar_z)
        self.snr_w1 = self.flux_w1 * np.sqrt(self.flux_ivar_w1)
        self.snr_w2 = self.flux_w2 * np.sqrt(self.flux_ivar_w2)

    def initialise_gaia_magnitudes(self):
        """Initialise the Gaia-based g-, b- and r-band MAGNITUDES."""
        self.gaia_g_mag = self.data.field('gaia_phot_g_mean_mag')
        self.gaia_b_mag = self.data.field('gaia_phot_bp_mean_mag')
        self.gaia_r_mag = self.data.field('gaia_phot_rp_mean_mag')

    def initialise_no_observations_maskbits(self):
        # Initialise the Number of observations for that brick
        self.nobs_g = self.data.field('nobs_g')
        self.nobs_r = self.data.field('nobs_r')
        self.nobs_z = self.data.field('nobs_z')
        # Retrieving the maskbits and fitbits for quasar detection
        self.maskbits = self.data.field('maskbits')
        self.fitbits = self.data.field('fitbits')

    def initialise_fluxes(self):
        """ Initialise the various fluxes, fiber fluxes and inverse variance of the fluxes in the different bands"""

        self.flux_g = self.data.field('flux_g')
        self.flux_r = self.data.field('flux_r')
        self.flux_z = self.data.field('flux_z')
        self.flux_w1 = self.data.field('flux_w1')
        self.flux_w2 = self.data.field('flux_w2')

        # getting predicted -band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing
        self.fiberflux_g = self.data.field('fiberflux_g')
        self.fiberflux_r = self.data.field('fiberflux_r')
        self.fiberflux_z = self.data.field('fiberflux_z')

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

    def extinction_correction(self):
        """ Retrieves the milky_way transmission and uses it to correct given fluxes for extinction"""
        self.mw_transmission_g = self.data.field('mw_transmission_g')
        self.mw_transmission_r = self.data.field('mw_transmission_r')
        self.mw_transmission_z = self.data.field('mw_transmission_z')
        self.mw_transmission_w1 = self.data.field('mw_transmission_w1')
        self.mw_transmission_w2 = self.data.field('mw_transmission_w2')
        # correcting for extinction ---> divide by the transmission
        self.flux_g = self.flux_g / self.mw_transmission_g
        self.flux_r = self.flux_r / self.mw_transmission_r
        self.flux_z = self.flux_z / self.mw_transmission_z
        self.flux_w1 = self.flux_w1 / self.mw_transmission_w1
        self.flux_w2 = self.flux_w2 / self.mw_transmission_w2

        self.fiberflux_g = self.fiberflux_g / self.mw_transmission_g
        self.fiberflux_r = self.fiberflux_r / self.mw_transmission_r
        self.fiberflux_z = self.fiberflux_z / self.mw_transmission_z

    def set_south(self, south):
        # Initialises the boolean whether the given brick is south or north
        self.south = south

    def initialise_type(self):
        self.type = self.data.field('type')


    def classify_galaxies(self):
        """ Function goes through every object in the brick and defines whether it is of LRG, ELG or QSO type """

        """ Returns: an np array with boolean flags for different galaxy types. Calls the DESI Helper functions"""
        target_lrg = np.zeros(self.no_of_objects)
        target_elg = np.zeros(self.no_of_objects)
        target_qso = np.zeros(self.no_of_objects)

        is_LRG = isLRG(gflux=self.flux_g, rflux=self.flux_r, zflux=self.flux_z, w1flux=self.flux_w1,
                       w2flux=self.flux_w2,
                       zfiberflux=self.fiberflux_z, rfluxivar=self.flux_ivar_r, zfluxivar=self.flux_ivar_z,
                       w1fluxivar=self.flux_ivar_w1, gaiagmag=self.gaia_g_mag, gnobs=self.nobs_g, rnobs=self.nobs_r,
                       znobs=self.nobs_z, maskbits=self.maskbits, zfibertotflux=self.fibertotflux_z, primary=None,
                       south=self.south)

        target_lrg[np.where(is_LRG == True)] = 1


        is_ELG, is_ELGVLO = isELG(gflux=self.flux_g, rflux=self.flux_r, zflux=self.flux_z, w1flux=self.flux_w1,
                                  w2flux=self.flux_w2,
                                  gfiberflux=self.fiberflux_g, gsnr=self.snr_g, rsnr=self.snr_r, zsnr=self.snr_z,
                                  gnobs=self.nobs_g, rnobs=self.nobs_r, znobs=self.nobs_z,
                                  maskbits=self.maskbits, south=self.south)

        target_elg[np.where(is_ELG == True)] = 1
        target_elg[np.where(is_ELGVLO == True)] = 1


        is_QSO = isQSO_cuts(gflux=self.flux_g, rflux=self.flux_r, zflux=self.flux_z, w1flux=self.flux_w1,
                            w2flux=self.flux_w2,
                            w1snr=self.snr_w1, w2snr=self.snr_w2, maskbits=self.maskbits,
                            gnobs=self.nobs_g, rnobs=self.nobs_r, znobs=self.nobs_z,
                            objtype=self.type, primary=None, optical=False, south=self.south)

        target_qso[np.where(is_QSO == True)] = 1

        target_objects = np.stack((self.ids, self.ra, self.dec, target_lrg,target_elg,target_qso), axis=1)
        mask = (target_lrg == True) | (target_elg == True) | (target_qso == True)
        target_objects = target_objects[mask]
        return target_objects

    def get_stellar_objects(self):
        """ Classifies Stellar Objects """
        is_PSF = (self.type == 'PSF') & (self.mag_r > 17) & (self.mag_r < 18)
        stacked_array = np.stack((self.ra, self.dec, self.mag_g, self.mag_r, self.mag_z),
                                 axis=1)

        return stacked_array[np.where(is_PSF == True)]


    def append_brick(self, data, south):
        """Helper Function that was used when extracting using Random Forests to speed up the process"""
        brick = Brick(data)
        brick.initialise_brick_for_galaxy_classification(south)
        self.ra = np.append(self.ra , brick.ra )
        self.dec = np.append(self.dec , brick.dec )
        self.ids = np.append(self.ids , brick.ids)
        self.objid = np.append(self.objid , brick.objid )
        self.flux_g = np.append(self.flux_g , brick.flux_g )
        self.flux_r = np.append(self.flux_r , brick.flux_r)
        self.flux_z = np.append(self.flux_z, brick.flux_z )
        self.flux_w1 = np.append(self.flux_w1 , brick.flux_w1 )
        self.flux_w2 = np.append(self.flux_w2 , brick.flux_w2 )
        self.fiberflux_g = np.append(self.fiberflux_g , brick.fiberflux_g )
        self.fiberflux_r = np.append(self.fiberflux_r , brick.fiberflux_r )
        self.fiberflux_z = np.append(self.fiberflux_z , brick.fiberflux_z )
        self.fibertotflux_g = np.append(self.fibertotflux_g , brick.fibertotflux_g )
        self.fibertotflux_r = np.append(self.fibertotflux_r , brick.fibertotflux_r )
        self.fibertotflux_z = np.append(self.fibertotflux_z , brick.fibertotflux_z )
        self.flux_ivar_g = np.append(self.flux_ivar_g , brick.flux_ivar_g )
        self.flux_ivar_r = np.append(self.flux_ivar_r , brick.flux_ivar_r )
        self.flux_ivar_z = np.append(self.flux_ivar_z , brick.flux_ivar_z )
        self.flux_ivar_w1 = np.append(self.flux_ivar_w1 , brick.flux_ivar_w1 )
        self.flux_ivar_w2 = np.append(self.flux_ivar_w2 , brick.flux_ivar_w2 )
        self.mw_transmission_g = np.append(self.mw_transmission_g , brick.mw_transmission_g )
        self.mw_transmission_r = np.append(self.mw_transmission_r , brick.mw_transmission_r )
        self.mw_transmission_z = np.append(self.mw_transmission_z , brick.mw_transmission_z )
        self.mw_transmission_w1 = np.append(self.mw_transmission_w1 , brick.mw_transmission_w1 )
        self.mw_transmission_w2 = np.append(self.mw_transmission_w2 , brick.mw_transmission_w2 )
        self.gaia_g_mag = np.append(self.gaia_g_mag , brick.gaia_g_mag )
        self.gaia_b_mag = np.append(self.gaia_b_mag , brick.gaia_b_mag )
        self.gaia_r_mag = np.append(self.gaia_r_mag, brick.gaia_r_mag )
        self.nobs_g = np.append(self.nobs_g , brick.nobs_g )
        self.nobs_r = np.append(self.nobs_r , brick.nobs_r )
        self.nobs_z = np.append(self.nobs_z, brick.nobs_z )
        self.maskbits = np.append(self.maskbits , brick.maskbits )
        self.fitbits = np.append(self.fitbits , brick.fitbits )
        self.snr_g = np.append(self.snr_g , brick.snr_g )
        self.snr_r = np.append(self.snr_r , brick.snr_r )
        self.snr_z = np.append(self.snr_z , brick.snr_z )
        self.snr_w1 = np.append(self.snr_w1 , brick.snr_w1 )
        self.snr_w2 = np.append(self.snr_w2 , brick.snr_w2 )
        self.mag_g = np.append(self.mag_g , brick.mag_g )
        self.mag_r = np.append(self.mag_r , brick.mag_r )
        self.mag_z = np.append(self.mag_z , brick.mag_z )
        #self.gmr = np.append(self.gmr , brick.gmr)
        #self.rmz = np.append(self.rmz , brick.rmz)
        #self.south = np.append(self.south , brick.south )
        self.type = np.append(self.type , brick.type )
        self.no_of_objects = len(self.flux_r)





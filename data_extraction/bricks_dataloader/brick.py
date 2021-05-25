import numpy as np

from galaxy_classification import isLRG, isELG, isQSO_cuts


class Brick:
    'Represents all attributes necessary to classify all galaxies in a brick into their categories'

    def __init__(self, data):
        # The init method calls other methods to initialise the brick, considered bad practice to do so but with
        # over 25 attributes this looked more than messy
        self.data = data
        self.initialise_brick()

    def initialise_brick(self):
        """Calls different Initialisation Methods"""
        self.initialise_fluxes()
        self.initialise_no_observations_maskbits()
        self.initialise_gaia_magnitudes()
        self.extinction_correction()
        self.calculate_signal_to_noise()
        self.south = None

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
        # Retrieving the maskbits for quasar detection and boolean for brick is in north
        self.maskbits = self.data.field('maskbits')

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
        # Initialises the boolean whether the given brick is south
        self.south = south

    def classify_galaxies(self):
        """ Function goes through every object in the brick and defines whether it is of LRG, ELG or QSO type """

        """ Returns: an np array with 0 indicating no galaxy type of interest, 1 = LRG; 2 = ELG, 3 = QSO """
        target_type = np.zeros(len(self.flux_g))

        is_LRG = isLRG(gflux=self.flux_g, rflux=self.flux_r, zflux=self.flux_z, w1flux=self.flux_w1,
                       w2flux=self.flux_w2,
                       zfiberflux=self.fiberflux_z, rfluxivar=self.flux_ivar_r, zfluxivar=self.flux_ivar_z,
                       w1fluxivar=self.flux_ivar_w1, gaiagmag=self.gaia_g_mag, gnobs=self.nobs_g, rnobs=self.nobs_r,
                       znobs=self.nobs_z, maskbits=self.maskbits, zfibertotflux=self.fibertotflux_z, primary=None,
                       south=self.south)

        target_type[np.where(is_LRG == True)] = 1

        is_ELG, is_ELGVLO = isELG(gflux=self.flux_g, rflux=self.flux_r, zflux=self.flux_z, w1flux=self.flux_w1,
                                  w2flux=self.flux_w2,
                                  gfiberflux=self.fiberflux_g, gsnr=self.snr_g, rsnr=self.snr_r, zsnr=self.snr_z,
                                  gnobs=self.nobs_g, rnobs=self.nobs_r, znobs=self.nobs_z,
                                  maskbits=self.maskbits, south=self.south)

        target_type[np.where(is_ELG == True)] = 2
        target_type[np.where(is_ELGVLO == True)] = 2

        is_QSO = isQSO_cuts(gflux=self.flux_g, rflux=self.flux_r, zflux=self.flux_z, w1flux=self.flux_w1,
                            w2flux=self.flux_w2,
                            w1snr=self.snr_w1, w2snr=self.snr_w2, maskbits=self.maskbits,
                            gnobs=self.nobs_g, rnobs=self.nobs_r, znobs=self.nobs_z,
                            objtype=None, primary=None, optical=False, south=self.south)

        target_type[np.where(is_QSO == True)] = 3

        return target_type
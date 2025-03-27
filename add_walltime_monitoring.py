import bagpipes as pipes
import numpy as np
import time
from bagpipes.models.model_galaxy import model_galaxy

# Code from Adam Carnall, adding option to check each
# likelihood call's walltime and calculation time
# Allows for checking if code is running efficiently when parallized
def lnlike(self, x, ndim=0, nparam=0):
        """ Returns the log-likelihood for a given parameter vector. """

        if self.time_calls:
            time0 = time.time()

            if self.n_calls == 0:
                self.wall_time0 = time.time()

        # Update the model_galaxy with the parameters from the sampler.
        self._update_model_components(x)

        if self.model_galaxy is None:
            self.model_galaxy = model_galaxy(self.model_components,
                                             filt_list=self.galaxy.filt_list,
                                             spec_wavs=self.galaxy.spec_wavs,
                                             index_list=self.galaxy.index_list)

        self.model_galaxy.update(self.model_components)

        # Return zero likelihood if SFH is older than the universe.
        if self.model_galaxy.sfh.unphysical:
            return -9.99*10**99

        lnlike = 0.

        if self.galaxy.spectrum_exists and self.galaxy.index_list is None:
            lnlike += self._lnlike_spec()

        if self.galaxy.photometry_exists:
            lnlike += self._lnlike_phot()

        if self.galaxy.index_list is not None:
            lnlike += self._lnlike_indices()

        # Return zero likelihood if lnlike = nan (something went wrong).
        if np.isnan(lnlike):
            print("Bagpipes: lnlike was nan, replaced with zero probability.")
            return -9.99*10**99

        # Functionality for timing likelihood calls.
        if self.time_calls:
            self.times[self.n_calls] = time.time() - time0
            self.n_calls += 1

            if self.n_calls == 1000:
                self.n_calls = 0
                print("Mean likelihood call time:", np.round(np.mean(self.times), 4))
                print("Wall time per lnlike call:", np.round((time.time() - self.wall_time0)/1000., 4))

        return lnlike

pipes.fitting.fitted_model.lnlike = lnlike

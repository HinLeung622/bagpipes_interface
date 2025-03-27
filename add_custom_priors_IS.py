import numpy as np
import os
import pymultinest as pmn
import deepdish as dd
from copy import deepcopy
import warnings
import json
import time
import sys
import bagpipes as pipes

from scipy.stats import norm
#from scipy.stats import loguniform  # vela does not support a high enough version of scipy

# add some functions into bagpipes to sample from a custom prior through importance sampling
"""
Structure:
Within any free parameter being fitted, the prior is specified through:
As an example:
sfh['massformed'] = (6,13)                 # prior limits
sfh['massformed_prior'] = "Gaussian"       # the functional form of the prior
sfh['massformed_prior_mu'] = 10            # mean of the Gaussian distribution
sfh['massformed_prior_sigma'] = 0.5        # std of the Gaussian distribution

To turn on importance sampling, we pass two additional inputs to the dictionary
# PDF of the custom prior in the form of a callable function with inputs (x,args)
sfh['massformed_prior_IS_func'] = custom_prior
# the arguments that go into the rest of the callable function, in list form
sfh['massformed_prior_IS_params'] = custom_prior_args

For every likelihood evalution, the factor log10(P(theta)/g(theta)) will then be
added to the log likelihood of the data given parameters theta, where
P(theta) is the custom prior's PDF as given, and g(theta) is the PDF of the
uninformatie prior that we are sampling from. In the example, it would be the Gaussian.
"""

from bagpipes.fitting.prior import prior, dirichlet
from bagpipes.fitting import posterior
from bagpipes.fitting.calibration import calib_model
from bagpipes.fitting.noise import noise_model
from bagpipes.models.model_galaxy import model_galaxy

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

except ImportError:
    rank = 0

def loguniform_pdf(x, a, b):
    return 1/(x*np.log(b/a))

def make_IS_weight_func(self):
    func = []
    for i in range(len(self.params)):
        # detect if there is importance sampling for this parameter
        if 'IS_func' in self.hyper_params[i].keys():
            hyper_dict = self.hyper_params[i]
            #print(self_.params[i])
            # set the denominator function as the pdf of the sampled prior
            if self.pdfs[i] == 'uniform':
                g_x = lambda x: 1
            elif self.pdfs[i] == 'log_10':
                g_x = lambda x, bound_i=i: loguniform_pdf(x, self.limits[bound_i][0], self.limits[bound_i][1])
            elif self.pdfs[i] == 'Gaussian':
                g_x = lambda x, bound_i=i: norm.pdf(x, loc=self.hyper_params[bound_i]['mu'],
                                                    scale=self.hyper_params[bound_i]['sigma'])
            # the main function
            func.append(lambda x, bound_g_x=g_x, hyper_dict=hyper_dict: np.log10(hyper_dict['IS_func'](x, *hyper_dict['IS_params'])) -
                        np.log10(bound_g_x(x)))
        else:
            # append dummy functions
            func.append(lambda x: 0)

    #print(len(func))
    # sum of ln probabilities
    self.lnIS_weight = lambda x: np.sum([fi(x[i]) for i,fi in enumerate(func)])

def lnlike_IS(self, x, ndim=0, nparam=0):
    """ Returns the log-likelihood for a given parameter vector. """

    if self.time_calls:
        time0 = time.time()

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
        
    ######### add importance sampling weight
    lnlike += self.lnIS_weight(x)

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
            print("Mean likelihood call time:", np.mean(self.times))

    return lnlike

def setup_IS_fit(self):
    self.fitted_model.make_IS_weight_func()
    
def fit_IS(self, verbose=False, n_live=400, use_MPI=True):
    """ Fit the specified model to the input galaxy data.

    Parameters
    ----------

    verbose : bool - optional
        Set to True to get progress updates from the sampler.

    n_live : int - optional
        Number of live points: reducing speeds up the code but may
        lead to unreliable results.
    """

    if "lnz" in list(self.results):
        if rank == 0:
            print("Fitting not performed as results have already been"
                  + " loaded from " + self.fname[:-1] + ".h5. To start"
                  + " over delete this file or change run.\n")

        return

    if rank == 0 or not use_MPI:
        print("\nBagpipes: fitting object " + self.galaxy.ID + "\n")

        start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pmn.run(self.fitted_model.lnlike_IS,
                self.fitted_model.prior.transform,
                self.fitted_model.ndim, n_live_points=n_live,
                importance_nested_sampling=False, verbose=verbose,
                sampling_efficiency="model",
                outputfiles_basename=self.fname, use_MPI=use_MPI)

    if rank == 0 or not use_MPI:
        runtime = time.time() - start_time

        print("\nCompleted in " + str("%.1f" % runtime) + " seconds.\n")

        # Load MultiNest outputs and save basic quantities to file.
        samples2d = np.loadtxt(self.fname + "post_equal_weights.dat")
        lnz_line = open(self.fname + "stats.dat").readline().split()

        self.results["samples2d"] = samples2d[:, :-1]
        self.results["lnlike"] = samples2d[:, -1]
        self.results["lnz"] = float(lnz_line[-3])
        self.results["lnz_err"] = float(lnz_line[-1])
        self.results["median"] = np.median(samples2d, axis=0)
        self.results["conf_int"] = np.percentile(self.results["samples2d"],
                                                 (16, 84), axis=0)

        # Save re-formatted outputs as HDF5 and remove MultiNest output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dd.io.save(self.fname[:-1] + ".h5", self.results)

        os.system("rm " + self.fname + "*")

        self._print_results()

        # Create a posterior object to hold the results of the fit.
        self.posterior = posterior(self.galaxy, run=self.run)

pipes.fitting.fitted_model.lnlike_IS = lnlike_IS
pipes.fitting.fitted_model.make_IS_weight_func = make_IS_weight_func

pipes.fitting.fit.setup_IS_fit = setup_IS_fit
pipes.fitting.fit.fit_IS = fit_IS

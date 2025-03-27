import numpy as np
import re
import h5py
import os
import time
import warnings
import corner
import bagpipes as pipes
from bagpipes.fitting import fitted_model
from bagpipes.fitting import posterior

try:
    import pymultinest as pmn
    multinest_available = True
except (ImportError, RuntimeError, SystemExit):
    print("Bagpipes: PyMultiNest import failed, fitting with MultiNest will " +
          "be unavailable.")
    multinest_available = False

try:
    from nautilus import Sampler
    nautilus_available = True
except (ImportError, RuntimeError, SystemExit):
    print("Bagpipes: Nautilus import failed, fitting with nautilus will be " +
          "unavailable.")
    nautilus_available = False

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

except ImportError:
    rank = 0
    
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def sample_from_weighted_samps(ln_weight, size):
    # generate samples from non-equal posterior samples
    new_ln_weight = ln_weight+np.log(size/np.sum(np.exp(ln_weight)))
    repeating_samps_ind = np.where(new_ln_weight > 0)[0]

    indices = []
    for ind in repeating_samps_ind:
        repeat_no = int(np.exp(new_ln_weight[ind]))
        indices += [ind]*repeat_no
        new_ln_weight[ind] = np.log(np.exp(new_ln_weight[ind])-repeat_no)

    indices += list(np.where((np.random.rand(len(new_ln_weight)) <
                          np.exp(new_ln_weight)))[0])
    
    return indices

def init_posterior(self, galaxy, run=".", n_samples=500):

    self.galaxy = galaxy
    self.run = run
    self.n_samples = n_samples

    fname = "pipes/posterior/" + self.run + "/" + self.galaxy.ID + ".h5"

    # Check to see whether the object has been fitted.
    if not os.path.exists(fname):
        raise IOError("Fit results not found for " + self.galaxy.ID + ".")

    # Reconstruct the fitted model.
    file = h5py.File(fname, "r")

    fit_info_str = file.attrs["fit_instructions"]
    fit_info_str = fit_info_str.replace("array", "np.array")
    fit_info_str = fit_info_str.replace("float", "np.float")
    fit_info_str = re.sub("<[^>]*>", "None", fit_info_str)
    self.fit_instructions = eval(fit_info_str)

    self.fitted_model = fitted_model(self.galaxy, self.fit_instructions)

    # 2D array of samples for the fitted parameters only.
    self.samples2d = np.array(file["samples2d"])
    
    # grab the weights of samples if they exist
    if "ln_weight" in file.keys():
        self.ln_weight = np.array(file["ln_weight"])
    else:
        self.ln_weight = None

    # If fewer than n_samples exist in posterior, reduce n_samples
    if self.samples2d.shape[0] < self.n_samples:
        self.n_samples = self.samples2d.shape[0]
        
    # Randomly choose points to generate posterior quantities
    if self.ln_weight is not None:
        self.indices = sample_from_weighted_samps(self.ln_weight, self.n_samples)
        self.n_samples = len(self.indices)
    else:
        self.indices = np.random.choice(self.samples2d.shape[0],
                                        size=self.n_samples, replace=False)

    self.samples = {}  # Store all posterior samples

    dirichlet_comps = []  # Do any parameters follow Dirichlet dist

    # Add 1D posteriors for fitted params to the samples dictionary
    for i in range(self.fitted_model.ndim):
        param_name = self.fitted_model.params[i]

        if "dirichlet" in param_name:
            dirichlet_comps.append(param_name.split(":")[0])

        self.samples[param_name] = self.samples2d[self.indices, i]

    self.get_dirichlet_tx(dirichlet_comps)

    self.get_basic_quantities()
    
# adding control on fitting.fit.fit, allowing to turn on or off the deletion of raw
# multinest or nautilus files after fitting, to sample for equal or non-equal weight
# samples
def fit(self, verbose=False, n_live=400, use_MPI=True,
           sampler="multinest", n_eff=0, discard_exploration=False,
           n_networks=4, equal_weight_samples=True,
           pool=4, delete_raw_files=True, n_like_max=np.inf):
    """ Fit the specified model to the input galaxy data.

    Parameters
    ----------

    verbose : bool - optional
        Set to True to get progress updates from the sampler.

    n_live : int - optional
        Number of live points: reducing speeds up the code but may
        lead to unreliable results.

    sampler : string - optional
        The sampler to use. Available options are "multinest" and
        "nautilus".

    n_eff : float - optional
        Target minimum effective sample size. Only used by nautilus.

    discard_exploration : bool - optional
        Whether to discard the exploration phase to get more accurate
        results. Only used by nautilus.

    n_networks : int - optional
        Number of neural networks. Only used by nautilus.
        
    equal_weight_samples : bool - optional
        Set to True to save only equal weight posterior samples, otherwise
        saves all samples along with their weights. Only used by nautilus.

    pool : int - optional
        Pool size used for parallelization. Only used by nautilus.
        MultiNest is parallelized with MPI.
        
    delete_raw_files : bool - optional
        Set to True to remove the raw files generated by multinest or
        nautilus after fitting completes.
        
    n_like_max : int - optional
        Maximum total (accross multiple runs) number of likelihood evaluations.
        Regardless of progress, the sampler will not start new likelihood
        computations if this value is reached. Note that this value includes
        likelihood calls from previous runs, if applicable. Default is infinity.
        Only used by nautilus.
    """

    if "lnz" in list(self.results):
        if rank == 0:
            print("Fitting not performed as results have already been"
                  + " loaded from " + self.fname[:-1] + ".h5. To start"
                  + " over delete this file or change run.\n")

        return

    sampler = sampler.lower()

    if (sampler == "multinest" and not multinest_available and
            nautilus_available):
        sampler = "nautilus"
        print("MultiNest not available. Switching to nautilus.")
    elif (sampler == "nautilus" and not nautilus_available and
            multinest_available):
        sampler = "multinest"
        print("Nautilus not available. Switching to MultiNest.")
    elif sampler not in ["multinest", "nautilus"]:
        raise ValueError("Sampler {} not supported.".format(sampler))
    elif not (multinest_available or nautilus_available):
        raise RuntimeError(
            "Neither MultiNest nor nautilus could be loaded.")

    if rank == 0 or not use_MPI:
        print("\nBagpipes: fitting object " + self.galaxy.ID + "\n")

        start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler_success = False

        if sampler == "multinest":
            pmn.run(self.fitted_model.lnlike,
                    self.fitted_model.prior.transform,
                    self.fitted_model.ndim, n_live_points=n_live,
                    importance_nested_sampling=False, verbose=verbose,
                    sampling_efficiency="model",
                    outputfiles_basename=self.fname, use_MPI=use_MPI)
            sampler_success = True

        elif sampler == "nautilus":
            n_sampler = Sampler(self.fitted_model.prior.transform,
                                self.fitted_model.lnlike, n_live=n_live,
                                n_networks=n_networks, pool=pool,
                                n_dim=self.fitted_model.ndim,
                                filepath=self.fname + "nautilus.h5")

            sampler_success = n_sampler.run(
                verbose=verbose, n_eff=n_eff,
                discard_exploration=discard_exploration,
                n_like_max=n_like_max
            )

    if rank == 0 or not use_MPI:
        runtime = time.time() - start_time

        if sampler_success == False:
            print("\nSampler timeout, runtime " + str("%.1f" % runtime) + " seconds.\n")
            return
        
        print("\nCompleted in " + str("%.1f" % runtime) + " seconds.\n")

        # Load MultiNest outputs and save basic quantities to file.
        if sampler == "multinest":
            samples2d = np.loadtxt(self.fname + "post_equal_weights.dat")
            lnz_line = open(self.fname + "stats.dat").readline().split()
            self.results["samples2d"] = samples2d[:, :-1]
            self.results["lnlike"] = samples2d[:, -1]
            self.results["lnz"] = float(lnz_line[-3])
            self.results["lnz_err"] = float(lnz_line[-1])

        elif sampler == "nautilus":
            samples2d, log_w, log_l = n_sampler.posterior(
                equal_weight=equal_weight_samples)
            self.results["samples2d"] = samples2d
            self.results["lnlike"] = log_l
            self.results["lnz"] = n_sampler.log_z
            self.results["lnz_err"] = 1.0 / np.sqrt(n_sampler.n_eff)
            if equal_weight_samples == False:
                self.results["ln_weight"] = log_w

        if sampler == "nautilus" and equal_weight_samples == False:
            Ndim = np.shape(self.results["samples2d"])[1]
            self.results["median"] = np.zeros(Ndim)
            self.results["conf_int"] = np.zeros((2, Ndim))
            
            # use weighted percentile function to get percentiles
            for i in range(Ndim):
                self.results["median"][i] = weighted_quantile(samples2d[:,i], 0.5, sample_weight=np.exp(log_w))
                self.results["conf_int"][:,i] = weighted_quantile(samples2d[:,i], (0.16,0.84), sample_weight=np.exp(log_w))
        else:
            self.results["median"] = np.median(samples2d, axis=0)
            self.results["conf_int"] = np.percentile(self.results["samples2d"],
                                                    (16, 84), axis=0)

        file = h5py.File(self.fname[:-1] + ".h5", "w")

        # This is necessary for converting large arrays to strings
        np.set_printoptions(threshold=10**7)
        file.attrs["fit_instructions"] = str(self.fit_instructions)
        np.set_printoptions(threshold=10**4)

        for k in self.results.keys():
            file.create_dataset(k, data=self.results[k])

        self.results["fit_instructions"] = self.fit_instructions

        file.close()

        if delete_raw_files:
            os.system("rm " + self.fname + "*")

        self._print_results()

        # Create a posterior object to hold the results of the fit.
        self.posterior = posterior(self.galaxy, run=self.run,
                                   n_samples=self.n_posterior)
    
    
pipes.fitting.posterior.__init__ = init_posterior
pipes.fitting.fit.fit = fit

from bagpipes.plotting.general import *

# adjust corner plot to take in the weight of the 2d samples too
def plot_corner(fit, show=False, save=True, bins=25, type="fit_params"):
    """ Make a corner plot of the fitted parameters. """

    update_rcParams()

    names = fit.fitted_model.params
    
    samples = np.copy(fit.posterior.samples2d)
    if 'ln_weight' not in fit.results.keys():
        weights = None
    else:
        weights = np.exp(fit.results['ln_weight']).copy()
    
    # down sample if Nsamp > 10000
    if len(fit.posterior.samples2d) > 10000:
        samp_ind = np.random.choice(len(fit.posterior.samples2d), size=10000, replace=False)
        samples = samples[samp_ind]
        if weights is not None:
            weights = weights[samp_ind]

    # Set up axis labels
    if tex_on:
        labels = fix_param_names(names)

    else:
        labels = fit.fitted_model.params
        
    print(len(samples))

    # Log any parameters with log_10 priors to make them easier to see
    for i in range(fit.fitted_model.ndim):
        if fit.fitted_model.pdfs[i] == "log_10":
            samples[:, i] = np.log10(samples[:, i])

            if tex_on:
                labels[i] = "$\\mathrm{log_{10}}(" + labels[i][1:-1] + ")$"

            else:
                labels[i] = "log_10(" + labels[i] + ")"

    # Replace any r parameters for Dirichlet distributions with t_x vals
    j = 0
    for i in range(fit.fitted_model.ndim):
        if "dirichlet" in fit.fitted_model.params[i]:
            comp = fit.fitted_model.params[i].split(":")[0]
            n_x = fit.fitted_model.model_components[comp]["bins"]
            t_percentile = int(np.round(100*(j+1)/n_x))

            samples[:, i] = fit.posterior.samples[comp + ":tx"][:, j]
            j += 1

            if tex_on:
                labels[i] = "$t_{" + str(t_percentile) + "}\ /\ \mathrm{Gyr}$"

            else:
                labels[i] = "t" + str(t_percentile) + " / Gyr"

    # Make the corner plot
    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 13},
                        smooth=1., smooth1d=1., bins=bins, weights=weights)

    # Save the corner plot to file
    if save:
        plotpath = ("pipes/plots/" + fit.run + "/" + fit.galaxy.ID
                    + "_corner.pdf")

        plt.savefig(plotpath, bbox_inches="tight")
        plt.close(fig)

    # Alternatively show the corner plot
    if show:
        plt.show()
        plt.close(fig)

    return fig

pipes.plotting.plot_corner = plot_corner


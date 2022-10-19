import bagpipes as pipes
import numpy as np
from bagpipes.models.model_galaxy import model_galaxy

def get_advanced_quantities(self):
    """Calculates advanced derived posterior quantities, these are
    slower because they require the full model spectra. """

    if "spectrum_full" in list(self.samples):
        return

    self.fitted_model._update_model_components(self.samples2d[0, :])
    self.model_galaxy = model_galaxy(self.fitted_model.model_components,
                                     filt_list=self.galaxy.filt_list,
                                     spec_wavs=self.galaxy.spec_wavs,
                                     index_list=self.galaxy.index_list)

    all_names = ["photometry", "spectrum", "spectrum_full", "uvj",
                 "indices"]

    all_model_keys = dir(self.model_galaxy)
    quantity_names = [q for q in all_names if q in all_model_keys]

    for q in quantity_names:
        try:
            size = getattr(self.model_galaxy, q).shape[0]
        except IndexError:
            if getattr(self.model_galaxy, q).ndim == 0:
                size = 1
        self.samples[q] = np.zeros((self.n_samples, size))

    if self.galaxy.photometry_exists:
        self.samples["chisq_phot"] = np.zeros(self.n_samples)

    if "dust" in list(self.fitted_model.model_components):
        size = self.model_galaxy.spectrum_full.shape[0]
        self.samples["dust_curve"] = np.zeros((self.n_samples, size))

    if "calib" in list(self.fitted_model.model_components):
        size = self.model_galaxy.spectrum.shape[0]
        self.samples["calib"] = np.zeros((self.n_samples, size))

    if "noise" in list(self.fitted_model.model_components):
        type = self.fitted_model.model_components["noise"]["type"]
        if type.startswith("GP"):
            size = self.model_galaxy.spectrum.shape[0]
            self.samples["noise"] = np.zeros((self.n_samples, size))

    for i in range(self.n_samples):
        param = self.samples2d[self.indices[i], :]
        self.fitted_model._update_model_components(param)
        self.fitted_model.lnlike(param)

        if self.galaxy.photometry_exists:
            self.samples["chisq_phot"][i] = self.fitted_model.chisq_phot

        if "dust" in list(self.fitted_model.model_components):
            dust_curve = self.fitted_model.model_galaxy.dust_atten.A_cont
            self.samples["dust_curve"][i] = dust_curve

        if "calib" in list(self.fitted_model.model_components):
            self.samples["calib"][i] = self.fitted_model.calib.model

        if "noise" in list(self.fitted_model.model_components):
            type = self.fitted_model.model_components["noise"]["type"]
            if type.startswith("GP"):
                self.samples["noise"][i] = self.fitted_model.noise.mean()

        for q in quantity_names:
            if q == "spectrum":
                spectrum = getattr(self.fitted_model.model_galaxy, q)[:, 1]
                self.samples[q][i] = spectrum
                continue

            self.samples[q][i] = getattr(self.fitted_model.model_galaxy, q)
            
pipes.fitting.posterior.get_advanced_quantities = get_advanced_quantities

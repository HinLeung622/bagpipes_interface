import bagpipes as pipes
import numpy as np
import spectres

from bagpipes import config

def _calculate_spectrum(self, model_comp):
    """ This method generates predictions for observed spectroscopy.
    It optionally applies a Gaussian velocity dispersion then
    resamples onto the specified set of observed wavelengths. """

    zplusone = model_comp["redshift"] + 1.

    if "veldisp" in list(model_comp):
        vres = 3*10**5/config.R_spec/2.
        sigma_pix = model_comp["veldisp"]/vres
        k_size = 4*int(sigma_pix+1)
        x_kernel_pix = np.arange(-k_size, k_size+1)

        kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
        kernel /= np.trapz(kernel)  # Explicitly normalise kernel

        spectrum = np.convolve(self.spectrum_full, kernel, mode="valid")
        redshifted_wavs = zplusone*self.wavelengths[k_size:-k_size]

    else:
        spectrum = self.spectrum_full
        redshifted_wavs = zplusone*self.wavelengths

    if "R_curve" in list(model_comp):
        oversample = 4  # Number of samples per FWHM at resolution R
        new_wavs = self._get_R_curve_wav_sampling(oversample=oversample)

        # spectrum = np.interp(new_wavs, redshifted_wavs, spectrum)
        spectrum = spectres.spectres_numba(new_wavs, redshifted_wavs,
                                            spectrum, fill=0)
        redshifted_wavs = new_wavs

        sigma_pix = oversample/2.35  # sigma width of kernel in pixels
        k_size = 4*int(sigma_pix+1)
        x_kernel_pix = np.arange(-k_size, k_size+1)

        kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
        kernel /= np.trapz(kernel)  # Explicitly normalise kernel

        # Disperse non-uniformly sampled spectrum
        spectrum = np.convolve(spectrum, kernel, mode="valid")
        redshifted_wavs = redshifted_wavs[k_size:-k_size]

    # Converted to using spectres in response to issue with interp,
    # see https://github.com/ACCarnall/bagpipes/issues/15
    # fluxes = np.interp(self.spec_wavs, redshifted_wavs,
    #                    spectrum, left=0, right=0)

    ######## Hin note: have swapped from using default spectres to the numba option
    # currently there is an issue with numba where numba looks for the file
    # spectral_resampling_numba.py in the working direction, not the package's directory
    # current fix is to also hold a copy in the working directry
    fluxes = spectres.spectres_numba(self.spec_wavs, redshifted_wavs,
                                        spectrum, fill=0)

    if self.spec_units == "mujy":
        fluxes /= ((10**-29*2.9979*10**18/self.spec_wavs**2))

    self.spectrum = np.c_[self.spec_wavs, fluxes]

pipes.models.model_galaxy._calculate_spectrum = _calculate_spectrum

# from H_alpha.py in pySEDM by Yirui
# with routines adapted from GALAXEV, clyman.f, from http://www.bruzual.org
# with equations based on section 2.3 of Kennicutt 1998
# HL added updated conversion based on Murphy et al. 2011, which Kennicutt cites in the updated review Kennicutt & Evans 2012

import numpy as np

def CountLyman(wave, sed):
    """
    Compute number of Lyman continuum photons in sed sed(wave).
    Assumes wavelength is in Angstroms and sed in ergs/sec/Angstroms (physical flux)
    Based on GALAXEV, clyman.f   http://www.bruzual.org
    NOTE: use the intrinsic spectra, not the dust redden one!
    """
    wly = 912.0 # Angstroms
    c=2.997925e10 # cm/sec
    h=6.6262e-27 # Planck constant, erg*sec
    const=1.0E-8/h/c
    # for a single photon, E=hc/wavelength, so E*wave/h/c = 1
    # for n photons with same wavelength and total energy of E_tot
    # E_tot*wave/h/c = n
    # The constant above convert E*wave to count of photons
    short_idx = np.where(wave <= wly)[0] # the wavelength shorter than Lyman limit
#     print short_idx
    wave_max = wave[short_idx.max()]  # the longest wavelength that is shorter than Lyman limit
#     print wave_max
    if wave_max == wly :
        wave_short = wave[short_idx]
        sed_short = sed[short_idx]
    else:
        # there is a partial bin, wave_max < wly < wave_next
        # need count the engery between wave_max and wly
        # use linear interpolation
        next_idx = short_idx.max() +1    # the wavelength has been sorted to be ascending
        f_wly = np.interp( wly, wave[next_idx-1 : next_idx+1], sed[next_idx-1 : next_idx+1])
        wave_short = np.append(wave[short_idx], wly)
        sed_short = np.append(sed[short_idx], f_wly)

    y=wave_short*sed_short
    clyman = const*np.trapz(y=y, x=wave_short ) # Integrate over all wavelength

    return clyman


def L_H_alpha(wave, sed, dusted=False, mu_d=0.3, tauv=1.0):
    """
    The luminosity of H_alpha, caculated with Q.
    Assumes wavelength is in Angstroms and sed in ergs/sec/Angstroms
    The output is in ergs/sec/Angstroms as well.
    NOTE: use the intrinsic spectra, not the dust redden one!
    The function gives the intrinsic L(H_alpha) by defualt,
    if dusted is set to be True, a default 2-component dust model will be applied.
    """
    Q = CountLyman(wave, sed)
    #  L_Ha = Q / 7.9e-42 * 1.08e-53 # in ergs/sec/Angstroms, according to Kennicutt 1998
    # the old method is unsafe, Q is usually quite large
    # Q/7.9e-42 may result in overflow
    L_Ha = Q * 1.08e-53 / 7.9e-42 # in ergs/sec/Angstroms, according to Kennicutt 1998
    # L_Ha = Q * 7.29e−54 / 5.37e−42 # in ergs/sec, according to Murphy et al. 2011, which Kennicutt cites in the updated review Kennicutt & Evans 2012
    if not dusted:
        return L_Ha
    else:
        tau_young_ha = mu_d*tauv*( (5500./w_ha)**0.7) + (1-mu_d)*tauv*( (5500./w_ha)**1.3)
        L_Ha_tau = L_Ha * np.exp(-tau_young_ha)
        return L_Ha_tau

def L_H_alpha2011(wave, sed, dusted=False, mu_d=0.3, tauv=1.0):
    """
    The luminosity of H_alpha, caculated with Q.
    Assumes wavelength is in Angstroms and sed in ergs/sec/Angstroms
    The output is in ergs/sec/Angstroms as well.
    NOTE: use the intrinsic spectra, not the dust redden one!
    The function gives the intrinsic L(H_alpha) by defualt,
    if dusted is set to be True, a default 2-component dust model will be applied.
    """
    Q = CountLyman(wave, sed)
    L_Ha = Q * 7.29e-54 / 5.37e-42 # in ergs/sec, according to Murphy et al. 2011, which Kennicutt cites in the updated review Kennicutt & Evans 2012
    if not dusted:
        return L_Ha
    else:
        tau_young_ha = mu_d*tauv*( (5500./w_ha)**0.7) + (1-mu_d)*tauv*( (5500./w_ha)**1.3)
        L_Ha_tau = L_Ha * np.exp(-tau_young_ha)
        return L_Ha_tau

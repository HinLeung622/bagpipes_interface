import bagpipes as pipes
import numpy as np

# adding psb_delayed to star_formation_history.py
"""
older component = e^-(t/tau)
younger component = double powerlaw
structure:
psb_delayed["massformed"] =
psb_delayed["metallicity"] =
psb_delayed["age"] = Time since beginning of star formation of older component
psb_delayed["tau"] = Timescale of decrease: Gyr
psb_delayed["burstage"] = age at z - tburst
psb_delayed["alpha"] = alpha of the burst component
psb_delayed["beta"] = beta of the burst component
psb_delayed["fburst"] = fburst
"""

def psb_delayed(self, sfr, param):
    """
    A 2-component SFH for post-starburst galaxies. A delayed-exponential
    compoent represents the existing stellar population before the
    starburst, while a double power law makes up the burst.
    The weight of mass formed between the two is controlled by a
    fburst factor: thefraction of mass formed in the burst.
    Modified based on psb_wild2020 form
    """
    age = param["age"]*10**9
    tau = param["tau"]*10**9
    burstage = param["burstage"]*10**9
    alpha = param["alpha"]
    beta = param["beta"]
    fburst = param["fburst"]

    ind = (np.where((self.ages < age) & (self.ages > burstage)))[0]
    texp = age - self.ages[ind]
    sfr_exp = texp*np.exp(-texp/tau)
    sfr_exp_tot = np.sum(sfr_exp*self.age_widths[ind])

    mask = self.ages < self.age_of_universe
    tburst = self.age_of_universe - self.ages[mask]
    tau_plaw = self.age_of_universe - burstage
    sfr_burst = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
    sfr_burst_tot = np.sum(sfr_burst*self.age_widths[mask])

    sfr[ind] = (1-fburst) * texp * np.exp(-texp/tau) / sfr_exp_tot

    dpl_form = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
    sfr[mask] += fburst * dpl_form / sfr_burst_tot
    
pipes.models.star_formation_history.psb_delayed = psb_delayed


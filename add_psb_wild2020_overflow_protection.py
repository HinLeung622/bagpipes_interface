import bagpipes as pipes
import numpy as np

# adding overflow protection to psb_wild2020 in star_formation_history.py

def psb_wild2020(self, sfr, param):
    """
    A 2-component SFH for post-starburst galaxies. An exponential
    compoent represents the existing stellar population before the
    starburst, while a double power law makes up the burst.
    The weight of mass formed between the two is controlled by a
    fburst factor: thefraction of mass formed in the burst.
    For more detail, see Wild et al. 2020
    (https://ui.adsabs.harvard.edu/abs/2020MNRAS.494..529W/abstract)
    """
    age = param["age"]*10**9
    tau = param["tau"]*10**9
    burstage = param["burstage"]*10**9
    alpha = param["alpha"]
    beta = param["beta"]
    fburst = param["fburst"]

    ind = (np.where((self.ages < age) & (self.ages > burstage)))[0]
    texp = age - self.ages[ind]
    sfr_exp = np.exp(-texp/tau)
    sfr_exp_tot = np.sum(sfr_exp*self.age_widths[ind])

    mask = self.ages < self.age_of_universe
    tburst = self.age_of_universe - self.ages[mask]
    tau_plaw = self.age_of_universe - burstage
    # using masks to avoid numpy64 float overflow
    ratio = tburst/tau_plaw
    mask_overflow = ((np.log10(ratio) * alpha < 250) & (np.log10(ratio) * -beta < 250))
    sfr_burst = np.zeros_like(tburst)
    sfr_burst[mask_overflow] = ((tburst[mask_overflow]/tau_plaw)**alpha + (tburst[mask_overflow]/tau_plaw)**-beta)**-1
    sfr_burst_tot = np.sum(sfr_burst*self.age_widths[mask])

    sfr[ind] = (1-fburst) * np.exp(-texp/tau) / sfr_exp_tot

    dpl_form = sfr_burst #((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
    sfr[mask] += fburst * dpl_form / sfr_burst_tot
    
pipes.models.star_formation_history.psb_wild2020 = psb_wild2020

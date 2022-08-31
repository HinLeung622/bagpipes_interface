import bagpipes as pipes
import numpy as np

# adding psb_twin to star_formation_history.py
"""
structure:
psb_twin["massformed"] = 
psb_twin["metallicity"] = 
psb_twin["age"] = peak of the older component
psb_twin["alpha1"] = alpha of the older component
psb_twin["beta1"] = beta of the older component
psb_twin["burstage"] = age at z - tburst
psb_twin["alpha2"] = alpha of the burst component
psb_twin["beta2"] = beta of the burst component
psb_twin["fburst"] = fburst
"""

def psb_twin(self, sfr, param):
    """
    Modified form of psb_wild2020. A 2-component SFH for post-starburst galaxies. 
    Replaced the exponential part with a double powerlaw, which creates 2x double
    powerlaws in this functional form, hence the name twin. alpha1 and beta1 refers
    to the older component, while 2 refers to the newer. 
    """
    age = param["age"]*10**9
    alpha1 = param["alpha1"]
    beta1 = param["beta1"]
    burstage = param["burstage"]*10**9
    alpha2 = param["alpha2"]
    beta2 = param["beta2"]
    fburst = param["fburst"]

    ind = (np.where((self.ages < self.age_of_universe) & (self.ages > burstage)))[0]
    told = self.age_of_universe - self.ages[ind]
    tau_old = self.age_of_universe - age
    sfr_old = ((told/tau_old)**alpha1 + (told/tau_old)**-beta1)**-1
    sfr_old_tot = np.sum(sfr_old*self.age_widths[ind])

    mask = self.ages < self.age_of_universe
    tburst = self.age_of_universe - self.ages[mask]
    tau_plaw = self.age_of_universe - burstage
    sfr_burst = ((tburst/tau_plaw)**alpha2 + (tburst/tau_plaw)**-beta2)**-1
    sfr_burst_tot = np.sum(sfr_burst*self.age_widths[mask])

    old_dpl_form = ((told/tau_old)**alpha1 + (told/tau_old)**-beta1)**-1
    sfr[ind] = (1-fburst) * old_dpl_form / sfr_old_tot

    burst_dpl_form = ((tburst/tau_plaw)**alpha2 + (tburst/tau_plaw)**-beta2)**-1
    sfr[mask] += fburst * burst_dpl_form / sfr_burst_tot
    
pipes.models.star_formation_history.psb_twin = psb_twin

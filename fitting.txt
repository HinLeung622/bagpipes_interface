import bagpipes as pipes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from scipy.optimize import curve_fit

class fitting:
    def __init__(self, skylines_path, data, z, binby, sky_masking=True, 
                 full=True, model_galaxy_SNR=None):
        self.skylines_path = skylines_path
        if type(data) is str:
            self.data_path = data
            self.loadfromfile = True
        else:
            self.model_galaxy = data
            self.loadfromfile = False
        self.z = z
        self.binby = binby
        self.sky_masking = sky_masking
        self.full = full
        self.model_galaxy_SNR = model_galaxy_SNR
        self.mask_em_vals = [3727.092,3729.875,4102.892, 
                             4341.692,4862.683,4960.295,5008.24,
                             6564.61,6549.8490,6585.2784, 6717., 
                             6731., 5893., 6302.046, 6918.6, 3870.]
        self.load_sky()
        
    def bin(self, spectrum, binn):
        """ Bins up two or three column spectral data by a specified factor. """
    
        binn = int(binn)
        nbins = len(spectrum)/binn
        binspec = np.zeros((int(nbins), spectrum.shape[1]))
    
        for i in range(binspec.shape[0]):
            spec_slice = spectrum[i*binn:(i+1)*binn, :]
            binspec[i, 0] = np.mean(spec_slice[:, 0])
            binspec[i, 1] = np.mean(spec_slice[:, 1])
    
            if spectrum.shape[1] == 3:
                binspec[i,2] = (1./float(binn)
                                *np.sqrt(np.sum(spec_slice[:, 2]**2)))
    
        return binspec
        
    def load_sky(self):
        np_arr = np.loadtxt(self.skylines_path+'/skylines.txt')
        lines_air_df = pd.DataFrame(np_arr, columns=['wavelength', 'width', 'flux'])
        self.lines_vac_sky = pyasl.airtovac2(lines_air_df[lines_air_df['flux']>=5]['wavelength'])

    def mask_sky(self, wave):
        """ Masks strong night sky emission lines that are often not removed 
        properly in the data processing. From masksky.pro """
    
        masksize=10
        lines_vac = self.lines_vac_sky #np.array([5578.5486,4279.2039,6301.7423,6365.7595])
        lines = pyasl.vactoair2(lines_vac)
        
        mask = []
        for i in range(lines.shape[0]):
            ind = np.where((wave>lines[i]-masksize) & (wave<lines[i]+masksize))
            mask.extend(ind[0])
    
        return mask

    def mask_em(self, wave):
        """ from maskem.pro OII, Hgama, Hbeta, OIII, OIII: Vacuum """
    
        lines_vac = np.array(self.mask_em_vals)
        lines = pyasl.vactoair2(lines_vac)
        
        mask = []
        for i in range(lines.shape[0]):
            if lines[i] > 6500 and lines[i] < 6600 or lines[i]==5893:
                masksize = 10
            else:
                masksize = 5
    
            ind = np.where((wave>lines[i]-masksize) & (wave<lines[i]+masksize))
            mask.extend(ind[0])
    
        # MgII  2796.352 ,2803.531 
        #ind = np.where((wave>2766.4) & (wave<2833.5))
    
        # remove everything bluewards of 3000A
        #ind = np.where(wave<3000)
        #mask.extend(ind[0])    
    
        return mask
    
    def load_manga_spec(self, ID):
    
        # load spectral data
        if self.loadfromfile:
            # load from save, csv file
            spectrum = np.loadtxt(fname=self.data_path+'/Spectrum_'+
                                  ID+".csv", delimiter=',', skiprows=1)
            
            spectrum[:,1] *= 10**-16
            spectrum[:,2] *= 10**-16
        
        else:
            # load from existing galaxy object
            spectrum = self.model_galaxy.spectrum.copy()
            spectrum_noise = spectrum[:,1]/self.model_galaxy_SNR
            spectrum = np.hstack([spectrum, np.expand_dims(spectrum_noise, axis=1)])
    
        # blow up the errors associated with any bad points in the spectrum and photometry
        for i in range(len(spectrum)):
            if spectrum[i,1] == 0 or spectrum[i,2] <= 0:
                spectrum[i,1] = 0.
                spectrum[i,2] = 9.9*10**99.
        
        # nebular emission lines and interstellar absorption lines
        mask = self.mask_em(spectrum[:,0]/(1+self.z))
        spectrum[mask, 2] = 9.9*10**99.
        
        # skylines
        if self.sky_masking:
            linemask = self.mask_sky(spectrum[:,0])
            spectrum[linemask, 2] = 9.9*10**99.
        
        for j in range(len(spectrum)):
            if (spectrum[j, 1] == 0) or (spectrum[j, 2] <= 0):
                spectrum[j, 2] = 9.9*10**99.
        
        # O2 telluric
        #mask = ((spectrum[:,0] > 7580.) & (spectrum[:,0] < 7650.))
        #spectrum[mask, 2] = 9.9*10**99.
        
        if self.full == False:
            endmask = (spectrum[:,0]/(1+self.z) < 7500) # just miles range
        else:
            endmask = (spectrum[:,0]>0)
    
        if self.binby > 1:
            return self.bin(spectrum[endmask], self.binby)
        else:
            return spectrum[endmask]
        
        
class get_ceh_array:
    """
    Evaluates the metallicity values at a list of ages (in lb time) given the
    metallicity model choice and model parameters.
    """
    def delta(ages, sfh_dict):
        return np.ones(len(ages))*sfh_dict['metallicity']
    
    def two_step(ages, sfh_dict):
        pre_step_ind = np.where(ages > sfh_dict['metallicity_step_age'])
        post_step_ind = np.isin(np.arange(len(ages)), pre_step_ind, invert=True)
        ceh = np.zeros(len(ages))
        ceh[pre_step_ind] = sfh_dict['metallicity_old']
        ceh[post_step_ind] = sfh_dict['metallicity_new']
        return ceh
    
    def psb_two_step(ages, sfh_dict):
        pre_step_ind = np.where(ages > sfh_dict['burstage'])
        post_step_ind = np.isin(np.arange(len(ages)), pre_step_ind, invert=True)
        ceh = np.zeros(len(ages))
        ceh[pre_step_ind] = sfh_dict['metallicity_old']
        ceh[post_step_ind] = sfh_dict['metallicity_burst']
        return ceh

# plotting functions
# extracted from bagpipes.models.star_formation_history.py, with a bit of tweaking
def psb_wild2020(age_list, age, tau, burstage, alpha, beta, fburst, Mstar):
    age_lhs = pipes.utils.make_bins(np.log10(age_list)+9, make_rhs=True)[0]
    age_list = age_list*10**9
    age_lhs = 10**age_lhs
    age_lhs[0] = 0.
    age_lhs[-1] = 10**9*pipes.utils.age_at_z[pipes.utils.z_array == 0.]
    age_widths = age_lhs[1:] - age_lhs[:-1]
    sfr = np.zeros(len(age_list))
    
    age_of_universe = 10**9*np.interp(0, pipes.utils.z_array,
                                               pipes.utils.age_at_z)
    
    age = age*10**9
    tau = tau*10**9
    burstage = burstage*10**9

    ind = (np.where((age_list < age) & (age_list > burstage)))[0]
    texp = age - age_list[ind]
    sfr_exp = np.exp(-texp/tau)
    sfr_exp_tot = np.sum(sfr_exp*age_widths[ind])

    mask = age_list < age_of_universe
    tburst = age_of_universe - age_list[mask]
    tau_plaw = age_of_universe - burstage
    sfr_burst = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
    sfr_burst_tot = np.sum(sfr_burst*age_widths[mask])

    sfr[ind] = (1-fburst) * np.exp(-texp/tau) / sfr_exp_tot

    dpl_form = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
    sfr[mask] += fburst * dpl_form / sfr_burst_tot
    
    return sfr*10**Mstar

# a copy of the function, with a bit of tweaking
def psb_twin_(age_list, age, alpha1, beta1, burstage, alpha2, beta2, fburst, Mstar):
    age_lhs = pipes.utils.make_bins(np.log10(age_list)+9, make_rhs=True)[0]
    age_list = age_list*10**9
    age_lhs = 10**age_lhs
    age_lhs[0] = 0.
    age_lhs[-1] = 10**9*pipes.utils.age_at_z[pipes.utils.z_array == 0.]
    age_widths = age_lhs[1:] - age_lhs[:-1]
    sfr = np.zeros(len(age_list))
    
    age_of_universe = 10**9*np.interp(0, pipes.utils.z_array,
                                               pipes.utils.age_at_z)
    
    age = age*10**9
    burstage = burstage*10**9

    ind = (np.where((age_list < age_of_universe) & (age_list > burstage)))[0]
    told = age_of_universe - age_list[ind]
    tau_old = age_of_universe - age
    sfr_old = ((told/tau_old)**alpha1 + (told/tau_old)**-beta1)**-1
    sfr_old_tot = np.sum(sfr_old*age_widths[ind])

    mask = age_list < age_of_universe
    tburst = age_of_universe - age_list[mask]
    tau_plaw = age_of_universe - burstage
    sfr_burst = ((tburst/tau_plaw)**alpha2 + (tburst/tau_plaw)**-beta2)**-1
    sfr_burst_tot = np.sum(sfr_burst*age_widths[mask])

    old_dpl_form = ((told/tau_old)**alpha1 + (told/tau_old)**-beta1)**-1
    sfr[ind] = (1-fburst) * old_dpl_form / sfr_old_tot

    burst_dpl_form = ((tburst/tau_plaw)**alpha2 + (tburst/tau_plaw)**-beta2)**-1
    sfr[mask] += fburst * burst_dpl_form / sfr_burst_tot
    
    return sfr*10**Mstar

def load_model_sfh(filepath):
    # load in true SFH
    #age_at_z = pipes.utils.cosmo.age(0).value
    sim_data = np.loadtxt(filepath)
    model_sfh = sim_data[:,2]
    model_ages = sim_data[:,0]
    mask = model_ages > 0
    model_ages = model_ages[mask].copy()
    model_sfh = model_sfh[mask].copy()
    return model_ages, model_sfh

def get_advanced_quantities(fit):
    # a workaround of having to recalculate the advanced
    # quantities upon every re-loading of results
    import os
    import deepdish as dd
    if "spectrum_full" in list(fit.posterior.samples):
        return
    elif os.path.exists(fit.fname + "full_samp.h5"):
        # load and replace samples from file
        fit.posterior.samples = dd.io.load(fit.fname + "full_samp.h5")
        fit.posterior.fitted_model._update_model_components(fit.posterior.samples2d[0, :])
        fit.posterior.model_galaxy = pipes.models.model_galaxy(
            fit.posterior.fitted_model.model_components,
            filt_list=fit.posterior.galaxy.filt_list,
            spec_wavs=fit.posterior.galaxy.spec_wavs,
            index_list=fit.posterior.galaxy.index_list
        )
    else:
        fit.posterior.get_advanced_quantities()
        # save it, path is pipes/[runID]/[galID]_full_samp.h5
        dd.io.save(fit.fname + "full_samp.h5", fit.posterior.samples)
        print(f'Advanced quantities saved in {fit.fname + "full_samp.h5"}.')

def plot_spec(fit, ID, runID, save=True):

    # Make the figure
    matplotlib.rcParams.update({'font.size': 16})
    params = {'legend.fontsize': 16,
              'legend.handlelength': 1}
    matplotlib.rcParams.update(params)
    matplotlib.rcParams['text.usetex'] = True
    get_advanced_quantities(fit)

    naxes=1
    fig = plt.figure(figsize=(12, 5.*naxes))

    gs1 = matplotlib.gridspec.GridSpec(4, 1, hspace=0., wspace=0.)
    ax1 = plt.subplot(gs1[:3])
    ax3 = plt.subplot(gs1[3])

    mask = fit.galaxy.spectrum[:, 2] < 1.
    fit.galaxy.spectrum[mask, 2] = 0.

    y_scale = pipes.plotting.add_spectrum(fit.galaxy.spectrum, ax1)
    pipes.plotting.add_spectrum_posterior(fit, ax1, y_scale=y_scale)

    post_median = np.median(fit.posterior.samples["spectrum"], axis=0)

    #ax1.plot(fit.galaxy.spectrum[:,0],
    #         post_median*10**-y_scale,
    #         color="black", lw=1.0,zorder=11)

    ax3.axhline(0, color="black", ls="--", lw=1)
    ax3.plot(fit.galaxy.spectrum[:,0],(post_median - fit.galaxy.spectrum[:,1])*10**-y_scale, color="sandybrown")
    #ax1.set_xlim([3000,4200])
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xlabel("$\\lambda / \\mathrm{\\AA}$")
    ax3.set_ylabel('residual')
    if save:
        fig.savefig('pipes/plots/'+runID+'/'+ID+'_fit.pdf')
    plt.show()
    
def integrate_sfh(ages, sfh, Mstar=None):
    """ 
    takes a sfh and integrates it to return a cumulative SFH (normalized to run from 0 to 1) fraction of 
    mass formed
    """
    if Mstar is None:
        Mstar = np.trapz(y=sfh,x=ages)
    c_sfh = np.zeros(len(sfh))
    for i,sfhi in enumerate(sfh):
        c_sfh[i] = np.trapz(sfh[:i+1],x=ages[:i+1]/Mstar)
    return c_sfh

def fit_f_burst(ages, sfh, age_at_z, SFH_comp):
    # using scipy curve fit to get a fit to the true SFH
    if SFH_comp == "psb2" or SFH_comp == "psb_wild2020":
        popt,pcov = curve_fit(psb_wild2020, ages, sfh, 
                              bounds=([10,1,0,10,10,0,10],[13,10,2,1000,1000,1,12]))
        [age, tau, burstage, alpha, beta, fburst, Mstar] = popt
    elif SFH_comp == "psb_twin":
        popt,pcov = curve_fit(psb_twin_, ages, sfh, 
                              bounds=([10,0.01,100,0,10,10,0,10],[13,1000,10000,2,1000,1000,1,12]))
        [age, alpha1, beta1, burstage, alpha2, beta2, fburst, Mstar] = popt
    #tform = age_at_z - age
    tburst = age_at_z - burstage
    return fburst, tburst

def plot_sfh(fit, model_lookbacktime, model_sfh, ID, runID, plot_mean=False, model_f_burst=None, 
             model_burstage=None, ninty_region=False, samples=0, save=True):
    """
    Plots the regular SFH (SFR vs age of universe) plot on the top, cumulative SFH plot on the bottom
    """
    if 'redshift' in fit.posterior.samples.keys():
        post_z = np.median(fit.posterior.samples['redshift'])
    else: post_z = 0.04
    age_at_z = pipes.utils.cosmo.age(post_z).value
    
    #identify SFH component used
    if "psb2" in fit.fit_instructions.keys():
        SFH_comp = "psb2"
    elif "psb_wild2020" in fit.fit_instructions.keys():
        SFH_comp = "psb_wild2020"
    elif "psb_twin" in fit.fit_instructions.keys():
        SFH_comp = "psb_twin"

    #posterior sfh
    post_sfh = fit.posterior.samples['sfh']
    #median_sfh = np.median(post_sfh,axis=0)
    mean_sfh = np.mean(post_sfh,axis=0)
    age_of_universe = np.interp(post_z, pipes.utils.z_array, pipes.utils.age_at_z)
    post_ages = age_of_universe - fit.posterior.sfh.ages*10**-9
    post_ages_int = post_ages.copy()[::-1]*10**9
    #post_m_total = np.trapz(y=median_sfh[::-1], x=post_ages_int)
    # integrate to get cumulative median
    #c_median_sfh = integrate_sfh(post_ages_int, median_sfh[::-1], Mstar=post_m_total)

    #model sfh
    model_sfh = model_sfh.copy()
    model_ages = age_at_z-model_lookbacktime.copy()
    model_ages_int = model_ages.copy()[::-1]*10**9
    model_m_total = np.trapz(y=model_sfh[::-1], x=model_ages_int)
    # integrate to get cumulative of model sfh
    c_model_sfh = integrate_sfh(model_ages_int, model_sfh[::-1], Mstar=model_m_total)
    
    print('only recovered',10**np.median(fit.posterior.samples[SFH_comp+":massformed"])
          /model_m_total,'of total mass formed.')
    print(np.median(fit.posterior.samples[SFH_comp+":massformed"]), np.log10(model_m_total))

    #calculating posterior tx and their uncertainties
    mass_percentiles = np.linspace(0,1,5)[1:-1]
    txs = np.zeros([len(mass_percentiles), fit.posterior.n_samples])
    c_sfh_samples = []
    for i,sfh_sample in enumerate(fit.posterior.samples['sfh']):
        sfh_ = sfh_sample[::-1]
        c_sfh_ = integrate_sfh(post_ages_int, sfh_)
        c_sfh_samples.append(c_sfh_)
        txs[:,i] = np.interp(mass_percentiles, c_sfh_, post_ages_int)
    txs = txs/10**9
    tx_percentiles = []
    for i,txi in enumerate(txs):
        tx_percentiles.append(np.percentile(txi, (16,50,84)))
    tx_percentiles = np.array(tx_percentiles)
    #print(tx_percentiles)
    c_sfh_percentiles = np.percentile(c_sfh_samples, (16,50,84), axis=0)
    c_sfh_mean = np.mean(c_sfh_samples, axis=0)
    
    # check if using complex CEH models
    plot_metallicity = False
    if "metallicity_type" in fit.fit_instructions[SFH_comp].keys():
        if fit.fit_instructions[SFH_comp]["metallicity_type"] != 'delta':
            plot_metallicity = True
            zmet_evo = np.zeros([fit.posterior.n_samples, len(fit.posterior.sfh.ages)])
            for i in range(fit.posterior.n_samples):
                sfh_dict = {}
                for sfh_key in fit.fit_instructions[SFH_comp]:
                    try:
                        sfh_dict[sfh_key] = fit.posterior.samples[f'{SFH_comp}:{sfh_key}'][i]
                    except KeyError:
                        pass
                zmet_evo[i] = getattr(get_ceh_array,
                               fit.fit_instructions[SFH_comp]["metallicity_type"])(
                               fit.posterior.sfh.ages/10**9, sfh_dict)
            zmet_evo_percentiles = np.percentile(zmet_evo, (16,50,84), axis=0)
    
    ################# plotting 
    
    if plot_metallicity:
        fig = plt.figure(figsize=[15,13])
        gs = fig.add_gridspec(5,1, hspace=0.4)
        ax1 = plt.subplot(gs[:2])
        ax2 = plt.subplot(gs[2:4])
        ax3 = plt.subplot(gs[4])
        ax = [ax1, ax2, ax3]
    else:
        fig, ax = plt.subplots(2,1, figsize=[15,10])
    pipes.plotting.add_sfh_posterior(fit, ax[0], z_axis=False, zorder=9)
    if plot_mean:
        ax[0].plot(post_ages, mean_sfh, color='k', ls='--', zorder=7)
    if ninty_region:
        ninty_sfh = np.percentile(post_sfh, (5,95), axis=0)
        ax[0].fill_between(post_ages, ninty_sfh[0], ninty_sfh[1], color='gray', 
                           alpha=0.3, zorder=6)
    ax[0].plot(model_ages, model_sfh, zorder=10)
    ylim = ax[0].get_ylim()

    #calculate model burst fraction
    if model_f_burst is None and model_burstage is None:
        model_f_burst, model_t_burst = fit_f_burst(
            model_lookbacktime.copy(), model_sfh, age_at_z, SFH_comp)
    else:
        model_t_burst = age_at_z - model_burstage
    print('model f_burst and t_burst:',model_f_burst,model_t_burst)
    ax[0].vlines(model_t_burst, 0, ylim[1], color='red', ls='--', zorder=8)
    ax[0].arrow(age_at_z,ylim[1]*0.8,-(age_at_z-model_t_burst),0.0,color='red',head_width=np.max(ylim)/20., 
             head_length=0.1,length_includes_head=True, zorder=8)

    #use psb2's built in fburst and tburst posteriors to plot arrows
    post_f_burst = np.percentile(fit.posterior.samples[SFH_comp+":fburst"], (16,50,84))
    post_t_burst = age_of_universe-np.percentile(fit.posterior.samples[SFH_comp+":burstage"], (84,50,16))

    print('posterior f_burst and t_burst:',post_f_burst,post_t_burst)
    ax[0].vlines(post_t_burst[1], 0, ylim[1], color='sandybrown', ls='--', zorder=8)
    ax[0].arrow(age_of_universe,ylim[1]*0.9,-(age_of_universe-post_t_burst[1]),0.0,color='sandybrown',
             head_width=np.max(ylim)/20., head_length=0.1,length_includes_head=True, zorder=8)

    #plot vertical bands of tx percentiles
    for i,[l,m,u] in enumerate(tx_percentiles):
        ax[0].vlines(m, 0, 10*ylim[1], color = 'k', ls='--', alpha=0.5, zorder=1)
        ax[0].fill_betweenx([0,10*ylim[1]], l, u, facecolor='royalblue', alpha=(1.5-(i+1)/len(txs))/2.5,
                           zorder=1)
    
    ax[0].set_ylim(ylim)
    #add text about z, age at z, poster f_burst and t_burst
    f_burst_r = [np.round(post_f_burst[1],2),np.round(post_f_burst[2]-post_f_burst[1],2),
                 np.round(post_f_burst[1]-post_f_burst[0],2)]
    f_burst_text = f'post f\_burst={f_burst_r[0]}+{f_burst_r[1]}-{f_burst_r[2]}\n '
    t_burst_r = [np.round(post_t_burst[1],2),np.round(post_t_burst[2]-post_t_burst[1],2),
                 np.round(post_t_burst[1]-post_t_burst[0],2)]
    t_burst_text = f'post t\_burst={t_burst_r[0]}+{t_burst_r[1]}-{t_burst_r[2]}Gyr \n '
    ax[0].text(0.03,0.6,
            f'redshift={np.round(post_z,3)}\n ' + 
            f'age at z={np.round(age_at_z,2)}Gyr\n ' + 
            f_burst_text + 
            f'true f\_burst={np.round(model_f_burst,2)}\n ' +
            t_burst_text +
            f'true t\_burst={np.round(model_t_burst,2)}Gyr',
            fontsize=14, transform=ax[0].transAxes, bbox=dict(boxstyle='round', facecolor='white'))
    
    ax[0].set_xlim(ax[0].get_xlim()[::-1])
    pipes.plotting.add_z_axis(ax[0])
    
    ax[1].plot(model_ages[::-1], c_model_sfh, zorder=9)
    ax[1].plot(post_ages[::-1], c_sfh_percentiles[1], color='k', zorder=8)
    if plot_mean:
        ax[1].plot(post_ages[::-1], c_sfh_mean, color='k', ls='--', zorder=6)
    ax[1].fill_between(post_ages[::-1], c_sfh_percentiles[0], c_sfh_percentiles[2], color='gray', 
                       alpha=0.6, zorder=7)
    if ninty_region:
        c_ninty_sfh = np.percentile(c_sfh_samples, (5,95), axis=0)
        ax[1].fill_between(post_ages[::-1], c_ninty_sfh[0], c_ninty_sfh[1], color='gray', 
                           alpha=0.3, zorder=5)
    ax[1].errorbar(tx_percentiles[:,1], np.linspace(0,1,5)[1:-1], xerr=[tx_percentiles[:,1]-tx_percentiles[:,0],
                                                                        tx_percentiles[:,2]-tx_percentiles[:,1]],
              color='red', label='calculated equivilent tx times (assuming 4 bins)', fmt='o', zorder=10)
    
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel(ax[0].get_xlabel())
    ax[1].set_ylabel('fraction of cumulative mass formed')
    
    if plot_metallicity:
        # third plot
        ax[2].plot(post_ages, zmet_evo_percentiles[1], color='k', zorder=8)
        ax[2].fill_between(post_ages, zmet_evo_percentiles[0], zmet_evo_percentiles[2],
                       color='gray', alpha=0.6, zorder=7)
        if plot_mean:
            zmet_evo_mean = np.mean(zmet_evo, axis=0)
            ax[2].plot(post_ages, zmet_evo_mean, color='k', ls='--', zorder=6)
        if ninty_region:
            zmet_evo_ninty = np.percentile(zmet_evo, (5,95), axis=0)
            ax[2].fill_between(post_ages, zmet_evo_ninty[0], zmet_evo_ninty[1], color='gray',
                               alpha=0.3, zorder=5)
        zmet_ylims = ax[2].get_ylim()
        # vertical band of jump age
        if fit.fit_instructions[SFH_comp]['metallicity_type'] == 'psb_two_step':
            step_age_percentiles = age_of_universe - np.percentile(
                fit.posterior.samples[f'{SFH_comp}:burstage'], (16,50,84))
            ax[2].axvline(step_age_percentiles[1], color='steelblue', zorder=1)
            ax[2].fill_between([step_age_percentiles[0], step_age_percentiles[2]],
                               [zmet_ylims[0]]*2, [zmet_ylims[1]]*2, color='steelblue',
                               alpha=0.3, zorder=0)
        elif fit.fit_instructions[SFH_comp]['metallicity_type'] == 'two_step':
            step_age_percentiles = age_of_universe - np.percentile(
                fit.posterior.samples[f'{SFH_comp}:metallicity_step_age'], (16,50,84))
            ax[2].axvline(step_age_percentiles[1], color='steelblue', zorder=1)
            ax[2].fill_between([step_age_percentiles[0], step_age_percentiles[2]],
                               [zmet_ylims[0]]*2, [zmet_ylims[1]]*2, color='steelblue',
                               alpha=0.3, zorder=0)
        ax[2].set_xlim(ax[0].get_xlim())
        ax[2].set_ylim(zmet_ylims)
        ax[2].set_xlabel(ax[0].get_xlabel())
        ax[2].set_ylabel('$\\mathrm{Z_{*}}/Z_{\\odot}$')
        ax[2].text(0.03,0.90,
            f"model:{fit.fit_instructions[SFH_comp]['metallicity_type'].replace('_',' ')}",
            fontsize=14, transform=ax[2].transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white'), zorder=20)
    
    if samples > 0:
        samp_ind = np.random.randint(fit.posterior.n_samples, size=samples)
        for samp_i in samp_ind:
            ax[0].plot(post_ages, fit.posterior.samples['sfh'][samp_i], color='black', alpha=0.3, ls='--',
                       zorder=5)
            ax[1].plot(post_ages[::-1], c_sfh_samples[samp_i], color='black', alpha=0.3, ls='--', zorder=5)
            if plot_metallicity:
                ax[2].plot(post_ages, zmet_evo[samp_i], color='black', alpha=0.3, ls='--', zorder=5)
    
    if save:
        fig.savefig('pipes/plots/'+runID+'/'+ID+'_combined_sfh.pdf')
    plt.show()
    
    return fig,ax

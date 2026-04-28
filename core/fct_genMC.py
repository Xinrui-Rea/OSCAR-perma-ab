"""
Copyright: IIASA (International Institute for Applied Systems Analysis), 2016-2021; CEA (Commissariat a L'Energie Atomique) & UVSQ (Universite de Versailles et Saint-Quentin), 2016
Contributor(s): Thomas Gasser (gasser@iiasa.ac.at), Yann Quilcaille

This software is a computer program whose purpose is to simulate the behavior of the Earth system, with a specific but not exclusive focus on anthropogenic climate change.

This software is governed by the CeCILL license under French law and abiding by the rules of distribution of free software.  You can use, modify and/ or redistribute the software under the terms of the CeCILL license as circulated by CEA, CNRS and INRIA at the following URL "http://www.cecill.info". 

As a counterpart to the access to the source code and rights to copy, modify and redistribute granted by the license, users are provided only with a limited warranty and the software's author, the holder of the economic rights, and the successive licensors have only limited liability. 

In this respect, the user's attention is drawn to the risks associated with loading, using, modifying and/or developing or reproducing the software by the user in light of its specific status of free software, that may mean that it is complicated to manipulate, and that also therefore means that it is reserved for developers and experienced professionals having in-depth computer knowledge. Users are therefore encouraged to load and test the software's suitability as regards their requirements in conditions enabling the security of their systems and/or data to be ensured and,  more generally, to use and operate it in the same conditions as regards security. 

The fact that you are presently reading this means that you have had knowledge of the CeCILL license and that you accept its terms.
"""

##################################################
##################################################

import random
import warnings
import numpy as np
import xarray as xr
import scipy.stats as st

from scipy.integrate import quad
from scipy.optimize import fsolve


##################################################
##   1. ANCILLARY FUNCTIONS
##################################################

## function to get lognorm distrib parameters
def lognorm_distrib_param(mean, std):
    mu = np.log(mean / np.sqrt(1. + std**2./mean**2.))
    sigma = np.sqrt(np.log(1. + std**2./mean**2.))
    return mu, sigma


## function to infer logitnorm distrib parameters
def logitnorm_distrib_param(mean, std):
    ## error function
    def err(par):
        exp, _ = quad(lambda x, mu, sigma: 1/(1.-x) * 1./np.sqrt(2*np.pi*sigma**2.) * np.exp(-0.5*(np.log(x/(1.-x))-mu)**2./sigma**2.), 0, 1, args=tuple(par), limit=100)
        var, _ = quad(lambda x, mu, sigma: x/(1.-x) * 1./np.sqrt(2*np.pi*sigma**2.) * np.exp(-0.5*(np.log(x/(1.-x))-mu)**2./sigma**2.), 0, 1, args=tuple(par), limit=100)
        return np.array([exp-mean, np.sqrt(var-exp**2)-std])**2
    ## minimize error function
    try:
        par, _, fsolve_flag, _ = fsolve(err, [np.log(mean/(1.-mean)), np.sqrt(std/mean)], full_output=True)
        mu, sigma = par[0], np.abs(par[1])
    except ZeroDivisionError:
        fsolve_flag = 0
    ## return
    if fsolve_flag == 1: return mu, sigma
    else: return np.nan, np.nan


##################################################
## 2. GENERATE MONTE CARLO PARAMETERS
##################################################

## generate all Monte Carlo configurations 
def generate_config(Par0, nMC, kde_to_mod=False, mod_to_unc=False, mod_noise=0.1, kde_bw=None, seed=None):
    '''
    Function to generate Monte Carlo configuration (= parameters) for OSCAR.
    
    Input:
    ------
    Par0 (xr.Dataset)       dataset containing initial parameters
    nMC (int)               number of MC elements
    
    Output:
    -------
    Par_mc (xr.Dataset)     dataset containing MC parameters

    Options:
    --------
    kde_to_mod (bool)       turn all kde_ options to mod_ options;
                            default = False
    mod_to_unc (bool)       turn all mod_ options to unc_ options;
                            default = False
    mod_noise (float)       equivalent s.d. of relative noise added on top of mod_ options;
                            default = 0.1
    kde_bw                  bandwith option for kde_ options forwarded to scipy.stats.gaussian_kde;
                            default = None
    seed (int)              seed for random number generation forwarded to numpyp.random.default_rng;
                            default = None
    '''

    print('generating MC configurations')

    ## copy as precaution
    Par = Par0.copy(deep=True)

    ## list mod_ and kde_ dimensions
    mod_list = [coo for coo in Par.coords if coo[:4] == 'mod_']
    kde_list = [coo for coo in Par.coords if coo[:4] == 'kde_']

    ## list uncertainty options, parameters and check no mixing
    par_unc_list, par_mod_list, par_kde_list = [], [], []
    for par in Par:
        is_unc = any(['unc_' in dim for dim in Par[par].dims])
        is_mod = any(['mod_' in dim for dim in Par[par].dims])
        is_kde = any(['kde_' in dim for dim in Par[par].dims])
        if is_unc + is_mod + is_kde > 1:
            raise RuntimeError("Cannot mix unc_, mod_ and/or kde_ approaches; change parameter '{}'".format(par))     
        elif is_unc: par_unc_list.append(par)
        elif is_mod: par_mod_list.append(par)
        elif is_kde: par_kde_list.append(par)

    ## turn kde_ into mod_ (if requested)
    if kde_to_mod:
        Par = Par.rename({kde: kde.replace('kde_', 'mod_', 1) for kde in kde_list})
        mod_list, kde_list = mod_list + kde_list, []
        par_mod_list, par_kde_list = par_mod_list + par_kde_list, []

    ## turn mod_ to unc_ (if requested)
    ## assumes functional form based on provided values
    if mod_to_unc:
        for par in par_mod_list:
            if (Par[par] == Par[par].mean()).all():
                Par[par] = xr.DataArray(Par[par].mean(), attrs=Par[par].attrs)
                par_mod_list.remove(par)
            elif (Par[par] == Par[par]**2).all(): # switch
                Par[par] = xr.DataArray([0, 1], coords=['mini', 'maxi'], dims='unc_Choice', attrs=Par[par].attrs)
            elif (Par[par] >= 0).all() and (Par[par] <= 1).all():
                Par[par] = xr.DataArray([Par[par].mean(), Par[par].std()], coords=['mean', 'std'], dims='unc_LogitNorm', attrs=Par[par].attrs)
            elif (Par[par] >= 0).all() or (Par[par] <= 0).all():
                Par[par] = xr.DataArray([Par[par].mean(), Par[par].std()], coords=['mean', 'std'], dims='unc_LogNorm', attrs=Par[par].attrs)
            else:
                Par[par] = xr.DataArray([Par[par].mean(), Par[par].std()], coords=['mean', 'std'], dims='unc_Norm', attrs=Par[par].attrs)
        mod_list = []
        par_unc_list, par_mod_list = par_unc_list + par_mod_list, []

    ## initialize MC dataset
    Par_mc = xr.Dataset()
    Par_mc.coords['config'] = np.arange(nMC)

    ## set random state
    rng = np.random.default_rng(seed)

    ## draw unc_ configurations
    for par in par_unc_list:
        distrib = [dim.split('unc_')[-1] for dim in Par[par].dims if 'unc_' in dim]
        assert len(distrib) == 1, "Parameter '{}' has multiple uncertainty dimensions: {}".format(par, Par[par].dims)
        distrib = distrib[0]

        Par_mc[par] = sum([xr.zeros_like(Par[par].coords[coord], dtype=float) for coord in Par[par].coords if 'unc_' not in coord] + [Par_mc.coords['config']])
        dim_names = [dim for dim in Par[par].dims if 'unc_' not in dim]
        dim_sizes = [Par.sizes[dim] for dim in dim_names]
        index_iterator = np.ndindex(*dim_sizes)

        for indices in index_iterator:
            index_sel = {dim_name: index for dim_name, index in zip(dim_names, indices)}
            par_sel = Par[par].isel(index_sel, drop=True)
            ## Normal distrib
            if distrib == 'Norm':
                mean = par_sel.sel(unc_Norm='mean', drop=True)
                std = par_sel.sel(unc_Norm='std', drop=True)
                mu, sigma = mean, np.absolute(std)
                Norm = xr.DataArray(st.norm.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
                if len(dim_sizes) > 0:
                    Par_mc[par].isel(index_sel)[:] = mu + sigma * Norm
                else:
                    Par_mc[par] = mu + sigma * Norm

            ## LogNormal distrib
            elif distrib == 'LogNorm':
                mean = par_sel.sel(unc_LogNorm='mean', drop=True)
                std = par_sel.sel(unc_LogNorm='std', drop=True)
                mu, sigma = lognorm_distrib_param(abs(mean), np.absolute(std))
                Norm = xr.DataArray(st.norm.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
                if len(dim_sizes) > 0:
                    Par_mc[par].isel(index_sel)[:] = np.sign(mean) * np.exp(mu + sigma * Norm)
                else:
                    Par_mc[par] = np.sign(mean) * np.exp(mu + sigma * Norm)

            ## LogitNormal distrib
            elif distrib == 'LogitNorm':
                mean = par_sel.sel(unc_LogitNorm='mean', drop=True)
                std = par_sel.sel(unc_LogitNorm='std', drop=True)
                mu, sigma = logitnorm_distrib_param(abs(mean), np.abs(std))
                if np.isnan([mu, sigma]).any(): raise RuntimeError('Could not infer LogitNorm distribution for parameter {}'.format(par)) 
                Norm = xr.DataArray(st.norm.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
                if len(dim_sizes) > 0:
                    Par_mc[par].isel(index_sel)[:] = (1 + np.exp(mu + sigma * Norm)**-1)**-1
                else:
                    Par_mc[par] = (1 + np.exp(mu + sigma * Norm)**-1)**-1

            ## two HalfNormal distribs
            elif distrib == '2HalfNorm':
                mean = par_sel.sel(unc_2HalfNorm='mean', drop=True)
                std_neg = par_sel.sel(unc_2HalfNorm='std_neg', drop=True)
                std_pos = par_sel.sel(unc_2HalfNorm='std_pos', drop=True)
                mu, sigma_neg, sigma_pos = mean, abs(std_neg), np.abs(std_pos)
                Bool = xr.DataArray(st.randint(0, 2).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
                HalfNorm = xr.DataArray(st.halfnorm.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
                if len(dim_sizes) > 0:
                    Par_mc[par].isel(index_sel)[:] = mu + (Bool * sigma_pos - (1 - Bool) * sigma_neg) * HalfNorm
                else:
                    Par_mc[par] = mu + (Bool * sigma_pos - (1 - Bool) * sigma_neg) * HalfNorm

            ## Uniform distrib
            elif distrib == 'Uniform':
                mini = par_sel.sel(unc_Uniform='mini', drop=True)
                maxi = par_sel.sel(unc_Uniform='maxi', drop=True)
                Uniform = xr.DataArray(st.uniform.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
                if len(dim_sizes) > 0:
                    Par_mc[par].isel(index_sel)[:] = mini + (maxi - mini) * Uniform
                else:
                    Par_mc[par] = mini + (maxi - mini) * Uniform
                    
            ## Triangle distrib
            elif distrib == 'Triangle':
                mode = par_sel.sel(unc_Uniform='mode', drop=True)
                mini = par_sel.sel(unc_Uniform='mini', drop=True)
                maxi = par_sel.sel(unc_Uniform='maxi', drop=True)
                Triang = xr.DataArray(st.triang(c=(mode-mini)/(maxi-mini)).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
                if len(dim_sizes) > 0:
                    Par_mc[par].isel(index_sel)[:] = mini + (maxi - mini) * Triang
                else:
                    Par_mc[par] = mini + (maxi - mini) * Triang

            ## Discrete Uniform distrib
            elif distrib == 'Choice':
                mini = par_sel.sel(unc_Choice='mini', drop=True)
                maxi = par_sel.sel(unc_Choice='maxi', drop=True)
                mini, maxi = np.minimum(mini, maxi), np.maximum(mini, maxi)
                Choice = xr.DataArray(st.randint(mini, maxi + 1).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
                if len(dim_sizes) > 0:
                    Par_mc[par].isel(index_sel)[:] = Choice
                else:
                    Par_mc[par] = Choice

            ## error otherwise
            else:
                raise RuntimeError("Distribution {} not implemented for parameter '{}'".format(distrib, par))   

    ## draw mod_ configurations
    ## discrete draw of each mod
    Mod = xr.Dataset()
    for mod in mod_list:
        Mod[mod] = xr.DataArray(st.randint(0, len(Par[mod])).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
    ## applying selection (this keeps mod_ as secondary coordinate)
    Par_mod = xr.merge([Par[par] for par in par_mod_list])
    Par_mc = xr.merge([Par_mc, Par_mod.isel({mod: Mod[mod] for mod in mod_list})])
    ## adding noise based on Von Mises (if requested)
    if mod_noise > 0:
        for par in par_mod_list:
            if Par[par].dtype.name in ['int64', 'bool']:
                warnings.warn("Cannot add noise to parameter '{}' with dtype '{}'; skipping noise addition".format(par, Par[par].dtype.name))
                continue
            Noise = xr.DataArray(st.vonmises_line(kappa=1/mod_noise**2).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config}) / np.pi
            Par_mc[par] *= 1 + Noise
        for mod in mod_list: del Par_mc[mod]

    ## draw kde_ configurations
    for par in par_kde_list: assert len(Par[par].dims) == 1
    for kde in kde_list:
        par_list = [par for par in par_kde_list if kde in Par[par].dims]
        kde_draw = st.gaussian_kde(np.array([Par[par].values for par in par_list]), bw_method=kde_bw).resample(nMC)
        for n, par in enumerate(par_list):
            Par_mc[par] = ('config', kde_draw[n, :])

    ## add parameters without uncertainty
    for par in [par for par in Par if par not in par_unc_list + par_mod_list + par_kde_list]:
        Par_mc[par] = Par[par]

    ## copy attributes
    for par in Par:
        Par_mc[par].attrs = Par[par].attrs

    ## return
    return Par_mc


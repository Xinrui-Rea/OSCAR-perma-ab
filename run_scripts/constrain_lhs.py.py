#!/usr/bin/env python
# coding: utf-8

# # 1. Packages and functions

# In[ ]:


# 1.0. load modules

import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.lines import Line2D
from scipy.stats import lognorm, norm, skewnorm
from scipy.optimize import minimize
from pathlib import Path

code_path = Path.cwd().parent
sys.path.insert(0, code_path)
os.chdir(code_path)

from run_scripts.utils_perma import *


# In[2]:


# 1.1. pre-defined constraints

def create_constraint_specs(var_list=['D_Tg', 'D_CO2', 'D_CH4', 'D_N2O', 'D_OHC', 'ERF_ari', 'ERF_aci', 'D_Focean', 'D_Fland - D_Eluc', 'RF_AERtot']):
    '''
    Constraint specifications with CO2 as lognormal distribution

    Output:
    -------
    (list)                   list of constraint specifications (dicts)

    Options:
    -------
    var_list (list)          list of variable names to create specifications for constraining
    
    '''

    list_specs = [
        {   # temperature constraint - normal distribution, 2014–2023
            # https://doi.org/10.5194/essd-16-2625-2024, Table 5, very likely range
            'name': 'D_Tg',
            'type': 'normal', 
            'base': (1850, 1900),
            'period': (2014, 2023),
            'mean': 1.19,
            # 'std': 0.11,
            'std': (1.30 - 1.06) / 3.29,  # 90% PI to std (assuming normal distribution)
            'units': 'K',
        },
        {   # CO2 constraint - lognormal distribution, 2023 minus 1750 level
            # https://doi.org/10.5194/essd-17-965-2025, p969
            # https://doi.org/10.1017/9781009157896.009, p948
            'name': 'D_CO2',
            'type': 'lognormal',
            'period': (2023, 2023),
            'mean': 419.3 - 278.38,
            'std': (419.3 - 278.38) * 0.05, # 5% uncertainty
            'units': 'ppm',
        },
        {   # CH4 constraint - lognormal distribution, 2023 minus 1750 level
            # https://doi.org/10.5194/essd-16-2625-2024, Table 10
            # https://doi.org/10.1017/9781009157896.009, p948
            'name': 'D_CH4',
            'type': 'lognormal',
            'period': (2023, 2023),
            'mean': 1922.5 - 729.2,
            'std': (1922.5 - 729.2) * 0.07, # 7% uncertainty
            'units': 'ppb',
        },
        {   # N2O constraint - lognormal distribution, 2023 minus 1750 level
            # https://doi.org/10.5194/essd-16-2625-2024, Table 10
            # https://doi.org/10.1017/9781009157896.009, p948
            'name': 'D_N2O',
            'type': 'lognormal',
            'period': (2023, 2023),
            'mean': 336.9 - 270.1,
            'std': (336.9 - 270.1) * 0.25, # 25% uncertainty
            'units': 'ppb',
        },
        {   # OHC constraint - normal distribution, 2006–2018
            # https://doi.org/10.1017/9781009157896.009, Table 7.1, very likely range
            'name': 'D_OHC',
            'type': 'normal',
            'base': (2006, 2006),
            'period': (2018, 2018),
            'mean': 138.8,
            'std': (191.3 - 86.4) / 3.29,  # 90% PI to std (assuming normal distribution)
            'units': 'ZJ',
        },
        {   # aerosol-radiation RF constraint - normal distribution, 1750–2023
            # https://doi.org/10.5194/essd-17-2641-2025, Table 3
            'name': 'ERF_ari',
            'type': 'percentile',
            'period': (2023, 2023),
            'median': -0.26,
            'p5': -0.50,
            'p95': -0.03,
            'units': 'W/m²',
        },
        {   # aerosol-cloud RF constraint - normal distribution, 1750–2023
            # https://doi.org/10.5194/essd-17-2641-2025, Table 3
            'name': 'ERF_aci',
            'type': 'percentile',
            'period': (2023, 2023),
            'median': -0.91,
            'p5': -1.80,
            'p95': -0.27,
            'units': 'W/m²',
        },
        # {   # ocean carbon uptake constraint - normal distribution, 2014–2023
        #     # https://doi.org/10.5194/essd-17-965-2025, Table 7, ±1σ
        #     'name': 'D_Focean',
        #     'type': 'normal',
        #     'period': (2014, 2023),
        #     'mean': 2.9,
        #     'std': 0.4,
        #     'units': 'PgC/yr',
        # },
        # {   # land carbon uptake constraint - normal distribution, 2014–2023
        #     # https://doi.org/10.5194/essd-17-965-2025, Table 7, ±1σ
        #     'name': 'D_Fland - D_Eluc',
        #     'type': 'normal',
        #     'period': (2014, 2023),
        #     'mean': 9.7 - 5.2 - 2.9,
        #     'std': 0.9,
        #     'units': 'PgC/yr',
        # },
        {   # ocean carbon uptake constraint - normal distribution, 2015–2023
            # https://essd.copernicus.org/preprints/essd-2025-659/essd-2025-659.pdf, Table 7, ±1σ
            'name': 'D_Focean',
            'type': 'normal',
            'period': (2015, 2023),
            'mean': (10 * 3.2 - 3.4) / 9,
            'std': ((100 * 0.4**2 + 0.4**2) / 81)**0.5,
            'units': 'PgC/yr',
        },
        {   # land carbon uptake constraint - normal distribution, 2015–2023 minus 2023
            # https://essd.copernicus.org/preprints/essd-2025-659/essd-2025-659.pdf, Table 5, ±1σ
            'name': 'D_Fland - D_Eluc',
            'type': 'normal',
            'period': (2015, 2023),
            'mean': (1 * 10 - 0.7) / 9,
            'std': ((100 * 1**2 + 1.1**2) / 81)**0.5,
            'units': 'PgC/yr',
        },
        {   # total AER RF constraint - normal distribution, 1750–2023
            'name': 'RF_AERtot',
            'type': 'percentile',
            'period': (2023, 2023),
            'median': -1.17,
            'p5': -1.97,
            'p95': -0.37,
            'units': 'W/m²',
        }
    ]

    return [spec for spec in list_specs if spec['name'] in var_list]


# In[3]:


# 1.2. default functions

def format_var(var, var_specs):
    '''
    format variable according to variable specifications

    Input:
    ------
    var (xr.DataArray)          variable values
    var_specs (dict)            variable specifications

    Output:
    -------
    var_new (xr.DataArray)      formatted variable values
    '''

    baseline = var_specs.get('base', None)
    period = var_specs.get('period', None)
    print(f'Formatting variable {var.name} based on {var_specs["name"]}: period={period}, baseline={baseline}')

    var_baseline = var.sel(year=slice(*baseline)).mean(dim='year') if baseline else 0
    var = var.sel(year=slice(*period)).mean(dim='year') if period else var
    var_new = var - var_baseline
    var_new.name = var_specs['name']
    
    return var_new

def mahalanobis_distance(x, mean, inv_cov):
    '''
    Compute the Mahalanobis distance of each row in x from the mean

    Input:
    ------
    x (np.ndarray)          array of shape (n_samples, n_features)
    mean (np.ndarray)       mean vector of shape (n_features,)
    inv_cov (np.ndarray)    inverse covariance matrix of shape (n_features, n_features)

    Output:
    -------
    (np.ndarray)            array of shape (n_samples,) with Mahalanobis distances
    '''
    diff = x - mean
    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

def dist_spec(spec, values, method='cdf'):
    '''
    Evaluate distribution specified by spec at given values
    Input:
    ------
    spec (dict)         distribution specification
    values (np.ndarray) values at which to evaluate

    Output:
    -------
    vals (np.ndarray)   evaluated values

    Options:
    --------
    method (str)       method to use: 'cdf', 'pdf', or 'ppf'
                       default = 'cdf'
    '''
    
    if spec['type'] == 'normal':
        # normal distribution
        if method == 'cdf': vals = norm.cdf(values, loc=spec['mean'], scale=spec['std'])
        if method == 'pdf': vals = norm.pdf(values, loc=spec['mean'], scale=spec['std'])
        if method == 'ppf': vals = norm.ppf(values, loc=spec['mean'], scale=spec['std'])
        
    elif spec['type'] == 'lognormal':
        # lognormal distribution - convert mean/std to lognormal parameters
        mean, std = spec['mean'], spec['std']
        mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
        sigma = np.sqrt(np.log(1 + (std/mean)**2))
        if method == 'cdf': vals = lognorm.cdf(values, s=sigma, scale=np.exp(mu))
        if method == 'pdf': vals = lognorm.pdf(values, s=sigma, scale=np.exp(mu))
        if method == 'ppf': vals = lognorm.ppf(values, s=sigma, scale=np.exp(mu))
    
    elif spec['type'] == 'percentile':
        # percentile-based distribution (e.g., 5th, median, 95th)
        p5, median, p95 = spec['p5'], spec['median'], spec['p95']
        
        if abs((median - p5) - (p95 - median)) < 1e-10:
            # symmetric percentiles - use normal approximation
            mean = median
            std = (p95 - p5) / (norm.ppf(0.95) - norm.ppf(0.05))
            if method == 'cdf': vals = norm.cdf(values, loc=mean, scale=std)
            if method == 'pdf': vals = norm.pdf(values, loc=mean, scale=std)
            if method == 'ppf': vals = norm.ppf(values, loc=mean, scale=std)
        else:
            # non-symmetric - use properly fitted skew-normal distribution
            loc, scale, shape = fit_skewnorm_from_percentiles(p5, median, p95)
            if method == 'cdf': vals = skewnorm.cdf(values, shape, loc=loc, scale=scale)
            if method == 'pdf': vals = skewnorm.pdf(values, shape, loc=loc, scale=scale)
            if method == 'ppf': vals = skewnorm.ppf(values, shape, loc=loc, scale=scale)

    elif spec['type'] == 'skewnormal':
        # direct skew-normal parameters
        loc, scale, shape = spec['loc'], spec['scale'], spec['shape']
        if method == 'cdf': vals = skewnorm.cdf(values, shape, loc=loc, scale=scale)
        if method == 'pdf': vals = skewnorm.pdf(values, shape, loc=loc, scale=scale)
        if method == 'ppf': vals = skewnorm.ppf(values, shape, loc=loc, scale=scale)

    return vals

def fit_skewnorm_from_percentiles(p5, median, p95, max_iter=100):
    '''
    Properly fit skew-normal distribution to percentiles using optimization

    Input:
    ------
    p5 (float)          5th percentile
    median (float)      50th percentile (median)
    p95 (float)         95th percentile

    Output:
    -------
    loc (float)         location parameter of fitted skew-normal
    scale (float)       scale parameter of fitted skew-normal
    shape (float)       shape parameter of fitted skew-normal

    Options:
    --------
    max_iter (int)      maximum iterations for optimization
                        default = 100
    '''
    def objective(params):
        loc, scale, shape = params
        try:
            dist = skewnorm(shape, loc=loc, scale=scale)
            p5_est = dist.ppf(0.05)
            median_est = dist.ppf(0.5)
            p95_est = dist.ppf(0.95)
            
            # weight errors by importance (focus on matching percentiles)
            error = (abs(p5_est - p5) + 
                    2 * abs(median_est - median) +  # Emphasize median
                    abs(p95_est - p95))
            return error
        except:
            return np.inf
    
    # better initial guesses
    initial_guesses = [
        [median, (p95 - p5)/3.0, 0.0],      # near-normal
        [median, (p95 - p5)/2.5, 2.0],      # positive skew
        [median, (p95 - p5)/2.5, -2.0],     # negative skew
        [(p5 + median + p95)/3, (p95 - p5)/2.0, 1.0],  # balanced
    ]
    
    best_params = None
    best_error = np.inf
    
    for init_guess in initial_guesses:
        try:
            result = minimize(objective, init_guess, 
                            bounds=[(None, None), (1e-6, None), (None, None)],
                            method='L-BFGS-B',
                            options={'maxiter': max_iter})
            
            if result.success and result.fun < best_error:
                best_error = result.fun
                best_params = result.x
        except:
            continue
    
    if best_params is None:
        # fallback: use empirical CDF
        print(f'Warning: Skew-normal fit failed for percentiles p5={p5}, median={median}, p95={p95}')
        print('Falling back to empirical distribution')
        return None
    
    loc, scale, shape = best_params
    
    # verify the fit
    dist = skewnorm(shape, loc=loc, scale=scale)
    p5_fit = dist.ppf(0.05)
    median_fit = dist.ppf(0.5)
    p95_fit = dist.ppf(0.95)
    
    print(f'Skew-normal fit: p5={p5_fit:.3f} (target {p5:.3f}), '
          f'median={median_fit:.3f} (target {median:.3f}), '
          f'p95={p95_fit:.3f} (target {p95:.3f})')
    
    return loc, scale, shape

def LHS_configs(simulated_results, constraint_specs, N_post, frac_ma=1, use_scipy=True):
    '''
    Select configurations using Latin Hypercube sampling in constraint space
    Returns the selected CONFIG INDICES
    
    Input:
    ------
    simulated_results (np.ndarray)          simulated results for each configuration
    constraint_specs (list of dicts)        specification for temperature and CO2 distributions
    N_post (int)                            number of configurations to select
    
    Output:
    --------
    selected_indices (np.ndarray of int)    config indices that were selected

    Options:
    --------
    frac_ma (float)                         fraction of Mahalanobis distance in combined distance metric
                                            default = 1 (only Mahalanobis distance)
    use_scipy (bool)                        whether to use scipy's LHS sampler
                                            default = True
    '''
    
    n_constraints = simulated_results.shape[1]
    
    # create LHS space in constraint space
    if use_scipy:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_constraints)
        lhs_design = sampler.random(n=N_post)
        print('Using scipy\'s LatinHypercube sampler')
    else:
        # Generate optimal Latin Hypercube design
        lhs_design = np.zeros((N_post, n_constraints), dtype=float)
        for j in range(n_constraints):
            ## ? whether to use fixed or random position within each bin
            lhs_design[:, j] = (np.random.permutation(N_post) + np.random.uniform(0.1, 0.9)) / N_post
        print('Using custom LatinHypercube sampler')

    uniform_space = np.zeros((N_post, n_constraints), dtype=float)
    
    for j, spec in enumerate(constraint_specs):
        values = lhs_design[:, j]
        uniform_space[:, j] = dist_spec(spec, values, method='ppf')

    # calculate covariance for Mahalanobis distance
    cov_matrix = np.cov(uniform_space.T)
    try:
        inv_cov_ma = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov_ma = np.eye(n_constraints)

    inv_cov_eu = np.eye(n_constraints)

    # for each LHS point drawn from the observational space, find the closest prior sample
    selected_indices = []
    used_indices = set()
    
    for lhs_point in uniform_space:
        available_mask = ~np.isin(np.arange(len(simulated_results)), list(used_indices))
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) == 0: break
            
        available_points = simulated_results[available_indices]
        
        # calculate both distances
        dist_mahalanobis = mahalanobis_distance(available_points, lhs_point, inv_cov_ma)
        dist_euclidean = mahalanobis_distance(available_points, lhs_point, inv_cov_eu)

        # combine distances
        combined_distances = frac_ma * dist_mahalanobis + (1 - frac_ma) * dist_euclidean
        
        best_idx = available_indices[np.argmin(combined_distances)]
        selected_indices.append(best_idx)
        used_indices.add(best_idx)
    
    return selected_indices

def analyze_selection(selected_indices, constraint_specs, all_simulated):
    print('\n=== LATIN HYPERCUBE SELECTION RESULTS ===')
    print(f'Selected {len(selected_indices)} configurations')
    print(f'Selected config indices: {selected_indices[:10]}...')  # Show first 10

    for j, spec in enumerate(constraint_specs):
        selected_vals = all_simulated[:, j][selected_indices]
        all_vals = all_simulated[:, j]
        
        print(f'\n--- {spec['name']} ({spec['type']}) ---')
        print(f'Observed constraint: {spec}')
        print(f'All configs:    mean={all_vals.mean():.2f}, std={all_vals.std():.2f}')
        print(f'Selected configs: mean={selected_vals.mean():.2f}, std={selected_vals.std():.2f}')
        print(f'Selected range: [{selected_vals.min():.2f}, {selected_vals.max():.2f}]')
        
        if spec['type'] == 'lognormal':
            # log-space analysis
            log_selected = np.log(selected_vals)
            log_all = np.log(all_vals)
            print(f'Log-space - All: μ={log_all.mean():.3f}, σ={log_all.std():.3f}')
            print(f'Log-space - Selected: μ={log_selected.mean():.3f}, σ={log_selected.std():.3f}')
        if spec['type'] == 'percentile':
            # percentile analysis
            p5, median, p95 = np.percentile(selected_vals, [5, 50, 95])
            print(f'Percentiles of selected: 5th={p5:.2f}, 50th={median:.2f}, 95th={p95:.2f}')

def plot_selection_results(selected_indices, constraint_specs, all_simulated):

    if len(all_simulated[1]) > 2:
        fig, axes = plt.subplots(2, (len(all_simulated[1]) + 1) // 2, figsize=((len(all_simulated[1]) + 1) // 2 * 4, 8))
    else:
        fig, axes = plt.subplots(1, len(all_simulated[1]), figsize=(len(all_simulated[1]) * 4, 4))

    for i in range(len(all_simulated[1])):
        if len(all_simulated[1]) != 1:
            ax = axes[i % 2, i // 2] if len(all_simulated[1]) > 2 else axes[i]
        else:
            ax = axes
        var = all_simulated[:, i]
        selected_var = all_simulated[:, i][selected_indices]

        ax.hist(var, bins=30, alpha=0.5, density=True, label='All', color='gray')
        ax.hist(selected_var, bins=20, alpha=0.5, density=True, label='Sel', color='blue')
        x = np.linspace(var.min(), var.max(), 100)
        ax.plot(x, dist_spec(constraint_specs[i], x, method='pdf'), color='red', linestyle='--', lw=2, label='Obs')
        ax.set_ylabel('Density')
        ax.set_xlabel(f'{constraint_specs[i]['units']}')
        ax.set_title(f'{constraint_specs[i]['name']}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()


# # 2. Constraining

# In[9]:


# 2.1. main workflow

if __name__ == '__main__':

    ## choose main OSCAR results as the baseline
    print('Loading prior data...')
    OUTPUT = True

    dir = './results/abrupt/'
    dir_ind = './results/abrupt/'
    var_all = ['D_Tg', 'D_CO2', 'D_CH4', 'D_N2O', 'D_OHC', 'RF_AERtot', 'RF_cloud2', 'D_Focean', 'D_Fland']
    # var_all = ['D_Tg', 'D_CO2', 'D_CH4', 'D_N2O', 'D_OHC', 'RF_AERtot', 'RF_cloud2', 'D_Focean']
    var_cons = ['D_Tg', 'D_CO2', 'D_CH4', 'D_N2O', 'D_OHC', 'D_Focean', 'RF_AERtot']

    # define constraint specifications
    constraint_specs_all = create_constraint_specs()
    constraint_specs = create_constraint_specs(var_list=var_cons)

    N_post = 600

    for LU_data in ['Houghton-FRA2020', 'LUH2-TRENDYv12']:
        fn_ind_out = f'{dir_ind}selected_indices_{LU_data}_{N_post}.csv'

        ds_list = []
        for varname in var_all:
            ds_list.append(xr.load_dataarray(f'{dir}{varname}_hist.nc').sel(data_LULCC=LU_data, drop=True))
            if varname == 'D_Fland': 
                ds_list.append(xr.load_dataarray(f'{dir}D_Eluc_hist.nc').sel(data_LULCC=LU_data, drop=True))

        ds = xr.merge(ds_list)

        D_Tg = format_var(ds['D_Tg'], [f for f in constraint_specs_all if f['name'] == 'D_Tg'][0])
        print(f'{D_Tg.name} {D_Tg.mean().values:.2f} ± {D_Tg.std().values:.2f} K')
        D_CO2 = format_var(ds['D_CO2'], [f for f in constraint_specs_all if f['name'] == 'D_CO2'][0])
        print(f'{D_CO2.name} {D_CO2.mean().values:.2f} ± {D_CO2.std().values:.2f} ppm')
        D_CH4 = format_var(ds['D_CH4'], [f for f in constraint_specs_all if f['name'] == 'D_CH4'][0])
        print(f'{D_CH4.name} {D_CH4.mean().values:.2f} ± {D_CH4.std().values:.2f} ppb')
        D_N2O = format_var(ds['D_N2O'], [f for f in constraint_specs_all if f['name'] == 'D_N2O'][0])
        print(f'{D_N2O.name} {D_N2O.mean().values:.2f} ± {D_N2O.std().values:.2f} ppb')
        D_OHC = format_var(ds['D_OHC'], [f for f in constraint_specs if f['name'] == 'D_OHC'][0])
        print(f'{D_OHC.name} {D_OHC.mean().values:.2f} ± {D_OHC.std().values:.2f} ZJ/yr')

        ERF_aci = format_var(ds['RF_cloud2'], [f for f in constraint_specs_all if f['name'] == 'ERF_aci'][0])
        print(f'{ERF_aci.name} {ERF_aci.mean().values:.2f} ± {ERF_aci.std().values:.2f} W/m²')
        RF_AERtot = format_var(ds['RF_AERtot'], [f for f in constraint_specs_all if f['name'] == 'RF_AERtot'][0])
        RF_AERtot.name = 'RF_AERtot'
        print(f'{RF_AERtot.name} {RF_AERtot.mean().values:.2f} ± {RF_AERtot.std().values:.2f} W/m²')
        ERF_ari = format_var(ds['RF_AERtot'] - ds['RF_cloud2'], [f for f in constraint_specs_all if f['name'] == 'ERF_ari'][0])
        print(f'{ERF_ari.name} {ERF_ari.mean().values:.2f} ± {ERF_ari.std().values:.2f} W/m²')

        D_Focean = format_var(ds['D_Focean'], [f for f in constraint_specs_all if f['name'] == 'D_Focean'][0])
        print(f'{D_Focean.name} {D_Focean.mean().values:.2f} ± {D_Focean.std().values:.2f} PgC/yr')

        D_Fland = format_var(ds['D_Fland'] - ds['D_Eluc'], [f for f in constraint_specs_all if f['name'] == 'D_Fland - D_Eluc'][0])
        print(f'{D_Fland.name} {D_Fland.mean().values:.2f} ± {D_Fland.std().values:.2f} PgC/yr')
        data_all = [D_Tg, D_CO2, D_CH4, D_N2O, D_OHC, ERF_ari, ERF_aci, D_Focean, D_Fland, RF_AERtot]
        # data_all = [D_Tg, D_CO2, D_CH4, D_N2O, D_OHC, ERF_ari, ERF_aci, D_Focean, RF_AERtot]

        data_in = []
        data_out = []
        for i, var in enumerate(data_all):
            if var.name in var_cons:
                data_in.append(var)
            else:
                data_out.append(var)

        simulated_results = np.column_stack(data_in)

        if os.path.exists(fn_ind_out):
            print(f'Selected indices for {LU_data} already exist. Loading file.')
            OUTPUT = False
            selected_indices = np.loadtxt(fn_ind_out, dtype=int)
        else:
            print(f'\nSelecting {N_post} configurations ...')
            selected_indices = LHS_configs(simulated_results, constraint_specs, N_post=N_post, use_scipy=True, frac_ma=1)
        analyze_selection(selected_indices, constraint_specs, simulated_results)
        plot_selection_results(selected_indices, constraint_specs, simulated_results)

        print(f'\n=== FINAL SELECTED CONFIG INDICES ===')
        print(f'Total selected: {len(selected_indices)} configs')
        print(f'Config indices range: {np.array(selected_indices).min()} to {np.array(selected_indices).max()}')
        print(f'First 20 selected indices:\n{np.array(selected_indices)[:20]}')

        ## plot other unselected variables
        if len(data_out) > 0:
            print('\nPlotting unselected variables...')
            others_simulated = np.column_stack(data_out)
            plot_selection_results(selected_indices, create_constraint_specs(var_list=[var.name for var in data_out]), others_simulated)

        # save the selected config indices
        if OUTPUT: np.savetxt(fn_ind_out, selected_indices, fmt='%d')
        print(f'\nSaved selected config indices to "{fn_ind_out}"')


# In[10]:


# 2.2. time series: scenario

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

N_post = 600
varname = 'D_Tg'
title = r'ΔGMST (°C)'
# title = r'LULCC emissions (PgC yr$^{-1}$)'

PLOT_HIST = False
PLOT_PRIOR = False
AVG_NGFS = False

estimator = 'median'
# errorbar = ('pi', 100)
errorbar = None

dir_hist = './results/abrupt/'
dir_scen = './results/abrupt/'
dir_ind = './results/abrupt/'

sel_dict = {'scen':scens_sorted, 'mod': mods}
if varname in ['dmg_SLR', 'dmg_T', 'dmg_tot']: 
    base_year = (1986, 2005)
elif varname in ['D_Tg']:
    base_year = (1850, 1900)
elif varname in ['RF_CO2', 'D_CO2', 'D_N2O', 'D_CH4', 'D_Eluc']:
    base_year = (1750, 1750)
var_hist = xr.open_dataarray(f'{dir_hist}{varname}_hist.nc')
var_hist_base = var_hist.sel(year=slice(*base_year)).mean('year')
var_hist = var_hist - var_hist_base
for attr in ['unit', 'units']:
    if attr in var_hist.attrs:
        unit = f' ({var_hist.attrs[attr]})'
        break
else:
    print(f'No unit attribute found in {varname}_hist.nc')
    unit = ''
var_scen = xr.open_dataarray(f'{dir_scen}{varname}_scen.nc').sel(**sel_dict)
var_scen = var_scen - var_hist_base

var_list_hist = []
var_list_scen = []
for LU_data in ['Houghton-FRA2020', 'LUH2-TRENDYv12']:
        selected_indices = np.loadtxt(f'{dir_ind}selected_indices_{LU_data}_{N_post}.csv', dtype=int)
        var_hist_sel = var_hist.sel(data_LULCC=LU_data)
        var_hist_sel = var_hist_sel.isel(config=selected_indices)
        var_hist_sel = var_hist_sel.expand_dims(data_LULCC=[LU_data])
        var_hist_sel.coords['config'] = np.arange(len(var_hist_sel.config))
        var_list_hist.append(var_hist_sel)
        var_scen_sel = var_scen.sel(data_LULCC=LU_data)
        var_scen_sel = var_scen_sel.isel(config=selected_indices)
        var_scen_sel = var_scen_sel.expand_dims(data_LULCC=[LU_data])
        var_scen_sel.coords['config'] = np.arange(len(var_scen_sel.config))
        var_list_scen.append(var_scen_sel)

var_hist_sel = xr.concat(var_list_hist, dim='data_LULCC')
var_scen_sel = xr.concat(var_list_scen, dim='data_LULCC')
var_hist_sel.name = varname
var_scen_sel.name = varname
print(f'{len(selected_indices)} out of {len(var_hist.config)} configs selected')

dim_avg = ['data_LULCC', 'mod']
for dim in dim_avg:
    try:
        print(f'Averaging over dimension: {dim}')
        var_hist = var_hist.mean(dim=dim)
        var_hist_sel = var_hist_sel.mean(dim=dim)
    except ValueError:
        print(f'Dimension {dim} not found in historical data array, skipping averaging over this dimension.')
    try:
        print(f'Averaging over dimension: {dim}')
        var_scen = var_scen.mean(dim=dim)
        var_scen_sel = var_scen_sel.mean(dim=dim)
    except ValueError:
        print(f'Dimension {dim} not found in scenario data array, skipping averaging over this dimension.')

try:
    var_ngfs = xr.load_dataarray(f'results/ngfs/{varname}_ngfs.nc').sel(**sel_dict).squeeze()
    var_ngfs = var_ngfs.sel(percentile=50) if 'percentile' in var_ngfs.coords else var_ngfs
    if AVG_NGFS:
        var_ngfs = var_ngfs.mean('mod') if 'mod' in var_ngfs.coords else var_ngfs
    PLOT_NGFS = True
except Exception:
    PLOT_NGFS = False

fig, axes = plt.subplots(
    2, len(sel_dict['scen']) // 2 + 1, figsize=(9, 5), 
    **{'sharex': True, 'sharey': True}, dpi=300
)
axes = axes.flatten()
if PLOT_PRIOR:
    legend_handles = [
        Line2D([0], [0], color='gray', ls='--'), 
        Line2D([0], [0], color='blue', ls='-')
    ]
    legend_labels = ['Prior', 'Posterior']
else:
    legend_handles = [Line2D([0], [0], color='blue', ls='-')]
    legend_labels = ['OSCAR']

median_hist_prior = var_hist.sel(year=slice(2014, 2023)).median().values
p5_hist_prior = var_hist.sel(year=slice(2014, 2023)).quantile(0.05).values
p95_hist_prior = var_hist.sel(year=slice(2014, 2023)).quantile(0.95).values
median_hist_post = var_hist_sel.sel(year=slice(2014, 2023)).median().values
p5_hist_post = var_hist_sel.sel(year=slice(2014, 2023)).quantile(0.05).values
p95_hist_post = var_hist_sel.sel(year=slice(2014, 2023)).quantile(0.95).values
print(f'Historical {varname} in 2014–2023: ')
print(f'Prior: {median_hist_prior:.2f} [{p5_hist_prior:.2f}–{p95_hist_prior:.2f}]')
print(f'Posterior: {median_hist_post:.2f} [{p5_hist_post:.2f}–{p95_hist_post:.2f}]')

for i, ax in enumerate(axes):
    try:
        ax.tick_params(axis='both', which='major', labelsize='small')

        if PLOT_NGFS:
            if 'mod' in var_ngfs.coords:
                style = 'mod'
            else:
                style = None
            sns.lineplot(
                data=var_ngfs.sel(scen=sel_dict['scen'][i]).to_dataframe(), x='year', y=varname, 
                color='green', alpha=0.8,
                style=style,
                dashes=mod_ls if style else None, 
                estimator=None,
                ax=ax
            )
            if style: 
                handles, labels = ax.get_legend_handles_labels()
                labels = ['NGFS-'+l[:3] for l in labels]
            else:
                handles, labels = [Line2D([0], [0], color='green', ls='-')], ['NGFS']
            try:
                ax.legend_.remove()
            except AttributeError:
                pass

        if PLOT_HIST:
            df_hist = var_hist.to_dataframe()
            df_hist_sel = var_hist_sel.to_dataframe()

            if PLOT_PRIOR:
                sns.lineplot(
                    data=df_hist, x='year', y=varname, 
                    estimator=estimator, errorbar=errorbar, 
                    color='gray', alpha=0.7, ls='--',
                    ax=ax
                )

            sns.lineplot(
                data=df_hist_sel, x='year', y=varname, 
                estimator=estimator, errorbar=errorbar, 
                color='blue', alpha=0.5,
                legend=False,
                ax=ax
            )

        df_scen = var_scen.sel(scen=sel_dict['scen'][i]).to_dataframe()
        df_scen_sel = var_scen_sel.sel(scen=sel_dict['scen'][i]).to_dataframe()
        
        if PLOT_PRIOR:
            sns.lineplot(
                data=df_scen, x='year', y=varname, 
                estimator=estimator, errorbar=errorbar, 
                color='gray', alpha=0.7, ls='--', label=False, 
                ax=ax
            )

        sns.lineplot(
            data=df_scen_sel, x='year', y=varname, 
            estimator=estimator, errorbar=errorbar, 
            color='blue', alpha=0.7,
            legend=False,
            ax=ax
        )

        try:
            ax.legend_.remove()
        except AttributeError:
            pass

        ax.grid(ls=':', alpha=0.5, which='both')
        if ax.get_ylim()[0] < 0:
            ax.axhline(0, color='black', lw=1, ls=':', alpha=0.7)
        ax.axvline(2050, ls=':', color='k', lw=1, alpha=0.7)
        ax.axvline(2100, ls=':', color='k', lw=1, alpha=0.7)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(standardize_scen_names([sel_dict['scen'][i]])[0], fontsize='large', fontstyle='italic')
        ax.xaxis.set_major_locator(plt.FixedLocator([2025, 2050, 2075, 2100]))
        ax.tick_params(axis='both', which='both', direction='in')

        print(f'\nScenario: {sel_dict["scen"][i]}')
        for sel_year in [2050, 2100]:
            median_prior = var_scen.sel(scen=sel_dict["scen"][i], year=sel_year).median(dim="config").values.item()
            p5_prior = var_scen.sel(scen=sel_dict["scen"][i], year=sel_year).quantile(0.05, dim="config").values.item()
            p95_prior = var_scen.sel(scen=sel_dict["scen"][i], year=sel_year).quantile(0.95, dim="config").values.item()
            median_post = var_scen_sel.sel(scen=sel_dict["scen"][i], year=sel_year).median(dim="config").values.item()
            p5_post = var_scen_sel.sel(scen=sel_dict["scen"][i], year=sel_year).quantile(0.05, dim="config").values.item()
            p95_post = var_scen_sel.sel(scen=sel_dict["scen"][i], year=sel_year).quantile(0.95, dim="config").values.item()
            print(f'before constraining in {sel_year}: {median_prior:.2f} [{p5_prior:.2f}–{p95_prior:.2f}]')
            print(f'after constraining in {sel_year}: {median_post:.2f} [{p5_post:.2f}–{p95_post:.2f}]')
            if PLOT_NGFS: 
                if 'mod' in var_ngfs.dims:
                    print(f'NGFS in {sel_year}: {var_ngfs.mean("mod").sel(scen=sel_dict["scen"][i], year=sel_year).values:.2f}')
                else:
                    print(f'NGFS in {sel_year}: {var_ngfs.sel(scen=sel_dict["scen"][i], year=sel_year).values:.2f}')

    except IndexError:
        ax.axis('off')
        try:
            if estimator == 'median':
                ax.text(
                    0.05, 0.95, f'{errorbar[1]}% {errorbar[0].upper()} around {estimator}', 
                    transform=ax.transAxes, fontsize='medium', va='top'
                )
            elif estimator == 'mean':
                ax.text(
                    0.05, 0.95, f'{errorbar[1]} {errorbar[0].upper()} around {estimator}', 
                    transform=ax.transAxes, fontsize='medium', va='top' 
                )
        except TypeError:
            pass
        
fig.supxlabel('Year', fontsize='large')
fig.supylabel(f'{title}{unit}', fontsize='large')

if PLOT_NGFS:
    for h, l in zip(handles, labels):
        legend_handles.append(h)
        legend_labels.append(l)

fig.legend(
    handles=legend_handles,
    labels=legend_labels,
    bbox_to_anchor=(0.9, 0.15), 
    loc='lower right', 
    ncol=1,
    frameon=False
)
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.9, wspace=0.1, hspace=0.2)
plt.show()


# In[25]:


# 2.2. time series: historical

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

from scipy import stats

varname = 'D_Htot'
units = 'mm'
title = f'SLR ({units})'
# title = r'ΔGMST (°C)'
# title = r'LULCC emissions (PgC yr$^{-1}$)'
N_post = 500

PLOT_HIST = True
PLOT_PRIOR = True
PLOT_OBS = True

dir_hist = './results/abrupt/'
dir_scen = './results/abrupt/'
dir_ind = './results/abrupt/'

sel_dict = {'scen':scens_sorted, 'mod': mods}
if varname in ['dmg_SLR', 'dmg_T', 'dmg_tot']: 
    base_year = (1986, 2005)
elif varname in ['D_Tg']:
    base_year = (1850, 1900)
elif varname in ['RF_CO2', 'D_CO2', 'D_N2O', 'D_CH4', 'D_Eluc']:
    base_year = (1750, 1750)
var_hist = xr.open_dataarray(f'{dir_hist}{varname}_hist.nc')
var_hist_base = var_hist.sel(year=slice(*base_year)).mean('year')
var_hist = var_hist - var_hist_base
for attr in ['unit', 'units']:
    if attr in var_hist.attrs:
        unit = f' ({var_hist.attrs[attr]})'
        break
else:
    print(f'No unit attribute found in {varname}_hist.nc')
    unit = ''
var_scen = xr.open_dataarray(f'{dir_scen}{varname}_scen.nc').sel(**sel_dict)
var_scen = var_scen - var_hist_base

var_list_hist = []
var_list_scen = []
for LU_data in ['Houghton-FRA2020', 'LUH2-TRENDYv12']:
        selected_indices = np.loadtxt(f'{dir_ind}selected_indices_{LU_data}_{N_post}.csv', dtype=int)
        var_hist_sel = var_hist.sel(data_LULCC=LU_data)
        var_hist_sel = var_hist_sel.isel(config=selected_indices)
        var_hist_sel = var_hist_sel.expand_dims(data_LULCC=[LU_data])
        var_hist_sel.coords['config'] = np.arange(len(var_hist_sel.config))
        var_list_hist.append(var_hist_sel)
        var_scen_sel = var_scen.sel(data_LULCC=LU_data)
        var_scen_sel = var_scen_sel.isel(config=selected_indices)
        var_scen_sel = var_scen_sel.expand_dims(data_LULCC=[LU_data])
        var_scen_sel.coords['config'] = np.arange(len(var_scen_sel.config))
        var_list_scen.append(var_scen_sel)

var_hist_sel = xr.merge(var_list_hist).to_dataarray()
var_scen_sel = xr.merge(var_list_scen).to_dataarray()
var_hist_sel.name = varname
var_scen_sel.name = varname
print(f'{len(selected_indices)} out of {len(var_hist.config)} configs selected')

dim_avg = ['data_LULCC', 'mod']
for dim in dim_avg:
    try:
        print(f'Averaging over dimension: {dim}')
        var_hist = var_hist.mean(dim=dim)
        var_hist_sel = var_hist_sel.mean(dim=dim)
    except ValueError:
        print(f'Dimension {dim} not found in historical data array, skipping averaging over this dimension.')
    try:
        print(f'Averaging over dimension: {dim}')
        var_scen = var_scen.mean(dim=dim)
        var_scen_sel = var_scen_sel.mean(dim=dim)
    except ValueError:
        print(f'Dimension {dim} not found in scenario data array, skipping averaging over this dimension.')

estimator = 'median'
errorbar = ('pi', 100)
# errorbar = None

slope_sim = stats.theilslopes(var_hist_sel.median('config').sel(year=slice(2014, 2023)).squeeze().values, np.arange(2014, 2024)).slope
print(f'Simulated slope (2014-2023): {slope_sim:.3f} {units}/yr')

try:
    if varname == 'D_Tg':
        var_obs = xr.load_dataset('results/global-temperature_IGCC.nc')['GSAT'].rename('D_Tg')
    elif varname == 'D_Htot':
        df_obs = pd.read_csv('results/SLR_IGCC.csv')
        df_long = df_obs.melt(id_vars=['scenario', 'type', 'unit', 'variable', 'version', 'model', 'region'], var_name='year', value_name='val')
        var_obs = df_long.assign(year=df_long['year'].astype(int)).set_index(['year'])['val'].to_xarray() * 10 # convert from cm to mm
        var_obs = var_obs.rename('D_Htot')
    slope_obs = stats.theilslopes(var_obs.sel(year=slice(2014, 2023)).squeeze().values, np.arange(2014, 2024)).slope
    print(f'Observed slope (2014-2023): {slope_obs:.3f} {units}/yr')
    PLOT_OBS = True
except Exception:
    PLOT_OBS = False


fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
if PLOT_PRIOR:
    legend_handles = [
        Line2D([0], [0], color='gray', ls='--'), 
        Line2D([0], [0], color='blue', ls='-')
    ]
    legend_labels = ['Prior', 'Posterior']
else:
    legend_handles = [Line2D([0], [0], color='blue', ls='-')]
    legend_labels = ['OSCAR']

ax.tick_params(axis='both', which='major', labelsize='small')

if PLOT_OBS:
    sns.lineplot(
        data=var_obs.to_dataframe(), x='year', y=varname,
        color='red', alpha=0.8,
        ax=ax
    )
    legend_handles.append(Line2D([0], [0], color='red', ls='-'))
    legend_labels.append('Observations')

if PLOT_HIST:
    df_hist = var_hist.to_dataframe()
    df_hist_sel = var_hist_sel.to_dataframe()
    
    if PLOT_PRIOR:
        sns.lineplot(
            data=df_hist, x='year', y=varname, 
            estimator=estimator, errorbar=errorbar, 
            color='gray', alpha=0.7, ls='--',
            ax=ax
        )

    sns.lineplot(
        data=df_hist_sel, x='year', y=varname, 
        estimator=estimator, errorbar=errorbar, 
        color='blue', alpha=0.5,
        legend=False,
        ax=ax
    )
ax.set_xlabel('Year', fontsize='large')
ax.set_ylabel(f'{title}{unit}', fontsize='large')
ax.tick_params(axis='both', which='both', direction='in')
ax.set_xlim(1850, 2023)

fig.legend(
    handles=legend_handles,
    labels=legend_labels,
    bbox_to_anchor=(0.1, 0.95), 
    fontsize='medium',
    loc='upper left', 
    ncol=1,
    frameon=False
)
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.9, wspace=0.1, hspace=0.2)
plt.show()


# In[ ]:


# 2.3. single distribution

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

varname = 'D_CO2'
dir = './results/abrupt/'
dir_ind = './results/abrupt/'
period = 'hist'
N_post = 500
sel_dict = {}
var = xr.load_dataarray(f'{dir}{varname}_{period}.nc').sel(sel_dict)

(mean, std) = (419.3 - 278.38, (419.3 - 278.38) * 0.05)  # observational constraint parameters
mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
sigma = np.sqrt(np.log(1 + (std/mean)**2))

var_list = []
mod = mods[0]
scen = scens[4]
for LU_data in ['Houghton-FRA2020', 'LUH2-TRENDYv12']:
    selected_indices = np.loadtxt(f'{dir_ind}selected_indices_{LU_data}_{N_post}.csv', dtype=int)
    var_sel = var.sel(data_LULCC=LU_data)
    var_sel = var_sel.isel(config=selected_indices)
    var_sel = var_sel.expand_dims(data_LULCC=[LU_data])
    var_sel.coords['config'] = np.arange(len(var_sel.config))
    var_list.append(var_sel)
var_sel = xr.concat(var_list, dim='data_LULCC')

for attr in ['unit', 'units']:
    if attr in var.attrs:
        unit = f' ({var.attrs[attr]})'
        break
else:
    print(f'No unit attribute found in {varname}_scen_{LU_data}.nc')
    unit = ''

# var = var.sel(year=slice(2014, 2023)).mean(dim='year') - var.sel(year=slice(1850, 1900)).mean(dim='year')
# var_sel = var_sel.sel(year=slice(2014, 2023)).mean(dim='year') - var_sel.sel(year=slice(1850, 1900)).mean(dim='year')
var = var.sel(year=2023)
var_sel = var_sel.sel(year=2023)

var = var.stack(new_dim=('data_LULCC', 'config'))
var_sel = var_sel.stack(new_dim=('data_LULCC', 'config'))

fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
## histogram over configs
sns.histplot(
    var, element='step', stat='density', common_norm=False, kde=True, 
    alpha=0.5, label='Prior', color='gray', ax=ax
)
sns.histplot(
    var_sel, element='step', stat='density', common_norm=False, kde=True, 
    alpha=0.5, label='Posterior', color='blue', ax=ax
)

x = np.linspace(var.min().values, var.max().values, 200)
pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
ax.plot(x, pdf, 'r--', lw=2, label='Obs')

# plt.text(
#     0.7, 0.98, f'Prior: {var.mean().values:.2f} ± {var.std().values:.2f}{unit}', 
#     transform=ax.transAxes, fontsize='medium', va='top'
# )
# plt.text(
#     0.7, 0.92, f'Posterior: {var_sel.mean().values:.2f} ± {var_sel.std().values:.2f}{unit}', 
#     transform=ax.transAxes, fontsize='medium', va='top'
# )
# plt.text(
#     0.7, 0.86, f'Obs: {mu} ± {sigma}{unit}', 
#     transform=ax.transAxes, fontsize='medium', va='top'
# )

title =f'{varname} in ' + ', '.join([f'{v}' for k, v in sel_dict.items()]) if sel_dict else f'{varname}'
plt.ylabel('Density', fontsize='large')
plt.xlabel(f'ΔCO$_2$ (ppm)', fontsize='large')
plt.legend(fontsize='medium', loc='upper right')


# In[ ]:


# 2.4. statistics

for dir in ['../results/noperma/', '../results/gradual/', '../results/abrupt/']:
    print(f'\n=== Checking directory: {dir} ===')
    for varname in ['D_Tg']:
        print(f'\n--- Variable: {varname} ---')
        var_hist = xr.open_dataarray(f'{dir}{varname}_hist.nc').sel(year=2020)
        var_scen = xr.open_dataarray(f'{dir}{varname}_scen.nc').sel(year=2020).isel(mod=0, scen=0)
        print(var_hist.mean().values, var_scen.mean().values, var_hist.mean().values - var_scen.mean().values)
        for LU_data in ['Houghton-FRA2020', 'LUH2-TRENDYv12']:
            print(f'Using LULCC data: {LU_data}')
            selected_indices = np.loadtxt(f'selected_indices_{LU_data}.csv', dtype=int)
            var_hist_sel = var_hist.sel(data_LULCC=LU_data).isel(config=selected_indices)
            var_scen_sel = var_scen.sel(data_LULCC=LU_data).isel(config=selected_indices)
            print(var_hist_sel.mean().values, var_scen_sel.mean().values, var_hist_sel.mean().values - var_scen_sel.mean().values)


# In[ ]:


# 2.5. comparison

varname = 'D_Tg'

dir1 = '../results/gradual/offset/'
dir2 = '../results/gradual/lulcc/'

var1 = xr.open_dataarray(f'{dir1}{varname}_hist.nc')
var2 = xr.open_dataarray(f'{dir2}{varname}_hist.nc')

var1.mean(dim=['config', 'data_LULCC']).plot(x='year', label='offset')
var2.mean(dim=['config', 'data_LULCC']).plot(x='year', label='lulcc')


# # 3. Output

# In[5]:


# 3.1. select the constrained results

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

var_keep = ['D_Tg', 'D_CO2', 'D_OHC', 'D_Htot', 'dmg_T', 'dmg_SLR']
var_keep_slr = ['D_Hais', 'D_Hgis', 'D_Hgla', 'D_Hthx']
var_keep_perma = ['D_Epf_ab_CO2', 'D_Epf_ab_CH4', 'D_Epf_gr_CO2', 'D_Epf_gr_CH4']

dir_ind = 'results/abrupt/'
for dir in ['results/abrupt/']:
    for period in ['scen']:
        for varname in var_keep_perma:        
            print(f'\nProcessing variable: {varname}')

            var_list = []
            for LU_data in ['Houghton-FRA2020', 'LUH2-TRENDYv12']:
                selected_ind = np.sort(np.loadtxt(f'{dir_ind}selected_indices_{LU_data}.csv', dtype=int))
                ds_scen = xr.load_dataarray(f'{dir}{varname}_{period}.nc').sel(data_LULCC=LU_data)
                selected_scen = ds_scen.isel(config=selected_ind)
                selected_scen['config'] = np.arange(len(selected_ind))
                var_list.append(selected_scen)

            xr.concat(var_list, dim='data_LULCC').to_netcdf(f'{dir}constrained/{varname}_{period}.nc', mode='w')
            print(f'Saved constrained results to "{dir}constrained/{varname}_{period}.nc"')


# In[ ]:


# 3.2. output percentiles into a csv file

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

varname = 'dmg_tot'
sel_dict = {'year': 2100, 'mod': mods[0], 'scen': [scens[id] for id in [0, 1, 5, 6]]}
if varname in ['dmg_SLR', 'dmg_T', 'dmg_tot']: 
    base_year = (1986, 2005)
else:
    base_year = (1850, 1900)

dir1 = './results/noperma/constrained/'
dir2 = './results/gradual/constrained/'
dir3 = './results/abrupt/constrained/'

var_scen1 = xr.load_dataarray(f'{dir1}{varname}_scen.nc')
sel_dict = {k: v for k, v in sel_dict.items() if k in var_scen1.dims}
for attr in ['unit', 'units']:
    if attr in var_scen1.attrs:
        unit = f'{var_scen1.attrs[attr]}'
        break
else:
    print(f'No unit attribute found in {varname}_scen.nc')
    unit = ''

var_scen1 = var_scen1.sel(**sel_dict, drop=True)
var_scen1_base = xr.load_dataarray(f'{dir1}{varname}_hist.nc').sel(year=slice(*base_year)).mean('year')
var1 = (var_scen1 - var_scen1_base)
var1.name = 'no_permafrost'
var_scen2 = xr.load_dataarray(f'{dir2}{varname}_scen.nc').sel(**sel_dict, drop=True)
var_scen2_base = xr.load_dataarray(f'{dir2}{varname}_hist.nc').sel(year=slice(*base_year)).mean('year')
var2 = (var_scen2 - var_scen2_base)
var2.name = 'gradual'
var_scen3 = xr.load_dataarray(f'{dir3}{varname}_scen.nc').sel(**sel_dict, drop=True)
var_scen3_base = xr.load_dataarray(f'{dir3}{varname}_hist.nc').sel(year=slice(*base_year)).mean('year')
var3 = (var_scen3 - var_scen3_base)
var3.name = 'gradual+abrupt'

dims = [dim for dim in var1.dims if dim not in ['config']]

## print p5, p50, p95 into a csv file
with open(f'{varname}.csv', 'w') as f:
    f.write(f'Simulation,{" ,".join(dims)},5th Percentile,Median,95th Percentile,unit\n')
    for var in [var1.stack(z=dims), var2.stack(z=dims), var3.stack(z=dims)]:
        for i in range(var.sizes['z']):
            var_sub = var.isel(z=i)
            p5 = var_sub.quantile(0.05)
            p50 = var_sub.quantile(0.5)
            p95 = var_sub.quantile(0.95)
            label = f'{var.name},' + ','.join([str(z) for z in var_sub.z.values.tolist()])
            f.write(f'{label},{p5:.4f},{p50:.4f},{p95:.4f},{unit}\n')


# In[ ]:


# 3.3. combine two variables

var_list = ['dmg_T', 'dmg_SLR']
var_new = 'dmg_tot'

dir1 = './results/noperma/constrained/'
dir2 = './results/gradual/constrained/'
dir3 = './results/abrupt/constrained/'

for dir in [dir1, dir2, dir3]:
    for period in ['hist', 'scen']:
        print(f'Processing period: {period}')
        var_dataarrays = []
        for varname in var_list:
            print(f'  Loading variable: {varname}')
            var_da = xr.load_dataarray(f'{dir}{varname}_{period}.nc')
            var_dataarrays.append(var_da)
        var_combined = sum(var_dataarrays)
        var_combined.name = var_new
        var_combined.attrs['unit'] = var_dataarrays[0].attrs.get('unit', '')
        var_combined.to_netcdf(f'{dir}{var_new}_{period}.nc', mode='w')
        print(f'  Saved combined variable to "{dir}{var_new}_{period}.nc"')


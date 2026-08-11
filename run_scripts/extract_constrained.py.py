#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## 0. load packages

import os, sys
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

code_path = Path.cwd().parent
sys.path.insert(0, code_path)
os.chdir(code_path)

from run_scripts.utils_perma import *


# # 1. Parameters

# In[ ]:


## 1.1 select constrained parameter sets

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

dir_ind = 'results/abrupt/'
Par = xr.load_dataset('results/Par/Par.nc')

par_config = [par for par in Par.data_vars if 'config' in Par[par].dims]
par_others = [par for par in Par.data_vars if par not in par_config]
Par_config = Par[par_config]

par_list = []
for LU_data in ['Houghton-FRA2020', 'LUH2-TRENDYv12']:
    selected_ind = np.sort(np.loadtxt(f'{dir_ind}selected_indices_{LU_data}.csv', dtype=int))
    try:
        Par_sel = Par_config.sel(data_LULCC=LU_data, drop=True)
    except:
        pass
    Par_sel = Par_config.isel(config=selected_ind)
    Par_sel['config'] = np.arange(len(selected_ind))
    par_list.append(Par_sel.expand_dims({'data_LULCC': [LU_data]}))

Par_config_cons = xr.concat(par_list, dim='data_LULCC')
Par_config_cons = xr.merge([Par_config_cons, Par[par_others]])
Par_config_cons.to_netcdf(f'{dir_ind}constrained/Par_cons.nc', mode='w')
print(f'Saved constrained results to "{dir_ind}constrained/Par_cons_2023.nc"')


# In[ ]:


## 1.2 constrained abrupt permafrost parameters

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

dir = 'results/Par/'
Par_steady = xr.load_dataset('results/Par/Par_steady.nc')
par_config = [par for par in Par_steady.data_vars if 'config' in Par_steady[par].dims]
par_others = [par for par in Par_steady.data_vars if par not in par_config]
Par_config = Par_steady[par_config]

random_indices = np.random.choice(Par_config.coords['config'], size=500, replace=True)
Par_config = Par_config.isel({'config': random_indices})
Par_config.coords['config'] = np.arange(len(Par_config.config))
Par_config = xr.merge([Par_config, Par_steady[par_others]])
Par_config.to_netcdf(f'{dir}Par_steady_cons.nc', mode='w')
print(f'Saved constrained steady-state results to "{dir}Par_steady_cons.nc"')


# In[4]:


## 1.3 constrained SLR parameters

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

dir = 'results/Par/'
Par_steady = xr.load_dataset('results/Par/Par_SLR_post.nc')
par_config = [par for par in Par_steady.data_vars if 'config' in Par_steady[par].dims]
par_others = [par for par in Par_steady.data_vars if par not in par_config]
Par_config = Par_steady[par_config]

random_indices = np.random.choice(Par_config.coords['config'], size=500, replace=True)
Par_config = Par_config.isel({'config': random_indices})
Par_config.coords['config'] = np.arange(len(Par_config.config))
Par_config = xr.merge([Par_config, Par_steady[par_others]])
Par_config.to_netcdf(f'{dir}Par_SLR_post_cons.nc', mode='w')
print(f'Saved constrained steady-state results to "{dir}Par_SLR_post_cons.nc"')


# In[5]:


## 1.4 constrained damage parameters

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

dir = 'results/Par/'
Par_steady = xr.load_dataset('results/Par/Par_dmg.nc')
par_config = [par for par in Par_steady.data_vars if 'config' in Par_steady[par].dims]
par_others = [par for par in Par_steady.data_vars if par not in par_config]
Par_config = Par_steady[par_config]

random_indices = np.random.choice(Par_config.coords['config'], size=500, replace=True)
Par_config = Par_config.isel({'config': random_indices})
Par_config.coords['config'] = np.arange(len(Par_config.config))
Par_config = xr.merge([Par_config, Par_steady[par_others]])
Par_config.to_netcdf(f'{dir}Par_dmg_cons.nc', mode='w')
print(f'Saved constrained steady-state results to "{dir}Par_dmg_cons.nc"')


# # 2. Run

# In[ ]:


## 2.1. create constrained intial conditions

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

dir = 'results/abrupt/'
dir_ind = 'results/abrupt/'

batch_size = 100
n_loops = 2400 // batch_size
hist_list = []
for i in range(n_loops):
    hist = xr.load_dataset(f'{dir}/hist_batch{i}.nc').sel(year=2023)
    hist_list.append(hist)

hist_all = xr.concat(hist_list, dim='config')

hist_list = []
for LU_data in ['Houghton-FRA2020', 'LUH2-TRENDYv12']:
    selected_ind = np.sort(np.loadtxt(f'{dir_ind}selected_indices_{LU_data}.csv', dtype=int))
    hist_sel = hist_all.sel(data_LULCC=LU_data, drop=True).isel(config=selected_ind)
    hist_sel['config'] = np.arange(len(selected_ind))
    hist_list.append(hist_sel)

hist_sel = xr.concat(hist_list, dim='data_LULCC')
hist_sel.to_netcdf(f'{dir}constrained/hist_init_2023.nc', mode='w')
print(f'Saved constrained initial conditions to "{dir}constrained/hist_init_2023.nc"')


# In[ ]:


## 2.2. run constrained projections for main model

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

sim = 'noperma'

ini_year = 2023
dir = f'results/{sim}/'
ini = xr.load_dataset(f'{dir}constrained/hist_init_{ini_year}.nc')

Par = xr.load_dataset(f'results/gradual/constrained/Par_cons_{ini_year}.nc')
if sim == 'gradual':
    from core.mod_process import OSCAR
elif sim == 'abrupt':
    from core.OSCAR_perma_ab import OSCAR_perma_ab
    OSCAR = OSCAR_perma_ab(option='online')
    Par = xr.merge([Par, xr.load_dataset(f'{dir}constrained/Par_steady_cons.nc')])
elif sim == 'noperma':
    from core.OSCAR_perma_ab import OSCAR_noperma
    OSCAR = OSCAR_noperma()
    
print(OSCAR.var_prog)

## offset emissions
print('Offsetting emissions...')
For_hist = xr.load_dataset('results/For/For_hist_fair.nc')
For_scen = xr.load_dataset('results/For/For_scen_fair.nc')
vars = [var for var in For_hist.data_vars if var.startswith('E_')] + ['Eff']
for var in vars:
    For_hist[var] = For_hist[var] - For_hist[var].sel(year=1750, drop=True)
    For_scen[var] = For_scen[var] - For_hist[var].sel(year=1750, drop=True)

var_keep = ['RF_CH4', 'D_CH4', 'RF_CO2', 'D_CO2', 'RF', 'D_Tg', 'D_OHC', 'RF_cloud2', 'RF_AERtot', 'D_Eluc', 'D_Fland', 'D_Focean', 'D_N2O']

# create small batches
batch_size = 100
n_loops = len(Par.config) // batch_size

result_list = []
for i in range(n_loops):
    print(f'Running batch {i+1} of {n_loops}...')
    batch_selected = np.arange(i * batch_size, (i + 1) * batch_size)
    mod_list = []
    for mod in For_scen.mod.values:
        scen_list = []
        for scen in For_scen.scen.values:
            Out_scen = OSCAR(Ini=ini.sel(config=batch_selected), Par=Par.sel(config=batch_selected), For=For_scen.sel(year=slice(ini_year, None), mod=mod, scen=scen, drop=True), var_keep=var_keep, adapt_nt=True)
            scen_list.append(Out_scen.expand_dims(scen=[scen]))
        mod_list.append(xr.concat(scen_list, dim='scen').expand_dims(mod=[mod]))
    result_list.append(xr.concat(mod_list, dim='mod'))
Out_scen = xr.concat(result_list, dim='config')
Out_scen.to_netcdf(f'{dir}constrained/scen_cons.nc', mode='w')


# In[ ]:


## 2.3. SLR module

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

from core.OSCAR_SLR import OSCAR_SLR
OSCAR_SLR = OSCAR_SLR(option='offline')

dir = 'results/noperma/constrained/'
Par_SLR = xr.load_dataset('results/Par/Par_SLR_post_cons.nc')

## baseline in Pathfinder is 1850-1900 period
For_scen = xr.merge([xr.load_dataset(f'{dir}D_Tg_scen.nc'), xr.load_dataset(f'{dir}D_OHC_scen.nc')])
For_hist = xr.merge([xr.load_dataset(f'{dir}D_Tg_hist.nc'), xr.load_dataset(f'{dir}D_OHC_hist.nc')])
print(f'Historical time range: {For_hist["year"].min().values} - {For_hist["year"].max().values}')
if For_hist['year'].min().values > 1850 or For_hist['year'].max().values < 1900:
    raise ValueError('Historical forcing does not cover 1850-1900 period!')

D_Tg_base = For_hist['D_Tg'].sel(year=slice(1850, 1900)).mean(dim='year')
D_OHC_base = For_hist['D_OHC'].sel(year=slice(1850, 1900)).mean(dim='year')

For_hist['D_Tg'] = For_hist['D_Tg'] - D_Tg_base
For_scen['D_Tg'] = For_scen['D_Tg'] - D_Tg_base

For_hist['D_OHC'] = For_hist['D_OHC'] - D_OHC_base
For_scen['D_OHC'] = For_scen['D_OHC'] - D_OHC_base

## run simulations

var_keep = ['D_Htot']

Out_hist = OSCAR_SLR(Ini=None, For=For_hist, Par=Par_SLR, var_keep=var_keep)
for var in var_keep:
    Out_hist[var].to_netcdf(f'{dir}{var}_hist.nc')
Out_scen = OSCAR_SLR(Ini=Out_hist.sel(year=2023, drop=True), For=For_scen, Par=Par_SLR, var_keep=var_keep)
for var in var_keep:
    Out_scen[var].to_netcdf(f'{dir}{var}_scen.nc')
print(f'Finished SLR simulations and saved to {dir}')


# In[ ]:


## 2.4. damage module

from core.OSCAR_dmg import OSCAR_dmg
OSCAR_dmg = OSCAR_dmg()

Par = xr.load_dataset('results/Par/Par_dmg_cons.nc')

## load drivers
dir = 'results/noperma/constrained/'
For_scen = xr.load_dataarray(dir + 'D_Tg_scen.nc')
For_hist = xr.load_dataarray(dir + 'D_Tg_hist.nc')
print(f'Historical time range: {For_hist["year"].min().values} - {For_hist["year"].max().values}')
if For_hist['year'].min().values > 1850 or For_hist['year'].max().values < 1900:
    raise ValueError('Historical forcing does not cover 1850-1900 period!')

For_scen_slr = xr.load_dataarray(dir + 'D_Htot_scen.nc')
For_hist_slr = xr.load_dataarray(dir + 'D_Htot_hist.nc')
print(f'Historical time range: {For_hist_slr["year"].min().values} - {For_hist_slr["year"].max().values}')
if For_hist_slr['year'].min().values > 1850 or For_hist_slr['year'].max().values < 1900:
    raise ValueError('Historical forcing does not cover 1850-1900 period!')

##! baseline for temperature damage functions is 1986-2005
## the shift of 0.6K is to account for the difference between pre-industrial (1850-1900) and 1986-2005
## https://utrechtuniversity.github.io/mimosa/components/damages/
D_Tg_base = For_hist.sel(year=slice(1850, 1900)).mean('year') - 0.6
D_Htot_base = For_hist_slr.sel(year=slice(1850, 1900)).mean('year')

For_hist = For_hist - D_Tg_base
For_scen = For_scen - D_Tg_base
For_hist_slr = For_hist_slr - D_Htot_base
For_scen_slr = For_scen_slr - D_Htot_base

For_hist = xr.merge([For_hist, For_hist_slr])
For_scen = xr.merge([For_scen, For_scen_slr])

var_keep = ['dmg_T', 'dmg_SLR', 'dmg_tot']
Out_hist = OSCAR_dmg(Ini=None, For=For_hist, Par=Par, var_keep=var_keep)
for var in Out_hist.data_vars:
    Out_hist[var].to_netcdf(dir + f'{var}_hist.nc')

Out_scen = OSCAR_dmg(Ini=Out_hist.sel(year=2023, drop=True), For=For_scen, Par=Par, var_keep=var_keep)
for var in Out_scen.data_vars:
    Out_scen[var].to_netcdf(dir + f'{var}_scen.nc')
    
print(f'Done! Results saved to {dir}')


# # 3. Results

# In[ ]:


## 3.1. extract scen variables from constrained runs

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

sim = 'abrupt'
dir = f'results/{sim}/'
for var in ['D_OHC']:
    os.system(f'ncks -O -v {var} {dir}constrained/scen_cons.nc {dir}constrained/{var}_scen.nc')


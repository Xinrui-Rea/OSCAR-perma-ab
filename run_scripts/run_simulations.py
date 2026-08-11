#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 0. options
import os, sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

code_path = Path.cwd().parent
sys.path.insert(0, code_path)
os.chdir(code_path)


# ## 1. normal run

# In[ ]:


## 1.1 historical simulation

dir = 'results/abrupt/'

Par = xr.load_dataset('results/Par/Par.nc')
For_hist = xr.load_dataset('results/For/For_hist_fair.nc')

from core.mod_process import OSCAR
print(OSCAR.var_prog)

var_keep = ['RF_CH4', 'D_CH4', 'RF_CO2', 'D_CO2', 'RF', 'D_Tg']

## run OSCAR
Out_hist = OSCAR(Ini=None, Par=Par, For=For_hist, var_keep=var_keep)
# Out_hist.to_netcdf(dir + 'hist_fair.nc')


# In[ ]:


## 1.2 future simulation

(yr_start, yr_end) = (2020, 2100)

Par = xr.load_dataset('results/Par.nc')
For_scen = xr.load_dataset('results/For_scen.nc')
Out_hist = xr.load_dataset(dir + 'hist.nc')

var_keep = ['RF_CH4', 'D_CH4', 'RF_CO2', 'D_CO2', 'RF', 'D_Tg']

Out_scen = OSCAR(Ini=Out_hist.sel(year=yr_start, drop=True), Par=Par, For=For_scen, var_keep=var_keep)
Out_scen.to_netcdf(dir + 'scen.nc')


# In[3]:


## 1.3 combine all modules

from core.mod_process import OSCAR

from core.OSCAR_SLR import OSCAR_SLR
from core.OSCAR_dmg import OSCAR_dmg

OSCAR_all = OSCAR.merge(OSCAR_SLR()).merge(OSCAR_dmg())
for lvl, keys in OSCAR_all.proc_levels().items():
    for key in keys:
        if ('CO2' in key) | ('N2O' in key):
            print(f'Level {lvl} - {key}')
            print('Input:', OSCAR_all._processes[key].In)
for var in OSCAR_all.var_prog:
    if var in ['D_Tg', 'D_CO2', 'D_CH4', 'D_N2O', 'D_OHC', 'D_Focean', 'RF_AERtot']:
        print(f'Constrained variable: {var}')
        
OSCAR_all.display(random=False)


# ## 2. batched run

# In[ ]:


# 2.1. abrupt (dynamic)
# dynamic (1900-2020) stages

start_yr = 2023
sim = 'abrupt'
dir = f'results/{sim}/'
var_keep = ['RF_CH4', 'D_CH4', 'RF_CO2', 'D_CO2', 'RF', 'D_Tg']
var_keep_perma = ['D_Apf_up', 'D_Apf_mi', 'D_Apf_or', 'D_Epf_up_CO2', 'D_Epf_up_CH4', 'D_Epf_mi_CO2', 'D_Epf_mi_CH4', 'D_Epf_or_CO2', 'D_Epf_or_CH4', 'D_Epf_CO2', 'D_Epf_CH4']
var_keep = var_keep + var_keep_perma
Par = xr.load_dataset('results/Par/Par.nc')
Par_dyn = xr.load_dataset('results/Par/Par_dyn.nc')

For_hist = xr.load_dataset('results/For/For_hist_fair.nc')
For_scen = xr.load_dataset('results/For/For_scen_fair.nc')

from core.OSCAR_perma_ab import OSCAR_perma_ab
OSCAR_perma_on = OSCAR_perma_ab(option='online')

## create initial values
Ini = xr.load_dataset('results/gradual/hist_fair.nc').sel(year=1900, drop=True)
for proc in OSCAR_perma_on._processes.values():
    if proc.Out in OSCAR_perma_on.var_prog and proc.Out not in Ini:
        Ini[proc.Out] = sum([xr.zeros_like(Par_dyn[dim], dtype=float) if dim in Par_dyn.coords else xr.zeros_like(Par[dim], dtype=float) for dim in proc.core_dims])

# create small batches
batch_size = 100
n_loops = len(Par.config) // batch_size

for i in range(n_loops):
    print(f'Running batch {i+1} of {n_loops}...')

    # generate configurations for the current batch
    batch_selected = np.arange(i * batch_size, (i + 1) * batch_size)
    hist_perma = OSCAR_perma_on(
        Ini=Ini.sel(config=batch_selected), Par=xr.merge([Par_dyn, Par]).sel(config=batch_selected), 
        For=For_hist.sel(year=slice(1900, None)), var_keep=var_keep, adapt_nt=True
    )
    hist_perma.to_netcdf(f'{dir}hist_batch{i}.nc')

    for mod in For_scen.mod.values:
        for scen in For_scen.scen.values:
            Out_scen = OSCAR_perma_on(
                Ini=hist_perma.sel(year=start_yr, drop=True), Par=xr.merge([Par_dyn, Par]).sel(config=batch_selected), 
                For=For_scen.sel(year=slice(start_yr, None), mod=mod, scen=scen, drop=True), 
                var_keep=var_keep, adapt_nt=True
            )
            Out_scen.to_netcdf(f'{dir}scen_batch{i}_{mod[:3]}_{scen[:3]}.nc')


# In[ ]:


# 2.2. gradual thaw

sim = 'gradual'
start_yr = 2023
dir = f'results/{sim}/'
var_keep = ['RF_CH4', 'D_CH4', 'RF_CO2', 'D_CO2', 'RF', 'D_Tg']
Par = xr.load_dataset('results/Par/Par.nc')

For_hist = xr.load_dataset('results/For/For_hist_fair.nc')
For_scen = xr.load_dataset('results/For/For_scen_fair.nc')

from core.mod_process import OSCAR

# create small batches
batch_size = 100
n_loops = len(Par.config) // batch_size

# Loop through the batches
for i in range(n_loops):
    print(f'Running batch {i+1} of {n_loops}...')

    # Generate configurations for the current batch
    batch_selected = np.arange(i * batch_size, (i + 1) * batch_size)
    hist = OSCAR(Ini=None, Par=Par.sel(config=batch_selected), For=For_hist, var_keep=var_keep, adapt_nt=True)
    hist.to_netcdf(f'{dir}hist_batch{i}.nc')

    for mod in For_scen.mod.values:
        for scen in For_scen.scen.values:
            Out_scen = OSCAR(
                Ini=hist.sel(year=start_yr, drop=True), Par=Par.sel(config=batch_selected), 
                For=For_scen.sel(year=slice(start_yr, None), mod=mod, scen=scen, drop=True), 
                var_keep=var_keep, adapt_nt=True
            )
            Out_scen.to_netcdf(f'{dir}scen_batch{i}_{mod[:3]}_{scen[:3]}.nc')


# In[ ]:


# 2.3. no permafrost thaw

sim = 'noperma'
start_yr = 2023
dir = f'results/{sim}/'
var_keep = ['RF_CH4', 'D_CH4', 'RF_CO2', 'D_CO2', 'RF', 'D_Tg']
Par = xr.load_dataset('results/Par/Par.nc')

For_hist = xr.load_dataset('results/For/For_hist_fair.nc')
For_scen = xr.load_dataset('results/For/For_scen_fair.nc')

from core.OSCAR_perma_ab import OSCAR_noperma
OSCAR_noperma = OSCAR_noperma()

# create small batches
batch_size = 100
n_loops = len(Par.config) // batch_size

# Loop through the batches
for i in range(n_loops):
    print(f'Running batch {i+1} of {n_loops}...')

    # Generate configurations for the current batch
    batch_selected = np.arange(i * batch_size, (i + 1) * batch_size)
    hist = OSCAR_noperma(Ini=None, Par=Par.sel(config=batch_selected), For=For_hist, var_keep=var_keep, adapt_nt=True)
    hist.to_netcdf(f'{dir}hist_batch{i}.nc')

    for mod in For_scen.mod.values:
        for scen in For_scen.scen.values:
            Out_scen = OSCAR_noperma(
                Ini=hist.sel(year=start_yr, drop=True), Par=Par.sel(config=batch_selected), 
                For=For_scen.sel(year=slice(start_yr, None), mod=mod, scen=scen, drop=True), 
                var_keep=var_keep, adapt_nt=True
            )
            Out_scen.to_netcdf(f'{dir}scen_batch{i}_{mod[:3]}_{scen[:3]}.nc')


# In[ ]:


# 2.4. abrupt (hybrid)
# static (1900-2000) + dynamic (2000-2020) stages: same setting as Truesky et al., 2020

sim = 'abrupt'
start_yr = 2023
var_keep = ['RF_CH4', 'D_CH4', 'RF_CO2', 'D_CO2', 'RF', 'D_Tg']
var_keep_perma = ['D_Apf_up', 'D_Apf_mi', 'D_Apf_or', 'D_Epf_up_CO2', 'D_Epf_up_CH4', 'D_Epf_mi_CO2', 'D_Epf_mi_CH4', 'D_Epf_or_CO2', 'D_Epf_or_CH4', 'D_Epf_CO2', 'D_Epf_CH4']
var_keep = var_keep + var_keep_perma
Par = xr.load_dataset('results/Par.nc')

## static stage
Par_sta = xr.load_dataset('results/abrupt/Par_sta.nc')
Par_dyn = xr.load_dataset('results/abrupt/Par_dyn.nc')

For_hist = xr.load_dataset('results/For_hist_fair.nc')
For_scen = xr.load_dataset('results/For_scen_fair.nc')

from core.OSCAR_perma_ab import OSCAR_perma_ab
OSCAR_perma_on = OSCAR_perma_ab(option='online')

## create initial values
Ini = xr.load_dataset('results/hist_fair.nc').sel(year=1900, drop=True)
for proc in OSCAR_perma_on._processes.values():
    if proc.Out in OSCAR_perma_on.var_prog and proc.Out not in Ini:
        Ini[proc.Out] = sum([
            xr.zeros_like(Par_sta[dim], dtype=float) if dim in Par_sta.coords else xr.zeros_like(Par[dim], dtype=float) for dim in proc.core_dims
        ])

# create small batches
batch_size = 100
n_loops = len(Par.config) // batch_size

for i in range(n_loops):
    print(f'Running batch {i+1} of {n_loops}...')

    # generate configurations for the current batch
    batch_selected = np.arange(i * batch_size, (i + 1) * batch_size)
    hist_perma_sta = OSCAR_perma_on(
        Ini=Ini.sel(config=batch_selected), Par=xr.merge([Par_sta, Par]).sel(config=batch_selected), 
        For=For_hist.sel(year=slice(1900, 2000)), var_keep=var_keep, adapt_nt=True
    )
    hist_perma_sta.to_netcdf(f'results/abrupt/hybrid/hist_perma_sta_batch{i}.nc')

    hist_perma_dyn = OSCAR_perma_on(
        Ini=hist_perma_sta.isel(year=-1, drop=True).sel(config=batch_selected), 
        Par=xr.merge([Par_dyn, Par]).sel(config=batch_selected), For=For_hist.sel(year=slice(2000, None)), 
        var_keep=var_keep, adapt_nt=True
    )
    hist_perma_dyn.to_netcdf(f'results/{sim}/hist_batch{i}.nc')

    for mod in For_scen.mod.values:
        for scen in For_scen.scen.values:
            Out_scen = OSCAR_perma_on(
                Ini=hist_perma_dyn.sel(year=start_yr, drop=True).sel(config=batch_selected), 
                Par=xr.merge([Par_dyn, Par]).sel(config=batch_selected), 
                For=For_scen.sel(year=slice(start_yr, None), mod=mod, scen=scen, drop=True), 
                var_keep=var_keep, adapt_nt=True
            )
            Out_scen.to_netcdf(f'results/abrupt/hybrid/scen_batch{i}_{mod[:3]}_{scen[:3]}.nc')


# ## 3. recursive run

# In[ ]:


## 3.1 normal simulation

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

dir = 'results/gradual/'
For = xr.load_dataset('./results/For/For_hist_fair.nc')
Par = xr.load_dataset('./results/Par/Par.nc')
Out = xr.load_dataset(f'{dir}hist_fair.nc')

from core.mod_process import OSCAR

for varname in ['RF_AERtot']:
    Out_ext = OSCAR[varname](Out, Par, For, recursive=True).to_dataset(name=varname)
    # Out_ext = Out_ext.sum('reg_pf', min_count=1, keep_attrs=True)
    Out_ext.to_netcdf(f'{dir}{varname}_hist.nc')


# In[3]:


## 3.2 batched: historical simulation

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

dir = 'results/gradual/'
## offset emissions
print('Offset emissions...')
For_hist = xr.load_dataset('results/For/For_hist_fair.nc')
For_scen = xr.load_dataset('results/For/For_scen_fair.nc')
vars = [var for var in For_hist.data_vars if var.startswith('E_')] + ['Eff']
for var in vars:
    For_hist[var] = For_hist[var] - For_hist[var].sel(year=1750, drop=True)
    For_scen[var] = For_scen[var] - For_hist[var].sel(year=1750, drop=True)

Par = xr.load_dataset('./results/Par/Par.nc')
Par_dyn = xr.load_dataset('./results/Par/Par_dyn.nc')
Par = xr.merge([Par_dyn, Par])

if 'gradual' in dir:
    from core.mod_process import OSCAR
if 'abrupt' in dir:
    from core.OSCAR_perma_ab import OSCAR_perma_ab
    OSCAR = OSCAR_perma_ab(option='online')
if 'noperma' in dir:
    from core.OSCAR_perma_ab import OSCAR_noperma
    OSCAR = OSCAR_noperma()

# create small batches
batch_size = 100
n_loops = len(Par.config) // batch_size

for varname in ['D_Epf_CO2', 'D_Epf_CH4']:
    var_list = []
    # loop through the batches
    for i in range(n_loops):
        print(f'Running batch {i+1} of {n_loops} for {varname}...')
        batch_selected = np.arange(i * batch_size, (i + 1) * batch_size)

        Out_hist = xr.load_dataset(f'{dir}hist_batch{i}.nc')
        Out_ext = OSCAR[varname](Out_hist, Par.sel(config=batch_selected), For_hist, recursive=True).to_dataset(name=varname)
        var_list.append(Out_ext)
    var = xr.concat(var_list, dim='config')
    if dir == 'results/gradual/':
        var = var.sum('reg_pf', min_count=1, keep_attrs=True)
    var.to_netcdf(f'{dir}{varname}_hist.nc')


# In[2]:


## 3.3 batched: future simulation

sim = 'gradual'
dir = f'results/{sim}/'

## offset emissions
print('Offset emissions...')
For_hist = xr.load_dataset('results/For/For_hist_fair.nc')
For_scen = xr.load_dataset('results/For/For_scen_fair.nc')
vars = [var for var in For_hist.data_vars if var.startswith('E_')] + ['Eff']
for var in vars:
    For_hist[var] = For_hist[var] - For_hist[var].sel(year=1750, drop=True)
    For_scen[var] = For_scen[var] - For_hist[var].sel(year=1750, drop=True)
    
Par = xr.load_dataset('./results/Par/Par.nc')

if sim == 'gradual':
    from core.mod_process import OSCAR
elif sim == 'abrupt':
    from core.OSCAR_perma_ab import OSCAR_perma_ab
    OSCAR = OSCAR_perma_ab(option='online')
    Par = xr.merge([xr.load_dataset('./results/Par/Par_steady.nc'), Par])
elif sim == 'noperma':
    from core.OSCAR_perma_ab import OSCAR_noperma
    OSCAR = OSCAR_noperma()

# create small batches
batch_size = 100
n_loops = len(Par.config) // batch_size

var_keep_perma = ['D_Epf_CO2', 'D_Epf_CH4']
for varname in var_keep_perma:
    var_list = []
    # loop through the batches
    for i in range(n_loops):
        print(f'Running batch {i+1} of {n_loops} for {varname}...')
        batch_selected = np.arange(i * batch_size, (i + 1) * batch_size)
        mod_list = []
        for mod in For_scen.mod.values:
            scen_list = []
            for scen in For_scen.scen.values:
                For_scen_batch = For_scen.sel(year=slice(2023, None), mod=mod, scen=scen, drop=True)
                Par_batch = Par.sel(config=batch_selected)
                if os.path.exists(f'{dir}scen_batch{i}_{mod[:3]}_{scen[:3]}.nc'):
                    Out_scen_batch = xr.load_dataset(f'{dir}scen_batch{i}_{mod[:3]}_{scen[:3]}.nc')
                else:
                    print(f'File {dir}scen_batch{i}_{mod[:3]}_{scen[:3]}.nc does not exist. Skipping...')
                    continue

                Out_ext = OSCAR[varname](Out_scen_batch, Par_batch, For_scen_batch, recursive=True).to_dataset(name=varname)
                
                scen_list.append(Out_ext.expand_dims({'scen': [scen]}))
            mod_list.append(xr.concat(scen_list, dim='scen').expand_dims({'mod': [mod]}))
        var_list.append(xr.concat(mod_list, dim='mod'))
    var = xr.concat(var_list, dim='config')
    if sim == 'gradual':
        var = var.sum('reg_pf', min_count=1, keep_attrs=True)
    var.to_netcdf(f'{dir}{varname}_scen.nc')
    print(f'Saved {dir}{varname}_scen.nc')


# ## 4. module run

# In[5]:


## 4.1 sea level rise

from core.OSCAR_SLR import OSCAR_SLR

start_yr = 2023
sim = 'noperma'
dir = f'results/{sim}/'
Par = xr.load_dataset(f'results/Par/Par_SLR_post.nc')

OSCAR_SLR = OSCAR_SLR(option='offline')

For_scen = xr.merge([xr.load_dataset(f'{dir}D_Tg_scen.nc'), xr.load_dataset(f'{dir}D_OHC_scen.nc')])
For_hist = xr.merge([xr.load_dataset(f'{dir}D_Tg_hist.nc'), xr.load_dataset(f'{dir}D_OHC_hist.nc')])
print(f'Historical time range: {For_hist["year"].min().values} - {For_hist["year"].max().values}')
if For_hist['year'].min().values > 1850 or For_hist['year'].max().values < 1900:
    raise ValueError('Historical forcing does not cover 1850-1900 period!')

##! baseline for damage functions is 1850-1900
D_Tg_base = For_hist['D_Tg'].sel(year=slice(1850, 1900)).mean(dim='year')
D_OHC_base = For_hist['D_OHC'].sel(year=slice(1850, 1900)).mean(dim='year')

For_hist['D_Tg'] = For_hist['D_Tg'] - D_Tg_base
For_scen['D_Tg'] = For_scen['D_Tg'] - D_Tg_base

For_hist['D_OHC'] = For_hist['D_OHC'] - D_OHC_base
For_scen['D_OHC'] = For_scen['D_OHC'] - D_OHC_base

var_keep = ['D_Htot']

Out_hist = OSCAR_SLR(Ini=None, For=For_hist, Par=Par, var_keep=var_keep)
for var in var_keep:
    Out_hist[var].to_netcdf(f'{dir}{var}_hist.nc')
Out_scen = OSCAR_SLR(Ini=Out_hist.sel(year=start_yr, drop=True), For=For_scen, Par=Par, var_keep=var_keep)
for var in var_keep:
    Out_scen[var].to_netcdf(f'{dir}{var}_scen.nc')
print(f'Finished SLR simulations and saved to {dir}')


# In[5]:


## 4.2 damage calculation

from core.OSCAR_dmg import OSCAR_dmg

start_yr = 2023
sim = 'noperma'
dir = f'results/{sim}/'
Par = xr.load_dataset('results/Par/Par_dmg.nc')

OSCAR_dmg = OSCAR_dmg()

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
D_Tg_base = For_hist.sel(year=slice(1986, 2005)).mean('year')
D_Htot_base = For_hist_slr.sel(year=slice(1986, 2005)).mean('year')

For_hist = For_hist - D_Tg_base
For_scen = For_scen - D_Tg_base
For_hist_slr = For_hist_slr - D_Htot_base
For_scen_slr = For_scen_slr - D_Htot_base

For_hist = xr.merge([For_hist, For_hist_slr])
For_scen = xr.merge([For_scen, For_scen_slr])

var_keep = ['dmg_T', 'dmg_SLR']
Out_hist = OSCAR_dmg(Ini=None, For=For_hist, Par=Par, var_keep=var_keep)
for var in Out_hist.data_vars:
    Out_hist[var].to_netcdf(dir + f'{var}_hist.nc')

Out_scen = OSCAR_dmg(Ini=Out_hist.sel(year=start_yr, drop=True), For=For_scen, Par=Par, var_keep=var_keep)
for var in Out_scen.data_vars:
    Out_scen[var].to_netcdf(dir + f'{var}_scen.nc')
    
print(f'Done! Results saved to {dir}')


# ## 5. prescribed LULCC emissions

# In[ ]:


## 5.1 prescribed LULCC emissions

dir = 'results/gradual/lulcc/'
Par = xr.load_dataset('results/Par/Par.nc')

from core.mod_process import OSCAR
print(OSCAR.var_in)

## offset emissions
print('Offset emissions...')
For_hist = xr.load_dataset('results/For/For_hist_fair.nc')
vars = [var for var in For_hist.data_vars if var.startswith('E_')] + ['Eff']
for var in vars:
    For_hist[var] = For_hist[var] - For_hist[var].sel(year=1750, drop=True)

For_hist = xr.merge([For_hist, xr.load_dataset('results/For/D_Eluc_hist_fair.nc')])

var_keep = ['RF_CH4', 'D_CH4', 'RF_CO2', 'D_CO2', 'RF', 'D_Tg', 'D_OHC', 'RF_cloud2', 'RF_AERtot', 'D_Eluc', 'D_Fland', 'D_Focean', 'D_N2O']

# create small batches
batch_size = 100
n_loops = len(Par.config) // batch_size

# loop through the batches
for i in range(n_loops):
    print(f'Running batch {i+1} of {n_loops}...')

    # generate configurations for the current batch
    batch_selected = np.arange(i * batch_size, (i + 1) * batch_size)
    hist = OSCAR(Ini=None, Par=Par.sel(config=batch_selected), For=For_hist, var_keep=var_keep, adapt_nt=True)
    hist.to_netcdf(f'{dir}hist_batch{i}.nc')


# In[ ]:


## 5.2 compare results

dir1 = 'results/gradual/offset/'
dir2 = 'results/gradual/lulcc/'

var1 = xr.load_dataarray(f'{dir1}D_Tg_hist.nc')
var2 = xr.load_dataarray(f'{dir2}D_Tg_hist.nc')

var1.mean(dim=['config', 'data_LULCC']).plot(x='year', label='offset')
var2.mean(dim=['config', 'data_LULCC']).plot(x='year', label='lulcc')

plt.legend()


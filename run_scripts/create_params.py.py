#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## 0. options
import os, sys
import numpy as np
import xarray as xr
import scipy.stats as st
from datetime import datetime
from pathlib import Path

code_path = Path.cwd().parent
sys.path.insert(0, code_path)
os.chdir(code_path)

nMC = 2400
seed = 1997
mod_region = 'RCP_5reg'
mod_noise = 0.1


# In[ ]:


## 1. OSCAR parameters

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

from core.fct_loadP import load_all_param
from core.fct_genMC import generate_config

#! parameters can be downloaded from https://github.com/tgasser/OSCAR/tree/master/input_data/parameters
#! downloaded parameters should be placed in the folder "input_data/parameters" under the code path

Par = load_all_param(mod_region=mod_region)
print(Par)


# In[4]:


## 2. dynamic abrupt permafrost parameters

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

from core.Par_perma_ab import load_permafrost_abrupt
Par_dyn = load_permafrost_abrupt(add_unc=True)
print(Par_dyn)


# In[5]:


## 3. SLR parameters

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

## load posterior parameters
Par_SLR = xr.load_dataset('input_data/parameters/Par_v1_SLR.nc')
random_indices = np.random.choice(Par_SLR.coords['config'], size=nMC, replace=True)
Par_SLR = Par_SLR.isel({'config': random_indices})
Par_SLR.coords['config'] = np.arange(len(Par_SLR.config))

## convert the unit of Lthx
Par_SLR['Lthx'] = Par_SLR['Lthx'] * 1E21 / ((3600*24*365.25) * 510.1E12)  # from mm m2 W-1 yr-1 to mm ZJ-1
Par_SLR['Lthx'].attrs['unit'] = 'mm ZJ-1'
print(Par_SLR)


# In[6]:


## 4. damage function parameters

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

from core.Par_dmg import load_dmg_all
Par_dmg = load_dmg_all(add_unc=True)

mini = Par_dmg['quantile'].min().values
maxi = Par_dmg['quantile'].max().values
rng = np.random.default_rng(seed=seed)
Uniform = st.uniform.rvs(size=nMC, random_state=rng) * (maxi - mini) + mini

## interpolate to MC samples
Par_mc = xr.Dataset()
Par_mc['f_dmg_SLR'] = Par_dmg['f_dmg_SLR'].interp(quantile=Uniform, method='linear')
Par_mc['f_dmg_T'] = Par_dmg['f_dmg_T'].interp(quantile=Uniform, method='linear')
Par_mc = Par_mc.rename({'quantile': 'config'})
Par_mc.coords['config'] = np.arange(nMC)
print(Par_mc)

Par_dmg = Par_dmg.drop_vars(['f_dmg_SLR', 'f_dmg_T', 'quantile'])
print(Par_dmg)


# In[ ]:


## 5. combine parameters

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

os.chdir(code_path)
Par_all = xr.merge([Par, Par_dyn, Par_dmg])
from core.fct_genMC import generate_config
Par_all = generate_config(Par_all, nMC=nMC, seed=seed, mod_noise=mod_noise)

Par_all = xr.merge([Par_all, Par_SLR, Par_mc])
print(Par_all)

# Par_all.to_netcdf('results/Par/Par.nc')

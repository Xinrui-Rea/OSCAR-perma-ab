#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## 1. load packages

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


# In[4]:


## 2. extract a single variable

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

xr.set_options(keep_attrs=True)

nMC = 2400
batch_size = 100
FROM_BATCH = True
HIST = False
SCEN = True

var_keep = ['D_Tg', 'D_CO2', 'D_CH4', 'D_N2O', 'D_OHC', 'RF_AERtot', 'RF_cloud2', 'D_Focean', 'D_Fland', 'D_Eluc']
var_keep_perma = ['D_Epf_up_CO2', 'D_Epf_up_CH4']

for dir in ['./results/abrupt/']:
    for varname in var_keep_perma:
        print(f'Concatenating {varname}')
        if not FROM_BATCH:
            if HIST:
                Out_hist = xr.open_dataset(f'{dir}hist_fair.nc')[varname]
                Out_hist.to_netcdf(f'{dir}{varname}_hist.nc')
            if SCEN:
                Out_scen = xr.open_dataset(f'{dir}scen_fair.nc')[varname]
                Out_scen.to_netcdf(f'{dir}{varname}_scen.nc')

        else:
            list_hist = []
            list_scen = []
            for batch_number in range(nMC // batch_size):
                if HIST:
                    # historical
                    var_hist = xr.load_dataset(f'{dir}hist_batch{batch_number}.nc')[varname]
                    list_hist.append(var_hist)

                if SCEN:
                    list_all_mods = []
                    for mod in mods:
                        list_all_scens = []
                        for scen in scens:
                            # scenario
                            fn = os.path.join(dir, f'scen_batch{batch_number}_{mod[:3]}_{scen[:3]}.nc')
                            var_scen = xr.load_dataset(fn)[varname]
                            list_all_scens.append(var_scen.expand_dims({'mod': [mod], 'scen': [scen]}))
                        list_all_mods.append(xr.concat(list_all_scens, dim='scen'))
                    list_scen.append(xr.concat(list_all_mods, dim='mod'))

            if HIST: 
                ds_hist = xr.concat(list_hist, dim='config')
                ds_hist.to_netcdf(f'{dir}{varname}_hist.nc')
            if SCEN:
                ds_scen = xr.concat(list_scen, dim='config')
                ds_scen.to_netcdf(f'{dir}{varname}_scen.nc')


# In[ ]:


## 3. concatenate varaibles

var_keep = ['D_Tg', 'D_CO2', 'D_OHC']

dir1 = './results/noperma/'
dir2 = './results/abrupt/'

for var in var_keep:
    print(f'Processing variable: {var}')
    hist1 = xr.load_dataset(f'{dir1}{var}_hist.nc')
    hist2 = xr.load_dataset(f'{dir2}{var}_hist.nc')

    hist = xr.merge([hist1.sel(year=slice(None, 1899)), hist2])
    hist.to_netcdf(f'{dir2}{var}_hist.nc')


# In[6]:


## 4. sum variables

dir = './results/abrupt/'
D_Epf_gr_CO2 = xr.load_dataarray(f'{dir}D_Epf_CO2_scen.nc') - xr.load_dataarray(f'{dir}D_Epf_ab_CO2_scen.nc')
D_Epf_gr_CO2.name = 'D_Epf_gr_CO2'
D_Epf_gr_CO2.to_netcdf(f'{dir}D_Epf_gr_CO2_scen.nc')
D_Epf_gr_CH4 = xr.load_dataarray(f'{dir}D_Epf_CH4_scen.nc') - xr.load_dataarray(f'{dir}D_Epf_ab_CH4_scen.nc')
D_Epf_gr_CH4.name = 'D_Epf_gr_CH4'
D_Epf_gr_CH4.to_netcdf(f'{dir}D_Epf_gr_CH4_scen.nc')


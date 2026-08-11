#!/usr/bin/env python
# coding: utf-8

# # 0. load packages and data

# In[ ]:


## 0.1 load packages

import os, io, re, sys
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from scipy.stats import gaussian_kde
from pathlib import Path

code_path = Path.cwd().parent
sys.path.insert(0, code_path)
os.chdir(code_path)

from run_scripts.utils_perma import *


# In[2]:


## 0.2 supporting functions

def check_variables_by_keyword(dataset: xr.Dataset, keyword: str):
    '''
    Checks which variables in an xarray Dataset contain a specific keyword
    (case-insensitive).

    Input:
    ------
    dataset (xr.Dataset)        The xarray Dataset to check.
    keyword (str)               The keyword to search for.

    output:
    -------
    A list of variable names that contain the keyword (case-insensitive).
    '''
    matching_variables = []
    keyword_lower = keyword.lower()
    for var in dataset.data_vars:
        if keyword_lower in var.lower():
            matching_variables.append(var)
    return matching_variables

def process_var(ds, var_pref, coords=None):
    ds_all = []
    for key, val in var_pref.items():
        if coords is None:
            try:
                ds_all.append(ds[key].rename(val))
            except KeyError:
                raise KeyError(f'{key} cannot be found in dataset')
        else:
            ds0 = []
            for dim, coord in coords.items():
                for coord_val in coord:
                    if key+coord_val in ds.data_vars:
                        try:
                            ds0.append(ds[key+coord_val].expand_dims(dim, -1).assign_coords({dim:[coord_val]}).assign_attrs(ds[key+coord_val].attrs).rename(val))
                        except KeyError:
                            raise KeyError(f'{key+coord_val} cannot be found in dataset')
            ds_all.append(xr.concat(ds0, dim=dim))
    return xr.merge(ds_all)

def align_future_emissions(hist_ds, future_ds,
                          interpolation_method='linear'):
    '''
    Align future emissions scenarios to historical data using year dimension.
    
    Input:
    ------
    hist_ds (xarray.dataarray)      Historical emissions data (1750-2022, annual) with 'year' dimension
    future_ds (xarray.dataarray)    Future emissions scenarios (2020-2100, 5-year) with 'year' dimension
    var_name (str)                  Name of the emissions variable

    Output:
    -------
    (xarray.dataarray)              Aligned future emissions with annual resolution

    Options:
    --------
    interpolation_method (str)      Interpolation method ('linear', 'cubic', 'nearest')
    '''
    
    # interpolate missing years
    hist_years = hist_ds.year.values 
    if len(hist_years) == max(hist_years) - min(hist_years) + 1:
        print('Historical data is already annual')
    else:
        print('Interpolating historical data to annual resolution...')
        annual_years = np.arange(hist_years.min(), hist_years.max() + 1)
        hist_interp = hist_ds.interp(year=annual_years, method=interpolation_method)
        hist_ds = hist_ds.copy()
        hist_ds = hist_interp
        hist_ds = hist_ds.assign_coords(year=annual_years)
        hist_years = hist_ds.year.values
           
    future_years = future_ds.year.values
    if len(future_years) == max(future_years) - min(future_years) + 1:
        print('Future data is already annual')
    else:
        print('Interpolating future data to annual resolution...')
        annual_years = np.arange(future_years.min(), future_years.max() + 1)
        future_interp = future_ds.interp(year=annual_years, method=interpolation_method)
        future_ds = future_ds.copy()
        future_ds = future_interp
        future_ds = future_ds.assign_coords(year=annual_years)
        future_years = future_ds.year.values
    
    # overlap -> scale future data
    if max(hist_years) >= min(future_years):
        print(f'Scaling using overlap period: {min(future_years)}-{max(hist_years)}')
        overlap_years = np.intersect1d(hist_years, future_years)
        
        hist_mean = hist_ds.sel(year=overlap_years).mean('year')
        future_mean = future_ds.sel(year=overlap_years).mean('year')
        
        scaling_factors = (hist_mean / future_mean).compute()
        future_scaled = future_ds.copy()
        future_scaled = future_ds * scaling_factors
        future_scaled = xr.concat([future_scaled.sel(year=[yr for yr in future_years if yr not in overlap_years]), hist_ds.sel(year=overlap_years)], dim='year')
        
    # no overlap -> interpolate and scale
    else:
        raise RuntimeError('No overlap between historical and future data')
    
    return future_scaled


# In[3]:


## 0.3 scenario emission drivers

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

## load excel data
fn = code_path / 'input_data' / 'ngfs' / f'IAM_data.xlsx'
df = pd.DataFrame(pd.read_excel(fn, header=0))

# Identify non-year and variable columns
non_year_columns = ['Model', 'Scenario', 'Region']
variable_column = 'Variable'
unit_column = 'Unit'
year_columns = [col for col in df.columns if col.isdigit()]

# check if required columns exist
required_columns = non_year_columns + [variable_column, unit_column] + year_columns
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Missing required columns in Excel file. Expected: {required_columns}, Found: {df.columns.tolist()}")

# create a dictionary mapping variables to their units
unit_mapping = df.set_index(variable_column)[unit_column].to_dict()

# set multi-index for non-year columns
try:
    df = df.set_index(non_year_columns + [variable_column])
except KeyError as e:
    raise KeyError(f"Could not set index with columns: {non_year_columns + [variable_column]}. Missing column: {e}")

# select only year columns
df_years = df[year_columns]

# stack the year columns to create a long format
df_stacked = df_years.stack(dropna=False)
df_stacked.index.names = df_stacked.index.names[:-1] + ['Year']

# convert the stacked Series to a DataFrame
df_stacked = df_stacked.rename('value').reset_index()

# pivot the DataFrame to have 'Variable' as separate variables in xarray
df_pivoted = df_stacked.pivot(
    index=non_year_columns + ['Year'],
    columns=variable_column,
    values='value'
).reset_index()

# convert the pivoted DataFrame to an xarray Dataset
try:
    ds = df_pivoted.set_index(non_year_columns + ['Year']).to_xarray()
except KeyError as e:
    raise KeyError(f"Could not set index for xarray with columns: {non_year_columns + ['Year']}. Missing column: {e}")

# assign units as attributes to the variables
for var_name, units in unit_mapping.items():
    if var_name in ds:
        ds[var_name].attrs['units'] = units

## change year into numeric
ds.coords['Year'] = ds.coords['Year'].astype(int)
ds = ds.sortby('Year')

ds_world = ds.sel(Region='World', drop=True)
ds_world = ds_world.drop_vars([var for var in ds_world.data_vars if ds[var].notnull().sum() == 0]) 
ds_world = ds_world.rename({'Year': 'year', 'Model':'mod', 'Scenario':'scen'})

## use only R5 regions
regs = {'Asia (R5)':1, 'Latin America (R5)':2, 'Middle East & Africa (R5)':3, 'OECD & EU (R5)':4, 'Reforming Economies (R5)':5}
ds = ds.sel(Region=list(regs.keys()), drop=True)
ds = ds.drop_vars([var for var in ds.data_vars if ds[var].notnull().sum() == 0]) 
new_region_values = [regs[reg] for reg in ds.Region.values]
ds = ds.assign_coords(Region=new_region_values)
ds = ds.rename({'Year': 'year', 'Model':'mod', 'Scenario':'scen', 'Region': 'reg_mask'})
print(ds)


# In[4]:


## 0.4 downscaled level variables in NGFS dataset

print('+'*50)
current_datetime = datetime.now()
print(f'Current date and time: {current_datetime}')
print('+'*50, '\n')

## load excel data
ds_list = []
for mod in mods:
    fn = code_path / 'input_data' / 'ngfs' / f'Downscaled_{mod}_data.xlsx'
    df = pd.DataFrame(pd.read_excel(fn, header=0))
    df = df.dropna(axis=1, how='all')
    df = df[df['Model'] == f'Downscaling[{mod}]']

    # Identify non-year and variable columns
    non_year_columns = ['Model', 'Scenario', 'Region']
    variable_column = 'Variable'
    unit_column = 'Unit'
    year_columns = [col for col in df.columns if col.isdigit()]

    # check if required columns exist
    required_columns = non_year_columns + [variable_column, unit_column] + year_columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in Excel file. Expected: {required_columns}, Found: {df.columns.tolist()}")

    # create a dictionary mapping variables to their units
    unit_mapping = df.set_index(variable_column)[unit_column].to_dict()

    # set multi-index for non-year columns
    try:
        df = df.set_index(non_year_columns + [variable_column])
    except KeyError as e:
        raise KeyError(f"Could not set index with columns: {non_year_columns + [variable_column]}. Missing column: {e}")

    # select only year columns
    df_years = df[year_columns]

    # stack the year columns to create a long format
    df_stacked = df_years.stack(dropna=False)
    df_stacked.index.names = df_stacked.index.names[:-1] + ['Year']

    # convert the stacked Series to a DataFrame
    df_stacked = df_stacked.rename('value').reset_index()
    
    # pivot the DataFrame to have 'Variable' as separate variables in xarray
    df_pivoted = df_stacked.pivot(
        index=non_year_columns + ['Year'],
        columns=variable_column,
        values='value'
    ).reset_index()

    # convert the pivoted DataFrame to an xarray Dataset
    try:
        ds = df_pivoted.set_index(non_year_columns + ['Year']).to_xarray()
    except KeyError as e:
        raise KeyError(f"Could not set index for xarray with columns: {non_year_columns + ['Year']}. Missing column: {e}")

    # assign units as attributes to the variables
    for var_name, units in unit_mapping.items():
        if var_name in ds:
            ds[var_name].attrs['units'] = units

    ## change year into numeric
    ds.coords['Year'] = ds.coords['Year'].astype(int)
    ds = ds.sortby('Year')
    ds.coords['Model'] = [mod]

    ds_list.append(ds.rename({'Year': 'year', 'Scenario':'scen', 'Region':'region', 'Model':'mod'}))

ds_national = xr.merge(ds_list)
print(ds_national)


# In[5]:


## 0.5 check variables by keyword

keyword = 'GDP|PPP|Counterfactual without damage'

REG_FLAG = 'global'

if REG_FLAG == 'global':
    print(f'Global level variables in NGFS dataset:')
    list_global = check_variables_by_keyword(ds_world, keyword)
    print(f'Found {len(list_global)} variables containing the keyword "{keyword}":')
    for item in list_global:
        print(item)

elif REG_FLAG == 'regional':
    print(f'\nRegional level variables in NGFS dataset:')
    list_regional = check_variables_by_keyword(ds, keyword)
    print(f'Found {len(list_regional)} variables containing the keyword "{keyword}":')
    for item in list_regional:
        print(item)

elif REG_FLAG == 'national':
    list_national = check_variables_by_keyword(ds_national, keyword)
    print(f'\nNational level variables in NGFS dataset:')
    print(f'Found {len(list_national)} variables containing the keyword "{keyword}":')
    for item in list_national:
        print(item)


# In[ ]:


## 0.6 for percentiles

varname = 'RF_CO2'
sel_dict = {'mod': mods, 'scen': scens}
ds_list = []
var_list = list_global
print(f'{varname}: {var_list}')
for var in var_list:
    y = float(var.split('|')[-1].split('th')[0])
    try:
        var = ds_world[var].sel(**sel_dict).expand_dims({'percentile': [y]})
        var.name = varname
        ds_list.append(var)
    except KeyError:
        continue
ds_ngfs = xr.concat(ds_list, dim='percentile')
ds_ngfs = ds_ngfs.sortby('percentile')
ds_ngfs.to_netcdf(os.path.join('results', 'ngfs', f'{varname}_ngfs.nc'))


# In[5]:


## 0.7 sum up emissions

var_list = ['Emissions|CO2|Energy', 'Emissions|CO2|Industrial Processes']
var = ds_national[var_list[0]] + ds_national[var_list[1]]
var = var.dropna('year', how='all')
var.name = 'Eff'
var.to_netcdf(os.path.join('results', 'ngfs', 'Eff_national_ngfs.nc'))


# In[ ]:


## 0.8 output emissions

varname = 'D_Eluc'
sel_dict = {'mod': mods, 'scen': scens}
print(f'{varname}: {list_global}')
var = list_global[0]
try:
    var = ds_world[var].sel(**sel_dict)
    # from MtCO2/yr to PgC/yr
    scaling_factor = molecular_weight['C'] / molecular_weight['CO2']
    var = var * scaling_factor / 1000 
    var.name = varname
except KeyError:
    print(f'{var} not found')
    pass
var.to_netcdf(os.path.join('results', 'ngfs', f'{varname}_ngfs.nc'))


# In[10]:


da = ds_world[list_global[0]].sel(mod=mods)
da.attrs = {'long_name': da.name, 'units': ds_world[list_global[0]].attrs['units']}
da.name = 'GDP'
da.to_netcdf(os.path.join('results', 'ngfs', f'GDP_ngfs.nc'))


# ## 1. check data

# In[4]:


## check downscaled emissions

fn = os.path.join('results', 'ngfs', 'Eff_national_ngfs.nc')
eff = xr.open_dataarray(fn)

print(eff.drop_sel(region=['EU27']).sel(year=2050, scen='Below 2°C', mod='GCAM 6.0 NGFS').sum('region'))


# ## 2. plot

# In[ ]:


## 2.1 distribution of global mean temperature

varname = 'D_Tg'
ylabel = r'ΔGMST (°C)'
sel_dict = {'scen': [scens_sorted[id] for id in range(7)], 'mod': mods, 'year': [2025, 2050, 2075, 2100]}
if varname in ['dmg_SLR', 'dmg_T', 'dmg_tot']: 
    base_year = (1986, 2005)
elif varname in ['D_Tg']:
    base_year = (1850, 1900)
elif varname in ['RF_CO2', 'D_CO2', 'D_N2O', 'D_CH4']:
    base_year = (1750, 1750)

dir1 = './results/noperma/constrained/'
dir2 = './results/gradual/constrained/'
dir3 = './results/abrupt/constrained/'

var_scen1 = xr.load_dataarray(f'{dir1}{varname}_scen.nc')
sel_dict = {k: v for k, v in sel_dict.items() if k in var_scen1.dims}
for attr in ['unit', 'units']:
    if attr in var_scen1.attrs:
        unit = f' ({var_scen1.attrs[attr]})'
        break
else:
    print(f'No unit attribute found in {varname}_scen.nc')
    unit = ''

var_scen1 = var_scen1.sel(**sel_dict, drop=True)
var_scen1_base = xr.load_dataarray(f'{dir1}{varname}_hist.nc')
base_dict = {k:v for k, v in sel_dict.items() if k in var_scen1_base.dims}
base_dict['year'] = slice(*base_year)
var_scen1_base = var_scen1_base.sel(**base_dict).mean('year').squeeze()
var1 = var_scen1 - var_scen1_base
# var_scen2 = xr.load_dataarray(f'{dir2}{varname}_scen.nc').sel(**sel_dict, drop=True)
# var_scen2_base = xr.load_dataarray(f'{dir2}{varname}_hist.nc').sel(**base_dict).mean('year').squeeze()
# var2 = var_scen2 - var_scen2_base
var_scen3 = xr.load_dataarray(f'{dir3}{varname}_scen.nc').sel(**sel_dict, drop=True)
var_scen3_base = xr.load_dataarray(f'{dir3}{varname}_hist.nc').sel(**base_dict).mean('year').squeeze()
var3 = var_scen3 - var_scen3_base

ds_ngfs = xr.load_dataarray(os.path.join('results', 'ngfs', f'{varname}_ngfs.nc'))

fig, axes = plt.subplots(2, 2, figsize=(6, 4), **{'sharex': True, 'sharey': True}, dpi=300)
for year in sel_dict['year']:
    ax = axes.flatten()[sel_dict['year'].index(year)]
    ax.set_xlim(0, 5)
    for scen in sel_dict['scen']:
        sns.lineplot(
            x=var3.sel(mod=mods, scen=scen, year=year).mean('mod').quantile(np.arange(0,101)/100, dim=['config', 'data_LULCC']).values, 
            y=np.arange(0,101),
            color=scen_colors[scen], alpha=0.7, ax=ax
            )
        ax.plot(
            ds_ngfs.sel(scen=scen, mod=mods, year=year).mean('mod').values, ds_ngfs['percentile'].values, 
            marker='o', ls='none', color=scen_colors[scen], ms=4, alpha=0.7
        )
    ax.set_title(f'{year}', fontsize='medium')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.grid(ls='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='major', direction='in')

legend_handler = [Line2D([0], [0], color=scen_colors[scen], label=standardize_scen_names([scen])[0]) for scen in sel_dict['scen'][::-1]]
fig.supxlabel(ylabel, y=0.02, fontsize='medium')
fig.supylabel('Percentile', fontsize='medium')
fig.legend(
    handles=legend_handler, bbox_to_anchor=(0.5, 0.03), ncol=4, loc='upper center', 
    prop={'size': 'small', 'style': 'italic'}, frameon=False
)
plt.subplots_adjust(bottom=0.1, left=0.12, right=0.95, hspace=0.2, wspace=0.1)
plt.show()


# In[ ]:


## 2.2 distribution of global mean temperature

varname = 'RF_CO2'
sel_dict = {'scen': [scens[id] for id in range(1,2)], 'mod': mods}
if varname in ['dmg_SLR', 'dmg_T', 'dmg_tot']: 
    base_year = (1986, 2005)
elif varname in ['D_Tg']:
    base_year = (1850, 1900)
elif varname in ['RF_CO2', 'D_CO2', 'D_N2O', 'D_CH4']:
    base_year = (1750, 1750)

dir2 = './results/gradual/constrained/'
var_scen2 = xr.load_dataarray(f'{dir2}{varname}_scen.nc').sel(**sel_dict, drop=True)
sel_dict = {k: v for k, v in sel_dict.items() if k in var_scen2.dims}
for attr in ['unit', 'units']:
    if attr in var_scen2.attrs:
        unit = f' ({var_scen2.attrs[attr]})'
        break
else:
    print(f'No unit attribute found in {varname}_scen.nc')
    unit = ''


ds_ngfs = xr.load_dataarray(os.path.join('results', 'ngfs', f'{varname}_ngfs.nc'))

fig, ax = plt.subplots(figsize=(8, 6))
for scen in sel_dict['scen']:
    sns.lineplot(
        x=var_scen2.year.values, 
        y=var_scen2.sel(mod=mods, scen=scen).mean(['data_LULCC', 'mod']).median(dim='config').values,
        color=scen_colors[scen], alpha=0.7
        )
    ax.plot(
        ds_ngfs.year.values, ds_ngfs.sel(scen=scen, mod=mods).mean('mod').squeeze().values, 
        marker='o', ls=':', color=scen_colors[scen], ms=4, alpha=0.7
    )
ax.set_title(f'{varname} median', fontsize='medium')
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
ax.tick_params(axis='both', which='major', labelsize='small')
ax.tick_params(axis='x', which='major', direction='in')

legend_handler = [Line2D([0], [0], color='none', label='Scenario')] + \
    [Line2D([0], [0], color=scen_colors[scen], label=scen if scen != scens[5] else 'NDCs') for scen in sel_dict['scen']] + \
    [Line2D([0], [0], color='k', ls=ls, label=lb) for ls, lb in [('-', 'OSCAR'), (':', 'NGFS')]]
fig.supxlabel('Year', y=0.03, fontsize='medium')
fig.supylabel(f'{varname}{unit}', fontsize='medium')
fig.legend(handles=legend_handler, bbox_to_anchor=(0.1, 0.85), ncol=2, loc='upper left', fontsize='medium')
plt.subplots_adjust(bottom=0.1, left=0.08, right=0.95, hspace=0.2, wspace=0.1)
plt.show()

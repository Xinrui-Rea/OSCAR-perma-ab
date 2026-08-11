#!/usr/bin/env python
# coding: utf-8

# # 0. Basic functions

# In[ ]:


## 0.1 load packages

import os, sys, csv, warnings
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from datetime import datetime
from scipy.stats import skew, kurtosis
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import ticker
from pathlib import Path

code_path = Path.cwd().parent
sys.path.insert(0, code_path)
os.chdir(code_path)

from run_scripts.utils_perma import *

map_mask = xr.load_dataset('./input_data/regions/region_mask_0p5deg.nc')['frac_reg']



map_mask = xr.load_dataset('../CROP/input_data/regions/region_mask_0p5deg.nc')['frac_reg']


# # Paper figure

# In[12]:


## 1.1 timeseries of all relevant variables
## add anthropogenic emissions
## quantile(PCF - no PCF)

varname1 = 'D_Tg'
varname2 = 'D_Htot'
varname3 = 'dmg_tot'
sel_dict = {'reg_IMAGE': 'World', 'adaptation': True}

dir1 = './results/noperma/constrained/'
dir2 = './results/abrupt/constrained/'
var_list = []
for dir in [dir1, dir2]:
    for varname in [varname1, varname2, varname3]:
        var = xr.load_dataarray(f'{dir}{varname}_scen.nc')
        sel_dict_var = {k: v for k, v in sel_dict.items() if k in var.dims}
        var = var.sel(**sel_dict_var, drop=True)
        var_base = xr.load_dataarray(f'{dir}{varname}_hist.nc')
        base_dict = {k:v for k, v in sel_dict_var.items() if k in var_base.dims}
        base_year = get_baseline_year(varname)
        base_dict['year'] = slice(*base_year)
        var_base = var_base.sel(**base_dict).mean('year')
        var = (var - var_base).sel(scen=scens_sorted).squeeze()
        var_list.append(var.expand_dims({'sim': [f'noperma' if dir == dir1 else 'abrupt']}).assign_coords(sim=[f'noperma' if dir == dir1 else 'abrupt']))

        for attr in ['unit', 'units']:
            if attr in var.attrs:
                unit = f' ({var.attrs[attr]})'
                break
        else:
            print(f'No unit attribute found in {varname}_scen.nc')
            unit = ''
    
ds = xr.merge(var_list)

fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True, dpi=300)

label_x = -0.2
label_y = 1.05

sel_years = [2050, 2100]

qt = 0.5
ax = axes[0, 0]
ax.text(label_x, label_y, 'a', transform=ax.transAxes, fontsize='large', fontweight='bold')
ds_var = xr.load_dataset('results/For/For_scen_fair.nc')['Eff'].sel(year=slice(2023, 2100))
for scen in scens_sorted:
    ds_scen = ds_var.sel(scen=scen).sum(['reg_land'], min_count=1).cumsum('year')
    quantile = ds_scen.quantile(qt, dim=['mod'])
    ax.plot(quantile['year'], quantile.values, color=scen_colors[scen])
    ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(ls='--', alpha=0.5)
    ax.axvline(x=2050, color='k', ls='--', lw=1)
    ax.axvline(x=2100, color='k', ls='--', lw=1)
    ax.set_ylabel('Anthropogenic\nCO$_2$ (PgC)', fontsize='large')
    for sel_year in sel_years:
        print(f'{sel_year}: {quantile.sel(year=sel_year).values:<4.2f}', end='\t')
    print(f'Cumulative CO₂ emission|{standardize_scen_names([scen])[0]}|baseline')

ax = axes[1, 0]
ax.text(label_x, label_y, 'e', transform=ax.transAxes, fontsize='large', fontweight='bold')
# ds_var = (xr.load_dataarray(dir2 + 'D_Epf_CO2_scen.nc') + xr.load_dataarray(dir2 + 'D_Epf_CH4_scen.nc') / 1.0e3).cumsum('year')
ds_var = xr.load_dataarray(dir2 + 'D_Epf_CO2_scen.nc').cumsum('year')
for scen in scens_sorted:
    ds_scen = ds_var.sel(scen=scen)
    quantile = ds_scen.quantile(qt, dim=['config', 'data_LULCC', 'mod'])
    ax.plot(quantile['year'], quantile.values, color=scen_colors[scen])
    for sel_year in sel_years:
        print(f'{sel_year}: {quantile.sel(year=sel_year).values:<4.2f}', end='\t')
    print(f'Cumulative CO₂ emission|{standardize_scen_names([scen])[0]}|permafrost')
ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.grid(ls='--', alpha=0.5)
ax.axvline(x=2050, color='k', ls='--', lw=1)
ax.axvline(x=2100, color='k', ls='--', lw=1)
ax.set_ylabel('Permafrost\nCO$_2$ (PgC)', fontsize='large')

var_ngfs = xr.load_dataarray('results/ngfs/GDP_ngfs.nc')

for i, sim in enumerate(['abrupt', 'noperma']):
    for j, varname in enumerate([varname1, varname2, varname3]):
        ax = axes[i, j + 1]
        ax.text(label_x, label_y, chr(98 + j if i == 0 else 102 + j), transform=ax.transAxes, fontsize='large', fontweight='bold')
        ax.set_xlim(2020, 2105)
        ds_var = ds[varname].sel(sim=sim)
        if sim =='noperma':
            ds_var = ds[varname].sel(sim='abrupt') - ds_var
        for scen in scens_sorted:
            ds_scen = ds_var.sel(scen=scen)
            quantile = ds_scen.quantile(qt, dim=['config', 'data_LULCC', 'mod'])
            ax.plot(quantile['year'], quantile.values, color=scen_colors[scen])
            ax.tick_params(axis='both', which='major', direction='in', labelsize='medium', pad=5)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            fmt = '.3f' if varname != varname2 else '.0f'
            for sel_year in sel_years:
                if varname == varname3:
                    gdp = var_ngfs.sel(year=sel_year, scen=scen, drop=True).mean('mod').values
                    print(f'{sel_year}: {quantile.sel(year=sel_year).values:{fmt}}% × {gdp*1e-3:.2f}={quantile.sel(year=sel_year).values*gdp*1e-5:.2f}', end='\t')
                else:
                    print(f'{sel_year}: {quantile.sel(year=sel_year).values:{fmt}}', end='\t')
            print(f'{sim if sim == "abrupt" else "baseline"}|{standardize_scen_names([scen])[0]}|{varname}')

        ## add zoom-inset
        if varname in [varname1, varname2] and i == 1:
            axins = ax.inset_axes([0.25, 0.35, 0.3, 0.5])
            for scen in scens_sorted:
                ds_scen = ds_var.sel(scen=scen)
                quantile = ds_scen.quantile(qt, dim=['config', 'data_LULCC', 'mod'])
                axins.plot(quantile['year'], quantile.values, color=scen_colors[scen])
            x1, x2 = 2095, 2100
            y1, y2 = None, None
            axins.set_xlim(x1, x2)
            if varname == varname1:
                y1, y2 = 0.07, 0.08
            elif varname == varname2:
                y1, y2 = 12, 13.2
            axins.set_ylim(y1, y2)
            axins.tick_params(axis='both', which='major', direction='in', labelsize='medium')
            axins.tick_params(axis='x', which='major', pad=5)
            axins.yaxis.set_major_locator(plt.FixedLocator([float(f'{y1:.2g}'), float(f'{y2:.2g}')]))
            axins.yaxis.set_major_formatter(formatter)
            axins.grid(ls='--', alpha=0.5)
            ax.indicate_inset_zoom(axins, edgecolor='black')

        if varname == varname1:
            ax.set_ylabel(r'ΔGMST (°C)', fontsize='large')
        elif varname == varname2:
            ax.set_ylabel(f'SLR (mm)', fontsize='large')
        elif varname == varname3:
            ax.set_ylabel(f'DC (% of GDP)', fontsize='large')

        ax.tick_params(axis='both', which='major', direction='in', labelsize='large')
        ax.xaxis.set_major_locator(plt.FixedLocator([2025, 2050, 2075, 2100]))
        ax.grid(ls='--', alpha=0.5)
        ax.axvline(x=2050, color='k', ls='--', lw=1)
        ax.axvline(x=2100, color='k', ls='--', lw=1)
        
legend_handles = [Line2D([0], [0], color=scen_colors[scen], lw=2) for scen in scens_sorted]
legend_labels = standardize_scen_names(scens_sorted)
fig.legend(
    handles=legend_handles[::-1], labels=legend_labels[::-1],
    prop={'size': 'large', 'style': 'italic'}, 
    frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01)
)
fig.supxlabel('Year', y=0.13, fontsize='x-large')
for i in range(4):
    fig.align_ylabels(axes[:, i])
plt.subplots_adjust(bottom=0.24, top=0.95, left=0.05, right=0.98, hspace=0.2, wspace=0.45)
plt.show()


# In[ ]:


## 1.2 timeseries of all relevant variables
## add anthropogenic emissions
## quantile(PCF) - quantile(no PCF)

varname1 = 'D_Tg'
varname2 = 'D_Htot'
varname3 = 'dmg_tot'
sel_dict = {'reg_IMAGE': 'World', 'adaptation': True}

dir1 = './results/noperma/constrained/'
dir2 = './results/abrupt/constrained/'
var_list = []
for dir in [dir1, dir2]:
    for varname in [varname1, varname2, varname3]:
        var = xr.load_dataarray(f'{dir}{varname}_scen.nc')
        sel_dict_var = {k: v for k, v in sel_dict.items() if k in var.dims}
        var = var.sel(**sel_dict_var, drop=True)
        var_base = xr.load_dataarray(f'{dir}{varname}_hist.nc')
        base_dict = {k:v for k, v in sel_dict_var.items() if k in var_base.dims}
        base_year = get_baseline_year(varname)
        base_dict['year'] = slice(*base_year)
        var_base = var_base.sel(**base_dict).mean('year')
        var = (var - var_base).sel(scen=scens_sorted).squeeze()
        var_list.append(var.expand_dims({'sim': [f'noperma' if dir == dir1 else 'abrupt']}).assign_coords(sim=[f'noperma' if dir == dir1 else 'abrupt']))

        for attr in ['unit', 'units']:
            if attr in var.attrs:
                unit = f' ({var.attrs[attr]})'
                break
        else:
            print(f'No unit attribute found in {varname}_scen.nc')
            unit = ''
    
ds = xr.merge(var_list)

fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True)

label_x = -0.2
label_y = 1.05
qt = 0.5

ax = axes[0, 0]
ax.text(label_x, label_y, 'a', transform=ax.transAxes, fontsize='large', fontweight='bold')
ds_var = xr.load_dataset('results/For/For_scen_fair.nc')['Eff'].sel(year=slice(2023, 2100))
for scen in scens_sorted:
    ds_scen = ds_var.sel(scen=scen).sum(['reg_land'], min_count=1).cumsum('year')
    quantile = ds_scen.quantile(qt, dim=['mod'])
    ax.plot(quantile['year'], quantile.values, color=scen_colors[scen])
    ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(ls='--', alpha=0.5)
    ax.set_ylabel('Anthropogenic\nCO$_2$ (PgC)', fontsize='large')
    print(f'{quantile.sel(year=2100).values:<4.2f}\tCO₂ emission|{standardize_scen_names([scen])[0]}|baseline')

ax = axes[1, 0]
ax.text(label_x, label_y, 'e', transform=ax.transAxes, fontsize='large', fontweight='bold')
ds_var = (xr.load_dataarray(dir2 + 'D_Epf_CO2_scen.nc') + xr.load_dataarray(dir2 + 'D_Epf_CH4_scen.nc') / 1.0e3).cumsum('year')
for scen in scens_sorted:
    ds_scen = ds_var.sel(scen=scen)
    quantile = ds_scen.quantile(qt, dim=['config', 'data_LULCC', 'mod'])
    ax.plot(quantile['year'], quantile.values, color=scen_colors[scen])
    print(f'{quantile.sel(year=2100).values:<4.2f}\tcarbon emission|{standardize_scen_names([scen])[0]}|permafrost')
ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.grid(ls='--', alpha=0.5)
ax.set_ylabel('Permafrost\ncarbon (PgC)', fontsize='large')

for i, sim in enumerate(['abrupt', 'noperma']):
    for j, varname in enumerate([varname1, varname2, varname3]):
        ax = axes[i, j + 1]
        ax.text(label_x, label_y, chr(98 + j if i == 0 else 102 + j), transform=ax.transAxes, fontsize='large', fontweight='bold')
        ax.set_xlim(2020, 2105)
        ds_var = ds[varname].sel(sim=sim).quantile(qt, dim=['config', 'data_LULCC', 'mod'])
        if sim =='noperma':
            ds_var -= ds[varname].sel(sim='abrupt').quantile(qt, dim=['config', 'data_LULCC', 'mod'])
            ds_var = -ds_var
        for scen in scens_sorted:
            ds_scen = ds_var.sel(scen=scen)
            ax.plot(ds_scen['year'], ds_scen.values, color=scen_colors[scen])
            ax.tick_params(axis='both', which='major', direction='in', labelsize='medium', pad=5)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            fmt = '.2f' if varname != varname2 else '.0f'
            print(f'{ds_scen.sel(year=2100).values:{fmt}}\t{varname}|{standardize_scen_names([scen])[0]}|{sim}')

        ## add zoom-inset
        if varname in [varname1, varname2] and i == 1:
            axins = ax.inset_axes([0.25, 0.35, 0.3, 0.5])
            for scen in scens_sorted:
                ds_scen = ds_var.sel(scen=scen)
                axins.plot(ds_scen['year'], ds_scen.values, color=scen_colors[scen])
            x1, x2 = 2095, 2100
            y1, y2 = None, None
            axins.set_xlim(x1, x2)
            if varname == varname1:
                y1, y2 = 0.07, 0.09
            elif varname == varname2:
                y1, y2 = 14, 16.5
            axins.set_ylim(y1, y2)
            axins.tick_params(axis='both', which='major', direction='in', labelsize='medium')
            axins.tick_params(axis='x', which='major', pad=5)
            axins.yaxis.set_major_locator(plt.FixedLocator([float(f'{y1:.2g}'), float(f'{y2:.2g}')]))
            axins.yaxis.set_major_formatter(formatter)
            axins.grid(ls='--', alpha=0.5)
            ax.indicate_inset_zoom(axins, edgecolor='black')

        if varname == varname1:
            ax.set_ylabel(r'ΔGMST (°C)', fontsize='large')
        elif varname == varname2:
            ax.set_ylabel(f'SLR (mm)', fontsize='large')
        elif varname == varname3:
            ax.set_ylabel(f'DC (% of GDP)', fontsize='large')

        ax.tick_params(axis='both', which='major', direction='in', labelsize='large')
        ax.xaxis.set_major_locator(plt.FixedLocator([2025, 2050, 2075, 2100]))
        ax.grid(ls='--', alpha=0.5)
        
legend_handles = [Line2D([0], [0], color=scen_colors[scen], lw=2) for scen in scens_sorted]
legend_labels = standardize_scen_names(scens_sorted)
fig.legend(
    handles=legend_handles[::-1], labels=legend_labels[::-1],
    prop={'size': 'large', 'style': 'italic'}, 
    frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01)
)
fig.supxlabel('Year', y=0.13, fontsize='x-large')
for i in range(4):
    fig.align_ylabels(axes[:, i])
plt.subplots_adjust(bottom=0.22, top=0.95, left=0.05, right=0.98, hspace=0.2, wspace=0.45)
plt.show()
            

# In[18]:


## 1.3 timeseries of relative changes

varname1 = 'D_Tg'
varname2 = 'D_Htot'
varname3 = 'dmg_tot'
sel_dict = {'reg_IMAGE': 'World', 'adaptation': True}

dir1 = './results/noperma/constrained/'
dir2 = './results/abrupt/constrained/'
var_list = []
for dir in [dir1, dir2]:
    for varname in [varname1, varname2, varname3]:
        var = xr.load_dataarray(f'{dir}{varname}_scen.nc')
        sel_dict_var = {k: v for k, v in sel_dict.items() if k in var.dims}
        var = var.sel(**sel_dict_var, drop=True)
        var_base = xr.load_dataarray(f'{dir}{varname}_hist.nc')
        base_dict = {k:v for k, v in sel_dict_var.items() if k in var_base.dims}
        base_year = get_baseline_year(varname)
        base_dict['year'] = slice(*base_year)
        var_base = var_base.sel(**base_dict).mean('year')
        var = (var - var_base).sel(scen=scens_sorted).squeeze()
        var_list.append(var.expand_dims({'sim': [f'noperma' if dir == dir1 else 'abrupt']}).assign_coords(sim=[f'noperma' if dir == dir1 else 'abrupt']))

        for attr in ['unit', 'units']:
            if attr in var.attrs:
                unit = f' ({var.attrs[attr]})'
                break
        else:
            print(f'No unit attribute found in {varname}_scen.nc')
            unit = ''
    
ds = xr.merge(var_list)

fig, axes = plt.subplots(1, 4, figsize=(8, 2.5), sharex=True, dpi=300)
ax = axes[0]
ax.text(-0.05, 1.02, 'a', transform=ax.transAxes, fontsize='large', fontweight='bold')
ds_var_base = xr.load_dataset('./results/For/For_scen_fair.nc')['Eff'].sel(year=slice(2023, 2100))
ds_var_add = xr.load_dataarray(dir2 + 'D_Epf_CO2_scen.nc').sel(year=slice(2023, 2100)).cumsum('year')
sel_years = [2050, 2100]
print('Permafrost contributions to cumulative CO₂ emission')
for scen in scens_sorted:
    ds_scen_base = ds_var_base.sel(scen=scen).sum(['reg_land'], min_count=1).cumsum('year')
    ds_scen_add = ds_var_add.sel(scen=scen)
    frac = ds_scen_add.median(dim=['config', 'data_LULCC', 'mod']) / ds_scen_base.median(dim=['mod']) * 100
    ax.plot(frac['year'], frac, color=scen_colors[scen])
    for sel_year in sel_years:
        print(f'{sel_year}: {frac.sel(year=sel_year).values:.2f}%', end='\t')
    print(f'{standardize_scen_names([scen])[0]}')
ax.grid(ls='--', alpha=0.5)
ax.axvline(x=2050, color='k', ls='--', lw=1)
ax.axvline(x=2100, color='k', ls='--', lw=1)
ax.set_ylabel('Permafrost contributions to' + '\n'  + r'cumulative CO$_2$ (%)', fontsize='medium')
ax.tick_params(axis='both', which='major', direction='in', labelsize='small')

for j, varname in enumerate([varname1, varname2, varname3]):
    print(f'Permafrost contribution to {varname}')
    ax = axes[j + 1]
    ax.text(-0.05, 1.02, chr(98 + j), transform=ax.transAxes, fontsize='large', fontweight='bold')
    ax.set_xlim(2020, 2105)
    ds_var = ds[varname]
    for scen in scens_sorted:
        ds_scen = ds_var.sel(scen=scen)
        median = (ds_scen.sel(sim='abrupt') - ds_scen.sel(sim='noperma')).median(dim=['config', 'data_LULCC', 'mod']) / ds_scen.sel(sim='abrupt').median(dim=['config', 'data_LULCC', 'mod']).values * 100
        ax.plot(median['year'], median.values, color=scen_colors[scen])
        for sel_year in sel_years:
            print(f'{sel_year}: {median.sel(year=sel_year).values:.2f}%', end='\t')
        print(f'{standardize_scen_names([scen])[0]}')

    if varname == varname1:
        ax.set_ylabel(r'ΔGMST (%)', fontsize='medium')
    elif varname == varname2:
        ax.set_ylabel(f'SLR (%)', fontsize='medium')
    elif varname == varname3:
        ax.set_ylabel(f'Damage cost (%)', fontsize='medium')

    ax.tick_params(axis='both', which='major', direction='in', labelsize='small')
    ax.grid(ls='--', alpha=0.5)
    ax.axvline(x=2050, color='k', ls='--', lw=1)
    ax.axvline(x=2100, color='k', ls='--', lw=1)
    ax.xaxis.set_major_locator(plt.FixedLocator([2025, 2050, 2075, 2100]))
    

legend_handles = [Line2D([0], [0], color=scen_colors[scen], lw=2) for scen in scens_sorted]
legend_labels = standardize_scen_names(scens_sorted)
fig.legend(
    handles=legend_handles[::-1], labels=legend_labels[::-1],
    prop={'size': 'medium', 'style': 'italic'}, 
    frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01)
)
plt.subplots_adjust(bottom=0.35, top=0.95, left=0.05, right=0.98, hspace=0.1, wspace=0.3)
plt.show()


# In[6]:


## 1.4 timeseries of permafrost carbon emissions

dir = './results/abrupt/constrained/'

fig, axes = plt.subplots(2, 2, figsize=(5, 5), **{'sharex': True, 'sharey': 'col'}, dpi=300)

label_x = -0.2
label_y = 1.05

qt = 0.5
for i in range(2):
    ax = axes[i, 0]
    if i==0 : ylabel = 'Gradual'; varname = 'D_Epf_gr_CO2_scen.nc'
    elif i==1: ylabel = 'Abrupt'; varname = 'D_Epf_ab_CO2_scen.nc'
    print(f'\n {ylabel}')
    print(f'Mean\tStd\tMedian\tMin\tMax\tVariable|Scenario|Type')
    ax.text(label_x, label_y, 'a', transform=ax.transAxes, fontsize='large', fontweight='bold')
    ds_var = (xr.load_dataarray(dir + varname)).cumsum('year') + \
        (xr.load_dataarray(dir + varname.replace('CO2', 'CH4')) / 1.0e3).cumsum('year')
    # ds_var_hist = (xr.load_dataarray(dir + 'D_Epf_CO2_hist.nc').sel(year=slice(1870, 2023))).cumsum('year').sel(year=2023, drop=True)
    # ds_var = ds_var_scen + ds_var_hist
    # print(ds_var_hist)
    for scen in scens_sorted:
        ds_scen = ds_var.sel(scen=scen)
        quantile = ds_scen.quantile(qt, dim=['config', 'data_LULCC', 'mod'])
        ax.plot(quantile['year'], quantile.values, color=scen_colors[scen])
        ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 2))
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(ls='--', alpha=0.5)
        ax.axvline(x=2050, color='k', ls='--', lw=1)
        ax.axvline(x=2100, color='k', ls='--', lw=1)
        ax.set_ylabel(f'{ylabel}\nCO$_2$ (PgC)', fontsize='large')
        fmt = '.0f'
        stats = [ds_scen.sel(year=2100).mean().values, 
               ds_scen.sel(year=2100).std().values, 
               quantile.sel(year=2100).values, 
               ds_scen.sel(year=2100).min().values,
               ds_scen.sel(year=2100).max().values]
        print('\t'.join([f'{s:{fmt}}' for s in stats])+f'\tCO₂ emission|{standardize_scen_names([scen])[0]}|baseline')

for i in range(2):
    ax = axes[i, 1]
    if i==0: ylabel = 'Gradual'; varname = 'D_Epf_gr_CO2_scen.nc'
    elif i==1: ylabel = 'Abrupt'; varname = 'D_Epf_ab_CO2_scen.nc'
    print(f'\n {ylabel}')
    ax.text(label_x, label_y, 'b', transform=ax.transAxes, fontsize='large', fontweight='bold')
    ds_var = xr.load_dataarray(dir + varname) + xr.load_dataarray(dir + varname.replace('CO2', 'CH4')) / 1.0e3
    for scen in scens_sorted:
        ds_scen = ds_var.sel(scen=scen)
        quantile = ds_scen.quantile(qt, dim=['config', 'data_LULCC', 'mod'])
        ax.plot(quantile['year'], quantile.values, color=scen_colors[scen])
        print(f'{quantile.sel(year=2100).values:<4.2f}\tCO₂ emission|{standardize_scen_names([scen])[0]}|permafrost')
    ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(ls='--', alpha=0.5)
    ax.axvline(x=2050, color='k', ls='--', lw=1)
    ax.axvline(x=2100, color='k', ls='--', lw=1)
    ax.set_ylabel(f'{ylabel}\nCO$_2$ (PgC yr$^{{-1}}$)', fontsize='large')
    ax.xaxis.set_major_locator(plt.FixedLocator([2025, 2050, 2075, 2100]))

legend_handles = [Line2D([0], [0], color=scen_colors[scen], lw=2) for scen in scens_sorted]
legend_labels = standardize_scen_names(scens_sorted)
fig.legend(
    handles=legend_handles[::-1], labels=legend_labels[::-1],
    prop={'size': 'medium', 'style': 'italic'}, 
    frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01)
)
fig.supxlabel('Year', y=0.15, fontsize='large')
plt.subplots_adjust(bottom=0.25, top=0.98, left=0.05, right=0.98, hspace=0.2, wspace=0.45)
plt.show()


# In[ ]:


## 1.5 timeseries of permafrost carbon emissions

fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharex=True, dpi=300)

label_x = -0.2
label_y = 1.05

qt = 0.5
ax = axes[0]
ax.text(label_x, label_y, 'a', transform=ax.transAxes, fontsize='large', fontweight='bold')
ds_var = (xr.load_dataarray(dir2 + 'D_Epf_CH4_scen.nc') / 1.0e3).cumsum('year')
for scen in scens_sorted:
    ds_scen = ds_var.sel(scen=scen)
    quantile = ds_scen.quantile(qt, dim=['config', 'data_LULCC', 'mod'])
    ax.plot(quantile['year'], quantile.values, color=scen_colors[scen])
    ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(ls='--', alpha=0.5)
    ax.axvline(x=2050, color='k', ls='--', lw=1)
    ax.axvline(x=2100, color='k', ls='--', lw=1)
    ax.set_ylabel('Permafrost CH$_4$ (PgC)', fontsize='large')
    print(f'{quantile.sel(year=2100).values:<4.2f}\tCH₄ emission|{standardize_scen_names([scen])[0]}|baseline')

ax = axes[1]
ax.text(label_x, label_y, 'b', transform=ax.transAxes, fontsize='large', fontweight='bold')
# ds_var = (xr.load_dataarray(dir2 + 'D_Epf_CO2_scen.nc') + xr.load_dataarray(dir2 + 'D_Epf_CH4_scen.nc') / 1.0e3).cumsum('year')
ds_var = (xr.load_dataarray(dir2 + 'D_Epf_CH4_scen.nc') * (130 / np.exp(0.01384 * (xr.load_dataarray(dir2 + 'D_Epf_CH4_scen.nc').year - 2023)) + 0.16) / 1.0e3).cumsum('year')  # convert CH4 to CO2e using GWP100 and decay factor
for scen in scens_sorted:
    ds_scen = ds_var.sel(scen=scen)
    quantile = ds_scen.quantile(qt, dim=['config', 'data_LULCC', 'mod'])
    ax.plot(quantile['year'], quantile.values, color=scen_colors[scen])
    print(f'{quantile.sel(year=2100).values:<4.2f}\tCH₄ emission|{standardize_scen_names([scen])[0]}|permafrost')
ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.grid(ls='--', alpha=0.5)
ax.axvline(x=2050, color='k', ls='--', lw=1)
ax.axvline(x=2100, color='k', ls='--', lw=1)
ax.set_ylabel('Permafrost CO$_2$e (PgC)', fontsize='large')
ax.xaxis.set_major_locator(plt.FixedLocator([2025, 2050, 2075, 2100]))

legend_handles = [Line2D([0], [0], color=scen_colors[scen], lw=2) for scen in scens_sorted]
legend_labels = standardize_scen_names(scens_sorted)
fig.legend(
    handles=legend_handles[::-1], labels=legend_labels[::-1],
    prop={'size': 'medium', 'style': 'italic'}, 
    frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01)
)
fig.supxlabel('Year', y=0.18, fontsize='large')
plt.subplots_adjust(bottom=0.3, top=0.98, left=0.05, right=0.98, hspace=0.2, wspace=0.45)
plt.show()


# In[ ]:


## 1.6 timeseries of DC

varname = 'dmg_tot'
sel_dict = {'reg_IMAGE': 'World', 'adaptation': True}

dir1 = './results/noperma/constrained/'
dir2 = './results/abrupt/constrained/'
var_list = []
for dir in [dir1, dir2]:
    var = xr.load_dataarray(f'{dir}{varname}_scen.nc')
    sel_dict_var = {k: v for k, v in sel_dict.items() if k in var.dims}
    var = var.sel(**sel_dict_var, drop=True)
    var_base = xr.load_dataarray(f'{dir}{varname}_hist.nc')
    base_dict = {k:v for k, v in sel_dict_var.items() if k in var_base.dims}
    base_year = get_baseline_year(varname)
    base_dict['year'] = slice(*base_year)
    var_base = var_base.sel(**base_dict).mean('year')
    var = (var - var_base).sel(scen=scens_sorted).squeeze()
    var_list.append(var.expand_dims({'sim': [f'noperma' if dir == dir1 else 'abrupt']}).assign_coords(sim=[f'noperma' if dir == dir1 else 'abrupt']))

    for attr in ['unit', 'units']:
        if attr in var.attrs:
            unit = f' ({var.attrs[attr]})'
            break
    else:
        print(f'No unit attribute found in {varname}_scen.nc')
        unit = ''
    
ds = xr.merge(var_list)

fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharex=True, dpi=300)

label_x = -0.2
label_y = 1.05

qt = 0.5
ax = axes[0]
ax.text(label_x, label_y, 'a', transform=ax.transAxes, fontsize='large', fontweight='bold')
gdp = xr.load_dataarray('results/ngfs/GDP_ngfs.nc').mean('mod') * 1e-3
for scen in scens_sorted:
    ds_scen = gdp.sel(scen=scen)
    ax.plot(ds_scen.year, ds_scen.values, color=scen_colors[scen])
    ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((0, 2))
    # ax.yaxis.set_major_formatter(formatter)
    ax.grid(ls='--', alpha=0.5)
    ax.axvline(x=2050, color='k', ls='--', lw=1)
    ax.axvline(x=2100, color='k', ls='--', lw=1)
    ax.set_ylabel('GDP (trillion 2010 US$)', fontsize='large')
    print(f'{quantile.sel(year=2100).values:<4.2f}\tGDP|{standardize_scen_names([scen])[0]}|baseline')

ax = axes[1]
ax.text(label_x, label_y, 'b', transform=ax.transAxes, fontsize='large', fontweight='bold')
ds_var = (ds[varname].sel(sim='abrupt', drop=True) - ds[varname].sel(sim='noperma', drop=True)).quantile(qt, dim=['config', 'data_LULCC', 'mod'])
ds_var = ds_var * gdp * 1e-2
for scen in scens_sorted:
    ds_scen = ds_var.sel(scen=scen)
    ax.plot(ds_scen.year, ds_scen.values, color=scen_colors[scen])
    print(f'{ds_scen.sel(year=2100).values:<4.2f}\tGDP loss|{standardize_scen_names([scen])[0]}|permafrost')
ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.grid(ls='--', alpha=0.5)
ax.set_ylim(0, None)
ax.axvline(x=2050, color='k', ls='--', lw=1)
ax.axvline(x=2100, color='k', ls='--', lw=1)
ax.set_ylabel('GDP loss (trillion 2010 US$)', fontsize='large')
ax.xaxis.set_major_locator(plt.FixedLocator([2025, 2050, 2075, 2100]))

legend_handles = [Line2D([0], [0], color=scen_colors[scen], lw=2) for scen in scens_sorted]
legend_labels = standardize_scen_names(scens_sorted)
fig.legend(
    handles=legend_handles[::-1], labels=legend_labels[::-1],
    prop={'size': 'medium', 'style': 'italic'}, 
    frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01)
)
fig.supxlabel('Year', y=0.18, fontsize='large')
plt.subplots_adjust(bottom=0.3, top=0.98, left=0.05, right=0.98, hspace=0.2, wspace=0.45)
plt.show()


# In[19]:


## 1.7 timeseries of DC

varname = 'dmg_tot'
sel_dict = {'reg_IMAGE': 'World'}

dir1 = './results/noperma/constrained/'
dir2 = './results/abrupt/constrained/'
var_list = []
for dir in [dir1, dir2]:
    var = xr.load_dataarray(f'{dir}{varname}_scen.nc')
    sel_dict_var = {k: v for k, v in sel_dict.items() if k in var.dims}
    var = var.sel(**sel_dict_var, drop=True)
    var_base = xr.load_dataarray(f'{dir}{varname}_hist.nc')
    base_dict = {k:v for k, v in sel_dict_var.items() if k in var_base.dims}
    base_year = get_baseline_year(varname)
    base_dict['year'] = slice(*base_year)
    var_base = var_base.sel(**base_dict).mean('year')
    var = (var - var_base).sel(scen=scens_sorted).squeeze()
    var_list.append(var.expand_dims({'sim': [f'noperma' if dir == dir1 else 'abrupt']}).assign_coords(sim=[f'noperma' if dir == dir1 else 'abrupt']))

    for attr in ['unit', 'units']:
        if attr in var.attrs:
            unit = f' ({var.attrs[attr]})'
            break
    else:
        print(f'No unit attribute found in {varname}_scen.nc')
        unit = ''
    
ds = xr.merge(var_list)

fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharex=True, dpi=300)

label_x = -0.2
label_y = 1.05

qt = 0.5
ax = axes[0]
ax.text(label_x, label_y, 'a', transform=ax.transAxes, fontsize='large', fontweight='bold')
ds_var = ds[varname].sel(sim='noperma', adaptation=False).quantile(qt, dim=['config', 'data_LULCC', 'mod'])
ds_var = ds_var / ds[varname].sel(sim='noperma', adaptation=True).quantile(qt, dim=['config', 'data_LULCC', 'mod'])
for scen in scens_sorted:
    ds_scen = ds_var.sel(scen=scen)
    ax.plot(ds_scen.year, ds_scen.values, color=scen_colors[scen])
    ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((0, 2))
    # ax.yaxis.set_major_formatter(formatter)
    ax.grid(ls='--', alpha=0.5)
    ax.axvline(x=2050, color='k', ls='--', lw=1)
    ax.axvline(x=2100, color='k', ls='--', lw=1)
    ax.set_ylabel('Baseline ratio', fontsize='large')
    print(f'{quantile.sel(year=2100).values:<4.2f}\tratio|{standardize_scen_names([scen])[0]}|baseline')

ax = axes[1]
ax.text(label_x, label_y, 'b', transform=ax.transAxes, fontsize='large', fontweight='bold')
ds_var = (ds[varname].sel(sim='abrupt', adaptation=False) - ds[varname].sel(sim='noperma', adaptation=False)).quantile(qt, dim=['config', 'data_LULCC', 'mod'])
ds_var = ds_var / (ds[varname].sel(sim='abrupt', adaptation=True) - ds[varname].sel(sim='noperma', adaptation=True)).quantile(qt, dim=['config', 'data_LULCC', 'mod'])
for scen in scens_sorted:
    ds_scen = ds_var.sel(scen=scen)
    ax.plot(ds_scen.year, ds_scen.values, color=scen_colors[scen])
    print(f'{ds_scen.sel(year=2100).values:<4.2f}\tratio|{standardize_scen_names([scen])[0]}|permafrost')
ax.tick_params(axis='both', which='major', direction='in', labelsize='large', pad=5)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.grid(ls='--', alpha=0.5)
ax.axvline(x=2050, color='k', ls='--', lw=1)
ax.axvline(x=2100, color='k', ls='--', lw=1)
ax.set_ylabel('Ratio', fontsize='large')
ax.xaxis.set_major_locator(plt.FixedLocator([2025, 2050, 2075, 2100]))

legend_handles = [Line2D([0], [0], color=scen_colors[scen], lw=2) for scen in scens_sorted]
legend_labels = standardize_scen_names(scens_sorted)
fig.legend(
    handles=legend_handles[::-1], labels=legend_labels[::-1],
    prop={'size': 'medium', 'style': 'italic'}, 
    frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01)
)
fig.supxlabel('Year', y=0.18, fontsize='large')
plt.subplots_adjust(bottom=0.3, top=0.98, left=0.05, right=0.98, hspace=0.2, wspace=0.45)
plt.show()


# In[ ]:


## 2.1 cumulative distribution function

varname1 = 'D_Tg'
varname2 = 'D_Htot'
varname3 = 'dmg_tot'
# sel_dict = {'scen': scens_sorted, 'reg_IMAGE': 'World', 'adaptation': True}
sel_dict = {'scen':[scens_sorted[i] for i in [5, 3, 2, 0]], 'reg_IMAGE': 'World', 'adaptation': True}

dir1 = './results/noperma/constrained/'
dir2 = './results/abrupt/constrained/'
var_list = []
for dir in [dir1, dir2]:
    for varname in [varname1, varname2, varname3]:
        var = xr.load_dataarray(f'{dir}{varname}_scen.nc')
        sel_dict_var = {k: v for k, v in sel_dict.items() if k in var.dims}
        var = var.sel(**sel_dict_var, drop=True)
        var_base = xr.load_dataarray(f'{dir}{varname}_hist.nc')
        base_dict = {k:v for k, v in sel_dict_var.items() if k in var_base.dims}
        base_year = get_baseline_year(varname)
        base_dict['year'] = slice(*base_year)
        var_base = var_base.sel(**base_dict).mean('year')
        var = (var - var_base).sel(scen=sel_dict['scen'], year=2100).squeeze()
        var_list.append(var.expand_dims({'sim': [f'noperma' if dir == dir1 else 'abrupt']}).assign_coords(sim=[f'noperma' if dir == dir1 else 'abrupt']))

        for attr in ['unit', 'units']:
            if attr in var.attrs:
                unit = f' ({var.attrs[attr]})'
                break
        else:
            print(f'No unit attribute found in {varname}_scen.nc')
            unit = ''

print('sel_dict:', sel_dict)    
ds = xr.merge(var_list)

csv_out = 'results/percentiles.csv'
width = 12
reversed = False
fig, axes = plt.subplots(
    len(sel_dict['scen']), 3, figsize=(2 * len(sel_dict['scen']), 6), 
    **{'sharex': 'col', 'sharey': True, 'dpi': 300}
    )
with open(csv_out, 'w') as f_csv:
    f_csv.write('Variable,Scenario,With,p50,p90,p95\n')
    for j, varname in enumerate([varname1, varname2, varname3]):
        print(f'\n{varname:<{width}} {"baseline":<{width}} {"with":<{width}} {"scenario":<{width}}')
        for i, scen in enumerate(sel_dict['scen']):
            ax = axes[i, j]
            scen_name = standardize_scen_names([scen])[0]
            ax.text(
                0.0, 1.05, 
                chr(97 + i * 3 + j), 
                transform=ax.transAxes, 
                fontsize='large', fontweight='bold'
            )
            var_noperma = ds[varname].sel(sim='noperma', scen=scen).to_dataframe().reset_index()
            var_abrupt = ds[varname].sel(sim='abrupt', scen=scen).to_dataframe().reset_index()
            kurt_noperma = kurtosis(var_noperma[varname], fisher=True, bias=False)
            kurt_abrupt = kurtosis(var_abrupt[varname], fisher=True, bias=False)
            print(f'{varname:<{width}} {kurt_noperma:<{width}.2f} {kurt_abrupt:<{width}.2f} {scen:<{width}}')

            q50_abrupt = var_abrupt[varname].quantile(0.5).round(2 if varname != 'D_Htot' else 0)
            q90_abrupt = var_abrupt[varname].quantile(0.9).round(2 if varname != 'D_Htot' else 0)
            q95_abrupt = var_abrupt[varname].quantile(0.95).round(2 if varname != 'D_Htot' else 0)
            q50_noperma = var_noperma[varname].quantile(0.5).round(2 if varname != 'D_Htot' else 0)
            q90_noperma = var_noperma[varname].quantile(0.9).round(2 if varname != 'D_Htot' else 0)
            q95_noperma = var_noperma[varname].quantile(0.95).round(2 if varname != 'D_Htot' else 0)

            f_csv.write(f'{varname},{scen},Baseline,{q50_noperma},{q90_noperma},{q95_noperma}\n')
            f_csv.write(f'{varname},{scen},With permafrost,{q50_abrupt},{q90_abrupt},{q95_abrupt}\n')
            f_csv.write(f'{varname},{scen},Increment,{q50_abrupt - q50_noperma},{q90_abrupt - q90_noperma},{q95_abrupt - q95_noperma}\n')

            if varname == varname1:
                f_exp = 3
                inv = 0.2
                ticks_inset = np.arange(1, 3.4, inv)
                if reversed:
                    ax.set_ylim(0, 5)
                    ax.set_ylabel(r'ΔGMST (°C)' if i == 0 else '', labelpad=1, fontsize='large')
                else:
                    ax.set_xlim(0, 5)
                    ax.set_xticks([0, 1, 2, 3, 4, 5])
                    ax.set_xlabel(r'ΔGMST (°C)' if i == len(sel_dict['scen']) - 1 else '', labelpad=5, fontsize='large')
            elif varname == varname2:
                f_exp = 5
                inv = 40
                ticks_inset = np.arange(420, 820, inv)
                if reversed:
                    ax.set_ylim(0, 1200)
                    ax.set_ylabel(f'SLR (mm)' if i == 0 else '', labelpad=1, fontsize='large')
                else:
                    ax.set_xlim(0, 1200)
                    ax.set_xticks([0, 400, 800, 1200])
                    ax.set_xlabel(f'SLR (mm)' if i == len(sel_dict['scen']) - 1 else '', labelpad=5, fontsize='large')
            elif varname == varname3:
                f_exp = 2
                inv = 1
                ticks_inset = np.arange(0, 12, inv)
                if reversed:
                    ax.set_ylim(0, 15)
                    ax.set_ylabel(f'DC (% of GDP)' if i == 0 else '', labelpad=1, fontsize='large')
                else:
                    ax.set_xlim(0, 15)
                    ax.set_xticks([0, 5, 10, 15])
                    ax.set_xlabel(f'DC (% of GDP)' if i == len(sel_dict['scen']) - 1 else '', labelpad=5, fontsize='large')

            if reversed:
                sns.ecdfplot(
                    data=var_noperma, y=varname, color=sim_colors[0], lw=1, 
                    label='No permafrost', ax=ax
                )
                sns.ecdfplot(
                    data=var_abrupt, y=varname, color=sim_colors[2], lw=1, 
                    label='With permafrost', ax=ax
                )
            else:
                sns.ecdfplot(
                    data=var_noperma, x=varname, color=sim_colors[0], lw=1, 
                    label='No permafrost', ax=ax
                )
                sns.ecdfplot(
                    data=var_abrupt, x=varname, color=sim_colors[2], lw=1, 
                    label='With permafrost', ax=ax
                )

            ## box plot
            ax_box = ax.twiny() if reversed else ax.twinx()
            box_lim = (0, 2)
            if reversed: 
                ax_box.set_xlim(box_lim)
                ax_box.tick_params(axis='x', top=False, labeltop=False)
            else:
                ax_box.set_ylim(box_lim)
                ax_box.tick_params(axis='y', right=False, labelright=False)
            box_props = dict(
                whis=(10, 90), notch=True, showmeans=False, 
                medianprops=dict(color='k'), patch_artist=True, 
                showfliers=False, widths=0.15,
                orientation='vertical' if reversed else 'horizontal'
            )
            ax_box.boxplot(
                var_noperma[varname], positions=[0.2],
                **box_props, boxprops=dict(facecolor=sim_colors[0], alpha=0.8)
            )
            ax_box.boxplot(
                var_abrupt[varname], positions=[0.4],
                **box_props, boxprops=dict(facecolor=sim_colors[2], alpha=0.8)
            )
            ax_box.tick_params(axis='x', top=False, labeltop=False)

            ## add first inset
            x1, x2 = 0.85, 0.95
            y1, y2 = var_noperma[varname].quantile(x1), var_abrupt[varname].quantile(x2)
            pt_range = y2 - y1
            if reversed:
                axins1 = ax.inset_axes([
                    0.68, 0.9 - f_exp * pt_range / (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                    (x2 - x1) * 3, f_exp * pt_range / (ax.get_ylim()[1] - ax.get_ylim()[0])])
                axins1.set_ylim(y1, y2)
            else:
                axins1 = ax.inset_axes([
                    0.95 - f_exp * pt_range / (ax.get_xlim()[1] - ax.get_xlim()[0]), 0.6, 
                    f_exp * pt_range / (ax.get_xlim()[1] - ax.get_xlim()[0]), (x2 - x1) * 3])
                axins1.set_xlim(y1, y2)
                ticks_inset1 = ticks_inset[(ticks_inset >= y1) & (ticks_inset <= y2)]
                if len(ticks_inset1) == 1: ticks_inset1 = np.array([ticks_inset1[0] - inv/2, ticks_inset1[0] + inv/2])
            plot_clipped(var_noperma[varname], sim_colors[0], axins1, method='ecdf',vals=[x1 * 100, x2 * 100], reversed=reversed)
            plot_clipped(var_abrupt[varname], sim_colors[2], axins1, method='ecdf', vals=[x1 * 100, x2 * 100], reversed=reversed)
            ax.indicate_inset_zoom(axins1, edgecolor='black')

            axins1.set_ylabel('')
            axins1.set_xlabel('')
            axins1.grid(ls='--', alpha=0.5)
            axins1.tick_params(axis='both', which='major', direction='in', labelsize='medium', pad=1)
            axins1.tick_params(axis='both', which='minor', direction='in')
            if reversed:
                axins1.tick_params(axis='x', left=False, labelleft=False)
                axins1.xaxis.set_major_locator(plt.FixedLocator([(x1 + x2) / 2]))
                axins1.yaxis.set_major_locator(ticker.MaxNLocator(2))
                if axins1.get_ylim()[1] - axins1.get_ylim()[0] < 0.1:
                    axins1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                elif axins1.get_ylim()[1] - axins1.get_ylim()[0] < 1:
                    axins1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                else:
                    axins1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
            else:
                axins1.tick_params(axis='y', left=False, labelleft=False)
                axins1.xaxis.set_major_locator(plt.FixedLocator(ticks_inset1))
                if axins1.get_xlim()[1] - axins1.get_xlim()[0] < 0.1:
                    axins1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                elif axins1.get_xlim()[1] - axins1.get_xlim()[0] < 1:
                    axins1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                else:
                    axins1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
                axins1.yaxis.set_major_locator(ticker.MaxNLocator(4))
                axins1.tick_params(axis='y', left=False, labelleft=False)


            ## add second inset
            x1, x2 = 0.45, 0.55
            if reversed:
                axins2 = ax.inset_axes([
                    0.18, 0.9 - f_exp * pt_range / (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                    (x2 - x1) * 3, f_exp * pt_range / (ax.get_ylim()[1] - ax.get_ylim()[0])])
                y1, y2 = var_noperma[varname].quantile(x1), var_abrupt[varname].quantile(x2)
                axins2.set_ylim((y1 + y2 - pt_range) / 2 , (y1 + y2 + pt_range) / 2)
            else:
                axins2 = ax.inset_axes([
                    0.95 - f_exp * pt_range / (ax.get_xlim()[1] - ax.get_xlim()[0]), 0.15, 
                    f_exp * pt_range / (ax.get_xlim()[1] - ax.get_xlim()[0]), (x2 - x1) * 3])
                y1, y2 = var_noperma[varname].quantile(x1), var_abrupt[varname].quantile(x2)
                axins2.set_xlim((y1 + y2 - pt_range) / 2 , (y1 + y2 + pt_range) / 2)
                ticks_inset2 = ticks_inset[(ticks_inset >= (y1 + y2 - pt_range) / 2) & (ticks_inset <= (y1 + y2 + pt_range) / 2)]
                if len(ticks_inset2) == 1: ticks_inset2 = np.array([ticks_inset2[0] - inv/2, ticks_inset2[0] + inv/2])
            plot_clipped(var_noperma[varname], sim_colors[0], axins2, method='ecdf', vals=[x1 * 100, x2 * 100], reversed=reversed)
            plot_clipped(var_abrupt[varname], sim_colors[2], axins2, method='ecdf', vals=[x1 * 100, x2 * 100], reversed=reversed)
            ax.indicate_inset_zoom(axins2, edgecolor='black')

            axins2.set_ylabel('')
            axins2.set_xlabel('')
            axins2.grid(ls='--', alpha=0.5)
            axins2.tick_params(axis='both', which='major', direction='in', labelsize='medium', pad=1)
            axins2.tick_params(axis='both', which='minor', direction='in')

            if reversed:
                axins2.tick_params(axis='x', left=False, labelleft=False)
                axins2.xaxis.set_major_locator(plt.FixedLocator([(x1 + x2) / 2]))
                axins2.yaxis.set_major_locator(ticker.MaxNLocator(2))
                if axins2.get_ylim()[1] - axins2.get_ylim()[0] < 0.1:
                    axins2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                elif axins2.get_ylim()[1] - axins2.get_ylim()[0] < 1:
                    axins2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                else:
                    axins2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
            else:
                axins2.tick_params(axis='y', left=False, labelleft=False)
                axins2.xaxis.set_major_locator(plt.FixedLocator(ticks_inset2))
                # if axins2.get_xlim()[1] - axins2.get_xlim()[0] < 0.1:
                #     axins2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                # elif axins2.get_xlim()[1] - axins2.get_xlim()[0] < 1:
                #     axins2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                # else:
                #     axins2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
                axins2.yaxis.set_major_locator(ticker.MaxNLocator(4))
                axins2.tick_params(axis='y', left=False, labelleft=False)

            if reversed:
                ax.set_xlabel('')
                ax.set_title(scen_name if i == 0 else '', fontsize='large', fontstyle='italic')
                if ax.get_ylim()[1] - ax.get_ylim()[0] < 1:
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                else:
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                ax.xaxis.set_major_locator(plt.FixedLocator(np.linspace(0, 1, 6)))
                formatter = ticker.FuncFormatter(lambda x, _: f'{x:g}')
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            else:
                ax.set_ylabel(scen_name if j == 0 else '', fontsize='large', fontstyle='italic')
                if ax.get_xlim()[1] - ax.get_xlim()[0] < 1:
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                else:
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_locator(plt.FixedLocator(np.linspace(0, 1, 6)))
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
                formatter = ticker.FuncFormatter(lambda y, _: f'{y:g}')
                ax.yaxis.set_major_formatter(formatter)
            
            ax.tick_params(axis='both', which='major', direction='in', labelsize='medium')
            ax.tick_params(axis='both', which='minor', direction='in')
            ax.grid(which='major', axis='both', ls='--', alpha=0.5)

f_csv.close()

legend_handles = [
    Line2D([0], [0], color=sim_colors[0], lw=2, markersize=6),
    Line2D([0], [0], color=sim_colors[2], lw=2, markersize=6)
]
fig.legend(
    handles=legend_handles, labels=['Baseline', 'With permafrost impacts'],
    fontsize='large', frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.01)
)
fig.align_ylabels(axes[:, 0])
plt.subplots_adjust(bottom=0.08, top=0.95, left=0.05, right=0.98, hspace=0.25, wspace=0.1)
plt.show()


# In[22]:


## 2.2 risk distribution

varname1 = 'D_Tg'
varname2 = 'D_Htot'
varname3 = 'dmg_tot'
sel_dict = {'scen':[scens_sorted[i] for i in [5, 3, 2, 0]], 'reg_IMAGE': 'World', 'adaptation': True}

dir1 = './results/noperma/constrained/'
dir2 = './results/abrupt/constrained/'
var_list = []
for dir in [dir1, dir2]:
    for varname in [varname1, varname2, varname3]:
        var = xr.load_dataarray(f'{dir}{varname}_scen.nc')
        sel_dict_var = {k: v for k, v in sel_dict.items() if k in var.dims}
        var = var.sel(**sel_dict_var, drop=True)
        var_base = xr.load_dataarray(f'{dir}{varname}_hist.nc')
        base_dict = {k:v for k, v in sel_dict_var.items() if k in var_base.dims}
        base_year = get_baseline_year(varname)
        base_dict['year'] = slice(*base_year)
        var_base = var_base.sel(**base_dict).mean('year')
        var = (var - var_base).sel(scen=sel_dict['scen'], year=2100).squeeze()
        var_list.append(var.expand_dims({'sim': [f'noperma' if dir == dir1 else 'abrupt']}).assign_coords(sim=[f'noperma' if dir == dir1 else 'abrupt']))

        for attr in ['unit', 'units']:
            if attr in var.attrs:
                unit = f' ({var.attrs[attr]})'
                break
        else:
            print(f'No unit attribute found in {varname}_scen.nc')
            unit = ''

print('sel_dict:', sel_dict)    
ds = xr.merge(var_list)

reversed = False

fig, axes = plt.subplots(
    len(sel_dict['scen']), 3, figsize=(2 * len(sel_dict['scen']), 6), 
    sharex='col', sharey='col', dpi=300
    )
for j, varname in enumerate([varname1, varname2, varname3]):
    for i, scen in enumerate(sel_dict['scen']):
        ax = axes[i, j]
        scen_name = standardize_scen_names([scen])[0]
        ax.text(
            -0.02, 1.02, 
            chr(97 + i * 3 + j), 
            transform=ax.transAxes, 
            fontsize='large', fontweight='bold'
        )

        var_noperma = ds[varname].sel(sim='noperma', scen=scen).to_dataframe().reset_index()
        var_abrupt = ds[varname].sel(sim='abrupt', scen=scen).to_dataframe().reset_index()
        if reversed:
            sns.kdeplot(
                data=var_noperma, y=varname, fill=False, color=sim_colors[0], lw=1, 
                label='No permafrost', ax=ax
            )
            sns.kdeplot(
                data=var_abrupt, y=varname, fill=False, color=sim_colors[2], lw=1, 
                label='With permafrost', ax=ax
            )
        else:
            sns.kdeplot(
                data=var_noperma, x=varname, fill=False, color=sim_colors[0], lw=1, 
                label='No permafrost', ax=ax
            )
            sns.kdeplot(
                data=var_abrupt, x=varname, fill=False, color=sim_colors[2], lw=1, 
                label='With permafrost', ax=ax
            )

        ## box plot
        ax_box = ax.twiny() if reversed else ax.twinx()
        lim_box = (0, 2)
        if reversed:
            ax_box.set_xlim(lim_box)
            ax_box.set_xlabel('')
            ax_box.tick_params(axis='x', top=False, labeltop=False)
        else:
            ax_box.set_ylim(lim_box)
            ax_box.set_ylabel('')
            ax_box.tick_params(axis='y', right=False, labelright=False)
        
        box_props = dict(
            whis=(10, 90), notch=True, showmeans=False, 
            medianprops=dict(color='k'), patch_artist=True, 
            showfliers=False, widths=0.15,
            orientation='vertical' if reversed else 'horizontal'
        )

        ax_box.boxplot(
            var_noperma[varname], positions=[0.4],
            **box_props, boxprops=dict(facecolor=sim_colors[0], alpha=0.8)
        )
        ax_box.boxplot(
            var_abrupt[varname], positions=[0.2],
            **box_props, boxprops=dict(facecolor=sim_colors[2], alpha=0.8)
        )

        ## add inset for zoom-in
        x1, x2 = var_noperma[varname].quantile(0.85), var_abrupt[varname].quantile(0.95)
        axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
        if reversed:
            axins.set_ylim(x1, x2)
        else:
            axins.set_xlim(x1, x2)
        plot_clipped(var_noperma[varname], sim_colors[0], axins, vals=[85, 95], reversed=reversed)
        plot_clipped(var_abrupt[varname], sim_colors[2], axins, vals=[85, 95], reversed=reversed)

        if varname == varname1:
            if reversed:
                ax.set_xlim(0, 2.5)
                ax.set_ylim(0, 5)
                ax.set_ylabel(r'ΔGMST (°C)' if i == 0 else '', labelpad=1, fontsize='large')
            else:
                ax.set_xlim(0, 5)
                ax.set_xticks([0, 1, 2, 3, 4, 5])
                ax.set_ylim(0, 2.5)
                ax.set_yticks([])
                ax.set_xlabel(r'ΔGMST (°C)' if i == len(sel_dict['scen']) - 1 else '', labelpad=5, fontsize='large')
        elif varname == varname2:
            if reversed:
                ax.set_xlim(0, 0.01)
                ax.set_ylim(0, 1500)
                ax.set_ylabel(f'SLR (mm)' if i == 0 else '', labelpad=1, fontsize='large')
            else:
                ax.set_xlim(0, 1200)
                ax.set_xticks([0, 400, 800, 1200])
                ax.set_ylim(0, 0.01)
                ax.set_yticks([])
                ax.set_xlabel(f'SLR (mm)' if i == len(sel_dict['scen']) - 1 else '', labelpad=5, fontsize='large')
        elif varname == varname3:
            if reversed:
                ax.set_xlim(0, 1)
                ax.set_ylim(None, 15)
                ax.set_ylabel(f'DC (% of GDP)' if i == 0 else '', labelpad=1, fontsize='large')
            else:
                ax.set_xlim(None, 15)
                ax.set_xticks([0, 5, 10, 15])
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.set_xlabel(f'DC (% of GDP)' if i == len(sel_dict['scen']) - 1 else '', labelpad=5, fontsize='large')
        
        axins.set_ylabel('')
        axins.set_xlabel('')
        axins.grid(ls='--', alpha=0.5)
        axins.tick_params(axis='both', which='major', direction='in', labelsize='small')
        axins.tick_params(axis='both', which='minor', direction='in')

        if reversed:    
            axins.tick_params(axis='x', bottom=False, labelbottom=False)
            axins.yaxis.set_major_locator(ticker.MaxNLocator(2))

            if axins.get_ylim()[1] - axins.get_ylim()[0] < 0.1:
                axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            elif axins.get_ylim()[1] - axins.get_ylim()[0] < 1:
                axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            else:
                axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.indicate_inset_zoom(axins, edgecolor='black')

        else:
            axins.tick_params(axis='y', left=False, labelleft=False)
            axins.xaxis.set_major_locator(ticker.MaxNLocator(2))

            if axins.get_xlim()[1] - axins.get_xlim()[0] < 0.1:
                axins.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            elif axins.get_xlim()[1] - axins.get_xlim()[0] < 1:
                axins.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            else:
                axins.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.indicate_inset_zoom(axins, edgecolor='black')

        ax.tick_params(axis='both', which='major', direction='in', labelsize='medium')
        ax.tick_params(axis='both', which='minor', direction='in')
        ax.grid(which='both', ls='--', alpha=0.5)

        txt_prop = dict(fontsize='large', fontstyle='italic', transform=ax.transAxes)
        if reversed:
            ax.text(
                0.5, 0.99, 
                scen_name if i == 0 else '',
                **txt_prop, 
                ha='center', va='bottom',
                rotation=0
            )
        else:
            ax.set_ylabel('')
            ax.text(
                -0.05, 0.5,
                scen_name if j == 0 else '', 
                **txt_prop,
                ha='right', va='center',
                rotation=90
            )

legend_handles = [
    Line2D([0], [0], color=sim_colors[0], lw=1, markersize=6),
    Line2D([0], [0], color=sim_colors[2], lw=1, markersize=6)
]
fig.legend(
    handles=legend_handles, labels=['Baseline', 'With permafrost impacts'],
    fontsize='large', frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.01)
)
fig.align_ylabels(axes[:, 0])
plt.subplots_adjust(bottom=0.08, top=0.95, left=0.05, right=0.98, hspace=0.25, wspace=0.1)
plt.show()


# In[6]:


## 2.3 risk difference distribution

# varname1 = 'D_Tg'
# var_title1 = r'ΔGMST (°C)'
var_title1 = r'ΔGMST-related DC (% of GDP)'
# varname2 = 'D_Htot'
var_title2 = 'SLR (mm)'
var_title2 = r'SLR-related DC (% of GDP)'
varname1 = 'dmg_T'
varname2 = 'dmg_SLR'
varname3 = 'dmg_tot'
var_title3 = 'Total DC (% of GDP)'
sel_dict = {'scen':[scens_sorted[i] for i in [5, 3, 2, 0]], 'reg_IMAGE': 'FSU', 'adaptation': True}
# sel_dict = {'scen':scens_sorted[::-1], 'reg_IMAGE': 'World', 'adaptation': True}

dir1 = './results/noperma/constrained/'
dir2 = './results/abrupt/constrained/'
var_list = []
for dir in [dir1, dir2]:
    for varname in [varname1, varname2, varname3]:
        var = xr.load_dataarray(f'{dir}{varname}_scen.nc')
        sel_dict_var = {k: v for k, v in sel_dict.items() if k in var.dims}
        var = var.sel(**sel_dict_var, drop=True)
        var_base = xr.load_dataarray(f'{dir}{varname}_hist.nc')
        base_dict = {k:v for k, v in sel_dict_var.items() if k in var_base.dims}
        base_year = get_baseline_year(varname)
        base_dict['year'] = slice(*base_year)
        var_base = var_base.sel(**base_dict).mean('year')
        var = (var - var_base).sel(scen=sel_dict['scen'], year=2100).squeeze()
        var_list.append(var.expand_dims({'sim': [f'noperma' if dir == dir1 else 'abrupt']}).assign_coords(sim=[f'noperma' if dir == dir1 else 'abrupt']))

        for attr in ['unit', 'units']:
            if attr in var.attrs:
                unit = f' ({var.attrs[attr]})'
                break
        else:
            print(f'No unit attribute found in {varname}_scen.nc')
            unit = ''

print('sel_dict:', sel_dict)    
ds = xr.merge(var_list)

reversed = False
fig, axes = plt.subplots(
    len(sel_dict['scen']), 3, figsize=(2 * len(sel_dict['scen']), 6), 
    sharex='col', sharey='col', dpi=300
    )
print('varname\tmin\tp1\tp50\tp90\tp95\tmax\tnegative percent')
for j, varname in enumerate([varname1, varname2, varname3]):
    print(f'{varname}')
    for i, scen in enumerate(sel_dict['scen']):
        ax = axes[i, j]
        scen_name = standardize_scen_names([scen])[0]
        ax.text(
            -0.02, 1.02, 
            chr(97 + i * 3 + j), 
            transform=ax.transAxes, 
            fontsize='large', fontweight='bold'
        )
        var_diff = (ds[varname].sel(sim='abrupt', scen=scen) - ds[varname].sel(sim='noperma', scen=scen)).to_dataframe().reset_index()
        fmt = '.3f' if varname != 'D_Htot' else '.0f'
        stats =  [
            var_diff[varname].min(), 
            var_diff[varname].quantile(0.01),
            var_diff[varname].quantile(0.5), 
            var_diff[varname].quantile(0.9), 
            var_diff[varname].quantile(0.95), 
            var_diff[varname].max(),
            var_diff[varname].where(var_diff[varname] < 0).count()/var_diff[varname].notnull().count()
        ]
        print(f'\t' + '\t'.join([f'{s:{fmt}}' for s in stats]) + f'\t{standardize_scen_names([scen])[0]}')

        if reversed:
            sns.kdeplot(
                data=var_diff, y=varname, fill=False, color='k', 
                label='No permafrost', ax=ax
            )
        else:
            sns.kdeplot(
                data=var_diff, x=varname, fill=False, color='k', 
                label='No permafrost', ax=ax
            )

        ## box plot
        ax_box = ax.twiny() if reversed else ax.twinx()
        box_lim = (0, 2)
        if reversed:
            ax_box.set_xlim(box_lim)
            ax_box.tick_params(axis='x', top=False, labeltop=False)
        else:
            ax_box.set_ylim(box_lim)
            ax_box.tick_params(axis='y', right=False, labelright=False)
            
        ax_box.boxplot(
            var_diff[varname], positions=[0.4],
            whis=(10, 90), notch=True, showmeans=False, 
            medianprops=dict(color='k'),
            patch_artist=True, boxprops=dict(facecolor=sim_colors[0], alpha=0.8),
            showfliers=False,
            orientation='vertical' if reversed else 'horizontal',
            widths=0.1
        )
        ax.set_xlabel('')
        ax.set_ylabel('')

        if varname == varname1:
            if reversed:
                ax.set_ylabel(var_title1 if i == 0 else '', labelpad=1, fontsize='large')
            else:
                # ax.set_xlim(None, 4)
                ax.set_xlabel(var_title1 if i == len(sel_dict['scen']) - 1 else '', labelpad=5, fontsize='large')
        elif varname == varname2:
            if reversed:
                ax.set_ylabel(var_title2 if i == 0 else '', labelpad=1, fontsize='large')
            else:
                ax.set_xlabel(var_title2 if i == len(sel_dict['scen']) - 1 else '', labelpad=5, fontsize='large')
        elif varname == varname3:
            if reversed:
                ax.set_ylabel(var_title3 if i == 0 else '', labelpad=1, fontsize='large')
            else:
                ax.set_xlim(None, 4)
                ax.set_xlabel(var_title3 if i == len(sel_dict['scen']) - 1 else '', labelpad=5, fontsize='large')

        txt_prop = dict(fontsize='large', fontstyle='italic', transform=ax.transAxes)
        if reversed:
            ax.text(
                0.5, 0.99, 
                scen_name if i == 0 else '',
                **txt_prop, 
                ha='center', va='bottom',
                rotation=0
            )
            ax.set_xticks([])
        else:
            ax.set_ylabel('')
            ax.text(
                -0.05, 0.5,
                scen_name if j == 0 else '', 
                **txt_prop,
                ha='right', va='center',
                rotation=90
            )
            ax.set_yticks([])

        ax.tick_params(axis='both', which='major', direction='in', labelsize='large')
        ax.tick_params(axis='both', which='minor', direction='in')
        ax.grid(which='both', ls='--', alpha=0.5)

plt.subplots_adjust(bottom=0.08, top=0.95, left=0.05, right=0.98, hspace=0.25, wspace=0.1)
plt.show()


# In[8]:


## 3.1 three different variables for each column

varname = 'dmg_tot'
sel_dict = {'year': 2100, 'scen': [scens_sorted[i] for i in [5, 3, 2, 0]], 'adaptation': False}
dir1 = 'results/noperma/constrained/'
dir3 = 'results/abrupt/constrained/'

base_year = get_baseline_year(varname)

var_scen1 = xr.load_dataarray(f'{dir1}{varname}_scen.nc')
sel_dict = {k: v for k, v in sel_dict.items() if k in var_scen1.dims}
var_scen1 = var_scen1.sel(**sel_dict, drop=True)
for attr in ['unit', 'units']:
    if attr in var_scen1.attrs:
        unit = f' ({var_scen1.attrs[attr]})'
        break
else:
    print(f'No unit attribute found in {varname}_scen.nc')
    unit = ''

var_base1 = xr.load_dataarray(f'{dir1}{varname}_hist.nc')
base_dict = {k:v for k, v in sel_dict.items() if k in var_base1.dims}
base_dict['year'] = slice(*base_year)
var_base1 = var_base1.sel(**base_dict).mean('year')
var1 = (var_scen1 - var_base1).sel(scen=sel_dict['scen']).squeeze().rename({'reg_IMAGE': 'COACCH'})

var_scen3 = xr.load_dataarray(f'{dir3}{varname}_scen.nc').sel(**sel_dict, drop=True)
var_base3 = xr.load_dataarray(f'{dir3}{varname}_hist.nc').sel(**base_dict).mean('year')
var3 = (var_scen3 - var_base3).sel(scen=sel_dict['scen']).squeeze().rename({'reg_IMAGE': 'COACCH'})

print('sel_dict:', sel_dict)
dims_median = ['config', 'data_LULCC', 'mod']

estimator = 'median'
errorbar = ('pi', 80)
markers = ['_', 'o', 's']
qt = 0.9

fig = plt.figure(figsize=(10, 8), dpi=300)
gs = gridspec.GridSpec(
    len(sel_dict['scen']), 3, figure=fig, 
    height_ratios=[1, 1, 1, 1], 
    width_ratios=[1, 1, 1], 
    wspace=0.1, hspace=0.1, 
    top=0.9, bottom=0.1, left=0.05, right=0.95
)
lvls1 = np.linspace(2, 14, 7)
lvls2 = np.linspace(0.2, 1.4, 7)
lvls3 = np.linspace(2, 14, 7)
alpha = [0.2, 0.4, 0.6, 0.8, 1.0]
qt = 0.9

for j in range(len(sel_dict['scen'])):
    var_base = var1.sel(scen=sel_dict['scen'][j]).quantile(qt, dim=dims_median).drop_sel(COACCH='World') 
    var_plot = var3.sel(scen=sel_dict['scen'][j]).quantile(qt, dim=dims_median).drop_sel(COACCH='World') - var_base
    ratio = var_plot / var_base * 100

    ax0 = fig.add_subplot(gs[j, 0], projection=ccrs.PlateCarree(central_longitude=0.0))
    ax0.text(-0.04, 1.02, chr(97 + j * 3), transform=ax0.transAxes, fontsize='large', fontweight='bold')
    ax0, cf0 = create_global_map(var_base, lvls1, 
                mask=map_mask, 
                axis='COACCH',
                ax=ax0,
                axis_label=['left', 'bottom'] if j == 3 else ['left'],
                cb_on=False, 
                contourf_kwargs={'cmap': 'YlOrRd', 'extend': 'both'}, 
                colorbar_kwargs={'extend': 'both', 'extendrect': False}
            )

    ax1 = fig.add_subplot(gs[j, 1], projection=ccrs.PlateCarree(central_longitude=0.0))
    ax1.text(-0.04, 1.02, chr(97 + j * 3 + 1), transform=ax1.transAxes, fontsize='large', fontweight='bold')
    ax1, cf1 = create_global_map(var_plot, lvls2,
                mask=map_mask, 
                axis='COACCH',
                ax=ax1,
                axis_label=['bottom'] if j == 3 else [],
                cb_on=False, 
                contourf_kwargs={'cmap': 'Reds',  'extend': 'both'}, 
                colorbar_kwargs={'extend': 'both', 'extendrect': False}
            )
    
    ax2 = fig.add_subplot(gs[j, 2], projection=ccrs.PlateCarree(central_longitude=0.0))
    ax2.text(-0.04, 1.02, chr(97 + j * 3 + 2), transform=ax2.transAxes, fontsize='large', fontweight='bold')
    ax2, cf2 = create_global_map(ratio, lvls3,
                mask=map_mask, 
                axis='COACCH',
                ax=ax2,
                axis_label=['right', 'bottom'] if j == 3 else ['right'],
                cb_on=False, 
                contourf_kwargs={'cmap': 'RdPu', 'extend': 'both'}, 
                colorbar_kwargs={'extend': 'both', 'extendrect': False}
            )
    
    text = f'{sel_dict["scen"][j] if sel_dict["scen"][j] != scens[5] else "NDCs"}'
    ax0.text(-0.2, 0.5, text, 
        fontsize='large', fontstyle='italic',
        va='center', ha='right', rotation=90,
        transform=ax0.transAxes
    )
    ax0.tick_params(axis='both', which='major', labelsize='medium')
    ax1.tick_params(axis='both', which='major', labelsize='medium')
    ax2.tick_params(axis='both', which='major', labelsize='medium')

cbar_ax0 = fig.add_axes([0.07, 0.05, 0.25, 0.02])  # [left, bottom, width, height]
cb0 = fig.colorbar(
    cf0, cax=cbar_ax0, 
    label=rf'Baseline cost (% of GDP)', 
    ticks=lvls1, 
    orientation='horizontal', 
    extend='both', extendrect=False
)
cb0.set_label(rf'Baseline cost (% of GDP)', fontsize='large')

cbar_ax1 = fig.add_axes([0.38, 0.05, 0.25, 0.02])
cb1 = fig.colorbar(
    cf1, cax=cbar_ax1, 
    label=rf'Additional cost (% of GDP)', 
    ticks=lvls2, 
    orientation='horizontal', 
    extend='both', extendrect=False
)
cb1.set_label(rf'Additional cost (% of GDP)', fontsize='large')

cbar_ax2 = fig.add_axes([0.69, 0.05, 0.25, 0.02])
cb2 = fig.colorbar(
    cf2, cax=cbar_ax2, 
    label=rf'Additional cost (% of baseline)', 
    ticks=lvls3, 
    orientation='horizontal', 
    extend='both', extendrect=False
)
cb2.set_label(rf'Additional cost (% of baseline)', fontsize='large')
# fig.suptitle(f'{qt*100:.0f}th percentile', fontsize='medium')
plt.show()


# In[ ]:


## 3.2 global map for damage cost

varname = 'dmg_tot'
sel_dict = {'year': 2100, 'scen': [scens_sorted[i] for i in [0, 2, 3, 5]], 'adaptation': True}
dir1 = 'results/noperma/constrained/'
dir3 = 'results/abrupt/constrained/'

base_year = get_baseline_year(varname)

var_scen1 = xr.load_dataarray(f'{dir1}{varname}_scen.nc')
sel_dict = {k: v for k, v in sel_dict.items() if k in var_scen1.dims}
var_scen1 = var_scen1.sel(**sel_dict, drop=True)
for attr in ['unit', 'units']:
    if attr in var_scen1.attrs:
        unit = f' ({var_scen1.attrs[attr]})'
        break
else:
    print(f'No unit attribute found in {varname}_scen.nc')
    unit = ''

var_base1 = xr.load_dataarray(f'{dir1}{varname}_hist.nc')
base_dict = {k:v for k, v in sel_dict.items() if k in var_base1.dims}
base_dict['year'] = slice(*base_year)
var_base1 = var_base1.sel(**base_dict).mean('year')
var1 = (var_scen1 - var_base1).sel(scen=sel_dict['scen']).squeeze()

var_scen3 = xr.load_dataarray(f'{dir3}{varname}_scen.nc').sel(**sel_dict, drop=True)
var_base3 = xr.load_dataarray(f'{dir3}{varname}_hist.nc').sel(**base_dict).mean('year')
var3 = (var_scen3 - var_base3).sel(scen=sel_dict['scen']).squeeze()

print('sel_dict:', sel_dict)
dims_median = ['config', 'data_LULCC', 'mod']

estimator = 'median'
errorbar = ('pi', 80)
markers = ['_', 'o', 's']
quantiles = [0.5, 0.9, 0.95]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(
    4, 3, figure=fig, 
    height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1], 
    wspace=0.1, hspace=0.1, bottom=0.15, left=0.05, right=0.95
)
axes = np.empty((3, 3), dtype=object)
lvls = np.linspace(0.2, 1, 5)
cmap = plt.cm.get_cmap('YlOrRd', 6)
colors = [cmap(i) for i in range(1, 6)]
alpha = [0.2, 0.4, 0.6, 0.8, 1.0]
rgba_map = np.zeros((len(lvls1) * len(lvls2), 4))
for i, qt in enumerate(quantiles):
    for j in range(4):
        if i == 0: 
            axis_label = ['left']
            if j == 3: axis_label.append('bottom')
        if i == 1: 
            if j == 3: 
                axis_label = ['bottom']
            else:
                axis_label = []
        if i == 2: 
            axis_label = ['right']
            if j == 3: axis_label.append('bottom')
        ax = fig.add_subplot(gs[j, i], projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.text(-0.04, 1.02, chr(97 + j * 3 + i), transform=ax.transAxes, fontsize='large', fontweight='bold')
        var_base = var1.sel(scen=sel_dict['scen'][j]).quantile(qt, dim=dims_median).drop_sel(reg_IMAGE='World') 
        var_plot = var3.sel(scen=sel_dict['scen'][j]).quantile(qt, dim=dims_median).drop_sel(reg_IMAGE='World') - var_base
        
        ax, cf = create_global_map(var_plot, lvls, 
                    mask=map_mask, 
                    axis='reg_IMAGE',
                    ax=ax, 
                    axis_label=axis_label,
                    cb_on=False, 
                    contourf_kwargs={'cmap': cmap, 'extend': 'both'}, 
                    colorbar_kwargs={'extend': 'both', 'extendrect': False}
                )
        ax.tick_params(axis='both', which='major', labelsize='small')
        if i == 0:
            text = f'{sel_dict["scen"][j] if sel_dict["scen"][j] != scens[5] else "NDCs"}'
            ax.text(-0.25, 0.5, text, 
                fontsize='small',
                va='center', ha='right', rotation=90,
                transform=ax.transAxes
            )
        if j == 0:
            text = f'{int(qt * 100)}th percentile'
            ax.text(
                0.5, 1.2, text, 
                fontsize='small',
                va='bottom', ha='center',
                transform=ax.transAxes
            )

        ax.grid(ls='--', alpha=0.5)

cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]
fig.colorbar(
    cf, cax=cbar_ax, 
    label=rf'Additional cost (% of GDP)', 
    ticks=lvls, 
    orientation='horizontal', 
    extend='both', extendrect=False
)
plt.show()

##################################################
##################################################

import os
import numpy as np
import pandas as pd
import xarray as xr
from core.paths import path_data

##====================================
## Damage functions for temperature
##====================================
## van der Wijst et al. (2023).
## https://doi.org/10.1038/s41558-023-01636-1
## baseline damage is calibrated for the 1986-2005 period

def load_dmg_temp(add_unc=True, **useless):
    ## use quadratic functions
    df = pd.read_excel(path_data + 'parameters/damage_coefficients-COACCH-WP4.xlsx', sheet_name='NoSLR-QuadraticQuantReg', skiprows=2)
    df = df.set_index(df.columns[0])
    ## only choose world and IMAGE regions
    df = df.loc[df.index.str.startswith('I_') | (df.index == 'World')]
    df.index.name = 'reg_IMAGE'
    df.index = df.index.str.replace('I_', '')

    Par = xr.Dataset.from_dataframe(df)
    Par.coords['unc_Norm'] = ['mean', 'std']
    Par.coords['quantile'] = [0.025, 0.05, 0.16, 0.25, 0.33, 0.5, 0.67, 0.75, 0.84, 0.95, 0.975]

    b1_mean = Par.b1.expand_dims({'unc_Norm':['mean']})
    b1_std = ((Par.b1_high - Par.b1_low) / 2).expand_dims({'unc_Norm':['std']})
    Par['b1_dmg_T'] = xr.concat([b1_mean, b1_std], dim='unc_Norm')
    b2_mean = Par.b2.expand_dims({'unc_Norm':['mean']})
    b2_std = ((Par.b2_high - Par.b2_low) / 2).expand_dims({'unc_Norm':['std']})
    Par['b2_dmg_T'] = xr.concat([b2_mean, b2_std], dim='unc_Norm')

    Par['f_dmg_T'] = xr.concat([Par[f'a (q={str(qt)})'].expand_dims({'quantile':[qt]}) for qt in Par.coords['quantile'].values], dim='quantile')
    Par = Par[['f_dmg_T', 'b1_dmg_T', 'b2_dmg_T']]

    Par = Par.transpose('reg_IMAGE', 'quantile', 'unc_Norm')
    return Par if add_unc else Par.sel(unc_Norm='mean', drop=True)


##====================================
## Damage functions for sea level rise
##====================================
def load_dmg_SLR(add_unc=True, **useless):
    ## SLR with adaptation
    df1 = pd.read_excel(path_data + 'parameters/damage_coefficients-COACCH-WP4.xlsx', sheet_name='SLR-Ad-LinearQuantReg', skiprows=2)
    df1 = df1.set_index(df1.columns[0])
    df1 = df1.loc[df1.index.str.startswith('I_') | (df1.index == 'World')]
    df1.index.name = 'reg_IMAGE'
    df1.index = df1.index.str.replace('I_', '')

    Par1 = xr.Dataset.from_dataframe(df1)
    Par1.coords['unc_Norm'] = ['mean', 'std']
    Par1.coords['quantile'] = [0.025, 0.05, 0.16, 0.25, 0.33, 0.5, 0.67, 0.75, 0.84, 0.95, 0.975]

    b1_mean = Par1.b1.expand_dims({'unc_Norm':['mean']})
    b1_std = ((Par1.b1_high - Par1.b1_low) / 2).expand_dims({'unc_Norm':['std']})
    Par1['b1_dmg_SLR'] = xr.concat([b1_mean, b1_std], dim='unc_Norm')

    Par1['f_dmg_SLR'] = xr.concat([Par1[f'a (q={str(qt)})'].expand_dims({'quantile':[qt]}) for qt in Par1.coords['quantile'].values], dim='quantile')
    Par1 = Par1[['f_dmg_SLR', 'b1_dmg_SLR']]
    Par1['b2_dmg_SLR'] = xr.zeros_like(Par1.b1_dmg_SLR)

    ## SLR without adaptation - Linear quantile regression
    df2 = pd.read_excel(path_data + 'parameters/damage_coefficients-COACCH-WP4.xlsx', sheet_name='SLR-NoAd-LinearQuantReg', skiprows=2)
    df2 = df2.set_index(df2.columns[0])
    df2 = df2.loc[df2.index.str.startswith('I_') | (df2.index == 'World')]
    df2.index.name = 'reg_IMAGE'
    df2.index = df2.index.str.replace('I_', '')

    Par2 = xr.Dataset.from_dataframe(df2)
    Par2.coords['unc_Norm'] = ['mean', 'std']
    Par2.coords['quantile'] = [0.025, 0.05, 0.16, 0.25, 0.33, 0.5, 0.67, 0.75, 0.84, 0.95, 0.975]

    b1_mean = Par2.b1.expand_dims({'unc_Norm':['mean']})
    b1_std = ((Par2.b1_high - Par2.b1_low) / 2).expand_dims({'unc_Norm':['std']})
    Par2['b1_dmg_SLR'] = xr.concat([b1_mean, b1_std], dim='unc_Norm')
    Par2['b2_dmg_SLR'] = xr.zeros_like(Par2.b1_dmg_SLR)

    Par2['f_dmg_SLR'] = xr.concat([Par2[f'a (q={str(qt)})'].expand_dims({'quantile':[qt]}) for qt in Par2.coords['quantile'].values], dim='quantile')
    Par2 = Par2[['f_dmg_SLR', 'b1_dmg_SLR', 'b2_dmg_SLR']]

    ## only keep parameters for INDIA, JAP, NAF, RSAS, SEAS, SSA, WEU
    Par2 = Par2.sel(reg_IMAGE=['INDIA', 'JAP', 'NAF', 'RSAS', 'SEAS', 'SSA', 'WEU'])

    ## SLR without adaptation - quadratic quantile regression
    df3 = pd.read_excel(path_data + 'parameters/damage_coefficients-COACCH-WP4.xlsx', sheet_name='SLR-NoAd-QuadraticQuantReg', skiprows=2)
    df3 = df3.set_index(df3.columns[0])
    df3 = df3.loc[df3.index.str.startswith('I_') | (df3.index == 'World')]
    df3.index.name = 'reg_IMAGE'
    df3.index = df3.index.str.replace('I_', '')

    Par3 = xr.Dataset.from_dataframe(df3)
    Par3.coords['unc_Norm'] = ['mean', 'std']
    Par3.coords['quantile'] = [0.025, 0.05, 0.16, 0.25, 0.33, 0.5, 0.67, 0.75, 0.84, 0.95, 0.975]

    b1_mean = Par3.b1.expand_dims({'unc_Norm':['mean']})
    b1_std = ((Par3.b1_high - Par3.b1_low) / 2).expand_dims({'unc_Norm':['std']})
    Par3['b1_dmg_SLR'] = xr.concat([b1_mean, b1_std], dim='unc_Norm')
    b2_mean = Par3.b2.expand_dims({'unc_Norm':['mean']})
    b2_std = ((Par3.b2_high - Par3.b2_low) / 2).expand_dims({'unc_Norm':['std']})
    Par3['b2_dmg_SLR'] = xr.concat([b2_mean, b2_std], dim='unc_Norm')

    Par3['f_dmg_SLR'] = xr.concat([Par3[f'a (q={str(qt)})'].expand_dims({'quantile':[qt]}) for qt in Par3.coords['quantile'].values], dim='quantile')
    Par3 = Par3.drop_sel(reg_IMAGE=['INDIA', 'JAP', 'NAF', 'RSAS', 'SEAS', 'SSA', 'WEU'])
    Par3 = Par3[['f_dmg_SLR', 'b1_dmg_SLR', 'b2_dmg_SLR']]

    Par = xr.merge([Par1.expand_dims({'adaptation':[True]}), xr.concat([Par2, Par3], dim='reg_IMAGE').expand_dims({'adaptation':[False]})])
    
    Par = Par.transpose('reg_IMAGE', 'adaptation', 'quantile', 'unc_Norm')
    return Par if add_unc else Par.sel(unc_Norm='mean', drop=True)

## wrapping all damage function parameters
def load_dmg_all(add_unc=True, **useless):
    '''
    Wrapper function to load all nitrogen parameters.

    Input:
    ------
    mod_region (str)        regional aggregation name
    recalibrate (bool)      whether to recalibrate all possible parameters;
                            WARNING: currently not working;
                            default = False

    Output:
    -------
    Par (xr.Dataset)        merged dataset
    '''

    print('loading damage function parameters')

    ## list of loading fuctions
    load_list = [
        load_dmg_temp, load_dmg_SLR
    ]

    ## return all
    return xr.merge([load(add_unc=add_unc) for load in load_list]).transpose('reg_IMAGE', 'adaptation', 'quantile', ...)
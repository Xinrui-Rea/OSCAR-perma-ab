##################################################
##################################################

import os
import numpy as np
import xarray as xr

## ========================
## Permafrost - abrupt thaw
## ========================

## permafrost carbon parameters
## (based on Turetsky et al., 2020)
def load_permafrost_abrupt(dyn_mode=True, par_from_paper=True, add_unc=True, **useless):
    
    ## initialization
    Par = xr.Dataset()
    if add_unc:
        Par.coords['unc_Norm'] = ['mean', 'std']
        Par.coords['unc_LogNorm'] = ['mean', 'std']

    ## ==============================
    ## A. upland thermokarst
    ## ==============================

    Par.coords['pf_thaw_up'] = ['Und', 'Act', 'Sta']
    Par.coords['pf_up_from'] = Par.coords['pf_thaw_up'].values
    Par.coords['pf_up_to'] = Par.coords['pf_thaw_up'].values
    Par.coords['pf_dyn'] = [True, False]

    ## initial areas
    Par['Apf_up_0'] = xr.DataArray([896350, 2275, 11375], dims=('pf_thaw_up',), attrs={'units': 'km2'})
    if add_unc:
        Apf_up_0_mean = Par['Apf_up_0']
        Apf_up_0_std = 0.4 * Par['Apf_up_0']
        Par['Apf_up_0'] = xr.concat(
            [Apf_up_0_mean.expand_dims({'unc_LogNorm':['mean']}), Apf_up_0_std.expand_dims({'unc_LogNorm':['std']})], 
            dim='unc_LogNorm'
        )

    ## coefficients of transition rates
    Par['a0_pf_up'] = xr.DataArray(np.zeros((2, 3, 3)), dims=('pf_dyn', 'pf_up_from', 'pf_up_to'))
    Par['a1_pf_up'] = xr.DataArray(np.zeros((2, 3, 3)), dims=('pf_dyn', 'pf_up_from', 'pf_up_to'))
    Par['a2_pf_up'] = xr.DataArray(np.zeros((2, 3, 3)), dims=('pf_dyn', 'pf_up_from', 'pf_up_to'))
    Par['a3_pf_up'] = xr.DataArray(np.zeros((2, 3, 3)), dims=('pf_dyn', 'pf_up_from', 'pf_up_to'))
    Par['a0_pf_up'].attrs['units'] = 'yr-1'
    Par['a1_pf_up'].attrs['units'] = 'K-1 yr-1'
    Par['a2_pf_up'].attrs['units'] = 'K-2 yr-1'
    Par['a3_pf_up'].attrs['units'] = 'K-3 yr-1'

    ## characteristic transition rates (yr-1)
    ## from Turetsky et al. (2020) Supplementary Table 1
    Par['a_pf_up'] = xr.DataArray([0.0002, 0.1, 0.02], dims=('pf_thaw_up',))

    ## set lower/upper limit to the transition rates
    Par['a_pf_up_max'] = xr.DataArray([0.03, 1, 1], dims=('pf_thaw_up', ))

    ## ------------------------------
    ## A1. Undisturbed -> Active
    ## ------------------------------
    ## static transition rates
    Par['a0_pf_up'].loc[{'pf_dyn': False, 'pf_up_from': 'Und', 'pf_up_to': 'Act'}] = 0.0002

    ## dynamic transition rates
    Par['a0_pf_up'].loc[{'pf_dyn': True, 'pf_up_from': 'Und', 'pf_up_to': 'Act'}] = -0.008880002476160125
    Par['a1_pf_up'].loc[{'pf_dyn': True, 'pf_up_from': 'Und', 'pf_up_to': 'Act'}] = 0.00948272317080108
    Par['a2_pf_up'].loc[{'pf_dyn': True, 'pf_up_from': 'Und', 'pf_up_to': 'Act'}] = -0.0018037063987277457
    Par['a3_pf_up'].loc[{'pf_dyn': True, 'pf_up_from': 'Und', 'pf_up_to': 'Act'}] = 0.00013044346305150992

    ## ------------------------------
    ## A2. Active -> Stabilized
    ## ------------------------------
    ## same transtion rates for static and dynamic stages
    Par['a0_pf_up'].loc[{'pf_up_from': 'Act', 'pf_up_to': 'Sta'}] = 0.1

    ## ------------------------------
    ## A3. Stabilized -> Undisturbed
    ## ------------------------------
    ## same transtion rates for static and dynamic stages
    Par['a0_pf_up'].loc[{'pf_up_from': 'Sta', 'pf_up_to': 'Und'}] = 0.02

    if par_from_paper:
    ## CO2 and CH4 emissions
    ## from Turetsky et al. (2020) Supplementary Table 1
        Par['ef_up_CO2'] = xr.DataArray(np.array([11, -95, 34]) * (-1.0E-6), 
                            dims=('pf_thaw_up',), attrs={'units': 'TgC km-2 yr-1'})
    else:
    ## CO2 emission factor from R code
    ## https://github.com/mturetsky/Abrupt-thaw-carbon-model/blob/master/upland-thermokarst_July10_2019_fme.Rmd
        Par['ef_up_CO2'] = xr.DataArray(np.array([10.5, -95, 34]) * (-1.0E-6), 
                            dims=('pf_thaw_up',), attrs={'units': 'TgC km-2 yr-1'})
        
    Par['ef_up_CH4'] = xr.DataArray(
        np.array([-4, -4, 0]) * (-1.0E-6), 
        dims=('pf_thaw_up',), attrs={'units': 'TgC km-2 yr-1'}
    )
    Par['ef_up_DOC'] = xr.DataArray(
        np.array([-2, -2188, -2]) * (-1.0E-6), 
        dims=('pf_thaw_up',), attrs={'units': 'TgC km-2 yr-1'}
    )
    Par['ef_up_ss'] = xr.DataArray(
        np.array([3.6, -99, 31]) * (-1.0E-6), 
        dims=('pf_thaw_up',), attrs={'units': 'TgC km-2 yr-1'}
    )
    Par['ef_up_ds'] = xr.DataArray(
        np.array([0, -2189., 1]) * (-1.0E-6),
        dims=('pf_thaw_up',), attrs={'units': 'TgC km-2 yr-1'}
    )
    if add_unc:
        ## normal distribution
        ef_up_CO2_mean = Par['ef_up_CO2']
        ef_up_CO2_std = [0.5, 0.25, 0.25] * Par['ef_up_CO2']
        Par['ef_up_CO2'] = xr.concat(
            [ef_up_CO2_mean.expand_dims({'unc_Norm':['mean']}), ef_up_CO2_std.expand_dims({'unc_Norm':['std']})], 
            dim='unc_Norm'
        )
        ef_up_ss_mean = Par['ef_up_ss']
        ef_up_ss_std = [0.1, 0.25, 0.34] * Par['ef_up_ss']
        Par['ef_up_ss'] = xr.concat(
            [ef_up_ss_mean.expand_dims({'unc_Norm':['mean']}), ef_up_ss_std.expand_dims({'unc_Norm':['std']})], 
            dim='unc_Norm'
        )
        ef_up_ds_mean = Par['ef_up_ds']
        ef_up_ds_std = [0., 0.45, 0.3] * Par['ef_up_ds']
        Par['ef_up_ds'] = xr.concat(
            [ef_up_ds_mean.expand_dims({'unc_Norm':['mean']}), ef_up_ds_std.expand_dims({'unc_Norm':['std']})], 
            dim='unc_Norm'
        )

        ## log-normal distribution
        ef_up_CH4_mean = Par['ef_up_CH4']
        ef_up_CH4_std = 0.4 * Par['ef_up_CH4']
        Par['ef_up_CH4'] = xr.concat(
            [ef_up_CH4_mean.expand_dims({'unc_LogNorm':['mean']}), ef_up_CH4_std.expand_dims({'unc_LogNorm':['std']})], 
            dim='unc_LogNorm'
        )

        ## normal distribution
        ef_up_DOC_mean = Par['ef_up_DOC']
        ef_up_DOC_std = [0.15, 0.3, 0.2] * Par['ef_up_DOC']
        Par['ef_up_DOC'] = xr.concat(
            [ef_up_DOC_mean.expand_dims({'unc_Norm':['mean']}), ef_up_DOC_std.expand_dims({'unc_Norm':['std']})], 
            dim='unc_Norm'
        )

    ## fraction of upland DOC that is mineralized as CO2
    Par['p_up_CO2'] = xr.DataArray([0.97, 2/3 * 0.95, 0], dims=('pf_thaw_up',), attrs={'units': '1'})
    ## fraction of upland DOC that is mineralized as CH4
    Par['p_up_CH4'] = xr.DataArray([0.03, 2/3 * 0.05, 0], dims=('pf_thaw_up',), attrs={'units': '1'})


    ## ==============================
    ## B. lowland mineral thermokarst
    ## ==============================

    Par.coords['pf_thaw_mi'] = ['Und', 'Act', 'Sta', 'Dra']
    Par.coords['pf_mi_from'] = Par.coords['pf_thaw_mi'].values
    Par.coords['pf_mi_to'] = Par.coords['pf_thaw_mi'].values
    Par.coords['soil_type'] = ['yedoma', 'non-yedoma']

    ## initial areas
    if par_from_paper:
    ## initial area of organic thermokarst from Turetsky et al. (2020) Supplementary Table 2
        Par['Apf_mi_0'] = xr.DataArray(
            [[171500, 2800, 44500, 57250], [602500, 3618, 116982, 200850]],
            dims=('soil_type', 'pf_thaw_mi'), 
            attrs={'units': 'km2'}
        )
    else:
    ## initial area of organic thermokarst from Turetsky et al. (2020) R code
    ## https://github.com/mturetsky/Abrupt-thaw-carbon-model/blob/master/lake-thermokarst_July10_2019_fme.Rmd
    ## lines 178-181, 203-206
        Par['Apf_mi_0'] = xr.DataArray(
            [[171750, 2370, 76630, 57250], [602550, 3618, 116982, 200850]], 
            dims=('soil_type', 'pf_thaw_mi'), 
            attrs={'units': 'km2'}
        )
    if add_unc:
        Apf_mi_0_mean = Par['Apf_mi_0']
        Apf_mi_0_std = 0.3 * Par['Apf_mi_0']
        Par['Apf_mi_0'] = xr.concat(
            [Apf_mi_0_mean.expand_dims({'unc_LogNorm':['mean']}), Apf_mi_0_std.expand_dims({'unc_LogNorm':['std']})], 
            dim='unc_LogNorm'
        )

    ## coefficients of transition rates
    Par['a0_pf_mi'] = xr.DataArray(np.zeros((2, 4, 4, 2)), dims=('pf_dyn', 'pf_mi_from', 'pf_mi_to', 'soil_type'))
    Par['a1_pf_mi'] = xr.DataArray(np.zeros((2, 4, 4, 2)), dims=('pf_dyn', 'pf_mi_from', 'pf_mi_to', 'soil_type'))
    Par['a2_pf_mi'] = xr.DataArray(np.zeros((2, 4, 4, 2)), dims=('pf_dyn', 'pf_mi_from', 'pf_mi_to', 'soil_type'))
    Par['a3_pf_mi'] = xr.DataArray(np.zeros((2, 4, 4, 2)), dims=('pf_dyn', 'pf_mi_from', 'pf_mi_to', 'soil_type'))
    Par['a0_pf_mi'].attrs['units'] = 'yr-1'
    Par['a1_pf_mi'].attrs['units'] = 'K-1 yr-1'
    Par['a2_pf_mi'].attrs['units'] = 'K-2 yr-1'
    Par['a3_pf_mi'].attrs['units'] = 'K-3 yr-1'

    ## characteristic transition rates (yr-1)
    ## from Turetsky et al. (2020) Supplementary Table 2
    Par['a_pf_mi'] = xr.DataArray([0.0033, 0.004, 0.0003, 0.0003], dims=('pf_thaw_mi',))

    ## set lower/upper limit to the transition rates
    Par['a_pf_mi_max'] = xr.DataArray(np.ones((4, 2)), dims=('pf_thaw_mi', 'soil_type'))

    ## ------------------------------
    ## B1. Undisturbed -> Active
    ## ------------------------------
    ## static transition rates
    ## https://github.com/mturetsky/Abrupt-thaw-carbon-model/blob/master/lake-thermokarst_July10_2019_fme.Rmd
    ## line 172
    Par['a0_pf_mi'].loc[{'pf_dyn':False, 'pf_mi_from': 'Und', 'pf_mi_to': 'Act'}] = 0.0002

    ## dynamic transition rates
    ## ! the dynamic transition rates for the two soil types are different in the R code
    ## lines 250-251, 278-279
    Par['a_pf_mi_max'].loc[{'pf_thaw_mi': 'Und', 'soil_type': 'yedoma'}] = 0.012
    Par['a_pf_mi_max'].loc[{'pf_thaw_mi': 'Und', 'soil_type': 'non-yedoma'}] = 0.007

    Par['a0_pf_mi'].loc[{'pf_dyn': True, 'pf_mi_from': 'Und', 'pf_mi_to': 'Act'}] = [0.0006093320630746738, 0.0021529326500057725]
    Par['a1_pf_mi'].loc[{'pf_dyn': True, 'pf_mi_from': 'Und', 'pf_mi_to': 'Act'}] = [0.0028448171049510165, 0.0012327540340279347]
    Par['a2_pf_mi'].loc[{'pf_dyn': True, 'pf_mi_from': 'Und', 'pf_mi_to': 'Act'}] = [-0.0005411119516971109, -0.00023448183636407902]
    Par['a3_pf_mi'].loc[{'pf_dyn': True, 'pf_mi_from': 'Und', 'pf_mi_to': 'Act'}] = [3.913304085992152e-05, 1.6957650470860372e-05]

    ## ------------------------------
    ## B2. Active -> Stabilized
    ## ------------------------------
    ## always constant
    ## lines 174, 199
    Par['a0_pf_mi'].loc[{'pf_dyn':False, 'pf_mi_from': 'Act', 'pf_mi_to': 'Sta'}] = 0.01
    ## lines 252, 280
    Par['a0_pf_mi'].loc[{'pf_dyn':True, 'pf_mi_from': 'Act', 'pf_mi_to': 'Sta'}] = 0.004

    ## ------------------------------
    ## B3. Stabilized -> Drained
    ## ------------------------------
    ## static transition rates
    ## line 175
    Par['a0_pf_mi'].loc[{'pf_dyn':False, 'pf_mi_from': 'Sta', 'pf_mi_to': 'Dra'}] = 0.001

    ## dynamic transition rates
    ## lines 250-251, 278-279
    Par['a_pf_mi_max'].loc[{'pf_thaw_mi': 'Sta'}] = 0.006

    Par['a0_pf_mi'].loc[{'pf_dyn': True, 'pf_mi_from': 'Sta', 'pf_mi_to': 'Dra'}] = -0.001482667537271174
    Par['a1_pf_mi'].loc[{'pf_dyn': True, 'pf_mi_from': 'Sta', 'pf_mi_to': 'Dra'}] = 0.0018965446661209778
    Par['a2_pf_mi'].loc[{'pf_dyn': True, 'pf_mi_from': 'Sta', 'pf_mi_to': 'Dra'}] = -0.0003607412862764562
    Par['a3_pf_mi'].loc[{'pf_dyn': True, 'pf_mi_from': 'Sta', 'pf_mi_to': 'Dra'}] = 2.6088693001206287e-05

    ## ------------------------------
    ## B4. Drained -> Undisturbed
    ## ------------------------------
    ## always constant
    Par['a0_pf_mi'].loc[{'pf_dyn':False, 'pf_mi_from': 'Dra', 'pf_mi_to': 'Und'}] = 0.0003
    if par_from_paper:
        Par['a0_pf_mi'].loc[{'pf_dyn':True, 'pf_mi_from': 'Dra', 'pf_mi_to': 'Und'}] = 0.0003
    else:
        Par['a0_pf_mi'].loc[{'pf_dyn':True, 'pf_mi_from': 'Dra', 'pf_mi_to': 'Und'}] = 0.000333333

    if par_from_paper:
    ## CO2 and CH4 emission factors
    ## from Turetsky et al. (2020) Supplementary Table 2
        Par['ef_mi_CO2'] = xr.DataArray(np.array(
            [[11, -450, -181, 26], [11, -149, -54, 30]]) * (-1.0E-6), 
            dims=('soil_type', 'pf_thaw_mi'), 
            attrs={'units': 'TgC km-2 yr-1'}
        )
        Par['ef_mi_CH4'] = xr.DataArray(
            np.array([[-5, -130, -10, -5], [-4, -38, -7, -2]]) * (-1.0E-6), 
            dims=('soil_type', 'pf_thaw_mi'), 
            attrs={'units': 'TgC km-2 yr-1'}
        )
    else:
    ## CO2 emission factors from R code
    ## https://github.com/mturetsky/Abrupt-thaw-carbon-model/blob/master/lake-thermokarst_July10_2019_fme.Rmd
        Par['ef_mi_CO2'] = xr.DataArray(
            np.array([
                [[10.5, -450, -181, 26.1], [10.5, -149, -54, 29.5]],
                [[10.5, -450, -181, 26], [10.5, -149, -54, 30]]
                ]) * (-1.0E-6), 
            dims=('pf_dyn', 'soil_type', 'pf_thaw_mi',), 
            attrs={'units': 'TgC km-2 yr-1'}
        )
        Par['ef_mi_CH4'] = xr.DataArray(
            np.array([
                [[-5, -130, -10, -5], [-4, -38, -7, -3]],
                [[-5, -130, -10, -5], [-5, -38, -7, -3]]
                ]) * (-1.0E-6), 
            dims=('pf_dyn', 'soil_type', 'pf_thaw_mi',), 
            attrs={'units': 'TgC km-2 yr-1'}
        )
        Par['ef_mi_DOC'] = xr.DataArray(
            np.array([
                [[-3, 0, 0, 0], [-2, 0, 0, 0]],
                [[-2.5, 0, 0, 0], [-3, 0, 0, 0]]
                ]) * (-1.0E-6), 
            dims=('pf_dyn', 'soil_type', 'pf_thaw_mi',), 
            attrs={'units': 'TgC km-2 yr-1'}
        )
    if add_unc:
        ## normal distribution
        ef_mi_CO2_mean = Par['ef_mi_CO2']
        ef_mi_CO2_std = xr.DataArray([[0.1, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4]], dims=('soil_type', 'pf_thaw_mi')) * Par['ef_mi_CO2']
        Par['ef_mi_CO2'] = xr.concat(
            [ef_mi_CO2_mean.expand_dims({'unc_Norm':['mean']}), ef_mi_CO2_std.expand_dims({'unc_Norm':['std']})], 
            dim='unc_Norm'
        )
        ## log-normal distribution
        ef_mi_CH4_mean = Par['ef_mi_CH4']
        ef_mi_CH4_std = 0.4 * Par['ef_mi_CH4']
        Par['ef_mi_CH4'] = xr.concat(
            [ef_mi_CH4_mean.expand_dims({'unc_LogNorm':['mean']}), ef_mi_CH4_std.expand_dims({'unc_LogNorm':['std']})], 
            dim='unc_LogNorm'
        )

    ## ==============================
    ## C. lowland organic thermokarst
    ## ==============================

    Par.coords['pf_thaw_or'] = ['Und', 'Act1', 'Act2', 'Bog', 'Fen']
    Par.coords['pf_or_from'] = Par.coords['pf_thaw_or'].values
    Par.coords['pf_or_to'] = Par.coords['pf_thaw_or'].values
    Par.coords['biom_type'] = ['boreal', 'tundra']
    
    ## initial areas
    if par_from_paper:
    ## initial area of organic thermokarst from Turetsky et al. (2020) Supplementary Table 3
        Par['Apf_or_0'] = xr.DataArray(
            [[523500, 52350, 52350, 209400, 209400], [193500, 19350, 19350, 77400, 77400]],
            dims=('biom_type', 'pf_thaw_or'), 
            attrs={'units': 'km2'}
        )
    else:
    ## initial area of organic thermokarst from Turetsky et al. (2020) R code
    ## https://github.com/mturetsky/Abrupt-thaw-carbon-model/blob/master/wetland-thermokarst_July10_2019_fme.Rmd
    ## lines 222-226, 275-279 
        Par['Apf_or_0'] = xr.DataArray(
            [[523500, 40000, 55350, 209400, 209400], [193500, 10000, 19350, 77400, 77400]],
            dims=('biom_type', 'pf_thaw_or'), 
            attrs={'units': 'km2'}
        )
    if add_unc:
        Apf_or_0_mean = Par['Apf_or_0']
        Apf_or_0_std = 0.35 * Par['Apf_or_0']
        Par['Apf_or_0'] = xr.concat(
            [Apf_or_0_mean.expand_dims({'unc_LogNorm':['mean']}), Apf_or_0_std.expand_dims({'unc_LogNorm':['std']})], 
            dim='unc_LogNorm'
        )

    ## coefficients of transition rates
    Par['a0_pf_or'] = xr.DataArray(np.zeros((2, 5, 5, 2)), dims=('pf_dyn', 'pf_or_from', 'pf_or_to', 'biom_type'))
    Par['a1_pf_or'] = xr.DataArray(np.zeros((2, 5, 5, 2)), dims=('pf_dyn', 'pf_or_from', 'pf_or_to', 'biom_type'))
    Par['a2_pf_or'] = xr.DataArray(np.zeros((2, 5, 5, 2)), dims=('pf_dyn', 'pf_or_from', 'pf_or_to', 'biom_type'))
    Par['a3_pf_or'] = xr.DataArray(np.zeros((2, 5, 5, 2)), dims=('pf_dyn', 'pf_or_from', 'pf_or_to', 'biom_type'))
    Par['a0_pf_or'].attrs['units'] = 'yr-1'
    Par['a1_pf_or'].attrs['units'] = 'K-1 yr-1'
    Par['a2_pf_or'].attrs['units'] = 'K-2 yr-1'
    Par['a3_pf_or'].attrs['units'] = 'K-3 yr-1'

    ## characteristic transition rates (yr-1)
    ## from Turetsky et al. (2020) Supplementary Table 3
    Par['a_pf_or'] = xr.DataArray(
        [[0.003, 0.005, 0.0066, 0.002, 0.001], [0.002, 0.005, 0.0066, 0.002, 0.001]], 
        dims=('biom_type', 'pf_thaw_or')
    )

    ## set lower/upper limit to the transition rates
    Par['a_pf_or_max'] = xr.DataArray(np.ones((5, 2)), dims=('pf_thaw_or', 'biom_type'))

    ## ------------------------------
    ## C1. Undisturbed -> Active 1/2
    ## ------------------------------
    ## static transition rates
    Par['a0_pf_or'].loc[{'pf_dyn': False, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'boreal'}] = 0.0009
    Par['a0_pf_or'].loc[{'pf_dyn': False, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'tundra'}] = 0.0007

    ## dynamic transition rates
    ## thaw rates between boreal and tundra biome types are different
    Par['a_pf_or_max'].loc[{'pf_thaw_or': 'Und', 'biom_type': 'boreal'}] = 0.006
    Par['a0_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'boreal'}] = 0.00209199971088766
    Par['a1_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'boreal'}] = 0.0009482723487305511
    Par['a2_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'boreal'}] = -0.0001803706463071699
    Par['a3_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'boreal'}] = 1.3044346688111554e-05

    Par['a_pf_or_max'].loc[{'pf_thaw_or': 'Und', 'biom_type': 'tundra'}] = 0.004
    Par['a0_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'tundra'}] = 0.001273599726897314
    Par['a1_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'tundra'}] = 0.0007586179127200741
    Par['a2_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'tundra'}] = -0.00014429652418021716
    Par['a3_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Und', 'pf_or_to': ['Act1', 'Act2'], 'biom_type': 'tundra'}] = 1.0435477786986533e-05

    ## ------------------------------
    ## C2. Active 1 -> Bog
    ## ------------------------------
    Par['a0_pf_or'].loc[{'pf_dyn': False, 'pf_or_from': 'Act1', 'pf_or_to': 'Bog'}] = 0.005

    ## different dynamic transition rates for boreal and tundra biome types
    Par['a_pf_or_max'].loc[{'pf_thaw_or': 'Act1', 'biom_type': 'boreal'}] = 0.017
    Par['a0_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'boreal'}] = 0.0013679987844384552
    Par['a1_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'boreal'}] = 0.0037930894254246035
    Par['a2_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'boreal'}] = -0.0007214825899607749
    Par['a3_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'boreal'}] = 5.217738698227155e-05

    Par['a_pf_or_max'].loc[{'pf_thaw_or': 'Act1', 'biom_type': 'tundra'}] = 0.014
    Par['a0_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'tundra'}] = 0.0022759992838302743
    Par['a1_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'tundra'}] = 0.002844816934172832
    Par['a2_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'tundra'}] = -0.0005411119167892692
    Par['a3_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'tundra'}] = 3.9133038776271785e-05
    
    ## ------------------------------
    ## C3. Active 2 -> Bog
    ## ------------------------------
    ## same transtion rates for static and dynamic stages and two biome types
    Par['a0_pf_or'].loc[{'pf_or_from': 'Act2', 'pf_or_to': 'Bog'}] = 0.0066

    ## ------------------------------
    ## C4. Bog -> Fen
    ## ------------------------------
    ## same transtion rates for static and dynamic stages and two biome types
    Par['a0_pf_or'].loc[{'pf_or_from': 'Bog', 'pf_or_to': 'Fen'}] = 0.002

    ## ------------------------------
    ## C5. Bog/Fen -> Undisturbed
    ## ------------------------------
    Par['a0_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': ['Bog', 'Fen'], 'pf_or_to': 'Und', 'biom_type': 'boreal'}] = 0.0005
    Par['a0_pf_or'].loc[{'pf_dyn': True, 'pf_or_from': ['Bog', 'Fen'], 'pf_or_to': 'Und', 'biom_type': 'tundra'}] = 0.001

    Par['a0_pf_or'].loc[{'pf_dyn': False, 'pf_or_from': ['Bog', 'Fen'], 'pf_or_to': 'Und'}] = 0.001

    if par_from_paper:
    ## CO2 and CH4 emission factors
    ## from Turetsky et al. (2020) Supplementary Table 4
        Par['ef_or_CO2'] = xr.DataArray(np.array([19, -270, -417, 53, 60]) * (-1.0E-6), 
                            dims=('pf_thaw_or',), attrs={'units': 'TgC km-2 yr-1'})
        Par['ef_or_CH4'] = xr.DataArray(np.array([0, -75, -13, -1, -34]) * (-1.0E-6), 
                            dims=('pf_thaw_or',), attrs={'units': 'TgC km-2 yr-1'})
    else:
    ## CO2 emission factors from R code
    ## https://github.com/mturetsky/Abrupt-thaw-carbon-model/blob/master/wetland-thermokarst_July10_2019_fme.Rmd
        Par['ef_or_CO2'] = xr.DataArray(
            np.array([[20.4, -270, -422, 37, 60], [19, -270, -417, 53, 60]]) * (-1.0E-6), 
            dims=('pf_dyn', 'pf_thaw_or'), attrs={'units': 'TgC km-2 yr-1'}
        )
        ##? in R code line 256, the bog_NEE = 1, which should be a typo?
        Par['ef_or_CH4'] = xr.DataArray(
            np.array([[0, -74.8, -13, -1, -33.5], [0, -74.8, -13, -1, -34]]) * (-1.0E-6), 
            dims=('pf_dyn', 'pf_thaw_or'), attrs={'units': 'TgC km-2 yr-1'}
        )
        Par['ef_or_DOC'] = xr.DataArray(
            np.array([0, 0, 0, 0, 0]) * (-1.0E-6), 
            dims=('pf_thaw_or',), attrs={'units': 'TgC km-2 yr-1'}
        )
        
    if add_unc:
        ## normal distribution
        ef_or_CO2_mean = Par['ef_or_CO2']
        ef_or_CO2_std = 0.4 * Par['ef_or_CO2']
        Par['ef_or_CO2'] = xr.concat(
            [ef_or_CO2_mean.expand_dims({'unc_Norm':['mean']}), ef_or_CO2_std.expand_dims({'unc_Norm':['std']})], 
            dim='unc_Norm'
        )

    ## ====================
    ## return the parameter
    ## ====================
    Par = Par.transpose('pf_thaw_up', 'pf_thaw_mi', 'pf_thaw_or', 'soil_type', 'biom_type', 'pf_up_from', 'pf_up_to', 'pf_mi_from', 'pf_mi_to', 'pf_or_from', 'pf_or_to', ...)
    
    return Par.sel(pf_dyn=True, drop=True) if dyn_mode else Par.sel(pf_dyn=False, drop=True)

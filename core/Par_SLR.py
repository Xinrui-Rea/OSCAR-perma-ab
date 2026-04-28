##################################################
##################################################

import os
import numpy as np
import xarray as xr
from core.paths import path_data

##================
## Sea level rise
##================

## based on Bossy, T., Gasser, T., & Ciais, P. (2022). Pathfinder v1.0.1
## https://github.com/tgasser/Pathfinder/blob/master/core_fct/fct_default.py
def load_SLR(recalibrate=False, ipcc='AR6', **useless):
    assert ipcc in ['AR5', 'AR6'], f'IPCC version {ipcc} not supported'

    param_unc = ['Lthx', 'Lgis1', 'Lgis3', 'Ggla1', 'Ggla3', 'ggla', 'Lais_smb', 'aais'] + ['lgla0', 'lgis0', 'lais0']
    param_unc_norm = ['lgla0', 'lgis0', 'lais0', 'Ggla1', 'aais']
    param_unc_lognorm = ['Lthx', 'Lgis1', 'Lgis3', 'Ggla3', 'ggla', 'Lais_smb']
    param_fix = ['Lgla', 'Lais'] + [] + ['tgla', 'tgis', 'tais']


    ## initialization
    Par = xr.Dataset()
    Par.coords['unc_Norm'] = ['mean', 'std']
    Par.coords['unc_LogNorm'] = Par.coords['unc_Norm'].values
    for var in param_unc_norm:
        Par[var] = np.nan * xr.zeros_like(Par['unc_Norm'], dtype=np.float64)
    for var in param_unc_lognorm:
        Par[var] = np.nan * xr.zeros_like(Par['unc_LogNorm'], dtype=np.float64)
    for var in param_fix:
        Par[var] = np.nan

    ## ! the unit here has been converted to conform with OSCAR
    ## linear thermosteric SLR
    ## AR6: (Fox-Kemper et al., 2021; doi:10.1017/9781009157896.011) (Section 9.2.4.1)
    ## AR5: (Kuhlbrodt & Gregory, 2012; doi:10.1029/2012GL052952) (CMIP5 value)
    if ipcc == 'AR6': 
        Par['Lthx'][:] = np.array([113, 13]) / 1E3
    elif ipcc == 'AR5': 
        Par['Lthx'][:] = np.array([110, 10]) / 1E3
    Par['Lthx'].attrs['units'] = 'mm ZJ-1'
    Par['Lthx'].attrs['bounds'] = (0., np.inf)


    ## characteristic times for ice components
    ## (Mengel et al., 2016; doi:10.1073/pnas.1500515113) (Table S1)
    ## note: derived assuming log-norm distribution and 90% range provided
    Par['tgla'] = 190. # [190., 62.] # from (98, 295)
    Par['tgis'] = 481. # [481., 292.] # from (99.7, 927)
    Par['tais'] = 2093. # [2093., 482.] # from (1350, 2910)
    Par['tgla'].attrs['units'] = Par['tgis'].attrs['units'] = Par['tais'].attrs['units'] = 'yr'


    ## holocene trends in ice components
    ## (Fox-Kemper et al., 2021; doi:10.1017/9781009157896.011) (Table 9.5)
    ## note: taken as earliest period available; likely overestimated but range increased (from 90% to 1-sigma)
    Par['lgla0'][:] = [0.58, 0.5*(0.82-0.34)]
    Par['lgis0'][:] = [0.33, 0.5*(0.47-0.18)]
    Par['lais0'][:] = [0.00, 0.5*(0.11+0.10)]
    Par['lgla0'].attrs['units'] = Par['lgis0'].attrs['units'] = Par['lais0'].attrs['units'] = 'mm'
    Par['lgla0'].attrs['bounds'] = Par['lgis0'].attrs['bounds'] = Par['lais0'].attrs['bounds'] = (-np.inf, np.inf)


    ## maximum contribution from glaciers
    ## (Fox-Kemper et al., 2021; doi:10.1017/9781009157896.011) (Section 9.6.3.2 & Table 9.5)
    Par['Lgla'] = 320. + np.round(67.2 - 7.5)
    Par['Lgla'].attrs['units'] = 'mm'


    ## equilibrium sensitivity of AIS
    ## (Church et al., 2013; doi:10.1017/CBO9781107415324.026) (Figure 13.14)
    Par['Lais'] = 1200.
    Par['Lais'].attrs['units'] = 'mm K-1'


    ## SLR sensitivity parameters
    ## load from Edwards et al. models
    with xr.open_dataset(path_data + 'parameters/param_slr.nc') as TMP: Par_tmp = TMP.load()
    ## take mean and unbiased std
    for var in ['Ggla1', 'Ggla3', 'ggla', 'Lgis1', 'Lgis3', 'Lais_smb', 'aais']:
        Par[var][:] = [Par_tmp[var].mean(), Par_tmp[var].std() * np.sqrt((len(Par_tmp[var]) - 1.) / ((len(Par_tmp[var]) - 1.5)))]
        Par[var].attrs['units'] = Par_tmp[var].units
    Par['Ggla1'].attrs['bounds'] = Par['aais'].attrs['bounds'] = (-np.inf, np.inf)
    Par['Ggla3'].attrs['bounds'] = Par['ggla'].attrs['bounds'] = Par['Lgis1'].attrs['bounds'] = Par['Lgis3'].attrs['bounds'] = Par['Lais_smb'].attrs['bounds'] = (0., np.inf)
    
    return Par

## This file contains utility functions and constants for the this project. 
## It includes physical constants, unit conversions, model and scenario definitions, plotting styles, and functions for standardizing scenario names and retrieving baseline years for specific variables.

##########################
## Constants
##########################

## physical constants and unit conversions
molecular_weight = {
    'CO2': 44.01, 
    'CH4': 16.04, 
    'N2O': 44.01, 
    'BC': 12.01, 
    'CO': 28.01, 
    'NH3': 17.03, 
    'NOX': 46.01, 
    'OC': 12.01, 
    'SO2': 64.07, 
    'C': 12.01, 
    'N': 14.01, 
    'S': 32.07
}

units_std = {
    'CO2': 'PgC yr-1', 
    'CH4': 'TgC yr-1',
    'BC': 'TgC yr-1',
    'CO': 'TgC yr-1',
    'OC': 'TgC yr-1',
    'N2O': 'TgN yr-1',
    'NH3': 'TgN yr-1',
    'NOX': 'TgN yr-1',
    'SO2': 'TgS yr-1',
    'VOC': 'Tg yr-1',
    'Xhalo': 'Gg yr-1'
}

molecular_scale = {
    'CO2': molecular_weight['C']/molecular_weight['CO2'],
    'CH4': molecular_weight['C']/molecular_weight['CH4'],
    'BC': molecular_weight['C']/molecular_weight['BC'],
    'CO': molecular_weight['C']/molecular_weight['CO'],
    'OC': molecular_weight['C']/molecular_weight['OC'],
    'N2O': 2*molecular_weight['N']/molecular_weight['N2O'],
    'NH3': molecular_weight['N']/molecular_weight['NH3'],
    'NOX': molecular_weight['N']/molecular_weight['NOX'],
    'SO2': molecular_weight['S']/molecular_weight['SO2'],
    'VOC': 1,
    'XHalo': 1
}

##########################
## Defaults
##########################

## models and scenarios
mods = ['GCAM 6.0 NGFS', 'MESSAGEix-GLOBIOM 2.0-M-R12-NGFS', 'REMIND-MAgPIE 3.3-4.8']
scens = ['Below 2°C', 'Current Policies', 'Delayed transition', 'Fragmented World', 
    'Low demand', 'Nationally Determined Contributions (NDCs)', 'Net Zero 2050'
    ]
scens_sorted = [
    'Current Policies', 'Fragmented World', 'Nationally Determined Contributions (NDCs)', 
    'Below 2°C', 'Delayed transition', 'Net Zero 2050', 'Low demand'
]

## plotting style
sim_colors = ["#947E7E", "#16A709", "#f2a310"]
mod_colors = ['#2dade9', "#c70c0c", "#F2DD24"]
scen_colors = {scen: color for scen, color in zip(scens_sorted, ["#C00F0F", "#D48805", "#EFCE28", "#78A553", "#67B0DA", "#1971A3", "#3E3370"])}

mod_ls = {mods[0]: (2, 3), mods[1]: (5, 2, 5, 2), mods[2]: (1, 1)}

mod_hatches = {mods[0]: '', mods[1]: '////', mods[2]: '....'}

##########################
## Functions
##########################

def standardize_scen_names(scenarios):
    mapping = {
        'Nationally Determined Contributions (NDCs)': 'NDCs',
        'Delayed transition': 'Delayed Transition',
        'Low demand': 'Low Demand'
    }
    
    return [mapping.get(s, s) for s in scenarios]

def get_baseline_year(varname):
    if varname in ['dmg_SLR', 'dmg_T', 'dmg_tot']: 
        return (1986, 2005)
    elif varname in ['D_Tg', 'D_Htot']:
        return (1850, 1900)
    elif varname in ['RF_CO2', 'D_CO2', 'D_N2O', 'D_CH4']:
        return (1750, 1750)
    else:
        raise ValueError(f'Unknown variable name: {varname}')

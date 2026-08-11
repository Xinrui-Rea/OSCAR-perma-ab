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

import os, sys, csv, warnings
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs

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

def aggreg_region(ds_in, mod_region, 
    weight_dict={}, 
    old_axis='reg_code', 
    new_axis='reg_code_new', 
    time_axis='year', 
    dir='input_data/regions/',
    debug=False):
    '''
    Function to aggregate data onto OSCAR regions. It uses dictionnaries mapping ISO regions to OSCAR regions defined in 'input_data/regions' by user.
    
    Input:
    ------
    ds_in (xr.Dataset)  input dataset to be aggregated
    mod_region (str)    name of regional aggregation (must be a valid option)
        
    Output:
    -------
    ds_out (xr.Dataset) output dataset

    Options:
    --------
    weight_dict (dict)  keys variables are weighted using values variables when aggregating; 
                        this is necessary for intensive variables (e.g. temperature that needs to be weighted by area); 
                        keys and values are names (str) of ds_in variables;
                        keys are variables to be weighted, values are weight variables;
                        default = {}
    old_axis (str)      name of regional axis that will be aggregated (must a dim of ds_in);
                        default = 'reg_code'
    new_axis (str)      name of new aggregated regional axis (must NOT be in ds_in, and will be in ds_out);
                        default = 'reg_code_new'
    time_axis (str)     name of time axis (to ensure it is first dim in ds_out);
                        default = 'year'
    dir (str)           path to directory containing regional mapping files;
                        default = 'input_data/regions/'
    debug (bool)        whether or not to print debug information
                        default = False
    '''
   
    ## check old axis in ds_in and new_axis not in ds_in
    assert old_axis in ds_in.coords and new_axis not in ds_in.coords
    ## check all weight variables in ds_in
    for key, val in weight_dict.items():
        if key not in ds_in.data_vars:
            raise KeyError(f'Weight variable "{key}" not found in dataset.')
        if val not in ds_in.data_vars:
            raise KeyError(f'Weight variable "{val}" not found in dataset.')

    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    warnings.filterwarnings('ignore')

    ## make deep copy to be safe
    ds_out = ds_in.copy(deep=True)

    ## region mapping files to be loaded
    list_load = [zou for zou in os.listdir(dir) if all([_  in zou for _ in ['dict', '.csv']])]

    ## load and create combined dictionary
    dico = {}
    for zou in list_load:
        with open(dir + zou) as f: TMP = np.array([line for line in csv.reader(f)])
        dico = {**dico, **{key:val for key, val in zip(TMP[1:,0], TMP[1:,TMP[0,:].tolist().index(mod_region)])}}

    ## load long region names
    with open(dir+'OSCAR_reg_names_crop.csv') as f: 
        for line in csv.reader(f):
            if line[0] == mod_region: 
                long_name = line[1:]
                break
        else:
            raise KeyError(f'Long names for region "{mod_region}" not found.')
    long_name = {name_pair.split(':')[0]: name_pair.split(':')[1] for name_pair in long_name if name_pair != ''}
    assert all([reg in long_name for reg in dico.values()]), 'Some long names are missing for the given region.'

    ## apply weights to weighted variables
    for key, val in weight_dict.items():
        ## deal with nan values in weight variable
        ds_out[val] = ds_out[val].fillna(0)
        ds_out[key] = ds_out[key] * ds_out[val]

    ## extract variables without regional axis
    ds_non = ds_out.drop([var for var in ds_out if old_axis in ds_out[var].dims] + [old_axis])
    ds_out = ds_out.drop([var for var in ds_out if old_axis not in ds_out[var].dims])

    ## new regional aggregation
    ds_out.coords[new_axis] = xr.DataArray([dico[reg] for reg in ds_out[old_axis].values], dims=old_axis)
    ds_out = ds_out.groupby(new_axis).sum(old_axis, keep_attrs=True, min_count=1)
    ds_out.coords[new_axis + '_name_'+mod_region] = xr.DataArray([long_name[reg] for reg in ds_out[new_axis].values], dims=new_axis)

    ## remove weights
    for key, val in weight_dict.items():
        ds_out[key] = xr.where(ds_out[val] != 0, ds_out[key] / ds_out[val], np.nan)

    ## merge with extracted variables
    ds_out = xr.merge([ds_out, ds_non])

    ## make sure time axis is first
    if time_axis in ds_out.coords: 
        ds_out = ds_out.transpose(time_axis,...)
    
    ## return
    return ds_out

def convert_reg_code(reg, region_from='reg_code', region_to='National', debug=False):
    '''
    Function to convert between different region codes
    Input:
    ------
    reg (str)           region to be disaggregated
                        
    Output:
    ------
    reg_new (list)      new list of regional code
    
    Options:
    --------
    region_from (str)   name of regional aggregation
                        default = 'reg_code'
    region_to (str)     name of regional aggregation
                        default = 'National'
    debug (bool)        whether or not to print debug information
                        default = False
    '''
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    ## region mapping files to be loaded
    list_load = [zou for zou in os.listdir('./input_data/regions/') if all([_  in zou for _ in ['dict', '.csv']])]
    
    ## load and create combined dictionary
    dico = {}
    for zou in list_load:
        with open('./input_data/regions/' + zou) as f: TMP = np.array([line for line in csv.reader(f)])
        if region_from in TMP[0,:].tolist() and region_to in TMP[0,:].tolist():
            for key, val in zip(TMP[1:,TMP[0,:].tolist().index(region_from)], TMP[1:,TMP[0,:].tolist().index(region_to)]):
                if reg == key: dico.setdefault(reg, []).append(val)
    try:
        reg_new = sorted(set(dico[reg]))
    except KeyError:
        print(f'Region {reg} not found in {region_from} to {region_to} mapping file.')
        return []
        
    return reg_new

## stacked plots
def plot_scens(varname, data1, data2=None, color=[None, None], style=None, estimator='pi', errorbar=90, ax=None):

    if data1.index.name == 'year':
        x_values = data1.index.unique()
    else:
        x_values = data1['year'].unique()

    ax.tick_params(axis='both', which='major', labelsize='small')

    sns.lineplot(
        data=data1, x='year', y=varname, 
        style=style, 
        err_style='bars', err_kws={'errorevery':10, 'elinewidth':2, 'capsize':5, 'capthick':3},
        estimator=estimator, errorbar=errorbar,
        color=color[0],
        dashes=mod_ls if style else None, 
        legend=False,
        ax=ax
    )

    ax.fill_between(
        x=x_values,
        y1=data1.groupby('year')[varname].mean() if estimator == 'mean' else data1.groupby('year')[varname].median(),
        y2=0,
        color=color[0],
        alpha=0.8,
    )

    if data2 is not None:
        sns.lineplot(
            data=data2, x='year', y=varname, 
            style=style, 
            err_style='bars', err_kws={'errorevery':10, 'elinewidth':2, 'capsize':5, 'capthick':3},
            estimator=estimator, errorbar=errorbar,
            color=color[1],
            dashes=mod_ls if style else None,
            legend=False,
            ax=ax
        )

        ax.fill_between(
            x=x_values,
            y1=data2.groupby('year')[varname].mean() if estimator == 'mean' else data2.groupby('year')[varname].median(),
            y2=0,
            color=color[1],
            alpha=0.4,
        )
        
    ax.set_xlabel('')
    ax.set_ylabel('')
    return ax

def plot_clipped(data, color, ax_target, method='kde', vals=[90, 95], reversed=False):
    ## make sure vals[1] > vals[0]
    assert len(vals) == 2, "vals should be a list of two percentiles"
    assert vals[1] > vals[0], "vals[1] should be greater than vals[0]"

    # calculate percentiles
    p1, p2 = np.percentile(data.dropna(), vals)
    
    # generate the KDE coordinates without plotting them yet
    fig, temp_ax = plt.subplots()
    if method == 'kde':
        if reversed:
            temp = sns.kdeplot(y=data, ax=temp_ax)
        else:
            temp = sns.kdeplot(x=data, ax=temp_ax)
    elif method == 'ecdf':
        if reversed:
            temp = sns.ecdfplot(y=data, ax=temp_ax)
        else:
            temp = sns.ecdfplot(x=data, ax=temp_ax)
    line = temp.lines[-1]
    x, y = line.get_data()
    plt.close()
    
    # create the mask for the specified percentiles
    if reversed:
        mask = (y >= p1) & (y <= p2)
    else:
        mask = (x >= p1) & (x <= p2)
    
    # plot ONLY the masked segment on the inset
    ax_target.plot(x[mask], y[mask], color=color, lw=2, label=f'{vals[0]}-{vals[1]}% segment')
    if method == 'ecdf': 
        if reversed:
            ax_target.set_xlim(vals[0]/100, vals[1]/100)
        else:
            ax_target.set_ylim(vals[0]/100, vals[1]/100)
    
## create global map for regional data
def create_global_map(var_in, levels,
        mask=None,
        axis='reg_code',
        crs=ccrs.PlateCarree(central_longitude=0.0),
        map_extent=[-180, 180, -90, 90],
        ax=None,
        title=None,
        draw_labels=False,
        cb_on=True,
        axis_label=['left', 'bottom'],
        contourf_kwargs={},
        colorbar_kwargs={},
        debug=False):
    '''
    Function to create a global map of a given variable
    
    Input:
    ------
    var_in (xr.DataArray)       1-D array, containing regional values to be plotted
    levels (np.array)           levels of contour map
        
    Output:
    -------
    ax (mpl.axes._axes.Axes)    axes containing the plot
    cf (QuadContourSet)         contour set of the plot

    Options:
    --------
    mask (xr.DataArray)         mask dataarray
                                default = None
    axis (str)                  regional axis
                                default = 'reg_code'
    region (str)                regional level
                                default = 'sub-national'
    crs (cartopy.crs)           coordinate reference system for the plot
                                default = ccrs.PlateCarree(central_longitude=0.0)
    map_extent (list)           map extent in the form [lon_min, lon_max, lat_min, lat_max]
                                default = [-180, 180, -90, 90]
    ax (mpl.axes._axes.Axes)    axes to draw plot on
                                default = None
    title (str)                 title of the plot
                                default = None
    draw_labels (boolean)       whether to draw grid labels
                                default = False
    cb_on (boolean)             whether to draw colorbar
                                default = False
    contourf_kwargs             keyword arguments control the contour plot
                                default = {}
    colorbar_kwargs             keyword arguments control the colorbar
                                default = {}
    debug (bool)                whether or not to print debug information
                                default = False
    '''
    
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    warnings.filterwarnings('ignore')

    ## check old axis in ds_in and new_axis not in ds_in
    assert var_in.ndim == 1 and len(var_in[axis]) == len(var_in), 'Input data must be 1-D array with length equal to the length of the given axis.'
    
    if mask is None:
        print('Please load mask dataarray before plotting regional data.')
        raise RuntimeError
    
    var = sum([np.nan * xr.zeros_like(mask.coords[dim], dtype=float) for dim in ['lat', 'lon']])
    for reg in var_in.coords[axis]:
        if var_in.loc[{axis:reg.item()}].notnull().sum() > 0:
            try:
                if axis != 'reg_code': reg_list = convert_reg_code(reg.item(), region_from=axis, region_to='reg_code', debug=debug)
                for reg_sub in (reg_list if axis != 'reg_code' else [reg]):
                    var = xr.where(mask.loc[{'reg_code': reg_sub}] > 0, var_in.loc[{axis:reg.item()}].values, var)
            except KeyError:
                continue
    
    var = var.fillna(np.nan)
    if var.notnull().sum() == 0:
        print('No valid data to plot.')
        return None, None
    else:
        try:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        except ImportError:
            print('"cartopy" libraries must be installed')
            return None, None
        finally:
            if debug: print('Plotting contour map ...')
            if ax is None:
                ax = plt.subplot(111, projection=crs)
            if debug: print(f'Using projection: {type(ax.projection)}')
            ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='#E6F7FF')
            ax.add_feature(ocean, zorder=0)
            
            if title is not None: ax.set_title(title)
            cf = ax.contourf(var.lon, var.lat, var, levels, transform=ccrs.PlateCarree(), zorder=4, **contourf_kwargs)

            ax.add_feature(cfeature.BORDERS, zorder=4, linewidth=0.15)
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), zorder=4, linewidth=0.15)
            
            if cb_on:
                cb_defaults = {'orientation': 'horizontal', 'aspect': 30, 
                            'shrink': 0.8, 'pad': 0.08, 'extend': 'both'}
                cb_defaults.update(colorbar_kwargs)
                cb = plt.colorbar(cf, **cb_defaults)

            ax.set_extent(map_extent, crs=ccrs.PlateCarree())
            
            is_polar = isinstance(crs, (ccrs.NorthPolarStereo, ccrs.SouthPolarStereo))
            if is_polar:
                if draw_labels:
                    gl = ax.gridlines(draw_labels=True, dms=True,
                                    linewidth=0.5, color='gray', alpha=0.5, 
                                    linestyle='--', zorder=3)
                    
                    # configure label positions for polar projection
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.left_labels = True
                    gl.bottom_labels = True
                    
                    # set formatters
                    gl.xformatter = LongitudeFormatter(zero_direction_label=False)
                    gl.yformatter = LatitudeFormatter()
                    
                    # style the labels
                    gl.xlabel_style = {'size': 'small', 'color': 'black'}
                    gl.ylabel_style = {'size': 'small', 'color': 'black'}
                
            if not is_polar:
                try:
                    ax.set_xticks(np.arange(-180, 180, 60), crs=ccrs.PlateCarree())
                    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
                except RuntimeError:
                    pass
                try:
                    ax.set_yticks(np.arange(-60, 90, 30), crs=ccrs.PlateCarree())
                    ax.yaxis.set_major_formatter(LatitudeFormatter())
                except RuntimeError:
                    pass
                
                # add gridlines with labels for non-polar
                if draw_labels:
                    gl = ax.gridlines(draw_labels=draw_labels, linewidth=0.5, 
                                    color='gray', alpha=0.5, zorder=3)
                    gl.ylocator = ticker.FixedLocator([val for val in np.arange(-60, 90, 30) 
                                                    if val > map_extent[2] and val < map_extent[3]])
                    gl.yformatter = LatitudeFormatter()
                    gl.xformatter = LongitudeFormatter(zero_direction_label=False)


            if not is_polar and len(axis_label) > 0:
                ax.spines[['left', 'right', 'top', 'bottom']].set_linewidth(0.5)
                for side in axis_label:
                    ax.spines[side].set_visible(True)

            if not is_polar:
                ax.tick_params(
                    left=False if 'left' not in axis_label else True, 
                    right=False if 'right' not in axis_label else True, 
                    top=False if 'top' not in axis_label else True, 
                    bottom=False if 'bottom' not in axis_label else True,
                    labelleft=True if 'left' in axis_label else False,
                    labelright=True if 'right' in axis_label else False,
                    labeltop=True if 'top' in axis_label else False,
                    labelbottom=True if 'bottom' in axis_label else False
                )
            else:
                # for polar projections, turn off all default tick labels
                ax.tick_params(
                    left=False, right=False, top=False, bottom=False,
                    labelleft=False, labelright=False, labeltop=False, labelbottom=False
                )
            return ax, cf
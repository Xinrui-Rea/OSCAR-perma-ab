##################################################
##################################################

import numpy as np
import xarray as xr

from core.cls_main import Model

##################################################
##   0. CONSTANTS
##################################################

## conversion factors
PgC_to_TgC = 1E3

##################################################
##  ABRUPT PERMAFROST THAW
##################################################

## initialize
def OSCAR_perma_ab(option='offline'):
    '''
    Options:
    ------
    option (str)        choose to run online or offline simulation

    '''
    
    if option == 'offline':
        from core.cls_main import Model
        model = Model('OSCAR_perma_ab')
    if option == 'online':
        from core.mod_process import OSCAR
        model = OSCAR.copy(add_name='_perma_ab')

    ## UPLAND TRANSITION RATES
    model.process(
        Out = 'r_pf_up', 
        In = ('D_Tg', ),
        Eq = lambda Var, Par: Eq__r_pf_up(Var, Par),
        unit = 'yr-1')

    def Eq__r_pf_up(Var, Par):
        r_pf_up = xr.where(
            Var.D_Tg > 0, 
            Par.a3_pf_up * Var.D_Tg ** 3 + Par.a2_pf_up * Var.D_Tg ** 2 + Par.a1_pf_up * Var.D_Tg + Par.a0_pf_up, 
            Par.a0_pf_up
        )
        r_pf_up.loc[{'pf_up_from': 'Und', 'pf_up_to': 'Act'}] = r_pf_up.loc[
            {'pf_up_from': 'Und', 'pf_up_to': 'Act'}
            ].clip(max=Par.a_pf_up_max.sel(pf_thaw_up='Und'))
        return r_pf_up

    ## METRIC: UPLAND THERMOKARST AREA CHANGE RATE
    model.process(
        Out = 'd_Apf_up',
        In = ('D_Apf_up', 'r_pf_up'),
        Eq = lambda Var, Par: Eq__d_Apf_up(Var, Par),
        unit = 'km2 yr-1')

    def Eq__d_Apf_up(Var, Par):
        ## ensure non-negative areas
        Apf_up = (Par.Apf_up_0 + Var.D_Apf_up).clip(min=0.)
        transitions = [
            ('Und', 'Act'),
            ('Act', 'Sta'),
            ('Sta', 'Und')
        ]
        results = []
        states = ['Und', 'Act', 'Sta']
        for state in states:
            inflows = []
            outflows = []
            for src, dst in transitions:
                if dst == state:  # inflow
                    inflows.append(Apf_up.sel(pf_thaw_up=src) * Var.r_pf_up.sel(pf_up_from=src, pf_up_to=dst, drop=True))
                elif src == state:  # outflow
                    outflows.append(Apf_up.sel(pf_thaw_up=src) * Var.r_pf_up.sel(pf_up_from=src, pf_up_to=dst, drop=True))
            delta = sum(inflows) - sum(outflows)
            results.append(delta.assign_coords(pf_thaw_up=state))
        return xr.concat(results, dim='pf_thaw_up')

    ## PROGNOSTIC: UPLAND THERMOKARST AREA CHANGE RELATIVE TO PRE-INDUSTRIAL
    model.process(
        Out = 'D_Apf_up',
        In = ('D_Apf_up', 'd_Apf_up'),
        DiffEq = lambda Var, Par: DiffEq__D_Apf_up(Var, Par),
        vLin = lambda Par: vLin__D_Apf_up(Par),
        unit = 'km2',
        core_dims = ['pf_thaw_up'])

    def DiffEq__D_Apf_up(Var, Par):
        return Var.d_Apf_up

    def vLin__D_Apf_up(Par):
        return Par.a_pf_up
    
    ## LOWLAND MINERAL TRANSITION RATES
    model.process(
        Out = 'r_pf_mi',
        In = ('D_Tg', ),
        Eq = lambda Var, Par: Eq__r_pf_mi(Var, Par),
        unit = 'yr-1')
    
    def Eq__r_pf_mi(Var, Par):
        r_pf_mi = xr.where(
            Var.D_Tg > 0, 
            Par.a3_pf_mi * Var.D_Tg ** 3 + Par.a2_pf_mi * Var.D_Tg ** 2 + Par.a1_pf_mi * Var.D_Tg + Par.a0_pf_mi, 
            Par.a0_pf_mi
        )
        ## set upper/lower limit to the transition rates
        r_pf_mi.loc[{'pf_mi_from': 'Und', 'pf_mi_to': 'Act', 'soil_type': 'yedoma'}] = r_pf_mi.loc[
            {'pf_mi_from': 'Und', 'pf_mi_to': 'Act', 'soil_type': 'yedoma'}
            ].clip(max=Par.a_pf_mi_max.sel(pf_thaw_mi='Und', soil_type='yedoma'))
        r_pf_mi.loc[{'pf_mi_from': 'Und', 'pf_mi_to': 'Act', 'soil_type': 'non-yedoma'}] = r_pf_mi.loc[
            {'pf_mi_from': 'Und', 'pf_mi_to': 'Act', 'soil_type': 'non-yedoma'}
            ].clip(max=Par.a_pf_mi_max.sel(pf_thaw_mi='Und', soil_type='non-yedoma'))
        r_pf_mi.loc[{'pf_mi_from': 'Sta', 'pf_mi_to': 'Dra'}] = r_pf_mi.loc[
            {'pf_mi_from': 'Sta', 'pf_mi_to': 'Dra'}
            ].clip(max=Par.a_pf_mi_max.sel(pf_thaw_mi='Sta', soil_type='yedoma'))
        return r_pf_mi

    ## METRIC: LOWLAND MINERAL THERMOKARST AREA CHANGE RATE
    model.process(
        Out = 'd_Apf_mi',
        In = ('D_Apf_mi', 'r_pf_mi'),
        Eq = lambda Var, Par: Eq__d_Apf_mi(Var, Par),
        unit = 'km2 yr-1')

    def Eq__d_Apf_mi(Var, Par):
        ## ensure non-negative area
        Apf_mi = (Par.Apf_mi_0 + Var.D_Apf_mi).clip(min=0.)
        transitions = [
            ('Und', 'Act'),
            ('Act', 'Sta'),
            ('Sta', 'Dra'),
            ('Dra', 'Und')
        ]
        results = []
        states = ['Und', 'Act', 'Sta', 'Dra']
        for state in states:
            inflows = []
            outflows = []
            for src, dst in transitions:
                if dst == state:  # inflow
                    inflows.append(Apf_mi.sel(pf_thaw_mi=src) * Var.r_pf_mi.sel(pf_mi_from=src, pf_mi_to=dst, drop=True))
                elif src == state:  # outflow
                    outflows.append(Apf_mi.sel(pf_thaw_mi=src) * Var.r_pf_mi.sel(pf_mi_from=src, pf_mi_to=dst, drop=True))
            delta = sum(inflows) - sum(outflows)
            results.append(delta.assign_coords(pf_thaw_mi=state))
        return xr.concat(results, dim='pf_thaw_mi')

    ## PROGNOSTIC: LOWLAND MINERAL THERMOKARST AREA CHANGE RELATIVE TO PRE-INDUSTRIAL
    model.process(
        Out = 'D_Apf_mi',
        In = ('D_Apf_mi', 'd_Apf_mi'),
        DiffEq = lambda Var, Par: DiffEq__D_Apf_mi(Var, Par),
        vLin = lambda Par: vLin__D_Apf_mi(Par),
        unit = 'km2',
        core_dims = ['pf_thaw_mi', 'soil_type'])

    def DiffEq__D_Apf_mi(Var, Par):
        return Var.d_Apf_mi
    
    def vLin__D_Apf_mi(Par):
        return Par.a_pf_mi

    ## LOWLAND ORGANIC TRANSITION RATES
    model.process(
        Out = 'r_pf_or',
        In = ('D_Tg', ),
        Eq = lambda Var, Par: Eq__r_pf_or(Var, Par),
        unit = 'yr-1')
    
    def Eq__r_pf_or(Var, Par):
        r_pf_or = xr.where(
            Var.D_Tg > 0, 
            Par.a3_pf_or * Var.D_Tg ** 3 + Par.a2_pf_or * Var.D_Tg ** 2 + Par.a1_pf_or * Var.D_Tg + Par.a0_pf_or, 
            Par.a0_pf_or
        )
        ## set upper/lower limit to the transition rates
        r_pf_or.loc[{'pf_or_from': 'Und', 'pf_or_to':['Act1', 'Act2'], 'biom_type': 'boreal'}] = r_pf_or.loc[
            {'pf_or_from': 'Und', 'pf_or_to':['Act1', 'Act2'], 'biom_type': 'boreal'}
            ].clip(max=Par.a_pf_or_max.sel(pf_thaw_or='Und', biom_type='boreal'))
        r_pf_or.loc[{'pf_or_from': 'Und', 'pf_or_to':['Act1', 'Act2'], 'biom_type': 'tundra'}] = r_pf_or.loc[
            {'pf_or_from': 'Und', 'pf_or_to':['Act1', 'Act2'], 'biom_type': 'tundra'}
            ].clip(max=Par.a_pf_or_max.sel(pf_thaw_or='Und', biom_type='tundra'))
        r_pf_or.loc[{'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'boreal'}] = r_pf_or.loc[
            {'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'boreal'}
            ].clip(max=Par.a_pf_or_max.sel(pf_thaw_or='Act1', biom_type='boreal'))
        r_pf_or.loc[{'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'tundra'}] = r_pf_or.loc[
            {'pf_or_from': 'Act1', 'pf_or_to': 'Bog', 'biom_type': 'tundra'}
            ].clip(max=Par.a_pf_or_max.sel(pf_thaw_or='Act1', biom_type='tundra'))
        return r_pf_or

    ## METRIC: LOWLAND ORGANIC THERMOKARST AREA CHANGE RATE
    model.process(
        Out = 'd_Apf_or',
        In = ('D_Apf_or', 'r_pf_or'),
        Eq = lambda Var, Par: Eq__d_Apf_or(Var, Par),
        unit = 'km2 yr-1')

    def Eq__d_Apf_or(Var, Par):
        ## ensure non-negative areas
        Apf_or = (Par.Apf_or_0 + Var.D_Apf_or).clip(min=0.)
        transitions = [
            ('Und', 'Act1'), ('Und', 'Act2'),
            ('Act1', 'Bog'),
            ('Act2', 'Bog'),
            ('Bog', 'Und'), ('Bog', 'Fen'),
            ('Fen', 'Und')
        ]
        results = []
        states = ['Und', 'Act1', 'Act2', 'Bog', 'Fen']
        for state in states:
            inflows = []
            outflows = []
            for src, dst in transitions:
                if dst == state:  # inflow
                    inflows.append(Apf_or.sel(pf_thaw_or=src) * Var.r_pf_or.sel(pf_or_from=src, pf_or_to=dst, drop=True))
                elif src == state:  # outflow
                    outflows.append(Apf_or.sel(pf_thaw_or=src) * Var.r_pf_or.sel(pf_or_from=src, pf_or_to=dst, drop=True))
            delta = sum(inflows) - sum(outflows)
            results.append(delta.assign_coords(pf_thaw_or=state))
        return xr.concat(results, dim='pf_thaw_or')

    ## PROGNOSTIC: LOWLAND ORGANIC THERMOKARST AREA CHANGE RELATIVE TO PRE-INDUSTRIAL
    model.process(
        Out = 'D_Apf_or',
        In = ('D_Apf_or', 'd_Apf_or'),
        DiffEq = lambda Var, Par: DiffEq__D_Apf_or(Var, Par),
        vLin = lambda Par: vLin__D_Apf_or(Par),
        unit = 'km2',
        core_dims = ['pf_thaw_or', 'biom_type'])

    def DiffEq__D_Apf_or(Var, Par):
        return Var.d_Apf_or
    
    def vLin__D_Apf_or(Par):
        return Par.a_pf_or

    ## ========================================
    ##     EMISSIONS
    ## ========================================
    ## DIAGNOSTIC: UPLAND THERMOKARST EMISSIONS
    model.process(
        Out = 'D_Epf_up_CO2',
        In = ('D_Apf_up', ),
        Eq = lambda Var, Par: Eq__D_Epf_up_CO2(Var, Par),
        unit = 'PgC yr-1')
    
    def Eq__D_Epf_up_CO2(Var, Par):
        return (Par.ef_up_CO2 + Par.p_up_CO2 * Par.ef_up_DOC) * Var.D_Apf_up / PgC_to_TgC
    
    model.process(
        Out = 'D_Epf_up_CH4',
        In = ('D_Apf_up', ),
        Eq = lambda Var, Par: Eq__D_Epf_up_CH4(Var, Par),
        unit = 'TgC yr-1')

    def Eq__D_Epf_up_CH4(Var, Par):
        return (Par.ef_up_CH4 + Par.p_up_CH4 * Par.ef_up_DOC) * Var.D_Apf_up

    ## DIAGNOSTIC: LOWLAND MINERAL THERMOKARST EMISSIONS
    model.process(
        Out = 'D_Epf_mi_CO2',
        In = ('D_Apf_mi', ),
        Eq = lambda Var, Par: Eq__D_Epf_mi_CO2(Var, Par),
        unit = 'PgC yr-1')
    
    def Eq__D_Epf_mi_CO2(Var, Par):
        return Par.ef_mi_CO2 * Var.D_Apf_mi / PgC_to_TgC
    
    model.process(
        Out = 'D_Epf_mi_CH4',
        In = ('D_Apf_mi', ),
        Eq = lambda Var, Par: Eq__D_Epf_mi_CH4(Var, Par),
        unit = 'TgC yr-1')

    def Eq__D_Epf_mi_CH4(Var, Par):
        return Par.ef_mi_CH4 * Var.D_Apf_mi

    ## DIAGNOSTIC: LOWLAND ORGANIC THERMOKARST EMISSIONS
    model.process(
        Out = 'D_Epf_or_CO2',
        In = ('D_Apf_or', ),
        Eq = lambda Var, Par: Eq__D_Epf(Var, Par),
        unit = 'PgC yr-1')
    
    def Eq__D_Epf(Var, Par):
        return Par.ef_or_CO2 * Var.D_Apf_or / PgC_to_TgC

    model.process(
        Out = 'D_Epf_or_CH4',
        In = ('D_Apf_or', ),
        Eq = lambda Var, Par: Eq__D_Epf_or_CH4(Var, Par),
        unit = 'TgC yr-1')
    
    def Eq__D_Epf_or_CH4(Var, Par):
        return Par.ef_or_CH4 * Var.D_Apf_or

    '''    
    model.process(
        Out = 'D_Fpf_up',
        In = ('D_Fpf_up', 'd_Apf_up'),
        DiffEq = lambda Var, Par: DiffEq__D_Fpf_up(Var, Par),
        unit = 'PgC yr-1',
        core_dims = ['pf_thaw_up'])

    def DiffEq__D_Fpf_up(Var, Par):
        return (Par.ef_up_ds + Par.ef_up_ss) * Var.d_Apf_up / PgC_to_TgC
    '''
    ## ===========================================
    ##     ALTER PROCESSES BASED ON SELECTED MODE
    ## ===========================================
    if option == 'offline':
        ## run only the aburupt permafrost processes
        pass
    else:
        ##! rewrite the total permafrost CO2 emissions
        for var in ['D_Epf_CO2', 'D_Epf_CH4', 'd_CO2', 'D_CO2', 'D_CH4', 'AF', 'kS']:
            model.__delitem__(var)

        model.process(
            Out = 'D_Epf_CO2',
            In = ('D_Epf_up_CO2', 'D_Epf_mi_CO2', 'D_Epf_or_CO2', 'D_Epf'),
            Eq = lambda Var, Par: Eq__D_Epf_CO2(Var, Par),
            unit = 'PgC yr-1')

        def Eq__D_Epf_CO2(Var, Par):
            return Var.D_Epf_up_CO2.sum('pf_thaw_up', min_count=1) + Var.D_Epf_mi_CO2.sum('pf_thaw_mi', min_count=1).sum('soil_type', min_count=1) + Var.D_Epf_or_CO2.sum('pf_thaw_or', min_count=1).sum('biom_type', min_count=1) + ((1 - Par.p_pf_CH4) * Var.D_Epf).sum('reg_pf', min_count=1)

        model.process(
            Out = 'D_Epf_CH4',
            In = ('D_Epf_up_CH4', 'D_Epf_mi_CH4', 'D_Epf_or_CH4', 'D_Epf'),
            Eq = lambda Var, Par: Eq__D_Epf_CH4(Var, Par),
            unit = 'TgC yr-1')

        def Eq__D_Epf_CH4(Var, Par):
            return Var.D_Epf_up_CH4.sum('pf_thaw_up', min_count=1) + Var.D_Epf_mi_CH4.sum('pf_thaw_mi', min_count=1).sum('soil_type', min_count=1) + Var.D_Epf_or_CH4.sum('pf_thaw_or', min_count=1).sum('biom_type', min_count=1) + (PgC_to_TgC * Par.p_pf_CH4 * Var.D_Epf).sum('reg_pf', min_count=1)

        ## METRIC: atmospheric growth rate of CO2
        model.process(
            Out = 'd_CO2', 
            In = ('Eff', 'D_Eluc', 'D_Epf_CO2', 'D_Fland', 'D_Focean', 'D_Foxi_CH4'), 
            Eq = lambda Var, Par: Eq__d_CO2(Var, Par), 
            unit='ppm yr-1')

        def Eq__d_CO2(Var, Par):
            return 1 / Par.a_CO2 * (Var.Eff.sum('reg_land', min_count=1) + Var.D_Eluc + Var.D_Epf_CO2 - Var.D_Fland - Var.D_Focean + Var.D_Foxi_CH4)
        
        ## PROGNOSTIC: atmospheric CO2
        model.process(
            Out = 'D_CO2', 
            In = ('D_CO2', 'd_CO2'), 
            DiffEq = lambda Var, Par: DiffEq__D_CO2(Var, Par), 
            vLin = lambda Par: vLin__D_CO2(Par), 
            unit='ppm')

        def DiffEq__D_CO2(Var, Par):
            return Var.d_CO2
        
        def vLin__D_CO2(Par):
            return 1E-18
        
        
        ## METRIC: airborne fraction
        model.process(
            Out = 'AF', 
            In = ('Eff', 'D_Eluc', 'D_Epf_CO2', 'D_Fland', 'D_Focean', 'D_Foxi_CH4'), 
            Eq = lambda Var, Par: Eq__AF(Var, Par), 
            unit='1')

        def Eq__AF(Var, Par):
            return 1 + (Var.D_Epf_CO2 - Var.D_Fland - Var.D_Focean + Var.D_Foxi_CH4) / (Var.Eff.sum('reg_land', min_count=1) + Var.D_Eluc)

        ## METRIC: carbon sinks rate
        model.process(
            Out = 'kS', 
            In = ('D_Epf_CO2', 'D_Fland', 'D_Focean', 'D_Foxi_CH4', 'D_CO2'), 
            Eq = lambda Var, Par: Eq__kS(Var, Par), 
            unit='yr-1')

        def Eq__kS(Var, Par):
            return 1 / Par.a_CO2 * -(Var.D_Epf_CO2 - Var.D_Fland - Var.D_Focean + Var.D_Foxi_CH4) / Var.D_CO2
        
        ## PROGNOSTIC: atmospheric CH4
        model.process(
            Out = 'D_CH4', 
            In = ('D_CH4', 'E_CH4', 'D_Ewet', 'D_Ebb', 'D_Epf_CH4', 'D_Fsink_CH4'), 
            DiffEq = lambda Var, Par: DiffEq__D_CH4(Var, Par), 
            vLin = lambda Par: vLin__D_CH4(Par), 
            unit='ppb')

        def DiffEq__D_CH4(Var, Par):
            return 1 / Par.a_CH4 * (Var.E_CH4.sum('reg_land', min_count=1) + Var.D_Ewet.sum('reg_land', min_count=1) + Var.D_Ebb.sel({'spc_bb':'CH4'}, drop=True).sum('bio_land', min_count=1).sum('reg_land', min_count=1) + Var.D_Epf_CH4 - Var.D_Fsink_CH4)

        def vLin__D_CH4(Par):
            return 1 / Par.w_t_OH / Par.t_OH_CH4 + 1 / Par.w_t_hv / Par.t_hv_CH4 + 1 / Par.t_soil_CH4 + 1 / Par.t_ocean_CH4

    ## RETURN model
    return model


##################################################
##  NO PERMAFROST THAW
##################################################
def OSCAR_noperma():

    from core.mod_process import OSCAR
    model = OSCAR.copy(add_name='_noperma')

    ## remove gradual permafrost thaw
    for var in ['f_resp_pf', 'D_pthaw_bar', 'd_pthaw', 'D_pthaw', 'D_Fthaw', 'D_Ethaw', 'D_Epf', 'D_Epf_CO2', 'D_Epf_CH4', 'D_Cfroz', 'D_Cthaw', 'd_CO2', 'D_CO2', 'D_CH4', 'AF', 'kS']:
        model.__delitem__(var)

    ## METRIC: atmospheric growth rate of CO2
    model.process(
        Out = 'd_CO2', 
        In = ('Eff', 'D_Eluc', 'D_Fland', 'D_Focean', 'D_Foxi_CH4'), 
        Eq = lambda Var, Par: Eq__d_CO2(Var, Par), 
        unit = 'ppm yr-1')

    def Eq__d_CO2(Var, Par):
        return 1 / Par.a_CO2 * (Var.Eff.sum('reg_land', min_count=1) + Var.D_Eluc - Var.D_Fland - Var.D_Focean + Var.D_Foxi_CH4)

    ## PROGNOSTIC: atmospheric CO2
    model.process(
        Out = 'D_CO2', 
        In = ('D_CO2', 'd_CO2'), 
        DiffEq = lambda Var, Par: DiffEq__D_CO2(Var, Par), 
        vLin = lambda Par: vLin__D_CO2(Par), 
        unit = 'ppm')

    def DiffEq__D_CO2(Var, Par):
        return Var.d_CO2

    def vLin__D_CO2(Par):
        return 1E-18

    ## METRIC: airborne fraction
    model.process(
        Out = 'AF', 
        In = ('Eff', 'D_Eluc', 'D_Fland', 'D_Focean', 'D_Foxi_CH4'), 
        Eq = lambda Var, Par: Eq__AF(Var, Par), 
        unit = '1')

    def Eq__AF(Var, Par):
        return 1 + (- Var.D_Fland - Var.D_Focean + Var.D_Foxi_CH4) / (Var.Eff.sum('reg_land', min_count=1) + Var.D_Eluc)


    ## METRIC: carbon sinks rate
    model.process(
        Out = 'kS', 
        In = ('D_Fland', 'D_Focean', 'D_Foxi_CH4', 'D_CO2'), 
        Eq = lambda Var, Par: Eq__kS(Var, Par), 
        unit = 'yr-1')

    def Eq__kS(Var, Par):
        return 1 / Par.a_CO2 * -(- Var.D_Fland - Var.D_Focean + Var.D_Foxi_CH4) / Var.D_CO2
    
    ## PROGNOSTIC: atmospheric CH4
    model.process(
        Out = 'D_CH4', 
        In = ('D_CH4', 'E_CH4', 'D_Ewet', 'D_Ebb', 'D_Fsink_CH4'), 
        DiffEq = lambda Var, Par: DiffEq__D_CH4(Var, Par), 
        vLin = lambda Par: vLin__D_CH4(Par), 
        unit = 'ppb')

    def DiffEq__D_CH4(Var, Par):
        return 1 / Par.a_CH4 * (Var.E_CH4.sum('reg_land', min_count=1) + Var.D_Ewet.sum('reg_land', min_count=1) + Var.D_Ebb.sel({'spc_bb':'CH4'}, drop=True).sum('bio_land', min_count=1).sum('reg_land', min_count=1) - Var.D_Fsink_CH4)

    def vLin__D_CH4(Par):
        return 1 / Par.w_t_OH / Par.t_OH_CH4 + 1 / Par.w_t_hv / Par.t_hv_CH4 + 1 / Par.t_soil_CH4 + 1 / Par.t_ocean_CH4

    return model
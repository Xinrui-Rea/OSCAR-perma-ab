##################################################
##################################################

import numpy as np
import xarray as xr

##################################################
##  DAMAGE FUNCTIONS
##################################################

## initialize
def OSCAR_dmg(option='offline'):
    '''
    Options:
    ------
    option (str)        choose to run online or offline simulation

    '''
    
    if option == 'offline':
        from core.cls_main import Model
        model = Model('OSCAR_dmg')
    if option == 'online':
        from core.OSCAR_SLR import OSCAR_SLR
        model = OSCAR_SLR(option='offline').copy(add_name='_dmg')

    ## DIAGNOSTIC
    model.process(
        Out = 'dmg_SLR',
        In = ('D_Htot', ),
        Eq = lambda Var, Par: Eq__dmg_SLR(Var, Par),
        unit = '%')

    def Eq__dmg_SLR(Var, Par):
        return (Par.b1_dmg_SLR * (Var.D_Htot / 1E3 ) + Par.b2_dmg_SLR * (Var.D_Htot / 1E3 ) ** 2) * Par.f_dmg_SLR
    
    model.process(
        Out = 'dmg_T',
        In = ('D_Tg', ),
        Eq = lambda Var, Par: Eq__dmg_T(Var, Par),
        unit = '%')

    def Eq__dmg_T(Var, Par):
        return (Par.b1_dmg_T * Var.D_Tg + Par.b2_dmg_T * Var.D_Tg**2) * Par.f_dmg_T
    
    model.process(
        Out = 'dmg_tot',
        In = ('dmg_SLR', 'dmg_T', ),
        Eq = lambda Var, Par: Eq__dmg_tot(Var, Par),
        unit = '%')

    def Eq__dmg_tot(Var, Par):
        return Var.dmg_SLR + Var.dmg_T

    return model
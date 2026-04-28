##################################################
##################################################

import numpy as np
import xarray as xr

from core.cls_main import Model

##################################################
##   0. CONSTANTS
##################################################

## conversion factors

##################################################
##   SEA LEVEL RISE
##################################################

## initialize
def OSCAR_SLR(option='offline'):
    '''
    Options:
    ------
    option (str)        choose to run online or offline simulation

    '''
    
    if option == 'offline':
        from core.cls_main import Model
        model = Model('OSCAR_SLR')
    if option == 'online':
        from core.mod_process import OSCAR
        model = OSCAR.copy(add_name='_SLR')

    ## DIAGNOSTIC: THERMAL EXPANSION
    model.process(
        Out = 'D_Hthx',
        In = ('D_OHC',),
        Eq = lambda Var, Par: Eq__D_Hthx(Var, Par),
        unit = 'mm')

    def Eq__D_Hthx(Var, Par):
        return Par.Lthx * Var.D_OHC
    
    ## PROGNOSTIC: GREENLAND ICE SHEET
    model.process(
        Out = 'D_Hgis',
        In = ('D_Hgis', 'D_Tg'),
        DiffEq = lambda Var, Par: DiffEq__D_Hgis(Var, Par),
        vLin = lambda Par: vLin__D_Hgis(Par),
        unit = 'mm')

    def DiffEq__D_Hgis(Var, Par):
        return Par.lgis0 + (Par.Lgis1 * Var.D_Tg + Par.Lgis3 * Var.D_Tg**3 - Var.D_Hgis) / Par.tgis

    def vLin__D_Hgis(Par):
        return 1 / Par.tgis

    ## PROGNOSTIC: GLACIER SEA LEVEL RISE
    model.process(
        Out = 'D_Hgla',
        In = ('D_Hgla', 'D_Tg'),
        DiffEq = lambda Var, Par: DiffEq__D_Hgla(Var, Par),
        # vLin = lambda Par: vLin__D_Hgla(Par),
        unit = 'mm')

    def DiffEq__D_Hgla(Var, Par):
        return Par.lgla0 + (Par.Lgla * (1. - np.exp(-Par.Ggla1 * Var.D_Tg - Par.Ggla3 * Var.D_Tg**3)) - Var.D_Hgla) / Par.tgla * np.exp(Par.ggla * Var.D_Tg)

    def vLin__D_Hgla(Par):
        return 1 / Par.tgla


    ## PROGNOSTIC: ANTARCTIC ICE SHEET
    ## include surface mass balance and solid ice discharge
    model.process(
        Out = 'D_Hais_smb',
        In = ('D_Hais_smb', 'D_Tg'),
        DiffEq = lambda Var, Par: DiffEq__D_Hais_smb(Var, Par),
        vLin = lambda Par: vLin__D_Hais_smb(Par),
        unit = 'mm'
    )

    def DiffEq__D_Hais_smb(Var, Par):
        return -Par.Lais_smb * Var.D_Tg

    def vLin__D_Hais_smb(Par):
        return 1.0E-9

    model.process(
        Out = 'D_Hais_sid',
        In = ('D_Hais_sid', 'D_Tg'),
        DiffEq = lambda Var, Par: DiffEq__D_Hais_sid(Var, Par),
        vLin = lambda Par: vLin__D_Hais_sid(Par),
        unit = 'mm')

    def DiffEq__D_Hais_sid(Var, Par):
        return Par.lais0 + (Par.Lais * Var.D_Tg - Var.D_Hais_sid) / Par.tais * (1 + Par.aais * Var.D_Hais_sid)

    def vLin__D_Hais_sid(Par):
        return 1. / Par.tais
    
    model.process(
        Out = 'D_Hais',
        In = ('D_Hais_smb', 'D_Hais_sid'),
        Eq = lambda Var, Par: Eq__D_Hais(Var, Par),
        unit = 'mm')

    def Eq__D_Hais(Var, Par):
        return Var.D_Hais_smb + Var.D_Hais_sid

    ## DIAGNOSTIC: TOTAL SEA LEVEL RISE
    model.process(
        Out = 'D_Htot',
        In = ('D_Hthx', 'D_Hgla', 'D_Hgis', 'D_Hais'),
        Eq = lambda Var, Par: Eq__D_Htot(Var, Par),
        unit = 'mm')

    def Eq__D_Htot(Var, Par):
        return Var.D_Hthx + Var.D_Hgla + Var.D_Hgis + Var.D_Hais
       
    return model
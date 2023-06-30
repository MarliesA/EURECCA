# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:22:39 2021

@author: marliesvanderl
"""
import pandas as pd
import numpy as np

def read_knmi_uurgeg(knmiFile,stationNumber):

    headerLines = 0
    with open(knmiFile) as fp:
                for line in (fp):
                    # pdb.set_trace()
                    if line[0]!='#':
                        headerLines+=1
                        continue
                    else:
                        break         
                    
    knmi = pd.read_csv(knmiFile,
                       skip_blank_lines = False, 
                       header =headerLines,
                       skipinitialspace=True)     
    # knmi = knmi[knmi['# STN'].notna()]
    knmi = knmi[knmi['# STN']==stationNumber] #make sure only the Kooij data is in there
    
    t0 = pd.to_datetime(knmi.iloc[0]['YYYYMMDD'],format='%Y%m%d')+pd.Timedelta('{}H'.format(knmi.iloc[0]['HH']))
    t = pd.date_range(t0.to_datetime64(),periods=len(knmi),freq='1H')
    # if len(t)!=len(pa):
    #     print('reconstruct time array line by line. This is much more work!')
    #     t = [pd.to_datetime(ix,format='%Y%m%d')+pd.Timedelta('{}H'.format(ih)) for ix,ih in zip(date,h)]
    knmi['t']=t
    knmi.set_index('t',inplace=True)
    
    variables = knmi.keys()
    if 'P' in variables:
        knmi['P'] = knmi['P']*10 # change 0.1hPa to Pascal
    if 'FH' in variables:
        knmi['FH'] = knmi['FH']/10 #to 0.1 m/s to m/s
    if 'FF' in variables:
        knmi['FF'] = knmi['FF']/10 #to 0.1 m/s to m/s
    if 'FX' in variables:
        knmi['FX'] = knmi['FX']/10 #to 0.1 m/s to m/s       
    if 'T' in variables:
        knmi['T'] = knmi['T']/10 #to 0.1 deg to deg
    if 'T10N' in variables:
        knmi['T10N'] = knmi['T10N']/10 #to 0.1 deg to deg
    if 'TD' in variables:
        knmi['TD'] = knmi['TD']/10 #to 0.1 deg to deg   
    if 'SQ' in variables:
        knmi['SQ'] = knmi['SQ']/10 #to 0.1 uur to uur
    if 'DR' in variables:
        knmi['DR'] = knmi['DR']/10 #to 0.1 uur to uur   
    if 'RH' in variables:
        knmi['RH'] = knmi['RH']/10 #to 0.1 mm to mm
    if 'DD' in variables:
        knmi.loc[knmi.DD==990,'DD']=np.nan
       
                
    
    return knmi
        
def load_uurgeg_from_knmi(filePath,stationNumber = 258, variables = 'all'):
    '''
    
    reads the text file and casts data in a dataframe in the unit of Pascals
    example usage:
    D = load_uurgeg_from_knmi(filePath,variables=['FH','FF','FX'])
    Parameters
    ----------
    filePath : TYPE
        DESCRIPTION.
    stationNumber : TYPE, optional
        DESCRIPTION. The default is 258.
    variables : TYPE, optional
        DESCRIPTION. The default is 'all'.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    '''
    

    

            
    hashHasPassed = False; D=[]
    with open(filePath) as fp:
        for line in (fp):
            if not hashHasPassed and not line[0]=='#':
                continue
            if not hashHasPassed and line[0]=='#':
                variableNames = line[1:].split(',')
                variableNames = [x.strip() for x in variableNames]
                
                hashHasPassed = True
            elif hashHasPassed and not line=='\n':
               x =line.split(',')
               x = [ix.strip() for ix in x]
               x = [0 if len(ix)==0 else float(ix) for ix in x]
               
               data= {};
               for ix,key in enumerate(variableNames):
                   data[key] = x[ix]
               D.append(data)
               
    D = pd.DataFrame(D)   
    
    t0 = pd.to_datetime(D.iloc[0]['YYYYMMDD'],format='%Y%m%d')+pd.Timedelta('{}H'.format(D.iloc[0]['HH']))
    t = pd.date_range(t0,periods=len(D),freq='1H')
    D['t'] = t
    D = D.set_index('t')
    
    tend = pd.to_datetime(D.iloc[-1]['YYYYMMDD'],format='%Y%m%d')+pd.Timedelta('{}H'.format(D.iloc[-1]['HH']))
    if not tend==t[-1]:
        print('reconstruct time array line by line. This is much more work!')
        t = [pd.to_datetime(ix,format='%Y%m%d')+pd.Timedelta('{}H'.format(ih)) for ix,ih in zip(D['YYYYMMDD'],D['HH'])]    
    
    D = D[D['STN']==stationNumber]
    
    if 'P' in variables:
        D['P'] = D['P']*10 # change 0.1hPa to Pascal
    if 'FH' in variables:
        D['FH'] = D['FH']/10 #to 0.1 m/s to m/s
    if 'FF' in variables:
        D['FF'] = D['FF']/10 #to 0.1 m/s to m/s
    if 'FX' in variables:
        D['FX'] = D['FX']/10 #to 0.1 m/s to m/s       
    if 'T' in variables:
        D['T'] = D['T']/10 #to 0.1 deg to deg
    if 'T10N' in variables:
        D['T10N'] = D['T10N']/10 #to 0.1 deg to deg
    if 'TD' in variables:
        D['TD'] = D['TD']/10 #to 0.1 deg to deg   
    if 'SQ' in variables:
        D['SQ'] = D['SQ']/10 #to 0.1 uur to uur
    if 'DR' in variables:
        D['DR'] = D['DR']/10 #to 0.1 uur to uur   
    if 'RH' in variables:
        D['RH'] = D['RH']/10 #to 0.1 mm to mm
    if 'DD' in variables:
        D.loc[D.DD==990,'DD']=np.nan
    if not variables=='all':
        D = D[variables]       
                
    return D




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:26:06 2024

@author: mc276490

This script aims to provide functions to translate ADDA outputs, DDscat outputs
and DustEM inputs into Polaris input files (Efficiencies, Scattering matrixes,
                                            Calorimetry.)

"""


import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import time
from scipy import signal
import os
from scipy.spatial.transform import Rotation 
from datetime import datetime
import shutil
import copy

# Considering average over all directions (sphere or orientationally avg part)
# For one particle (so one aeff only)
# Mainly the same as Utilitary.read_q
# XXX DDsat has wavelength and aeff in um.


 
def integ_dust_size(quantity,a_eff,a_min,a_max):
    
    # This method comes from Polaris.
    
    nr_of_dust_species = len(a_eff)
    res = 0

    # If maximum grain size is smaller than the minimum grain size -> return zero
    if(a_max < a_min):
        return res

    # Return the quantity value if only one size is used
    if(nr_of_dust_species == 1):
        return quantity[0]

    for i in range(1,nr_of_dust_species):
        m = (quantity[i] - quantity[i - 1]) / (a_eff[i] - a_eff[i - 1])
        if(a_eff[i - 1] >= a_min and a_eff[i] < a_max):

            dx = a_eff[i] - a_eff[i - 1]
            y0 = quantity[i - 1]
        
        elif(a_eff[i - 1] < a_min and a_eff[i] >= a_max):
        
            dx = a_max - a_min
            y0 = quantity[i - 1] + (a_min - a_eff[i - 1]) * m
            if(dx == 0):
                return y0
        
        elif(a_eff[i] >= a_min and a_eff[i - 1] < a_min):
        
            dx = a_eff[i] - a_min
            y0 = quantity[i]
            m *= -1
            
        elif(a_eff[i - 1] < a_max and a_eff[i] >= a_max):
            dx = a_max - a_eff[i - 1]
            y0 = quantity[i - 1]
        
        else:
            continue

        res += y0 * dx + 0.5 * dx * dx * m
    

    return res

def read_Q_Polaris(path):
    with open(path,'r') as f:
        lines = f.readlines()
        lines_ok = [line.split() for line in lines if not line.lstrip().startswith('#') and not not line.split()]
        idx_ok = [i for i,line in enumerate(lines) if not line.lstrip().startswith('#') and not not line.split()]
    # Read sizes (third uncommented line)
    aeff = np.array(lines_ok[2],dtype = 'float')
    # Read waves (fourth uncommented line)    
    wave = np.array(lines_ok[3],dtype = 'float')
    # Read Q (all the rest by block of wavelegth; a_eff1,a_eff2...  )
    Q = np.array(lines_ok[4:],dtype = 'float')
    if lines[idx_ok[4]-1].lstrip()[1] == 'Q' :
        Q_header = lines[idx_ok[4]-1].lstrip()[1:].split()
    elif lines[idx_ok[4]-2].lstrip()[1] == 'Q' :
        Q_header = lines[idx_ok[4]-2].lstrip()[1:].split()
    index = pd.MultiIndex.from_product([wave,aeff],names =['wave','aeff'])
    df_Q = pd.DataFrame(Q,columns=Q_header,index=index)
    df_Q['Qext'] = (df_Q['Qext1'] +df_Q['Qext2'])/2
    df_Q['Qabs'] = (df_Q['Qabs1'] +df_Q['Qabs2'])/2
    df_Q['Qsca'] = (df_Q['Qsca1'] +df_Q['Qsca2'])/2
    return df_Q
    
    
def read_Q_DDscat(path):
    """

    Parameters
    ----------
    path : str, path to file that is read

    Returns
    -------
    data : pandas dataframe, table containing the wavelength, aeff and corresponding q etc. Names are in headers.
    aeff and wavelength are in meters

    """
    with open(path+'/qtable','r') as f:
        lines = f.readlines()
    
    first = lines[0].split()[0]
    i=0
    while first != 'aeff':
        i +=1
        first = lines[i].split()[0]
    data = pd.read_fwf(path+'/qtable',skiprows=i+1,widths=[11,11,11,11,11,12,11,11,8],names=lines[i].split()) 
    
    data = data.rename(columns = {'Q_ext':'Qext','Q_sca':'Qsca','Q_abs':'Qabs','g(1)=<cos>':'g0','Wave':'wave'})
    # microns to meters
    data['wave'] = data['wave']*10**(-6)
    data['aeff'] = data['aeff']*10**(-6)
    return data


# Considering average over all directions (sphere or orientationally avg part)
# For one particle (so one aeff only)
# XXX ADDA has wavelength and aeff in um.
def read_Q_ADDA(path):
    i=0
    df_tot = pd.DataFrame()
    df_mueller = pd.DataFrame()
    print()
    while os.path.isfile(path+f'/w{str(i).zfill(4)}_CrossSec-avg'):
        df_m = pd.read_table(path+'/w'+str(i).zfill(4)+'_mueller_scatgrid',index_col=[0],engine='python',delim_whitespace=True)
        df = pd.read_table(path+f'/w{str(i).zfill(4)}_CrossSec-avg',sep='\t= ',index_col=0, header=None,skiprows=[6],engine='python').T
        df_X = pd.read_table(path+f'/w{str(i).zfill(4)}_CrossSec-avgX',sep='\t= ',index_col=0, header=None,skiprows=[6],engine='python').T
        df_Y = pd.read_table(path+f'/w{str(i).zfill(4)}_CrossSec-avgY',sep='\t= ',index_col=0, header=None,skiprows=[6],engine='python').T
        df["Qpol"] = (float(df_X["Qext"])-float(df_Y["Qext"]))/2
        df['g'].iloc[0] = np.fromstring(df['g'].iloc[0][1:-1],sep=',')
        df['Csca.g'].iloc[0] = np.fromstring(df['Csca.g'].iloc[0][1:-1],sep=',')
        df_tot = pd.concat([df_tot,df],ignore_index=True)
        # print(df["wave"].iloc[0])
        df_m['wave'] = float(df['wave'].iloc[0])
        df_m['aeff'] = float(df['aeff'].iloc[0])
        df_mueller = pd.concat([df_mueller,df_m])
        i+=1
    for col in df_tot.columns.values.tolist():
        if col != 'g' and col != 'Csca.g':
            df_tot[col] = pd.to_numeric(df_tot[col])
            
    df_tot['g0'] = np.array(df_tot['g'].to_list())[:,2]
    
    df_tot = df_tot.rename(columns = {'Q_ext':'Qext','Q_sca':'Qsca','Q_abs':'Qabs','g(1)=<cos>':'g0','Wave':'wave'})
    # microns to meters
    df_tot['wave'] = df_tot['wave']*10**(-6)
    df_tot['aeff'] = df_tot['aeff']*10**(-6)
    df_mueller['wave'] = df_mueller['wave']*10**(-6)
    df_mueller['aeff'] = df_mueller['aeff']*10**(-6)
    return df_tot, df_mueller

# XXX values are in cm (yay) so TODO : convert
def auto_read_Dustem(path,grainpath):
    """ 
    path to the directory where Q, G and C files are.
    path to the Grain file
    """
    with open(grainpath,'r') as f:
        lines = f.readlines()
    comments = ()
    for i,line in enumerate(lines) :
        if line.startswith('#'):
            comments += (i,)    
    lines=np.delete(lines,comments)        
    import gc
    my_dust = np.empty((len(lines)-2,7),dtype = object)
    params = np.empty((len(lines)-2),dtype = dict)
    
    class dust :
        
        def __init__(self,line):
            line = l.split()
            # print('______________')
            # print(line)
            self.name = line[0]
            self.qty = float(line[3])
            self.a_min = float(line[5])*0.01 # convert from cm to m
            self.a_max = float(line[6])*0.01 # convert from cm to m
            # a = np.linspace(self.a_min, self.a_max,10000)
            a_log = np.logspace(np.log10(self.a_min), np.log10(self.a_max),10000)
            # a_s = np.linspace(self.a_min, self.a_max)
            # a_log_s = np.logspace(np.log10(self.a_min), np.log10(self.a_max))
            if line[2][:4]=='logn':
                self.a_0 = float(line[7])*0.01 # convert from cm to m
                self.sig = float(line[8]) # no unit
                # TODO Test the intergal with linspace and logspace, and add a normalisation.
                self.distrib = lambda a : ( 1/a*np.exp(-np.log(a/self.a_0)**2/2/self.sig**2))*np.heaviside(a - self.a_min,1)* np.heaviside(self.a_max - a,0)# dn/da 
            elif line[2][:4]=='plaw':
                self.alpha = float(line[7])
                self.distrib = lambda a : (a ** self.alpha)*np.heaviside(a - self.a_min,1)* np.heaviside(self.a_max - a,0)# dn/da 
            elif line[2][:7]=='plaw-ed':
                self.alpha = float(line[7])
                self.a_t = float(line[8])*0.01 # convert from cm to m
                self.a_c = float(line[9])*0.01 # convert from cm to m
                self.gamma =float(line[10])
                self.distrib = lambda a : (a ** self.alpha)*(np.heaviside(self.a_t - a,1)+np.heaviside(a - self.a_t,0)*np.exp(-((a-self.a_t)/self.a_c)**self.gamma))*np.heaviside(a - self.a_min,1)* np.heaviside(self.a_max - a,0)# dn/da 
            # norm = integ_dust_size(self.distrib(a),a,self.a_min,self.a_max)
            norm_log = integ_dust_size(self.distrib(a_log),a_log,self.a_min,self.a_max)
            # print("random test:",self.distrib(np.array([1.50e-08, 2.00e-08, 2.50e-08, 5.00e-08, 7.50e-08, 1.00e-07])))
            # norm_s = integ_dust_size(self.distrib(a_s),a_s,self.a_min,self.a_max)
            # norm_log_s = integ_dust_size(self.distrib(a_log_s),a_log_s,self.a_min,self.a_max)
            # print(norm,norm_log,norm_s,norm_log_s)
            self.distrib_norm = lambda a: self.distrib(a)/norm_log
            #Mass distribution
            self.distrib_mass = lambda a: self.distrib(a)*a**3
            norm_log_mass = integ_dust_size(self.distrib_mass(a_log),a_log,self.a_min,self.a_max)
            self.distrib_mass_norm = lambda a: self.distrib(a)*a**3/norm_log_mass
            
            self.df_QG = read_Q_G_12_Dustem(path, self.name) 
            self.df_C = read_C_Dustem(path,self.name)
            # print("Parameters: ", params)
        
    for i,l in enumerate(lines[2:]):
        d = dust(line)
        my_dust[i][0] = d.name
        my_dust[i][1] = d.distrib_norm
        my_dust[i][2] = d.df_QG
        my_dust[i][3] = d.df_C
        my_dust[i][4] = d.qty
        my_dust[i][5] = d.distrib_mass_norm
        my_dust[i][6] = (d.a_min,d.a_max)
            
    return my_dust

def write_Q_C_Dustem(path,name,df_QG,df_C): #TODO : manage conversions ! waves, aeff = um; densities : g/cm3; T : K; C : *10 (J/m3 to erg/cm3)
    
    wave = np.unique(df_QG.index.get_level_values('wave'))
    
    ##### LAMBDA #####
    df_wave = pd.DataFrame(wave)*1e6
    df_wave.to_csv(path+f'/LAMBDA.DAT',sep='\t',header=[str(len(df_wave))],index=False)
    
    
    if 'Qabs1' in df_QG:
        df_Qabs1 = df_QG['Qabs1'].unstack('aeff')
        df_Qabs2 = df_QG['Qabs2'].unstack('aeff')
        df_Qsca1 = df_QG['Qsca1'].unstack('aeff')
        df_Qsca2 = df_QG['Qsca2'].unstack('aeff')
        df_dens = df_QG['dens'].unstack('aeff')*1e-3
        df_aeff = pd.DataFrame([df_Qabs1.columns])*1e6
        
        
        df_aeff.to_csv(path+f'/Q1_{name}.DAT',sep='\t',header=[str(len(df_aeff.iloc[0]))]+['' for i in range(len(df_aeff.iloc[0])-1)],index=False,float_format='%.4e')
        # df_dens.iloc[0:1].to_csv(path+f'/Q1_{name}.DAT',sep='\t',header=False,index=False,mode='a',float_format='%.4e')
        df_Qabs1.to_csv(path+f'/Q1_{name}.DAT',sep='\t',header=["#Qabs"]+['' for i in range(len(df_Qabs1.iloc[0])-1)],index=False,mode='a',float_format='%.4e')
        df_Qsca1.to_csv(path+f'/Q1_{name}.DAT',sep='\t',header=["#Qsca"]+['' for i in range(len(df_Qsca1.iloc[0])-1)],index=False,mode='a',float_format='%.4e')
        
        
        df_aeff.to_csv(path+f'/Q2_{name}.DAT',sep='\t',header=[str(len(df_aeff.iloc[0]))]+['' for i in range(len(df_aeff.iloc[0])-1)],index=False,float_format='%.4e')
        # df_dens.iloc[0:1].to_csv(path+f'/Q2_{name}.DAT',sep='\t',header=False,index=False,mode='a',float_format='%.4e')
        df_Qabs2.to_csv(path+f'/Q2_{name}.DAT',sep='\t',header=["#Qabs"]+['' for i in range(len(df_Qabs2.iloc[0])-1)],index=False,mode='a',float_format='%.4e')
        df_Qsca2.to_csv(path+f'/Q2_{name}.DAT',sep='\t',header=["#Qsca"]+['' for i in range(len(df_Qsca2.iloc[0])-1)],index=False,mode='a',float_format='%.4e')
        
    ##### Q #####
    df_Qabs = df_QG['Qabs'].unstack('aeff')
    df_Qsca = df_QG['Qsca'].unstack('aeff')
    df_dens = df_QG['dens'].unstack('aeff')*1e-3
    df_aeff = pd.DataFrame([df_Qabs.columns])*1e6
    
    df_aeff.to_csv(path+f'/Q_{name}.DAT',sep='\t',header=[str(len(df_aeff.iloc[0]))]+['' for i in range(len(df_aeff.iloc[0])-1)],index=False,float_format='%.4e')
    df_dens.iloc[0:1].to_csv(path+f'/Q_{name}.DAT',sep='\t',header=False,index=False,mode='a',float_format='%.4e')
    df_Qabs.to_csv(path+f'/Q_{name}.DAT',sep='\t',header=["#Qabs"]+['' for i in range(len(df_Qabs.iloc[0])-1)],index=False,mode='a',float_format='%.4e')
    df_Qsca.to_csv(path+f'/Q_{name}.DAT',sep='\t',header=["#Qsca"]+['' for i in range(len(df_Qsca.iloc[0])-1)],index=False,mode='a',float_format='%.4e')
    
    ##### G #####
    df_G = df_QG['g0'].unstack('aeff')
    
    df_aeff.to_csv(path+f'/G_{name}.DAT',sep='\t',header=[str(len(df_aeff.iloc[0]))]+['' for i in range(len(df_aeff.iloc[0])-1)],index=False,float_format='%.4e')
    df_G.to_csv(path+f'/G_{name}.DAT',sep='\t',header=["#g-factor"]+['' for i in range(len(df_G.iloc[0])-1)],index=False,mode='a',float_format='%.4e')
    
    ##### C #####
    df_CC = np.log10(df_C*10)
    df_CC.index = np.log10(df_CC.index)
    df_aeff.to_csv(path+f'/C_{name}.DAT',sep='\t',header=[str(len(df_aeff.iloc[0]))]+['' for i in range(len(df_aeff.iloc[0])-1)],index=False,float_format='%.4e')
    df_CC.to_csv(path+f'/C_{name}.DAT',sep='\t',header=[str(len(df_CC))]+['' for i in range(len(df_CC.iloc[0])-1)],index=True,mode='a',float_format='%.4e')
    
    
    
    
    
    
        
# XXX  lambdas are un um !
def read_Q_Dustem(path,filepath):
    """ path = pth to lambdas
    """
    df_lambdas = pd.read_table(path+'/LAMBDA.DAT',engine='python', comment='#')
    df_lambdas.columns = ['wave']
    with open(filepath,'r') as f:
        lines = f.readlines()
    # XXX : read by blocks between comments - Qabs and Qsca shoul be the last two blocks. n of sizes and sizes should be two first lines of first block.
    gr_start,gr_end = [],[]
    last_comm = True
    for i,l in enumerate(lines):
       if last_comm and not l[0] =='#':
           gr_start = gr_start + [i]
       elif not last_comm and l[0] =='#':
           gr_end = gr_end+[i]
       if l[0] =='#':
           last_comm = True
       else:
           last_comm = False
    if len(gr_end) < len(gr_start):
        gr_end = gr_end+[len(lines)]
    if len(gr_start) != 3:
        raise ValueError("Please check the Q file structure")
    n_sizes = int(lines[gr_start[0]])
    df_aeff = pd.read_table(filepath,delim_whitespace=True,engine='python', index_col=False,names =[i for i in range(n_sizes)],skiprows = gr_start[0]+1,header=None,skipfooter = len(lines)-(gr_start[0]+2))
    if gr_end[0]-gr_start[0]>2:
        df_dens = pd.read_table(filepath,delim_whitespace=True,engine='python', index_col=False,names =[i for i in range(n_sizes)],skiprows = gr_start[0]+2,header=None,skipfooter = len(lines)-(gr_start[0]+3))
    
    df_qabs = pd.read_table(filepath,delim_whitespace=True,engine='python', index_col=False,names =[i for i in range(n_sizes)],skiprows = gr_start[1],header=None,skipfooter = len(lines)-gr_end[1])
    df_qsca = pd.read_table(filepath,delim_whitespace=True,engine='python', index_col=False,names =[i for i in range(n_sizes)],skiprows = gr_start[2],header=None,skipfooter = len(lines)-gr_end[2])
    df_qabs = pd.DataFrame(df_qabs.stack())
    df_qabs['aeff'] = df_qabs.index.get_level_values(1)
    df_qabs['aeff'] = df_qabs['aeff'].map(df_aeff.T[0])
    if gr_end[0]-gr_start[0]>2:
        df_qabs['dens'] = df_qabs.index.get_level_values(1)
        df_qabs['dens'] = df_qabs['dens'].map(df_dens.T[0]) # density is in g/cm3 (ok) # no ! for polaris we need kg/m3
        df_qabs['dens'] = df_qabs['dens']*1000 # g/cm3 to kg/m3
    df_qabs['wave'] = df_qabs.index.get_level_values(0)
    df_qabs['wave'] = df_qabs['wave'].map(df_lambdas['wave'])
    df_qsca = pd.DataFrame(df_qsca.stack())
    df_qsca['aeff'] = df_qsca.index.get_level_values(1)
    df_qsca['aeff'] = df_qsca['aeff'].map(df_aeff.T[0])
    if gr_end[0]-gr_start[0]>2:
        df_qsca['dens'] = df_qsca.index.get_level_values(1)
        df_qsca['dens'] = df_qsca['dens'].map(df_dens.T[0])
        df_qsca['dens'] = df_qsca['dens']*1000 # g/cm3 to kg/m3
    df_qsca['wave'] = df_qsca.index.get_level_values(0)
    df_qsca['wave'] = df_qsca['wave'].map(df_lambdas['wave'])
    
    df_qabs = df_qabs.rename(columns = {0:'Qabs'})
    df_qsca = df_qsca.rename(columns = {0:'Qsca'})
    df_tot = pd.merge(left = df_qabs, right = df_qsca)
    df_tot['Qext'] = df_tot['Qabs'] + df_tot['Qsca'] 
    
    # microns to meters
    df_tot['wave'] = df_tot['wave']*10**(-6)
    df_tot['aeff'] = df_tot['aeff']*10**(-6)
    return df_tot

def read_G_Dustem(path,filename):
    df_lambdas = pd.read_table(path+'/LAMBDA.DAT',engine='python', comment='#')
    df_lambdas.columns = ['wave']
    with open(path+f'/G_{filename}.DAT','r') as f:
        lines = f.readlines()
    # XXX : read by blocks between comments - G shoul be the last blocks. n of sizes and sizes should be two first lines of first block.
    gr_start,gr_end = [],[]
    last_comm = True
    for i,l in enumerate(lines):
       if last_comm and not l[0] =='#':
           gr_start = gr_start + [i]
       elif not last_comm and l[0] =='#':
           gr_end = gr_end+[i]
       if l[0] =='#':
           last_comm = True
       else:
           last_comm = False
    if len(gr_end) < len(gr_start):
        gr_end = gr_end+[len(lines)]
    if len(gr_start) != 2:
        raise ValueError("Please check the Q file structure")
    n_sizes = int(lines[gr_start[0]])
    df_aeff = pd.read_table(path+f'/G_{filename}.DAT',delim_whitespace=True,engine='python', index_col=False,names =[i for i in range(n_sizes)],skiprows = gr_start[0]+1,header=None,skipfooter = len(lines)-(gr_start[0]+2))
    df_g = pd.read_table(path+f'/G_{filename}.DAT',delim_whitespace=True,engine='python', index_col=False,names =[i for i in range(n_sizes)],skiprows = gr_start[1],header=None,skipfooter = len(lines)-gr_end[1])
    df_g = pd.DataFrame(df_g.stack())
    df_g['aeff'] = df_g.index.get_level_values(1)
    df_g['aeff'] = df_g['aeff'].map(df_aeff.T[0])
    df_g['wave'] = df_g.index.get_level_values(0)
    df_g['wave'] = df_g['wave'].map(df_lambdas['wave'])
    
    df_g = df_g.rename(columns = {0:'g0'})
    
    # microns to meters
    df_g['wave'] = df_g['wave']*10**(-6)
    df_g['aeff'] = df_g['aeff']*10**(-6)
    return df_g

def read_C_Dustem(path,filename):
    with open(path+f'/C_{filename}.DAT','r') as f:
        lines = f.readlines()
    lines = np.array(lines)[[not l.strip().startswith('#') for l in lines]]
    n_sizes = int(lines[0])
    n_temp = int(lines[2])
    # XXX Be careful, in C files all is log 
    # XXX C are in erg/K/cm3 - Polaris requires in J/K/m^3
    df_aeff = pd.read_table(path+f'/C_{filename}.DAT',delim_whitespace=True,engine='python', index_col=False,names =[i for i in range(n_sizes)],comment='#',header=0,skipfooter = len(lines)-2)
    df_C = pd.read_table(path+f'/C_{filename}.DAT',delim_whitespace=True,engine='python', index_col=False,names =['logT']+[i for i in range(n_sizes)],comment='#')
    df_C = df_C.drop([0,1,2])
    df_C['T'] = np.power(10,df_C['logT'])
    df_C = df_C.drop(columns='logT')
    df_C = df_C.set_index('T') 
    df_C = np.power(10,df_C)*0.1 #conversion
    # Lines below are in fact not needed for the polaris file writing.
    # df_C = pd.DataFrame(df_C.stack())
    # df_C['aeff'] = df_C.index.get_level_values(1)
    # df_C['aeff'] = df_C['aeff'].map(df_aeff.T[0])
    # # microns to meters
    # df_C['aeff'] = df_C['aeff']*10**(-6)
    # df_C =df_C.set_index(['aeff'],append=True,verify_integrity=True)
    # df_C =df_C.reset_index(level =1,drop=True)
    return df_C


def read_C_Polaris(path,name):
        with open(path+'/'+name+'/calorimetry.dat','r') as f:
            lines = f.readlines()
        lines = np.array(lines)[[not l.strip().startswith('#') for l in lines]]
        n_temp = int(lines[0])
        n_sizes = len(lines[3].split())
        # XXX Be careful, in C files all is log 
        # XXX C are in erg/K/cm3 - Polaris requires in J/K/m^3
        df_T = pd.read_table(path+'/'+name+'/calorimetry.dat',delim_whitespace=True,engine='python', index_col=False,names =[i for i in range(n_temp)],comment='#',header=0,skipfooter = len(lines)-2)
        df_C = pd.read_table(path+'/'+name+'/calorimetry.dat',delim_whitespace=True,engine='python', index_col=False,names =[i for i in range(n_sizes)],comment='#')
        df_C = df_C.drop([0,1,2])
        df_C = df_C.reset_index(drop=True)
        df_C['T'] = df_T.transpose()
        df_C = df_C.set_index('T')
        return df_C
        
# TODO : these function are supposed to be finished : test them !
def read_Q_G_12_Dustem(path,filename):
    if os.path.isfile(path+f'/Q1_{filename}.DAT'):
        df_q1 = read_Q_Dustem(path,path+f'/Q1_{filename}.DAT')
        df_q1 = df_q1.rename(columns = {'Qext':'Qext1','Qsca':'Qsca1','Qabs':'Qabs1'})
        df_q2 = read_Q_Dustem(path,path+f'/Q2_{filename}.DAT')
        df_q2 = df_q2.rename(columns = {'Qext':'Qext2','Qsca':'Qsca2','Qabs':'Qabs2'})
        df_q = pd.merge(left = df_q1, right = df_q2)
        df_q['dens'] =  read_Q_Dustem(path,path+f'/Q_{filename}.DAT')['dens']
        df_q['Qext'] = (df_q['Qext1']+df_q['Qext2'])/2
        df_q['Qsca'] = (df_q['Qsca1']+df_q['Qsca2'])/2
        df_q['Qabs'] = (df_q['Qabs1']+df_q['Qabs2'])/2
    else:
        df_q = read_Q_Dustem(path,path+f'/Q_{filename}.DAT')
    df_g =  read_G_Dustem(path,filename)
    df_tot = pd.merge(left = df_q, right = df_g)
    # TODO : keep or leave ? 
    df_tot =df_tot.set_index(['aeff','wave'])
    
    return df_tot 
    
def read_Q_DDscat_a_eff(paths):
    data_tot = pd.DataFrame()
    for path in paths:
        data = read_Q_DDscat(path)
        data_tot = pd.concat([data_tot,data],ignore_index=True)
    data_tot =data_tot.set_index(['aeff','wave'])
    return data_tot

def read_Q_ADDA_a_eff(paths):
    data_tot = pd.DataFrame()
    for path in paths:
        data,_ = read_Q_ADDA(path)
        data_tot = pd.concat([data_tot,data],ignore_index=True)
    data_tot =data_tot.set_index(['aeff','wave'])
    return data_tot



# def expand_Q(df_tot,a_exp):
    
#     wave = np.unique(df_tot.index.get_level_values('wave')) # if in indices
    
#     for w in wave:
#         df = df_tot.iloc[df_tot.index.get_level_values('wave') == w]
#         df_exp = pd.DataFrame()
#         for col in df.columns : 
#             df_exp[col] = np.interp(a_all,df.index.get_level_values('aeff'),df[col],left=0,right =0)
#         df_exp['aeff'] = a_exp
#         df_exp =df_exp.set_index(['aeff'])
#         df = df_exp
        
    

def write_polaris_Q_file(path,name,df_tot,ratio,dens,sub_temp,delta,expand_a = None):
    gs = df_tot.columns[df_tot.columns.str.startswith('g')].tolist()
    n_i = len(gs)
    
    if 'Qext1' not in df_tot:
        df_tot['Qext1'] = df_tot['Qext']
        df_tot['Qext2'] = df_tot['Qext']
        df_tot['Qabs1'] = df_tot['Qabs']
        df_tot['Qabs2'] = df_tot['Qabs']
        df_tot['Qsca1'] = df_tot['Qsca']
        df_tot['Qsca2'] = df_tot['Qsca']
    # wave = np.unique(df_tot['wave'])
    wave = np.unique(df_tot.index.get_level_values('wave')) # if in indices
    # aeff = np.unique(df_tot['aeff'])
    
    #Adding zeros if columns don't exist
    df_tot['dQphas'] = df_tot.get('dQphas',0)
    qtrq = ['Qtrq'+g[1:] for g in gs]
    for g in gs:
        df_tot['Qtrq'+g[1:]] = df_tot.get('Qtrq'+g[1:],0)
    
    print(qtrq,gs)
    
    print(df_tot)
    
    if expand_a is not None:
        aeff = expand_a
    else:
        aeff = np.unique(df_tot.index.get_level_values('aeff')) # if in indices
        
    print(wave)
    lines = ['\n' for i in range(25+len(wave)*len(aeff))]
    lines[0] = "# CARPINE Marie-Anne\n"
    lines[1] = f"# File generated {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
    lines[9]= "#string ID\n"
    lines[10]= name+'\n'
    lines[12] = "#nr. of dust species #wavelength #inc. angles #aspect ratio #density [kg/m^3] #sub.temp #delta  #align\n"
    lines[13] = f'{len(aeff)}\t{len(wave)}\t{n_i}\t{ratio}\t{dens}\t{sub_temp}\t{delta}\t{0}\n'
    lines[15] = "#a_eff\n"
    lines[16] = "#"+'\t'.join(map(str, range(len(aeff))))+'\n'
    lines[17] = '\t'.join(map(str, aeff))+'\n'
    lines[19] = "#wavelength\n"
    lines[20] = "#"+'\t'.join(map(str, range(len(wave))))+'\n'
    lines[21] = '\t'.join(map(str, wave))+'\n'
    lines[23] = "#"+'\t'.join(['Qext1','Qext2','Qabs1','Qabs2','Qsca1','Qsca2','dQphas']+qtrq+gs)+'\n'
    # lines[23] = "#Qext1\tQext2\tQabs1\tQabs2\tQsca1\tQsca2\tdQphas\tQtrq0\tg0\n"
    lines[24] = "#"+'\t'.join(map(str, range(len(['Qext1','Qext2','Qabs1','Qabs2','Qsca1','Qsca2','dQphas']+qtrq+gs))))+'\n'

    i=25
    for w in wave:
        # df = df_tot[df_tot["wave"]==w]
        df = df_tot.iloc[df_tot.index.get_level_values('wave') == w]
        # df = df_tot.loc[[w]]# if in indices
        if len(df) !=  len(aeff) and expand_a is None:
            raise ValueError("The data is missing some radius or wavelength or Some (aeff,wavelength) data is duplicated!!!!")
            
        if expand_a is not None:
            df_exp = pd.DataFrame()
            for col in df.columns : 
                # df_exp[col] = np.interp(aeff,df.index.get_level_values('aeff'),df[col],left=0,right =0)
                df_exp[col] = np.interp(aeff,df.index.get_level_values('aeff'),df[col])
            df_exp['aeff'] = aeff
            df_exp = df_exp.set_index(['aeff'])
            df = df_exp
            
        
        for a in aeff:
            # df_unique = df[df["aeff"]==a]
            df_unique = df.iloc[df.index.get_level_values('aeff') == a]
            # print(df_unique[['Qext1','Qext2','Qabs1','Qabs2','Qsca1','Qsca2','dQphas']+qtrq+gs])
            # df_unique = df.loc[[a]]# if in indices
            if len(df_unique) != 1:
                raise ValueError("Some (aeff,wavelength) data is duplicated!!")
            # lines[i] = '\t'.join(map(str, df_unique[['Qext1','Qext2','Qabs1','Qabs2','Qsca1','Qsca2']].iloc[0]))+f"\t0.0\t0.0\t{df_unique['g0'].iloc[0]}\n"
            lines[i] = '\t'.join(map(str, df_unique[['Qext1','Qext2','Qabs1','Qabs2','Qsca1','Qsca2','dQphas']+qtrq+gs].iloc[0]))+'\n'
            i+=1
    with open(path+'/'+name+'.dat','w+') as f:
        f.writelines(lines)
        
    return read_Q_Polaris(path+'/'+name+'.dat')
    return lines


def write_polaris_C_file(path,name,df_C,expand_a = None,original_a=None):
    # wave = np.unique(df_tot['wave'])
    temp = np.unique(df_C.index.get_level_values('T')) # if in indices
    lines = ['\n' for i in range(12)]
    lines[0] = "# CARPINE Marie-Anne\n"
    lines[1] = f"# File generated {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
    lines[3] = "# nr. of temperatures\n"
    lines[4] = f'{len(temp)}\n'
    lines[5] = '#  temperature: T [K]\n'
    lines[6] = '\t'.join(map(str, temp))+'\n'
    lines[7] = "# type of calorimetry\n"
    lines[8] = '0\n'
    lines[9] = '# heat capacity C [J/K/m^3]\n'
    lines[10] = '# C(T_0, a_0), C(T_0, a_1), C(T_0, a_2), ... \n'
    lines[11] = '# C(T_1, a_0), C(T_1, a_1), C(T_1, a_2), ... \n'

    
    os.makedirs(path+'/'+name, exist_ok=True)
    with open(path+'/'+name+'/calorimetry.dat','w+') as f:
        f.writelines(lines)
    
                   
    
    if expand_a is not None:
        df_exp = pd.DataFrame(index=df_C.index)
        for j in range(len(expand_a)):
            df_exp[j] = pd.Series(dtype='int')
            
        for i in range(len(df_C)):
            df_exp.iloc[i] = np.interp(expand_a,original_a,df_C.iloc[i],left=0,right =0)
        df_C = df_exp
        
    df_C.to_csv(path+'/'+name+'/calorimetry.dat',sep='\t',header=False,index=False,mode='a')
    
    return df_C
    return lines

def get_df_closest(df,name,value):
    # np.abs(df_Q_3.index.get_level_values('aeff').to_numpy() - 1.00e-06)
    idx = np.nonzero(np.abs(df.index.get_level_values(name).to_numpy() - value) == np.amin(np.abs(df.index.get_level_values(name).to_numpy() - value)))
    if len(np.unique(df.index.get_level_values(name).to_numpy()[idx]))>1:
        print("two closest values, choosing the smallest")
        idx = np.nonzero(df.index.get_level_values(name).to_numpy() == np.unique(df.index.get_level_values(name).to_numpy()[idx])[0])
    # print(idx)
    return(df.iloc[idx])

# XX for the sut distribution, dustem and polaris seem to use the same format.


# ### For radMC
def write_radmc_file_averaged(path,name,df_tot,distrib,a_min =None,a_max=None):
    
    if a_min is not None:
        coeff = lambda a : np.heaviside(a - a_min,1)* np.heaviside(a_max - a,0)
    else :
        coeff = lambda a : 1
    
    print(path,name,df_tot)
    ## Mass-weighted opacity (<=> number-weighted cross-section / mean mass)
    
    # df_tot['kabs'] = df_tot['Qabs']*3/(4*df_tot.index.get_level_values('aeff')*100*df_tot['dens']) #Needs to be in cm2/g so convert aeff in cm (dens is in g/cm3 from dustem input)
    
    wave = np.unique(df_tot.index.get_level_values('wave')) # if in indices
    
    aeff,idx_aeff = np.unique(df_tot.index.get_level_values('aeff'),return_index=True) # if in indices
    
    df_weighted = pd.DataFrame(0,columns=['Cext','Cabs','Csca','g0','Qext','Qabs','Qsca'],index = pd.Index(wave,name = 'wave'),)
    
   

        
    for w in wave: 
        df_weighted.loc[df_weighted.index.get_level_values('wave') == w,'Cabs'] = integ_dust_size(df_tot.iloc[df_tot.index.get_level_values('wave') == w]['Qabs'].to_numpy()*np.pi*aeff**2*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)
        df_weighted.loc[df_weighted.index.get_level_values('wave') == w,'Cext'] = integ_dust_size(df_tot.iloc[df_tot.index.get_level_values('wave') == w]['Qext'].to_numpy()*np.pi*aeff**2*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)
        df_weighted.loc[df_weighted.index.get_level_values('wave') == w,'Csca'] = integ_dust_size(df_tot.iloc[df_tot.index.get_level_values('wave') == w]['Qsca'].to_numpy()*np.pi*aeff**2*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)
        df_weighted.loc[df_weighted.index.get_level_values('wave') == w,'Qabs'] = integ_dust_size(df_tot.iloc[df_tot.index.get_level_values('wave') == w]['Qabs'].to_numpy()*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)
        df_weighted.loc[df_weighted.index.get_level_values('wave') == w,'Qext'] = integ_dust_size(df_tot.iloc[df_tot.index.get_level_values('wave') == w]['Qext'].to_numpy()*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)
        df_weighted.loc[df_weighted.index.get_level_values('wave') == w,'Qsca'] = integ_dust_size(df_tot.iloc[df_tot.index.get_level_values('wave') == w]['Qsca'].to_numpy()*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)
        df_weighted.loc[df_weighted.index.get_level_values('wave') == w,'gCsca'] = integ_dust_size(df_tot.iloc[df_tot.index.get_level_values('wave') == w]['Qsca'].to_numpy()*np.pi*aeff**2*df_tot.iloc[df_tot.index.get_level_values('wave') == w]['g0'].to_numpy()*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)
    
    
    number = integ_dust_size(distrib(aeff)*coeff(aeff),aeff,a_min,a_max)
    size = integ_dust_size(aeff*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)/number #mean size
    volume = integ_dust_size(4/3*np.pi*aeff**3*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)/number  #mean volume
    mass = integ_dust_size(4/3*np.pi*aeff**3*df_tot.iloc[idx_aeff]['dens'].to_numpy()*distrib(aeff)*coeff(aeff),aeff,a_min,a_max)/number #mean mass
    
    
    # mean values
    df_weighted['Cabs'] = df_weighted['Cabs']/number
    df_weighted['Cext'] = df_weighted['Cext']/number
    df_weighted['Csca'] = df_weighted['Csca']/number
    df_weighted['Qabs'] = df_weighted['Qabs']/number
    df_weighted['Qext'] = df_weighted['Qext']/number
    df_weighted['Qsca'] = df_weighted['Qsca']/number
    df_weighted['kabs'] = df_weighted['Cabs']/mass
    df_weighted['kext'] = df_weighted['Cext']/mass
    df_weighted['ksca'] = df_weighted['Csca']/mass
    df_weighted['gCsca'] = df_weighted['gCsca']/number
    df_weighted['g0'] = df_weighted['gCsca']/df_weighted['Csca']
    
    lines = ['\n' for i in range(5)]
    lines[0] = "# CARPINE Marie-Anne\n"
    lines[1] = f"# File generated {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
    lines[2] = '3 \n'
    lines[3] = f'{len(df_weighted)}\n'
    
    with open(path+f'/dustkappa_{name}.inp','w+') as f:
        f.writelines(lines)
    
    df_towrite =  df_weighted[['kabs','ksca']]*10 # convert m2/kg to cm2/g
    df_towrite['g0'] =  df_weighted['g0']
    df_towrite.index = df_towrite.index*1e6 # convert wave to um
    df_towrite[['kabs','ksca','g0']].to_csv(path+f'/dustkappa_{name}.inp',sep='\t',header=False,index=True,mode='a',float_format='%.6e') 
    print("df_weighted",df_weighted)
    return df_weighted, mass
    # Now we create the weighted opacity : we compute total Cross sec, total mass and we make the ratio.


#TODO : separation of a species in multiple species and writing + writing of differetn species at once.    
    
def write_radmc_file_multiple_species(path,name,df_tot,distrib,proportions,a_min =None,a_max=None):
    """
    All params = lists of same length except path

    convertion from mass proportion X to number propotion x with mi mean mass of the species
    
    mi = Mi/Ni et xi = Ni/Ntot, Xi = Mi/Mtot
    
    Xi = xi*mi/(Sum_j(xj*mj))
    car
    xi*mi = Mi/Ntot
    Sum_j(xj*mj) = Mtot/Ntot
    
    et xi = Xi/mi/(Sum_j(Xj/mj))
    car
    Xi/mi = Ni/Mtot
    Sum_j(Xj/mj) = Ntot/Mtot
    
    """
    
    
    proportions = proportions/np.sum(proportions)
    n_spec = len(name)
    
    dfs_weighted, masses = np.empty((np.sum(n_spec)),dtype=object),np.empty((np.sum(n_spec)),dtype=object)
        #Write dust_kappa files
    i=0
    if a_min is not None:
        for n,df,dis,a_min_b,a_max_b in zip(name,df_tot,distrib,a_min,a_max):
            df_weighted, mass = write_radmc_file_averaged(path,n,df,dis,a_min_b,a_max_b)
            dfs_weighted[i] = df_weighted
            masses[i] = mass
            i+=1
    else:
        for n,df,dis in zip(name,df_tot,distrib):
            df_weighted, mass = write_radmc_file_averaged(path,n,df,dis)
            dfs_weighted[i] = df_weighted
            masses[i] = mass
            i+=1
        
    #Write dust opacity file
    lines = ['\n' for i in range(3+4*n_spec)]
    lines[0] = "2               Format number of this file\n"
    lines[1] = f"{n_spec}               Nr of dust species\n"
    lines[2] = '============================================================================\n'
    
    for i in range(n_spec):
        lines[3+i*4+0] = '1               Way in which this dust species is read\n' #for dust_kappe. If scat matrix : 10
        lines[3+i*4+1] = '0               0=Thermal grain\n'
        lines[3+i*4+2] = f'{name[i]}        Extension of name of dustkappa_***.inp file\n'
        lines[3+i*4+3] = '----------------------------------------------------------------------------\n'
    
    with open(path+f'/dustopac.inp','w+') as f:
        f.writelines(lines)
    
    #Duplicate the dust densities    
    if n_spec>1:
        with open(path+f'/dust_density.inp','r') as f:
            lines=f.readlines()
            
        n_spec_orig = int(lines[2])
        n_cell = int(lines[1])
        
        if n_spec_orig != 1:
            raise ValueError("The original file contains more than one dust density, cannot be converted.")
            
        table = np.loadtxt(path+f'/dust_density.inp',skiprows=3)
        
        if len(table) != n_cell:
            raise ValueError("The original file has an inconsistent numer of lines.")
        
        new_lines = ['\n' for i in range(3)]
        new_lines[0] = lines[0]
        new_lines[1] = f'{n_cell}\n'
        new_lines[2] = f'{n_spec}\n'
        
        shutil.copyfile(path+f'/dust_density.inp', path+f'/dust_density_old.inp')

        
        with open(path+f'/dust_density.inp','w') as f:
            f.writelines(new_lines)
            
        for s in range(n_spec):
            with open(path+f'/dust_density.inp', "a") as f:
                np.savetxt(f, table*proportions[s],fmt='%.6e')
                
    return dfs_weighted, masses
                
                
def make_radmc_separate_sizes(df_tot,distrib,a_min,a_max,n_bins=None):
    
    # Need to sort out if the dust_density file is density in mass or in number ! Those are mass dust densi. Use dust mass proportions.
    
    a_eff,idx_aeff = np.unique(df_tot.index.get_level_values('aeff'),return_index=True) # if in indices
    
    # a_min = np.nonzero(distrib(a_eff))[0][0]
    # a_max = np.nonzero(distrib(a_eff))[0][-1]
    
    aeff = a_eff[np.concatenate((np.nonzero(distrib(a_eff))[0],[np.nonzero(distrib(a_eff))[0][-1]+1]))] #We add the amax which was ruled out for symmetry
    
    if n_bins is None or len(aeff)-1<n_bins:
        bins = aeff # The bins borders. 
        if len(aeff)<n_bins :
            print("More bins than actual available aeff were asked for. Setting available aeff as bins")
            
    else :
        # Now we select 'regular' aeffs as the bins borders to have a continuous integration.
        q = (len(aeff)-1)//(n_bins)
        r = (len(aeff)-1)%(n_bins) 
        idx = [i*(q+1) for i in range(r)] + [r*(q+1)+i*q for i in range(n_bins+1-r)]
        bins = aeff[idx]
    print("aeff",aeff)    
    print("bins",bins)    
        
    prop = np.empty((len(bins)-1))
    masses = np.empty((len(bins)-1))
    i=0
    for a_min_b, a_max_b in zip(bins[:-1],bins[1:]):
        # a_loc = np.logspace(np.log10(a_min), np.log10(a_max),10000)
        print(a_min_b, a_max_b)
        print(integ_dust_size(distrib(a_eff),a_eff,a_min,a_max))
        print(integ_dust_size(distrib(a_eff)*np.heaviside(a_eff - a_min_b,1)* np.heaviside(a_max_b - a_eff,0),a_eff,a_min,a_max))
        print('______')
        proportion = integ_dust_size(distrib(a_eff)*np.heaviside(a_eff - a_min_b,1)* np.heaviside(a_max_b - a_eff,0),a_eff,a_min,a_max)/integ_dust_size(distrib(a_eff),a_eff,a_min,a_max)
        number = integ_dust_size(distrib(a_eff)*np.heaviside(a_eff - a_min_b,1)* np.heaviside(a_max_b - a_eff,0),a_eff,a_min,a_max)
        mass = integ_dust_size(4/3*np.pi*a_eff**3*df_tot.iloc[idx_aeff]['dens'].to_numpy()*distrib(a_eff)*np.heaviside(a_eff - a_min_b,1)* np.heaviside(a_max_b - a_eff,0),a_eff,a_min,a_max)/number
        
        prop[i] = proportion #in numbers not in mass ! # Heavyside de droite à 0 pour les bins pour avoir des proportions à 1...
        masses[i] = mass
        i+=1
        
    prop_mass = prop*masses/(np.sum(prop*masses))
    #We can calculate the mass proportion of the bin/the species. Xi = xi*mi/(Sum_j(xj*mj))
    
    bins[0] = a_min
    bins[-1] = a_max
    # Then we multiply the bin fraction with the species fraction and tada ! 
    
    return df_tot,distrib,prop_mass,prop,bins[:-1],bins[1:]
    
def write_radmc_separate_sizes_multiplt_species(path,name,df_tot,distrib,proportions,n_bins,a_min,a_max):
    
    proportions = proportions/np.sum(proportions)
    #TODO beware of distribs as variables change...
    name_s = np.empty((np.sum(n_bins)),dtype= object)
    df_s = np.empty((np.sum(n_bins)),dtype=object)
    distrib_s = np.empty((np.sum(n_bins)),dtype=object)
    prop_mass = np.empty((np.sum(n_bins)))
    prop_number = np.empty((np.sum(n_bins))) #just for info
    a_min_s = np.empty((np.sum(n_bins)))
    a_max_s = np.empty((np.sum(n_bins)))
    
    i=0
    
    for n,df,dis,prop,n_b,a_min_b,a_max_b in zip(name,df_tot,distrib,proportions,n_bins,a_min,a_max): # for each spec,ies
        df_loc,distrib_loc,prop_size,prop_num,a_min,a_max  = make_radmc_separate_sizes(df,dis,a_min_b,a_max_b,n_b) # get the info in diffferent sizes
        
        for j in range(n_b):
            print(n+f'_bin_{j}')
            name_s[i] = n+f'_bin_{j}'
            df_s[i] = df_loc
            distrib_s[i] = copy.deepcopy(distrib_loc)
            prop_mass[i] = prop_size[j]*prop 
            prop_number[i] = prop_num[j]#XXX beware the propnum is just for info : it is valid within a species but is not ponderated between one species and another. Just for basic tests that were positive.
            a_min_s[i] = a_min[j]
            a_max_s[i] = a_max[j]
            i+=1
        
    print(name_s,a_min_s,a_max_s)
    
    dfs_weighted, masses = write_radmc_file_multiple_species(path,name_s,df_s,distrib_s,prop_mass,a_min_s,a_max_s)
    
    return path,name_s,df_s,distrib_s,prop_mass,a_min_s,a_max_s, dfs_weighted, masses ,prop_number
        
   
# # TEST     plot size distrib 
# my_dust = auto_read_Dustem('/Users/mc276490/Documents/THEMIS2/Données DustEM THEMIS2/THEMIS_2_for_dustem 2','/Users/mc276490/Documents/THEMIS2/Données DustEM THEMIS2/inputs_for_DustEM_13perc_Lenz/GRAIN_2.DAT')
# plt.figure()
# for d in my_dust:
#     name = d[0]
#     a = np.unique(d[2].index.get_level_values('aeff'))
#     not_zero = np.where(d[1](a))
#     plt.plot(a[not_zero],d[1](a)[not_zero],label =name)
#     print(name,a,d[1](a))
# plt.loglog()
# plt.legend()
# plt.show()

# # Polaris requires lenghts in meters.
# # TODO : compute the averaged density from the size/mass distribution and put it in the file.
# # TODO : if we don't read from Dustem, we need the densities : cf what was given by nathalie and make the densities.
# # When creating the Q files, we also need to create the density files. What about C files and size distribution ?


# Test writing pol from dustem


# path = '/Users/mc276490/Documents/THEMIS2/Données DustEM THEMIS2/THEMIS_2_for_dustem 2'
# my_dust = auto_read_Dustem(path,'/Users/mc276490/Documents/THEMIS2/Données DustEM THEMIS2/inputs_for_DustEM_13perc_Lenz/GRAIN_2.DAT')

# a_list = np.empty(3,dtype = object)
# a_all = np.empty(0)
# i=0
 
# # np.interp(all_a,a,df_,left=0,right =0)

# plt.figure()
# for d in my_dust:
#     name = d[0]
#     df_QG = d[2]
#     df_C = d[3]
#     distrib = d[1]
#     (a_min,a_max) = d[6]
#     a = df_QG.index.get_level_values('aeff')
    
#     # plt.scatter(a,i*np.ones_like(a))
#     # print(a)
    
#     df_wave = df_QG.iloc[df_QG.index.get_level_values('wave') == df_QG.index.get_level_values('wave')[1]]
#     df_a = np.unique(df_QG.index.get_level_values('aeff'))
#     ii = np.searchsorted(a_all, df_a)
#     a_all = np.insert(a_all, ii, df_a)
    
#     # a_list[i] = df_a
#     # i+=1
    
#     plt.plot(df_a,df_wave['Qext'])
    
    
#     ### For simple polaris tests, with a modified distribution    
#     # if 'Qext1' in df_QG:
#     #     df_QG = df_QG.drop(columns=["Qabs1","Qsca1","Qext1","Qabs2","Qsca2","Qext2"])
#     # distrib = lambda a : a**(-3.5)*np.heaviside(a - 1e-8,1)* np.heaviside(2.1e-8 - a,0)
#     ### End of simple test
    
    
    
#     dens = df_QG["dens"] #convert and divide by what is ok
#     dens_m = integ_dust_size(distrib(a)*a**3*dens,a,a_min,a_max)/integ_dust_size(distrib(a)*a**3,a,a_min,a_max)# TODO : finish and test the density computation averaging.
#     if name.endswith("prolate2"):
#         ratio = 1/2
#     elif name.endswith("prolate1.3"):
#         ratio = 1/1.3
#     elif name.endswith("oblate2"):
#         ratio = 2
#     elif name.endswith("oblate1.3"):
#         ratio = 1.3
#     else:
#         ratio=1
        
#     sub_temp = 1500
#     delta = 1
    
        
        
#     write_polaris_Q_file(path+'/Polaris2',name,df_QG,ratio,dens_m,sub_temp,delta)
#     write_polaris_C_file(path+'/Polaris2',name,df_C)
    
    
# a_all = np.unique(a_all)


# # plt.plot(a_all,np.interp(a_all,df_wave.index.get_level_values('aeff'),df_wave['Qext'],left=0,right =0),marker ="+")
# plt.show()


# plt.figure()

# for d in my_dust:
#     name = d[0]
#     df_QG = d[2]
#     df_C = d[3]
#     distrib = d[1]
    # (a_min,a_max) = d[6]
#     a = df_QG.index.get_level_values('aeff')
    
#     # plt.scatter(a,i*np.ones_like(a))
#     # print(a)
    
#     df_wave = df_QG.iloc[df_QG.index.get_level_values('wave') == df_QG.index.get_level_values('wave')[1]]
#     df_a = np.unique(df_QG.index.get_level_values('aeff'))
    
#     # a_list[i] = df_a
#     # i+=1
    
    
#     dens = df_QG["dens"]
#     dens_m = integ_dust_size(distrib(a)*a**3*dens,a,a_min,a_max)/integ_dust_size(distrib(a)*a**3,a,a_min,a_max)# TODO : finish and test the density computation averaging.
#     if name.endswith("prolate2"):
#         ratio = 1/2
#     elif name.endswith("prolate1.3"):
#         ratio = 1/1.3
#     elif name.endswith("oblate2"):
#         ratio = 2
#     elif name.endswith("oblate1.3"):
#         ratio = 1.3
#     else:
#         ratio=1
        
#     sub_temp = 1500
#     delta = 1
    
#     df_QG = write_polaris_Q_file(path+'/Polaris',name+'_bis_expand',df_QG,ratio,dens_m,sub_temp,delta,expand_a = a_all)
#     df_C = write_polaris_C_file(path+'/Polaris',name+'_bis_expand',df_C,expand_a = a_all,original_a=df_a)
    
#     df_wave = df_QG.iloc[df_QG.index.get_level_values('wave') == df_QG.index.get_level_values('wave')[1]]
#     df_a = np.unique(df_QG.index.get_level_values('aeff'))
    
#     plt.plot(df_a,df_wave['Qext'])

# # plt.scatter(a_all,i*np.ones_like(a_all))







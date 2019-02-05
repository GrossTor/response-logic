#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 18:01:05 2019

@author: torsten
"""

import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd
import os
from scipy.stats import ttest_ind
import pickle

analysis_folder=os.path.dirname(__file__)
data_folder=os.path.join(analysis_folder, 'data/')

SW48_raw=pd.read_csv(os.path.join(data_folder,'SW-48_RPPA_perturbation_data.csv'))

readouts=SW48_raw.readout.unique()
inhibitors=SW48_raw.inhibitor.unique()
ligands=SW48_raw.ligand.unique()

inhibitor_alias={
    'Erlotinib': 'EGFRi',
    'GDC-0068': 'AKTi',
    'GDC-0973':'MEKi',
    'DMSO':'DMSO'
        }

perturbations=[l for l in ligands if l!='PBS']
perturbations.extend([inhibitor_alias[i] for i in inhibitors if i!='DMSO'])

meta_data={'perturbations':perturbations,
           'readouts':readouts,
           }

with open(os.path.join(data_folder,'meta_data.pkl'), 'wb') as handle:
    pickle.dump(meta_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

SW48_raw.inhibitor.replace(inhibitor_alias,inplace=True)



#%%build response matrix
mean_raw=SW48_raw.groupby([ 'cell line', 'inhibitor', 'ligand', 'readout']).mean()\
    .drop(columns='TP conc. (mg/mL)').reset_index()

lists_raw=SW48_raw.groupby([ 'cell line', 'inhibitor', 'ligand', 'readout']).apply(
        lambda g: [val for val in g.raw_val.tolist() if not np.isnan(val)])\
        .reset_index().rename(columns={0:'raw_val_list'})

#For inhibitors I extract only the combination experiment with the ligand that 
#caused the largest stimulation at the inhibited kinase. Like this:
response_inhib=mean_raw.loc[
        ( (mean_raw.inhibitor=='EGFRi') & (mean_raw.ligand=='EGF') ) | \
        ( (mean_raw.inhibitor=='MEKi') & (mean_raw.ligand=='EGF') ) | \
        ( (mean_raw.inhibitor=='AKTi') & (mean_raw.ligand=='IGF') ) ]

response_inhib=pd.merge(response_inhib,lists_raw,how='left',
                        on=['cell line','readout','ligand','inhibitor'])

#and I normalize it to the according stimulation
response_inhib=pd.merge(response_inhib,mean_raw.loc[
        (mean_raw.inhibitor=='DMSO'),['cell line','ligand','readout','raw_val']
        ],
        how='left', on=['cell line','readout','ligand'], suffixes=['','_base'] )

response_inhib=pd.merge(response_inhib,lists_raw.loc[
        (lists_raw.inhibitor=='DMSO'),['cell line','ligand','readout','raw_val_list']
        ],
        how='left', on=['cell line','readout','ligand'], suffixes=['','_base'] )

response_inhib=response_inhib.drop(columns='ligand').rename(columns={'inhibitor':'perturbation'})

#For ligands I select only the uninhibited stimulation
response_stimu=mean_raw.loc[
        ( (mean_raw.inhibitor=='DMSO') & (mean_raw.ligand=='EGF') ) | \
        ( (mean_raw.inhibitor=='DMSO') & (mean_raw.ligand=='IGF') ) | \
        ( (mean_raw.inhibitor=='DMSO') & (mean_raw.ligand=='HGF') )]
        
response_stimu=pd.merge(response_stimu,lists_raw,how='left',
                        on=['cell line','readout','ligand','inhibitor'] )

#and normalize to unperturbed
response_stimu=pd.merge(response_stimu,mean_raw.loc[
        (mean_raw.inhibitor=='DMSO') & (mean_raw.ligand=='PBS'),['cell line','readout','raw_val']
        ],
        how='left', on=['cell line','readout'], suffixes=['','_base'] )

response_stimu=pd.merge(response_stimu,lists_raw.loc[
        (lists_raw.inhibitor=='DMSO')& (lists_raw.ligand=='PBS'),['cell line','readout','raw_val_list']
        ],
        how='left', on=['cell line','readout'], suffixes=['','_base'] )

response_stimu=response_stimu.drop(columns='inhibitor').rename(columns={'ligand':'perturbation'})

response=response_inhib.append(response_stimu)

response['p_val']=response.apply(lambda r: ttest_ind(r.raw_val_list,r.raw_val_list_base)[1],axis=1)

response.to_pickle(os.path.join(data_folder, 'response.pkl'))


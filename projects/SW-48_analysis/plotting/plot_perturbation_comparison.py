#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:05:52 2018

@author: grosstor
"""

import seaborn as sns
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

analysis_folder=os.path.join(os.path.dirname(__file__),'../')
plot_folder=os.path.join(analysis_folder, 'plots/')
data_folder=os.path.join(analysis_folder, 'data/')



SW48_raw=pd.read_csv(os.path.join(data_folder,'SW-48_RPPA_perturbation_data.csv'))

inhibitor_alias={
    'Erlotinib': 'EGFRi',
    'GDC-0068': 'AKTi',
    'GDC-0973':'MEKi',
    'DMSO':'DMSO'
        }
SW48_raw.inhibitor.replace(inhibitor_alias,inplace=True)


readouts=['EGFR.Y1068.XP','PDGF.Rec.beta.Y751','Shc.Y317.1',
           'ERK1.2.T202.Y204','p90.RSK.S380','Akt.S473.XP.1','p70.S6.Ki?se.T412',
           'S6.Ribo.Prot..S235.236..XP','Bad.S112','eIF4G.S1108'
           ]
player_names_to_clear_names={
     'EGFR.Y1068.XP':'EGFR Y1068',
     'PDGF.Rec.beta.Y751': 'PDGFRB Y751',
     'Shc.Y317.1':'Shc Y317',
     'ERK1.2.T202.Y204':'ERK1/2 T202/Y204',
     'p90.RSK.S380':'p90RSK S380',
     'Akt.S473.XP.1': 'Akt S473',
     'p70.S6.Ki?se.T412':'p70RSK T412',
     'S6.Ribo.Prot..S235.236..XP':'RPS6 S235/236',
     'Bad.S112':'Bad S112',
     'eIF4G.S1108':'eIF4G S1108',
        }


sns.set_style('white')



x_label_order=['EGFRi','MEKi','AKTi','EGF','HGF','IGF']

fig,axs=plt.subplots(nrows=len(readouts),ncols=3,figsize=[6.5,7.5],sharex=True,sharey='row')
plt.subplots_adjust(left=.06, bottom=0.125, right=0.75, top=.97,wspace=0.1, hspace=0.12)
for row,readout in enumerate(readouts):
    for col,cell_line in enumerate(['SW48 parental', 'SW48 E545K', 'SW48 H1047R']):
        ax=axs[row,col]
        data=SW48_raw[(SW48_raw['readout']==readout) & (SW48_raw['cell line']==cell_line)]
        comparison=pd.DataFrame(columns=pd.Index(['perturbation','raw_val','perturbed']))
        
        prt=data.loc[(data.inhibitor=='EGFRi') & (data.ligand=='EGF')].copy()
        prt['perturbed']='perturbed'
        prt['perturbation']='EGFRi'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')
        prt=data.loc[(data.inhibitor=='DMSO') & (data.ligand=='EGF')].copy()
        prt['perturbed']='unperturbed'
        prt['perturbation']='EGFRi'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')
        
        prt=data.loc[(data.inhibitor=='MEKi') & (data.ligand=='EGF')].copy()
        prt['perturbed']='perturbed'
        prt['perturbation']='MEKi'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')
        prt=data.loc[(data.inhibitor=='DMSO') & (data.ligand=='EGF')].copy()
        prt['perturbed']='unperturbed'
        prt['perturbation']='MEKi'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')
        
        prt=data.loc[(data.inhibitor=='AKTi') & (data.ligand=='IGF')].copy()
        prt['perturbed']='perturbed'
        prt['perturbation']='AKTi'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')
        prt=data.loc[(data.inhibitor=='DMSO') & (data.ligand=='IGF')].copy()
        prt['perturbed']='unperturbed'
        prt['perturbation']='AKTi'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')

        prt=data.loc[(data.inhibitor=='DMSO') & (data.ligand=='EGF')].copy()
        prt['perturbed']='perturbed'
        prt['perturbation']='EGF'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')
        prt=data.loc[(data.inhibitor=='DMSO') & (data.ligand=='PBS')].copy()
        prt['perturbed']='unperturbed'
        prt['perturbation']='EGF'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')

        prt=data.loc[(data.inhibitor=='DMSO') & (data.ligand=='HGF')].copy()
        prt['perturbed']='perturbed'
        prt['perturbation']='HGF'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')
        prt=data.loc[(data.inhibitor=='DMSO') & (data.ligand=='PBS')].copy()
        prt['perturbed']='unperturbed'
        prt['perturbation']='HGF'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')

        prt=data.loc[(data.inhibitor=='DMSO') & (data.ligand=='IGF')].copy()
        prt['perturbed']='perturbed'
        prt['perturbation']='IGF'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')
        prt=data.loc[(data.inhibitor=='DMSO') & (data.ligand=='PBS')].copy()
        prt['perturbed']='unperturbed'
        prt['perturbation']='IGF'
        comparison=comparison.append(prt[['perturbation','raw_val','perturbed']],ignore_index='perturbed')
        
        
        
        sns.stripplot(x='perturbation',y='raw_val',hue='perturbed',data=comparison,
                      order=x_label_order,hue_order=['perturbed','unperturbed'],
                    #width=0.85,linewidth=1.,
                    dodge=True,jitter=False,s=3,palette={'perturbed':'#e7745b','unperturbed':'#5a78e4'},#'tab10',
                    ax=ax)
        #5a78e4
        #e7745b
        
        xticklabels=x_label_order
        x_min,x_max=ax.get_xlim()
        #inhib_borders=np.linspace(xticks[0],xticks[-1],len(xticklabels)+1)
        inhib_borders=np.linspace(x_min,x_max,len(xticklabels)+1)
        
        for line_x in inhib_borders[1:-1]:
            ax.axvline(line_x,color=[0.75,0.75,0.75],linewidth=.5,zorder=0)

        if row==len(readouts)-1:
            
            xticks=inhib_borders[1:]-(inhib_borders[1]-inhib_borders[0])/2
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels,rotation=90)
            
            if col!=1:
                ax.set_xlabel('')
            else:
                #pass
                ax.set_xlabel(ax.get_xlabel(),labelpad=10)

#            xtick_labels=ax.get_xticklabels()
#            ax.set_xticklabels(xtick_labels,rotation=90)
        elif row==0:
            ax.set_title(cell_line)
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        if col==2:
            ax.yaxis.set_label_position("right")
            #ytick_labels=ax.get_yticklabels()
            #ax.set_yticklabels(ytick_labels)
            
            ax.set_ylabel(player_names_to_clear_names[readout],rotation=0,labelpad=5,
                          **{'horizontalalignment':'left','verticalalignment':'center'})

        elif col==0:
            ax.locator_params(axis='y',prune='both',min_n_ticks=2)
            ax.set_ylabel('')
        else:
            #ax.set_yticklabels([])
            ax.set_ylabel('')
        
        if col!=1 or row!=len(readouts)-1:
            ax.legend_.remove()
        else:
            ax.legend(#title='perturbed',
                      loc='upper center',
                      bbox_to_anchor=(2.18, -1.02),ncol=4,frameon=True)
        
        
#        basal=data[ (data['inhibitor']=='DMSO') & (data['ligand']=='PBS')]
#        data['perturbation']=data.apply(lambda r: r['inhibitor']

plt.savefig(os.path.join(plot_folder, 'perturbation_comparison.pdf'))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
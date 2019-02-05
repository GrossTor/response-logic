#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:06:43 2018

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


SW48_raw['ligand/inhibitor']=SW48_raw.apply(lambda r: '{0}/{1}'.format(r['ligand'], r['inhibitor']),1)

plot_order_ligands=['EGF', 'HGF', 'IGF', 'PBS']
plot_order_inhibs=['EGFRi', 'MEKi','AKTi', 'DMSO']

x_order=['{0}/{1}'.format(ligand,inhib) for inhib in plot_order_inhibs for ligand in plot_order_ligands]

sns.set_style('white')

fig,axs=plt.subplots(nrows=len(readouts),ncols=3,figsize=[6.5,7.5],sharex=True,sharey='row')
plt.subplots_adjust(left=.06, bottom=0.155, right=0.75, top=.97,wspace=0.1, hspace=0.15)
for row,readout in enumerate(readouts):
    for col,cell_line in enumerate(['SW48 parental', 'SW48 E545K', 'SW48 H1047R']):
        ax=axs[row,col]
        data=SW48_raw[(SW48_raw['readout']==readout) & (SW48_raw['cell line']==cell_line)]
        sns.stripplot('ligand/inhibitor','raw_val','ligand',data,palette='tab10',
                      order=x_order,hue_order=plot_order_ligands,
                      ax=ax,s=3)
        
        
        xticklabels=plot_order_inhibs
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
                ax.set_xlabel(ax.get_xlabel(),labelpad=30)

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
            ax.legend(#title='ligand',
                      loc='upper center',
                      bbox_to_anchor=(.5, -1.08),ncol=4,frameon=True)
            
    
    
#plt.tight_layout(h_pad=.01,w_pad=.3)
#plt.tight_layout(rect=(0,0,.85,1))
plt.savefig(os.path.join(plot_folder,'raw_data.pdf'))
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 21:25:39 2017

@author: torsten
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import response_logic

N=5
Q=4
perturbed_nodes=[0,2,3,4]


#%%set up model
Sen_known=np.zeros([N,Q])
for i,pert_node in enumerate(perturbed_nodes):
    Sen_known[pert_node,i]=np.nan


z_thresh=1

z_vals=np.loadtxt('z_vals.tsv')
response_pattern = z_vals > z_thresh
confidence_pattern = np.full(response_pattern.shape,np.nan)
confidence_pattern[response_pattern]=(z_vals[response_pattern]-z_thresh)\
                    /(z_vals[response_pattern].max()-1)
confidence_pattern[~response_pattern]=(z_thresh-z_vals[~response_pattern])/z_thresh

response_pattern=response_pattern.astype(float)
response_pattern[np.isnan(z_vals)]=np.nan
confidence_pattern[np.isnan(z_vals)]=np.nan

model={
       'N':N,
       'Q':Q,
       'input':{
               'node_names':[str(i) for i in range(N)],#[str(i) for i in nodes],
               'pert_names':[str(i) for i in perturbed_nodes],#[str(i) for i in targets],
               'edges_known':{(1,2):1,(0,2):0},
               'Sen_known':Sen_known,
               },
       'response':{
               'nonzero':response_pattern,
               'confidence':confidence_pattern,
               }
       }

#%%infer networks

#no heuristics
response_logic.conform_response_pattern(model,scheme='general')
response_logic.brave_solving(model)
brave=model['response']['brave edge array']

#heuristics: parsimonious
response_logic.conform_response_pattern(model,scheme='general',heuristics={'parsimonious':True})
response_logic.brave_solving(model,heuristics={'parsimonious':True})
brave_parsimonious=model['response']['brave edge array']

#iterate solutions
all_parsimonious_nets=response_logic.iterate_conforming_networks(model,'all',heuristics={'parsimonious':True})

#local link number heuristics: 
response_logic.conform_response_pattern(model,scheme='general',heuristics={
        'local_max_nbr_outedges':[['1',1]] })
response_logic.brave_solving(model,heuristics={
        'local_max_nbr_outedges':[['1',1]] })
brave_local_constraint=model['response']['brave edge array']

#global number heuristics: 
response_logic.conform_response_pattern(model,scheme='general',heuristics={
        'global_min_nbr_inedges':1})
response_logic.brave_solving(model,heuristics={
        'global_min_nbr_inedges':1 })
brave_global_constraint=model['response']['brave edge array']




#%%Plot results
sns.set_style('whitegrid')
fig,axs=plt.subplots(nrows=4,ncols=2,figsize=[7,18],sharex=False,sharey=False)

sns.heatmap(z_vals,square=True,ax=axs[0,0])
axs[0,0].set_title('z-values')
sns.heatmap(2*(response_pattern-.5)*confidence_pattern,center=0,square=True,cmap='coolwarm',ax=axs[0,1],cbar=True)
axs[0,1].set_title('Resonse pattern')
sns.heatmap(brave,square=True,cmap='Greys',cbar=False,ax=axs[1,0])
axs[1,0].set_title('Union of all conforming networks')
sns.heatmap(brave_parsimonious,square=True,cmap='Greys',cbar=False,ax=axs[1,1])
axs[1,1].set_title('Union of parsimonious networks')
sns.heatmap(all_parsimonious_nets[0][0],square=True,cmap='Greys',cbar=False,ax=axs[2,0])
axs[2,0].set_title('Parsimonious network 1')
sns.heatmap(all_parsimonious_nets[0][1],square=True,cmap='Greys',cbar=False,ax=axs[2,1])
axs[2,1].set_title('Parsimonious network 2')
sns.heatmap(brave_local_constraint,square=True,cmap='Greys',cbar=False,ax=axs[3,0])
axs[3,0].set_title('at most one link out of node 1')
sns.heatmap(brave_global_constraint,square=True,cmap='Greys',cbar=False,ax=axs[3,1])
axs[3,1].set_title('at least one link into each node')


for i,j in ([0,0],[0,1]):
    axs[i,j].set_xlabel('Perturbed node')
    axs[i,j].set_ylabel('Observed node')


for i,j in ([1,0],[1,1],[2,0],[2,1],[3,0],[3,1]):
    axs[i,j].set_xlabel('Source node')
    axs[i,j].set_ylabel('Target node')
    for source,target in model['input']['edges_known'].keys():
        axs[i,j].scatter(source+0.5,target+0.5,marker='*',color='forestgreen',s=150)

plt.tight_layout()
plt.savefig('toy_model_inference.pdf')
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:26:22 2018

@author: grosstor
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np


analysis_folder=os.path.dirname(__file__)
KEGG_folder=os.path.join(analysis_folder, './KEGG_data/')
result_folder=os.path.join(analysis_folder, './results/')
plot_folder=os.path.join(analysis_folder,'./plots/')


#%%plot KEGG ensemble predictions

with open ('parsed_human_KEGG_pathways.pkl', 'rb') as fp:
    pathways = pd.DataFrame(pickle.load(fp))
pathways['N']=pathways['nodes'].apply(len)

classification_results_list=[]

for filename in os.listdir(result_folder):
    if filename[:16]=='result_KEGG_run_':
        with open(os.path.join(result_folder,filename), 'rb') as handle:
            classification_results_list.append( pickle.load(handle) )
        
classification=pd.DataFrame(list(np.concatenate(classification_results_list)))

classification_counts=classification.groupby(['KEGG_id','rate_misclassification','rate_missing_data']).count()
classification_mean=classification.groupby(['KEGG_id','rate_misclassification','rate_missing_data']).mean()
classification_mean=classification_mean.join( pathways[['KEGG_id','class']].set_index('KEGG_id'))


classification_mean=classification_mean.reset_index().melt(
        id_vars=['KEGG_id','N','class','effective_node_ratio','rate_misclassification','rate_missing_data'],
        value_vars=['precision','sensitivity'],
        value_name='mean')#,var_name=' ')

#take out metabolism (only gene regulation and signal processing)
classification_mean=classification_mean[~(classification_mean['class']=='Metabolism')]

classification_mean.rename(columns={'class':'KEGG pathway classification',
                                    'rate_missing_data':'$\epsilon_M$',
                                    'rate_misclassification':'$\epsilon_C$',
                                    },inplace=True)
classification_mean['data quality']=['$\epsilon_M={0}\quad\epsilon_C={1}$'.format(eM,eC) for eM,eC in zip(
        classification_mean['$\epsilon_M$'], classification_mean['$\epsilon_C$'])]

    
g=sns.FacetGrid(data=classification_mean[(classification_mean['$\epsilon_M$'].isin([0,0.1])) & (classification_mean['$\epsilon_C$'].isin([0,0.1]))],
                                         size=2.1,aspect=1.,legend_out=True,hue='data quality',
                                         col='variable',row=None,margin_titles=False)
def dateplot(**kwargs):
    ax = plt.gca()
    #print(kwargs.keys(),kwargs['color'])
    cmap=sns.light_palette(kwargs['color'],n_colors=6,as_cmap=True)
    data = kwargs.pop("data")
    sns.kdeplot(data['N'], data['mean'],cmap=cmap, cut=0,n_levels=5, shade=False,shade_lowest=False,ax=ax,kwargs={'alpha':0.5})#, **kwargs)

g=(g.map(plt.scatter,'N','mean',s=5,alpha=0.5,linewidths=.4))#,zorder=20
g.add_legend()

for ax in g.axes.flat:
    plt.setp(ax.texts, text="")
g.set_titles(row_template="{row_name}", col_template="{col_name}")

g.savefig(os.path.join(plot_folder,'mean_classification_scores_quality.pdf'))


#%%plot selected pathway predictions

classification_results_list=[]
for filename in os.listdir(result_folder):
    if filename[:19]=='single_pathway_run_' and filename[-5].isdecimal():
        with open(os.path.join(result_folder,filename), 'rb') as handle:
            classification_results_list.append( pickle.load(handle) )
single_pathways=pd.DataFrame([j for i in classification_results_list for j in i])            

single_pathways=single_pathways.join( pathways[['KEGG_id','title']].set_index('KEGG_id'),on='KEGG_id')
#single_pathways=single_pathways.reset_index().melt(
#        id_vars=['KEGG_id','N','title','effective_node_ratio','rate_misclassification','rate_missing_data'],
#        value_vars=['precision','sensitivity','specificity'],
#        value_name='score')#,var_name=' ')
single_pathways=single_pathways.reset_index().melt(
        id_vars=['KEGG_id','N','title','effective_node_ratio','rate_misclassification','rate_missing_data'],
        value_vars=['precision','sensitivity'],
        value_name='score')#,var_name=' ')
counts=single_pathways.groupby(['title','rate_missing_data','rate_misclassification']).count()


chosen_errors_M=[0,0.1,0.2,0.3,0.4,0.5]
chosen_errors_C=[0,0.05,0.1,0.15,0.2,0.25]#0.3,0.4,0.5]

single_pathways.rename(columns={'rate_missing_data':'$\epsilon_M$',
                                'rate_misclassification':'$\epsilon_C$',
                                },inplace=True)

g=sns.factorplot(data=single_pathways[(single_pathways['$\epsilon_C$']==0 ) &
                    (single_pathways['$\epsilon_M$'].isin(chosen_errors_M))],
                 x='$\epsilon_M$',y='score',kind="box", dodge=True,
                 hue='title',linewidth=.5,width=0.65,
                size=2.2,aspect=1.1,legend_out=True,fliersize=1,
                col='variable',row=None,margin_titles=False)
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.savefig(os.path.join(plot_folder,'single_pathways_missing_screen.pdf'))



g=sns.factorplot(data=single_pathways[(single_pathways['$\epsilon_M$']==0 ) &
                    (single_pathways['$\epsilon_C$'].isin(chosen_errors_C))],
                 x='$\epsilon_C$',y='score',kind="box", dodge=True,
                 hue='title',linewidth=.5,width=0.65,
                size=2.2,aspect=1.1,legend_out=True,fliersize=1,
                col='variable',row=None,margin_titles=False)
g.set_titles(row_template="{row_name}", col_template="{col_name}")


g.savefig(os.path.join(plot_folder,'single_pathways_misclas_screen.pdf'))





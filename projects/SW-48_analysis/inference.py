#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:31:27 2018

@author: grosstor
"""

draw_graphs=True #set to False if there are problems with graph_viz

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd
import itertools as it
import os
import pickle

import response_logic
import inference_input
if draw_graphs==True:
    from plotting.draw_graph import draw_graph


rank_randomization_repeats=100

analysis_folder=os.path.dirname(__file__)
plot_folder=os.path.join(analysis_folder, 'plots/')
result_folder=os.path.join(analysis_folder, 'results/')
data_folder=os.path.join(analysis_folder, 'data/')

with open(os.path.join(data_folder,'meta_data.pkl'), 'rb') as handle:
    meta_data = pickle.load(handle)
    
perturbations=meta_data['perturbations']
readouts=meta_data['readouts']

gold_edges=inference_input.gold_edges
pert_to_players=inference_input.pert_to_players

readouts_shortnames=[inference_input.readouts2short_name[readout] for readout in readouts]

readout_to_ind={readout:i for i,readout in enumerate(readouts)}

gold_edges_dict={}
for edge in it.product(readouts,repeat=2):
    if edge in gold_edges:
        gold_edges_dict[readout_to_ind[edge[0]],readout_to_ind[edge[1]]]=1
    else:
        gold_edges_dict[readout_to_ind[edge[0]],readout_to_ind[edge[1]]]=0

pert_to_players_shortnames={pert:inference_input.readouts2short_name[player] for pert,player in pert_to_players.items()}


def generate_conform_model(edges_known,heuristics,plot_response_matrices=False):
    thresh=.01 
    response=pd.read_pickle(os.path.join(data_folder, 'response.pkl'))

    response_cl=response.loc[response['cell line']==cl].copy()
    response_matrix=response_cl.pivot(index='readout',columns='perturbation',values='p_val')
    response_matrix=response_matrix.reindex(index=readouts,columns=perturbations)
    
    #determine nonzero and confidence matrices
    nonzero = (response_matrix < thresh).astype(float)
    nonzero[response_matrix.isna()]=np.nan
    
    response_cl['thresh_dist']=np.abs(response_cl['p_val']-thresh)
    response_cl['confidence']=response_cl.thresh_dist.rank()
    response_cl['confidence']=response_cl['confidence']/response_cl['confidence'].max()
    confidence=response_cl.pivot(index='readout',columns='perturbation',values='confidence')
    confidence=confidence.reindex(index=readouts,columns=perturbations)
    

    if plot_response_matrices:
        fig,axs=plt.subplots(ncols=2,figsize=[8,5],sharey=True)
        sns.heatmap(nonzero,square=True,ax=axs[0])
        sns.heatmap(confidence,square=True,ax=axs[1])
        plt.show()
    
    Sen_known=np.zeros([len(readouts),len(perturbations)])
    for col,pert in enumerate(perturbations):
        for row,readout in enumerate(readouts):
            if readout==pert_to_players[pert]:
                Sen_known[row,col]=np.nan
     
    model={'N':len(readouts),
           'P':len(perturbations),
           'Q':len(perturbations),
          'input':{
           'node_names':readouts,
           'pert_names':perturbations,
           'edges_known':edges_known,
           'Sen_known':Sen_known
           },
       'response':{
           'nonzero':nonzero.values,
           'confidence':confidence.values
           }
       }
    
    #to conform the response pattern the parsimonious heuristic does not make a difference,
    #because I only want to identify a conform response pattern, and if there is a
    #non-parsimonious net I know there is also a parsimonious one with the same response
    #pattern so I don't have to discard it. So I remove this heuristic for this step
    #because it is much faster
    conform_heuristics=heuristics.copy()
    conform_heuristics['parsimonious']=False
    response_logic.conform_response_pattern(model,scheme='general',heuristics=conform_heuristics,fact_timeout=100)
    
    return model


def plot_conforming_networks(model,heuristics,nbr_of_conforming_networks=5,file_suffix='',graph_label=''):
    conforming_networks,SolveResults=response_logic.iterate_conforming_networks(
            model,heuristics=heuristics,nbr_answer_sets=nbr_of_conforming_networks)
    for i,conforming_network in enumerate(conforming_networks):
        print(i)
        single_net=pd.DataFrame(conforming_network,index=readouts,columns=readouts)
        sns.heatmap(single_net,square=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder,'{0}_single_{1}.pdf'.format(plot_prefix,i)))
        plt.show()
        if draw_graphs==True:
            draw_graph(single_net,readouts_shortnames,os.path.join(plot_folder,'graphs/{0}_single_net_{1}_{2}'.format(
                    plot_prefix,file_suffix,i)),
                    perturbations=perturbations,pert_to_players=pert_to_players_shortnames,
                    label='single net {0} {1}'.format(i,graph_label))
    return

    
def plot_brave_network(model,heuristics,file_suffix='',graph_label=''):
    response_logic.brave_solving(model,heuristics=heuristics)
    brave_net=pd.DataFrame(model['response']['brave edge array'],index=readouts,columns=readouts)
    fix,ax=plt.subplots(figsize=[6,6])
    sns.heatmap(brave_net,square=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder,'{0}_brave.pdf'.format(plot_prefix)))
    plt.show()
    if draw_graphs==True:
        draw_graph(brave_net.values,readouts_shortnames,os.path.join(plot_folder,'graphs/{0}_brave_net_{1}'.format(plot_prefix,file_suffix)),
                            perturbations=perturbations,pert_to_players=pert_to_players_shortnames,
                            label='brave_net {}'.format(graph_label))
    return


def score_brave_model(model,heuristics,edges_known,rank_randomization_repeats,brave_array=None):
    '''Computes brave model. To compute AUCs score, ties are broken randomly.
    This is repeated and the averages are returned.
    If brave_array is given it is not re-computed.'''
    
    if brave_array is None:
        response_logic.brave_solving(model,heuristics=heuristics)
        brave_array=model['response']['brave edge array']
    pr_aucs=[]
    roc_aucs=[]
    for i in range(rank_randomization_repeats):
        randomly_ranked_prediction=response_logic.convenience.break_score_ties(
                brave_array,np.random.rand(len(readouts),len(readouts)))
        score_dict=response_logic.convenience.score_binary_classifier(gold_edges_dict,randomly_ranked_prediction,edges_known)
        pr_aucs.append(score_dict['pre_rec_auc'])
        roc_aucs.append(score_dict['roc_auc'])
    
    return {'ROC auc': np.mean(roc_aucs), 'PR auc':np.mean(pr_aucs)}

def score_scenarios(score_average_model=True):
    label_list=['no network knowledge, no heuristics',
                'partial network knowledge, no heuristics',
                'partial network knowledge, parsimonious constraint',
                'partial network knowledge, heuristics']
    
    short_label=['A','B','C','D']
    
    heuristics_list=[{'parsimonious': False},
                     {'parsimonious': False},
                     {'parsimonious': True},
                     heuristics,
                     ]
    
    
    pk_list=[{(i,i):1 for i in range(len(readouts))},
              edges_known,
              edges_known,
              edges_known]
    
    aucs_scenarios=[]
    for label,short_label,parsimonious,pk in zip(label_list,short_label,heuristics_list,pk_list):
        model=generate_conform_model(pk,parsimonious,plot_response_matrices=False)
        
        #plot_brave_network(model,parsimonious,file_suffix='parental_{}'.format(short_label),graph_label='scenario_plot')
        
        aucs=score_brave_model(model,parsimonious,pk,rank_randomization_repeats)
        prediction_edges=set(gold_edges_dict.keys())-set(pk.keys())
        positives=np.sum([gold_edges_dict[edge] for edge in prediction_edges])
        negatives=len(prediction_edges)-positives
        random_pr_auc=positives / (positives + negatives)
        random_roc_auc=0.5
        aucs_scenarios.append({'random_pr_auc':random_pr_auc,'random_roc_auc':random_roc_auc,
                               'label':label,'short_label':short_label})
        aucs_scenarios[-1].update(aucs)
    
    
    nbr_answer_sets=1000
    if score_average_model==True:
        conforming_networks,SolveResults=response_logic.iterate_conforming_networks(
            model,heuristics=heuristics,nbr_answer_sets=nbr_answer_sets)
        nbr_conforming_nets=len(conforming_networks)
        print(nbr_conforming_nets)
      
    if nbr_conforming_nets<nbr_answer_sets:
        aucs=score_brave_model(None,None,pk,rank_randomization_repeats,
                               brave_array=np.average(conforming_networks,0) )
        prediction_edges=set(gold_edges_dict.keys())-set(pk.keys())
        positives=np.sum([gold_edges_dict[edge] for edge in prediction_edges])
        negatives=len(prediction_edges)-positives
        random_pr_auc=positives / (positives + negatives)
        random_roc_auc=0.5
    
        aucs_scenarios.append({'random_pr_auc':random_pr_auc,'random_roc_auc':random_roc_auc,
                           'label':'partial network knowledge, heuristics, average net',
                           'short_label':'E'})
        aucs_scenarios[-1].update(aucs)    
    
    
    aucs_scenarios=pd.DataFrame(aucs_scenarios)
    
    aucs_scenarios.to_csv(os.path.join(result_folder, 'aucs_scenarios.pkl'),index=False)
    return aucs_scenarios


#%%parental inference and scoring

cl='SW48 parental'
case='parental_inference'

prior_net_knowledge,heuristics,plot_prefix=inference_input.return_inference_information(case=case)
edges_known={(i,i):1 for i in range(len(readouts))}
edges_known.update({(readout_to_ind[key[0]],readout_to_ind[key[1]]):val for key,val in prior_net_knowledge.items()})


model=generate_conform_model(edges_known,heuristics,plot_response_matrices=False)
plot_brave_network(model,heuristics,file_suffix='parental_{}'.format('C'),graph_label='parental')

##%%count conforming networks
conforming_networks,SolveResults=response_logic.iterate_conforming_networks(
            model,heuristics=heuristics,nbr_answer_sets=1000)
print('Number of conforming networks = ', len(conforming_networks))

aucs_scenarios=score_scenarios()
print(aucs_scenarios)


#%% compare cell lines
case='cl_comparison'

prior_net_knowledge,heuristics,plot_prefix=inference_input.return_inference_information(case=case)
edges_known={(i,i):1 for i in range(len(readouts))}
edges_known.update({(readout_to_ind[key[0]],readout_to_ind[key[1]]):val for key,val in prior_net_knowledge.items()})

for cl in ['SW48 parental', 'SW48 E545K', 'SW48 H1047R']:
    #cl is global variable in generate_conform_model
    print('\n\n\n\n\n########### {} ##############'.format(cl))
    model=generate_conform_model(edges_known,heuristics,plot_response_matrices=False)
    print('Union of conforming networks:')
    plot_brave_network(model,heuristics,file_suffix=cl.replace(' ',''),graph_label=cl)
    conforming_networks,SolveResults=response_logic.iterate_conforming_networks(
                model,heuristics=heuristics,nbr_answer_sets=100)
    
    print('Number of conforming networks = ', len(conforming_networks),':')
    plot_conforming_networks(model,heuristics,nbr_of_conforming_networks=5,file_suffix=cl.replace(' ',''),graph_label=cl)
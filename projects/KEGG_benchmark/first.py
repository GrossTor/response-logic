#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:39:18 2018

@author: grosstor
"""

import networkx as nx
import numpy as np
import response_logic
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os 
import pickle
import multiprocessing

#import itertools as it


#set niceness of process
#os.nice()


analysis_folder=os.path.dirname(__file__)
KEGG_folder=os.path.join(analysis_folder, 'KEGG_data/')
result_folder=os.path.join(analysis_folder, 'results/')
#temp_folder=os.path.join(analysis_folder, '../temp/')


#rate_missing_data=0.1
#rate_misclassification=0.1 #amongst non-missing data
#rate_perturbed_nodes=1. #I take this one out, and only work with missing data



#generate the graph and its response pattern (with confidence scores)#
######################################################################

#init_G=nx.MultiDiGraph()
#init_G.add_node(0)
#gold_net=nx.DiGraph( nx.generators.directed.scale_free_graph(N,create_using=init_G).to_directed() )
#gold_net=nx.DiGraph( nx.generators.directed.scale_free_graph(N).to_directed() )

def generate_test_data(pathway,rate_missing_data=0.,rate_misclassification=0.):
    '''Given a networkx.DiGraph this function generates a "model" that can be 
    used to infer via response logic.
    '''
    
    gold_response=pathway['gold_response']

    nodes=pathway['nodes']
    Sen_known=pathway['Sen_known']

    
    edges_known={}#currently
    
    
    N=len(nodes)
    P=Q=N
    nonzero=np.array(gold_response,dtype=float)
    confidence=np.random.rand(*Sen_known.shape)
    
    #introduce random missing data points#
    ######################################
    missing_indeces_flat=np.random.choice(nonzero.size,int(nonzero.size*rate_missing_data),replace=False)
    actual_rate_missing_data=len(missing_indeces_flat)/nonzero.size
    nonzero.flat[missing_indeces_flat]=np.nan
    confidence.flat[missing_indeces_flat]=np.nan

    #introduce random misclassifications#
    #####################################
    #it is done in such a way that a misclassification is chosen according to the 
    #confidence of the according data point
    
    nonmissing_indeces_flat=np.array(list(set(np.arange(nonzero.size)) - set(missing_indeces_flat)))
    nbr_misclassifications=int(nonmissing_indeces_flat.size*rate_misclassification)
    actual_rate_misclassification=nbr_misclassifications/len(nonmissing_indeces_flat)
    misclassification_proba=1-confidence.flat[nonmissing_indeces_flat]
    misclassification_proba /= np.sum(misclassification_proba)
    
    misclassification_indeces_flat=np.random.choice(nonmissing_indeces_flat,
            size=nbr_misclassifications,p=misclassification_proba)
    
    nonzero.flat[misclassification_indeces_flat]= ~nonzero.flat[misclassification_indeces_flat].astype(bool)
    
    #plt.hist(confidence.flat[misclassification_indeces_flat],bins=50)
    model={
           'N':N,
           'P':P,
           'Q':Q,
           'input':{
                   'node_names':nodes,#[str(i) for i in nodes],
                   'pert_names':nodes,#[str(i) for i in targets],
                   'edges_known':edges_known,
                   'Sen_known':Sen_known
                   },
           'response':{
                   'nonzero':nonzero,
                   'confidence':confidence
                   },
           'extra_info':{
                   #'gold_edges':gold_edges,
                   #'extended_confidence':extended_confidence,
                   #'actual_rate_perturbed_nodes':actual_rate_perturbed_nodes,
                   'actual_rate_missing_data':actual_rate_missing_data,
                   'actual_rate_misclassification':actual_rate_misclassification,
                   },
           }
    
           
    return model


def score_pathway(pathway,rate_missing_data,rate_misclassification):
    model=generate_test_data(pathway,rate_missing_data,rate_misclassification)
    response_logic.conform_response_pattern(model,scheme='star-model')
    response_logic.brave_solving(model)
    
#    brave_net=pd.DataFrame(model['response']['brave edge array'],
#            index=model['input']['node_names'],columns=model['input']['node_names'])
    
    #sparsify#

    #order ambiguous links by first taking those belonging to zero then nan then nonzero datapoints
    #then order them according to their confidence (zeros) or inverse confidence (nonzeros) (not possible for nan)
    ambiguous_links_zero_mask=np.logical_and(
        model['response']['brave edge array']==0.5,model['response']['nonzero']==0)
    ambiguous_links_nan_mask=np.logical_and(
        model['response']['brave edge array']==0.5,np.isnan(model['response']['nonzero']))
    ambiguous_links_nonzero_mask=np.logical_and(
        model['response']['brave edge array']==0.5,model['response']['nonzero']==1)

    ordered_zero_candidates=np.concatenate((
        np.transpose(np.where(ambiguous_links_zero_mask))[
            np.argsort(model['response']['confidence'][ambiguous_links_zero_mask])][::-1],
        np.transpose(np.where(ambiguous_links_nan_mask)),                    
        np.transpose(np.where(ambiguous_links_nonzero_mask))[
            np.argsort(model['response']['confidence'][ambiguous_links_nonzero_mask])]))
    
    
    response_logic.sparsify_brave_structure(model,ordered_zero_candidates)
    
    #compute sensitivity, specitivity, precision
    Jac_known=response_logic.convenience.edges_known2Jac_known(model['input']['edges_known'],model['N'])
    np.fill_diagonal(Jac_known,0)
    relevant_selection=np.isnan(Jac_known)
    relevant_prediction=model['response']['brave edge array sparsified'][relevant_selection]
    relevant_gold=pathway['gold_Jac'][relevant_selection]
    positives_inds=relevant_prediction==1
    nbr_true_pos=np.sum(relevant_gold[positives_inds]==1)
    nbr_false_pos=np.sum(relevant_gold[positives_inds]==0)
    negative_inds=relevant_prediction==0
    nbr_true_neg=np.sum(relevant_gold[negative_inds]==0)
    nbr_false_neg=np.sum(relevant_gold[negative_inds]==1)
    sensitivity=nbr_true_pos/(nbr_true_pos+nbr_false_neg)
    specificity=nbr_true_neg/(nbr_true_neg+nbr_false_neg)
    precision=nbr_true_pos/(nbr_true_pos+nbr_false_pos)
    
    #store results
    classification_result={
             'N':model['N'],
             'KEGG_id':pathway['KEGG_id'],
             'effective_node_ratio':pathway['effective_node_ratio'],
             'sensitivity':sensitivity,'specificity':specificity,'precision':precision,
#             'rate_perturbed_nodes':rate_perturbed_nodes,
#             'actual_rate_perturbed_nodes':model['extra_info']['actual_rate_perturbed_nodes'],
             'rate_missing_data':rate_missing_data,
             'actual_rate_missing_data':model['extra_info']['actual_rate_missing_data'],
             'rate_misclassification':rate_misclassification,
             'actual_rate_misclassification':model['extra_info']['actual_rate_misclassification'],
             }
            
    return classification_result


    
def analyse_KEGG_pathways(arguments):
    #I need to reset random seed so that multiple workers don't yield the same
    #results.
    np.random.seed()
    
    
    rate_missing_data,rate_misclassification,file_suffix=arguments['rate_missing_data'],arguments['rate_misclassification'],arguments['file_suffix']
#    gold_net=pathway['net']
#    print(len(gold_net))
    
    #load the parsed KEGG data
    with open (os.path.join(analysis_folder,'parsed_human_KEGG_pathways.pkl'), 'rb') as fp:
        pathways = pickle.load(fp)
    
    classification_results=[]
    #for pathway in pathways[:4]:
        
    for pathway_i,pathway in enumerate(pathways):
        classification_result=score_pathway(pathway,rate_missing_data,rate_misclassification)
        classification_results.append(classification_result)
        
    with open(os.path.join(result_folder,'result_KEGG_run_{0}.pkl'.format(file_suffix)), 'wb') as handle:
        pickle.dump(classification_results, handle)
        
    return classification_results



def explore_single_pathway(path_run):
    np.random.seed()
    #pathway_title='TGF-beta signaling pathway'
    pathway_title=path_run['pathway_title']
    run_id=path_run['run_id']
    with open (os.path.join(analysis_folder,'parsed_human_KEGG_pathways.pkl'), 'rb') as fp:
        pathways = pickle.load(fp)
    pathways=pd.DataFrame(pathways)
    
    pathway=pathways[pathways['title']==pathway_title].squeeze()
    
    classification_results=[]
    rate_misclassification=0
    for rate_missing_data in np.linspace(0,0.5,6):
        print(rate_missing_data)
        classification_result=score_pathway(pathway,rate_missing_data,rate_misclassification)
        classification_results.append(classification_result)    
    with open(os.path.join(result_folder,'single_pathway_run_{0}_missing_data_{1}.pkl'.format(pathway['KEGG_id'],run_id)), 'wb') as handle:
        pickle.dump(classification_results, handle)
    
    classification_results=[]
    rate_missing_data=0
    for rate_misclassification in np.linspace(0,0.5,6):    
        print(rate_misclassification)
        classification_result=score_pathway(pathway,rate_missing_data,rate_misclassification)
        classification_results.append(classification_result)    
    with open(os.path.join(result_folder,'single_pathway_run_{0}_misclassification_{1}.pkl'.format(pathway['KEGG_id'],run_id)), 'wb') as handle:
        pickle.dump(classification_results, handle)
    return




single_pathway_run=False
entire_KEGG_run=True


if single_pathway_run:
    pathway_titles=['TGF-beta signaling pathway','Ras signaling pathway','Wnt signaling pathway']
    path_runs=[]
    nbr_repeats=10
    for rep in range(nbr_repeats):
        for pathway_title in pathway_titles:
            path_runs.append({'pathway_title':pathway_title,'run_id':rep})
    

    max_nbr_workers=10
    nbr_workers=min(max_nbr_workers,len(path_runs))    

    pool = multiprocessing.Pool(nbr_workers)
    classification_results = pool.map(explore_single_pathway, path_runs)
    pool.close()
    pool.terminate()


if entire_KEGG_run:
    parallelized_run=True
    
    if not parallelized_run:
        
        KEGG_arguments={'rate_missing_data':0.0,'rate_misclassification':0.0,'file_suffix':'errorless'}
        classification_results_no_error=analyse_KEGG_pathways(KEGG_arguments)
    else:
        nbr_parallelized_repeats=10
        rate_missing_data=0.0
        rate_misclassification=0.1
        max_nbr_workers=10
        KEGG_arguments_list=[]
        for i in range(nbr_parallelized_repeats):
            KEGG_arguments_list.append({'rate_missing_data':rate_missing_data,
                                        'rate_misclassification':rate_misclassification,
                                        'file_suffix':'_emiss_{0}_emisclass_{1}_repeat_{2}'.format(
            str(rate_missing_data).split('.')[-1],str(rate_misclassification).split('.')[-1],str(i))})
        
        nbr_workers=min(max_nbr_workers,nbr_parallelized_repeats)
        
        import time
        start_time=time.time()
        pool = multiprocessing.Pool(nbr_workers)
        classification_results = pool.map(analyse_KEGG_pathways, KEGG_arguments_list)
        pool.close()
        end_time=time.time()
        pool.terminate()
            
        print('For a quality vector of len {0}, inference took {1} minutes.'.format(nbr_parallelized_repeats,(end_time-start_time)/60.))
        classification=pd.DataFrame(list(np.concatenate(classification_results)))
        
        
        #read the parallelized file dumps
        classification_results_list=[]
        for KEGG_arguments in KEGG_arguments_list:        
            with open(os.path.join(result_folder,'result_KEGG_run_{0}.pkl'.format(KEGG_arguments['file_suffix'])), 'rb') as handle:
                classification_results_list.append( pickle.load(handle) )
        classification=pd.DataFrame(list(np.concatenate(classification_results)))
        classification.to_pickle(os.path.join(result_folder,'classification_joined_results.pkl'))
    



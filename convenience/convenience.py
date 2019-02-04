#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:51:42 2017

@author: grosstor
"""

__all__=('edges_known2Jac_known','score_binary_classifier','generate_response_pattern',
         'generate_Pexpt_from_pd_index','generate_DiGraph_from_edge_array_matrix',
         'break_score_ties','order_zero_candidates_by_nonzero_groups',
         'noisify_response_pattern',
         )

import numpy as np
from sklearn.metrics import precision_recall_curve,auc,roc_curve
import networkx as nx
import pandas as pd


def edges_known2Jac_known(edges_known,N):
    '''
    Dictionary edges_known describes three discrete states of prior knowledge 
    of each edge: present/absent/unknown (missing entry), which are required in
    response logic analysis. For steady state fitting prior knowledge about the
    Jacobian is required, which is for every entry of array Jac_known either a
    float or NaN (unknown). Here Jac_known is generated from edges_known by
    assuming that no Jacobian value is known for present edges (thus turning 
    True to NaN).
    '''
    Jac_known=np.full([N,N],np.nan)
    for (N1,N2),exists in edges_known.items():
        if not exists:
            Jac_known[N2,N1]=0
    return Jac_known


def generate_Pexpt_from_pd_index(response_pattern,node_names,non_perturbed_label=None):
    '''
    If perturbations of different experiments are indicated as an index object
    of some dataframe, one can use this function to extract the Pexpt matrix from
    it. This also works for Multiindex objects that would label perturbation-combinations.
    In that case one must provide the non_perturbed_label (indicating the unper-
    turbed state on some Multiindex level).
    The iterable node_names is required to ensure the correct row ordering of 
    Pexpt. node_names and annotated perturbation targets must be the same.
    '''
        
    perts_frame=response_pattern.columns.to_frame()
    
    #Panel is 3D stack of DataFrames.
    Pexpt=pd.Panel({col:pd.get_dummies(perts_frame[col]) for col in perts_frame}).any(axis=0)
    if non_perturbed_label!= None:
        Pexpt.drop(columns=non_perturbed_label,inplace=True)
    Pexpt=Pexpt.T.reindex(node_names).values
    return Pexpt


def score_binary_classifier(edges_gold,Jac_predicted,edges_known):
    '''
    Computes ROC curve and Precision Recall curves for the (binary) prediction
    of the Jacobian matrix. Only links in gold_Jac that also have not been part
    of the prior knowledge will be considered for scoring.
    Input: 
        edges_gold: dictionary of type {(from,to):True/False, ...} 
            indicating the gold standard Jacobian Matrix (zero and one entries).
        Jac_predicted: a Matrix of the shape of the Jacobian,
            where entries denote the predicted probability of existence of the according edge.
        edges_known: dict of form {(from,to):True/False,...} indicating if from->to edges
            exists/or not, (or unknown state for missing (N1,N2) tuples). See
            .edges_known2Jac_known
    Returns:
        Score dictionary that is also accepted by plotting function
        ..visualize.plot_binary_classifier_characteristics .
    '''
    gold,pred=[],[]
    for edge_pair,linked in edges_gold.items():
        if edge_pair in edges_known: continue
        gold.append(linked)
        pred.append(Jac_predicted[edge_pair[1],edge_pair[0]])
    precision, recall, _ = precision_recall_curve(gold,pred)
    fpr, tpr,_ = roc_curve(gold,pred)
    pre_rec_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)
    return {'precision':precision, 'recall':recall,'fpr':fpr, 'tpr':tpr, 'pre_rec_auc':pre_rec_auc, 'roc_auc':roc_auc}


def generate_DiGraph_from_edge_array_matrix(edge_array):
    '''
    An edge array is for example computed by response.sparsify_brave_structure
    it is a binarized Jacobian matrix and thus the transpose of the adjacency 
    matrix.
    '''
    return nx.from_numpy_array(edge_array.T,create_using=nx.DiGraph())

def generate_response_pattern(DiGraph,Sen,Pexpt=None):
    '''
    ...I could probably replace parts of this with networkx.transitive_closure...
    
    Given a directed graph (DiGraph from networkx.DiGraph) and perturbation 
    information, the function will return a (binary) response matrix (indicating
    which nodes (rows) respond upon which perturbation (columns). The DiGraph
    can be created with utilities.convenience.generate_DiGraph_from_adjancency_matrix.
        
    A perturbed node will only show a response if there actually is an (explicite)
    path (either self-loop or via intermediates). This is mentioned explicitely
    because the nx.has_path function always indicates an existing path when
    source and target are identical.
    
    If Pexpt is not provided it is assumed to be diagonal, that is each perturbation
    type is individually applied once.
    '''
    
    if Pexpt is None:
        Pexpt=np.eye(Sen.shape[1])
        
    response_pattern=np.zeros([Sen.shape[0],Pexpt.shape[1]],dtype=bool)
    nodes=np.array(DiGraph.nodes())
    N=len(nodes)
#    response_pattern=np.empty([N,0])    
    for perturbation_ind,Pexpt_col in enumerate(Pexpt.T):
#        new_R_col=np.zeros([N,1],dtype=bool)
        Pexpt_col= Pexpt_col!=0
        perturbed_nodes=np.any(Sen[:,Pexpt_col],axis=1)
        for source_ind in np.arange(N)[perturbed_nodes]:
            for target_ind in range(N):  
                #I need to treat case of equal source and target seperatly because
                #has_path will always assume that there is a self loop
                #so I ask if there is a path from the source to any of the source's
                #predecessors
                if source_ind==target_ind:
                    for predecessor in DiGraph.predecessors(nodes[source_ind]):
                        if nx.has_path(DiGraph,nodes[source_ind],predecessor):
                            response_pattern[target_ind,perturbation_ind]=True
                            break
                else:
                    if nx.has_path(DiGraph,nodes[source_ind],nodes[target_ind]):
                        response_pattern[target_ind,perturbation_ind]=True    
    return response_pattern


def order_zero_candidates_by_nonzero_groups(brave_edge_array,nonzero,confidence):
    '''
    This function returns an array ordered_zero_candidates as required for the
    response.sparsify_brave_structure function. It requires that non_zero, confidence
    and brave_edge_array all have the same shape (can have cols of nan). It then
    associates the brave links to the entries in confidence and groups them according
    to non_zero/nan/zero and sorts each group accordingly.
    '''
    
    ambiguous_links_zero_mask=np.logical_and(
        brave_edge_array==0.5,nonzero==0)
    ambiguous_links_nan_mask=np.logical_and(
        brave_edge_array==0.5,np.isnan(nonzero))
    ambiguous_links_nonzero_mask=np.logical_and(
       brave_edge_array==0.5,nonzero==1)

    ordered_zero_candidates=np.concatenate((
        np.transpose(np.where(ambiguous_links_zero_mask))[
            np.argsort(confidence[ambiguous_links_zero_mask])][::-1],
        np.transpose(np.where(ambiguous_links_nan_mask)),                    
        np.transpose(np.where(ambiguous_links_nonzero_mask))[
            np.argsort(confidence[ambiguous_links_nonzero_mask])]))
    
    return ordered_zero_candidates

def break_score_ties(scores,weights):
    '''
    Take matrix scores and calculates new_scores such that the rank of different
    entries in scores is preserved and in case that there are groups of entries
    with the same score, they get new scores such that this group's ranking is
    equal to that of the according group of entries in the weights matrix.
    This is useful to refine the link predictions. Eg. one can rerank all ambigious
    links that have a score of 0.5 by ordering them according to the magnitude 
    of the global response matrix.
    '''
    
    scores=pd.DataFrame(scores,index=pd.Index(np.arange(scores.shape[0]),name='target'),
                        columns=pd.Index(np.arange(scores.shape[1]),name='source'))
    weights=pd.DataFrame(weights,index=pd.Index(np.arange(scores.shape[0]),name='target'),
                        columns=pd.Index(np.arange(scores.shape[1]),name='source'))
    
    scores=scores.reset_index().melt(id_vars='target')
    scores['rank']=scores.value.rank(method='dense')
    
    weights=weights.reset_index().melt(id_vars='target',value_name='weight')
    scores=pd.merge(scores,weights,on=['target','source'])
    #normalize weights to be within zero and one
    scores.weight=( scores.weight-scores.weight.min() ) / ( scores.weight.max()-scores.weight.min() )
    scores['weighted_ranks']=scores['rank']+scores.weight
    #print(scores.weight)
    #normalize new scores to be within zero and one
    #scores['weighted_ranks']=( scores.weighted_ranks-scores.weighted_ranks.min() ) / ( scores.weighted_ranks.max()-scores.weighted_ranks.min() )
    new_scores=scores.pivot(index='target',columns='source',values='weighted_ranks').values
    return new_scores



def noisify_response_pattern(response_pattern,confidence_pattern,flip_fraction):
    '''
    Changes the state of a "flip_fraction" of response_pattern entries. The 
    entries are chosen such that entries with higher associated confidence get
    changed less. This assumes confidence scores to be within 0 and 1.    
    '''
    noisified_response_pattern=response_pattern.copy().astype(bool)
    shape=confidence_pattern.shape
    nbr_flips=int(confidence_pattern.size*flip_fraction)
    sorted_inds=np.unravel_index(np.argsort(( confidence_pattern-np.random.uniform(size=shape))\
                                            .flatten())[:nbr_flips], shape)
    #plt.plot(confidence_pattern[sorted_inds])
    noisified_response_pattern[sorted_inds]=~noisified_response_pattern[sorted_inds]
    return noisified_response_pattern
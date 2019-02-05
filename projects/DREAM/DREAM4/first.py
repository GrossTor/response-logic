#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:17:11 2018

@author: grosstor
"""



import pandas as pd
import numpy as np
import response_logic
import os
import itertools as it


DREAM_Ns=[10,100]
data_set_numbers=[1,2,3,4,5]
confidence_cuts=np.linspace(0.0,0.25,26)#[0.0] #

analysis_folder=os.path.dirname(__file__)
result_folder=os.path.join(analysis_folder, 'results/')


def create_response_pattern(ko,wts,confidence_cut=0.):       

    mean=wts.mean()
    sig=ko.std(1)
    
    #############################################################################    
    deviation=np.abs( (ko.subtract(mean,axis='index')).divide(sig,axis='index') ).values
    #############################################################################  
        

    deviation_response= deviation>=1
    deviation_confidence=np.full(deviation_response.shape,np.nan)
    if np.any(deviation_response):
        deviation_confidence[deviation_response]=(deviation[deviation_response]-1)\
                            /(deviation[deviation_response].max()-1)
    deviation_confidence[~deviation_response]=(1-deviation[~deviation_response])

    if confidence_cut>0.:
        #discard all data points with confidence below given threshold
        discarded_data_point_mask= deviation_confidence<confidence_cut
        deviation_response[discarded_data_point_mask]=np.nan
        deviation_confidence[discarded_data_point_mask]=np.nan
        
    return deviation_response,deviation_confidence,deviation
    


def write_result_files(scores,filename,model):
    scores=pd.DataFrame(scores,index=pd.Index(model['input']['node_names'],name='target'),
                            columns=pd.Index(model['input']['node_names'],name='source'))
    
    scores=scores.reset_index().melt(id_vars='target').sort_values('value',ascending=False)
    scores['score']=np.linspace(1,0,scores.shape[0])
    f=open(filename,'w')
    for dfrow in scores.itertuples():
        if dfrow.target==dfrow.source:
            continue
        f.write('{0}\t{1}\t{2}\r\n'.format(dfrow.source,dfrow.target,dfrow.score))
    f.close()
    return

def run_DREAM_inference(DREAM_N,data_set_number,confidence_cut=0):

    data_root_folder=os.path.join(analysis_folder, './DREAM4 in silico challenge/DREAM4 in-silico challenge/Size {0}/DREAM4 training data'.format(DREAM_N))
    data_folder=os.path.join(data_root_folder, 'insilico_size{0}_{1}'.format(DREAM_N,data_set_number))
    timeseries=pd.read_csv(os.path.join(data_folder,'insilico_size{0}_{1}_timeseries.tsv'.format(DREAM_N,data_set_number)),
                   sep='\t')
    
    wts=timeseries[timeseries.Time.isin([0,1000])].drop(columns='Time')

    wts=wts.append(pd.read_csv(os.path.join(data_folder,'insilico_size{0}_{1}_wildtype.tsv'.format(
            DREAM_N,data_set_number)),sep='\t'))
    
    
    kd=pd.read_csv(os.path.join(data_folder,'insilico_size{0}_{1}_knockdowns.tsv'.format(DREAM_N,data_set_number)),
                   sep='\t').T               
    kd.columns=pd.Index(kd.index.values,name='target gene')
    kd.index.name='readout_gene'
    ko=pd.read_csv(os.path.join(data_folder,'insilico_size{0}_{1}_knockouts.tsv'.format(DREAM_N,data_set_number)),
                   sep='\t').T
    ko.columns=pd.Index(ko.index.values,name='target gene')
    ko.index.name='readout_gene'

    response_pattern,confidence_pattern,Rexpt=create_response_pattern(
            ko,wts,confidence_cut=confidence_cut)
            
    Sen_known=np.eye(DREAM_N)
    np.fill_diagonal(Sen_known,np.nan)
    
    model={
           'N':DREAM_N,
           'P':DREAM_N,
           'Q':DREAM_N,
           'input':{
                   'node_names':ko.index.values,#[str(i) for i in nodes],
                   'pert_names':ko.index.values,#[str(i) for i in targets],
                   'edges_known':{},
                   'Sen_known':Sen_known,
                   },
           'response':{
                   'nonzero':response_pattern,
                   'confidence':confidence_pattern,
                   }
           }
    response_logic.conform_response_pattern(model)
    response_logic.brave_solving(model)
    ordered_zero_candidates=response_logic.convenience.order_zero_candidates_by_nonzero_groups(
            model['response']['brave edge array'],response_pattern,confidence_pattern)
    response_logic.sparsify_brave_structure(model,ordered_zero_candidates)
    
    sparse_scores=model['response']['brave edge array sparsified']
    sparse_scores=response_logic.convenience.break_score_ties(sparse_scores,Rexpt)
    
    #write down results file
    curr_result_dir=os.path.join(result_folder,'{0}/conf_thresh_{1}/').format(
            DREAM_N,str(confidence_cut).replace('.','_'))
    if not os.path.exists(curr_result_dir):
        os.makedirs(curr_result_dir)
    results_filename=os.path.join(curr_result_dir,'conf_thresh_{2}_net_InSilico_Size{0}_{1}.txt'.format(
            DREAM_N,data_set_number,str(confidence_cut).replace('.','_')))
    write_result_files(sparse_scores,results_filename,model)
    return

for DREAM_N,data_set_number,confidence_cut in it.product(DREAM_Ns,data_set_numbers,confidence_cuts):
    run_DREAM_inference(DREAM_N,data_set_number,confidence_cut)


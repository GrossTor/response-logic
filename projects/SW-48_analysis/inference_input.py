#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:53:13 2018

@author: grosstor
"""
import numpy as np
import itertools as it
import pickle
import os

analysis_folder=os.path.dirname(__file__)
data_folder=os.path.join(analysis_folder, 'data/')

readouts2short_name={
    'EGFR.Y1068.XP':'EGFR',
    'PDGF.Rec.beta.Y751':'PDGFR',
    'Shc.Y317.1':'Shc',
    'ERK1.2.T202.Y204':'ERK1',
    'p90.RSK.S380':'p90RSK',
    'Akt.S473.XP.1':'Akt',
    'p70.S6.Ki?se.T412':'p70RSK',
    'S6.Ribo.Prot..S235.236..XP':'RPS6',
    'Bad.S112':'Bad',
    'eIF4G.S1108':'eIF4G',
    }

cell_line2shortname={
        'SW48 parental':'parental',
        'SW48 E545K':'E545K',
        'SW48 H1047R':'H1047R'
        }


#gold standard
gold_edges={
        ('EGFR.Y1068.XP','Shc.Y317.1'),
        ('Shc.Y317.1','ERK1.2.T202.Y204'),
        ('Shc.Y317.1','Akt.S473.XP.1'),
        ('PDGF.Rec.beta.Y751','Shc.Y317.1'),
        ('ERK1.2.T202.Y204','p90.RSK.S380'),
        #('ERK1.2.T202.Y204','p70.S6.Ki?se.T412'), weak evidence
        ('p70.S6.Ki?se.T412','S6.Ribo.Prot..S235.236..XP'),
        ('p90.RSK.S380','S6.Ribo.Prot..S235.236..XP'),
        ('p90.RSK.S380','Bad.S112'),
        ('Akt.S473.XP.1','p70.S6.Ki?se.T412'),
        ('Akt.S473.XP.1','eIF4G.S1108'),
        #feedbacks
        }

#readouts=np.array(['EGFR.Y1068.XP','PDGF.Rec.beta.Y751','Shc.Y317.1',
#           'ERK1.2.T202.Y204','p90.RSK.S380','Akt.S473.XP.1','p70.S6.Ki?se.T412',
#           'S6.Ribo.Prot..S235.236..XP','Bad.S112','eIF4G.S1108'
#           ])


pert_to_players={
    'EGFRi':'EGFR.Y1068.XP',
    'AKTi':'Akt.S473.XP.1',
    'MEKi':'ERK1.2.T202.Y204',#{'MEK1.2.S217.221'},
    'EGF':'EGFR.Y1068.XP',
    'HGF':'PDGF.Rec.beta.Y751',#{'Met.Y1234.1235.XP'},
    'IGF':'Akt.S473.XP.1'#{'IGF.IRb.Y1131.IRb.Y1146'}
    }

with open(os.path.join(data_folder,'meta_data.pkl'), 'rb') as handle:
    meta_data = pickle.load(handle)
    
perturbations=meta_data['perturbations']
readouts=meta_data['readouts']

def return_inference_information(case):
    '''
    Returns gold standard network and information needed for inference:
    prior_net_knowledge: 
        state known (source,target) pairs and whether a link exists (1) or not (0) between them
    heuristics:
        heuristics used in inference  
    '''

    if case=='parental_inference':
        prior_net_knowledge={                
                ('EGFR.Y1068.XP','Bad.S112'):0,
                ('EGFR.Y1068.XP','p90.RSK.S380'):0,
                ('EGFR.Y1068.XP','p70.S6.Ki?se.T412'):0,
                ('EGFR.Y1068.XP','eIF4G.S1108'):0,
                ('EGFR.Y1068.XP','S6.Ribo.Prot..S235.236..XP'):0,
                ('PDGF.Rec.beta.Y751','Bad.S112'):0,
                ('PDGF.Rec.beta.Y751','p90.RSK.S380'):0,
                ('PDGF.Rec.beta.Y751','p70.S6.Ki?se.T412'):0,
                ('PDGF.Rec.beta.Y751','eIF4G.S1108'):0,                
                ('PDGF.Rec.beta.Y751','S6.Ribo.Prot..S235.236..XP'):0,
                ('Shc.Y317.1','Bad.S112'):0,
                ('Shc.Y317.1','p90.RSK.S380'):0,
                ('Shc.Y317.1','p70.S6.Ki?se.T412'):0,
                ('Shc.Y317.1','eIF4G.S1108'):0,
                ('Shc.Y317.1','S6.Ribo.Prot..S235.236..XP'):0,
                }

        readout_to_ind={readout:i for i,readout in enumerate(readouts)}
        receptor_level=['EGFR.Y1068.XP','PDGF.Rec.beta.Y751','Shc.Y317.1']
#        intermediate_level=['ERK1.2.T202.Y204','Akt.S473.XP.1','p90.RSK.S380','p70.S6.Ki?se.T412']
#        target_level=['S6.Ribo.Prot..S235.236..XP',
#                   'Bad.S112','eIF4G.S1108']
        
        min_nbr_inputs=2
        heuristics={'parsimonious':True}
        local_edge_nbr_lims={
#            'local_min_nbr_inedges':#intermediate+target_level need at least one input
#                [(readout_to_ind[readout],min_nbr_inputs) for readout in intermediate_level+target_level],
            'local_min_nbr_outedges':#receptor_level need at least one output
                [(readout_to_ind[readout],min_nbr_inputs) for readout in receptor_level],
#            'local_max_nbr_outedges':#target level should not have outputs
#                [(readout_to_ind[readout],1) for readout in target_level],
            }
        heuristics.update(local_edge_nbr_lims)
        
        #heuristics={'parsimonious':False}
        plot_prefix='parental_inference'
        
        
    elif case=='cl_comparison': #no MEK
       
        prior_net_knowledge={}
        for edge in it.product(readouts,repeat=2):
            if edge[0]==edge[1]:
                prior_net_knowledge[edge]=1
            elif  \
                (edge[1]!= 'EGFR.Y1068.XP') and \
                (edge[0]!= 'Akt.S473.XP.1') and \
                (edge[1]!= 'Akt.S473.XP.1'): 

                if edge in gold_edges:
                    prior_net_knowledge[edge]=1
                else:                            
                    prior_net_knowledge[edge]=0
            
    
        prior_net_knowledge.update(
                {                
                ('EGFR.Y1068.XP','Bad.S112'):0,
                ('EGFR.Y1068.XP','p90.RSK.S380'):0,
                ('EGFR.Y1068.XP','p70.S6.Ki?se.T412'):0,
                ('EGFR.Y1068.XP','S6.Ribo.Prot..S235.236..XP'):0,
                ('PDGF.Rec.beta.Y751','Bad.S112'):0,
                ('PDGF.Rec.beta.Y751','p90.RSK.S380'):0,
                ('PDGF.Rec.beta.Y751','p70.S6.Ki?se.T412'):0,
                ('PDGF.Rec.beta.Y751','S6.Ribo.Prot..S235.236..XP'):0,
                ('Shc.Y317.1','Bad.S112'):0,
                ('Shc.Y317.1','p90.RSK.S380'):0,
                ('Shc.Y317.1','p70.S6.Ki?se.T412'):0,
                ('Shc.Y317.1','S6.Ribo.Prot..S235.236..XP'):0,
                }
                )

        heuristics={'parsimonious':True}
        
#        readout_to_ind={readout:i for i,readout in enumerate(readouts)}
#        receptor_level=['EGFR.Y1068.XP','PDGF.Rec.beta.Y751','Shc.Y317.1']
#        intermediate_level=['ERK1.2.T202.Y204','Akt.S473.XP.1','p90.RSK.S380','p70.S6.Ki?se.T412']
#        target_level=['S6.Ribo.Prot..S235.236..XP',
#                   'Bad.S112','eIF4G.S1108']
#        
#        min_nbr_inputs=2
#        local_edge_nbr_lims={
#            'local_min_nbr_inedges':#intermediate+target_level need at least one input
#                [(readout_to_ind[readout],min_nbr_inputs) for readout in intermediate_level+target_level],
#            'local_min_nbr_outedges':#receptor_level+intermediate_level need at least one output
#                [(readout_to_ind[readout],min_nbr_inputs) for readout in receptor_level],#+intermediate_level],
#            'local_max_nbr_outedges':#target level should not have outputs
#                [(readout_to_ind[readout],1) for readout in target_level],
#            }
#        #heuristics.update(local_edge_nbr_lims)
        
        plot_prefix='cl_comparison'        
    return prior_net_knowledge,heuristics,plot_prefix

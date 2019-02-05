#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:02:04 2018

@author: grosstor
"""
from Bio.KEGG import REST as KEGG_REST
from Bio.KEGG.KGML import KGML_parser
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np
import response_logic



analysis_folder=os.path.dirname(__file__)
KEGG_data_folder=os.path.join(analysis_folder, 'KEGG_data/')

import urllib.request

#Download KEGG onthology
if not os.path.isfile(os.path.join(KEGG_data_folder,'ko00001.json')):
    url='https://www.genome.jp/kegg-bin/download_htext?htext=ko00001&format=json&filedir='
    urllib.request.urlretrieve(url, os.path.join(KEGG_data_folder,'ko00001.json'))


#get all human pathways
hsa_path_list=KEGG_REST.kegg_list('pathway','hsa')
identifiers=[]
for line in hsa_path_list:
   identifiers.append(line.partition('\t')[0][5:])


for identifier in identifiers:
   KGML_handle=KEGG_REST.kegg_get(identifier,option='kgml')
   file=open(os.path.join(KEGG_data_folder,identifier+'.kgml'),'w')
   file.write(KGML_handle.read())
   file.close()

#%%
parse_pathways=1
if parse_pathways:
    
    pathways=[]
    lengths=[]
    for filename in os.listdir(KEGG_data_folder):
        if not filename.endswith('kgml'): continue
        #pathways.append({})    
        KGML_handle=open(os.path.join(KEGG_data_folder,filename))#open('/home/grosstor/Desktop/steady_ready_projects/response_logic_synthetic_benchmarks/KEGG/hsa00515.xml')
        #KGML_handle=open(identifier+'.kgml')
        pathway=KGML_parser.read(KGML_handle)
        net=nx.DiGraph()
        for relation in pathway.relations:
            #print(relation.entry1.id,'<->',relation.entry2.id)
            net.add_edge(relation.entry1.id,relation.entry2.id)
        lengths.append(len(net))
        if (len(net)>5) & (len(net)<100):
    #        print(len(net))
            #generate some extra information from net#
            
            gold_Jac=nx.adj_matrix(net).toarray().T
            cycle_list=[set(cycle) for cycle in nx.simple_cycles(net) ] #maybe needed later to correlate results with number of nodes that are in loops
            #find distinct aggregated cycles
            aggregated_cycles=[]
            while cycle_list:        
                aggr_cycle=cycle_list.pop()
                #keep_indeces=[]
                finished=False
                while not finished:
                    for i,cycle in enumerate(cycle_list):
                        if aggr_cycle.intersection(cycle):
                            aggr_cycle.update(cycle)
                            del(cycle_list[i])
                            break
                    else:
                        finished=True
                        aggregated_cycles.append(aggr_cycle)
            
            #effective node ratio counts nodes that are in a cycle as one node
            nbr_noncyclic_nodes=len(net)-np.sum((len(i) for i in aggregated_cycles))
            effective_node_ratio=(nbr_noncyclic_nodes+len(aggregated_cycles))/len(net)
            
            Sen_known=np.eye(len(net))
            np.fill_diagonal(Sen_known,np.nan)
            gold_response=response_logic.convenience.generate_response_pattern(net,Sen_known)
            
            
            
            pathways.append({'title':pathway.title,'KEGG_id':pathway.name,'net':net,
                             'nodes':np.array(net.nodes()),
                             'gold_Jac':gold_Jac,'gold_response':gold_response,
                             'Sen_known':Sen_known,
                             'aggregated_cycles':aggregated_cycles,
                             'cycle_list':cycle_list,
                             'effective_node_ratio':effective_node_ratio})
            
with open('parsed_human_KEGG_pathways.pkl', 'wb') as fp:
   pickle.dump(pathways, fp)  
    
    
#%% class annotations of pathways
import json
    
with open(os.path.join(KEGG_data_folder,'ko00001.json'), "r") as read_file:
    annotation = json.load(read_file)



def get_all_names(d,names=[]):
    d_is_dict=isinstance(d, dict)
    if d_is_dict and d["name"][-13:-9]=='PATH':
        names.append(d['name'][:5])
    else:
#    if len(d)>1:
        if isinstance(d, dict):
            for v in d.values():
                get_all_names(v,names)
        elif isinstance(d, list):
            for i in range(len(d)):
                get_all_names( d[i],names)   
#    elif isinstance(d, dict):
        #print(d)

#        names.append(d['name'][1:6])
            #print("{0} : {1}".format(k, v))
    return names

KEGG_ids_to_class={}
for classes_dict in annotation['children']:
    class_name=classes_dict['name']
    names=[]
    names=get_all_names(classes_dict,names=[])
    KEGG_ids_to_class[frozenset(names)]=class_name

with open (os.path.join('parsed_human_KEGG_pathways.pkl'), 'rb') as fp:
    pathways = pickle.load(fp)
#%%    
for pathway in pathways:
    pathway_classes=[]
    for id_set,KO_class in KEGG_ids_to_class.items():
        if pathway['KEGG_id'][-5:] in id_set:
            pathway_classes.append(KO_class)
    pathway['class']=pathway_classes[0] #I checked that the list never has more than one entry
    

with open('parsed_human_KEGG_pathways.pkl', 'wb') as fp:
    pickle.dump(pathways, fp)
fp.close()





#pathways.loc[(pathways['N']>35) & (pathways['N']<100) & (pathways['class']=='Environmental Information Processing')][['KEGG_id','title','N']]


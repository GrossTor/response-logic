#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:55:48 2018

@author: grosstor
"""


import graphviz as gv
from subprocess import call
import numpy as np
import matplotlib
import pandas as pd
import os 
from inference_input import readouts2short_name


analysis_folder=os.path.dirname(__file__)
plot_folder=os.path.join(analysis_folder, 'plots/')


#    gv_player_pos={
#    player_names_191to187['IGF.IRb.Y1131.IRb.Y1146']:[0,0],
#    player_names_191to187['Akt.S473.XP.1']:[0,2],
#    player_names_191to187['mTOR.S2448']:[0,3],
#    player_names_191to187['S6.Ribo.Prot..S235.236..XP']:[1,4],
#    player_names_191to187['Met.Y1234.1235.XP']:[1,0],
#    player_names_191to187['EGFR.Y1068.XP']:[2,0],
#    player_names_191to187['Shc.Y317.1']:[2,1],
#    #'B.Raf.T401':[2,2],
#    #'MEK1.2.S217.221':[2,3],
#    player_names_191to187['ERK1.2.T202.Y204']:[2,2],
#    player_names_191to187['p90.RSK.S380']:[2,3]
#    }


def create_gv_graph(nodes,edges,node_positions={},edge_strengths={},perturbed_nodes=set([]),
               perturbed_nodes_strengths={},label='SW48',edge_strength_thresh=0.,
               gv_x_stretch=.9,gv_y_stretch=.9,edge_width_scale=1.3,arrowsize_scale=1,
               edge_len='3',
               colormap_edges="Greys",colormap_perturbed_nodes="coolwarm"):
    if not edge_strengths:
        edge_strengths={edge:1 for edge in edges}
    
    if not perturbed_nodes_strengths:
        perturbed_nodes_strengths={perturbed_node:1 for perturbed_node in perturbed_nodes}
    
    cmap_edges = matplotlib.cm.get_cmap(colormap_edges)
    colormap_perturbed_nodes = matplotlib.cm.get_cmap(colormap_perturbed_nodes)
    
    #Set graphviz positions by hand:
#    gv_x_pos=[0,3,6]
#    gv_y_pos=[5,4,3,2,1,0]
#    #gv_x_shift=[-.3,.3,-.6,.3,-.3,.3]
#    gv_x_shift=[0,0,0,0,0,0]
    

    graph=gv.Digraph(engine='neato',format='ps2')
    #graph=gv.Digraph(engine='neato',format='pdf')
    graph.attr(labelloc="t",label=label,fontsize='20')
    #graph.attr(outputorder='edgesfirst',splines='ortho')
    graph.attr(outputorder='edgesfirst',splines='spline')
    
    #graph.node_attr.update(style='filled', color='black',margin='0.',fontsize='12')
    graph.node_attr.update(style='rounded', color='black',margin='-0.',fontsize='12')

     
    #b7cfd3ff
    for node in nodes:
        if node in perturbed_nodes:
            color=matplotlib.colors.to_hex(colormap_perturbed_nodes(
                    perturbed_nodes_strengths[node]))
            color='#ffcc00ff'
        else:
            color='#b7cfd3ff'

#        graph.node(node,pos="{0},{1}!".format(
#            gv_x_pos[gv_player_pos[player][0]]+gv_x_shift[gv_player_pos[player][1]],
#            gv_y_pos[gv_player_pos[player][1]]),
#            color=color)
        if node in node_positions:
            graph.node(node,pos="{0},{1}!".format(
                node_positions[node][0]*gv_x_stretch,
                node_positions[node][1]*gv_y_stretch),
                #color=color, shape='box')#shape='ellipse')# shape='polygon',sides='6')
                shape='box',fillcolor=color,color='#23373bff', style="rounded,filled",
                width='.7',height='0.35',
                fontname='DejaVuSans')
                
        else:
            graph.node(node,color=color,shape='polygon',sides='6')
    for (node1,node2),edge_strength in edge_strengths.items():
        if node1!=node2 and edge_strength>edge_strength_thresh:
            color=matplotlib.colors.to_hex(cmap_edges(edge_strength))
            #color='#b7cfd3ff'
            if edge_strength<1:#dashed edges
                graph.edge(node1,node2,penwidth=str(edge_width_scale*edge_strength),
                   arrowsize=str(arrowsize_scale*edge_strength),
                   headclip='True',color=color,len=str(edge_len),
                   #style='dashed')
                   )
            else:
                graph.edge(node1,node2,penwidth=str(edge_width_scale*edge_strength),
                   arrowsize=str(arrowsize_scale*edge_strength),
                   headclip='True',color=color,len=str(edge_len))
    return graph

    
    
#%%

def draw_graph(Jac,readouts,filename,perturbations=[],pert_to_players=[],
               **graph_paras):
    graph_paras['node_positions']=node_positions
    Jac=pd.DataFrame(Jac.astype(float),
                     index=pd.Index(readouts,name='target'),
                     columns=pd.Index(readouts,name='source'))
    nodes=Jac.index.values
    edges=Jac.reset_index().melt(id_vars='target',value_name='edge_strength')
    edges=edges[edges['edge_strength']!=0]
    edge_strengths=edges.set_index(['source','target']).to_dict()['edge_strength']
    edges=edges[['source','target']].values
    perturbed_nodes={pert_to_players[perturbation] for perturbation in perturbations}    
    gv_graph=create_gv_graph(nodes,edges,edge_strengths=edge_strengths,
                perturbed_nodes=perturbed_nodes,**graph_paras)
    gv_graph.render(filename)
    call(["ps2pdf",filename+'.ps2',filename+'.pdf'])
    
    return


#node_positions_old={
#        'EGFR.Y1068.XP':[2.,0.], 
#        'ErbB2.HER2.Y1248':[3.,0.], 
#        'PDGF.Rec.beta.Y751':[4.,0.],
#        'Shc.Y317.1':[3.,1.],
#        'Src.Family.Y416':[3.,2.],
#        'MEK1.2.S217.221':[3.,3.],
#        'ERK1.2.T202.Y204':[3.,4.],
#        'p90.RSK.S380':[3.,5.],
#        'Akt.S473.XP.1':[1.,2.],
#        'PRAS40.T246':[2.,3.],
#        'p70.S6.Ki?se.T412':[2.,5.],
#        'S6.Ribo.Prot..S235.236..XP':[3.,6.],
#        'Bad.S112':[2.,6.],
#        'eIF4G.S1108':[1.,5.]
#       }



#node_positions={
#        'EGFR.Y1068.XP':[4.,2.], 
#        #'ErbB2.HER2.Y1248':[3.,0.], 
#        'PDGF.Rec.beta.Y751':[2.,2.],
#        'Shc.Y317.1':[3.,1.5],
#        #'Src.Family.Y416':[3.,2.],
#        #'MEK1.2.S217.221':[3.,3.],
#        'ERK1.2.T202.Y204':[4.,1.],
#        'p90.RSK.S380':[3.,0.5],
#        'Akt.S473.XP.1':[2.,1.],
#        #'PRAS40.T246':[2.,3.],
#        'p70.S6.Ki?se.T412':[2.,0.],
#        'S6.Ribo.Prot..S235.236..XP':[3.,-.5],
#        'Bad.S112':[4.,0.],
#        'eIF4G.S1108':[1.,0.5]
#       }


node_positions={
        'EGFR.Y1068.XP':[2.75,2.], 
        #'ErbB2.HER2.Y1248':[3.,0.], 
        'PDGF.Rec.beta.Y751':[0.25,2.],
        'Shc.Y317.1':[1.5,1.75],
        #'Src.Family.Y416':[3.,2.],
        #'MEK1.2.S217.221':[3.,3.],
        'ERK1.2.T202.Y204':[2.75,1.25],
        'p90.RSK.S380':[2.1,.625],
        'Akt.S473.XP.1':[.25,1.25],
        #'PRAS40.T246':[2.,3.],
        'p70.S6.Ki?se.T412':[.9,.625],
        'S6.Ribo.Prot..S235.236..XP':[1.5,0.],
        'Bad.S112':[2.75,0.],
        'eIF4G.S1108':[.25,0.]
       }




node_positions.update(
        {readouts2short_name[readout]:pos for readout,pos in node_positions.items()}
        )


#draw_graph(Jac_known.values,readouts_shortnames,os.path.join(plot_folder,'graphs/gold_net.pdf'),
#           label='Literature network')

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:11:19 2018

@author: grosstor
"""

from dreamtools import D3C4
import pandas as pd
import os 

c = D3C4()

analysis_folder=os.path.dirname(__file__)
result_folder=os.path.join(analysis_folder, 'results/')
scores=[]
for DREAM_N in [10,50,100]:
    N_folder=os.path.join(result_folder,'{0}/'.format(DREAM_N))
    subfolders=os.listdir(N_folder)
    for subfolder in subfolders:
        confidence_thresh=float('.'+subfolder.split('_')[-1])
        subfolder_full=os.path.join(N_folder,subfolder)
        predict_files=['{0}/{1}'.format(subfolder_full,f) for f in os.listdir(subfolder_full)]
        score=c.score(predict_files,DREAM_N)
        
        score['N']=DREAM_N
        score['confidence_thresh']=confidence_thresh
        scores.append(score)

scores=pd.DataFrame(scores)
scores.to_csv('DREAM3_scores.csv')

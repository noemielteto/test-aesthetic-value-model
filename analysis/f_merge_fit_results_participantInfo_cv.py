#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 2021
Last updated Thu Jan 20 2022

@author: aennebrielmann
"""

import os
import glob
import pandas as pd

#%% ---------------------------------------------------------
# Specify directories
#------------------------------------------------------------
os.chdir('..')
home_dir = os.getcwd()
dataDir = home_dir
# list the names of all models as in the saved .csv files
allResFiles = glob.glob(dataDir + '/results/*.csv')

#%% ---------------------------------------------------------
# Load, merge, save the data
#------------------------------------------------------------
infoDf = pd.read_csv(dataDir + '/merged_participantInfo.csv')
df = infoDf.copy()

for file in allResFiles:
    modelDf = pd.read_csv(file)
    modelSpec = file.replace(dataDir+'/results', '')
    # windows/OS encoding differs a little in use of \\ vs /
    # remove both
    modelSpec = modelSpec.replace('\\','')
    modelSpec = modelSpec.replace('/','')
    modelSpec = modelSpec.replace('.csv','')
    modelSpec = modelSpec.replace('cv','')
    modelSpec = modelSpec.replace('_results','')
    modelSpec = modelSpec.replace('_paper_draftmodel','')
    resDf = modelDf.iloc[:,-5:-1].add_suffix(modelSpec)
    resDf['subj'] = modelDf['subj'].copy()
    df = pd.merge(df, resDf, on='subj')

# save this as a new, merged data frame
df.to_csv(dataDir + '/perParticipantResults_cv.csv', index=False)

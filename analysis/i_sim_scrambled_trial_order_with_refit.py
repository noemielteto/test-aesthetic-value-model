#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 2022
Based on script for simulating scrambled trial orders for all participants

@author: aennebrielmann
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

#%% ---------------------------------------------------------
# WARNING this script is computationally expensive, especially for the larger models
# be aware that it might take a long time to run even a few participants
#------------------------------------------------------------

#%% ---------------------------------------------------------
# Specify directories; settings
#------------------------------------------------------------
os.chdir('..')
os.chdir('..')
os.chdir('..')
home_dir = os.getcwd()
dataDir = home_dir + '/Papers/RoySocB/'
resDir = dataDir + 'analysis/results/individuals/'

modelSpec = '2vggfeat'
plot = True
save = True
niter = 100
n_base_stims = 7

exampleSubj = ['kwqcc','7df33'] # 
sigBetterWithLearn3D = ['mrx1q', 'oveao', 'quzfn', 'zv0ou']
sigBetterWithLearn2D = ['37o95', '6r8qk', '9gjbe', 'h647a', 'jmzs9',
                        'mbx6w', 'mrx1q', 'ms8r4', 'n0htt', 'njzz4',
                        'nq3sp', 'zv0ou']

#%% ---------------------------------------------------------
# Load results
#------------------------------------------------------------
infoDf = pd.read_csv(dataDir + 'perParticipantResults_cv.csv')
df = pd.read_csv(dataDir + 'merged_rating_data.csv')
participantList = infoDf.subj.unique()
participantList.sort()

#%% ---------------------------------------------------------
# Custom functions
#------------------------------------------------------------
sys.path.append((home_dir + "/python_packages"))
from aestheticsModel import fitPilot

if 'model' in modelSpec:
    # set specs accordingly
    if 'vgg' in modelSpec:
        dnnFeatures = 'vgg'
    else:
        dnnFeatures = ''
    if 'wrzero' in modelSpec:
        modelSpec = 'wrzero'
    elif 'wVzero' in modelSpec:
        modelSpec = 'wVzero'
    elif 'muStateFix' in modelSpec:
        modelSpec = 'muStateFix'
    elif 'alphazero' in modelSpec:
        modelSpec = 'alphazero'
    else:
        modelSpec = ''
        
if '2' in modelSpec:
    n_features = 2
elif '3' in modelSpec:
    n_features = 3
elif '4' in modelSpec:
    n_features = 4
else:
    ValueError('Cannot determine number of features')

if 'vgg' in modelSpec:
    # get (reduced) VGG features
    featureDf = pd.read_pickle(dataDir
                               + '/VGG_features/VGG_features_reduced_to_'
                               + str(n_features) + '.pkl')
    # now create an array that contains featuers of the images in the right order
    for imgInd in np.unique(df.imageInd):
        img = np.unique(df.image[df.imageInd==imgInd])[0]
        if imgInd==0:
            vggFeatures = featureDf.feature_array[featureDf.image==img].values[0]
        else:
            vggFeatures = np.vstack([vggFeatures,
                            featureDf.feature_array[featureDf.image==img].values[0]])
    fixedDict = {'features': vggFeatures, 'numStimsFit': 0}
elif modelSpec=='muStateFix':
    fixedDict = {'muState': np.zeros(n_features)}
elif modelSpec=='wVzero':
    fixedDict = {'w_V': 0}
elif modelSpec=='wrzero':
    fixedDict = {'w_r': 0}
elif modelSpec=='alphazero':
    fixedDict = {'alpha': 0}
else:
    fixedDict = None
    
if 'alphazero' not in modelSpec:
    bounds = ((0,0.1), (0,1e4),) # alpha, weight(s)
else:
    bounds = ((0,1e4),) # weight w_r
if 'wVzero' not in modelSpec and 'wrzero' not in modelSpec:
    bounds += ((0,1e4),)
bounds += ((-1,1),) # bias
if 'muStateFix' not in modelSpec:
   bounds += ((-10, 10), ) * n_features # mu state
bounds += ((0, 10), ) # var state
# below: no need for p_true if there is no learning
if 'alphazero' not in modelSpec:
    bounds += ((-10, 10), ) * n_features # mu true
    bounds += ((0, 10), ) # var true
   
def pred_ratings(parameters, data):
    pred = fitPilot.predict(parameters, data.iloc[:55],
                                  n_features=n_features,
                                  fixParameters=fixedDict)
    return pred

def cost_fn(parameters, data):
    ratings = data['rating'].values
    pred = pred_ratings(parameters, data)
    cost = np.sqrt(np.mean((ratings - pred)**2))
    return cost

nParams = len(bounds)

#%% ---------------------------------------------------------
# fetch and save predictions for the entire experiment based on best model
#------------------------------------------------------------
resDict = {'r_true': [], 'r_shuffled': [],
           'rmse_shuffled': [], 'rmse_true': [],
           'subj': [], 'alpha': [], 'w_r': [], 'w_V': []}

simRMSE = []
simR = []
alphas = []
resList = []
peepList = []
counter = 0

for peep in participantList:
    ratingData = df[df.subj==peep]

    res = np.load(dataDir + 'results/individuals/' + 'fit_' + peep 
                  + '_' + modelSpec + '.npy',
                  allow_pickle=True).tolist()
    modelParameters = res
    alpha, w_V, w_r, _, _, _, _, _, _ = fitPilot.unpackParameters(modelParameters,
                                                                  fixParameters=fixedDict)

    # get properly ordered image indices and rts
    imageSequence = ratingData.imageInd.tolist()[:55]
    rtSequence = ratingData.rt.tolist()[:55]
    ratingSequence = ratingData.rating.tolist()[:55]
    tmpDf = pd.DataFrame({'imageInd': imageSequence, 'rt': rtSequence,
                          'rating': ratingSequence})

    # first, we get the true predictions (and ratings)
    truePred = fitPilot.predict(modelParameters, tmpDf,
                                 n_features=n_features,
                                 fixParameters=fixedDict)

    # now, we simulate predictions for shuffled orders
    simPreds = []
    simDiff = []
    for sim in range(niter):
        tmpDf = tmpDf.sample(frac=1)
        tmp_res = []
        tmp_rmse = []
        
        for seed in range(10):
            np.random.seed(seed)
            randStartValues = np.random.rand(nParams)
            # adjust scaling for starting point of alpha
            if 'alphazero' not in fixedDict:
                randStartValues[0] = randStartValues[0]/100
            # the usual optimization
            thisRes = minimize(cost_fn, randStartValues,
                               args=(tmpDf,),
                               method='SLSQP',
                               options={'maxiter': 1e4, 'ftol': 1e-06},
                               bounds=bounds)

            tmp_res.append(thisRes)
            tmp_rmse.append(thisRes.fun)
            
        res = tmp_res[tmp_rmse.index(np.nanmin(tmp_rmse))]
        resList.append(res)
        simPred = fitPilot.predict(res.x, tmpDf,
                                 n_features=n_features,
                                 fixParameters=fixedDict)
        simPreds.append(simPred)
        simDiff.append(simPred-tmpDf.rating)
        
        simRMSE.append(np.sqrt(np.mean((tmpDf.rating - simPred)**2)))
        simR.append(np.corrcoef(tmpDf.rating, simPred)[0,1])
        alphas.append(res.x[0])
        peepList.append(peep)
    
    backupDf = pd.DataFrame({'sim_rmse': simRMSE, 'sim_r': simR, 
                                'participant': peepList,
                                'alpha': alphas})
    backupDf.to_csv(dataDir + 'backup_results_refit_scrambled'
                    + modelSpec + '.csv',
                    index=False)
     
    avgShuffledPred = np.mean(simPreds,axis=0)
    sdShuffedPred = np.std(simPreds,axis=0)
    rmse_shuffled = np.mean(simRMSE, axis=0)
    r_shuffled = np.mean(simR, axis=0)
    
    rmse_model = np.sqrt(np.mean((truePred - ratingData.rating[:55])**2))
    r_model = np.corrcoef(truePred, ratingData.rating[:55])[0,1]
    
    resDict['r_true'] += [r_model]
    resDict['r_shuffled'] += [r_shuffled]
    resDict['rmse_true'] += [rmse_model]
    resDict['rmse_shuffled'] += [rmse_shuffled]
    resDict['subj'] += [peep]
    resDict['alpha'] += [alpha]
    resDict['w_r'] += [w_r]
    resDict['w_V'] += [w_V]
    
    counter += 1
    print('Done with ' + peep + '; ' + str(counter) + ' participants done.')
    
    
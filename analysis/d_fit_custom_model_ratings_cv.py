#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 5 2022
updated Feb 04 2022: cross-validation
updated Wed March 10 2022: system state L2-normalization option
Last updated Wed Jan 4 2023: cleaning up, updating directories
@author: aennebrielmann


"""
import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn import preprocessing


#%% ---------------------------------------------------------
# Specify directories; settings
#------------------------------------------------------------

os.chdir('..')
os.chdir('..')
os.chdir('..')
home_dir = os.getcwd()
dataDir = home_dir + '/Papers/RoySocB/'

n_features = 2 # dimensionality of assumed feature space
n_base_stims = 7 # unique images used as basis for morphing
scaleVariances = False # see fitPilot.py for details
if scaleVariances:
    varScaling = 'scaledVars'
else:
    varScaling = ''

truncatePredictions = '' # either '' for regular predictions or 'truncated'
normalizedFeatures = '' # either 'normed' or ''
constrainedMu = False
dnnFeatures = 'vgg' # either '' or spec of DNN (vgg, ResNet50)
fixParam = 'alphazero_wVzero' # mainly '' or 'alphazero_wVzero'; other are possible

save = False # save parameter values?
plot = True # plot data vs predictions?
preLoad = True # load parameter values rather than fit them?
fitMissingParticipants = False # fit data of participants that cannot be preloaded?
# NOTE that you have to set this to False if you want to preload data from all participants.

#%% ---------------------------------------------------------
# Load model; data
#------------------------------------------------------------
sys.path.append((home_dir + "/python_packages"))
from aestheticsModel import fitPilot
fixedDict = {}

# get data
df = pd.read_csv(dataDir + 'merged_rating_data.csv')

if dnnFeatures!='':
        # get (reduced) VGG features
    featureDf = pd.read_pickle(dataDir + '/' 
                               + dnnFeatures + '_features/' + dnnFeatures 
                               + '_features_reduced_to_'
                            + str(n_features) + '.pkl')
    # now create an array that contains featuers of the images in the right order
    for imgInd in np.unique(df.imageInd):
        img = np.unique(df.image[df.imageInd==imgInd])[0]
        if imgInd==0:
            dnnFeatValues = featureDf.feature_array[featureDf.image==img].values[0]
        else:
            dnnFeatValues = np.vstack([dnnFeatValues,
                            featureDf.feature_array[featureDf.image==img].values[0]])

    if normalizedFeatures=='normed':
        dnnFeatValues = preprocessing.normalize(dnnFeatValues, 'l2')

    fixedDict['features'] = dnnFeatValues
    fixedDict['numStimsFit'] = 0
if 'muStateFix' in fixParam:
    fixedDict['muState'] = np.zeros(n_features)
if 'alphazero' in fixParam:
    fixedDict['alpha'] = 0
    fixedDict['muTrue'] = np.zeros(n_features)
    fixedDict['varTrue'] = 1e4
if 'wVzero' in fixParam:
    fixedDict['w_V'] = 0
if 'wrzero' in fixParam:
    fixedDict['w_r'] = 0
if fixParam=='' and dnnFeatures=='':
    fixedDict = None

# IDs to choose from (not including excluded subjs):
participantList = ['q22fa', 'eunhf', 'fax28', '1z0ca', '0xmq6',
                    'z19vf', 'pu7uh', 's4tsr', 'n0htt', 'zouzq', 'nqfzn',
                    'oveao', 'mh0hn', 'mbx6w', '46vpz', 'sdx2b', 'jmzs9',
                    'jeu26', 'hvjjw', 'xj6bc', '3l6s4', 'chb6q', 'kwqcc',
                    'czgyu', 'nyp52', 'pshnw', 'faow1', 'kuoa6', '7df33',
                    'mrx1q', 'quzfn', 're1sg', 'j49v1', '19bq6', 'r4ron',
                    '8kyd7', 'nq3sp', 'rsw55', '37o95', 'wtr4q', 'ar5vx',
                    '6yaxg', 'jv7jg', 'ms8r4', '520tj', '9gjbe',
                    'njzz4', 'rq280', 'tg4o2', 'fpt0a', 'k02z1', 'u0pc9',
                    'hxal4', 'h647a', 'r1g8p', '89w87', 'zv0ou', 'qrbxe',
                    '6r8qk']

# sort the participant list for consistency
participantList.sort()

#%% ---------------------------------------------------------
# Set bounds for optimization
#------------------------------------------------------------
if 'alphazero' not in fixParam:
    bounds = ((0,0.1), (0,1e4),) # alpha, weight(s)
else:
    bounds = ((0,1e4),) # weight w_r
if 'wVzero' not in fixParam and 'wrzero' not in fixParam:
    bounds += ((0,1e4),)
bounds += ((-1,1),) # bias
if 'muStateFix' not in fixParam:
   bounds += ((-10, 10), ) * n_features # mu state
bounds += ((0, 10), ) # var state
# below: no need for p_true if there is no learning
if 'alphazero' not in fixParam:
    bounds += ((-10, 10), ) * n_features # mu true
    bounds += ((0, 10), ) # var true
if dnnFeatures=='':
   bounds += ((-10,10), ) * n_base_stims*n_features # stimuli

nParams = len(bounds)

# get a list of pairs (excluding source images)
tmp = df.pair.unique().tolist()
sourceImInds = [tmp.index(im) for im in ['5', '37', '41', '57', '64',  '92', '101']]
pairs = np.delete(tmp, sourceImInds)

#%% ---------------------------------------------------------
# Cost function
#------------------------------------------------------------
def pred_ratings(parameters, data):
    pred = fitPilot.predict(parameters, data.iloc[:55],
                                  n_features=n_features,
                                  fixParameters=fixedDict,
                                  scaleVariances=scaleVariances)
    return pred

def cost_fn(parameters, data, testPair, test=False):
    ratings = data['rating'].values

    pred = pred_ratings(parameters, data)

    if test:
        cost = np.sqrt(np.mean((ratings[data.pair==testPair]
                                 - pred[data.pair==testPair])**2))
    else:
        cost = np.sqrt(np.mean((ratings[data.pair!=testPair]
                                 - pred[data.pair!=testPair])**2))
    # print(cost) # only for initial or intermittent checks on speed and smoothness of convergence
    return cost

def L2_const(parameters):
    _, _, _, _, mu_0, _, _, _, _ = fitPilot.unpackParameters(parameters,
                                                             n_features,
                                                             fixParameters=fixedDict,
                                                             scaleVariances=scaleVariances)
    squared_mus = [m*2 for m in mu_0]
    sumSq = np.sum(squared_mus)
    return 1- sumSq
if constrainedMu:
    cons = {'type':'eq', 'fun': L2_const}


#%% ---------------------------------------------------------
# Setting up a dict to store all results for each participant
#------------------------------------------------------------
if preLoad:
    resDf = pd.read_csv(dataDir + 'results/allFits/' + 'allFits_'
                            + str(n_features) + dnnFeatures + 'feat'
                            + fixParam
                            + truncatePredictions
                            + normalizedFeatures
                            + varScaling
                            + '.csv')
    resDict = resDf.to_dict()
    if fitMissingParticipants:
        for peep in resDf.participant.unique():
            participantList.remove(peep)
else:
    resDict = {'res':[], 'rmse_fit': [],'rmse_pred': [],
               'participant': [], 'testPair': []}

#%% ---------------------------------------------------------
# Optimization; looping through participants
#------------------------------------------------------------
for peep in participantList[:10]:
    data = df[df.subj==peep]
    ratings = data['rating'].values

    if preLoad:
        bestParams = np.load(dataDir + 'results/individuals/fit_'
                                + peep
                                + '_'
                                + str(n_features) + dnnFeatures + 'feat'
                                + fixParam
                                + truncatePredictions
                                + normalizedFeatures
                                + varScaling
                                + '.npy',
                                allow_pickle=True).tolist()
    else:
        # preliminaries
        print('Fitting data of ' + peep)
        resList = []
        rmseFitList = []
        rmsePredList = []
        testPairList = []

        # loop through all possible 16 pairs to be held out for CV
        for testPair in pairs:
            # loop through several seeds
            tmp_res = []
            tmp_rmse = []
            for seed in range(10):
                np.random.seed(seed)
                randStartValues = np.random.rand(nParams)
                # adjust scaling for starting point of alpha
                if 'alphazero' not in fixParam:
                    randStartValues[0] = randStartValues[0]/100
                # the usual optimization
                if constrainedMu:
                    thisRes = minimize(cost_fn, randStartValues,
                                   args=(data.iloc[:55], testPair,),
                                   method='SLSQP',
                                   constraints = cons,
                                   options={'maxiter': 1e4, 'ftol': 1e-06},
                                   bounds=bounds)
                else:
                    thisRes = minimize(cost_fn, randStartValues,
                                   args=(data.iloc[:55], testPair,),
                                   method='SLSQP',
                                   options={'maxiter': 1e4, 'ftol': 1e-06},
                                   bounds=bounds)

                tmp_res.append(thisRes)
                tmp_rmse.append(thisRes.fun)
            res = tmp_res[tmp_rmse.index(np.nanmin(tmp_rmse))]
            resList.append(res)
            rmseFitList.append(res.fun)
            testPairList.append(testPair)

            rmse_test = cost_fn(res.x, data.iloc[:55], testPair, test=True)
            rmsePredList.append(rmse_test)

            resDict['participant'] += [peep]
            resDict['testPair'] += [testPair]
            resDict['res'] += [res]
            resDict['rmse_fit'] += [res.fun]
            resDict['rmse_pred'] += [rmse_test]

    if not preLoad:
        bestRes = resList[rmsePredList.index(np.nanmin(rmsePredList))]
        bestTestPair = testPairList[rmsePredList.index(np.nanmin(rmsePredList))]
        bestParams = bestRes.x
        print('Selected best fit for ' + peep)
        print(bestRes)
        print('-------------')
        print('-------------')

    bestPred = pred_ratings(bestParams, data)

#%% ---------------------------------------------------------
# Plot
#------------------------------------------------------------
    if plot:
        r_fit = np.corrcoef(ratings[:55], bestPred[:55])[0,1]
    
        fig = plt.figure(0, (5,5))
        plt.plot(ratings[:55], bestPred[:55], 'oC0', label='fit')
        plt.plot([0,1], [0,1], '--k')
        plt.text(.1,.8, '$r_{fit} = $' + str(np.round(r_fit,2)))
        plt.ylim((0,1))
        plt.xlim((0,1))
        plt.xlabel('Data')
        plt.ylabel('Model')
        plt.title(peep)
        plt.show()
        plt.close()

#%% ---------------------------------------------------------
# Save
#------------------------------------------------------------
    if save:
        np.save(dataDir + 'results/individuals/fit_'
                                + peep
                                + '_'
                                + str(n_features) + dnnFeatures + 'feat'
                                + fixParam
                                + truncatePredictions
                                + normalizedFeatures
                                + varScaling
                                + '.npy', bestParams)

#%% ---------------------------------------------------------
# Res dict to pandas dataframe and save as .csv
#------------------------------------------------------------
if save:
    resDf = pd.DataFrame(resDict)
    resDf.to_csv(dataDir + 'results/allFits/' + 'allFits_'
                                + str(n_features) + dnnFeatures + 'feat'
                                + fixParam
                                + truncatePredictions
                                + normalizedFeatures
                                + varScaling
                                + '.csv')



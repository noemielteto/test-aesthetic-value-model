#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21
@author: aennebrielmann
"""

import os,sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

#%% ---------------------------------------------------------
# Specify directories; settings
#------------------------------------------------------------
os.chdir('..')
home_dir = os.getcwd()
dataDir = home_dir + '/'
resDir = dataDir + 'results/individuals/'
figDir = dataDir + 'figures/'

modelSpec = '3vggfeat'
plot = True

sigBetterWithLearn3D = ['mrx1q', 'oveao', 'quzfn', 'zv0ou']
sigBetterWithLearn2D = ['37o95', '6r8qk', '9gjbe', 'h647a', 'jmzs9',
                        'mbx6w', 'mrx1q', 'ms8r4', 'n0htt', 'njzz4',
                        'nq3sp', 'zv0ou']

#%% ---------------------------------------------------------
# Load results
#------------------------------------------------------------
infoDf = pd.read_csv(dataDir + 'perParticipantResults_cv.csv')
backupDf = pd.read_csv(dataDir + 'backup_results_refit_scrambled_'
                       + modelSpec + '.csv')
df = pd.read_csv(dataDir + 'merged_rating_data.csv')
participantList = infoDf.subj.unique()
participantList.sort()

#%% ---------------------------------------------------------
# Custom functions
#------------------------------------------------------------
sys.path.append((home_dir))
import fitPilot

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
                               + 'vgg_features/vgg_features_reduced_to_'
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

for peep in participantList:
    ratingData = df[df.subj==peep]
    thisBackup = backupDf[backupDf.participant==peep]

    res = np.load(resDir + 'fit_' + peep + '_' + modelSpec + '.npy',
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

    r_shuffled = thisBackup.sim_r.mean()
    rmse_shuffled = thisBackup.sim_rmse.mean()

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

#%% ---------------------------------------------------------
# save a summary of these summary statistics
#------------------------------------------------------------
resDf= pd.DataFrame(resDict)
resDf['r2_true'] = resDf.r_true.values**2
resDf['r2_shuffled'] = resDf.r_shuffled.values**2
resDf['r2_diff'] = resDf.r2_true - resDf.r2_shuffled
resDf['r_diff'] = resDf.r_true - resDf.r_shuffled
resDf['rmse_diff'] = resDf.rmse_true -resDf.rmse_shuffled
resDf['alpha*w_V'] = resDf.alpha * resDf.w_V

resDf.to_csv(dataDir + 'resDict_summary_refit_scrambled'
             + modelSpec + '.csv', index=False)

#%% ---------------------------------------------------------
# plot summary figure
#------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5,5))
sns.scatterplot(data=resDf[resDf.alpha!=0], x='r2_true', y='r2_shuffled',
                hue='w_V', size='alpha',
                sizes=(30, 200), alpha=1, palette='crest')
# mark participant with alpha=0 special
sns.scatterplot(data=resDf[resDf.alpha==0], x='r2_true', y='r2_shuffled',
                hue='w_V', marker="$\circ$", ec="face", s=100,
                sizes=(30, 200), alpha=1, palette='crest')
# mark participants sig better with learning 2D
sns.scatterplot(data=resDf[resDf.subj.isin(sigBetterWithLearn2D)],
                x='r2_true', y='r2_shuffled',
                color='r', marker="x", s=100,
                sizes=(30, 200), alpha=1, palette='crest')
# mark participants sig better with learning 3D
sns.scatterplot(data=resDf[resDf.subj.isin(sigBetterWithLearn3D)],
                x='r2_true', y='r2_shuffled',
                color='k', marker="x", s=100,
                sizes=(30, 200), alpha=1, palette='crest')
handles, labels = ax.get_legend_handles_labels()
# Put the legend out of the figure, remove added elments
properLabels = labels[:10]
properLabels[0] = r'$w_V$'
properLabels[5] = r'$\alpha$'
ax.legend(handles=handles[:10], labels=properLabels, title="",
          bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.plot([0,.75], [0,.75], ':k', alpha=.33)
# plt.title(modelSpec)
plt.xlabel(r'$R^2$ true order')
plt.ylabel(r'$R^2$ shuffled order')
sns.despine()
plt.show()
fig.savefig(figDir + modelSpec + 'r2 scatter.png', dpi=300, bbox_inches = "tight")
plt.close()

#%% ---------------------------------------------------------
# get statistics
#------------------------------------------------------------
_,p = stats.shapiro(resDf.r2_diff)
if p < 0.05:
    W, pDiff = stats.wilcoxon(resDf.r2_diff)
else:
    T, pDiff = stats.ttest_1samp(resDf.r2_diff, popmean=0, nan_policy='omit')

print('corr r2 true - r2 diff')
print(stats.pearsonr(resDf.r2_true, resDf.r2_diff))

print('corr alpha r2 diff')
print(stats.pearsonr(resDf.alpha, resDf.r2_diff))
print('corr wV r2 diff')
print(stats.pearsonr(resDf.w_V, resDf.r2_diff))

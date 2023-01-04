# -*- coding: utf-8 -*-
"""
Created on Fri Jan 7 2022
Last updated Feb 04 2022: adjust for cross-validated results
@author: abrielmann
"""

import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # disable SettingWithCopyWarning


#%% ---------------------------------------------------------
# Specify directories; settings
#------------------------------------------------------------
os.chdir('..')
os.chdir('..')
os.chdir('..')
homeDir = os.getcwd()
dataDir = homeDir + '/Dogs of Instagram/analysis/'
saveDir = dataDir + 'results_paper_draft/'

plot = True
save = True

n_features = 2 # dimensionality of assumed feature space
scaleVariances = False
if scaleVariances:
    varScaling = 'scaledVars'
else:
    varScaling = ''
truncatePredictions = '' # either '' for regular predictions or 'truncated'
normalizedFeatures = '' # either 'normed' or ''
constrainedMu = False
dnnFeatures = 'ResNet50' # either '' or spec of DNN (vgg, ResNet50)
fixParam = ''

#%% ---------------------------------------------------------
# Load data
#------------------------------------------------------------
df = pd.read_csv(dataDir + 'merged_rating_data.csv')
initialFits = pd.read_csv(dataDir + 'results_paper_draft/backupFits/allFits_'
                         + str(n_features) + dnnFeatures + 'feat'
                                + fixParam
                                + truncatePredictions
                                + normalizedFeatures
                                + varScaling
                                + '.csv')
# followUpFits = pd.read_csv(dataDir + 'results_cv/backupFits/second batch/allFits_'
#                           + str(n_features) + dnnFeatures + 'feat'+ fixParam
#                           + '.csv')
participantList = initialFits.participant.unique().tolist()
# refitPeeps = followUpFits.participant.unique().tolist()

allFitInfo = initialFits.copy()
# for peep in refitPeeps:
#     allFitInfo[allFitInfo.participant==peep] = followUpFits[followUpFits.participant==peep]

#%% ---------------------------------------------------------
# Get avg/nanmedian rmse, r across cv folds
#------------------------------------------------------------
avg_rmse = []
med_rmse = []
med_rmse_fit = []
sd_rmse = []
min_rmse = []
max_rmse = []
peeps = []

for peep in participantList:
    thisDf = allFitInfo[allFitInfo.participant==peep]
    avg_rmse.append(np.mean(thisDf.rmse_pred))
    med_rmse.append(np.nanmedian(thisDf.rmse_pred))
    med_rmse_fit.append(np.nanmedian(thisDf.rmse_fit))
    sd_rmse.append(np.nanstd(thisDf.rmse_pred))
    min_rmse.append(np.nanmin(thisDf.rmse_pred))
    if np.nanmax(thisDf.rmse_pred) > 2:
         max_rmse.append(2)
    else:
        max_rmse.append(np.nanmax(thisDf.rmse_pred))

    peeps.append(peep)

#%% ---------------------------------------------------------
# save
#------------------------------------------------------------
d = {'avg_rmse': avg_rmse, 'med_rmse': med_rmse, 'med_rmse_fit': med_rmse_fit,
     'sd_rmse': sd_rmse, 'min_rmse': min_rmse, 'max_rmse': max_rmse,
          'subj': peeps}
df = pd.DataFrame(d)
if save:

    # save it with a reasonable .csv name
    df.to_csv((saveDir + '' + 'model_results_'
                + str(n_features) + dnnFeatures + 'feat'
                + fixParam
                + truncatePredictions
                + normalizedFeatures
                + varScaling
                + '.csv'), index=False)

#%% ---------------------------------------------------------
# plot with errorbars
#------------------------------------------------------------
from matplotlib import pyplot as plt
plt.errorbar(x=df.subj, y=df.med_rmse,
             yerr=np.array([df.med_rmse-df.min_rmse, df.max_rmse-df.med_rmse]))
plt.xticks([])
plt.ylabel('RMSE [min, median, max]')
plt.xlabel('participant')
plt.ylim((0,2))
plt.title('model ' + str(n_features) + dnnFeatures + ' features ' + fixParam
          + truncatePredictions + normalizedFeatures + varScaling)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 2021
Last updated Feb 04 2022: adapted for cv fits
@author: aennebrielmann
"""

import os, sys
import pandas as pd
from scipy import stats

#%% ---------------------------------------------------------
# Specify directories; settings
#------------------------------------------------------------
os.chdir('..')
os.chdir('..')
os.chdir('..')
home_dir = os.getcwd()
dataDir = home_dir + '/Papers/RoySocB'
figDir = dataDir + '/final paper figures/'

#%% ---------------------------------------------------------
# load data, functions
#------------------------------------------------------------
df = pd.read_csv(dataDir + '/perParticipantResults_cv.csv')
sys.path.append((home_dir + "/python_packages"))
from aestheticsModel import figureFunctions as ff

#%% ---------------------------------------------------------
# Re-format df to long
#------------------------------------------------------------
longDf = pd.wide_to_long(df,
                         stubnames=['med_rmse', 'avg_rmse'],
                         i=df.columns[:18],
                         j='model', sep='_', suffix='.*')
longDf = longDf.reset_index()
longDf['model'] = longDf['model'].str.replace('_results_','')
longDf['model'] = longDf['model'].str.replace('ure','')
longDf['model'] = longDf['model'].str.replace('glm_rating ~','')

#%% ---------------------------------------------------------
# Plot
#------------------------------------------------------------
plotDf = longDf.copy()
plotDf['model'] = pd.Categorical(plotDf.model)
remap_cats = {'fit_paper_draftmodel_3vggfeatalphazero_wVzero': '3D no learning',
       'fit_paper_draftmodel_2vggfeatalphazero_wVzero': '2D no learning',
       'fit_paper_draftmodel_4vggfeat': '4D full', 
       'fit_paper_draftmodel_3vggfeat': '3D full', 
       'fit_paper_draftmodel_2vggfeat': '2D full',
       'LOOavg': 'LOO-average'}
plotDf['model'] = plotDf.model.cat.rename_categories(remap_cats)
modelNames = plotDf.model.unique().tolist()
modelNames.pop(modelNames.index('LOO-average'))
fig = ff.scatter_model_comparison(plotDf, 'med_rmse', modelNames,
                              baselineModel = 'LOO-average',
                              minVal=0, maxVal=.5)
fig.suptitle('Median RMSE')
fig.axes[0].set_ylabel('3D no learning')
fig.axes[1].set_ylabel('2D no learning')
fig.axes[2].set_ylabel('3D full')
fig.axes[3].set_ylabel('2D full')
for axcount in range(4):
    fig.axes[axcount].set_title('')
fig.savefig(figDir + 'model comparison scatters.png', dpi=300, bbox_inches = "tight")

#%% ---------------------------------------------------------
# Get and print median RMSE, r per model
#------------------------------------------------------------
tableDf = pd.DataFrame(longDf.groupby(['model'])['med_rmse'].median())
tableDf['RMSE SD'] = longDf.groupby(['model'])['med_rmse'].std()
# tableDf['r_pred'] = longDf.groupby(['model'])['r_pred'].median()
# tableDf['r SD'] = longDf.groupby(['model'])['r_pred'].std()
# tableDf.reset_index(inplace=True)
print(tableDf.to_latex(float_format="{:0.3f}".format))

#%% ---------------------------------------------------------
# run paired t-tests
#------------------------------------------------------------
for model in plotDf.model.unique():
    _,p = stats.shapiro(plotDf[plotDf.model==model]['med_rmse'])
    if model != 'LOO-average':
        if p>0.05:
            print('Comparison to ' + model)
            print(stats.ttest_rel(plotDf[plotDf.model=='LOO-average']['med_rmse'],
                            (plotDf[plotDf.model==model]['med_rmse'])))
        else:
            print('Comparison to ' + model)
            print(stats.wilcoxon(plotDf[plotDf.model=='LOO-average']['med_rmse'],
                            (plotDf[plotDf.model==model]['med_rmse'])))
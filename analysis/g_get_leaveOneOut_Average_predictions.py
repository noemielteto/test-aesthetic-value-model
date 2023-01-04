#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 2021

@author: aennebrielmann
 As a benchmark for the true value of using individual ratings instead of
 population averages: look at the error and correlation values we get when
 predicting an individual's rating based on the average rating of the rest of the population

"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#%% ---------------------------------------------------------
# Specify directories; settings
#------------------------------------------------------------
os.chdir('..')
home_dir = os.getcwd()
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
participantList.sort()
dataDir = home_dir + '/'
save = True

#%% ---------------------------------------------------------
# Load data
#------------------------------------------------------------
df = pd.read_csv(dataDir + 'merged_rating_data.csv')
viewDf = pd.read_csv(dataDir + 'merged_viewing_data.csv')
imList = pd.unique(df.imageInd)

#%% ---------------------------------------------------------
# define RMSE and correlation as separate functions to keep things clean
#------------------------------------------------------------
def get_rmse(data, pred):
    rmse = np.mean(np.sqrt((data - pred)**2))
    return rmse

def get_corr(data, pred):
    r = np.corrcoef(data, pred)[0,1]
    return r

#%% ---------------------------------------------------------
# Get leave-one-pout averages and predictive quality measures for it
#------------------------------------------------------------
rmse = []
r = []
for participant in participantList:
    # fetch first ratings only
    thisDf = df[(df.subj==participant) & (df.block==1)]
    restDf = df[(df.subj!=participant) & (df.block==1)]

    avgRating = [restDf.loc[restDf.imageInd==im, 'rating'].mean() for im in imList]
    predRating = [avgRating[im] for im in thisDf.imageInd]

    rmse.append(get_rmse(thisDf.rating.values, predRating))
    r.append(get_corr(thisDf.rating.values, predRating))

#%% ---------------------------------------------------------
# Plot
#------------------------------------------------------------
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].hist(r)
axs[0].vlines(np.mean(r), 0, 20, 'r')
axs[0].set_title('r')
axs[1].hist(rmse)
axs[1].vlines(np.mean(rmse), 0, 20, 'r')
axs[1].set_title('RMSE')

#%% ---------------------------------------------------------
# Save
#------------------------------------------------------------
infoDf = pd.read_csv(dataDir + '/perParticipantResults_cv.csv')
d = {'subj': participantList, 'med_rmse_LOOavg': rmse, 'r_LOOavg': r}
resDf = pd.DataFrame(d)
df = pd.merge(infoDf, resDf, on='subj')
# save this as a new, merged data frame
df.to_csv(dataDir + '/perParticipantResults_cv.csv', index=False)

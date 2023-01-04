# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21

@author: abrielmann

"""
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # disable SettingWithCopyWarning
from matplotlib import pyplot as plt
import seaborn as sns
import pingouin as pg # for ICC
from scipy import stats # for testing distribution of ratings

#%% ---------------------------------------------------------
# Specify directories; settings
#------------------------------------------------------------
os.chdir('..')
homeDir = os.getcwd()
# IDs (not including excluded subjs):
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
dataDir = homeDir + '/'
plot = True
plotIndividuals = False # plots for individual participants' rating distribution

#%% ---------------------------------------------------------
# Load data
#------------------------------------------------------------
rawDf = pd.read_csv(dataDir + 'merged_rating_data.csv')
df = rawDf[rawDf['subj'].isin(participantList)]
df = df[df['block']==1]

#%% ---------------------------------------------------------
# ICC and rating distribution per image
#------------------------------------------------------------
icc = pg.intraclass_corr(data=df, targets='imageInd', raters='subj',
                         ratings='rating').round(3)
print(icc.set_index("Type"))

mdnRatingsPerImage = df.groupby(['imageInd'])['rating'].median().sort_values()
mdnImgs =  mdnRatingsPerImage[mdnRatingsPerImage==mdnRatingsPerImage.median()]

fig, ax = plt.subplots(figsize=(15,5))
sns.stripplot(data=df, x='imageInd', y='rating', alpha=.33, color='C0',
              order=mdnRatingsPerImage.index)
ax.set_xlabel('image #')
sns.despine()
plt.show()
plt.close()

#%% ---------------------------------------------------------
# Check distribution of ratings
#------------------------------------------------------------
pValShap = []

for peep in participantList:
    thisDf = df[df.subj==peep]

    # first, test whether ratings are normally distributed
    shapTest = stats.shapiro(thisDf['rating'])
    pValShap.append(shapTest.pvalue)

    # and illustrate distribution
    if plotIndividuals:
        g = sns.displot(data=thisDf, x='rating', hue='block',
                        kind='kde', palette='tab10')
        g.ax.set_xlim((0,1))
        plt.show()
        plt.close()

print('')
print('N participants with non-normal rating dist: ')
print(str(np.sum([p<0.05 for p in pValShap])))
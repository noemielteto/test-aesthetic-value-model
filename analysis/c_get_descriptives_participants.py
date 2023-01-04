# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 2022
Modified Wed Jan 4 2023: cleaning up

@author: abrielmann

"""
import os
import pandas as pd
pd.options.mode.chained_assignment = None # disable SettingWithCopyWarning

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
plot = False
plotIndividuals = True # plots for individual participants' rating distribution

#%% ---------------------------------------------------------
# Load data
#------------------------------------------------------------
rawDf = pd.read_csv(dataDir + 'merged_participantInfo.csv')
df = rawDf[rawDf['subj'].isin(participantList)]

#%% ---------------------------------------------------------
# demographics; use print() to display
#------------------------------------------------------------
df.age.describe()
df.education.value_counts()
df.nationality.value_counts()
df.dogPresent.value_counts()
df.dogPast.value_counts()
df.imgBrowseDur.value_counts()
df.imgBrowseLike.value_counts()
df.dogLiking.value_counts()
df.boredom.value_counts()

#%%---------------------------------------------------------
# related to rating behavior; use print() to display
#------------------------------------------------------------
df.corrRatingSliderPos.describe()
df.ratingCorrWithAvg.describe()



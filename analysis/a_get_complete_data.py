# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:57:12 2021
Last updated Thu Jan 20 2022: cleaning up

@author: abrielmann

read in the data and transform it such that we can store usable dataframe(s)
for each participant as well as big data frames for each type of response
across participants
"""

import os
import glob
import ast
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'; we are aware of the caveats but know that this is not an issue, here
import numpy as np
from scipy.stats import entropy, zscore

#%% ---------------------------------------------------------
# Specify directories
#------------------------------------------------------------
#   further specificcations regarding the rounding of RTs, optimization, etc.
#   can be changed in the respective places in the main code
os.chdir('..')
homeDir = os.getcwd()

# get data
dataDir = homeDir + '/data_experiment/*'
saveDir = homeDir + '/'

#%% ---------------------------------------------------------
# Initialize variables
#------------------------------------------------------------
dataList = glob.glob(dataDir)
# initialize variables for end-of-experiment questions
subjs_end = []
wantPrevImg = []
strategies = []
switchReasons = []
boredom = []
dogLiking = []
dogPast = []
dogPresent = []
imgBrowseDurs = []
imgBrowseLikes = []
ages = []
educations = []
genders = []
nationalities = []
rating_entropy = []
roundedRating_entropy = []
corr_rating_sliderPos = []
block = []

#initialize variables for rating data
ratings = []
zScored_ratings = []
subjs_rate = []
rts_rate = []
imgs_rate = []
sliderStart = []
trialInd_rate = []

# initialize variables for view time data
viewTimes = []
subjs_view = []
imgs_view = []
trialInd_view = []

#%% ---------------------------------------------------------
# Loop through all data files and fetch variables
#------------------------------------------------------------
for ss in range(len(dataList)):
    thisData = pd.read_csv(dataList[ss])
    ratingData = thisData[thisData.trial_type=='html-slider-response']

    # all viewing trials, excluding practice
    viewingData = thisData[thisData.trial_type=='timed-html-button-response'][5:]

    strategyData = ast.literal_eval(thisData.response.values[-3])
    experienceData = ast.literal_eval(thisData.response.values[-4])
    demographicData = ast.literal_eval(thisData.response.values[-5])

    ratings.extend(ratingData.response.tolist())
    zScored_ratings.extend(zscore(ratingData.response.astype(float)).tolist())
    subjs_rate.extend(ratingData.id.tolist())
    rts_rate.extend(ratingData.rt.tolist())
    imgs_rate.extend(ratingData.image_name.tolist())
    sliderStart.extend(ratingData.slider_start.tolist())
    trialInd_rate.extend(ratingData.trial_index.tolist())

    viewTimes.extend(viewingData.rt.tolist())
    subjs_view.extend(viewingData.id.tolist())
    imgs_view.extend(viewingData.image_name.tolist())
    trialInd_view.extend(viewingData.trial_index.tolist())

    # append, don't extend lists for demographics
    # we want to keep those strings together and not dissolve them into letters
    subjs_end.append(thisData.id[0])
    wantPrevImg.append(strategyData['previousImage'])
    strategies.append(strategyData['strategies'])
    switchReasons.append(strategyData['switchReason'])
    boredom.append(experienceData['boredom'])
    dogLiking.append(experienceData['dogLiking'])
    dogPast.append(experienceData['dogPast'])
    dogPresent.append(experienceData['dogPresent'])
    imgBrowseDurs.append(experienceData['imageBrowsingDuration'])
    imgBrowseLikes.append(experienceData['imageBrowsingLiking'])
    ages.append(demographicData['age'])
    educations.append(demographicData['education'])
    genders.append(demographicData['gender'])
    nationalities.append(demographicData['nationality'])

    # block number
    block.extend([int(trial>60)+1 for trial in ratingData.trial_index.astype(int)])

    # entropy across all ratings
    counts = ratingData.response.value_counts()
    rating_entropy.append(entropy(counts))
    ratingData['roundedRating'] = np.round(ratingData.response.astype(float)/50)
    counts = ratingData.roundedRating.value_counts()
    roundedRating_entropy.append(entropy(counts))

    # correlation between ratings and initial slider position
    corr = np.corrcoef(ratingData.response.astype(float),
                       ratingData.slider_start.astype(float))[0,1]
    corr_rating_sliderPos.append(corr)

#%% ---------------------------------------------------------
# Create df for rating data
#------------------------------------------------------------
d = {'subj':subjs_rate, 'rating':ratings, 'zScored_ratings': zScored_ratings,
     'rt':rts_rate, 'raw_image_name':imgs_rate,
     'sliderStart':sliderStart, 'trial': trialInd_rate, 'block': block}
df = pd.DataFrame(d)
df.rating = df.rating.astype(int) # convert ratings to numeric
df.rating = df.rating/500 # also convert them to % of rating scale
df.rt = df.rt/100 # also down-scale RTs

# clean up the image names
df['image'] = df['raw_image_name'].copy()
df['image'] = df['image'].str.replace('images/experiment/','', regex=True)
df['image'] = df['image'].str.replace('pair_','')
df['image'] = df['image'].str.replace('_start','')
df['image'] = df['image'].str.replace('.png','')
df['image'] = df['image'].str.replace('_out_tensor','')
df['image'] = df['image'].str.replace('(0.3333)','0.33')
df['image'] = df['image'].str.replace('(0.5000)','0.5')
df['image'] = df['image'].str.replace('(0.6667)','0.67')
df['image'] = df['image'].str.replace(r'\(','_')
df['image'] = df['image'].str.replace(r'\)','')

# In these cleanup lines, consider using regex=True, otherwise it gives a
# FutureWarning: The default value of regex will change from True to False in a
# future version.

# strip away morph stage, save as image pair variable
df['pair'] = df['image'].str.replace('0.33','')
df['pair'] = df['pair'].str.replace('0.5','')
df['pair'] = df['pair'].str.replace('0.67','')

# same

# now separately store variables for each present source image
sourceImages = df['image'].str.split(pat='_', expand=True)
df['sourceImageA'] = sourceImages[0]
df['sourceImageB'] = sourceImages[1]
df['morphPercent'] = sourceImages[2]

# same

# add idx, sorted such that they map onto the right stimulus parameters
idxDf = pd.read_csv(saveDir + '/map_imgName_imgIdx.csv')
df = df.merge(idxDf, on='image')
# !! important: after merging, we need to re-sort by trial number
df = df.sort_values(by=['subj', 'trial'])

# save
df.to_csv(saveDir + 'merged_rating_data.csv')

#%% ---------------------------------------------------------
# df for demographics and end-of-experiment questions
#------------------------------------------------------------
# the bits we have to calculate here are the ones that relate to the entire
# data across participants
avgRatingPerImage = []
for im in np.unique(df.imageInd):
    avgRatingPerImage.append(np.mean(df.loc[df.imageInd==im, 'rating']))

ratingCorrAvgRating = []
for subj in subjs_end:
    thisDf = df[df.subj==subj]
    theseRatings = df.loc[df.subj==subj,'rating']
    avgsArray = np.array(avgRatingPerImage)
    matchedAvgRatings = avgsArray[thisDf.imageInd.astype(int)]
    corr = np.corrcoef(theseRatings, matchedAvgRatings)[0,1]
    ratingCorrAvgRating.append(corr)

d = {'subj':subjs_end, 'wantPrevImg':wantPrevImg, 'strategy':strategies,
     'switchReason': switchReasons, 'boredom': boredom, 'dogLiking': dogLiking,
     'dogPast': dogPast, 'dogPresent': dogPresent,
     'imgBrowseDur': imgBrowseDurs, 'imgBrowseLike': imgBrowseLikes,
     'age': ages, 'education': educations,
     'gender': genders, 'nationality': nationalities,
     'rating_entropy': rating_entropy,
     'roundedRating_entropy': roundedRating_entropy,
     'ratingCorrWithAvg': ratingCorrAvgRating,
     'corrRatingSliderPos': corr_rating_sliderPos}
df = pd.DataFrame(d)

# save
df.to_csv(saveDir + '/merged_participantInfo.csv', index=False)

#%% ---------------------------------------------------------
# Df for free viewing data
#------------------------------------------------------------
d = {'subj':subjs_view, 'viewTime':viewTimes, 'image':imgs_view,
     'trial': trialInd_view}
df = pd.DataFrame(d)
#create a ranked view time variable
viewTimeRanks = []
counter = 0
for peep in subjs_end:
    thisDf = df[df.subj==peep]
    viewTimeRanks.extend(thisDf.viewTime.rank().values)
    counter += len(thisDf)
df['viewTimeRank'] = viewTimeRanks
df.viewTime = df.viewTime/100 # down-scale viewTimes

# clean up the image names
df['image'] = df['image'].str.replace('images/experiment/','')
df['image'] = df['image'].str.replace('pair_','')
df['image'] = df['image'].str.replace('_start','')
df['image'] = df['image'].str.replace('.png','')
df['image'] = df['image'].str.replace('_out_tensor','')
df['image'] = df['image'].str.replace('(0.3333)','0.33')
df['image'] = df['image'].str.replace('(0.5000)','0.5')
df['image'] = df['image'].str.replace('(0.6667)','0.67')
df['image'] = df['image'].str.replace(r'\(','_')
df['image'] = df['image'].str.replace(r'\)','')

# strip away morph stage, save as image pair variable
df['pair'] = df['image'].str.replace('0.33','')
df['pair'] = df['pair'].str.replace('0.5','')
df['pair'] = df['pair'].str.replace('0.67','')

# now separately store variables for each present source image
sourceImages = df['image'].str.split(pat='_', expand=True)
df['sourceImageA'] = sourceImages[0]
df['sourceImageB'] = sourceImages[1]
df['morphPercent'] = sourceImages[2]

# add idx, sorted such that they map onto the right stimulus parameters
idxDf = pd.read_csv(saveDir + 'map_imgName_imgIdx.csv')
df = df.merge(idxDf, on='image')
# after merging, we need to re-sort by trial number
df = df.sort_values(by=['subj', 'trial'])

# save
df.to_csv(saveDir + 'merged_viewing_data.csv')

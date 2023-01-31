# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:25:07 2021

@author: abrielmann
"""
import os
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#%% Settings
home_dir = os.getcwd()

n_features = 2
n_base_stims = 7
n_morphs = 5
show_outlier = True
dataDir = home_dir + '/'
imageDir = dataDir  + 'experiment_code/images/experiment/'
dnn = 'VGG'

# !!! DO pay attention to file order!
imageFiles = glob.glob(imageDir + '*')
imageFiles.sort()

#%% features, data (for image names)
df = pd.read_csv(dataDir + 'merged_rating_data.csv')
# get (reduced) dnn features
featureDf = pd.read_pickle(dataDir + dnn + '_features/' 
                           + dnn + '_features_reduced_to_'
                           + str(n_features) + '.pkl')
# now create an array that contains featuers of the images in the right order
# !!! careful here to load these in the same order as in imageFiles
rawImageNameList = pd.unique(df.raw_image_name)
imageList = pd.unique(df.image)
nameDf = pd.DataFrame({'raw': rawImageNameList, 'short': imageList})
nameDf.sort_values(by='raw', inplace=True)

for ii in range(len(imageList)):
    img = nameDf.iloc[ii].short
    if ii==0:
        dnnFeatures = featureDf.feature_array[featureDf.image==img].values[0]
    else:
        dnnFeatures = np.vstack([dnnFeatures,
                       featureDf.feature_array[featureDf.image==img].values[0]])

# for visualizations: scale dnnFeatures to a range from -250 to 250
dnnFeatures = ((dnnFeatures-np.min(dnnFeatures))
               /(np.max(dnnFeatures)-np.min(dnnFeatures))*500-250)

#%% Define the function that will place our images in a grid
def plot_img(arr_image, xy, ax, imSize, xDim=0, yDim=1):
    x = xy[xDim]
    y = xy[yDim]
    axin = ax.inset_axes([x-imSize/2, y-imSize/2, imSize,imSize],
                         transform=ax.transData) #scale
    axin.imshow(arr_image)
    axin.axis('off')

def plot_pair(imList, lim=(-250,250), size=50, xDim=0, yDim=1):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    for im in imList:
        thisIm = plt.imread(imageFiles[im], format='.png')
        xy = dnnFeatures[im]
        plot_img(thisIm, xy, ax, size, xDim, yDim)
    ax.set_xlabel('Dimension ' + str(xDim), fontsize=24)
    ax.set_ylabel('Dimension ' + str(yDim), fontsize=24)
    plt.xticks([])
    plt.yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # change all spines
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(3)

#%% plot, base stimuli only first, on a grid
imList = np.arange(n_base_stims).tolist()
if not show_outlier:
    imList.pop(4)
plot_pair(imList, xDim=0, yDim=1)
plt.title(dnn, fontsize=20)

#%% plot all (this might be a mess)
# change limits to see the outlier or no
imList = np.arange(55).tolist()
if not show_outlier:
    imList.pop(4)
plot_pair(imList, xDim=0, yDim=1)
plt.title(dnn, fontsize=20)

#%% plotting individual pairs will likely be a matter of hand-picking idxs
# pair_101_57_idxs = [0, 7, 8, 9, 3]
# plot_pair(pair_101_57_idxs, xDim=0, yDim=1)

# pair_101_64_idxs = [0, 10, 11, 12, 5]
# plot_pair(pair_101_64_idxs, xDim=0, yDim=1)

# pair_101_92_idxs = [0, 13, 14, 15, 6]
# plot_pair(pair_101_92_idxs, lim=(-180,-110), size=15)

# pair_37_41_idxs = [1, 19, 20,21, 2]
# plot_pair(pair_37_41_idxs, xDim=0, yDim=1)

#%% a nice thing for the paper: a triangle of stimuli

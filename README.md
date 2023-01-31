# Testing a computational model of aesthetic value
Data and analysis files for an experiment testing the viability of [a computational model of aesthetic value developed by Brielmann and Dayan (in press)](https://psyarxiv.com/eaqkc/).
Analyses and plotting functions used here depend on the [main package containing the core functions of the computational model of aesthetic value](https://github.com/aenneb/intro-aesthetic-value-model).

# Folder content

The main directory contains the pre-processed data from all participants as well as the results of the simulated, random-order refits.

In addition, it contains "map_imgName_imgIdx.csv" which provides a safe way to ensure that DNN-derived image features are mapped unto the correct images in the data files. 

As a bonus, it contains the "plot)images_inVGGspace.py" script which visualizes the location of the stimulus images in reduced DNN-feature space.

## analysis

Contains all scripts needed to reproduce the analyses reported in the paper as well as the code for re-creating the figures. NOTE that the model fitting may take a substantial amount of time, so if you do not want to tweak this, you might want to use the available model fitting results from the "results" folder.

## data_experiment

Contains all raw data that was collected. NOTE that these files also contain participants that were excluded from analyses.

## results

Contains the .csv and .npy files for all preliminary and final results of the analyses that are the basis for statistical analyses and plots.

## experiment_code

Contains the complete code for running the experiment as described in the paper *except* for the jspsych files. Download jspsych 6.3.1 and save the folder inside the experiment_code folder to run the experiment.

## VGG_features

Contains the feature values and their PCA-reduced version for all stimuli based on vanilla, pretrained VGG-16.

## ResNet50_features

Contains the feature values and their PCA-reduced version for all stimuli based on vanilla, pretrained ResNet50.

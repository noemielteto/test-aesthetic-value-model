#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:32:04 2020

@author: aennebrielmann

Collection of outsourced code for creating figures
"""
import numpy as np
from matplotlib import pyplot as plt

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def get_dist_plot_data(mu_X, cov_X, mu_true, cov_true,
                       xmin=-10, ymin = -10, xmax=10, ymax=10, N=25):

    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z_X = multivariate_gaussian(pos, mu_X, cov_X)
    Z_true = multivariate_gaussian(pos, mu_true, cov_true)

    return X, Y, Z_X, Z_true

def scatter_model_comparison(df, measure, modelNames,
                             baselineModel = 'model2fitFeats',
                             minVal=0, maxVal=1):
    nModels = len(modelNames)
    ncols = int(np.ceil(nModels/2))
    nPeeps = len(df.subj.unique())
    baseModelValues = df[df.model==baselineModel][measure]

    fig, axs = plt.subplots(2, ncols, figsize=(ncols*3,6))
    fig.suptitle(measure)
    for count in range(nModels):
        ax = axs.flatten()[count]
        ax.plot(baseModelValues,
                  df[df.model==modelNames[count]][measure],
                  'x', alpha=.5)
        # add averages as salient point
        ax.plot(baseModelValues.mean(),
                  df[df.model==modelNames[count]][measure].mean(),
                  'or', markersize=10)

        # add a count of num participants where baseModel outperforms
        # the other (in absolute terms)
        compModelValues = df[df.model==modelNames[count]][measure]
        if 'rmse' in measure:
            basemodelBetter = np.sum(baseModelValues.values < compModelValues.values)
            ax.set_title(('x better in '
                      + str(basemodelBetter) + '/' + str(nPeeps) + ' cases.'))
        elif 'r_' in measure:
            basemodelBetter = np.sum(baseModelValues.values > compModelValues.values)
            ax.set_title(('x better in '
                      + str(basemodelBetter) + '/' + str(nPeeps) + ' cases.'))

        ax.set_xlabel(baselineModel)
        ax.set_ylabel(modelNames[count])
        ax.set_xlim((minVal, maxVal))
        ax.set_ylim((minVal, maxVal))
        ax.plot([minVal, maxVal], [minVal,maxVal], ':k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        fig.tight_layout(pad=1)
    
    return fig
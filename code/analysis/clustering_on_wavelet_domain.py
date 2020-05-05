# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Testing clustering on wavelet basis 

# +
# %load_ext blackcellmagic

import sleepa as sl
import numpy as np
#from numba import njit
import pandas as pd

import pywt
#import matplotlib.pyplot as plt
#from numpy.fft import fft

# Interactive plotting stack
# import bokeh_catplot
# import bebi103
# import colorcet as cc
# import hvplot
# import hvplot.pandas 
# import bokeh.io
# import bokeh.layouts
# import bokeh.plotting
# import holoviews as hv
# from bokeh.themes import Theme
# from bokeh.plotting import figure
# from holoviews.operation import datashader
# from holoviews.operation.datashader import rasterize

# bokeh.io.output_notebook()
# hv.extension('bokeh')

# Set random seed
np.random.seed(34917)

import warnings
warnings.filterwarnings(action = 'ignore')

# -

# ### Load datasets

# +
# Set report data path 
path =  '../../../prober_lab/data/'

# Show current datasets 
# !ls {path}

# +
# Load clustering report df 
#report_df = pd.read_csv(path + '200426_clus_report_df.csv')

# Take a look 
#report_df.head(3)

# Load fish #7 from the Neuron paper
path_ahrens = '../../../prober_lab/data'
neuron_data = sl.load_ahrens_data_figshare('../../../prober_lab/data/subject_7/TimeSeries.h5')

# Initialize variables for time array
n_timepoints = neuron_data.shape[1] # timepoints
sampling_rate = 2 # Hz

# Generate time array in seconds 
time_arr = np.arange(n_timepoints) / 2

# Initialize scales array 
scales = np.arange(1, 2**6)
# -

# %%time
# Compute wavelet basis 
wavelet_basis = sl.get_wavelet_basis(
    neuron_data[5000, :],
    time_arr,
    scales,
    wavelet = 'gaus5'
)

pd.DataFrame(wavelet_basis).to_csv(path + 'fish_7_wavelet_basis.csv', index = False)

# %%time
# Apply clustering pipeline to wavelet basis 
plot_df = cluster_neuron_data_v2(
    neuron_data,
    pca_components= 0.6,
    n_neighbors= 30,
    max_clus = 20,
)

plot_df.to_csv(path + 'fish_7_umap_wavelet_basis.csv', index = False)

import sleepa as sl
import numpy as np
from numba import njit
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import h5py

@njit
def corr_numba(x1, x2):
    """
    Returns Pearson correlation coefficient in numba version
    """
    return numpy.corrcoef(x1, x2)


path = '../../../prober_lab/data/subject_10'
neurons_path = path + 'TimeSeries.h5'
mat_path = path + 'data_full.mat'

# Load neuron dataset
neuron_data = sl.load_ahrens_data_figshare(neurons_path)

# Load behavioral data
# Load mat file with behavior data
mat_file =  sio.loadmat(mat_path)

# Extract behavioral data: left channel
behavior_raw = mat_file['data']['Behavior_raw'][0][0]

# Set variables: number of neurons and number of timepoints
n_neurons, n_timepoints = neuron_data.shape

# Trim last data points
behavior_ = behavior[:n_timepoints]

# Initialize correlation array
corr_coefs = np.empty(n_neurons)

# Loop over neurons
for ix in tqdm(np.arange(neuron)):

    corr_coefs[ix] = corr_numba(rest_regressor, neuron_data[ix, :])

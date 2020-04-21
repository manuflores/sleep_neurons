# sleep_neurons


from umap import UMAP as umap 
from numba import njit 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
import numpy as np 
import h5py

from sklearn.decomposition import IncrementalPCA as iPCA
from sklearn.decomposition import PCA 
from sklearn.linear_model import Lasso 
import scipy.fftpack as spfft
import cvxpy as cvx



@njit
def rolling_mean_numba(x, window_size= 30):
    cumsum = np.cumsum(x)
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


# Load and preprocess data from the FigShare repository

def preprocess_ahrens_figshare(path, 
    #norm_method = 'rolling_mean'
    ):

    """
    Loads data from the FigShare repository. 
    The data comes from the Neuron 201? paper.

    """

    # Download data from figshare 
    # Uncompress 

    # Load data 
    dat = h5py.File(path, 'r')
    #'../Downloads/subject_7/TimeSeries.h5'

    # Get the neuron activity traces 
    neuron_data = dat['CellResp']

    # Turn data into numpy array
    # *To-do: Could we do sparse mat and apply function along axis ?
    data_arr = np.array(neuron_data)


    # Transpose array 
    transposed = data_arr.T

    return transposed 

def subsample_data(neuron_data, sample_size = 10000):
    """
    Acquires a subsample of the Neuron dataset. 
    This function samples a set of neurons without replacement.  

    Params 
    -----------

    Returns
    -----------
    rand_ix (array-like):
        Array containing the chosen indices

    sample_neurons (array-like ):
        Array with shape (sample_size, neuron_data.shape[1])
        containing a subset of the neuron traces. 

    """
    # Get random indices sampling without replacement
    rand_ix = np.random.choice(
        np.arange(neuron_data.shape[0]), size= sample_size, replace=False
    )

    # Get subsample by choosing indices along rows
    sample_neurons = neuron_data[rand_ix, :]

    return rand_ix, sample_neurons 



#def visualize_cluster_traces_bokeh():

#def visualize_cluster_traces_mpl():


def cluster_neuron_data(neuron_data, sample_size, window_size, **kwargs):

	"""
	Wrapper for first version of the pipeline . 
	"""

	rand_ixs, neuron_samples = subsample_data(neuron_data, sample_size = sample_size)

	# Apply rolling mean to each row 
	processed_neurons = np.apply_along_axis(
    	func1d=rolling_mean_numba,
    	axis=0,
    	arr = neuron_samples,
    	**{'window_size':window_size}
	)

	# Apply PCA to processed neurons
	pcs = PCA(n_components = 20).fit_transform(processed_neurons)


	# Apply UMAP to neurons' principal components
	embedding = umap(
	    min_dist =0, 
	    #n_neighbors=20,
	    n_components = 2, 
	    random_state = 3287354).fit_transform(pcs)

	# Apply DBSCAN clustering on UMAP space 

	clus_dens = DBSCAN(eps = 0.5).fit(embedding)


	# Get cluster labels 
	cluster_labels = clus_dens.labels_


	# Make a dataframe for visualization 
	plot_df = pd.DataFrame(
	    np.concatenate([embedding, cluster_labels.reshape(-1,1)], axis = 1),
	    columns = ['UMAP_1', 'UMAP_2', 'cluster_labels'])


	return rand_ixs, plot_df






# Apply rolling mean to each row 
processed_neurons = np.apply_along_axis(
    func1d=rolling_mean, axis=0, arr = sample_neurons
)

## Apply PCA to processed neurons
pcs = iPCA(n_components = 10).fit_transform(processed_neurons)


# Apply UMAP to neurons' principal components
embedding = umap(
	min_dist =0, 
	n_neighbors=20,
	n_components = 2, 
	random_state = 34).fit_transform(pcs)

# Apply clustering 

# clus = SpectralClustering(
# 	n_clusters = 8,	
# 	affinity = 'nearest_neighbors').fit(embedding)

clus_dens = DBSCAN(eps = 0.5).fit(embedding)

# Get cluster labels 
cluster_labels = clus_dens.labels_


# Make a dataframe for visualization 
plot_df = pd.DataFrame(
    np.concatenate([embedding, cluster_labels.reshape(-1,1)], axis = 1),
    columns = ['UMAP_1', 'UMAP_2', 'cluster_labels'])






def compressive_sensing_1d_cvxpy(signal, subsample_proportion): 

	"""
	
	Apply compressive sensing to 1D signal using cosine basis. 

	Params
	--------
	corrupted_signal (array-like)
		Numpy array containing the corrupted signal. This signal must constitute
		a random subsample of the original signal. 

	"""

	n = signal.shape[0]
	
	# Generate random sample of indices by sampling without remplacement
	rand_ixs = np.random.choice(n, subsample_size, replace=False) 

	# Initialize subsample size 
	subsample_size = int(n * subsample_proportion) # 10% sample

	# Create cosine transform operator
	A = spfft.idct(np.identity(n), norm='ortho', axis=0)

	# Make sampled operator A= ψϕ s.t. A.shape = [subsample_size, n]
	A_ = A[rand_ixs]

	# Initialize cvxpy variable
	vx = cvx.Variable(n)

	# Select random indices from signal 
	y = signal[rand_ixs]

	# Set optimization objective : L1-norm
	objective = cvx.Minimize(cvx.norm(vx, 1))

	# Set constrains for linear regression problem 
	constraints = [A*vx == y]


	# Initialize cvxpy problem
	prob = cvx.Problem(objective, constraints)
	result = prob.solve(verbose=True)

	# Get reconstructed signal as numpy array  
	x = np.array(vx.value)

	# Convert reconstructed signal from frequency to time domain
	reconstructed = spfft.idct(x, norm="ortho", axis=0)

	return reconstructed 


def compressive_sensing_1d_sklearn(signal, subsample_proportion, alpha= 0.001): 

	"""
	Compressive sensing of 1D signal using sci-kit learn. 
	This function uses cosine basis. 

	"""

	# Get signal shape 
	n = signal.shape[0]

	# Initialize subsample size 
	subsample_size = int(n * subsample_proportion)
	
	# Generate random sample of indices by sampling without remplacement
	rand_ixs = np.random.choice(n, subsample_size, replace=False) 

	# Initialize subsample size 
	subsample_size = int(n * subsample_proportion) # 10% sample
	
	# Initialize cosine transform operator
	A = spfft.idct(np.identity(n), norm='ortho', axis=0)

	# Make sampled operator A= ψϕ s.t. A.shape = [subsample_size, n]
	A_ = A[rand_ixs]

	# Get subsample of signal 
	y = signal[rand_ixs]

	# Initialize and fit linear regression with L1 norm
	lasso = Lasso(alpha = alpha)

	lasso.fit(A, y)

	# Get fourier coefficients
	fourier_coef = lasso.coef_

	# Convert back to time domain
	reconstructed_sklearn = spfft.idct(fourier_recon, norm='ortho', axis=0)




	
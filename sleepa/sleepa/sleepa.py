# -*- coding: utf-8 -*-
# sleep_neurons

# Streaming packages 
import toolz as tz 
import toolz.curried as c

# Numerical and data wrangling libraries 
from numba import njit 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import networkx as nx
import scipy.io as sio
from scipy.signal import find_peaks
import seaborn as sns
import numpy as np 
import h5py

# Signal processing libs
import scipy.fftpack as spfft
import pywt
import cvxpy as cvx


# Interactive plotting 

import bokeh 
from bokeh.plotting import figure
import holoviews as hv 
import bebi103
from bebi103.viz import fill_between
import panel as pn
import colorcet as cc 


# Unsupervised learning stack 
from umap import UMAP as umap 
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import IncrementalPCA #as iPCA
from sklearn.decomposition import PCA 
from sklearn.linear_model import Lasso 
from sklearn.mixture import BayesianGaussianMixture as bGMM
from sklearn.mixture import GaussianMixture as GMM



@njit
def rolling_mean_numba(x, window_size= 30):
    cumsum = np.cumsum(x)
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


#def get_all_datasets(download_dir = '../data/'):
	#"""Helper function to download the raw data from the Janelia FigShare repo. """

	# Download data from figshare 
    # Uncompress 

def load_ahrens_data_figshare(path):

    """
    Loads data from the FigShare repository. 
    The data comes from the Neuron 2018 paper.

    """

    # Load data 
    dat = h5py.File(path, 'r')
   

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



def cluster_neuron_data(neuron_data, sample_size, window_size,
	pca_components= 20, n_neighbors= 30, **kwargs):

	"""
	Wrapper for first version of the pipeline . 
	"""

	rand_ixs, neuron_samples = subsample_data(neuron_data, sample_size = sample_size)

	# Apply rolling mean to each row 
	processed_neurons = np.apply_along_axis(
    	func1d=rolling_mean_numba,
    	axis=1,
    	arr = neuron_samples,
    	**{'window_size':window_size}
	)

	# Normalize data to be standard normal, i.e., with mean = 0, var = 1
	scaler = StandardScaler()
	normalized_neurons = scaler.fit_transform(processed_neurons)

	# Apply PCA to processed neurons
	pcs = PCA(n_components = pca_components).fit_transform(normalized_neurons)


	# Apply UMAP to neurons' principal components
	embedding = umap(
	    min_dist =0, 
	    n_neighbors=n_neighbors,
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




def get_bayesian_information_criterion(max_clusters, data):
	
	"""
	Returns the bayesian information criterion for a number of Gaussian Mixture models.
	This is aimed to choose the number of clusters for a given dataset. 

	The number of clusters that minimizes the Bayesian information criterion, maximizes 
	the likelihood of the model best explaining the dataset. 

	Params
	--------

	max_clusters(int)
		Maximum number of clusters to run against. 

	data (array-like or pd.DataFrame)
		Dataset (n_samples, n_variables) to be clustered


	Returns
	--------

	bic(list)
		Bayesian information criterion score for each model. 

	"""

	# Initialize array for the number of clusters
	n_components = np.arange(1, max_clusters)

	# Run a GMM model for each of the number of components
	models = [GMM(n, covariance_type='full', random_state=0).fit(data)
	          for n in n_components]

	# Extract the Schwarz (bayesian) information criterion for each model
	bic = [m.bic(data) for m in models]

	return bic 




def cluster_neuron_data_v2(neuron_data, pca_components= 0.6, n_neighbors= 30, max_clus = 20, **kwargs):

	"""
	Wrapper for version 2 of the pipeline: GMM clustering  
	"""


	# Normalize data to be standard normal, i.e., with mean = 0, var = 1
	scaler = StandardScaler()
	normalized_neurons = scaler.fit_transform(neuron_data)

	# Apply PCA to processed neurons
	pcs = PCA(n_components = pca_components).fit_transform(normalized_neurons)

	three_pcs = pcs[:, :3]

	# Apply UMAP to neurons' principal components
	embedding = umap(
	    min_dist =0, 
	    n_neighbors=n_neighbors,
	    n_components = 2, 
	    random_state = 3287354).fit_transform(pcs)

	
	# Compute Shwarz information criterion
	bic = get_bayesian_information_criterion(max_clus, embedding)
	
	# Get optimal number of clusters
	n_clusters = np.argmin(bic)

	# Initialize Gaussian Mixture model 
	clustering_model = bGMM(n_clusters).fit(embedding)

	# Get cluster labels 
	cluster_labels = clustering_model.predict(embedding)


	# Make a dataframe for visualization 
	plot_df = pd.DataFrame(
	    np.concatenate([embedding, three_pcs, cluster_labels.reshape(-1,1)], axis = 1),
	    columns = ['UMAP_1', 'UMAP_2', 'PCA_1', 'PC_2', 'PC_3', 'GMM_labels'])


	return plot_df


#def compressive_sensing_reconstruction_1d():


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

	return reconstructed_sklearn


def dct2(x):
    
    """Two-dimensional discrete cosine transform wrapper."""

    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    
    """Two-dimensional inverse consine transform. """
    
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def get_scaleogram(signal, time, scales, waveletname = 'cmor'):
    
    """
    Returns a 2D scaleogram. 
	

    """
    
    # Get the inter sampling time 
    dt = time[1] - time[0]

    # Compute wavelet transform 
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)

    # Get the log power of signal 
    log_power = np.log((abs(coefficients)) ** 2)
    
    
    return log_power




def get_wavelet_transform(signal, time, scales, waveletname = 'gaus5'):
    
    """
    Returns flattened array of scaleogram of a 1D signal.
    Helper function to apply to a matrix where each row or column 
    represents a signal or timeseries. 
    
    Params
    --------

    signal (array-like)
    	Array containing 1D signal. 
    
    time (array-like)
    	Array of time corresponding to the time at which the signal
    	was sampled. 

    scales (array-like)
    	Pseudo-frequencies, they have units of [signal units (SU) / freq].
    	A high scale factor corresponds to smaller frequencies. 

    waveletname (str, default = 'gaus5')
    	Wavelet to perform the transform. 
    
    Returns
    --------

    wavelet_transform (array-like)
    	Flattened array of the log power of the wavelet transform. The log power is
    	a matrix with shape (n_scales, n_timepoints), thus the flattened version
    	will have (n_scales * n_timepoints, 1) dimensions. 
    
    """
    
    # Get the inter sampling time 
    dt = time[1] - time[0]

    # Compute wavelet transform 
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    
    # Get log power
    log_power = np.log((abs(coefficients)) ** 2)
    
    # Flatten array 
    wavelet_transform = log_power.flatten()
    
    return wavelet_transform




def get_wavelet_basis(neuron_data, time, scales, wavelet = 'gaus5'):
    
    """
    Compute the wavelet transform over the complete whole brain dataset. 
    
    Params 
    --------
    
    neuron_data (array-like)

	time (array-like)
		Array of time corresponding to the time at which the signal
    	was sampled. 

	scales (array-like)
    	Pseudo-frequencies, they have units of [signal units (SU) / freq].
    	A high scale factor corresponds to smaller frequencies.

	wavelet (str, default = 'gaus5')


    Returns
    --------
    
    """
    
    # Apply wavelet transform to each row 
    wavelet_basis = np.apply_along_axis(
        func1d= get_wavelet_transform,
        axis=1,
        arr = neuron_data,
        **{
            'time':time,
            'scales': scales,
            'waveletname' : wavelet
        }
    )
    
    return wavelet_basis




def get_knn_graph(data, n_neighbors = 30, **kwargs): 
    
    """
    Wrapper function to generate a kNN graph using sklearn and NetworkX. 
    
    """
    # Initialize Nearest Neighbors graph learning
    knn = NearestNeighbors(n_neighbors,**kwargs).fit(data)
    
    # Get adjacency matrix in sparse format
    knn_adjacency_mat = knn.kneighbors_graph(data)
    
    # Convert to nx.Graph object
    knn_graph = nx.from_scipy_sparse_matrix(knn_adjacency_mat)
    
    return knn_graph



def get_graph2vec_from_knn(list_of_data, n_neighbors, knn_kwargs={}, graph2vec_kwargs={}): 
    
    """
    Wrapper function to make a graph2vec embedding directly from a list of datasets. 
    
    Params 
    ---------
    list_of_data(array-like)
        List of datasets. Datasets could be pd.Dataframes or Numpy arrays with numerical 
        values only. 
    
    """
    
    # Generate a list of knn graphs 
    graph_list = [
        get_knn_graph(data, n_neighbors, **knn_kwargs)
    ]
    
    # Initialize and fit model
    if 'dimensions' not in graph2vec_kwargs: 
        graph2vec_kwargs['dimensions'] = 20
    
    # Initialize and fit graph2vec model to networks
    model = Graph2Vec(**graph2vec_kwargs)
    model.fit(graph_list)
    
    # Get the vector embeddings
    vector_embedding = model.get_embedding()
    
    return vector_embedding


def get_z_slice(path_to_mat, fish_number, data_path , slice_ix = 13, return_df = False):

	"""

	Params
	--------

	path_to_mat(str)
		Path to the .mat file containing the anatomical data. 

	fish_number (int)
		Number of the studied fish according to the Chen et al. notation. 

	data_path (str)
		Path to store the dataset. 

	slice_ix(int)
		Index of the slice to get. 

	return_df (bool)
		If set to True it will return the dataframe. 

	"""

	# Load mat file 
	dat = sio.loadmat(path_to_mat)

	# Extract fish voxels 
	fish_voxels = dat['data']['anat_stack'][0][0]

	# Get z_slice
	z_slice = fish_voxels[:, :, slice_ix]

	# Convert to dataframe
	df_z_slice = pd.DataFrame(z_slice)

	# Export dataframe 
	df_z_slice.to_csv(data_path + 'z_slice' + '_'+ str(fish_number) + '.csv', index = False )

	if return_df == True:
		return df_z_slice

	else: 
		pass 




def get_xy_coordinates(path_to_mat, fish_number, data_path, return_df = False):
	
	"""
	Exports the XYZ coordinates of valid neurons. 

	Params 
	---------

	path_to_mat(str)
		Path to the .mat file containing the anatomical data. 

	fish_number(int)
		Number of the fish to retrieve data from.

	data_path (str)
		Path to the datasets folder to export the df. 

	return_df (bool)
		If True, returns the dataframe. 

	Returns 
	---------

	df_xyz (pd.DataFrame)
		XY coordinates for the fish analyzed. 

	"""

	# Load mat file 
	dat = sio.loadmat(path_to_mat)

	# Get xyz coords 
	xyz_coordinates = dat['data']['CellXYZ'][0][0]

	# Get invalid ixs
	invalid_ixs = dat['data']['IX_inval_anat'][0][0].flatten()

	# Get number of neurons in data
	n_neurons= xyz_coordinates.shape[0] # in xyz_coords

	# Initialize boolean mask 
	mask = np.ones(n_neurons, dtype = bool)

	# Set to False on invalid ixs
	mask[invalid_ixs] = False

	# Get valid neurons
	xyz = xyz_coordinates[mask, :2]

	# Make dataframe
	df_xyz = pd.DataFrame(xyz, columns = ['X', 'Y'])

	# Export df 
	df_xyz.to_csv(data_path + 'xyz_coords' +'_' + str(fish_number) + '.csv', index = False)


	if return_df == True:
		return df_xyz

	else : 
		pass


def individual_trace_plot(clus_num, clus_neurons, time_arr, max_ix, fill_color, line_color, **kwargs):

	"""
	Wrapper function to make a single cluster trace plot. It makes use of the 
	bebi103.viz.fill_between function. 
	
	Params
	--------
	clus_num (int)
		Number of the cluster to be plotted. This is used for the title of the plot only.  
	
	clus_neurons (array-like or pd.DataFrame) 
		Cluster of neurons. It has a shape of (n_neurons, n_timepoints) unless otherwise stated.
		This could also work in the frequency domain or other representation of the data. 
	
	time_arr (array-like)
		Time array to be plotted. It should have a shape of (1, n_timepoints). It is intended 
		to be in seconds. If it is not, it should be explicitly specified using the x_axis_label 
		keyword argument. 
	
	max_ix (int)
		Maximum index to visualize the traceplots. 
	
	fill_color (str)
		RGB color for the percentile shading, as a hex string. We recommend to use the http://colorbrewer2.org/ website or 
		the Colorcet library https://colorcet.holoviz.org/user_guide/index.html. 
	
	line_color (str)
		RGB color for the mean trace line.


	**kwargs
		Keyword arguments supplied to the bokeh.figure. 


	Returns 
	--------

	fig (bokeh.figure)
		Trace lineplot for a cluster of neurons. 

	"""


def make_cluster_trace_plot(clus_num, clus_neurons, time_arr, max_ix, fill_color, line_color, n_samples = 10, plot_summary = False, **kwargs):

	"""
	Wrapper function to make a single cluster trace plot. It makes use of the 
	bebi103.viz.fill_between function. 
	
	Params
	--------
	clus_num (int)
		Number of the cluster to be plotted. This is used for the title of the plot only.  
	
	clus_neurons (array-like or pd.DataFrame) 
		Cluster of neurons. It has a shape of (n_neurons, n_timepoints) unless otherwise stated.
		This could also work in the frequency domain or other representation of the data. 
	
	time_arr (array-like)
		Time array to be plotted. It should have a shape of (1, n_timepoints). It is intended 
		to be in seconds. If it is not, it should be explicitly specified using the x_axis_label 
		keyword argument. 
	
	max_ix (int)
		Maximum index to visualize the traceplots. 
	
	fill_color (str)
		RGB color for the percentile shading, as a hex string. We recommend to use the http://colorbrewer2.org/ website or 
		the Colorcet library https://colorcet.holoviz.org/user_guide/index.html. 
	
	line_color (str)
		RGB color for the mean trace line.

	n_samples(int)
		Number of traces to plot 


	**kwargs
		Keyword arguments supplied to the bokeh.figure. 


	Returns 
	--------

	fig (bokeh.figure)
		Trace lineplot for a cluster of neurons. 

	"""

	# Compute 95th percentiles of cluster neurons
	ptiles = np.percentile(clus_neurons, [2.5, 97.5], axis = 0)

	# Compute mean neuron 
	mean_neuron = np.mean(clus_neurons, axis = 0)

	# Get minimum and maximum value for setting axes ranges
	y_min = ptiles[0].min()
	y_max = ptiles[1].max()

	if 'plot_width' not in kwargs: 
		kwargs['plot_width'] = 600

	if 'plot_height' not in kwargs: 
		kwargs['plot_height'] = 300

	if 'title' not in kwargs:
		kwargs['title'] = 'cluster ' + str(clus_num)

	if 'x_axis_label' not in kwargs:
		x_axis_label="time (seconds)"


	# Initialize bokeh figure
	p = figure(
	    y_range=(y_min - 0.1, y_max + 0.05),
	    y_axis_label = 'activity',
	    **kwargs	    
	)

	# Make plot for a limited index value across time 
	if max_ix is not None: 
		
		# Plot mean trace 
		p.line(x=time_arr[:max_ix], y=mean_neuron[:max_ix], line_width=3, line_color = line_color)

		# Plot percentiles using Bebi103 function 
		fig = fill_between(
		    x1 = time_arr[:max_ix] , #
		    y1 = ptiles[0, :max_ix],#
		    x2 = time_arr[:max_ix], 
		    y2 = ptiles[1, :max_ix],
		    patch_kwargs = {'fill_alpha' : 0.4, 'fill_color': fill_color},
		    line_kwargs = {'line_color': line_color },
		    p = p 
		)
		
	# Make plot for all of the timepoints
	else: 

		# Plot mean line 
		p.line(x=time_arr, y=mean_neuron, line_width=3, line_color = line_color)

		# Plot percentiles 
		fig = bebi103.viz.fill_between(
		    x1 = time_arr , #
		    y1 = ptiles[0],#
		    x2 = time_arr, 
		    y2 = ptiles[1],
		    patch_kwargs = {'fill_alpha' : 0.4, 'fill_color': fill_color},
		    line_kwargs = {'line_color': fill_color},
		    p = p 
		)

	return fig 



def trace_plot_all_clusters(processed_neurons, clus_df, time_arr, label_col,  ix_col, color_palette = None,
	max_ix= None, clus_num_list = None, **kwargs):
	
	"""
	Wrapper function to make a list of bokeh figures containing the distribution of traces
	for visualization of clustering results. 
	
	Params
	-------

	processed_neurons(array-like)
		np.array with shape (n_neurons, n_timepoints) after the data smoothing or transformation. 

	clus_df (pd.DataFrame)
		Pandas df containing the cluster labels for each neuron. The indices of this dataframe
		must correspond with the processed neurons. 

	time_arr(array-like)
		Time array to be plotted. It should have a shape of (1, n_timepoints). It is intended 
		to be in seconds. If it is not, it should be explicitly specified using the x_axis_label 
		keyword argument.

	label_col(str) 
		Name of the column which contains the cluster labels. 

	ix_col(str) 
		As of now, the this parameter is intended to keep track of the original indices explicitly. 
		In a future version this could be ignored and set from the main clustering function. 
	
	color_palette (array-like)
		List of RGB colors in HEX format. We recommend using the palettes of colorcet https://colorcet.holoviz.org/user_guide/index.html. 

	max_ix (int, default= None)
		Maximum index to be considered for the plot. Default is to use all timepoints. 
	
	clus_num_list (array-like, default= None)
		List of number of clusters. If not supplied, inferred from the clus_df. 
	
	**kwargs
		Keyword arguments supplied to the bokeh.figure. 

	Returns 
	--------

	fig_list (list of bokeh.figures)
		List of plots ready to be visualized with bokeh.layouts
	
	"""

	# Extract the number of clusters 
	if clus_num_list is None: 
		try: 
			clus_num_list = clus_df[label_col].unique()
		#else: 
		#	clus_num_list = np.arange(clus_df.label_col.astype(int))
		except: 
			print('Cluster number list was not supplied and could not be inferred. ')


	if max_ix == None: 
		max_ix = len(time_arr)
	else:
		pass

	if color_palette == None:
		color_palette = cc.glasbey_cool


	# Initialize list of figures. 
	fig_list = []

	# Make clusterPlot for each cluster
	for clus in clus_num_list: 

		# Extract cluster data 
		clus_ixs_ = clus_df[clus_df[label_col] == clus][ix_col].values

		# Extract data for given cluster with indices
		clus_data_ = processed_neurons[clus_ixs_]

		# Get cluster traceplot 

		trace_clus= make_cluster_trace_plot(
			clus,
			clus_data_,
			time_arr, 
			max_ix, 
			fill_color = color_palette[clus], 
			line_color= color_palette[clus],
			**kwargs)

		fig_list.append(trace_clus)

	return fig_list




def brain_cluster_plot_plt(z_slice, xy_coords, clus_number, cluster_labels, color): 
    """
    Returns a scatterplot of the neurons in a cluster on top of
    a z-slice of an image the zebrafish brain. 
        
    Params 
    -----

    z_slice (array-like)
    	Matrix as numpy array

    xy_coords (array-like)

    clus_number(int)
        Number of cluster to plot. 

    cluster_labels(array-like)
        Numpy array containing the cluster label for each neuron.

    color (str)
        Color in HEX format. 
    
    Returns
    --------
    
    fig (matplotlib.figure.Figure)
        Overlay of scatterplot of neurons on image of the brain.
    
    """

    # Initialize figure 
    fig, ax = plt.subplots()
    
    # Plot slice of z-axis of the brain
    ax.imshow(z_slice, cmap = 'bone')

    # Make x and y arrays 
    neurons_x = xy_coords[:, 1]
    neurons_y = xy_coords[:, 0]
    
    # Plot neurons scatterplot 
    ax.scatter(
    	neurons_x[cluster_labels == clus_number],
    	neurons_y[cluster_labels == clus_number],
    	s = 0.15,
        alpha = 0.08,
        color = color,
        label = 'cluster ' + str(clus_number)
	)
    
    return fig


def brain_cluster_plot_bokeh(z_slice, xy_coords, clus_number, cluster_labels, color):
    
    """
    Returns a scatterplot of the neurons in a cluster on top of
    a z-slice of an image the zebrafish brain. 
        
    Params 
    -----
    clus_number(int)
        Number of cluster to plot. 

    cluster_labels(array-like)
        List containing the cluster label for each neuron.

    color (str)
        Color in HEX format. 
    
    Returns
    --------
    
    p (bokeh.figure)
        Overlay of scatterplot of neurons on interactive plot
        of the brain.
    """
    
    # Initialize z_slice plot 
    p = bebi103.image.imshow(z_slice, title = 'cluster ' + str(clus_number), cmap = cc.gray)
    
    # Get x and y coordinates
    neurons_x = xy_coords[:, 1]
    neurons_y = xy_coords[:, 0]
    
    # Make scatter plot 
    p.scatter(
        neurons_x[cluster_labels == clus_number],
        neurons_y[cluster_labels == clus_number],
        color = color, 
        size = 1.8, 
        alpha = 0.4
    )
    
    return p




def get_all_brain_clus_bokeh(
    z_slice, xy_coords, cluster_labels, max_clusters, palette, **kwargs
):

    """
    Wrapper function to plot all clusters with bokeh. Works (best) in jupyter notebooks. 

    Warning: very computationally intensive in terms of graphical processing. 
    
    Params
    ---------
    cluster_labels (array-like or pd.Series)
        Cluster labels for each neuron. 
    
    palette (array-like)
        List of HEX colors. 
    
    Returns
    ---------
    grid (bokeh.models.layouts.Column) 
        Grid of plots. 
    """

    #  Initialize plot list
    plot_list = []

    #  Add individual cluster plot to list
    for ix in np.arange(max_clusters):
    
        plot_list.append(
            brain_cluster_plot_bokeh(
            z_slice,
            xy_coords,
            clus_number=ix,
            color=palette[ix], 
            cluster_labels=cluster_labels
            )
        )
    
    # Generate gridplot 
    grid = bokeh.layouts.gridplot(plot_list, ncols= int(max_clusters //2))

    return grid



def load_bright_palette():

	"Load custom bright palette."

	# Initialize custom palette 
	bright_palette = [
	    '#8dd3c7',
	    '#762a83', 
	    "#1debf2",
	    "#fca903",
	    "#3df514",
	    "#a41df2",
	    "#4b00bd",
	    "#9aba9e",
	    "#95fff2",
	    '#e01df2',
	    "#14f5f5",
	    '#fccde5',
	    '#ffffb3'
	]

	return bright_palette

def get_style_bokeh():
    
    '''
    Formats bokeh plotting envir∂oment similar 
    to that used in Physical Biology of the Cell, 2nd edition.
    Based on @gchure and @mrazomej's work.

    '''

    theme_json = {'attrs':
            {'Figure': {
                'background_fill_color': '#ffffff',
                'outline_line_color': '#000000',
            },
            'Axis': {
            'axis_line_color': "white",
            'major_tick_in': 7,
            'major_tick_line_width': 2,
            'major_tick_line_color': "white",
            'minor_tick_line_color': "grey",
            'axis_label_text_font': 'Helvetica Neue',
            'axis_label_text_font_style': 'normal'
            },
            'Grid': {
                'grid_line_color': 'white',
            },
            'Legend': {
                'background_fill_color': '#E3DCD0',
                'border_line_color': '#FFFFFF',
                'border_line_width': 1.5,
                'background_fill_alpha': 0.5
            },
            'Text': {
                'text_font_style': 'normal',
               'text_font': 'Helvetica'
            },
            'Title': {
                'background_fill_color': '#FFEDC0',
                'text_font_style': 'normal',
                'align': 'center',
                'text_font': 'Helvetica Neue',
                'offset': 2,
            }}}

    return theme_json


def set_plotting_style():
      
    """
    Plotting style parameters, based on the RP group. 
    """    
        
    tw = 1.5

    rc = {'lines.linewidth': 2,
        'axes.labelsize': 18,
        'axes.titlesize': 21,
        'xtick.major' : 16,
        'ytick.major' : 16,
        'xtick.major.width': tw,
        'xtick.minor.width': tw,
        'ytick.major.width': tw,
        'ytick.minor.width': tw,
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large',
        'font.family': 'sans',
        'weight':'bold',
        'grid.linestyle': ':',
        'grid.linewidth': 1.5,
        'grid.color': '#ffffff',
        'mathtext.fontset': 'stixsans',
        'mathtext.sf': 'fantasy',
        'legend.frameon': True,
        'legend.fontsize': 12, 
       "xtick.direction": "in","ytick.direction": "in"}



    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)

    #sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)




def make_slider(): 

	"""
	Integer slider for z_slice function. 
	"""
	slider = pn.widgets.IntSlider(start= 0, end = 29, step = 1, value = 10 )
	
	return slider 


#@pn.depends(slider.param.value)

# def plot_zslice(slider): 
	
# 	"""
# 	Helper function to generate a view of the fish anatomy across the z-axis
# 	with a slider. 
# 	"""

#     zslide = fish_voxels[:, :, slider]

#     im = rasterize(hv.Image(zslide).opts(cmap = 'viridis')).opts(width = 400, height = 600)

#     return im




#
#def streaming_wavelet_transform(): 


	"""
	Performs wavelet transform on the whole brain neuron matrix. 

	"""

	# 



@njit
def corr_numba(x1, x2):
    """
    Returns Pearson correlation coefficient in numba version
    """
    return np.corrcoef(x1, x2)[0,1]


def get_rest_regressor(behavior_arr, time_arr, min_dt, n_timepoints, thresh = 0.1, channel = None, mat_file = None):
	
	
	"""
	Compute the rest regressor defined as an indicator random variable 
	that is set to 1 when the time between activity spikes above 
	a given threshold. 

	Params
	------
	behavior_arr(array-like)
		1D array  corresponding to the activity trace of a given channel in the 
		activity matrix. 

	min_dt (int or float)
		Time threshold of inactivity. 

	n_timepoints (int) 
		Number of recorded data points. 

	thresh (float)
		Activity height threshold to determine a spike as activity. 

	channel (int, default None)
		If not None, set the desired channel to analyze. 

	mat_file (array-like)
		Raw mat file containing the behavioral data. 

	Returns 
	-------
	rest_regressor(array-like)
		Binary indicator random variable to a lack of activity under a given
		threshold.

	"""

	# if channel is not None and mat_file is not None: 

	# 	#
	# else: 
	# 	pass


	# Use scipy.signal to get peaks in behavior array. 
	ixs, peaks = find_peaks(behavior_arr, height = thresh)

	peak_heights = peaks['peak_heights']

	rest_regressor = np.zeros(n_timepoints)


	for ix_peak in np.arange(len(ixs)): 

		if time_arr[ixs[ix_peak]] - time_arr[ixs[ix_peak -1]]  > min_dt: 
			rest_regressor[ixs[ix_peak - 1 ]: ixs[ix_peak]] = 1
		else: 
			pass 

	return rest_regressor



#~~~~MMM~~~~~curry...

@tz.curry 
def curried_wavelet_transform(signal, scales, waveletname, dt):
	"""Helper curry function to compute wavelet transform."""
	return pywt.cwt(signal, scales, waveletname, dt)


@tz.curry
def get_wavelet_transform(signal, time, scales, waveletname = 'gaus5'):
    
    """
    Returns flattened array of scaleogram of a 1D signal.
    Helper function to apply to a matrix where each row or column 
    represents a signal or timeseries. 
    
    Params
    --------

    signal (array-like)
    	Array containing 1D signal. 
    
    time (array-like)
    	Array of time corresponding to the time at which the signal
    	was sampled. 

    scales (array-like)
    	Pseudo-frequencies, they have units of [signal units (SU) / freq].
    	A high scale factor corresponds to smaller frequencies. 

    waveletname (str, default = 'gaus5')
    	Wavelet to perform the transform. 
    
    Returns
    --------

    wavelet_transform (array-like)
    	Flattened array of the log power of the wavelet transform. The log power is
    	a matrix with shape (n_scales, n_timepoints), thus the flattened version
    	will have (n_scales * n_timepoints, 1) dimensions. 
    
    """
    
    # Get the inter sampling time 
    dt = time[1] - time[0]

    # Compute wavelet transform 
    [coefficients, frequencies] = curried_wavelet_transform(signal, scales, waveletname, dt)
    
    # Get log power
    log_power = np.log((abs(coefficients)) ** 2)
    
    # Flatten array 
    wavelet_transform = log_power.flatten()
    
    return wavelet_transform


def array_from_txt(line, sep='\t', dtype=np.float):

    """Helper function to stream a csv file. """

    return np.array(line.rstrip().split(sep), dtype=dtype)


def tsv_line_to_array(line):
    lst = [float(elem) for elem in line.rstrip().split('\t')]
    return np.array(lst)

def readtsv(filename):
    print('starting readtsv')
    with open(filename) as fin:
        for i, line in enumerate(fin):
            #print(f'reading line {i}')
            yield tsv_line_to_array(line)
    print('finished readtsv')


def streaming_pca_(data, n_components = 20, batch_size = 100):
    
    """
    Helper function to initialize streaming Incremental PCA. 

    Params 
    --------

    data (iterator)
        Csv file to be streamed. It assumes it does not have a header. 
        Be sure to remove header a priori. 

    n_components (int, default = 2)
        Number of principal components to extract. 

    batch_size (int, default = 100)
        Size of batches to make partial fits. 


    Returns
    ---------

    ipca_ (IncrementalPCA object)
    	Mini-batch-trained PCA object. 

    """

    ipca_ = IncrementalPCA(
        n_components=n_components,
        batch_size=batch_size
    )
    
    print('Starting minibatch training.')
    
    tz.pipe(
        data,  # iterator of 1D arrays
        c.partition(batch_size),  # iterator of tuples (tuples of sets (of size = batch size) of 1-D arrays)
        c.map(np.array),  # iterator of 2D arrays : for each batch, makes a matrix from the stream 
        c.map(ipca_.partial_fit),  # partial_fit on each batch 
        tz.last # Suck the stream of data through the pipeline
    ) 
    

    return ipca_


def wavelet_space_train(fname, time_arr, scales ): 

	"""
	Params 
	------

	fname (str)

		Path to file and file name of desired fish to analize. 
		Remember the dataset must be a plain tab-separated txt
		in order to be analized.

	time_arr ()

	scales (array-like)


	Returns
	-------
	

	"""


	with open(fname) as file: 

		ipca = tz.pipe(
			file, 
			c.map(array_from_txt), 
			c.map(get_wavelet_transform(time = time_arr, scales = scales)),
			streaming_pca_
		)

	return ipca 

def wavelet_transform(ipca, fname, time_arr, scales): 

	with open(fname) as file: 

		projected = tz.pipe(
			file, 
			c.map(array_from_txt), 
			c.map(get_wavelet_transform(time = time_arr, scales = scales)), 
			c.map(reshape(newshape = (1, -1))), 
			c.map(ipca.transform), 
			np.vstack

		)

	return projected 


# @tz.curry 
# class wavelet_pca_streamer: 


# 	def __init__(self, data, n_components, batch_size): 

# 		self.iPCA = IncrementalPCA(
# 	        n_components=n_components,
# 	        batch_size=batch_size
# 	    )

	   

# 		tz.pipe(
# 	        data,  # iterator of 1D arrays
# 	        c.partition(batch_size),  # iterator of tuples (tuples of sets (of size = batch size) of 1-D arrays)
# 	        c.map(np.array),  # iterator of 2D arrays : for each batch, makes a matrix from the stream 
# 	        c.map(self.pca.partial_fit),  # partial_fit on each batch 
# 	        tz.last # Suck the stream of data through the pipeline
# 	    ) 
	    


# 	def train(self, time_arr, scales): 

# 		with open(fname) as file: 

# 		ipca = tz.pipe(
# 			file, 
# 			c.map(array_from_txt), 
# 			c.map(get_wavelet_transform(time = time_arr, scales = scales))
# 			self.iPCA
# 		)

# 	return self.iPCA 


# 	def transform(self, data):

# 		return ipca 







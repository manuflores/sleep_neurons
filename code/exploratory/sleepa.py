# -*- coding: utf-8 -*-
# sleep_neurons


from umap import UMAP as umap 
from numba import njit 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
import numpy as np 
import h5py
from bokeh.plotting import figure
import colorcet as cc 
import bokeh 
import holoviews as hv 
from bebi103.viz import fill_between
import networkx as nx


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import IncrementalPCA as iPCA
from sklearn.decomposition import PCA 
from sklearn.linear_model import Lasso 
from sklearn.mixture import BayesianGaussianMixture as bGMM
from sklearn.mixture import GaussianMixture as GMM


import scipy.fftpack as spfft
import cvxpy as cvx


@njit
def rolling_mean_numba(x, window_size= 30):
    cumsum = np.cumsum(x)
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def load_ahrens_data_figshare(path, 
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



# def visualize_cluster_traces_bokeh():

# def visualize_cluster_traces_mpl():


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


def get_knn_graph(data, n_neighbors = 30, **kwargs): 
    """
    Wrapper function to generate a kNN graph using sklearn and networkx. 
    
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


def make_cluster_trace_plot(clus_num, clus_neurons, time_arr, max_ix, fill_color, line_color, **kwargs):

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


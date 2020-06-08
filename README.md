## Unsupervised learning of sleep-active neurons :microscope: :fish:


Repository to analyze neuron activity data from whole-brain calcium imaging in zebrafish. Focused on the discovery of sleep-active neurons. This work is ongoing research of the Prober Lab at Caltech. For more information please contact Manu ([manuflores@caltech.edu](mailto:manuflores@caltech.edu)) or Andrey ([aandrev@caltech.edu](mailto:aandrev@caltech.edu))


## Data 

You can download all of the datasets from the [Chen *et al.* Neuron 2018 paper](https://www.cell.com/neuron/pdf/S0896-6273(18)30844-4.pdf) by running :

`$ chmod a+x fetch_data.sh `

`$ ./fetch_data.sh `

on the command line. Alternatively, you can download a single dataset in order to run a basic tutorial for exploring the data using : 

`$ ./fetch_sample_dataset.sh`

The dataset used in this work can also be downloaded manually at: 

* Chen, Xiuye; Mu, Yu; Kuan, Aaron; Nikitchenko, Maxim; Randlett, Owen; Chen, Alex; et al. (2018): Whole-brain light-sheet imaging data. [https://doi.org/10.25378/janelia.7272617.v4](https://janelia.figshare.com/articles/Whole-brain_light-sheet_imaging_data/7272617/4)

We thank massively Dr. Chen for guidance in the use of this dataset. Moreover we acknowledge the amazing work done by the Ahrens lab in order to publish an open dataset of this quality and caliber. 


Some of the lessons learned after working with the dataset are the following: 

* Some of the behavior timeseries have more data points than the whole-brain imaging data. We recommend to trimming the last entries of the behavioral channels.
* To use the mask database one should use the `xyz_noorm` coordinates which are the standard coordinates based on [Z-brain] (https://engertlab.fas.harvard.edu/Z-Brain/).

## Code

The repo organization is as follows: 

* `code` 
  * `tutorials`: It contains a demo of the functionalities wrapped up in the `sleepa` package.
  * `exploratory`:  Documentation of the development of the functions in the package. 
  * `analysis`: Wrapper functions in order to perform our PCA-UMAP-GMM based clustering, as well as the more meaningful though computationally expensive wavelet transform based pipelines. 
  
* `sleepa`: Python module. It will be mainly aimed at unsupervised learning tasks such as dimensionality reduction and clustering. You can install the package by running the command : 

`$ pip install -e .`

in the root directory after cloning the repo. 


## License

The code in this repo is released under an MIT License. 

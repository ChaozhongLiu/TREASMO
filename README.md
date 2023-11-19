# TREASMO
[![License](https://img.shields.io/github/license/ChaozhongLiu/DyberPet.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)
![TREASMO Version](https://img.shields.io/badge/TREASMO-v1.1.1-green.svg)  
[![Documentation Status](https://readthedocs.org/projects/treasmo/badge/?version=latest)](https://treasmo.readthedocs.io/en/latest/?badge=latest)


**TREASMO** is a Transcription regulation analysis toolkit for single-cell multi-omics data. It quantifies the single-cell level gene-peak correlation strength, based on which a series of analysis and visualization functions are built to help researchers understand their multi-omics data.  

![](./Supplementary_Materials/Fig1.png)

## Installation
TREASMO is written in Python and is available from PyPI. Note that Python version should be 3.x.

### Install from PyPI
Please run the following command. Creating new environment using conda is recommended.
```
pip3 install treasmo

# For all the functions, please also install leidenalg and minisom
pip3 install leidenalg
pip3 install minisom
```
If the above method failed, please try create a new environment using conda first then re-install the three packages above.  
  
If things still doesn't work, try the method below.

### Use the package locally
If there is conflicts with other packages or something unexpected happened, try download the package and use it locally in a new conda environment.
```
# Create a new conda environment
conda create -n treasmo numpy=1.22
conda activate treasmo

# Setting up kernels for Jupyter Notebook if needed
conda install ipykernel
ipython kernel install --user --name=treasmo

# Install dependencies
conda install -c conda-forge scanpy python-igraph leidenalg
conda install -c conda-forge libpysal
conda install -c conda-forge hdf5plugin
conda install -c conda-forge muon
conda install -c conda-forge fa2
conda install -c conda-forge louvain

# Option 1 to install minisom
git clone https://github.com/JustGlowing/minisom.git
cd minisom
python setup.py install

# Option 2 to install minisom
pip3 install minisom

# install Homer if needed
```
  

  
## Quick start
- To test whether the package is working without error and go through important functions in TREASMO:
  - Download the toy example data via [Google Drive](https://drive.google.com/drive/folders/1pQY_Xj22KtizxYmxDzHekFFcGNMNCWa_?usp=sharing)
  - Follow this [Jupyter Notebook tutorial](example/quick_start.ipynb)  to get familiar with the package

- To prepare your own single-cell multiome data for TREASMO, see refer to Notebook [here](replication/1_Data_preprocessing.ipynb).


## Manuscript-related information
- For supplementary materials, please check the ``Supplementary_Materials`` folder
- For manuscript results replication, please check the ``replication`` folder


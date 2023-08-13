# TREASMO main results replication
This folder stores codes and related files to replicate the TREASMO paper results.

## 10X Multiome HSPC Data Availability
- Downnload the [H5 files](https://www.kaggle.com/competitions/open-problems-multimodal/data) into ``data/raw``


## Data Preprocessing
1. Run scripts in ``data/h52h5ad.ipynb/`` to get:
  - H5AD files of scRNA-seq and scATAC-seq data saved in ``data/h5ad``
2. Run ``1_Data_preprocessing.ipynb`` to get:
  - Processed scRNA-seq H5AD file in ``data/muon_out``
  - Processed scATAC-seq H5AD file in ``data/muon_out``
  - Combined and Processed multiome H5MU file in ``data/muon_out``



## Replication of all results
Please refer to the codes and Jupyter notebooks in this folder to replicate the results.
- 2_treasmo.ipynb:  
  - Generates all main results in TREASMO analysis pipeline
  - You can also refer to the raw results saved in ``Supplementary_Materials/``
  
  
- 3_functional_analysis,ipynb:  
  - Prepares and runs Homer motif enrichment analysis  
  - Sumarizes Homer motif enrichment results
  

- run_GO.R:
  - Runs gene ontology enrichment analysis for a given gene list
  

- plot_GO.R:
  - Produces plots in the manuscript
  
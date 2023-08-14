import numpy as np
import pandas as pd
import scanpy as sc
import random
import anndata as ad
import time

from treasmo.moran_vec import Moran
import treasmo.lee_vec as lee_vec
from libpysal.weights import W

import scipy as sp
from scipy import stats
#from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multitest import fdrcorrection

__author__ = "Chaozhong Liu"
__email__ = "czliubioinfo@gmail.com"



######
# add self-connection or not
# remove dropout: Default False
######



def Morans_I(mudata, mods=['rna','atac'], seed=1, max_RAM=16):
    """ 
    Function to calculate Moran's I for all the features in anndat_multiome

    Arguments
    --------------

        mudata: single-cell multi-omics data saved as MuData object

        mods: scRNA-seq and scATAC-seq modality name in MuData object

        seed: random seed to make the results reproducible

        max_RAM: maximum limitation of memory (Gb)


    Returns
    --------------
        mudata added with
            var['Morans.I']

    """

    # Check if mods exist in MuData
    if sum([i in mudata.mod.keys() for i in mods]) == len(mods):
        pass
    else:
        print("Not all mods provided are in MuData! Running stopped.")
        return

    # Start
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()

    # Construct connection matrix
    cong_mtx = mudata.obsp['connectivities'].toarray()
    #cong_mtx = cong_mtx/cong_mtx.sum(axis=1)[:,np.newaxis]
    neighbors = {}
    weights = {}

    for i in range(cong_mtx.shape[0]):
        neighbors[i] = np.nonzero(cong_mtx[i])[0].tolist()
        weights[i] = cong_mtx[i][np.nonzero(cong_mtx[i])[0]].tolist()
    w = W(neighbors, weights)

    mi = Moran(mudata.n_obs, w, permutations=0)
    mudata.var['Morans.I'] = np.NaN

    for mod in mods:
        mi.calc_i(mudata.mod[mod].X, seed=seed, max_RAM=max_RAM)
        mudata.mod[mod].var['Morans.I'] = mi.I

    mudata.var['Morans.I'] = np.concatenate([mudata.mod[mod].var['Morans.I'].to_numpy() for mod in mudata.mod.keys()])
    
    #print(mi.I)
    print("Finished calculating Moran's I. %.3fs past"%(time.time()-start))
    return mudata




def _check_array(mtx):
    if sp.sparse.issparse(mtx): #isinstance(mtx, sp.spmatrix):
        return mtx.toarray()
    elif isinstance(mtx, np.ndarray):
        return mtx
    else:
        raise Exception("Omics Data should be either numpy array or scipy sparse matrix.")




def _prepare_mtx(anndat, features):

    index_df = pd.DataFrame({'index':np.arange(anndat.n_vars)})
    index_df.index = anndat.var.index

    feature_index = index_df.loc[features,'index'].to_numpy()

    return _check_array(anndat.X[:,feature_index])




def Global_L(mudata, pairsDf, mods=['rna','atac'], permutations=0, percent=0.1, seed=1, max_RAM=16):
    """ 
    Function to calculate the global L index for all the pairs in anndat_multiome

    Arguments
    --------------

        mudata: single-cell multi-omics data saved as MuData object

        mods: scRNA-seq and scATAC-seq modality name in MuData object

        pairsDf: gene-peak pair DataFrame containing pairs to be calculated
                 self-prepared or call treasmo.tl.peaks_within_distance / TFBS_match

        permutations: number of permutations for significance test.
                      Default is 0, meaning no significance test
                      999 is a good choice for the test, but it might take a long time (hours) to finish depending on the number of pairs
        
        percent: percentage of cells to shuffle during permutation.
                 For most of the time, default 0.1 is already a good choice.
        
        seed: random seed to make the results reproducible

        max_RAM: maximum limitation of memory (Gb)


    Returns
    --------------
        pairsDf added with extra columns:
            Global L results
            QC metrics (sparsity)

    """

    random.seed(seed)
    np.random.seed(seed)
    start = time.time()

    print("Calculating KNN graph-based global correlation...")
    cong_mtx = mudata.obsp['connectivities'].toarray()
    np.fill_diagonal(cong_mtx, 1.0)
    eSP = lee_vec.Spatial_Pearson(cong_mtx, permutations=permutations)

    genes = pairsDf['Gene']
    peaks = pairsDf['Peak']

    gene_X = _prepare_mtx(mudata.mod[mods[0]], genes)
    peak_X = _prepare_mtx(mudata.mod[mods[1]], peaks)

    eSP = eSP.fit(gene_X, peak_X, percent=percent, seed=seed, max_RAM=max_RAM)

    pairsDf['L'] = eSP.association_
    pairsDf['L.p_value'] = eSP.significance_ if permutations else np.NaN

    if permutations:
        _,pairsDf['L.FDR'] = fdrcorrection(peaks_nearby['L.p_value'],
                                           alpha=0.05, method='indep')
    else:
        pairsDf['L.FDR'] = np.NaN

    pairsDf['gene.pct'] = mudata.mod[mods[0]].var.loc[genes,'Frac.all'].to_numpy()
    pairsDf['peak.pct'] = mudata.mod[mods[1]].var.loc[peaks,'Frac.all'].to_numpy()

    print("Finished calculating correlation. %.3fs past"%(time.time()-start))

    return pairsDf




def _calculate_LL(mudata, genes, peaks,
                  mods=['rna','atac'],
                  seed=1, max_RAM=16):
    random.seed(seed)
    np.random.seed(seed)

    cong_mtx = mudata.obsp['connectivities'].toarray()
    np.fill_diagonal(cong_mtx, 1.0)

    gene_X = _prepare_mtx(mudata.mod[mods[0]], genes)
    peak_X = _prepare_mtx(mudata.mod[mods[1]], peaks)

    eSP2 = lee_vec.Spatial_Pearson_Local(cong_mtx, permutations=0)
    eSP2 = eSP2.fit(gene_X,peak_X,seed=seed, max_RAM=max_RAM)

    '''
    if permutations:
        local_p_df = pd.DataFrame(eSP2.significance_)
        local_p_df.columns = pair_names
    '''

    L_mtx = eSP2.associations_
    pair_names = pd.Series(genes) + '~' + pd.Series(peaks)
    pair_names = pair_names.to_numpy()

    return L_mtx, pair_names




def Local_L(mudata, genes, peaks,
            mods=['rna','atac'],
            rm_dropout=False,
            seed=1, max_RAM=16):
    """ 
    Function to calculate the local L index for all the pairs in anndat_multiome

    Arguments
    --------------

        mudata: single-cell multi-omics data saved as MuData object

        mods: scRNA-seq and scATAC-seq modality name in MuData object

        genes, peaks: one-to-one lists containing gene and peak names

        rm_dropout: make local L index to be 0 if feature value is 0 (dropout)

        seed: random seed to make the results reproducible

        max_RAM: maximum limitation of memory (Gb)


    Returns
    --------------
        mudata added with
            Local L matrix and gene-peak pair names

    """

    random.seed(seed)
    np.random.seed(seed)
    start = time.time()

    print("Inferring KNN graph-based local correlation...")
    
    L_mtx, L_mtx_names = _calculate_LL(mudata, genes, peaks,
                                       mods=mods,
                                       seed=1, max_RAM=16)

    if rm_dropout:
        print("Setting local correlation to 0 for cells with no expression on either feature of a certain pair...")
        GP_names = L_mtx_names
        Dropout_mtx = _dropout_filter(mudata, GP_names, mods)
        L_mtx = L_mtx * Dropout_mtx

    mudata.uns['Local_L'] = L_mtx
    mudata.uns['Local_L_names'] = L_mtx_names

    print("Following changes made to the AnnData object:")
    print("\tKNN graph-based local correlation results saved in uns['Local_L']")
    print("\tGene-peak pair names saved in uns['Local_L_names']")

    print("Finished Inferring local correlation. %.3fs past"%(time.time()-start))
    return mudata




def _dropout_filter(mudata, GP_names, mods):

    # Gene dropout
    index_df = pd.DataFrame({'index':np.arange(len(mudata.mod[mods[0]].var_names))})
    index_df.index = mudata.mod[mods[0]].var_names
    GP_G = [gp.split('~')[0] for gp in GP_names]
    GP_genes_index = index_df.loc[GP_G,:]['index'].to_numpy()

    E_mtx = _check_array(mudata.mod[mods[0]].X)
    E_mtx_dropout_value = np.zeros(E_mtx.shape)
    Dropout_mtx = (~np.isclose(E_mtx_dropout_value,E_mtx, 1e-3)).astype(int)
    Dropout_mtx_G = Dropout_mtx[:,GP_genes_index]

    # Peak dropout
    index_df = pd.DataFrame({'index':np.arange(len(mudata.mod[mods[1]].var_names))})
    index_df.index = mudata.mod[mods[1]].var_names
    GP_P = [gp.split('~')[1] for gp in GP_names]
    GP_peaks_index = index_df.loc[GP_P,:]['index'].to_numpy()

    E_mtx = _check_array(mudata.mod[mods[1]].X)
    E_mtx_dropout_value = np.zeros(E_mtx.shape)
    Dropout_mtx = (~np.isclose(E_mtx_dropout_value,E_mtx, 1e-3)).astype(int)
    Dropout_mtx_P = Dropout_mtx[:,GP_peaks_index]
    
    Dropout_mtx = Dropout_mtx_G * Dropout_mtx_P
    
    return Dropout_mtx





#=========================================================================================
# Pearson correlation function for benchmark purpose
#=========================================================================================

def Pearsonr(mudata, genes, peaks,
                   mods=['rna','atac'],
                   p_value=False, seed=1):
    random.seed(seed)
    np.random.seed(seed)

    gene_X = _prepare_mtx(mudata.mod[mods[0]], genes)
    peak_X = _prepare_mtx(mudata.mod[mods[1]], peaks)

    gene_X = pd.DataFrame(gene_X, index=np.arange(gene_X.shape[0]), columns=np.arange(gene_X.shape[1]))
    peak_X = pd.DataFrame(peak_X, index=np.arange(peak_X.shape[0]), columns=np.arange(gene_X.shape[1]))

    global_P_df = pd.DataFrame(peak_X.corrwith(gene_X, method='pearson', axis=0))
    global_P_df.index = pd.Series(genes) + '~' + pd.Series(peaks)
    global_P_df.columns = ['r']

    if p_value:
        print("Calculating p-values...")
        p_list = []
        for i in range(global_P_df.shape[0]):
            p_list.append(stats.pearsonr(gene_X.iloc[:,i].to_numpy(), peak_X.iloc[:,i].to_numpy())[1])
        global_P_df['r.p_value'] = p_list
        print("Calculating FDR...")
        _, global_P_df['r.FDR'] = fdrcorrection(global_P_df['r.p_value'],
                                                alpha=0.05, method='indep')
    else:
        global_P_df['r.p_value'] = np.NaN
        global_P_df['r.FDR'] = np.NaN

    return global_P_df








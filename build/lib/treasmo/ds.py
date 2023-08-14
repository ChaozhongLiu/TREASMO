import numpy as np
import pandas as pd
import anndata as ad
import time

import treasmo.core as core
from minisom import MiniSom


from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
from sklearn.gaussian_process import GaussianProcessRegressor

import random
import scanpy as sc

import os
import sys
from sklearn import preprocessing

__author__ = "Chaozhong Liu"
__email__ = "czliubioinfo@gmail.com"


#=========================================================================================
# Cluster / Group marker identification
# Trajectory-based regulatory dynamics
# TF regulatory dynamics
# Statistical test for dynamics
#=========================================================================================




#=========================================================================================
# Group marker identification
#=========================================================================================

def FindAllMarkers(mudata, ident, mods=['rna','atac'], corrct_method='bonferroni', seed=1):
    """
    Function to discover regulatory markers in all groups

    Arguments
    ---------
        mudata: single-cell multi-omics data saved as MuData object
        
        ident: column name in mudata.obs containing group labels

        mods: scRNA-seq and scATAC-seq modality name in MuData object

        corrct_method: multi-test correction method, one of ['bonferroni', 'fdr']

        seed: random seed to make the results reproducible

    Returns
    ---------

        DataFrame - Differentially regulated pairs statistical test results


    """

    if 'Local_L' in mudata.uns.keys():
        local_L_df = pd.DataFrame(mudata.uns['Local_L'])
        local_L_df.columns = mudata.uns['Local_L_names']
    else:
        raise Exception("No previously calculated local correlation matrix found. \nPlease calculate the matrix first.")
        #mudata = core.Local_L(mudata, genes, peaks,
        #                      mods=mods,
        #                      rm_dropout=False,
        #                      seed=1, max_RAM=16)

        #local_L_df = pd.DataFrame(mudata.uns['Local_L'])
        #local_L_df.columns = mudata.uns['Local_L_names']
        #print("========= Finished ========")


    # cluster comparison
    print("Performing statistical test for correlation differences among identities...")
    start = time.time()
    # get clusters have at least 2 mini-bulk
    groups = mudata.obs[ident].value_counts() > 1
    groups = groups.loc[groups].index.to_list()

    local_L_df['clus'] = mudata.obs[ident].to_numpy()

    stat_df_all = pd.DataFrame(np.empty((0,13)))
    stat_df_all.columns = ['group','name','Mean.1','Mean.2','Std.1','Std.2',
                           'obsn.1','obsn.2','score','p.value','p.adj','Frac.gene.1','Frac.peak.1']

    for group in groups:
        stat_df = marker_test(local_L_df, group_1=group, group_2=None, corrct_method=corrct_method)
        try:
            stat_df['Frac.gene.1'], stat_df['Frac.peak.1'] = add_feature_sparsity(stat_df, mudata, mods=mods, group=group)
        except:
            print('\tCluster %s sparsity information not found. Skip.'%group)
            stat_df['Frac.gene.1'] = np.NaN
            stat_df['Frac.peak.1'] = np.NaN

        stat_df_all = pd.concat([stat_df_all,stat_df])
    
    print("Completed! %.2fs past."%(time.time()-start))

    return stat_df_all



def FindMarkers(mudata, ident, group_1, group_2, mods=['rna','atac'], corrct_method='bonferroni',seed=1, log=True):
    """
    Function to compare regulatory pairs between two group

    Arguments
    ---------
        mudata: single-cell multi-omics data saved as MuData object
        
        ident: column name in mudata.obs containing group labels

        group_1: first group name in ident to compare with the second

        group_2: second group name in ident to compare with the first

        mods: scRNA-seq and scATAC-seq modality name in MuData object

        corrct_method: multi-test correction method, one of ['bonferroni', 'fdr']

        seed: random seed to make the results reproducible

    Returns
    ---------

        DataFrame - Differentially regulated pairs statistical test results


    """

    if 'Local_L' in mudata.uns.keys():
        local_L_df = pd.DataFrame(mudata.uns['Local_L'])
        local_L_df.columns = mudata.uns['Local_L_names']
    else:
        raise Exception("No previously calculated local correlation matrix found. \nPlease calculate the matrix first.")
        #mudata = core.Local_L(mudata, genes, peaks,
        #                      mods=mods,
        #                      rm_dropout=False,
        #                      seed=1, max_RAM=16)

        #local_L_df = pd.DataFrame(mudata.uns['Local_L'])
        #local_L_df.columns = mudata.uns['Local_L_names']
        #print("========= Finished ========")


    # cluster comparison
    if log:
        print("Perform statistical test for correlation differences between selected two group...")
    start = time.time()

    local_L_df['clus'] = mudata.obs[ident].to_numpy()
    stat_df = marker_test(local_L_df, group_1=group_1, group_2=group_2, corrct_method=corrct_method)

    try:
        stat_df['Frac.gene.1'], stat_df['Frac.peak.1'] = add_feature_sparsity(stat_df, mudata, mods=mods, group=group_1)
        stat_df['Frac.gene.2'], stat_df['Frac.peak.2'] = add_feature_sparsity(stat_df, mudata, mods=mods, group=group_2)
    except:
        if log:
            print('\tSparsity information not found. Skip.')
        stat_df['Frac.gene.1'] = np.NaN
        stat_df['Frac.peak.1'] = np.NaN
        stat_df['Frac.gene.2'] = np.NaN
        stat_df['Frac.peak.2'] = np.NaN

    if log:
        print("Completed! %.2fs past."%(time.time()-start))

    return stat_df




def marker_test(local_L_df, group_1, group_2=None, corrct_method='bonferroni'):

    #stat_df.columns = ['group','name','Mean.1','Mean.2','Std.1',
    #                       'Std.2','obsn.1','obsn.2','score','p.value','p.adj']

    stat_df = group_stat(local_L_df, group_1=group_1, group_2=group_2)
    stat_df['score'] = np.NaN
    stat_df['p.value'] = np.NaN
    for i in range(stat_df.shape[0]):
        ttest = stats.ttest_ind_from_stats(
                mean1=stat_df['Mean.1'][i],
                std1=stat_df['Std.1'][i],
                nobs1=stat_df['obsn.1'][i],
                mean2=stat_df['Mean.2'][i],
                std2=stat_df['Std.2'][i],
                nobs2=stat_df['obsn.2'][i],
                equal_var=True,  # Welch's
                )
        stat_df.loc[stat_df.index[i],'score'] = ttest[0]
        stat_df.loc[stat_df.index[i],'p.value'] = ttest[1]
    if corrct_method == 'fdr':
        _, stat_df['p.adj'] = fdrcorrection(
                    stat_df['p.value'], alpha=0.05, method='indep'
                )
    elif corrct_method == 'bonferroni':
        stat_df['p.adj'] = np.minimum(stat_df['p.value'] * stat_df.shape[0], 1.0)
    else:
        #print("Please select correction methods from ['fdr', 'bonferroni']!")
        stat_df['p.adj'] = np.NaN

    #stat_df['Frac.gene'], stat_df['Frac.peak'] = _add_clus_sparsity(stat_df, anndat_multiome, group)

    #stat_df_all = pd.concat([stat_df_all,stat_df])

    return stat_df


def group_stat(local_L, group_1, group_2=None):
    groups_label = local_L['clus'].to_numpy()
    mask_1 = groups_label == group_1
    obsn1 = np.sum(mask_1)
    if group_2 is None:
        group_2 = 'others'
        mask_2 = groups_label != group_1
        obsn2 = len(mask_2) - obsn1
    else:
        mask_2 = groups_label == group_2
        obsn2 = np.sum(mask_2)

    local_L_df = local_L.iloc[:,:-1].copy()
    local_L_df['clus'] = 'others'
    local_L_df.loc[mask_1, 'clus'] = group_1
    local_L_df.loc[mask_2, 'clus'] = group_2

    mean_df = local_L_df.groupby('clus').mean() #.reset_index()
    mean_df = mean_df.loc[[group_1,group_2],:]
    std_df = local_L_df.groupby('clus').std()
    std_df = std_df.loc[[group_1,group_2],:]

    mean_df = mean_df.T
    mean_df.columns = ['Mean.1','Mean.2']

    std_df = std_df.T
    std_df.columns = ['Std.1','Std.2']

    stat_df = pd.concat([mean_df, std_df], axis=1)
    stat_df.insert(0,'name',stat_df.index)
    stat_df = stat_df.reset_index(drop=True)

    if group_2 == 'others':
        stat_df.insert(0,'group',group_1)
    else:
        stat_df.insert(0,'group.1',group_1)
        stat_df.insert(1,'group.2',group_2)

    stat_df['obsn.1'] = obsn1
    stat_df['obsn.2'] = obsn2
    
    return stat_df


def add_feature_sparsity(stat_df, mudata, group, mods=['rna','atac']):
    GP_G = [gp.split('~')[0] for gp in stat_df['name']]
    GP_P = [gp.split('~')[1] for gp in stat_df['name']]
    
    return mudata.mod[mods[0]].var.loc[GP_G,:]['Frac.%s'%group].to_numpy(),\
           mudata.mod[mods[1]].var.loc[GP_P,:]['Frac.%s'%group].to_numpy()




def MarkerFilter(statDf, min_pct_rna=0.1, min_pct_atac=0.05, mean_diff=1.0, p_cutoff=1e-12, plot=False):
    """
    Function to filter markers from statistical test results by sparsity, correlation difference, and p-value

    Arguments
    ---------
        statDf: Differentially regulated pairs statistical test results
        
        min_pct_rna: percentage of cells that express the gene as sparsity cutoff

        min_pct_atac: percentage of cells that have the peak as sparsity cutoff

        mean_diff: mean correlation difference between the group and background (all other groups)

        p_cutoff: adjusted p-value cutoff

        plot: if True, return volcano plot

    Returns
    ---------
        Filtered marker list with the same columns as stat_df

        if plot==True, return volcano plot


    """
    stat_df = statDf.copy()

    pd.options.mode.chained_assignment = None
    stat_df.loc[:,'score.abs'] = np.abs(stat_df['score'])
    if 'group' in stat_df.columns:
        stat_df = stat_df.sort_values(by=['group','score.abs'], ascending=False)
    else:
        stat_df = stat_df.sort_values(by=['group.1','score.abs'], ascending=False)

    mask = (stat_df['Frac.gene.1']>min_pct_rna) & (stat_df['Frac.peak.1']>min_pct_atac)
    stat_df = stat_df.loc[mask,:]

    filt = ((stat_df['p.adj']<p_cutoff) &
        (np.abs(stat_df['Mean.1']-stat_df['Mean.2']) > mean_diff))

    if plot:
        plt.scatter(stat_df['Mean.1'][~filt]-stat_df['Mean.2'][~filt],
                    -np.log10(stat_df['p.adj'][~filt]), s=2, marker='o', c='grey')
        plt.scatter(stat_df['Mean.1'][filt]-stat_df['Mean.2'][filt],
                    -np.log10(stat_df['p.adj'][filt]), s=2, marker='o', c='red')
        plt.grid(which='both', linestyle='-', linewidth='0')
        plt.xlabel('mean_1 - mean_2')
        plt.ylabel('-log10(p.adj)')

    return stat_df.loc[filt,:]




#=========================================================================================
# Trajectory Path Analysis
#=========================================================================================


def FindPathMarkers(mudata, ident, path, mods=['rna','atac'], corrct_method='bonferroni', seed=1):
    """
    One-to-one gene-peak pair correlation comparison among groups in the trajectory path

    Arguments
    ---------
        mudata: single-cell multi-omics data saved as MuData object
        
        ident: column name in mudata.obs containing group labels

        path: list of clusters ordered by their sequence on the trajectory. A path here should have no branch.

        mods: scRNA-seq and scATAC-seq modality name in MuData object

        corrct_method: multi-test correction method, one of ['bonferroni', 'fdr']

        seed: random seed to make the results reproducible

    Returns
    ---------

        DataFrame - Differentially regulated pairs statistical test results


    """

    # One-to-one comparison
    start = time.time()

    dfList = []

    for i in range(len(path)):

        for j in range(i+1, len(path)):

            stat_df = FindMarkers(mudata, ident, 
                                  group_1=path[j], group_2=path[i], 
                                  mods=mods, 
                                  corrct_method=corrct_method, 
                                  seed=seed, log=False)
            dfList.append(stat_df)

    statDf = pd.concat(dfList)

    print("Completed! %.3fs past."%(time.time()-start))

    return statDf




def _fit_data(timebinDf, xfit):
    xdata = timebinDf['time'].to_numpy()
    ydata = timebinDf['value'].to_numpy()

    filter_na = ~np.isnan(ydata)
    xdata = xdata[filter_na]
    ydata = ydata[filter_na]

    # Compute the Gaussian process fit
    gp = GaussianProcessRegressor(random_state=1)
    gp.fit(xdata[:, np.newaxis], ydata)
    #xfit = np.linspace(data['time'].min(), data['time'].max(), bins)
    yfit, _ = gp.predict(xfit[:, np.newaxis], return_std=True)

    return yfit



def TimeBinData(mudata, ident, path, pseudotime, features,
                bins=100, rm_outlier=False, fitted=None):

    cells_bool = mudata.obs[ident].isin(path).to_numpy()
    #features = mudata.uns['Local_L_names']
    CorDf = pd.DataFrame(mudata.uns['Local_L'], columns=mudata.uns['Local_L_names'])
    CorDf = CorDf.loc[cells_bool, features]

    CorDf['time'] = mudata.obs.loc[cells_bool, pseudotime].to_numpy()
    CorDf = CorDf.sort_values(by='time')

    # Construct time bins
    time_range = (CorDf['time'].min(), CorDf['time'].max())
    N_interval = bins
    interv_range = (time_range[1] - time_range[0]) / N_interval

    group_intev = pd.cut(CorDf['time'], np.arange(time_range[0]-1e-3, 
                                                  time_range[1]+interv_range,
                                                  interv_range))

    data = CorDf.groupby(group_intev).mean().dropna()
    print("Empty bins removed. %.i bins left"%(data.shape[0]))
    data.index = np.arange(data.shape[0])
    data.index.name = None

    if rm_outlier:
        data = data[np.abs(stats.zscore(data)) <= 2]

    data = data.sort_values(by='time')

    if fitted is not None:
        fitDf = pd.DataFrame(np.zeros((fitted, data.shape[1])), columns=data.columns)
        timefit = np.linspace(data['time'].min(), data['time'].max(), fitted)
        fitDf['time'] = timefit

        for feature in features:
            tmpDf = data[['time', feature]].copy()
            tmpDf.columns = ['time', 'value']
            fitDf[feature] = _fit_data(tmpDf, timefit)

        return data, fitDf

    else:
        return data




def TimeBinProportion(mudata, ident, path, pseudotime, bins=100):

    # Construct DataFrame
    cells_bool = mudata.obs[ident].isin(path).to_numpy()
    CorDf = pd.DataFrame(pd.get_dummies(mudata.obs.loc[cells_bool, ident]).astype('float32').loc[:,path].to_numpy(),
                         columns=path)
    CorDf['time'] = mudata.obs.loc[cells_bool, pseudotime].to_numpy()

    CorDf = CorDf.sort_values(by='time')

    # Construct time bins
    time_range = (CorDf['time'].min(), CorDf['time'].max())
    N_interval = bins
    interv_range = (time_range[1] - time_range[0]) / N_interval

    group_intev = pd.cut(CorDf['time'], np.arange(time_range[0]-1e-3, 
                                                  time_range[1]+interv_range,
                                                  interv_range))

    data = CorDf.groupby(group_intev).mean().dropna()
    print("Empty bins removed. %.i bins left"%(data.shape[0]))
    data.index = np.arange(data.shape[0])
    data.index.name = None
    data.columns.name = ident

    return data





def FindPathDynamics(mudata, ident, path, pseudotime, rm_outlier=True,
                     var_cutoff=0.1, range_cutoff=1.0, bins=100, plot=False):
    """
    Detect highly variable gene-peak pairs along the trajectory
    """
    # Construct DataFrame
    features = mudata.uns['Local_L_names']
    data = TimeBinData(mudata, ident, path, pseudotime, features,
                       bins=bins, rm_outlier=True)

    sumDf = pd.DataFrame(data[features].var(), columns=['variance'])
    sumDf['max'] = data[features].max()
    sumDf['min'] = data[features].min()
    sumDf['range'] = sumDf['max'] - sumDf['min']
    sumDf = sumDf.sort_values(by='variance',ascending=False)

    filt = ((sumDf['variance']>=var_cutoff) &(sumDf['range']>=range_cutoff))

    if plot:
        plt.scatter(sumDf['range'][~filt],
                    sumDf['variance'][~filt], s=2, marker='o', c='grey')
        plt.scatter(sumDf['range'][filt],
                    sumDf['variance'][filt], s=2, marker='o', c='red')
        plt.grid(which='both', linestyle='-', linewidth='0')
        plt.xlabel('min - max')
        plt.ylabel('variance')

    return sumDf[filt]





def PathDynamics(mudata, gene, peaks, ident, path, pseudotime, bins=100):
    """
    Quantify regulatory dynamics along the trajectory.

    Arguments
    ---------
        mudata: single-cell multi-omics data saved as MuData object
        
        gene: gene name

        peak: a list of peaks correlated with the gene (gene-peak pair should exist in mudata.uns['Local_L_names'])

        ident: ident label in mudata.obs columns to distinguish clusters listed in path

        path: list of clusters ordered by their sequence on the trajectory. A path here should have no branch.

        pseudotime: pseudotime label for the trajectory saved in mudata.obs

        bins: number of bins to divide the trajectory into


    Returns
    ---------

        DataFrame saved in mudata.uns['path_dynamics'] describing correlation and cluster proportion changes along the trajectory


    """

    features = [f"{gene}~{peak}" for peak in peaks]

    # Construct DataFrame
    cells_bool = mudata.obs[ident].isin(path).to_numpy()
    CorDf = pd.DataFrame(mudata.uns['Local_L'], columns=mudata.uns['Local_L_names'])
    CorDf = CorDf.loc[cells_bool, features]

    CorDf['time'] = mudata.obs.loc[cells_bool, pseudotime].to_numpy()
    CorDf[path] = pd.get_dummies(mudata.obs.loc[cells_bool, ident]).astype('float32').loc[:,path].to_numpy()

    CorDf = CorDf.sort_values(by='time')

    # Construct time bins
    time_range = (CorDf['time'].min(), CorDf['time'].max())
    N_interval = bins
    interv_range = (time_range[1] - time_range[0]) / N_interval

    group_intev = pd.cut(CorDf['time'], np.arange(time_range[0]-1e-3, 
                                                  time_range[1]+interv_range,
                                                  interv_range))
    '''
    agg_dict = {feature:['mean', 'std']}
    agg_dict['time'] = 'mean'
    for ct in path:
        agg_dict[ct] = 'mean'

    data = CorDf.groupby(group_intev).agg(agg_dict).dropna() #.mean().dropna()
    '''
    data = CorDf.groupby(group_intev).mean().dropna()

    print("Empty bins removed. %.i bins left"%(data.shape[0]))
    data.index = np.arange(data.shape[0])
    data.index.name = None

    path_name = '_'.join(path)

    if 'pathDym' not in mudata.uns.keys():
        mudata.uns['pathDym'] = {}

    if path_name not in mudata.uns['pathDym'].keys():
        mudata.uns['pathDym'][path_name] = {gene:data}
    else:
        mudata.uns['pathDym'][path_name][gene] = data
    
    return mudata



def DynamicModule(mudata, ident, path, pseudotime, features=None, bins=100, fitted=100,
                  num_iteration=5000, som_shape=(2,2), sigma=0.5, learning_rate=.1, random_seed=1):
    
    """
    Function to discover feature modules by Self-Organizing Map


    Arguments
    ---------
        mudata: single-cell multi-omics data saved as MuData object

        ident: ident label in mudata.obs columns to distinguish clusters listed in path

        path: list of clusters ordered by their sequence on the trajectory. A path here should have no branch.

        pseudotime: pseudotime label for the trajectory saved in mudata.obs

        bins: number of bins to divide the trajectory into

        re_outlier: whether to remove bins exceeding 2 std
    
        num_iteration: maximum number of iteration to optimize the SOM
        
        som_shape: (M, N) shape of the map, defines number and similarity structure of modules

        sigma: the radius of the different neighbors in the SOM

        learning_rate: optimization speed, how much weights are adjusted during each iteration

        random_seed: random seed to make the results reproducible


    Returns
    ---------
        time bin data labeled with module


    """
    if features is None:
        features = mudata.uns['Local_L_names']

    data, fitDf = TimeBinData(mudata, ident, path, pseudotime, features,
                              bins=bins, rm_outlier=False, fitted=fitted)


    som_mtx = fitDf.to_numpy()[:,:-1].T
    som_mtx = stats.zscore(som_mtx, axis=1)
    som_mtx[som_mtx > 3] = 3
    som_mtx[som_mtx < -3] = -3

    print("Start training...")
    som = MiniSom(som_shape[0], som_shape[1], som_mtx.shape[1], sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian', random_seed=random_seed)

    som.train_batch(som_mtx, num_iteration=num_iteration, verbose=True)

    # record module index
    winner_coordinates = np.array([som.winner(x) for x in som_mtx]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    somDict = {clsi:data.loc[:,np.concatenate([cluster_index==clsi, [True]])].copy() for clsi in np.unique(cluster_index)}

    return somDict




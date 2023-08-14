import numpy as np
import pandas as pd
import anndata as ad
import time
import math
import scipy as sp

import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.gaussian_process import GaussianProcessRegressor

import os
import sys
from sklearn import preprocessing


__author__ = "Chaozhong Liu"
__email__ = "czliubioinfo@gmail.com"


#=========================================================================================
# Local correlation heatmap
# Gene-peak correlation Feature plot
# Tractory-based regulatory dynamics
# Pairs Spearman correlation heatmap (demonstrate the relationship between regulatory pairs)
#=========================================================================================



# Heatmap
def LocalCor_Heatmap(mudata, pairs, groupby, cluster=True, save=None, **kwds):
    """
    Function to visualize the local L matrix by heatmap, and cluster features

    Arguments
    ---------
        mudata: single-cell multi-omics data saved as MuData object

        pairs: gene-peak pair lists to visualize

        cluster: cluster features or not

        groupby: if provided, cells / pseudo-bulks will be grouped by the label name

        save: if provided, heatmap will be saved in the file path

        **kwds: other arguments for sc.pl.heatmap()


    Returns
    ---------
        Local L index heatmap with features clustered or not


    """

    L_mtx = mudata.uns['Local_L']
    L_mtx_name = mudata.uns['Local_L_names']

    if not np.all(np.isin(np.asarray(pairs), L_mtx_name)):
        print("The following pairs listed are in the local correlation matrix:")
        print(", ".join(np.asarray(pairs)[~np.isin(np.asarray(pairs), L_mtx_name)]))
        pairs = np.asarray(pairs)[np.isin(np.asarray(pairs), L_mtx_name)]

    index_df = pd.DataFrame({'index':np.arange(L_mtx_name.shape[0])})
    index_df.index = L_mtx_name
    idx = index_df.loc[pairs,'index'].to_numpy()

    anndat_L = ad.AnnData(
        X = L_mtx[:,idx],
        obs = mudata.obs
    )
    anndat_L.var_names = pairs
    anndat_L.var_names_make_unique()

    if f'{groupby}_colors' in mudata.uns.keys():
        anndat_L.uns[f'{groupby}_colors'] = mudata.uns[f'{groupby}_colors']

    if cluster:
        _Heatmap_cluster(anndat_L, groupby=groupby, save=save, **kwds)
    else:
        _Heatmap_nocluster(anndat_L, groupby=groupby, save=save, **kwds)




def _Heatmap_cluster(anndat_L, groupby, save=None, **kwds):

    # Determine features order in heatmap
    model = AgglomerativeClustering(n_clusters=15, affinity='euclidean', 
                                  linkage='ward', compute_distances=True)
    model = model.fit(anndat_L.X.T)
    cluster_labels = model.labels_
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    R = dendrogram(linkage_matrix, no_plot=True)
    feature_order = R['leaves']

    anndat_L = anndat_L[:,feature_order]
    if save is None:
        sc.pl.heatmap(anndat_L, anndat_L.var_names,
                      groupby=groupby, **kwds)
                      #vmin=-1.0, vmax=1.0, dendrogram=True, cmap='vlag', figsize=(10,6)
    else:
        with plt.rc_context():  # Use this to set figure params like size and dpi
            sc.pl.heatmap(anndat_L, anndat_L.var_names,
                          show=False,
                          groupby=groupby, **kwds)
            plt.savefig(save)



def _Heatmap_nocluster(anndat_L, groupby, save=None, **kwds):

    if save is None:
        sc.pl.heatmap(anndat_L, anndat_L.var_names,
                      groupby=groupby, **kwds)
                      #vmin=-1.0, vmax=1.0, dendrogram=True, cmap='vlag', figsize=(10,6)
    else:
        with plt.rc_context():  # Use this to set figure params like size and dpi
            sc.pl.heatmap(anndat_L, anndat_L.var_names,
                          show=False,
                          groupby=groupby, **kwds)
            plt.savefig(save)


def _check_array(mtx):
    if sp.sparse.issparse(mtx): #isinstance(mtx, sp.spmatrix):
        return mtx.toarray()
    elif isinstance(mtx, np.ndarray):
        return mtx
    else:
        raise Exception("Omics Data should be either numpy array or scipy sparse matrix.")



# Feature Plot in UMAP
def visualize_marker(mudata, gene, peak, mods=['rna','atac'], 
                     cmaps='plasma', basis='umap',
                     vmins=None, vmaxs=None,
                     figsize=None, save=None, **kwds):

    """
    Function to visualize the gene-peak pair correlation in UMAP

    Arguments
    ---------
        mudata: single-cell multi-omics data saved as MuData object

        gene: gene name

        peak: peak name

        mods: scRNA-seq and scATAC-seq modality name in MuData object

        cmaps: Color map to use for continous variables. Could be either a single color_map or a list

        basis: the embeddings to plot. Could be either a single embedding space or a list for each of the feature

        vmins: min value to color. Could be either a single value or a list for gene, peak, and correlation

        vmaxs: max value to color. Same as vmins

        figsize: figure size

        save: str. if provided, heatmap will be saved in the file path

        **kwds: other arguments for sc.pl.embedding


    Returns
    ---------
        Embedding colored by the gene, peak, and the correlation between gene and peak


    """

    if isinstance(basis, str):
        basis = [basis, basis, basis]

    if isinstance(cmaps, str):
        cmaps = [cmaps, cmaps, cmaps]

    
    _, n_col = mudata.obs.shape

    anndat_sp_L = pd.DataFrame(mudata.uns['Local_L'])
    anndat_sp_L.columns = mudata.uns['Local_L_names']

    feature = f'{gene}~{peak}'
    mudata.obs[feature] = anndat_sp_L[feature].to_numpy()
    mudata.obs[gene] = _check_array(mudata.mod[mods[0]].X[:,mudata.mod[mods[0]].var_names == gene]).squeeze() #.toarray()
    mudata.obs[peak] = _check_array(mudata.mod[mods[1]].X[:,mudata.mod[mods[1]].var_names == peak]).squeeze() #.toarray()

    if vmins is None:
        vmins = [mudata.obs[gene].min(), mudata.obs[peak].min(), mudata.obs[feature].min()]

    if vmaxs is None:
        vmaxs = [mudata.obs[gene].max(), mudata.obs[peak].max(), mudata.obs[feature].max()]

    print('%s and %s'%(gene, peak))

    if save is None:
        _, axs = plt.subplots(ncols=3, figsize=figsize)

        for i in range(3):
            sc.pl.embedding(mudata, basis=basis[i],
                            ax=axs[i],
                            color=[gene, peak, feature][i],
                            cmap=cmaps[i],
                            vmin=vmins[i], vmax=vmaxs[i],
                            show=False,
                            **kwds)

        plt.show()

    else:
        with plt.rc_context():  # Use this to set figure params like size and dpi
            _, axs = plt.subplots(ncols=3, figsize=figsize)

            for i in range(3):
                sc.pl.embedding(mudata, basis=basis[i],
                                ax=axs[i],
                                color=[gene, peak, feature][i],
                                cmap=cmaps[i],
                                vmin=vmins[i], vmax=vmaxs[i],
                                show=False,
                                **kwds)
            plt.savefig(save)

    mudata.obs = mudata.obs.iloc[:,0:n_col]




# Visualize trajectory regulatory dynamic for a given gene
def PathDynamics(mudata, ident, path, gene, peaks=None,
                 xlim=None, ylim=None,
                 title=None, title_fontsize=15, ticks_fontsize=12,
                 x_label='Pseudotime', y_label='Correlation Strength', label_fontsize=12,
                 curve_colors=None, dot_size=5, linewidth=3,
                 ident_colors=None, show_legend=True, save=None):

    """
    Function to visualize the gene-peak pair correlation changes along pseudotime + cell type proportion visualization

    Arguments
    ---------
        mudata: single-cell multi-omics data saved as MuData object

        ident: ident label in mudata.obs columns to distinguish clusters listed in path

        path: list of clusters ordered by their sequence on the trajectory. A path here should have no branch.

        gene: gene name

        peaks: list of peak names to be paired with the gene

        *To visualize the results, need to run df.PathDynamics() first*

        xlim: (min, max) the pseudotime range limit

        ylim: (min, max) the correlation range limit, useful to remove outliers

        (title/ticks/label)_fontsize: fontsize of different element

        (x/y)_label: labels for x/y axis

        curve_colors: curve colors for each of the gene-peak pair correlation; if not specified, defaul color palette will be applied.

        dot_size: dot size in plot

        linewidth: curve width

        ident_colors: colors of each cluster to be plotted in the proportion bar. If not specified, function will look for uns[IDENT_colors] first. If not found, default color palette will be applied.

        show_legend: Show color legend or not

        save: str. if provided, heatmap will be saved in the file path


    Returns
    ---------
        Embedding colored by the gene, peak, and the correlation between gene and peak


    """

    path_name = '_'.join(path)

    try:
        data = mudata.uns['pathDym'][path_name][gene].copy()
    except:
        raise Exception(f"Path regulatory dynamics data not found in uns['pathDym']['{path_name}'']['{gene}'']. Please run ds.PathDynamics() first.")


    # Filter by xlim
    if xlim is not None:
        n_before = data.shape[0]
        data = data.loc[(data['time']>=xlim[0]) & (data['time']<=xlim[1]), :]
        n_after = data.shape[0]
        if n_after < n_before:
            print(f"{n_before-n_after} time bins removed from plotting")


    # Get the features name
    time_index = data.columns.tolist().index('time')
    if peaks is None:
        features = data.columns.tolist()[0:time_index]
    else:
        features = [f"{gene}~{peak}" for peak in peaks]
        data = data.loc[:, features+['time']+path]


    # Calculate cumulative cell proportion
    prpmtx = data[path].to_numpy()
    prpmtx = (prpmtx / prpmtx.sum(axis=1)[:, np.newaxis])
    data[path] = prpmtx
    data[path] = data[path].cumsum(axis=1)
    data[path[-1]] = 1.0

    # Select colors for features and clusters
    color_tmp = plt.cm.rainbow(np.linspace(0, 1, len(features)+len(path)))
    # curve color
    if curve_colors is None:
        curve_colors = []
        for i in range(len(features)):
            curve_colors.append(mcolors.to_hex(color_tmp[i]))
        #curve_colors = np.array(sc.pl.palettes.zeileis_28)[0:len(features)]
    else:
        if len(curve_colors) == len(features):
            pass
        else:
            raise Exception(f"Number of curve colors {len(curve_colors)} doesn't match number of features {len(features)}")
    # cluster color
    if ident_colors is None:
        if f'{ident}_colors' in mudata.uns.keys():
            ident_colors = list(mudata.uns[f'{ident}_colors'])
            path_index = [mudata.obs[ident].cat.categories.tolist().index(cluster) for cluster in path]
            ident_colors = [ident_colors[i] for i in path_index]
        else:
            print(f"{ident}_colors not found in mudata.uns, will use defaul color palette.")
            ident_colors = []
            for i in range(len(features), len(features)+len(path)):
                ident_colors.append(mcolors.to_hex(color_tmp[i]))
            #ident_colors = np.array(sc.pl.palettes.zeileis_28)[-len(path):]

    # Prepare for plot
    if title is None:
        title = f"{'-'.join(path)}: {gene}"

    curveDf = data.loc[:, features+['time']].copy()
    prpDf = data.loc[:, path+['time']].copy()
    prpDf.columns.name = ident

    # Set up the figure
    fig1 = plt.figure(constrained_layout=False)
    gs1 = fig1.add_gridspec(nrows=10, ncols=10, left=0.0, right=1.0, top=0.9, bottom=0.3, wspace=0.05, hspace=0.2)
    f1_ax1 = fig1.add_subplot(gs1[:-1, :-1])
    f1_ax2 = fig1.add_subplot(gs1[-1, :-1])
    f1_ax3 = fig1.add_subplot(gs1[:-1, -1])
    hfont = {'fontname':'sans'}
    fig1.suptitle(title, fontsize=title_fontsize, x=0.45 , y=0.95, verticalalignment='bottom', **hfont)

    _ComplexCurvePlot(curveDf, prpDf, f1_ax1, f1_ax2, xlim=xlim, ylim=ylim,
                      ticks_fontsize=ticks_fontsize,
                      x_label=x_label, y_label=y_label, label_fontsize=label_fontsize,
                      curve_colors=curve_colors, dot_size=dot_size, linewidth=linewidth,
                      ident_colors=ident_colors)

    # Legend
    if show_legend:
        legend_elements = [Line2D([0], [0], linewidth=0, label='Curves')]
        legend_elements += [Line2D([0], [0], linewidth=0, marker='o', color=curve_colors[i], label=features[i].split('~')[1], markersize=7) for i in range(len(features))]
        legend_elements += [Line2D([0], [0], linewidth=0, label='')]
        legend_elements += [Line2D([0], [0], linewidth=0, label=ident)]
        legend_elements += [Line2D([0], [0], linewidth=0, marker='o', color=ident_colors[i], label=path[i], markersize=7) for i in range(len(path))]
        #ax.set_title(feature)
        f1_ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.1, 0.5), frameon=False)
        f1_ax3.tick_params(left = False, right = False , labelleft = False ,
                           labelbottom = False, bottom = False)
        f1_ax3.grid(which='both', linestyle='-', linewidth='0')
        f1_ax3.axis('off')

    if save is None:
        plt.show()
    else:
        with plt.rc_context():
            plt.savefig(save)




def _ComplexCurvePlot(curveDf, prpDf, f1_ax1, f1_ax2, xlim=None, ylim=None,
                      ticks_fontsize=12,
                      x_label='Pseudotime', y_label='Correlation Strength', label_fontsize=12,
                      curve_colors=None, dot_size=5, linewidth=3,
                      ident_colors=None):
    
    # Prepare some arguments
    if isinstance(curveDf, dict):
        features = list(curveDf.keys())
    else:
        features = curveDf.columns.tolist()
        features.remove('time')
    if isinstance(linewidth, int):
        linewidth = [linewidth] * len(features)
    path = prpDf.columns.tolist()
    path.remove('time')
    ident = prpDf.columns.name


    # Plot the curves
    if isinstance(curveDf, dict):
        time_range = (curveDf[list(curveDf.keys())[0]]['time'].min(), curveDf[list(curveDf.keys())[0]]['time'].max())
        i_key = 0
        for key_name, data_tmp in curveDf.items():
            _curve_plot(data_tmp, feature=key_name, colors=curve_colors[i_key], xlim=xlim, ylim=ylim, ax=f1_ax1, dot_size=dot_size, linewidth=linewidth[i_key])
            i_key += 1
    else:
        time_range = (curveDf['time'].min(), curveDf['time'].max())
        for i in range(len(features)):
            data_tmp = curveDf[['time',features[i]]].copy()
            data_tmp.columns = ['time', 'value']
            _curve_plot(data_tmp, feature=features[i], colors=curve_colors[i], xlim=xlim, ylim=ylim, ax=f1_ax1, dot_size=dot_size, linewidth=linewidth[i])
    
    if xlim is None:
        f1_ax1.set_xlim(time_range[0], time_range[1])
    else:
        f1_ax1.set_xlim(xlim[0], xlim[1]) #data['time'].min(), data['time'].max())
    if ylim is not None:
        f1_ax1.set_ylim(ylim[0], ylim[1]) 
    f1_ax1.tick_params(left = True, right = False , labelleft = True ,
                       labelbottom = False, bottom = False)
    f1_ax1.grid(which='both', linestyle='-', linewidth='0')
    f1_ax1.set_ylabel(y_label, fontsize=label_fontsize)
    #Font size in plots
    f1_ax1.yaxis.label.set_fontsize(label_fontsize)
    f1_ax1.tick_params(axis='both', which='major', labelsize=ticks_fontsize)


    # Plot the proportion bar
    if f1_ax2 is None:
        pass
    else:
        _prp_var(prpDf, path, colors=ident_colors, xlim=xlim, ax=f1_ax2)
        f1_ax2.set_xlabel(x_label, fontsize=label_fontsize)
        #Font size in plots
        f1_ax2.xaxis.label.set_fontsize(label_fontsize)
        f1_ax2.tick_params(axis='both', which='major', labelsize=ticks_fontsize)




def _curve_plot(data, feature, colors, dot_size, linewidth, xlim=None, ylim=None, ax=None):

    if ax is None:
        ax = plt.gca()

    #for i in range(len(features)):
    # define the model and draw some data
    xdata = data['time'].to_numpy()
    ydata = data['value'].to_numpy()

    #filter by ylim
    #if ylim is not None:
    #    filter_bool = (ydata >= ylim[0]) & (ydata <= ylim[1])
        #xdata = xdata[filter_bool]
        #ydata = ydata[filter_bool]
        #if np.sum(~filter_bool) > 0:
        #    print(f"{feature}: {np.sum(~filter_bool)} time bins removed based on provided ylim.")
    
    filter_na = ~np.isnan(ydata)
    xdata = xdata[filter_na]
    ydata = ydata[filter_na]

    # Compute the Gaussian process fit
    gp = GaussianProcessRegressor(random_state=1)
    gp.fit(xdata[:, np.newaxis], ydata)
    xfit = np.linspace(data['time'].min(), data['time'].max(), 1000)
    yfit, std = gp.predict(xfit[:, np.newaxis], return_std=True)
    dyfit = 1.96 * std #np.sqrt(MSE)
    
    # confidence interval
    '''
    stdDf = data[['time','value']].groupby('time').std()
    stdata = stdDf['value'].to_numpy()
    timePoint = stdDf.index.to_numpy()

    gp = GaussianProcessRegressor(random_state=1)
    gp.fit(timePoint[:, np.newaxis], timePoint)
    stdfit, _ = gp.predict(xfit[:, np.newaxis], return_std=True)

    # Generate samples from posterior distribution. 
    y_hat_samples = gp.sample_y(xfit[:, np.newaxis], n_samples=1000)
    # Compute the mean of the sample. 
    y_hat = np.apply_over_axes(func=np.mean, a=y_hat_samples, axes=1).squeeze()
    # Compute the standard deviation of the sample. 
    stdfit = np.apply_over_axes(func=np.std, a=y_hat_samples, axes=1).squeeze()
    '''
    
    #ax.plot(xdata, ydata, 'or', size=dot_size)
    ax.scatter(xdata, ydata, s=dot_size, c=colors)
    ax.plot(xfit, yfit, '-', color=colors, linewidth=linewidth, alpha=0.8)
    ax.fill_between(xfit, yfit-dyfit, yfit+dyfit,
                    color=colors, alpha=0.5)




def _prp_var(data, path, colors, xlim=None, ax=None):

    if ax is None:
        ax = plt.gca()

    for i in range(len(path)):
        ax.plot(data['time'], data[path[i]], '-', linewidth=0)
        if i == 0:
            ax.fill_between(data['time'], 0, data[path[i]],
                             color=colors[i], alpha=0.75)
        else:
            ax.fill_between(data['time'], data[path[i-1]], data[path[i]],
                             color=colors[i], alpha=0.75)
    
    
    if xlim is None:
        ax.set_xlim(data['time'].min(), data['time'].max())
    else:
        ax.set_xlim(xlim[0], xlim[1])

    ax.set_ylim(0, 1)
    #ax.tight_layout()
    ax.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = True, bottom = True)
    ax.grid(which='both', linestyle='-', linewidth='0')



def DynamicSumMtx(mudata, ident, path, gene, peaks=None, 
                  feature_colors=None, show_legend=True, save=None, **kwds):

    path_name = '_'.join(path)

    try:
        data = mudata.uns['pathDym'][path_name][gene].copy()
    except:
        raise Exception(f"Path regulatory dynamics data not found in uns['pathDym']['{path_name}'']['{gene}'']. Please run ds.PathDynamics() first.")

    time_index = data.columns.tolist().index('time')
    #path = data.columns.tolist()[time_index+1:]
    if peaks is None:
        features = data.columns.tolist()[0:time_index]
    else:
        features = [f"{gene}~{peak}" for peak in peaks]

    if len(features) <= 1:
        raise Exception("Please select at least two gene-peak pairs for analysis.")

    corrDf = data[features].corr(method='spearman')
    corrDf.index = corrDf.index.str.split('~').str[1]
    corrDf.columns = corrDf.columns.str.split('~').str[1]

    # plot the heatmap
    mask = np.triu(np.ones_like(corrDf))
    np.fill_diagonal(mask, 0.0)

    fig, ax = plt.subplots()
    sns.heatmap(corrDf, mask=mask, **kwds)
    plt.grid(which='both', linestyle='-', linewidth='0')
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    color_tmp = plt.cm.rainbow(np.linspace(0, 1, len(features)+len(path)))
    if feature_colors is None:
        feature_colors = []
        for i in range(len(features)):
            feature_colors.append(mcolors.to_hex(color_tmp[i]))
    else:
        if len(feature_colors) == len(features):
            pass
        else:
            raise Exception(f"Number of feature colors {len(feature_colors)} doesn't match number of features {len(features)}")

    symbolsx = ["\u26AB"] * len(features)
    colorsx = feature_colors
    ax.set_xticks([i+0.5 for i in range(len(symbolsx))])
    ax.set_xticklabels(symbolsx, size=25)
    for tick, color in zip(ax.get_xticklabels(), colorsx):
        tick.set_color(color)

    ax.set_yticks([i+0.5 for i in range(len(symbolsx))])
    ax.set_yticklabels(symbolsx, size=25)
    for tick, color in zip(ax.get_yticklabels(), colorsx):
        tick.set_color(color)

    if show_legend:
        legend_elements = [Line2D([0], [0], linewidth=0, label='Peaks')]
        legend_elements += [Line2D([0], [0], linewidth=0, marker='o', color=feature_colors[i], label=features[i].split('~')[1], markersize=7) for i in range(len(features))]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.4, 0.5), frameon=False)


    if save is None:
        plt.show()
    else:
        with plt.rc_context():
            plt.savefig(save)



def DynamicModule(mudata, somDict, prpDfin, xlim=None, ylim=None,
                  split=False, n_cols=3,
                  title=None, title_fontsize=15, ticks_fontsize=12,
                  x_label='Pseudotime', y_label='Correlation Strength', label_fontsize=12,
                  curve_colors=None, dot_size=5, linewidth=3,
                  ident_colors=None, show_legend=True, save=None):
    
    # Some variables
    modules = list(somDict.keys())
    prpDf = prpDfin.copy()
    ident = prpDf.columns.name
    path = prpDf.columns[:-1].tolist()


    # Select colors for modules and clusters
    color_tmp = plt.cm.rainbow(np.linspace(0, 1, len(modules)+len(path)))
    # curve color
    if curve_colors is None:
        curve_colors = []
        for i in range(len(modules)):
            curve_colors.append(mcolors.to_hex(color_tmp[i]))
    else:
        if len(curve_colors) == len(modules):
            pass
        else:
            raise Exception(f"Number of curve colors {len(curve_colors)} doesn't match number of modules {len(modules)}")
    # cluster color
    if ident_colors is None:
        if f'{ident}_colors' in mudata.uns.keys():
            ident_colors = list(mudata.uns[f'{ident}_colors'])
            path_index = [mudata.obs[ident].cat.categories.tolist().index(cluster) for cluster in path]
            ident_colors = [ident_colors[i] for i in path_index]
        else:
            print(f"{ident}_colors not found in mudata.uns, will use defaul color palette.")
            ident_colors = []
            for i in range(len(modules), len(modules)+len(path)):
                ident_colors.append(mcolors.to_hex(color_tmp[i]))


    # Prepare module-wise data for plotting
    dataDict = {}
    for sommod in somDict.keys():
        data = somDict[sommod].copy()

        # filter by xlim
        if xlim is not None:
            data = data.loc[(data['time']>=xlim[0]) & (data['time']<=xlim[1]), :]

        # wide to long
        long_list = [ pd.DataFrame(data[['time', feature]].to_numpy(), columns=['time','value']) for feature in data.columns[:-1]]
        for i in range(len(data.columns[:-1])):
            long_list[i].insert(2, 'feature', data.columns[:-1][i])
        longDf = pd.concat(long_list)
        longDf['module'] = sommod
        longDf.index = np.arange(longDf.shape[0])
        dataDict[sommod] = longDf

    # Prepare cell type proportion
    prpmtx = prpDf[path].to_numpy()
    prpmtx = (prpmtx / prpmtx.sum(axis=1)[:, np.newaxis])
    prpDf[path] = prpmtx
    prpDf[path] = prpDf[path].cumsum(axis=1)
    prpDf[path[-1]] = 1.0


    # Prepare for plot
    if title is None:
        title = f"{'-'.join(path)} Modules"


    if split:
        n_modules = len(modules)
        n_rows = math.ceil(n_modules/n_cols)

        fig1 = plt.figure(constrained_layout=False)
        gs = fig1.add_gridspec(n_rows, n_cols+1, #9*n_cols+1, 
                               left=0.0, right=(0.1+0.9*n_cols), 
                               bottom=0.0, top=(0.6*n_rows),
                               width_ratios = [9]*n_cols + [1],
                               hspace=0.15, wspace=0.2)
        #plt.subplots_adjust(wspace=None, hspace=None)

        ax_list = []
        max_col = 0
        max_row = {ir:0 for ir in range(n_cols)}
        for i in range(n_modules):
            index_col = i % n_cols
            if max_col < index_col:
                max_col = index_col

            index_row = i // n_cols
            if max_row[index_col] < index_row:
                max_row[index_col] = index_row
            
            gsij = gs[index_row:(index_row+1), index_col:(index_col+1)].subgridspec(10, 1, hspace=0.02)

            ax_list.append((
                            fig1.add_subplot(gsij[:-1,:]), 
                            fig1.add_subplot(gsij[-1,:])
                          ))

        f1_legend = fig1.add_subplot(gs[0:(index_row+1), (max_col+1):(max_col+2)])

        hfont = {'fontname':'sans'}
        fig1.suptitle(title, fontsize=title_fontsize, x=(0.1+0.9*n_cols)*(9*n_cols)/(9*n_cols+1)/2 , y=(0.6*n_rows)+0.05, verticalalignment='bottom' , **hfont)

        for i in range(len(ax_list)):
            index_col = i % n_cols
            index_row = i // n_cols
            
            tmp_ax1, tmp_ax2 = ax_list[i]
            data = somDict[modules[i]].copy()
            # filter by xlim
            if xlim is not None:
                data = data.loc[(data['time']>=xlim[0]) & (data['time']<=xlim[1]), :]

            mod_f = data.columns.tolist()
            mod_f.remove('time')
            data[modules[i]] = data[mod_f].mean(axis=1)

            _ComplexCurvePlot(data, prpDf, tmp_ax1, tmp_ax2, xlim=xlim, ylim=ylim,
                          ticks_fontsize=ticks_fontsize,
                          x_label=x_label if index_row==max_row[index_col] else '', 
                              y_label=y_label if index_col==0 else '', label_fontsize=label_fontsize,
                          curve_colors=['#e9ecec']*(data.shape[1]-1), dot_size=0, linewidth=1,
                          ident_colors=ident_colors)

            modsumDf = dataDict[modules[i]][['time','value']].copy() #data[['time',modules[i]]].copy()
            
            _ComplexCurvePlot(modsumDf, prpDf, tmp_ax1, None, xlim=xlim, ylim=ylim,
                          ticks_fontsize=ticks_fontsize,
                          x_label=x_label if index_row==max_row[index_col] else '', 
                              y_label=y_label if index_col==0 else '', label_fontsize=label_fontsize,
                          curve_colors=[curve_colors[i]], dot_size=0, linewidth=linewidth,
                          ident_colors=ident_colors)


    else:

        # Setup the plot
        fig1 = plt.figure(constrained_layout=False)
        gs1 = fig1.add_gridspec(nrows=10, ncols=10, left=0.0, right=1.0, top=0.9, bottom=0.3, wspace=0.05, hspace=0.2)
        f1_ax1 = fig1.add_subplot(gs1[:-1, :-1])
        f1_ax2 = fig1.add_subplot(gs1[-1, :-1])
        f1_legend = fig1.add_subplot(gs1[:-1, -1])
        hfont = {'fontname':'sans'}
        fig1.suptitle(title, fontsize=title_fontsize, x=0.45 , y=0.95, verticalalignment='bottom', **hfont)

        _ComplexCurvePlot(dataDict, prpDf, f1_ax1, f1_ax2, xlim=xlim, ylim=ylim,
                          ticks_fontsize=ticks_fontsize,
                          x_label=x_label, y_label=y_label, label_fontsize=label_fontsize,
                          curve_colors=curve_colors, dot_size=0, linewidth=linewidth,
                          ident_colors=ident_colors)

    # Legend
    if show_legend:
        legend_elements = [Line2D([0], [0], linewidth=0, label='Curves')]
        legend_elements += [Line2D([0], [0], linewidth=0, marker='o', color=curve_colors[i], label=modules[i], markersize=7) for i in range(len(modules))]
        legend_elements += [Line2D([0], [0], linewidth=0, label='')]
        legend_elements += [Line2D([0], [0], linewidth=0, label=ident)]
        legend_elements += [Line2D([0], [0], linewidth=0, marker='o', color=ident_colors[i], label=path[i], markersize=7) for i in range(len(path))]
        #ax.set_title(feature)
        f1_legend.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.1, 0.5), frameon=False)
        f1_legend.tick_params(left = False, right = False , labelleft = False ,
                           labelbottom = False, bottom = False)
        f1_legend.grid(which='both', linestyle='-', linewidth='0')
        f1_legend.axis('off')

    if save is None:
        plt.show()
    else:
        with plt.rc_context():
            plt.savefig(save)







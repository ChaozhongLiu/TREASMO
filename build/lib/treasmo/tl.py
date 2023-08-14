import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import random
import time

from sklearn.neighbors import KNeighborsRegressor


__author__ = "Chaozhong Liu"
__email__ = "czliubioinfo@gmail.com"


# ============= Toolkit functions =============
# For Future work
#   Pseudo-bulk Generation (and the functions mapping results back to single-cell data)
#   nearest neighbor map construction
#   imputation option


# Implemented function
#   Feature sparsity summary
#   Gene-peak annotation
#   TF-peak annotation
# =============================================



# Calculate feature sparsity =================================================

def _fill_miss_index(df, n):
    miss_ind = np.setdiff1d(np.arange(n), df['ind'].to_numpy())
    miss_pd = pd.DataFrame({'ind':miss_ind,'size':0})
    df = pd.concat([df,miss_pd], axis=0).sort_values('ind')
    return df


def feature_sparsity(mudata, group_by={}):
    """
    Function to add sparisity info in AnnData.var

    Arguments
    --------------
        mudata: single-cell multi-omics data saved as MuData object

        group_by: dict; If provided, calculate per group feature sparsity for each modality
                  Example: {'rna':'cell_type', 'atac':'cell_type'}

    Return
    --------------
        anndat with
            var['Frac.all']
            var['Frac.GroupName']

    """

    for mod in mudata.mod.keys():

        spsmtx = sp.sparse.csr_matrix(mudata.mod[mod].X)
        non_zero = spsmtx.nonzero()
        non_zero_row = non_zero[0]
        non_zero_ind = non_zero[1]

        ## All cell sparsity
        non_zero_pd = pd.DataFrame(non_zero_ind)
        non_zero_pd = non_zero_pd.groupby(by=0, as_index=False).size()
        non_zero_pd = pd.DataFrame(non_zero_pd)
        non_zero_pd.columns = ['ind','size']
        non_zero_pd = _fill_miss_index(non_zero_pd, mudata.mod[mod].n_vars)
        non_zero_pd['sparsity'] = non_zero_pd['size'] / mudata.mod[mod].n_obs

        mudata.mod[mod].var['Frac.all'] = non_zero_pd['sparsity'].to_numpy()

        ## Per group sparsity
        if mod in group_by.keys():

            non_zero_pd = pd.DataFrame({'ind':non_zero_ind,
                                        'group':mudata.mod[mod].obs[group_by[mod]].to_numpy()[non_zero_row]})
            non_zero_pd = non_zero_pd.groupby(by=['ind','group'], as_index=False).size().sort_values('group')
            non_zero_pd['n_cells'] = mudata.mod[mod].obs[group_by[mod]].value_counts()[non_zero_pd['group']].to_numpy()

            for ident in mudata.mod[mod].obs[group_by[mod]].unique():
                df = non_zero_pd.loc[non_zero_pd['group']==ident,:].copy()
                n_cells = df['n_cells'].tolist()[0]
                df = df.loc[:,['ind','size']]
                df = _fill_miss_index(df, mudata.mod[mod].n_vars)
                df['sparsity'] = df['size'] / n_cells
                mudata.mod[mod].var['Frac.%s'%ident] = df['sparsity'].to_numpy()


    return mudata




# Annotate genes with nearby peaks =================================================

def get_gloc_from_atac_data(peaks, split_symbol):
    """
    Method to get the genomic locations (including the middle point)of peaks
        Author: Linhua Wang Linhua.Wang@bcm.edu
        https://github.com/LiuzLab/Neurips2021/blob/master/task1_utils1.py
    """
    glocs = peaks.tolist()
    glocs = [c for c in glocs if 'chr' in c]
    chrms, ranges, sts, ends, midpoints = [], [], [], [], []
    for gl in glocs:
        chrms.append(gl.split(split_symbol[0],1)[0])
        st, end = int(gl.split(split_symbol[0],1)[1].split(split_symbol[1],1)[0]), int(gl.split(split_symbol[0],1)[1].split(split_symbol[1],1)[1])
        sts.append(st)
        ends.append(end)
        #midpoints.append(int((st + end)/2))
        #ranges.append("_".join(gl.split(split_symbol[0])[1].split(split_symbol[1])))
    gloc_df = pd.DataFrame({'chrm': chrms, #'grange': ranges,
                        'start': sts, 'end': ends}, index=glocs)
                        #'midpoint': midpoints}, index=glocs)
    return gloc_df




def nearby_peaks(g_array, min_op=1):
    g_chr = str(g_array[2])
    g_array_n = g_array[0:2].copy().astype(float)
    op = np.minimum(p_array[:,1].squeeze(),g_array_n[1]) - np.maximum(p_array[:,0].squeeze(),g_array_n[0])
    filter_bool = (op>=min_op) & (chr_list==g_chr)
    op_index = np.arange(op.shape[0])[filter_bool]
    return np.array(','.join(list(plist[op_index])), dtype=object)
    

def peaks_within_distance(genes, peaks, upstream, downstream, ref_gtf_fn,
                          no_intersect=True, id_type='Symbol', split_symbol=['-','-']):
    """
    Function to annotate genes with nearby peaks

    Arguments
    --------------
        genes: gene list to be annotated

        peaks: peak candidate list

        upstream: include peaks N bp upstream of the TSS

        downstream: include peaks N bp downstream of the TES

        ref_gtf_fn: GTF format file containing gene location information
                    see example at https://github.com/ChaozhongLiu/scGREAT/tree/main/replication/data
                    - Homo_sapiens.GRCh38.104.GeneLoc.Tab.txt
                    - Mus_musculus.GRCm38.100.GeneLoc.Tab.txt

        no_intersect: if peak within the range lies in another gene body, remove the peak or not

        id_type: ID type of the gene list, ref_gtf_fn should contain this ID

        split_symbol: how peak location ID is merged
                      'chr1-12345-23456' - split_symbol=['-','-']
                      'chr1:12345-23456' - split_symbol=[':','-']

    Return
    --------------
        DataFrame containing gene annotation infos

    """

    gloc_df = get_gloc_from_atac_data(peaks, split_symbol=split_symbol)
    
    ref_gtf = pd.read_csv(ref_gtf_fn, sep='\t')
    if id_type=='Symbol':
        ref_gtf = ref_gtf.loc[ref_gtf['GeneSymbol'].isin(genes)]
    elif id_type=='Ensembl':
        ref_gtf = ref_gtf.loc[ref_gtf['gene_id'].isin(genes)]
        ref_gtf['GeneSymbol'] = ref_gtf['gene_id'].copy()
    #upstream, downstream = 100000, 100000
    ref_gtf['start_exp'] = 0
    ref_gtf['end_exp'] = 0

    ref_gtf.loc[ref_gtf['Strand']=='+','start_exp'] = ref_gtf.loc[ref_gtf['Strand']=='+','start'] - upstream
    ref_gtf.loc[ref_gtf['Strand']=='-','start_exp'] = ref_gtf.loc[ref_gtf['Strand']=='-','start'] - downstream

    ref_gtf.loc[ref_gtf['Strand']=='+','end_exp'] = ref_gtf.loc[ref_gtf['Strand']=='+','end'] + downstream
    ref_gtf.loc[ref_gtf['Strand']=='-','end_exp'] = ref_gtf.loc[ref_gtf['Strand']=='-','end'] + upstream
    
    global p_array, plist, chr_list
    p_array = gloc_df[['start','end']].to_numpy()
    plist = gloc_df.index.to_numpy()
    chr_list = gloc_df['chrm'].to_numpy().astype(object)

    genes_loc = ref_gtf[['start_exp','end_exp','chr']].to_numpy()
    selected_peaks = np.apply_along_axis(nearby_peaks, axis=1, arr=genes_loc)
    
    ref_gtf['nearby_peaks'] = selected_peaks
    ref_gtf = ref_gtf.loc[~(ref_gtf['nearby_peaks']=='')]
    ref_gtf['nearby_peaks'] = ref_gtf['nearby_peaks'].str.split(',')
    
    peaks_nearby_new = pd.DataFrame(np.ones((0,6)), columns=['GeneSymbol','Strand','chr','start','end','nearby_peaks']) #ref_gtf.columns[[1,4,5,6,7,10]])
    for i in range(ref_gtf.shape[0]):
        n_lines = len(ref_gtf.loc[ref_gtf.index[i],'nearby_peaks'])
        peaks_nearby_new_tmp = ref_gtf.iloc[np.repeat(i,n_lines),:]
        peaks_nearby_new_tmp = peaks_nearby_new_tmp.loc[:,['GeneSymbol','Strand','chr','start','end','nearby_peaks']]
        peaks_nearby_new_tmp['nearby_peaks'] = ref_gtf.loc[ref_gtf.index[i],'nearby_peaks']

        peaks_nearby_new = pd.concat([peaks_nearby_new, peaks_nearby_new_tmp], axis=0)
        peaks_nearby_new.index = np.arange(peaks_nearby_new.shape[0])
    
    #return peaks_nearby_new

    peaks_nearby_new['midp'] = peaks_nearby_new['nearby_peaks'].apply(lambda x: \
                               int((int(x.split(split_symbol[0],1)[1].split(split_symbol[1],1)[0]) + int(x.split(split_symbol[0],1)[1].split(split_symbol[1],1)[1]))/2) )

    plus_strand = peaks_nearby_new['Strand']=='+'

    peaks_nearby_new['tss_dist'] = np.NaN
    peaks_nearby_new.loc[plus_strand,'tss_dist'] = peaks_nearby_new.loc[plus_strand,'midp'] - peaks_nearby_new.loc[plus_strand,'start']
    peaks_nearby_new.loc[~plus_strand,'tss_dist'] = peaks_nearby_new.loc[~plus_strand,'end'] - peaks_nearby_new.loc[~plus_strand,'midp']

    peaks_nearby_new['tts_dist'] = np.NaN
    peaks_nearby_new.loc[plus_strand,'tts_dist'] = peaks_nearby_new.loc[plus_strand,'midp'] - peaks_nearby_new.loc[plus_strand,'end']
    peaks_nearby_new.loc[~plus_strand,'tts_dist'] = peaks_nearby_new.loc[~plus_strand,'start'] - peaks_nearby_new.loc[~plus_strand,'midp']

    promoters = ((peaks_nearby_new['tss_dist'] >= -2000) & (peaks_nearby_new['tss_dist'] <= 0)).astype(int)
    genebodys = ((peaks_nearby_new['tss_dist'] > 0) & (peaks_nearby_new['tts_dist'] <= 0)).astype(int)

    peaks_nearby_new['pRegion'] = promoters
    peaks_nearby_new['gBody'] = genebodys
    
    peaks_nearby_final = peaks_nearby_new.loc[(peaks_nearby_new['tss_dist']>=-upstream)&
                                              (peaks_nearby_new['tts_dist']<=downstream)].copy()
    
    df = peaks_nearby_final.copy()
    df.columns = ['Gene','Strand','chr','start','end','Peak','midp','tss_dist','tts_dist','pRegion','gBody']
    df = df.loc[~df.iloc[:,[0,5]].duplicated(),:].copy()
    
    if no_intersect:
        print("Remove nearby peaks if it lies on the gene body or promoter regions of other genes.")
        duplicated_peaks = df['Peak'].duplicated(keep=False)
        dup_df = df.loc[duplicated_peaks,:].copy()
        dup_df = dup_df.groupby(by='Peak')[['pRegion','gBody']].max()
        body_peaks = dup_df.index[(dup_df['pRegion']==1) | (dup_df['gBody']==1)]

        peaks_rm = (df['Peak'].isin(body_peaks)) & (df['pRegion']==0) & (df['gBody']==0)

        df = df.loc[~peaks_rm,:]
    
    df = df.iloc[:,[0,5,7,9,8,10]].copy()
    df.index = np.arange(df.shape[0])
    
    return df




# Annotate TF with motif regions =================================================


def TFBS_match(genes, peaks, ref_fn, min_overlap=1, split_symbol=['-','-']):

    """
    Function to annotate TF with binding sites

    Arguments
    --------------
        genes: gene list to be annotated

        peaks: peak candidate list

        ref_fn: BED format file containing gene binding site location information
                see example below from JASPAR TFBS genome track (https://jaspar.genereg.net/genome-tracks/)
                    chr1    280     298     AGL3    821     -
                    chr1    309     327     AGL3    823     +
                    chr1    309     327     AGL3    882     -
                    chr1    1577    1595    AGL3    823     +
                    chr1    1577    1595    AGL3    883     -
        
        min_overlap: number; the minimum number of overlaped base pairs between peak and TFBS

        split_symbol: how peak location ID is merged
                      'chr1-12345-23456' - split_symbol=['-','-']
                      'chr1:12345-23456' - split_symbol=[':','-']

    Return
    --------------
        DataFrame containing TF annotation infos

    """


    gloc_df = get_gloc_from_atac_data(peaks, split_symbol=split_symbol)
    
    ref_gtf = pd.read_csv(ref_fn, sep='\t', header=None)
    ref_gtf.columns = ['chr', 'start', 'end', 'Gene', 'score', 'Strand']
    ref_gtf = ref_gtf.loc[ref_gtf['Gene'].isin(genes)]

    upstream, downstream = 0, 0
    ref_gtf['start_exp'] = 0
    ref_gtf['end_exp'] = 0

    ref_gtf.loc[ref_gtf['Strand']=='+','start_exp'] = ref_gtf.loc[ref_gtf['Strand']=='+','start'] - upstream
    ref_gtf.loc[ref_gtf['Strand']=='-','start_exp'] = ref_gtf.loc[ref_gtf['Strand']=='-','start'] - downstream

    ref_gtf.loc[ref_gtf['Strand']=='+','end_exp'] = ref_gtf.loc[ref_gtf['Strand']=='+','end'] + downstream
    ref_gtf.loc[ref_gtf['Strand']=='-','end_exp'] = ref_gtf.loc[ref_gtf['Strand']=='-','end'] + upstream

    global p_array, plist, chr_list
    p_array = gloc_df[['start','end']].to_numpy()
    plist = gloc_df.index.to_numpy()
    chr_list = gloc_df['chrm'].to_numpy().astype(object)

    genes_loc = ref_gtf[['start_exp','end_exp','chr']].to_numpy()
    selected_peaks = np.apply_along_axis(nearby_peaks, axis=1, arr=genes_loc)
    
    ref_gtf['TFBS'] = selected_peaks
    ref_gtf = ref_gtf.loc[~(ref_gtf['TFBS']=='')]
    ref_gtf['TFBS'] = ref_gtf['TFBS'].str.split(',')

    peaks_nearby_new = pd.DataFrame(np.ones((0,6)), columns=['Gene','Strand','chr','start','end','TFBS']) #ref_gtf.columns[[1,4,5,6,7,10]])
    for i in range(ref_gtf.shape[0]):
        n_lines = len(ref_gtf.loc[ref_gtf.index[i],'TFBS'])
        peaks_nearby_new_tmp = ref_gtf.iloc[np.repeat(i,n_lines),:]
        peaks_nearby_new_tmp = peaks_nearby_new_tmp.loc[:,['Gene','Strand','chr','start','end','TFBS']]
        peaks_nearby_new_tmp['TFBS'] = ref_gtf.loc[ref_gtf.index[i],'TFBS']

        peaks_nearby_new = pd.concat([peaks_nearby_new, peaks_nearby_new_tmp], axis=0)
        peaks_nearby_new.index = np.arange(peaks_nearby_new.shape[0])
    
    peaks_nearby_new['pstart'] = peaks_nearby_new['TFBS'].str.split(split_symbol[0]).str[1].str.split(split_symbol[1]).str[0]
    peaks_nearby_new['pstart'] = peaks_nearby_new['pstart'].astype(float)
    peaks_nearby_new['pend'] = peaks_nearby_new['TFBS'].str.split(split_symbol[0]).str[1].str.split(split_symbol[1]).str[1]
    peaks_nearby_new['pend'] = peaks_nearby_new['pend'].astype(float)

    peaks_nearby_new['overlap'] = peaks_nearby_new.apply(lambda x: min(x['pend'],x['end']) - max(x['pstart'],x['start']), axis=1)

    df = peaks_nearby_new.copy()
    df = df.drop(['pstart', 'pend'], axis=1)

    df.columns = ['Gene','Strand','chr','start','end','Peak','overlap']
    df = df.sort_values(by='overlap', ascending=False)
    df = df.loc[~df.iloc[:,[0,5]].duplicated(),:].copy()
    
    #df = df.iloc[:,[0,5,7,9,8,10]].copy()
    df.index = np.arange(df.shape[0])
    
    return df



#=========================================================================================
# Homer related tools
#=========================================================================================
def PFM2Motif(file_name, out_file, detection_threshold=0):
    file = open(file_name, 'r').readlines()

    df_dict = {}
    for i in range(1,5):
        line = file[i].strip('\n').split(' ')
        line = list(filter(lambda a: a != '', line))
        line.remove('[')
        line.remove(']')
        #line = line[1:]
        df_dict[line[0]] = [float(i) for i in line[1:]]
    df = pd.DataFrame(df_dict)
    df = df[['A','C','G','T']]
    df[df.columns] = df.to_numpy() / df.sum(axis=1).to_numpy()[:, np.newaxis]

    motif_seq = ''.join(df.columns[np.argmax(df, axis=1)])
    motif_header = file[0]
    motif_header = motif_header.strip('>')
    motif_header = motif_header.strip('\n').split('\t')
    motif_header = '_'.join(motif_header)
    motif_header = '>' + motif_seq + '\t' + motif_header + '\t0'

    motif_file = open(out_file,'w')
    motif_file.write(motif_header)
    for i in range(df.shape[0]):
        freq_list = df.iloc[i,:].tolist()
        freq_list = [str(round(j, 6)) for j in freq_list]
        motif_file.write('\n')
        motif_file.write('\t'.join(freq_list))
        
    motif_file.close()




def peak2HomerInput(peaks, out_file, filetype='peaks', split_symbol=['-','-']):

    if filetype == 'peaks':
        homer_df = pd.DataFrame({'index': np.arange(len(peaks)),
                                 'chrom': [i.split(split_symbol[0],1)[0] for i in peaks],
                                 'start': [i.split(split_symbol[0],1)[1].split(split_symbol[1])[0] for i in peaks],
                                 'end': [i.split(split_symbol[0],1)[1].split(split_symbol[1])[1] for i in peaks],
                                 'strand':'.'})

    elif filetype == 'bed':
        homer_df = pd.DataFrame({'chrom': [i.split(split_symbol[0],1)[0] for i in peaks],
                                 'start': [i.split(split_symbol[0],1)[1].split(split_symbol[1])[0] for i in peaks],
                                 'end': [i.split(split_symbol[0],1)[1].split(split_symbol[1])[1] for i in peaks],
                                 'index': np.arange(len(peaks)),
                                 'useless': '.',
                                 'strand':'.'})
    
    homer_df.to_csv(out_file, index=False, header=False, sep='\t')




def run_HOMER_motif(peaks, out_dir, prefix, ref_genome, 
                    homer_path=None, split_symbol=['-','-'], size=200):

    """
    Function to run Homer from Python script.
    It will prepare Homer required input file
    and output results in the directory specified

    Arguments
    ---------
        peaks: list or array-like, peaks of interests

        out_dir: output directory to save the results
    
        prefix: prefix of all files, folder called out_dir/homer_prefix will be created to save all the results

        ref_genome: str, reference genome name, e.g., 'hg19', 'hg38'

        homer_path: path to Homer software, if Homer already added to the PATH, argument can be ignored

        split_symbol: how peak location ID is merged
                      'chr1-12345-23456' - split_symbol=['-','-']
                      'chr1:12345-23456' - split_symbol=[':','-']
        
        size: Homer paramter, the size of the region used for motif finding


    Returns
    ---------
        Homer results saved in out_dir/homer_prefix
        Peak list DataFrame in BED format

    """

    homer_df = pd.DataFrame({'index': np.arange(len(peaks)),
                             'chrom': [i.split(split_symbol[0],1)[0] for i in peaks],
                             'start': [i.split(split_symbol[0],1)[1].split(split_symbol[1])[0] for i in peaks],
                             'end': [i.split(split_symbol[0],1)[1].split(split_symbol[1])[1] for i in peaks],
                             'strand':'.'})
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_file = os.path.join(out_dir, '%s.txt'%(prefix))
    print("Save peaks list in BED file format at %s"%out_file)
    homer_df.to_csv(out_file, index=False, header=False, sep='\t')

    homer_dir = os.path.join(out_dir, 'homer_%s'%(prefix))
    if not os.path.exists(homer_dir):
        print("Creat HOMER output folder at %s"%homer_dir)
        os.mkdir(homer_dir)


    if homer_path is None:
        try:
            os.system('findMotifsGenome.pl %s %s %s -size %s'%(out_file, ref_genome, homer_dir, size))
        except:
            print("HOMER run failed. The HOMER peak format DataFrame is returned.")
            return homer_df
    else:
        homer_cmd_path = os.path.join(homer_path,'bin/findMotifsGenome.pl')
        try:
            os.system('%s %s %s %s -size %s'%(homer_cmd_path, out_file, ref_genome, homer_dir, size))
        except:
            print("HOMER run failed. The HOMER peak format DataFrame is returned.")
            return homer_df

    print("HOMER finished successfully! Please check the HTML report for interesting motifs.")
    print("motif_summary can be run with the motif index for further analysis.")
    return homer_df




def motif_summary(peak_file, homer_dir, motif_index, ref_genome,
                  homer_path=None, size=200):

    """
    Function to extract related peaks from motifs of interests.


    Arguments
    ---------
        peak_file: out_dir/prefix.peaks.bed file generated in run_HOMER_motif()

        homer_dir: out_dir/homer_prefix in run_HOMER_motif() | Homer output folder
    
        motif_index: motif of interests index in homer_dir/knownResults.html

        ref_genome: str, reference genome name, e.g., 'hg19', 'hg38'

        homer_path: path to Homer software, if Homer already added to the PATH, argument can be ignored
        
        size: Homer paramter, the size of the region used for motif finding
              Keep the same as in run_HOMER_motif()


    Returns
    ---------
        Motif related peaks information saved in homer_dir/
        DataFrame containing peak list and motif matching quality information

    """

    motif_file = os.path.join(homer_dir, 'knownResults/known%s.motif'%motif_index)
    motif_out = os.path.join(homer_dir, 'motif%s.peaks.txt'%motif_index)
    if homer_path is None:
        try:
            os.system('findMotifsGenome.pl %s %s %s -find %s -size %s > %s'%(
                peak_file, ref_genome, homer_dir, motif_file, size, motif_out))
        except:
            print("HOMER run failed. Make sure homer_path is set and all file paths are right.")
            return
    else:
        homer_cmd_path = os.path.join(homer_path,'bin/findMotifsGenome.pl')
        try:
            os.system('%s %s %s %s -find %s -size %s > %s'%(
                homer_cmd_path, peak_file, ref_genome, homer_dir, motif_file, size, motif_out))
        except:
            print("HOMER run failed. Make sure homer_path is set and all file paths are right.")
            return

    print("HOMER finished successfully! Motif related peaks will be loaded and returned.")
    motif_peaks = pd.read_csv(motif_out,sep='\t')
    peak_df = pd.read_csv(peak_file, sep='\t',header=None)

    peaks = peak_df.loc[motif_peaks['PositionID'].tolist()]
    peaks = peaks[0] + '-' + peaks[1].astype(str) + '-' + peaks[2].astype(str)
    motif_peaks.insert(0, 'peaks', peaks.to_numpy())

    return motif_peaks










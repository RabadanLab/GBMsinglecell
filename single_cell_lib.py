import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import umap
import gseapy

from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def normalize(df, norm_constant = 10**6):
    """
    Performs TP(K/M) normalization, 
    log2(#*norm_constant/total_transcripts_in_cell + 1)

    Args:
        df: pandas df
            rows = cells, columns = genes
        norm_constant: int
            calculates transcripts per [norm_constant]

    Returns:
        normed pandas df
    """
    return np.log2(1 + df.mul(norm_constant/df.sum(axis = 1), axis=0))


def embed(df, components = 0.05, random = False, method = 'umap'):
    """
    PCA then UMAP onto normalized scRNA data.

    Args:
        df: pandas df
            rows = cells, columns = genes
        components: float between 0 and 1, or int
            what fraction of dimensions to keep for pca
        random: bool
            whether to randomize seed for UMAP
        method: ['umap', 'tsne']
            what embedding algorithm

    Returns:
        embedding as pandas df
    """
    pca = PCA(n_components = components if (components % 1 == 0) 
              else int(df.shape[1]*components))

    pca_data = pd.DataFrame(pca.fit_transform(df))

    print("First PC variance ratio: ", pca.explained_variance_ratio_[0])
    print("Total PCA variance ratio: ", sum(pca.explained_variance_ratio_))
    
    random_state = None if random else 42
    
    if method == 'umap':
        embedding = umap.UMAP(
            random_state = random_state).fit_transform(pca_data)
    elif method == 'tsne':
        tsne = TSNE(random_state = random_state)
        embedding = tsne.fit_transform(pca_data)
        
    return pd.DataFrame(embedding, 
        index = df.index)

def plot(embedding, gene_values=None, gene_list=None, show_batch=False, 
         batch_level = 0, ax=None, legend='first', cutoff=0, cmap = None,
         cmap_list = None, color_ceiling=1.0, **kwargs):
    """
    Plot embedding and gene expression or sample number
    
    Args:
        embedding: pandas df
            rows = cells, columns = position
        gene_values: pandas df
            rows = cells, columns = genes
        gene_list: nested list
            each entry is a list of genes
        show_batch: bool
            Sample origin for each cell (leave gene blank)
        batch_level: int
            which level of multi-index to use as batch
        ax: matplotlib axes
        legend: {'first', 'all'}
            for gene plot, only show first gene in each list 'first'
        cutoff: number
            don't plot expression for cells with less than cutoff
        cmap: colormap function
            for batch plotting
        cmap_list: list of cmaps
            for plotting multiple gene exp
        color_ceiling: float
            saturate colors by adjusting vmax of norm in plt.scatter
        kwargs: to pass to plotting method (partially implemented)

    Returns:
        if ax is None:
        fig, ax: matplotlib handles
    """    
    if ax is None:
        return_axes = True
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
    else:
        return_axes = False
        
    ax.get_xaxis().set_visible(False);
    ax.get_yaxis().set_visible(False);

    if 's' in kwargs.keys():
        point_size = kwargs['s']
    else:
        point_size = 1 + 10 /(1 + len(embedding)/1000)    
    
    embedding.plot.scatter(0,1, s=point_size, c='k', ax=ax, alpha=.2,
                           rasterized=True);
    
    if show_batch:
        batches = embedding.index.levels[batch_level]
        
        if cmap is None:
            if len(batches) <= 10:
                cmap = plt.cm.tab10 
            elif len(batches) <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = lambda x: plt.cm.gist_rainbow(x/len(batches))
        
        colors = cmap(embedding.index.codes[batch_level].values())
            
        embedding.plot.scatter(0,1, s=point_size, c=colors, 
                               ax=ax, alpha=.6, rasterized=True)
        legend_elements = [mpl.lines.Line2D(
            [0], [0], marker='o', linestyle=' ', 
            color=cmap(i), markersize=5) 
                           for i in range(len(batches))]
        ax.legend(legend_elements, batches, fontsize=11, 
                  ncol= int(np.ceil(len(batches)/10)))

    if gene_list is not None:
        
        if cmap_list is None: #default
            cmap_list = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens,
                         plt.cm.Purples, plt.cm.YlOrBr]
        counter = 0
        row_max = pd.concat([gene_values[gene].sum(axis=1) 
                             for gene in gene_list], 
                            axis=1).apply(lambda x: x == max(x), axis=1)
        
        for gene in gene_list:
            selected_cells = (gene_values[gene].sum(axis=1) 
                              > cutoff) & row_max[counter] #only max geneset
            color_values = gene_values[gene][selected_cells].sum(axis=1)
            ax.scatter(embedding[selected_cells][0],
                       embedding[selected_cells][1], 
                       s=point_size, c=color_values, cmap = cmap_list[counter],
                       vmax = color_ceiling*color_values.max(), alpha=.7, 
                       rasterized=True)
            counter += 1


        if legend != False:
            legend_elements = [mpl.lines.Line2D(
                [0], [0], marker='o', linestyle=' ', 
                color=cmap_list[i](.7), markersize=5) 
                               for i in range(len(gene_list))]
            ax.legend(legend_elements, 
                      gene_list if legend == 'all' 
                      else [gene[0] for gene in gene_list], 
                      loc='lower right', fontsize=14)

    
    if return_axes:
        return fig, ax
    else:
        return None, None
    
def cluster(embedding, gene_values, gene = None, manual_cluster = None, 
            method = 'dbscan', param = .4, comparison = False):
    """
    Plot clustering of cells via dbscan or kmeans, and select for clusters
    with overexpression of a given gene. Compares differential expression
    in the selected cluster through Welch's T-test.
    
    Args:
        embedding: pandas df
            rows = cells, columns = position
        method: {'dbscan', 'kmeans'}
            clustering method
        param: number
            main param for clustering method 
            (eps for dbscan, n_clusters for kmeans)
        gene_values: pandas df
            rows = cells, columns = genes
        gene: str
            gene to be expressed more than +1SD in selected pops
        comparison: bool
            whether to print other overexpressed genes
    Returns:
        cluster.labels_: list of ints of cluster for each cell
        diff_genes: differentially expressed genes in that cluster
    """    
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    ax.get_xaxis().set_visible(False);
    ax.get_yaxis().set_visible(False);
    
    if method == 'dbscan':
        cluster = DBSCAN(eps = param).fit(embedding)
    elif method == 'kmeans':
        cluster = KMeans(n_clusters = param, random_state=42).fit(embedding)

    embedding.plot.scatter(0,1, s=0.1, c=plt.cm.tab10(cluster.labels_),ax=ax,
                          rasterized=True);

    for i in set(cluster.labels_):
         ax.text(*embedding[cluster.labels_==i].mean(), 
                 str(i), fontdict={'size': 12})

    if gene is not None or manual_cluster is not None:
        
        if manual_cluster is None:
            gene_exp = gene_values[gene]
            cluster_exp = [gene_exp[cluster.labels_==i].mean() 
                           for i in set(cluster.labels_) if i >= 0]
            overexpression = (cluster_exp > np.mean(cluster_exp) 
                              + np.std(cluster_exp))
            selected_clusters = np.where(overexpression)[0]
        else:
            selected_clusters = manual_cluster
        print("Selected clusters: ", selected_clusters)     
    
        if comparison:
            selected_cells = np.isin(cluster.labels_, selected_clusters)
            welcht = stats.ttest_ind(gene_values[selected_cells],
                                     gene_values[~selected_cells], 
                                     equal_var=False)
            diff_genes = pd.Series(welcht[0], index = gene_values.columns)
            print('Low in selected: ', 
                  '%s' % ', '.join(map(str,(list(diff_genes.sort_values(
                      ascending=True).head(20).index)))))
            print('High in selected: ',              
                  '%s' % ', '.join(map(str,(list(diff_genes.sort_values(
                      ascending=False).head(20).index)))))

    
    return pd.Series(cluster.labels_, index=gene_values.index), \
            (diff_genes if comparison else None)

def signature_plot(embedding = None, gene_values = None, signature = None, 
                   second_signature = None, point_size = None,
                   labels = ['Macrophage-like', 'Microglia-like'],
                   method = 'nonparam', ylabel = 'Enrichment Score',
                   cmap = 'viridis', cbarloc = 'right', **kwargs):
    """
    Graph enrichment of gene list
    
    Args:
        embedding: pandas df
            rows = cells, columns = position. Leave blank to
            only compute rankings, no plots
        gene_values: pandas df
            rows = cells, columns = genes
        signature: list-like
            of genes
        second_signature: optional, list-like
            to compare against
        point_size: float
            size of points for scatter
        method: {'nonparam', 'gsea', 'none'}
            whether to use ssGSEA or a fully non-parametric
            system to assess enrichment. When using on paired
            sets, both will subtract enrichment scores. 
        ylabel: str
            for the colorbar
        cmap: cmap object
            colormap
        cbarloc: 'left' | 'right'
            where to put the colorbar
        kwargs:
            to pass to plt.scatter

    Returns:
        fig, ax: matplotlib handles, if embedding provided
        rankings: enrichment scores
    """       

    if signature is not None:
        signature1 = signature[signature.isin(gene_values.columns)]
    

    
    if method == 'gsea': #gseapy does not support multi-indexing
        saved_index = gene_values.index
        gene_values.index = range(gene_values.shape[0])
    
    if second_signature is None:
        if method == 'gsea':
            enrichment = gseapy.ssgsea(gene_values.T, 
                                       {'up': signature1}, 
                         outdir=None, min_size = 2,
                        no_plot=True, processes=11)
            rankings = pd.DataFrame(enrichment.resultsOnSamples).loc['up']
        elif method == 'nonparam':
            rankings = (gene_values[signature1]
                        .rank(pct=True).sum(1)).rank(pct=True)
        else:
            rankings = gene_values

    else: # subtract 2 signatures
        signature2 = second_signature[
            second_signature.isin(gene_values.columns)]

        if method == 'gsea':
            enrichment = gseapy.ssgsea(gene_values.T, 
                                       {'up': signature1,
                                       'down': signature2}, 
                         outdir=None, min_size = 2, 
                        no_plot=True, processes=11)
            rankings = pd.DataFrame(enrichment.resultsOnSamples)
            rankings = rankings.loc['up'] - rankings.loc['down']
        elif method == 'nonparam':
            rankings = (gene_values[signature1].rank(pct=True).sum(1) 
                        - gene_values[signature2].rank(pct=True).sum(1)
                       ).rank(pct=True)

    if method == 'gsea': #gseapy does not support multi-indexing
        gene_values.index = saved_index
        rankings.index = saved_index            
            
    if embedding is not None: #plot
        fig, ax = plt.subplots()
        fig.set_size_inches(9, 9)
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False);
        
        
        if point_size is None:
            point_size = 3 + 10 /(1 + len(embedding)/1000)
            
        scatter_plot = ax.scatter(embedding[0],embedding[1], 
                                  s=point_size, c=rankings, 
                                  rasterized=True,
                                  cmap = cmap, **kwargs)


        cbar = fig.colorbar(scatter_plot, ax=[ax], extend = 'both', fraction=0.04, 
                            pad=0.04, location = cbarloc)
        cbar.ax.set_ylabel(ylabel, fontsize = 16, 
                           rotation=270 if cbarloc is 'right' else 90, labelpad=20)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=12)

#         cmap_list = [ plt.cm.Greens, plt.cm.Purples]        
#         legend_elements = [mpl.lines.Line2D(
#             [0], [0], marker='o', linestyle=' ', 
#             color=cmap_list[i](.7), markersize=5) 
#                            for i in range(2)]
#         ax.legend(legend_elements, labels, 
#                   loc='upper left');
        return fig, ax, rankings

    else:
        return rankings

def plot_pvalue(data1, data2, ax, hloc, vloc, hlen = 0.02, 
                verbose = False, use_stars = True):
    """
    Mann-Whitney bar and star result
    
    Args:
        data1, data2: list-like of values
        ax: matplotlib ax
        hloc: [0,1] horizontal position
        vloc: [0,1] vertical position
        verbose: bool
            print p-value
        use_stars: bool
            whether to replace p-values

    Returns:
        str indicating significance
    """       
    pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')[1]
    if verbose:
        print(pvalue)
    if use_stars:    
        if pvalue < 0.001:
            text = u'\u2731\u2731\u2731'
        elif pvalue < 0.01:
            text = u'\u2731\u2731'
        elif pvalue < 0.05:
            text = u'\u2731'
        else:
            text = 'ns'
    else:
        #text = '{:.2g}'.format(pvalue)
        if pvalue < 0.001:
            text = '<0.001'
        elif pvalue < 0.01:
            text = '<0.01'
        else:
            text = '{:.2g}'.format(pvalue)
    ax.plot([hloc-hlen, hloc+hlen], [vloc,vloc], 'k', transform=ax.transAxes)
    ax.text(hloc, vloc+0.01, text, 
            horizontalalignment='center', transform=ax.transAxes)



def permute_enrich(df, selected_cells, gene_sets, method = 'ranksum', 
                   n_permutations = 10):
    """
    Calculates an enrichment p-value by permutation 
    of phenotypes
    
    Args:
        df: pandas df
            rows = cells, columns = genes: No 
        selected_cells: list of bools
            length of rows of df to be masked
        gene_sets: dict of gene sets
            {gs_name: [genes]}
        method: ['gsea', 'ranksum']
        n_permutations: int

    Returns:
        output: pandas df
            pvalue of each gene set
    """  
    
    
    output = []
    permuted_selection = np.copy(selected_cells)
    
    #first iteration calculates observed U-statistics
    for i in range(1 + n_permutations):
        
        welcht = stats.ttest_ind(df[permuted_selection],
                                 df[~permuted_selection], 
                                 equal_var=False)
        diff_genes = pd.Series(welcht[0], index = df.columns)
        if method == 'ranksum':
            es = {name: 
                  stats.mannwhitneyu(diff_genes[diff_genes.index.isin(gset)],
                                   diff_genes[~diff_genes.index.isin(gset)],
                                   alternative='two-sided')[0] #U-stat
                  for name, gset in gene_sets.items()}
        elif method == 'gsea':
            es = dict(gseapy.prerank(diff_genes, gene_sets, 
                                weighted_score_type=0, outdir=None,
                                permutation_num = 0, processes=11, 
                                no_plot=True).res2d['es'])

        output.append(pd.Series(es))
        np.random.shuffle(permuted_selection)
    output = pd.concat(output, axis=1)
    
    #compare permuted simulations to observed
    p_values = output.apply(lambda x: sum(x[1:]>x[0])/n_permutations, axis=1)
    
    #convert to two-tailed statistic
    return p_values.apply(lambda x: 2*x if x<0.5 else -2*(1-x))


#for gseaplot
class _MidpointNormalize(mpl.colors.Normalize):
    """ The MIT License (MIT) 
    Copyright (c) 2016-2017 Zhuoqing Fang"""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def gseaplot(rank_metric, term, hits_indices, RES,
              pheno_pos='', pheno_neg='', figsize=(6,5.5), 
              cmap='seismic', ofname=None, **kwargs):
    """This is the main function for reproducing the gsea plot, 
    copied from gseapy for further customization
    The MIT License (MIT) 
    Copyright (c) 2016-2017 Zhuoqing Fang

    :param rank_metric: pd.Series for rankings, rank_metric.values.
    :param term: gene_set name
    :param hits_indices: hits indices of rank_metric.index 
    presented in gene set S.
    :param RES: running enrichment scores.
    :param pheno_pos: phenotype label, positive correlated.
    :param pheno_neg: phenotype label, negative correlated.
    :param figsize: matplotlib figsize.
    :param ofname: output file name. If None, don't save figure 

    """
    # plt.style.use('classic')
    # center color map at midpoint = 0
    norm = _MidpointNormalize(midpoint=0)

    #dataFrame of ranked matrix scores
    x = np.arange(len(rank_metric))
    rankings = rank_metric.values
    # figsize = (6,6)
    phenoP_label = pheno_pos + ' (Positively Correlated)'
    phenoN_label = pheno_neg + ' (Negatively Correlated)'
    zero_score_ind = np.abs(rankings).argmin()
    z_score_label = 'Zero score at ' + str(zero_score_ind)
    im_matrix = np.tile(rankings, (2,1))

    # output truetype
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    # in most case, we will have many plots, so do not display plots
    # It's also usefull to run this script on command line.

    # GSEA Plots
    gs = plt.GridSpec(16,1)
    fig = plt.figure(figsize=figsize)

    # Ranked Metric Scores Plot
    ax1 =  fig.add_subplot(gs[11:])
    module = 'tmp' if ofname is None else ofname.split(".")[-2]
    if module == 'ssgsea':
        ax1.fill_between(x, y1=np.log(rankings), y2=0, color='#C9D3DB')
        ax1.set_ylabel("log ranked metric", fontsize=14)
    else:
        ax1.fill_between(x, y1=rankings, y2=0, color='#C9D3DB')
        ax1.set_ylabel("Ranked List Metric", fontsize=14)
    ax1.text(.05, .9, phenoP_label, color='red',
             horizontalalignment='left', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(.95, .05, phenoN_label, color='Blue',
             horizontalalignment='right', verticalalignment='bottom',
             transform=ax1.transAxes)
    # the x coords of this transformation are data, and the y coord are axes
    trans1 = mpl.transforms.blended_transform_factory(ax1.transData, 
                                                      ax1.transAxes)
    if module != 'ssgsea':
        ax1.vlines(zero_score_ind, 0, 1, linewidth=.5, transform=trans1,
                   linestyles='--', color='grey')
        ax1.text(zero_score_ind, 0.5, z_score_label,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=trans1)
    ax1.set_xlabel("Rank in Ordered Dataset", fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(axis='both', which='both', top=False, right=False, 
                    left=False)
    ax1.locator_params(axis='y', nbins=5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda tick_loc, tick_num: '{:.1f}'.format(tick_loc) ))

    # use round method to control float number
    # ax1.yaxis.set_major_formatter(plt.FuncFormatter(
    # lambda tick_loc,tick_num :  round(tick_loc, 1) ))

    # gene hits
    ax2 = fig.add_subplot(gs[8:10], sharex=ax1)

    # the x coords of this transformation are data, and the y coord are axes
    trans2 = mpl.transforms.blended_transform_factory(ax2.transData, 
                                                      ax2.transAxes)
    ax2.vlines(hits_indices, 0, 1,linewidth=.5,transform=trans2)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, right=False, left=False, 
                    labelleft=False)
    # colormap
    ax3 =  fig.add_subplot(gs[10], sharex=ax1)
    ax3.imshow(im_matrix, aspect='auto', norm=norm, cmap=cmap, 
               interpolation='none') # cm.coolwarm
    ax3.spines['bottom'].set_visible(False)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, right=False, left=False,labelleft=False)

    # Enrichment score plot
    ax4 = fig.add_subplot(gs[:8], sharex=ax1)
    ax4.plot(x, RES, linewidth=4, color ='#88C544')

    # the y coords of this transformation are data, and the x coord are axes
    trans4 = mpl.transforms.blended_transform_factory(ax4.transAxes, 
                                                      ax4.transData)
    ax4.hlines(0, 0, 1, linewidth=.5, transform=trans4, color='grey')
    ax4.set_ylabel("Enrichment Score (ES)", fontsize=14)
    ax4.set_xlim(min(x), max(x))
    ax4.tick_params(axis='both', which='both', bottom=False, top=False, 
                    labelbottom=False, right=False)
    ax4.locator_params(axis='y', nbins=5)
    # FuncFormatter need two argument, I don't know why. 
    # this lambda function used to format yaxis tick labels.
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda tick_loc, tick_num : '{:.1f}'.format(tick_loc)) )

    # fig adjustment
    fig.suptitle(term, fontsize=16, fontweight='bold')
    fig.subplots_adjust(hspace=0)
    # fig.tight_layout()
    if ofname is not None: 
        # canvas.print_figure(ofname, bbox_inches='tight', dpi=300)
        fig.savefig(ofname, bbox_inches='tight', dpi=300)
    return fig
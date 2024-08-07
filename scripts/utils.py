import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy import stats
from scipy.stats import fisher_exact, ranksums, ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import category_encoders as ce
from multiprocessing import Pool
from multiprocessing import cpu_count
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
from scipy.stats import median_abs_deviation
from itertools import product
import itertools

from BTC_Cell_Line_Atlas_Manuscript.scripts.io_library import MyLib
from BTC_Cell_Line_Atlas_Manuscript.scripts.visualization import MyVisualization
from BTC_Cell_Line_Atlas_Manuscript.scripts.clustering_utils import Clustering


def compute_corr(args):
    """
        For the parallel processing of the correlation test
    """
    df1, df2, col = args
    return df1.corrwith(df2[col])

class Utils:
    subtype_dic = {'AC': "#c9bb3a", 'ECC': "#35978f", 'GBC': "#9970ab",
                   'HCC': "#fa9fb5", 'ICC': "#bf812d"}

    @staticmethod
    def get_cell_lines_by_fusion(fusion_df, upset_df, gene):
        df = fusion_df[fusion_df['Gene'].str.contains(gene)].set_index('Gene').T
        cell_lines = df.index[(df.astype(bool).any(axis=1)) & df.index.isin(upset_df.index)]
        return cell_lines.tolist()

    @staticmethod
    def get_cell_lines_by_cnv(cnv_df, upset_df, gene):
        df = cnv_df[cnv_df['Gene']==gene]
        gp_df = df.groupby('CN')
        res_dic = {}
        for cn, cur_df in gp_df:
            cell_lines_s = set(cur_df['Sample_name'].tolist())
            res_dic[cn] = sorted(cell_lines_s.intersection(upset_df.index.tolist()))
        return res_dic

    @staticmethod
    def get_cell_lines_by_hotspot_dam_mutation(gene, hotspot_df, dam_df, upset_df):
        if gene in dam_df.columns:
            dam_df = dam_df.loc[upset_df.index, gene]
        else:
            dam_df = pd.Series(0, index=upset_df.index, name=gene)
        if gene in hotspot_df.columns:
            hotspot_df = hotspot_df.loc[upset_df.index, gene]
        else:
            hotspot_df = pd.Series(0, index=upset_df.index, name=gene)
        df = hotspot_df | dam_df
        upset_df[gene] = df
        return upset_df

    @staticmethod
    def match_rows_df(df1, df2):
        df = pd.merge(df1, df2, how='inner', left_index=True, right_index=True, sort=True)
        new_df1 = df1.loc[df.index, :]
        new_df2 = df2.loc[df.index, :]

        return (new_df1, new_df2)

    @staticmethod
    def match_columns_df(data_df1, data_df2):
        matched_features = sorted(set(data_df1.columns.tolist()).intersection(set(data_df2.columns.tolist())))
        # print('number of shared columns: ', len(matched_features))
        return data_df1[matched_features], data_df2[matched_features]

    @staticmethod
    def pdx_cell_line_distance(pdx_df, cl_df, save_figure=False):
        k = pdx_df.shape[0]
        knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn_model.fit(cl_df)
        distances, indices = knn_model.kneighbors(pdx_df)
        df = pd.DataFrame({'Cell_line':cl_df.index, 'PDX':pdx_df.index, 'K':np.nan})
        for i, pdx_sample in enumerate(pdx_df.index):
            df.loc[i, 'K'] = np.where(indices[i] == i)[0]+1
            # cl_df.iloc[indices[i], :].index
        df = df.sort_values(by=['K'], ascending=False)
        df['K'] = df['K'].astype(int)
        # title ='Dispersion of $K$ Nearest Cell Line Values in Cell Line-PDX Pairs'
        # MyVisualization.plot_histogram(df['K'], ytick_step=1, xtick_step=1, xlim=23, ylim=17,
        #                                xlabel=f'$K$', ylabel='Count', title=title)
        title = f'Dispersion of $K$ Nearest Cell Line Values'
        MyVisualization.plot_catplot(df[['K']], x_col='K', ytick_step=2, ylim=18, title_height=.98, color_code='gray',
                                       xlabel=f'$K$', ylabel='Count', title=title,
                                     figure_width=3, figure_height=4,
                                     save_figure=save_figure)
        df.set_index('Cell_line', inplace=True)
        return df

    @staticmethod
    def scatter_pdx_and_cell(pdx_df, cl_df, k_df, upset_df, size_dic, save_figure=False):

        pdx_df.index = cl_df.index + '_PDX'
        df = pd.DataFrame(index=cl_df.index, columns=['Correlation', 'Distance'])
        df['Subtype'] = upset_df.loc[df.index, 'Subtype']

        df['K'] = k_df['K'].apply(lambda x: size_dic[x])
        # max_k = k_df.max().values[1]
        # df['K'] = k_df['K'].apply(lambda x: (max_k-x+1)/max_k*100)
        for index, row in cl_df.iterrows():
            corr, _ = pearsonr(cl_df.loc[index, :], pdx_df.loc[index + '_PDX', :])
            df.loc[index, 'Correlation'] = corr
            df.loc[index, 'Distance'] = np.linalg.norm(cl_df.loc[index, :] - pdx_df.loc[index + '_PDX', :])
        df['colors'] = df['Subtype'].apply(lambda x: Utils.subtype_dic[x])

        MyVisualization.scatter_plot_comparison(X_data_sr=df['Correlation'], Y_data_sr=df['Distance'], force_text=.6,
                                                legend_anchor_x=.9, legend_anchor_y=1, points_size=df['K'],
                                                legend_label='Subtype',
                                                x_label='Correlation', y_label='Distance',
                                                title='Cell Line - PDX pairs', text_fontsize=11,
                                                colors_sr=df['colors'], legend_classes_dic=Utils.subtype_dic,
                                                legend_marker_size=90,
                                                legend_anchor_x_offset=.001, ylim_top=160, ylim_bottom=80,
                                                xlim_right=0.65,
                                                tick_fontsize=13, add_text=True, label_fontsize=13,
                                                legend_size_dic=size_dic,
                                                legend_num=list(size_dic.values()),
                                                save_file=save_figure)
        return df[['Correlation', 'Distance']].copy()

    @staticmethod
    def prepare_data_for_hue_plots(df, in_group_df, label1, label2, id_vars='index', test='ttest', reverse_groups=False):
        df1 = df[df.index.isin(in_group_df.index)]
        df2 = df[~df.index.isin(in_group_df.index)]
        if test == 'ranksum':
            _, p_value = ranksums(df1, df2, nan_policy='omit')
        else:
            _, p_value = ttest_ind(df1, df2, nan_policy='omit')

        df1_melt = pd.melt(df1.reset_index(), id_vars=id_vars)
        df1_melt['Group'] = label1
        df2_melt = pd.melt(df2.reset_index(), id_vars=id_vars)
        df2_melt['Group'] = label2

        if reverse_groups:
            merged_df = pd.concat([df2_melt, df1_melt])
        else:
            merged_df = pd.concat([df1_melt, df2_melt])
        return (merged_df, p_value[0])

    @staticmethod
    def fisher_test(data_df, x_col, y_col):
        pairs_dic = dict()
        for var1, var2 in itertools.combinations(sorted(data_df[x_col].unique()), 2):
            df = data_df[data_df[x_col].isin([var1, var2])].copy()
            contingency_table = pd.crosstab(df[x_col], df[y_col])
            odds_ratio, p_value = fisher_exact(contingency_table)
            pairs_dic[(var1, var2)] = p_value
            print(contingency_table)
            print("Odds Ratio:", odds_ratio)
            print("P-value:", p_value)

            if p_value < 0.05:
                print(f"There is a significant association between {var1} and {var2}")
            else:
                print(f"There is no significant association between {var1} and {var2}")
            print('===============================')
        return pairs_dic

    @staticmethod
    def visualize_feature_importance(gene, data_dir, featurs_df, targets_df, summary_df, feat_cutoff=10,
                                     wspace=0.3, save_figure=False):

        print(gene)
        file_name = None
        if save_figure:
            file_name = 'RF_' + gene
        corr = round(summary_df.loc[gene, 'corr'], 2)
        title = f"{gene.split(' (')[0]}"
        res_df = pd.read_csv(os.path.join(data_dir, f'RF_{gene}.csv'), skiprows=3, index_col=0, low_memory=False)
        scores_sr = res_df.iloc[0, :].astype('float64')
        total_importance = scores_sr.sum()
        percentage_sr = scores_sr.iloc[:feat_cutoff].apply(
            lambda importance: round((importance / total_importance) * 100, 1))
        data_df = pd.DataFrame({'importance': percentage_sr})
        # print(data_df.index)
        data_df['corr'] = data_df.index.map(
            lambda feat: round(np.corrcoef(featurs_df[feat], targets_df[gene])[0, 1], 2))
        data_df.index = data_df.index.map(
            lambda gene: gene.split('_')[0] + ' (' + gene.split('_')[-1] + ')')
        data_df = data_df.sort_values(by='importance', ascending=True)
        MyVisualization.plot_feature_importance(data_df, title=title, model_accuracy=f'{corr}', wspace=wspace,
                                                file_name=file_name)

    @staticmethod
    def get_feature_importance(gene, data_dir, summary_df, feat_cutoff=5):

        corr = round(summary_df.loc[gene, 'corr'], 2)
        res_df = pd.read_csv(os.path.join(data_dir, f'RF_{gene}.csv'), skiprows=3, index_col=0, low_memory=False)
        scores_sr = res_df.iloc[0, :].astype('float64')
        total_importance = scores_sr.sum()
        percentage_sr = scores_sr.iloc[:feat_cutoff].apply(
            lambda importance: round((importance / total_importance) * 100, 1))
        data_df = pd.DataFrame({'importance': percentage_sr})
        data_df.index = data_df.index.map(
            lambda gene: gene.split('_')[0] + ' (' + gene.split('_')[-1] + ')')
        data_df = data_df.sort_values(by='importance', ascending=False)
        return (data_df, corr)

    @staticmethod
    def cohend_effect_size(df1, df2):
        n1 = len(df1)
        n2 = len(df2)
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(df1, ddof=1) + (n2 - 1) * np.var(df2, ddof=1)) / (n1 + n2 - 2)
        )
        mean_difference = np.mean(df1) - np.mean(df2)
        effect_size = mean_difference / pooled_std
        return effect_size

    @staticmethod
    def apply_t_test(df1, df2, save_file=''):
        res_df = pd.DataFrame(index=df1.columns, columns=['stat', 'p-value', 'effect_size'], dtype='float')
        res_df['feature'] = df1.columns.values
        res_df['feature'] = res_df['feature'].apply(lambda x: x.split(' (')[0])
        for col in df2.columns:
            res_df.loc[col, 'stat'], res_df.loc[col, 'p-value'] = ttest_ind(df1[col], df2[col], nan_policy='omit')
            res_df.loc[col, 'effect_size'] = Utils.cohend_effect_size(df1[col], df2[col])
        res_df['q-value'] = fdrcorrection(res_df['p-value'].values)[1]

        in_group_data_sr = df1.mean()
        in_group_data_sr.name = 'in_group_gene_effect_mean'

        out_group_data_sr = df2.mean()
        out_group_data_sr.name = 'out_group_gene_effect_mean'

        res_df.index = res_df['feature']
        res_df = pd.concat([res_df, in_group_data_sr, out_group_data_sr], axis=1)
        res_df.reset_index(inplace=True, drop=True)
        res_df = res_df.sort_values('effect_size', ascending=False).sort_values('q-value', ascending=True)
        if save_file != '':
            MyLib.save_csv(res_df, save_file)
        return res_df

    @staticmethod
    def apply_ranksums_test(df1, df2, save_file=''):
        res_df = pd.DataFrame(index=df1.columns, columns=['feature', 'stat', 'p-value', 'effect_size'], dtype='float')
        res_df['feature'] = df1.columns.values
        res_df['feature'] = res_df['feature'].apply(lambda x: x.split(' (')[0])
        for col in df2.columns:
            res_df.loc[col, 'stat'], res_df.loc[col, 'p-value'] = ranksums(df1[col], df2[col], nan_policy='omit')
            res_df.loc[col, 'effect_size'] = res_df.loc[col, 'stat'] / np.sqrt(len(df1) + len(df2))
        res_df['q-value'] = fdrcorrection(res_df['p-value'].values)[1]

        in_group_data_sr = df1.mean()
        in_group_data_sr.name = 'in_group_mean'

        out_group_data_sr = df2.mean()
        out_group_data_sr.name = 'out_group_mean'

        res_df.index = res_df['feature']
        res_df = pd.concat([res_df, in_group_data_sr, out_group_data_sr], axis=1)
        res_df.reset_index(inplace=True, drop=True)
        res_df = res_df.sort_values('effect_size', ascending=False).sort_values('q-value', ascending=True)
        if save_file != '':
            MyLib.save_csv(res_df, save_file)
        return res_df

    @staticmethod
    def ranksums_test_all_pairs(data_df, upset_df, data_col, cluster_col, q_val_thr=0.1):
        print('\n----Pairwise Ranksums test----\n')
        pairs_dic = dict()
        for c1, c2 in itertools.combinations(sorted(upset_df[cluster_col].unique()), 2):
            c1_sr = data_df[upset_df[cluster_col] == c1][data_col]
            c2_sr = data_df[upset_df[cluster_col] == c2][data_col]
            p_value = ranksums(c1_sr, c2_sr)[1]
            pairs_dic[(c1, c2)] = p_value

        rows = [(key[0], key[1], value) for key, value in pairs_dic.items()]
        res_df = pd.DataFrame(rows, columns=[f"{cluster_col}_1", f"{cluster_col}_2", 'p-value'])
        res_df['q-value'] = fdrcorrection(res_df['p-value'].values)[1]
        res_df['Status'] = res_df['q-value'].apply(lambda x: 'significant' if x < q_val_thr else 'ns')
        res_df = res_df.sort_values('q-value', ascending=True)
        res_df.reset_index(inplace=True, drop=True)
        return res_df

    @staticmethod
    def filter_statistical_results(res_df, stat_col, q_val_thr=0.1, down_regulated=False, up_regulated=False):
        p_val_thr = 0.05
        res_filtered_df = res_df[res_df.apply(lambda x: (x['p-value'] < p_val_thr and
                                                         x['q-value'] < q_val_thr), axis=1)]
        down_df = None
        up_df = None
        if down_regulated:
            down_df = res_filtered_df[res_filtered_df[stat_col] < 0].sort_values(stat_col, ascending=True) \
                .sort_values('q-value', ascending=True)
        if up_regulated:
            up_df = res_filtered_df[res_filtered_df[stat_col] >= 0].sort_values(stat_col, ascending=False) \
                .sort_values('q-value', ascending=True)
        return (down_df, up_df)

    @staticmethod
    def apply_correlation_test_sr(sr, df, save_file=''):
        # df = Utils.drop_low_vars(df)
        df = df.loc[:, (df != 0).any(axis=0)]
        res_df = pd.DataFrame(index=df.columns, columns=['feature', 'corr', 'p-value'], dtype='float')
        res_df['feature'] = df.columns.values
        res_df['feature'] = res_df['feature'].apply(lambda x: x.split(' (')[0])
        res_df[['corr', 'p-value']] = df.corrwith(sr, method=lambda x, y: stats.pearsonr(x, y)).T
        # res_df[['corr', 'p-value']] = df.corrwith(sr, method=lambda x, y: stats.spearmanr(x, y)).T
        res_df['q-value'] = fdrcorrection(res_df['p-value'].values)[1]
        res_df = res_df.sort_values(by='corr', ascending=False)
        if save_file != '':
            MyLib.save_csv(res_df, save_file)
        return res_df

    @staticmethod
    def apply_correlation_paired_dfs(df1, df2, save_file=''):
        """
            df1 and df2 should have the same columns
        """
        # df = Utils.drop_low_vars(df)
        # df = df1.loc[:, (df1 != 0).any(axis=0)]
        res_df = pd.DataFrame(index=df1.columns, columns=['feature', 'corr', 'p-value'], dtype='float')
        res_df['feature'] = df1.columns.values
        for col in df2.columns:
            temp_df = pd.concat([df1[col], df2[col]], axis=1)
            temp_df.columns = [1,2]
            temp_df = temp_df.dropna()
            corr, pval = stats.pearsonr(temp_df[1], temp_df[2])
            res_df.loc[col, 'corr'] = corr
            res_df.loc[col, 'p-value'] = pval
        res_df['q-value'] = fdrcorrection(res_df['p-value'].values)[1]
        res_df = res_df.sort_values(by='corr', ascending=False)
        if save_file != '':
            MyLib.save_csv(res_df, save_file)
        return res_df

    @staticmethod
    def apply_correlation_dfs(df1, df2):
        """
            computes Pearson's correlation coefficient between columns of df1 and df2
        """
        df1, df2 = Utils.match_rows_df(df1, df2)
        pairs_col = list(product(df1.columns, df2.columns))
        res_df = pd.DataFrame(index=pairs_col, columns=['corr', 'p-value'])
        for col in df2.columns:
            cur_res_df = Utils.apply_correlation_test_sr(df2[col], df1)
            cur_indices = list(product(df1.columns, [col]))
            res_df.loc[cur_indices, 'corr'] = cur_res_df.loc[df1.columns, 'corr'].values
            res_df.loc[cur_indices, 'p-value'] = cur_res_df.loc[df1.columns, 'p-value'].values
        res_df['corr'] = res_df['corr'].astype(float)
        res_df['p-value'] = res_df['p-value'].astype(float)
        return res_df

    @staticmethod
    def pearson_corr(df1, df2):
        """
        Computes Pearson's correlation coefficient between columns of df1 and df2 using parallel processing.
        """
        stat_val_df = pd.DataFrame(index=df1.columns, columns=df2.columns)

        with Pool(cpu_count()) as pool:
            # Create argument list for parallel processing
            args_list = [(df1, df2, col) for col in df2.columns]
            results = pool.map(compute_corr, args_list)

        for col, result in zip(df2.columns, results):
            stat_val_df[col] = result

        return stat_val_df

    @staticmethod
    def pearson_corr_sequential(df1, df2):
        """
            computes Pearson's correlation coefficient between columns of df1 and df2
        """
        stat_val_df = pd.DataFrame(index=df1.columns, columns=df2.columns)
        p_val_df = pd.DataFrame(index=df1.columns, columns=df2.columns)
        for col in df2.columns:
            stat_val_df[col] = df1.corrwith(df2[col], method=lambda x, y: stats.pearsonr(x, y)[0]).values
            p_val_df[col] = df1.corrwith(df2[col], method=lambda x, y: stats.pearsonr(x, y)[1]).values

        return (stat_val_df, p_val_df)

    @staticmethod
    def statistical_test_and_volcano_plot(data_df1, data_df2, col=None, res_df=None, test='ttest', q_value=0.1,
                                          selected_genes_dic=None, alpha_top=1,
                                          xtick_step=None, ytick_step=None, title=None, xlim_right=None, ylim_top=None,
                                          ylim_bottom=None, xlim_left=None, cut_off_labels=5, y_thr_qval=None,
                                          add_label=True, yspine_center=False, plot_qqplot=False, plot_volcano=True,
                                          down_regulated=True, up_regulated=True, force_points=5, force_text=.9,
                                          add_arrow=True, axis_label_fontsize=None, title_fontsize=None,
                                          xlabel='', down_color=None, up_color=None, save_file='', save_sig=False,
                                          figure_width=3, figure_height=3, save_figure=False, return_sig=False):
        color_col = None
        # down_df = None
        up_df = None
        top_df = None

        if test == 'ttest':
            df1, df2 = Utils.match_columns_df(data_df1, data_df2)
            res_df = Utils.apply_t_test(df1, df2, save_file=save_file)
            x_col = 'effect_size'
            down_df, _ = Utils.filter_statistical_results(res_df, x_col, q_val_thr=q_value,
                                                          down_regulated=down_regulated)
        elif test == 'ranksum':
            df1, df2 = Utils.match_columns_df(data_df1, data_df2)
            res_df = Utils.apply_ranksums_test(df1, df2, save_file=save_file)
            x_col = 'effect_size'
            down_df, up_df = Utils.filter_statistical_results(res_df, x_col, q_val_thr=q_value,
                                                          down_regulated=down_regulated, up_regulated=up_regulated)
        else:  # test == 'correlation':
            if res_df is None:
                df1, df2 = Utils.match_rows_df(data_df1, data_df2)
                if col is not None:
                    res_df = Utils.apply_correlation_test_sr(df1[col], df2, save_file=save_file)
                else:
                    res_df = Utils.apply_correlation_paired_dfs(df1, df2, save_file=save_file)
            x_col = 'corr'
            down_df, up_df = Utils.filter_statistical_results(res_df, x_col, q_val_thr=q_value,
                                                              down_regulated=down_regulated, up_regulated=up_regulated)
        if y_thr_qval is not None:
            res_df = res_df[res_df['q-value'] < y_thr_qval]
            if down_df is not None:
                down_df = down_df[down_df['q-value'] < y_thr_qval]
            if up_df is not None:
                up_df = up_df[up_df['q-value'] < y_thr_qval]
        if selected_genes_dic is not None:
            color_col = 'color'
            res_df[color_col] = np.nan
            temp_df = res_df[res_df['feature'].isin(selected_genes_dic.keys())]
            res_df.loc[temp_df.index, color_col] = list(selected_genes_dic.values())
            if down_df is not None:
                top_df = down_df[~down_df['feature'].isin(selected_genes_dic.keys())].head(cut_off_labels)
            if up_df is not None:
                up_top_df = up_df[~up_df['feature'].isin(selected_genes_dic.keys())].head(cut_off_labels)
                top_df = pd.concat([top_df, up_top_df], axis=0)
            if down_color is None:  # If down_color is not provided, it means we don't need to distinguish lower significant points.
                down_df = None
            if up_color is None:
                up_df = None
        if save_sig:
            if down_df is not None and down_df.shape[0]>0:
                MyLib.save_csv(down_df, f'{title}_down.csv', index=False)
            if up_df is not None and up_df.shape[0]>0:
                MyLib.save_csv(up_df, f'{title}_up.csv', index=False)
        if plot_qqplot:
            MyVisualization.qqplot(res_df['p-value'], title=title)
        if plot_volcano:
            MyVisualization.Volcano_plot(res_df, x_col=x_col, y_col='q-value', color_col=color_col, title=title,
                                     x_label=xlabel, label_col='feature', down_df=down_df, up_df=up_df,
                                     force_text=force_text, axis_label_fontsize=axis_label_fontsize,
                                     down_color=down_color, up_color=up_color, cut_off_labels=cut_off_labels,
                                     add_label=add_label, title_fontsize=title_fontsize,
                                     xtick_step=xtick_step, ytick_step=ytick_step, xlim_left=xlim_left,
                                     yspine_center=yspine_center, top_df=top_df, alpha_top=alpha_top,
                                     force_points=force_points, add_arrow=add_arrow, xlim_right=xlim_right,
                                     ylim_top=ylim_top, ylim_bottom=ylim_bottom,
                                     figure_width=figure_width, figure_height=figure_height, save_figure=save_figure)
        if return_sig:
            return (down_df, up_df)
        return res_df

    @staticmethod
    def compare_visualize_cat_features(df, x, y, color_dic, y_label, x_label='', title=' ', test='ttest',
                                       label_fontsize=8,
                                       tick_fontsize=7, ylim_top=None, ylim_bottom=None, ytick_step=None,
                                       title_height=1., plot='swarmp',
                                       title_fontsize=10, tick_rotation=0, label_newline=True, box_plot_fliers=False,
                                       hline_type='mean', points_size=3, figure_width=3, figure_height=4, file_suffix='',
                                       save_figure=False):
        p_value = None
        if test is not None:
            if test == 'ranksum':
                stat, p_value = ranksums(df[df[x] == list(color_dic.keys())[0]][y],
                                         df[df[x] == list(color_dic.keys())[1]][y], nan_policy='omit')
            else:
                stat, p_value = stats.ttest_ind(df[df[x] == list(color_dic.keys())[0]][y],
                                                df[df[x] == list(color_dic.keys())[1]][y])
            print(f'{test} p-value={p_value}')
            if p_value < 0.05:
                print(
                    f"There is a significant association.")
            else:
                print(
                    f"There is no significant association.")
        if plot == 'swarmp':
            MyVisualization.swarm_plot(df, x_col=x, y_col=y, p_value=p_value, y_label=y_label, x_label=x_label,
                                       labels_fontsize=label_fontsize, title_fontsize=title_fontsize,
                                       title=title, palette_dic=color_dic, hue_col=x, ylim_top=ylim_top,
                                       ylim_bottom=ylim_bottom, hline_type=hline_type, label_newline=label_newline,
                                       star_pval=True, point_size=points_size, tick_fontsize=tick_fontsize,
                                       ytick_step=ytick_step, tick_rotation=tick_rotation,
                                       title_height=title_height, figure_width=figure_width,
                                       figure_height=figure_height, save_figure=save_figure, file_suffix=file_suffix)
        elif plot == 'violin':
            MyVisualization.violin_plot(df, x_col=x, y_col=y, color_dic=color_dic, p_value=p_value, y_label=y_label,
                                        x_label=x_label,
                                        title=title, ylim_top=ylim_top, ylim_bottom=ylim_bottom, ytick_step=ytick_step,
                                        star_pval=True, labels_fontsize=label_fontsize,
                                        title_height=title_height, alpha=1, showfliers=box_plot_fliers,
                                        figure_width=figure_width, figure_height=figure_height, save_figure=save_figure)

    @staticmethod
    def prepare_dorothea_results(dorothea_df):

        dorothea_df['q-value'] = fdrcorrection(dorothea_df['p_value'].values)[1]
        dorothea_df = dorothea_df.sort_values('q-value', ascending=True)
        dorothea_df['feature'] = dorothea_df['source'] + '--' + dorothea_df['condition']
        MyLib.save_csv(dorothea_df, 'dorothea_res.csv', index=False)

        dorothea_mat_df = dorothea_df.pivot(index='source', columns='condition', values='score')
        dorothea_mat_df = zscore(dorothea_mat_df, axis=0)
        return dorothea_mat_df

    @staticmethod
    def piechart(df):
        total_mutations = df.sum().sum()
        categories = df.sum()
        categories['WT'] = df.shape[0] - total_mutations
        mutation_percentages = (categories / df.shape[0]) * 100

        labels = mutation_percentages.index
        sizes = mutation_percentages.values
        explode = [0.1]*(df.shape[1]+1)
        plt.figure(figsize=(2, 3))
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140,
                        textprops={'fontsize': 5})
        plt.axis('equal')
        plt.show()

    @staticmethod
    def split_mutation_profiles(ssgsea_mut_df, R3_muts_l, R4_muts_l, color_dic, title='', file_name=None):
        x_col = 'High_ssGSEA'
        y_col = 'Mutation_Signature'
        ssgsea_mut_df[y_col] = 'WT'
        final_color_dic = dict()
        final_color_dic['WT'] = color_dic['WT']
        for id, row in ssgsea_mut_df.iterrows():
            has_R3_muts = row[R3_muts_l].eq(1).any()
            has_R4_muts = row[R4_muts_l].eq(1).any()
            if has_R3_muts.any() and has_R4_muts.any():
                ssgsea_mut_df.loc[id, y_col] = 'R3 & R4'
                final_color_dic['R3 & R4'] = color_dic['R3 & R4']
            elif has_R3_muts.any():
                ssgsea_mut_df.loc[id, y_col] = 'R3'
                final_color_dic['R3'] = color_dic['R3']
            elif has_R4_muts.any():
                ssgsea_mut_df.loc[id, y_col] = 'R4'
                final_color_dic['R4'] = color_dic['R4']

        final_color_dic = {key: color_dic[key] for key in color_dic if key in final_color_dic}  # reorder elements

        crosstab = pd.crosstab(index=ssgsea_mut_df[x_col],
                               columns=ssgsea_mut_df[y_col])
        print(crosstab)
        print('-------------------')
        temp_df = ssgsea_mut_df.copy()
        temp_df.loc[temp_df[y_col] == 'R3 & R4', y_col] = 'R4'
        temp_df.loc[temp_df[y_col] == 'R3', y_col] = 'WT'
        res1 = Utils.fisher_test(temp_df, x_col=y_col, y_col=x_col)
        print(res1)
        print('-------------------')
        temp_df = ssgsea_mut_df.copy()
        temp_df.loc[temp_df[y_col] == 'R3 & R4', y_col] = 'R3'
        temp_df.loc[temp_df[y_col] == 'R4', y_col] = 'WT'
        res2 = Utils.fisher_test(temp_df, x_col=y_col, y_col=x_col)
        print(res2)
        print('-------------------')
        # if list(color_dic.keys())[0] == 'R4':
        #     p_value = list(res1.values())[0]
        # else:
        p_value = [list(res2.values())[0]]
        p_value.append(list(res1.values())[0])
        MyVisualization.percentage_stacked_barplot(ssgsea_mut_df, x_col=x_col, y_col=y_col, title=title,
                                                   p_value=p_value, ylabel='Samples',
                                                   y_pval=.2, legend_title="Mutation Signature",
                                                   color_dic=final_color_dic,
                                                   figure_width=1.5, figure_height=2, file_name=file_name)

    @staticmethod
    def create_onehot_encoding_columns(data_df, cols, drop_base_variable=True, reference_ascending=True):
        encoder = ce.OneHotEncoder(use_cat_names=True)
        df = pd.DataFrame(index=data_df.index)
        for col in cols: # iterate one by one to handle NA values
            print(col)
            temp_df = data_df[col].dropna()
            temp_df = encoder.fit_transform(temp_df)
            temp_df = temp_df.sort_index(axis=1, ascending=reference_ascending)
            if temp_df.shape[1]<2:
                print(temp_df.shape)
                continue
            if drop_base_variable:
                if f'{col}_other' in temp_df.columns:
                    print(f'Reference level: other')
                    temp_df.drop(f'{col}_other', axis=1, inplace=True)
                elif f'{col}_Unknown' in temp_df.columns:
                    print(f'Reference level: Unknown')
                    temp_df.drop(f'{col}_Unknown', axis=1, inplace=True)
                else:
                    print(f'Reference level: {temp_df.columns[0]}')
                    temp_df = temp_df.iloc[:,1:]
            print('------')
            df = pd.concat([df, temp_df], axis=1)
        return df

    @staticmethod
    def load_visualize_de_proteomics(dir_path, q_val_thr=0.1, force_text=1, force_points=.2, plot_qqplot=False,
                                     alpha_top=.6, cut_off_texts = 10, volcano_axes_lim_dic=None, save_figure=False):
        p_val_thr = 0.05

        pval_col = 'P.Value'
        for i in range(1,6):
            title = f'P{i}'
            file_name = f'C{i}_vs_notC{i}_protein_differential_expression_v20240311.xlsx'
            res_df = MyLib.load_xls(os.path.join(dir_path,file_name), sheet_no=0)
            res_df['GeneName'] = res_df['GeneName'].astype(str).apply(
                lambda x: str(x).split('|')[1].strip() if '|' in x else x)
            # res_df.set_index('Gene', drop=True, inplace=True)
            res_filtered_df = res_df[res_df.apply(lambda x: (x[pval_col] < p_val_thr and
                                                             x['adj.P.Val'] < q_val_thr), axis=1)]
            logfc_col = f'logFC (C{i}/notC{i})'
            down_df = res_filtered_df[res_filtered_df[logfc_col] < 0].sort_values(logfc_col, ascending=True) \
                                                                 .sort_values('adj.P.Val', ascending=True)

            up_df = res_filtered_df[res_filtered_df[logfc_col] > 0].sort_values(logfc_col, ascending=True) \
                .sort_values('adj.P.Val', ascending=True)

            cut_off_labels = min(cut_off_texts, max(up_df.shape[0], down_df.shape[0]))

            if plot_qqplot:
                MyVisualization.qqplot(res_df[pval_col].tolist(), title=title)
            if up_df.shape[0] > 0 or down_df.shape[0] > 0:
                cluster_id = f'P{i}'
                ylim_top, ylim_bottom, xlim_right, xlim_left, y_step_size, x_step_size = None, None, None, None, None, None
                if (volcano_axes_lim_dic is not None) and (cluster_id in volcano_axes_lim_dic):
                    ylim_top, ylim_bottom, xlim_right, xlim_left, y_step_size, x_step_size = volcano_axes_lim_dic[
                        cluster_id]
                MyVisualization.Volcano_plot(res_df, x_col=logfc_col, y_col='adj.P.Val', title=title, force_text=force_text,
                                             force_points=force_points, alpha_top=alpha_top, label_fontsize=7,
                                             ylim_top=ylim_top, xlim_right=xlim_right, xlim_left=xlim_left,
                                             ylim_bottom=ylim_bottom, xtick_step=x_step_size, ytick_step=y_step_size,
                                             x_label='logFC', label_col='GeneName', down_df=down_df,  up_df=up_df,
                                             cut_off_labels=cut_off_labels, filename_suffix='_DEP', save_figure=save_figure)

    # @staticmethod
    # def statistical_test_and_volcano_plot_v2(df1, df2, col=None, res_df=None, points_size=7, test='ttest', q_value=0.1,
    #                                          points_labels_sr=None, add_label=True, selected_genes_dic=None,
    #                                          size_scale=1, hline_sig_cutoff=None,
    #                                          xtick_step=None, ytick_step=None, title=None, xlim_right=None,
    #                                          ylim_top=None,
    #                                          ylim_bottom=None, xlim_left=None, cut_off_labels=5, y_thr_qval=None,
    #                                          x_thr_leq=None,
    #                                          x_thr_geq=None, yspine_center=False, plot_qqplot=False, near_sig_qval=None,
    #                                          down_regulated=True, up_regulated=True, force_points=2, force_text=1,
    #                                          add_arrow=True,
    #                                          xlabel='', sig_color_up='red', sig_color_down='blue',
    #                                          near_sig_color_down='#47bcff',
    #                                          near_sig_color_up='#f0871f', add_up_labels=True, add_down_labels=True,
    #                                          add_legend=False, legend_title='', text_fontsize=7, legend_num=4,
    #                                          xlabel_fontsize=10, ylabel_fontsize=10, return_res=False,
    #                                          save_file='', save_figure=False):
    #     color_col = 'color'
    #     text_color_col = 'text_color'
    #     label_col = 'label'
    #     if test == 'ttest':
    #         df1, df2 = Utils.match_columns_df(df1, df2)
    #         res_df = Utils.apply_t_test(df1, df2, save_file=save_file)
    #         x_col = 'effect_size'
    #     elif test == 'correlation':
    #         if res_df is None:
    #             df1, df2 = Utils.match_rows_df(df1, df2)
    #             if col is not None:
    #                 res_df = Utils.apply_correlation_test_sr(df1[col], df2, save_file=save_file)
    #             else:
    #                 res_df = Utils.apply_correlation_paired_dfs(df1, df2, save_file=save_file)
    #         x_col = 'corr'
    #     else:
    #         print('Warning: Select ttest or correlation!')
    #         exit()
    #
    #     down_df, up_df = Utils.filter_statistical_results(res_df, x_col, q_val_thr=q_value,
    #                                                       down_regulated=down_regulated, up_regulated=up_regulated)
    #     res_df[color_col] = '#d4d4d4'
    #     res_df[text_color_col] = np.nan
    #     res_df[label_col] = np.nan
    #     if near_sig_qval is not None:
    #         ns_down_df, ns_up_df = Utils.filter_statistical_results(res_df, x_col, q_val_thr=near_sig_qval,
    #                                                                 down_regulated=down_regulated,
    #                                                                 up_regulated=up_regulated)
    #         if ns_down_df is not None:
    #             res_df.loc[ns_down_df.index, color_col] = near_sig_color_down
    #         if ns_up_df is not None:
    #             res_df.loc[ns_up_df.index, color_col] = near_sig_color_up
    #
    #     if down_df is not None:
    #         res_df.loc[down_df.index, color_col] = sig_color_down
    #         if add_down_labels:
    #             res_df.loc[down_df.index[:cut_off_labels], label_col] = down_df.iloc[:cut_off_labels]['feature']
    #             res_df.loc[down_df.index[:cut_off_labels], text_color_col] = 'k'
    #
    #     if up_df is not None:
    #         res_df.loc[up_df.index, color_col] = sig_color_up
    #         if add_up_labels:
    #             res_df.loc[up_df.index[:cut_off_labels], label_col] = up_df.iloc[:cut_off_labels]['feature']
    #             res_df.loc[up_df.index[:cut_off_labels], text_color_col] = 'k'
    #
    #     if y_thr_qval is not None:
    #         res_df = res_df[res_df['q-value'] < y_thr_qval]
    #     if x_thr_leq is not None:
    #         res_df = res_df[res_df[x_col] <= x_thr_leq]
    #     elif x_thr_geq is not None:
    #         res_df = res_df[res_df[x_col] >= x_thr_geq]
    #
    #     if not isinstance(points_size, int):
    #         points_size = points_size.astype(float)
    #         # points_size += 0.01
    #         # points_size *= 100
    #         points_size = points_size[res_df.index]
    #         print(points_size)
    #
    #     if points_labels_sr is not None:
    #         points_labels_sr = points_labels_sr[res_df.index]
    #     elif add_label:
    #         points_labels_sr = pd.Series(res_df.index.tolist(), index=res_df.index)
    #
    #     if selected_genes_dic is not None:
    #         temp_df = res_df[res_df['feature'].isin(selected_genes_dic.keys())]
    #         if temp_df.shape[0] < len(selected_genes_dic):
    #             print('Warning: selected genes are not present!')
    #             exit()
    #         res_df.loc[temp_df.index, color_col] = list(selected_genes_dic.values())
    #         res_df.loc[temp_df.index, text_color_col] = list(selected_genes_dic.values())
    #         res_df.loc[temp_df.index, label_col] = list(selected_genes_dic.keys())
    #
    #     if plot_qqplot:
    #         MyVisualization.qqplot(res_df['p-value'], title=title)
    #     MyVisualization.Volcano_plot_v2(res_df, x_col=x_col, y_col='q-value', color_col=color_col,
    #                                     legend_num=legend_num, hline_sig_cutoff=hline_sig_cutoff,
    #                                     text_color_col=text_color_col, add_label=add_label, points_size=points_size,
    #                                     points_labels_sr=points_labels_sr, add_legend=add_legend,
    #                                     legend_title=legend_title,
    #                                     title=title, x_label=xlabel, force_text=force_text, yspine_center=yspine_center,
    #                                     xtick_step=xtick_step, ytick_step=ytick_step, xlim_left=xlim_left,
    #                                     alpha_points=.7,
    #                                     text_fontsize=text_fontsize, xlabel_fontsize=xlabel_fontsize,
    #                                     ylabel_fontsize=ylabel_fontsize,
    #                                     force_points=force_points, add_arrow=add_arrow, xlim_right=xlim_right,
    #                                     size_scale=size_scale,
    #                                     ylim_top=ylim_top, ylim_bottom=ylim_bottom, save_figure=save_figure)
    #     if return_res:
    #         return (res_df)

    @staticmethod
    def statistical_test_and_volcano_plot_v2(df1, df2, col=None, res_df=None, points_size=7, test='ttest', q_value=0.1,
                                             points_labels_sr=None, add_label=True, selected_genes_dic=None,
                                             size_scale=1,
                                             xtick_step=None, ytick_step=None, title=None, xlim_right=None,
                                             ylim_top=None,
                                             ylim_bottom=None, xlim_left=None, cut_off_labels=5, y_thr_qval=None,
                                             x_thr_leq=None,
                                             x_thr_geq=None, yspine_center=False, plot_qqplot=False, near_sig_qval=None,
                                             down_regulated=True, up_regulated=True, force_points=2, force_text=1,
                                             add_arrow=True,
                                             xlabel='', sig_color_up='red', sig_color_down='blue',
                                             near_sig_color_down='#47bcff',
                                             near_sig_color_up='#f0871f', add_up_labels=True, add_down_labels=True,
                                             add_legend=False, legend_title='', text_fontsize=7, legend_num=4,
                                             xlabel_fontsize=10, ylabel_fontsize=10, return_res=False,
                                             hline_sig_cutoff=None,
                                             save_file='', save_figure=False):
        color_col = 'color'
        text_color_col = 'text_color'
        label_col = 'label'
        if test == 'ttest':
            df1, df2 = Utils.match_columns_df(df1, df2)
            res_df = Utils.apply_t_test(df1, df2, save_file=save_file)
            x_col = 'effect_size'
        elif test == 'correlation':
            if res_df is None:
                df1, df2 = Utils.match_rows_df(df1, df2)
                if col is not None:
                    res_df = Utils.apply_correlation_test_sr(df1[col], df2, save_file=save_file)
                else:
                    res_df = Utils.apply_correlation_paired_dfs(df1, df2, save_file=save_file)
            x_col = 'corr'
        else:
            print('Warning: Select ttest or correlation!')
            exit()

        down_df, up_df = Utils.filter_statistical_results(res_df, x_col, q_val_thr=q_value,
                                                          down_regulated=down_regulated, up_regulated=up_regulated)
        res_df[color_col] = '#d4d4d4'
        res_df[text_color_col] = np.nan
        res_df[label_col] = np.nan
        if near_sig_qval is not None:
            ns_down_df, ns_up_df = Utils.filter_statistical_results(res_df, x_col, q_val_thr=near_sig_qval,
                                                                    down_regulated=down_regulated,
                                                                    up_regulated=up_regulated)
            if ns_down_df is not None:
                res_df.loc[ns_down_df.index, color_col] = near_sig_color_down
            if ns_up_df is not None:
                res_df.loc[ns_up_df.index, color_col] = near_sig_color_up

        if down_df is not None:
            res_df.loc[down_df.index, color_col] = sig_color_down
            if add_down_labels:
                res_df.loc[down_df.index[:cut_off_labels], label_col] = down_df.iloc[:cut_off_labels]['feature']
                res_df.loc[down_df.index[:cut_off_labels], text_color_col] = 'k'

        if up_df is not None:
            res_df.loc[up_df.index, color_col] = sig_color_up
            if add_up_labels:
                res_df.loc[up_df.index[:cut_off_labels], label_col] = up_df.iloc[:cut_off_labels]['feature']
                res_df.loc[up_df.index[:cut_off_labels], text_color_col] = 'k'

        if y_thr_qval is not None:
            res_df = res_df[res_df['q-value'] < y_thr_qval]
        if x_thr_leq is not None:
            res_df = res_df[res_df[x_col] <= x_thr_leq]
        elif x_thr_geq is not None:
            res_df = res_df[res_df[x_col] >= x_thr_geq]

        if not isinstance(points_size, int):
            points_size = points_size.astype(float)
            # points_size += 0.01
            # points_size *= 100
            points_size = points_size[res_df.index]
            print(points_size)

        if points_labels_sr is not None:
            points_labels_sr = points_labels_sr[res_df.index]
        elif add_label:
            points_labels_sr = pd.Series(res_df.index.tolist(), index=res_df.index)

        if selected_genes_dic is not None:
            temp_df = res_df[res_df['feature'].isin(selected_genes_dic.keys())]
            if temp_df.shape[0] < len(selected_genes_dic):
                print('Warning: selected genes are not present!')
                exit()
            res_df.loc[temp_df.index, color_col] = list(selected_genes_dic.values())
            res_df.loc[temp_df.index, text_color_col] = list(selected_genes_dic.values())
            res_df.loc[temp_df.index, label_col] = list(selected_genes_dic.keys())

        if plot_qqplot:
            MyVisualization.qqplot(res_df['p-value'], title=title)
        MyVisualization.Volcano_plot_v2(res_df, x_col=x_col, y_col='q-value', color_col=color_col,
                                        legend_num=legend_num,
                                        text_color_col=text_color_col, add_label=add_label, points_size=points_size,
                                        hline_sig_cutoff=hline_sig_cutoff,
                                        points_labels_sr=points_labels_sr, add_legend=add_legend,
                                        legend_title=legend_title,
                                        title=title, x_label=xlabel, force_text=force_text, yspine_center=yspine_center,
                                        xtick_step=xtick_step, ytick_step=ytick_step, xlim_left=xlim_left,
                                        alpha_points=.7,
                                        text_fontsize=text_fontsize, xlabel_fontsize=xlabel_fontsize,
                                        ylabel_fontsize=ylabel_fontsize,
                                        force_points=force_points, add_arrow=add_arrow, xlim_right=xlim_right,
                                        size_scale=size_scale,
                                        ylim_top=ylim_top, ylim_bottom=ylim_bottom, save_figure=save_figure)
        if return_res:
            return (res_df)

    @staticmethod
    def visualize_de_genes(res_df, in_group, out_group='Rest', pval_col='P.Value', fdr_col='adj.P.Val', q_val_thr=0.1,
                     plot_qqplot=False, force_text=.6, force_points=.2, return_up=False, return_down=False,
                     cut_off_texts=10, up_regulated=True, down_regulated=True, cut_off_labels_down=np.inf,
                     y_hline=None, title=None, ylim_top=None, ylim_bottom=None, xlim_right=None, xlim_left=None,
                     ytick_step=None, xtick_step=None, save_figure=False, file_name=''):
        p_val_thr = 0.05

        res_filtered_df = res_df[res_df.apply(lambda x: (x[pval_col] < p_val_thr and
                                                         x[fdr_col] < q_val_thr), axis=1)]
        down_dim=0
        up_dim=0
        down_df=None
        up_df=None
        if down_regulated:
            down_df = res_filtered_df[res_filtered_df['logFC'] < 0].sort_values('logFC', ascending=True) \
                .sort_values(fdr_col, ascending=True)
            down_df['Gene'] = down_df['Gene'].apply(lambda x: x.split(' (')[0])
            down_dim = down_df.shape[0]
        if up_regulated:
            up_df = res_filtered_df[res_filtered_df['logFC'] > 0].sort_values('logFC', ascending=True) \
                .sort_values(fdr_col, ascending=True)
            up_df['Gene'] = up_df['Gene'].apply(lambda x: x.split(' (')[0])
            up_dim = up_df.shape[0]
        cut_off_labels = min(cut_off_texts, max(down_dim, up_dim))
        if title is None:
            title = f'{in_group} vs {out_group}'
        if plot_qqplot:
            # MyVisualization.qqplot(res_df[pval_col].tolist(), title=title, ylim_top=2)
            MyVisualization.qqplot(res_df[pval_col].tolist(), title=title)
        if up_dim > 0 or down_dim > 0:
            MyVisualization.Volcano_plot(res_df, x_col='logFC', y_col=fdr_col, title=title, y_hline=y_hline,
                                         x_label='logFC', label_col='Gene', down_df=down_df, force_text=force_text,
                                         force_points=force_points, cut_off_labels_down=cut_off_labels_down,
                                         up_df=up_df, cut_off_labels=cut_off_labels, label_fontsize=6, point_size=15,
                                         ylim_top=ylim_top, ylim_bottom=ylim_bottom, xlim_right=xlim_right, xlim_left=xlim_left,
                                         ytick_step=ytick_step, xtick_step=xtick_step,
                                         save_figure=save_figure)
        contrast = f'{in_group}_{out_group}'
        if up_dim > 0:
            up_df.set_index('Gene', inplace=True, drop=True)
            MyLib.save_csv(up_df, f'{file_name}{contrast}_up.csv')
        if down_dim > 0:
            down_df.set_index('Gene', inplace=True, drop=True)
            MyLib.save_csv(down_df, f'{file_name}{contrast}_down.csv')
        if return_up and return_down:
            return (up_df, down_df)
        elif return_up:
            return up_df
        elif return_down:
            return down_df

    @staticmethod
    def process_dependency_table(genes_table_df, upset_df, gene_effect_df, target_genes, gene_effect_thr_dic,
                                 data_df, cluster_col, thr_dic=None):
        shared_indices = list(set(data_df.index).intersection(upset_df.index))
        data_df = data_df.loc[shared_indices, :]
        gp = data_df.groupby(cluster_col)
        for index, df in gp:
            if index != 0:
                group_l = df.index.tolist()
                for gene in target_genes:
                    dependent_cell_lines = (gene_effect_df.loc[group_l, gene] < gene_effect_thr_dic[gene]).sum()
                    percentage = round((dependent_cell_lines / df.shape[0]) * 100)
                    if thr_dic is not None and percentage >= thr_dic[index]:
                        genes_table_df.loc[gene, f'{index}[n={df.shape[0]}] (%)'] = percentage
                    elif thr_dic is None and percentage > 0:
                        genes_table_df.loc[gene, f'{index}[n={df.shape[0]}] (%)'] = percentage
                        # genes_table_df.loc[gene, f'{index}(n={df.shape[0]})'] = f'{dependent_cell_lines} ({percentage}%)'

        return genes_table_df

    @staticmethod
    def apply_correlations_pairwise(df1, df2, index_name, row_label='', col_label='', title='', plot_heatmaps=True,
                                    filename='', figure_width=2, figure_height=2):

        corr_df, p_vals_df = Utils.pearson_corr_sequential(df1, df2)
        labels = Clustering.hierarchical_clustering(corr_df)
        corr_df = corr_df.loc[labels, labels]
        p_vals_df = p_vals_df.loc[labels, labels]

        p_vals_df = p_vals_df.rename_axis('Index')
        p_vals_tri_df = p_vals_df.where(np.tril(np.ones(p_vals_df.shape), k=-1).astype(bool))
        p_vals_melt_df = pd.melt(p_vals_tri_df.reset_index(), id_vars='Index')
        p_vals_melt_df.dropna(axis=0, inplace=True)
        p_vals_melt_df = p_vals_melt_df.rename(columns={'value': 'p-value'})
        p_vals_melt_df['q-value'] = fdrcorrection(p_vals_melt_df['p-value'].values)[1]
        p_vals_melt_df['fdr_log'] = -np.log10(p_vals_melt_df['q-value'])
        fdr_log = pd.DataFrame(0, index=p_vals_df.index, columns=p_vals_df.columns)

        for id, row in p_vals_melt_df.iterrows():
            fdr_log.loc[row['Index'], row[index_name]] = row['fdr_log']
            fdr_log.loc[row[index_name], row['Index']] = row['fdr_log']
            p_vals_melt_df.loc[id, 'corr'] = corr_df.loc[row[index_name], row['Index']]

        p_vals_melt_df = p_vals_melt_df.sort_values('corr', ascending=False).sort_values('q-value', ascending=True)

        if plot_heatmaps:
            MyVisualization.plot_heatmap(corr_df, row_label=row_label, col_label=col_label, cbar_title='Correlation',
                                         xtick_fontsize=8, ytick_fontsize=8, title=title, cbar_left_adjust=-.09,
                                        title_fontsize=9, figsize_w=figure_width, figsize_h=figure_height,
                                         filename_suffix=filename, save_figure=True)
            cbar_colors = ['white', "#d94801", "#a63603"]
            cabr_intervals = [0, 1, 1.5, 7]
            MyVisualization.plot_heatmap(fdr_log, row_label=row_label, col_label=col_label, cbar_title='-Log10 Adj P',
                                         xtick_fontsize=8, ytick_fontsize=8, title=title, cbar_left_adjust=-.09,
                                         cbar_pval=True, title_fontsize=9, figsize_w=figure_width, figsize_h=figure_height,
                                         filename_suffix=filename, save_figure=True)
        MyLib.save_csv(p_vals_melt_df[['Index', index_name, 'corr', 'p-value', 'q-value']],
                       f'{filename}_correlations.csv', index=False)
import numpy as np
import pandas as pd
import random
from netneurotools import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import median_abs_deviation
import bct
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ranksums
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score


from scripts.visualization import MyVisualization
from scripts.io_library import MyLib

seed = 2023

class Clustering:

    @staticmethod
    def init_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def select_top_variable_features(df, top_k=100):
        variances = df.var()
        sorted_variances = variances.sort_values(ascending=False)
        top_variables = sorted_variances.head(top_k).index.tolist()
        return df.loc[:, top_variables]

    @staticmethod
    def drop_low_vars(exp_df, threshold=1e-10):
        vars_sr = exp_df.var()
        mask = vars_sr > threshold
        exp_df = exp_df.loc[:, exp_df.columns[mask]]
        return exp_df

    @staticmethod
    def filter_sparse_samples(df, max_allowed_nans_zeros_frac=0.1):
        max_allowed_nans_zeros = df.shape[0] * max_allowed_nans_zeros_frac
        filtered_df = df.loc[:,(df.isnull().sum(axis=0) + (df == 0).sum(axis=0)) < max_allowed_nans_zeros]
        return filtered_df

    @staticmethod
    def mad_normalize(column):
        mad = median_abs_deviation(column, nan_policy='omit')
        if mad==0:
            print(f'Note MAD==0: {column.name}')
        median = column.median(skipna=True)
        return (column - median) / mad

    @staticmethod
    def select_variable_features(df, variance_percentile_cutoff=75):
        variance_sr = df.var(axis=0, skipna=True)
        variance_cutoff = np.percentile(variance_sr.dropna(), variance_percentile_cutoff)
        filtered_df = df[variance_sr[variance_sr > variance_cutoff].index]

        return filtered_df

    @staticmethod
    def hierarchical_clustering(df, reverse=False):
        """
            Produces the same output as clustermap of Seaborn
        """
        linked = linkage(df, 'ward', optimal_ordering=True)
        dd = dendrogram(linked,
                        # distance_sort='descending',
                        # count_sort='descending',
                        show_leaf_counts=False,
                        no_plot=True)
        labels = df.iloc[dd['leaves'],:].index.tolist()
        if reverse:
            labels.reverse()
        return labels

    @staticmethod
    def sort_samples(data_df, upset_df, reverse, cluster_column='Cluster', cluster_ids=None):
        """
            Sorts the samples within each cluster using hierarchical clustering
        """
        if cluster_ids is None:
            cluster_ids = sorted(upset_df[cluster_column].unique().tolist())
        sorted_samples = []
        for cluster_id in cluster_ids:
            cluster_data = data_df[upset_df[cluster_column] == cluster_id]
            if cluster_data.shape[0] > 1:
                sorted_samples += Clustering.hierarchical_clustering(cluster_data, reverse)
            else:
                sorted_samples += cluster_data.index.tolist()
        return sorted_samples

    @staticmethod
    def plot_heatmaps_subtypes(data_df, upset_df, clusters_ordering, cluster_column='Subtype',
                               heatmap_row_cluster=False, save_figure=False):
        mean_df = pd.DataFrame(index=data_df.columns, columns=clusters_ordering)
        mean_upset_df = pd.DataFrame(clusters_ordering, columns=[cluster_column], index=clusters_ordering)
        for cur_cluster in clusters_ordering:
            cur_cluster_df = data_df[upset_df[cluster_column] == cur_cluster].T

            mean_df[cur_cluster] = cur_cluster_df.mean(axis=1)
        row_min_sr = mean_df.min(axis=1)
        print('Minimum value for all clusters:')
        print(row_min_sr.sort_values())
        mean_df = mean_df[row_min_sr < -0.4]
        data_df = data_df[mean_df.index.tolist()].copy()
        sorted_features = MyVisualization.plot_heatmap(mean_df, mean_upset_df, cbar_title='Mean Gene Effect',
                                                       row_label='Gene',  # xdend=True,
                                                       row_cluster=heatmap_row_cluster, ytick_fontsize=7, cbar_CRISPR=True,
                                                       left_adjust=0.2,
                                                       bar_colors_ratio=0.05, cbar_left_adjust=0.015,
                                                       col_label_fontsize=8, row_label_fontsize=8, legend_bar=False,
                                                       legend_top_offset=0.7, figsize_w=3, figsize_h=3,
                                                       save_figure=save_figure)
        if sorted_features is not None:
            data_df = data_df.loc[:, sorted_features]
        sample_ids = Clustering.sort_samples(data_df, upset_df, reverse=True, cluster_column=cluster_column,
                                             cluster_ids=clusters_ordering)
        data_df = data_df.loc[sample_ids, :]
        upset_df = upset_df.loc[sample_ids, :]
        MyVisualization.plot_heatmap(data_df.T, upset_df, col_label='Cell Line',
                                     row_label='Gene', cbar_discrete=True, col_label_fontsize=8, row_label_fontsize=8,
                                     cbar_legened_offset=.35, row_cluster=False, ytick_fontsize=7, xtick_fontsize=7,
                                     cbar_title=f"Gene effect", bar_colors_ratio=0.05, cbar_left_adjust=0.015,
                                     left_adjust=0.07, legend_top_offset=0.7, figsize_w=7, figsize_h=4,
                                     save_figure=save_figure)
    @staticmethod
    def normalize_corr_matrix(corr_df):
        corr_matrix_df = corr_df.copy()
        corr_matrix_df[corr_matrix_df < 0] = 0
        return corr_matrix_df

    @staticmethod
    def louvain_clustering_ensemble(df, ite=100, gammas=None):
        if gammas is None:
            gammas = [1.5]
        nonegative_df = Clustering.normalize_corr_matrix(df)
        labels_np = np.column_stack([bct.community_louvain(nonegative_df.values, gamma=g, seed=seed + n)[0]
                                     for n in range(ite) for g in gammas])
        total_ite = len(gammas) * ite
        return (labels_np, total_ite)

    @staticmethod
    def prepare_labels(upset_df, labels, clusters_reordering=None, cluster_prefix='C'):
        """
        Adds a new column 'Cluster' for the labels.
        Converts integer labels to strings and,
        for visualization purposes, relabels clusters according to the provided ordering.
        """
        # relabeled_upset_df = upset_df.copy()
        relabeled_upset_df = pd.DataFrame(labels, columns=['Cluster'], index=upset_df.index)
        # relabeled_upset_df['Cluster'] = labels
        relabeled_upset_df['Cluster'] = relabeled_upset_df['Cluster'].astype('int32')
        if clusters_reordering is not None:
            if min(labels) == 1:
                label_mapping = {old_label: new_label + 1 for new_label, old_label in enumerate(clusters_reordering)}
            elif min(labels) == -1:
                label_mapping = {old_label - 2: new_label - 1 for new_label, old_label in
                                 enumerate(clusters_reordering)}
                print(f'Warning, cluster {cluster_prefix}1 of noise samples is moved to C{label_mapping[-1] + 2}')
            else:
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(clusters_reordering)}
            relabeled_upset_df['Cluster'] = relabeled_upset_df['Cluster'].map(label_mapping)
            # reordered_upset_df['Cluster'].astype('int32')
        if min(labels) == 0:
            relabeled_upset_df['Cluster'] = relabeled_upset_df['Cluster'].apply(
                lambda x: f'{cluster_prefix}{int(x + 1)}')
        elif min(labels) == -1:  # -1 labels are mapped to 1 indicating the noise cluster
            print('Min label -1')
            relabeled_upset_df['Cluster'] = relabeled_upset_df['Cluster'].apply(
                lambda x: f'{cluster_prefix}{int(x + 2)}')
        else:
            relabeled_upset_df['Cluster'] = relabeled_upset_df['Cluster'].apply(lambda x: f'{cluster_prefix}{int(x)}')
        relabeled_upset_df = pd.concat([relabeled_upset_df, upset_df], axis=1)
        relabeled_upset_df.sort_values('Cluster', inplace=True)
        return relabeled_upset_df

    @staticmethod
    def create_consensus_matrix(consensus_lbls, total_ite, index):
        consensus_np = bct.agreement(consensus_lbls, buffsz=consensus_lbls.shape[0])
        np.fill_diagonal(consensus_np, total_ite)
        consensus_df = pd.DataFrame(consensus_np, index=index, columns=index)
        return consensus_df

    @staticmethod
    def imputing_simple(data_df):
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')  # 'mean'
        X = imputer.fit_transform(data_df)
        new_df = pd.DataFrame(X, columns=data_df.columns, index=data_df.index)
        return new_df

    @staticmethod
    def plot_consensus_heatmaps(data_df, corr_df, upset_df, labels_np, consensus_ite, plot_corr=False, up=True,
                                ttest=False, q_val_thr=0.1,
                                feature_cutoff=np.inf, row_cluster=False, row_label='Gene', reverse=False,
                                clusters_type_filename='', add_mean_cols=False,
                                clusters_reordering=None, cluster_prefix='C', xticklabels=True, yticklabels=True,
                                ytick_fontsize=9, xtick_fontsize=9,
                                bar_colors_ratio=0.01, legend_fontsize=9, event_legend=True, event_color=None,
                                cbar_title='                 ',
                                borders_linewidths=0.01, legend_h_offset=0.06, filter=False, cbar_discrete=False,
                                plot_volcano=False,
                                volcano_axes_lim_dic=None, gene_effect_thr=None, cut_off_labels_top=10,
                                left_adjust=0.1, legend_top_offset=0.85, cbar_left_adjust=0.06, legend_diff_offset=0.11,
                                volcano_force_points=0.6, volcano_force_text=0.4, top_features=True, plot_qqplot=False,
                                title_consensus_mat='', title_corr_mat='', title_markers_mat='', figsize_w=13,
                                figsize_h=12,
                                save_figure=False):
        """
         This method manages the output of clustering algorithms that produce varying labels for the same input across
         different runs. It then plots these results based on consensus scores and the top features.
            input labels_np: each column correspond to labels of one run.
        """
        consensus_lbls = cluster.find_consensus(labels_np, seed=seed)  # To find consensus clustering labels
        consensus_df = Clustering.create_consensus_matrix(labels_np, consensus_ite, data_df.index)
        reordered_upset_df = Clustering.prepare_labels(upset_df, consensus_lbls, clusters_reordering,
                                                       cluster_prefix=cluster_prefix)

        # prepare dataframe to save clusters
        temp_upset = reordered_upset_df.copy()
        columns_to_drop = temp_upset.filter(regex='\$$', axis=1)
        temp_upset.drop(columns=columns_to_drop.columns, inplace=True)
        if clusters_type_filename == 'RNA_':
            temp_upset.rename(columns={'Cluster': 'RNA_Cluster'}, inplace=True)

        data_df = data_df.reindex(reordered_upset_df.index)
        features = Clustering.identify_significant_features(data_df, reordered_upset_df, up_regulated=up, ttest=ttest,
                                                            feature_cutoff=feature_cutoff, q_val_thr=q_val_thr,
                                                            plot_qqplot=plot_qqplot, add_mean_cols=add_mean_cols,
                                                            volcano_axes_lim_dic=volcano_axes_lim_dic,
                                                            gene_effect_thr=gene_effect_thr,
                                                            cut_off_labels_top=cut_off_labels_top,
                                                            plot_volcano=plot_volcano,
                                                            force_points=volcano_force_points,
                                                            force_text=volcano_force_text, filter=filter,
                                                            save_figure=save_figure)

        data_filtered_df = data_df.loc[:, features]
        if not data_filtered_df.isnull().values.any():
            sample_ids = Clustering.sort_samples(data_filtered_df, reordered_upset_df, reverse)
        else:
            impute_df = Clustering.imputing_simple(data_filtered_df)
            sample_ids = Clustering.sort_samples(impute_df, reordered_upset_df, reverse)

        reordered_upset_df = reordered_upset_df.reindex(sample_ids)

        # save clusters
        temp_upset = temp_upset.reindex(sample_ids)
        MyLib.save_csv(temp_upset, f'{clusters_type_filename}{row_label}_reordered_upset_df.csv')

        consensus_df = consensus_df.loc[sample_ids, sample_ids]
        MyVisualization.plot_heatmap(consensus_df, reordered_upset_df, bar_colors_ratio=bar_colors_ratio,
                                     legend_fontsize=legend_fontsize,
                                     col_label='Cell Line', row_label='Cell Line',
                                     borders_linewidths=borders_linewidths, event_legend=event_legend,
                                     event_color=event_color, title=title_consensus_mat, cbar_title='Consensus Score\n',
                                     legend_top_offset=legend_top_offset, left_adjust=left_adjust,
                                     ytick_fontsize=ytick_fontsize, xtick_fontsize=ytick_fontsize,
                                     cbar_left_adjust=cbar_left_adjust, legend_diff_offset=legend_diff_offset,
                                     legend_h_offset=legend_h_offset,
                                     figsize_w=figsize_w, figsize_h=figsize_h, save_figure=save_figure)

        if plot_corr:
            reordered_corr_df = corr_df.loc[sample_ids, sample_ids]
            MyVisualization.plot_heatmap(reordered_corr_df, reordered_upset_df, bar_colors_ratio=bar_colors_ratio,
                                         legend_fontsize=legend_fontsize, title=title_corr_mat,
                                         cbar_title='   Correlation Score\n',
                                         legend_top_offset=legend_top_offset, left_adjust=left_adjust,
                                         cbar_left_adjust=cbar_left_adjust,
                                         legend_h_offset=legend_h_offset, event_legend=event_legend,
                                         event_color=event_color,
                                         legend_diff_offset=legend_diff_offset, figsize_w=figsize_w,
                                         figsize_h=figsize_h)

        if top_features & (len(features) > 0):
            data_filtered_df = data_filtered_df.loc[sample_ids, features]
            MyVisualization.plot_heatmap(data_filtered_df.T, reordered_upset_df, col_label='Cell Line',
                                         row_label=row_label,
                                         row_cluster=row_cluster, xticklabels=xticklabels, yticklabels=yticklabels,
                                         ytick_fontsize=ytick_fontsize,
                                         xtick_fontsize=xtick_fontsize, cbar_discrete=cbar_discrete,
                                         event_legend=event_legend, event_color=event_color,
                                         bar_colors_ratio=bar_colors_ratio, legend_fontsize=legend_fontsize,
                                         title=title_markers_mat, legend_h_offset=legend_h_offset,
                                         cbar_title=cbar_title, legend_top_offset=legend_top_offset,
                                         left_adjust=left_adjust, borders_linewidths=borders_linewidths,
                                         cbar_left_adjust=cbar_left_adjust, legend_diff_offset=legend_diff_offset,
                                         figsize_w=figsize_w, figsize_h=figsize_h,
                                         save_figure=save_figure)
        return (features, reordered_upset_df)

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
    def identify_significant_features(data_df, upset_df, cluster_column='Cluster', ttest=False, up_regulated=True,
                                      feature_cutoff=np.inf, q_val_thr=0.1, plot_volcano=True, plot_qqplot=True,
                                      filter=False, add_mean_cols=False,
                                      force_points=0.6, force_text=0.4, cut_off_labels_top=10,
                                      volcano_axes_lim_dic=None,
                                      gene_effect_thr=None, exclusion_ids=None, add_non_hits_labels=False, title='',
                                      save_figure=None, filename_prefix=''):
        """

        :param data_df:
        :param upset_df:
        :param cluster_column:
        :param ttest:
        :param up_regulated:
        :param feature_cutoff: cut off for the number of sig features
        :param q_val_thr:
        :param plot_volcano:
        :param plot_qqplot:
        :param filter:
        :param force_points:
        :param force_text:
        :param cut_off_labels_top: cut off for the number of labels of sig features
        :param volcano_axes_lim_dic:
        :param gene_effect_thr:
        :param exclusion_ids:
        :param save_figure:
        :return:
        """
        p_val_thr = 0.05
        cluster_ids = upset_df[cluster_column].unique().tolist()
        if exclusion_ids is not None:
            for id in exclusion_ids:
                cluster_ids.remove(id)
        features_l = []
        top_df = None  # For gene effect, add label for those not meeting the 2x dependency condition
        for cluster_id in cluster_ids:
            cluster_data = data_df[upset_df[cluster_column] == cluster_id]
            other_clusters_data = data_df[upset_df[cluster_column] != cluster_id]
            p_values = []
            if (cluster_data.shape[0] > 1) and (other_clusters_data.shape[0] > 1):
                for col in data_df.columns:
                    if ttest:
                        stat, p_value = ttest_ind(cluster_data[col],
                                                  other_clusters_data[col])  ## two-sample (unpaired) t-test
                        # stat, p_value = ttest_ind(cluster_data[col], other_clusters_data[col], equal_var=False)  ## two-tailed Welch’s t-test
                        effect_size = Clustering.cohend_effect_size(cluster_data[col], other_clusters_data[col])
                    else:
                        stat, p_value = ranksums(cluster_data[col], other_clusters_data[col], nan_policy='omit')
                        # stat, p_value = mannwhitneyu(cluster_data[col], other_clusters_data[col])
                        effect_size = stat / np.sqrt(len(cluster_data) + len(other_clusters_data))  ## effect size r
                    p_values.append((col, p_value, effect_size))

                res_df = pd.DataFrame(p_values, columns=['feature', 'p-value', 'effect_size'])
                # MyVisualization.plot_distribution(res_df['p-value'])
                res_df['q-value'] = fdrcorrection(res_df['p-value'].values)[1]

                if add_mean_cols:
                    in_group_data_sr = cluster_data.mean()
                    in_group_data_sr.name = 'in_group_gene_effect_mean'

                    out_group_data_sr = other_clusters_data.mean()
                    out_group_data_sr.name = 'out_group_gene_effect_mean'

                    res_df.index = res_df['feature']
                    res_df = pd.concat([res_df, in_group_data_sr, out_group_data_sr], axis=1)
                    res_df.reset_index(inplace=True, drop=True)

                res_filtered_df = res_df[res_df.apply(lambda x: (x['p-value'] < p_val_thr and
                                                                 x['q-value'] < q_val_thr), axis=1)]
                if plot_qqplot:
                    MyVisualization.qqplot(res_df['p-value'].tolist(), title=cluster_id)
                if res_filtered_df.shape[0] > 0:

                    up_df = None
                    down_df = None
                    if up_regulated:
                        regulated = 'Up'
                        up_df = res_filtered_df[res_filtered_df['effect_size'] >= 0].sort_values('effect_size',
                                                                                                 ascending=False) \
                            .sort_values('q-value', ascending=True)
                        MyLib.save_csv(up_df, f'{filename_prefix}all_{cluster_id}_features_{regulated}.csv', index=False)
                        cur_feature_cut_off = min(up_df.shape[0], feature_cutoff)
                        up_df = up_df.iloc[:cur_feature_cut_off, ]
                        cur_cut_off_labels = min(cut_off_labels_top, up_df.shape[0])
                        temp_df = up_df.copy()

                    else:
                        regulated = 'Down'
                        down_df = res_filtered_df[res_filtered_df['effect_size'] < 0].sort_values('effect_size',
                                                                                                  ascending=True) \
                            .sort_values('q-value', ascending=True)
                        MyLib.save_csv(down_df, f'{filename_prefix}all_{cluster_id}_features_{regulated}.csv', index=False)
                        cur_feature_cut_off = min(down_df.shape[0], feature_cutoff)
                        down_df = down_df.iloc[:cur_feature_cut_off, ]
                        cur_cut_off_labels = min(cut_off_labels_top, down_df.shape[0])
                        temp_df = down_df.copy()
                        if gene_effect_thr is not None:
                            dependent_genes = cluster_data.columns[(cluster_data < gene_effect_thr).sum() >= 2].tolist()
                            # group_data_df = cluster_data.mean()
                            # group_data_df = group_data_df[group_data_df < gene_effect_thr]
                            if add_non_hits_labels:
                                top_df = down_df[~down_df['feature'].isin(dependent_genes)].head(cur_cut_off_labels)
                            down_df = down_df[down_df['feature'].isin(dependent_genes)]

                    if (not up_regulated and down_df.shape[0] > 0) or (up_regulated and up_df.shape[0] > 0):
                        if plot_volcano:
                            ylim_top, ylim_bottom, xlim_right, xlim_left, y_step_size, x_step_size = None, None, None, None, None, None
                            if (volcano_axes_lim_dic is not None) and (cluster_id in volcano_axes_lim_dic):
                                ylim_top, ylim_bottom, xlim_right, xlim_left, y_step_size, x_step_size = \
                                volcano_axes_lim_dic[cluster_id]

                            MyVisualization.Volcano_plot(res_df, y_col='q-value', x_col='effect_size',
                                                         title=cluster_id + title, x_label='Effect size',
                                                         label_col='feature', force_points=force_points,
                                                         force_text=force_text, xlim_right=xlim_right,
                                                         xlim_left=xlim_left, ylim_top=ylim_top,
                                                         ylim_bottom=ylim_bottom, cut_off_labels=cur_cut_off_labels,
                                                         down_df=down_df, up_df=up_df, top_df=top_df,
                                                         xtick_step=x_step_size, ytick_step=y_step_size,
                                                         save_figure=save_figure)
                        temp_df.set_index('feature', inplace=True, drop=True)
                        if filter:  # For visualization purpose only select genes that are dependencies in at least two cell lines
                            count_less_than_minus_0_5 = (cluster_data[temp_df.index.tolist()] < -0.5).sum()
                            data_filtered_df = cluster_data[temp_df.index.tolist()].loc[:,
                                               count_less_than_minus_0_5 >= 2]
                            hits_l = [f for f in data_filtered_df.columns if f not in features_l]
                            features_l += hits_l
                            MyLib.save_csv(temp_df.loc[hits_l], f'{filename_prefix}{cluster_id}_features_{regulated}_dep_hits.csv')
                        else:
                            features_l += [f for f in temp_df.index.tolist() if f not in features_l]
                        MyLib.save_csv(temp_df, f'{filename_prefix}{cluster_id}_features_{regulated}.csv')
        return features_l

    @staticmethod
    def permutation_test_adjusted_rand_score(cluster_id, feat_sr, permuted_clusters_l, observed_score):
        """

        """
        permuted_statistics = []

        for cluster_df in permuted_clusters_l:
            permuted_ari = adjusted_rand_score(feat_sr, cluster_df[cluster_id])
            permuted_statistics.append(permuted_ari)

        p_value = (np.array(permuted_statistics, dtype=np.float64) >= observed_score).mean()

        if p_value == 0:  # Extreme case
            p_value = 1 / len(permuted_clusters_l)

        return p_value

    @staticmethod
    def permute_clusters_labels(clusters_df, permutation_count=1):
        """
        Permutes the sample labels for each cluster.

        :param clusters_df: Pandas DataFrame whose columns represent clusters and rows are cell lines,
                            with a value of 1 indicating membership.
        :param permutation_count: Number of permutations to perform. Default is 1.
        """
        permuted_clusters_l = []
        for i in range(permutation_count):
            lbls_sr = pd.Series(index=clusters_df.index, dtype=np.int16)
            for lbl, c in enumerate(clusters_df.columns):
                lbls_sr[clusters_df[clusters_df[c] == 1].index] = lbl
            lbls_np = np.array(lbls_sr, dtype=np.int16)
            np.random.shuffle(lbls_np)
            lbls_sr = pd.Series(lbls_np, index=clusters_df.index, dtype=np.int16)
            new_clusters_df = pd.DataFrame(0, index=clusters_df.index, columns=clusters_df.columns)
            for lbl, c in enumerate(clusters_df.columns):
                new_clusters_df.loc[lbls_sr[lbls_sr == lbl].index, c] = 1
            permuted_clusters_l.append(new_clusters_df)
        return permuted_clusters_l

    @staticmethod
    def calc_q_values(df):
        p_vals_np = df.values.flatten()
        q_vals_np = fdrcorrection(p_vals_np)[1]
        q_vals_df = pd.DataFrame(q_vals_np.reshape(df.shape), index=df.index, columns=df.columns)
        return q_vals_df

    @staticmethod
    def clustering_scores_and_statistics(clusters_df, feat_df, split_feats_l=None, split_row=None, col_label='RNA Cluster', row_label='',
                         title='Clustering Similarities', seed=2023, filename='', figsize_w=3, figsize_h=3, save_figure=False):
        Clustering.init_seed(seed)
        permuted_clusters_l = Clustering.permute_clusters_labels(clusters_df, permutation_count=5000)
        p_vals_df = pd.DataFrame(index=feat_df.columns, columns=clusters_df.columns)
        paired_df = pd.DataFrame(np.nan, index=feat_df.columns, columns=clusters_df.columns)
        ari_df = pd.DataFrame(index=feat_df.columns, columns=clusters_df.columns, dtype=np.float64)
        for c in clusters_df:
            print(c)
            for f in feat_df.columns:
                ari = adjusted_rand_score(feat_df[f], clusters_df[c])  # Adjusted Rand Index
                ari_df.loc[f, c] = ari
                coverage_frac_complement = sum(feat_df[f] & (1-clusters_df[c])) / sum(1-clusters_df[c])
                coverage_frac = sum(feat_df[f] & clusters_df[c]) / sum(clusters_df[c])
                if coverage_frac_complement <= coverage_frac:
                    paired_df.loc[f, c] = 'paired'
                p_val = Clustering.permutation_test_adjusted_rand_score(cluster_id=c, feat_sr=feat_df[f],
                                                                        permuted_clusters_l=permuted_clusters_l,
                                                                        observed_score=ari)

                p_vals_df.loc[f, c] = p_val
        if split_feats_l is not None:
            group1_df = Clustering.calc_q_values(p_vals_df.loc[split_feats_l])
            group2_df = Clustering.calc_q_values(p_vals_df.loc[~p_vals_df.index.isin(split_feats_l)])
            q_vals_df = pd.concat([group2_df, group1_df], axis=0)
        else:
            q_vals_df = Clustering.calc_q_values(p_vals_df)

        empty_row1 = pd.DataFrame(columns=p_vals_df.columns, index=['p-value'])
        empty_row2 = pd.DataFrame(columns=p_vals_df.columns, index=['q-value'])
        empty_row3 = pd.DataFrame(columns=p_vals_df.columns, index=['ARI score'])
        empty_row4 = pd.DataFrame(columns=p_vals_df.columns, index=['Cluster Matching'])
        stats_df = pd.concat([empty_row1, p_vals_df,empty_row2, q_vals_df], axis=0)
        ari_comp_df = pd.concat([empty_row3, ari_df, empty_row4, paired_df], axis=0)
        MyLib.save_csv(stats_df, f'adjusted_rand_index_stats_{filename}.csv')
        MyLib.save_csv(ari_comp_df, f'adjusted_rand_scores_{title}{filename}.csv')

        MyVisualization.plot_heatmap(ari_df, None,
                                     col_label=col_label, title=title,
                                     row_label=row_label, xdend=True, cbar_title='Adjusted Rand Index',
                                     row_cluster=False, ytick_fontsize=7, cbar_CRISPR=False,
                                     left_adjust=0.2, split_row_l=split_row, legend_fontsize=6, title_fontsize=10,
                                     bar_colors_ratio=0.05, cbar_left_adjust=0.015, title_height=.94,
                                     col_label_fontsize=8, row_label_fontsize=8, legend_bar=False,
                                     legend_top_offset=0.7, figsize_w=figsize_w, figsize_h=figsize_h,
                                     save_figure=save_figure)

        return (p_vals_df, q_vals_df)

    @staticmethod
    def clustering_scores(clusters_df, feat_df, split_row=None, col_label='RNA Cluster', row_label='',
                         title='Clustering Similarities', figsize_w=3, figsize_h=3, save_figure=False, filename_suffix=''):
        features = sorted(feat_df.columns)
        ari_df = pd.DataFrame(index=feat_df.columns, columns=clusters_df.columns)
        ari_df = ari_df.astype(float)
        for c in clusters_df:
            for f in features:
                ari = adjusted_rand_score(feat_df[f], clusters_df[c]) #Adjusted Rand Index
                ari_df.loc[f, c] = ari
        MyLib.save_csv(ari_df, f'adjusted_rand_scores_{title}{filename_suffix}.csv')
        MyVisualization.plot_heatmap(ari_df, None,
                                   col_label=col_label, title=title,
                                   row_label=row_label, xdend=True, cbar_title='Adjusted Rand Index',
                                   row_cluster=False, ytick_fontsize=7, cbar_CRISPR=False,
                                   left_adjust=0.2, split_row_l=split_row, legend_fontsize=6, title_fontsize=10,
                                   bar_colors_ratio=0.05, cbar_left_adjust=0.015, title_height=.94,
                                   col_label_fontsize=8, row_label_fontsize=8, legend_bar=False,
                                   legend_top_offset=0.7, figsize_w=figsize_w, figsize_h=figsize_h,
                                     save_figure=save_figure, filename_suffix=filename_suffix)

    @staticmethod
    def run_louvain_on_multiple_subsamples(data_df, gammas, sample_fractions_l, subsampling_ite=50,
                                           louvain_ite=100, seed=2023, filename=''):
        """
            Applies subsampling without replacement and executes the Louvain method.
            At each iteration fills out the consensus matrix.
        """
        labels_dic = dict()
        Clustering.init_seed(seed)
        for frac in sample_fractions_l:
            print(f'started {subsampling_ite} runs with fraction {frac}')
            labels_df = pd.DataFrame(-1, index=data_df.index, columns=range(subsampling_ite))
            for i in range(subsampling_ite):
                data_subsampled_df = data_df.sample(frac=frac, replace=False)
                # corr_data_subsampled_df = pd.DataFrame(np.corrcoef(data_subsampled_df.values), columns=data_subsampled_df.index,
                #                                 index=data_subsampled_df.index)
                corr_data_subsampled_df = data_subsampled_df.transpose().corr()
                # if corr_data_subsampled_df.equals(corr_data_subsampled_df2):
                #     print('not equal')
                labels_np, consensus_ite = Clustering.louvain_clustering_ensemble(corr_data_subsampled_df, ite=louvain_ite, gammas=gammas)
                consensus_lbls_np = cluster.find_consensus(labels_np, seed=seed)
                labels_df.loc[data_subsampled_df.index, i] = consensus_lbls_np

            labels_dic[frac] = labels_df
        MyLib.save_pkl(labels_dic, f'{filename}_subsampling_labels_dic.pkl')
        return labels_dic

    @staticmethod
    def louvain_clustering_stability_analysis(downsampling_labels_dic, title='', filename=''):
        sample_fractions_l = sorted(downsampling_labels_dic.keys())
        subsampling_ite = downsampling_labels_dic[sample_fractions_l[0]].shape[1]
        max_no_cluster = max([downsampling_labels_dic[c].max().max() for c in downsampling_labels_dic])
        cluster_counts_l = list(range(1, max_no_cluster + 1))
        freq_df = pd.DataFrame(0, index=sample_fractions_l, columns=cluster_counts_l)
        for frac in downsampling_labels_dic:
            freq_sr = (downsampling_labels_dic[frac].nunique() - 1).value_counts()
            # print(freq_sr)
            freq_df.loc[frac, freq_sr.index] = freq_sr.values
        freq_df = freq_df.loc[:, freq_df.sum(axis=0) > 0].copy()
        color_dic = {key: MyVisualization.feat_colors_dic['Cluster_Count'][key] for key in freq_df.columns if
                     key in MyVisualization.feat_colors_dic['Cluster_Count']}

        MyVisualization.percentage_stacked_barplot(freq_df, x_col='', y_col='', crosstab=False,
                                                   legend_title='Cluster Count', percentage_sign='',
                                                   xlabel='Downsampling Fraction', title=title,
                                                   ylabel=f'Downsampling Runs (n={subsampling_ite})',
                                                   color_dic=color_dic, add_label_count=False, ylabel_fontsize=8,
                                                   xlabel_fontsize=8,
                                                   file_name=filename)

    @staticmethod
    def adjust_rand_index_downsampling_comparisons(main_clustering_sr, downsampling_labels_dic, frac_color_dic,
                                                   original_cluster_count, title_height=1., ylim_bottom=None,
                                                   ylim_top=1.1,
                                                   filename=''):
        res_df = pd.DataFrame(columns=['frac', 'cluster_count', 'ari'])
        count = 0
        for frac in downsampling_labels_dic:
            runs_df = downsampling_labels_dic[frac]
            for i in runs_df.columns:
                cur_run_sr = runs_df[runs_df[i] != -1][i]
                main_lbls_sr = main_clustering_sr.loc[cur_run_sr.index]
                ari = adjusted_rand_score(main_lbls_sr, cur_run_sr)
                res_df.loc[count] = [frac, cur_run_sr.nunique(), ari]
                count += 1
        color_dic = {key: MyVisualization.feat_colors_dic['Cluster_Count'][key] for key in list(range(1, 9))}

        MyVisualization.box_plot(res_df, y_col='ari', x_col='frac', y_label='Adjusted Rand Index',
                                 x_label='Downsampling Fraction',
                                 title='Similarity with Original Clusters', title_fontsize=8,
                                 star_pval=True, title_height=title_height, ylim_top=ylim_top, palette=color_dic,
                                 color_dic=frac_color_dic,
                                 color_col='cluster_count', box_aspect=1.1, add_counts=False, figure_width=4,
                                 figure_height=2,
                                 ylim_bottom=ylim_bottom, alpha_points=0.7, add_legend=True, filename=filename)

    @staticmethod
    def plot_TF_heatmap(data_df, col_upset_df, row_upset_df, labels_np, yticklabels=True, ytick_fontsize=9,
                        xtick_fontsize=9,
                        bar_colors_ratio=0.01, legend_fontsize=9, event_legend=True, event_color=None,
                        clusters_reordering=None, col_label='', row_label='TF', cbar_title='                 ',
                        borders_linewidths=0.01, legend_h_offset=0.06, filter=False, cbar_discrete=False,
                        plot_volcano=True, col_cluster=False, col_name_cluster='Cluster', reverse=True,
                        left_adjust=0.1, legend_top_offset=0.85, cbar_left_adjust=0.06, legend_diff_offset=0.11,
                        volcano_force_points=0.6, volcano_force_text=0.4, top_features=True, cluster_prefix='T',
                        title_consensus_mat='', title_corr_mat='', title_markers_mat='', figsize_w=13, figsize_h=12,
                        save_figure=False):

        ## Row clustering
        consensus_lbls = cluster.find_consensus(labels_np, seed=seed)
        row_reordered_upset_df = Clustering.prepare_labels(row_upset_df, consensus_lbls, clusters_reordering,
                                                           cluster_prefix=cluster_prefix)
        MyLib.save_csv(row_reordered_upset_df, f'{row_label}_{col_name_cluster}_reordered_upset_df.csv')
        data_df = data_df.loc[row_reordered_upset_df.index, :]
        row_reordered_upset_df = row_reordered_upset_df.rename(columns={'Cluster': 'TF_Cluster'})

        ## Column clustering
        if col_cluster:
            corr_df = pd.DataFrame(np.corrcoef(data_df.T.values), columns=data_df.columns, index=data_df.columns)
            gammas = [1, 1.5, 2, 2.5]
            labels_np = Clustering.louvain_clustering_ensemble(corr_df, ite=100, gammas=gammas)
            consensus_lbls = cluster.find_consensus(labels_np, seed=seed)
            col_reordered_upset_df = Clustering.prepare_labels(col_upset_df, consensus_lbls, clusters_reordering,
                                                               cluster_prefix=cluster_prefix)
            MyLib.save_csv(col_reordered_upset_df, f'{col_label}_reordered_upset_df.csv')
            data_df = data_df.loc[:, col_reordered_upset_df.index]

        else:
            # sample_ids = Clustering.sort_samples(data_df.T, col_upset_df, reverse, cluster_column=col_name_cluster)
            # data_df = data_df.loc[:, sample_ids]
            data_df = data_df.loc[:, col_upset_df.index]

        MyVisualization.plot_heatmap(data_df, upset_df=col_upset_df, row_upset_df=row_reordered_upset_df,
                                     borders_linewidths=borders_linewidths,
                                     cbar_title=cbar_title, xdend=True, yticklabels=yticklabels,
                                     ytick_fontsize=ytick_fontsize, xtick_fontsize=xtick_fontsize,
                                     event_color=event_color, row_label=row_label, col_label=col_label,
                                     title=title_consensus_mat,
                                     legend_top_offset=legend_top_offset, left_adjust=left_adjust,
                                     cbar_left_adjust=cbar_left_adjust, legend_diff_offset=legend_diff_offset,
                                     legend_h_offset=legend_h_offset,
                                     figsize_w=figsize_w, figsize_h=figsize_h, save_figure=save_figure)

#========
Clustering.init_seed(seed)
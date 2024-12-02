import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import ttest_ind
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection


from scripts.io_library import MyLib
from scripts.visualization import MyVisualization

seed = 2023
class DifferentialDependency:

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
    def apply_test(col_info):
        np.random.seed(seed)
        col, cluster_data, other_clusters_data, test = col_info
        p_values = []

        if test == 'ttest':
            stat, p_value = ttest_ind(cluster_data[col], other_clusters_data[col])
            effect_size = DifferentialDependency.cohend_effect_size(cluster_data[col], other_clusters_data[col])
        else:# test == 'ranksum':
            stat, p_value = ranksums(cluster_data[col], other_clusters_data[col])#, nan_policy='omit'
            effect_size = stat / np.sqrt(len(cluster_data) + len(other_clusters_data))

        p_values.append((col, p_value, effect_size))
        return p_values

    @staticmethod
    def identify_significant_features_clusters(data_df, upset_df, cluster_column='Cluster', test='ttest'):


        cluster_ids = upset_df[cluster_column].unique()

        for cluster_id in cluster_ids:
            cluster_data = data_df[upset_df[cluster_column] == cluster_id]
            other_clusters_data = data_df[upset_df[cluster_column] != cluster_id]

            columns = data_df.columns
            col_info = [(col, cluster_data, other_clusters_data, test) for col in columns]

            with Pool() as pool:
                print(pool._processes,'-------------------------------')
                results = pool.map(DifferentialDependency.apply_test, col_info)

            p_values = [item for sublist in results for item in sublist]

            res_df = pd.DataFrame(p_values, columns=['feature', 'p-value', 'effect_size'], dtype=np.float128)
            # res_df['q-value'] = fdrcorrection(res_df['p-value'].values)[1]
            res_df.to_csv(f'output_files/{cluster_id}_features_stats.csv', index=False)

    @staticmethod
    def differential_dependency(data_df, upset_df, test, group, filename=''):
        print(group)
        cluster_data = data_df[upset_df[group] == 1]
        other_clusters_data = data_df[upset_df[group] != 1]

        drop_cols = [col for col in cluster_data.columns if cluster_data[col].nunique() == 1]
        cluster_data = cluster_data.drop(columns=drop_cols)
        other_clusters_data = other_clusters_data.drop(columns=drop_cols)

        columns = cluster_data.columns
        col_info = [(col, cluster_data, other_clusters_data, test) for col in columns]

        with Pool() as pool:
            print('Number of processes:', pool._processes, '-------------------------------')
            results = pool.map(DifferentialDependency.apply_test, col_info)

        p_values = [item for sublist in results for item in sublist]

        res_df = pd.DataFrame(p_values, columns=['feature', 'p-value', 'effect_size'])
        res_df['q-value'] = fdrcorrection(res_df['p-value'].values)[1]

        in_group_data_sr = cluster_data.mean()
        in_group_data_sr.name = 'in_group_gene_effect_mean'

        out_group_data_sr = other_clusters_data.mean()
        out_group_data_sr.name = 'out_group_gene_effect_mean'

        res_df.index = res_df['feature']
        res_df = pd.concat([res_df, in_group_data_sr, out_group_data_sr], axis=1)
        res_df.reset_index(inplace=True, drop=True)

        res_df = res_df.sort_values('effect_size', ascending=False).sort_values('q-value', ascending=True)
        MyLib.save_csv(res_df, f'{group}_{filename}all_features_stats.csv', index=False)
        return res_df

    @staticmethod
    def select_differential_dependency(res_df, gene, in_group_col, q_val_thr=0.1, p_val_thr=0.05, gene_effect_thr=-0.5,
                                       plot_qqplot=True, force_points=1, force_text=1, top_labels_cutoff=0, cut_off_texts=10,
                                       title='', xtick_step=None, ytick_step=None, xlim_right=None, xlim_left=None,
                                       ylim_top=None, ylim_bottom=None, return_features=False, save_figure=False):

        temp_df = None
        feature_cutoff = np.inf
        up_regulated = False
        top_df = None
        res_df['feature'] = res_df['feature'].apply(lambda x: x.split(' (')[0])
        res_df = res_df.dropna(axis=1, how='all')
        res_df = res_df.dropna(axis=0)
        # res_df = res_df[res_df['feature'].isin(gene_effect_df.columns.tolist())]
        res_df['q-value'] = fdrcorrection(res_df['p-value'].values)[1]

        res_filtered_df = res_df[res_df.apply(lambda x: (x['p-value'] < p_val_thr and
                                                         x['q-value'] < q_val_thr), axis=1)]

        if plot_qqplot:
            MyVisualization.qqplot(res_df['p-value'].tolist(), title=gene)

        if res_filtered_df.shape[0] > 0:
            cut_off_labels = cut_off_texts
            up_df = None
            down_df = None
            if up_regulated:
                regulated = 'Up'
                up_df = res_filtered_df[res_filtered_df['effect_size'] >= 0].sort_values('effect_size',
                                                                                         ascending=False) \
                    .sort_values('q-value', ascending=True)
                cur_feature_cut_off = min(up_df.shape[0], feature_cutoff)
                up_df = up_df.iloc[:cur_feature_cut_off, ]
                cut_off_labels = min(cut_off_labels, up_df.shape[0])
                temp_df = up_df.copy()
                up_df['feature'] = up_df['feature'].apply(lambda x: x.split(' (')[0])

            else:
                regulated = 'Down'
                down_df = res_filtered_df[res_filtered_df['effect_size'] < 0].sort_values('effect_size',
                                                                                          ascending=True) \
                    .sort_values('q-value', ascending=True)

                cur_feature_cut_off = min(down_df.shape[0], feature_cutoff)
                down_df = down_df.iloc[:cur_feature_cut_off, ]
                if gene_effect_thr is not None:
                    group_data_l = res_df[res_df[in_group_col] < gene_effect_thr]['feature'].tolist()
                    top_df = down_df[~down_df['feature'].isin(group_data_l)].head(top_labels_cutoff)
                    down_df = down_df[down_df['feature'].isin(group_data_l)]
                cut_off_labels = min(cut_off_labels, down_df.shape[0])
                temp_df = down_df.copy()
                down_df['feature'] = down_df['feature'].apply(lambda x: x.split(' (')[0])

            # volcano plot
            if (~up_regulated and down_df.shape[0] > 0) or (up_regulated and up_df.shape[0] > 0):
                MyVisualization.Volcano_plot(res_df, x_col='effect_size', y_col='q-value',
                                             title=gene.split(' (')[0] + ' ' + title,
                                             x_label='Effect size', label_col='feature', down_df=down_df,
                                             xtick_step=xtick_step, ytick_step=ytick_step, top_df=top_df,
                                             xlim_right=xlim_right, xlim_left=xlim_left, ylim_top=ylim_top,
                                             ylim_bottom=ylim_bottom,
                                             force_points=force_points, force_text=force_text, up_df=up_df,
                                             cut_off_labels=cut_off_labels,
                                             point_size=20,
                                             save_figure=save_figure)
                temp_df.set_index('feature', inplace=True, drop=True)

                # features_l += [f for f in temp_df.index.tolist() if f not in features_l]
                MyLib.save_csv(temp_df, f'{gene}_features_{regulated}_{title}.csv')
        if return_features and temp_df is not None:
            features_l = temp_df.sort_values(by=in_group_col).index.tolist()
            return features_l
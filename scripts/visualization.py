import os.path
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import gcf
import matplotlib.gridspec as gridspec
from adjustText import adjust_text
import mplcursors
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap, to_hex
from scipy.stats import beta
import networkx as nx
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from statannotations.Annotator import Annotator
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage
import forestplot as fp
import warnings
import umap
import re



import scripts.io_library as io_library

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
seed = 2023
np.random.seed(seed)

class MyVisualization:
    feat_colors_dic = {'Subtype': {'AC': "#c9bb3a", 'ECC': "#35978f", 'GBC': "#9970ab",
                                          'HCC': "#fa9fb5", 'ICC': "#bf812d"},
                       'RNA_Subtype': {'Bipotential': '#225ea8', 'Bile-duct like': '#4daf4a',
                                              'Hepatocyte like': '#fa9fb5', 'EMT': '#e2e82c'},
                       'Cluster': {'C1':'#4383fa', 'C2':"#4ae89e", 'C3':"#9342f5", 'C4':"#f7e72f",  #"#9e6ab8",
                                   'C5':"#f03043", 'C6':"#615e61", 'C7':'#f8fc03', 'C8':'#CEE86D',
                                   'D1': '#A8E2EF', 'D2': "#E8E38D", 'D3': "#F4B7F5", 'D4': "#D9EC94",  # "#9e6ab8",
                                   'D5': "#B2A0DE", 'D6': "#FD8A48", 'D5+D6':"#4ae89e",
                                   'R1': '#10118C', 'R2': "#7FCBFF", 'R3': "#FF785C", 'R4': "#F7E72F",

                                   'P1': '#CD41AC', 'P2': "#8AADAA", 'P3': "#6C498A", 'P4': "#16580A",
                                   'P5': "#BC6764",
                                   'T1': '#4383fa', 'T2': "#4ae89e", 'T3': "#9342f5", 'T4': "#f7e72f"
                                   },#''
                       'TF_Cluster': {'T3': '#225ea8',  'T1': '#8c4574', 'T2': '#5aad87', 'T4': '#7f8899', 'T5': '#37A5EE',
                                      },
                       'Event': {'NONSENSE': '#07578D', 'FRAME_SHIFT_DEL': '#458a2c', 'FRAME_SHIFT_INS': '#8391f7',
                                     'MISSENSE':'#f586f7', 'SILENT':  '#37A5EE', 'SPLICE_SITE':  '#c73316',
                                     'IN_FRAME_INS':'#758e4f', 'NONSTOP': '#F2FF49', 'IN_FRAME_DEL': '#815355',
                                 'FUSION': '#8D0741'},
                       'RNA_Cluster': {'R1': '#10118C', 'R2': "#7FCBFF", 'R3': "#FF785C", 'R4': "#F7E72F"},
                       'Disease':{'BTC':'#2C4E78', 'Other CCLE':'#CEDEE0'},
                       'Cluster_Count': {1: '#07578D', 2: '#458a2c', 3: '#004A17',
                                         4: '#0A003F', 5: '#C9AA00', 6: '#831748',
                                         7: '#758E45', 8: '#DE760D', 9: '#815355',
                                         10: '#8D0741'},
                       'Lineage': {"bile duct": "#fb6a4a", "esophagus": "#E190D6",
                                   "gastric": "#62BE79", "liver": "#D8A06A",
                                   "colorectal": "#4D4D4D", "pancreas": "#48B8DE"}
                       }

    @staticmethod
    def get_stat_text(stats_value, sig_threshold=0.05, star=False, fdr=False, ns_text=True):
        if star:
            if stats_value > sig_threshold:
                stats_text = 'ns'
            elif stats_value < 0.0001:
                stats_text = '****'
            elif stats_value < 0.001:
                stats_text = '***'
            elif stats_value < 0.01:
                stats_text = '**'
            elif stats_value < sig_threshold:
                stats_text = '*'
        else:
            if fdr:
                stats_text = 'q'
            else:
                stats_text = 'p'
            if stats_value < 0.0001:
                stats_text += '<0.0001'  # {p_value:.3f}
            elif (stats_value > sig_threshold) and ns_text:
                stats_text = 'ns'
            elif stats_value < 0.001:
                stats_text += '<0.001'
            else:
                stats_text += f'={stats_value:.2f}' if int(stats_value * 1000) % 10 == 0 else f'={stats_value:.3f}'

        return stats_text

    @staticmethod
    def plot_distribution(data_sr, title='', x_vline=None):
        plt.style.use('default')
        plt.figure(figsize=(5, 3), facecolor="w")
        plt.grid(False)
        sns.kdeplot(data_sr, shade=True)
        if x_vline is not None:
            plt.axvline(x=x_vline, color='#A00000', linestyle='--')
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_catplot(data_df, x_col, xlabel, ylabel, title='', ytick_step=None, title_height=1,
                     tick_fontsize=None, ylim=None, title_fontsize=10, color_code='#a63603',
                     figure_width=5, figure_height=5, save_figure=False):
        plt.style.use('default')
        # sns.set(style='ticks', context='talk')
        # plt.figure(figsize=(figure_width, figure_height))
        plt.figure(figsize=(figure_width, figure_height))

        ax = sns.countplot(data=data_df, x=x_col, color=color_code, edgecolor='w')

        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.suptitle(title, fontsize=title_fontsize, y=title_height)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.xticks(rotation=45, ha='right')
        if ytick_step is not None:
            ax.yaxis.set_major_locator(plt.MultipleLocator(ytick_step))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_ylim(top=ylim)
        plt.grid(False)
        if save_figure:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"catplot.pdf"),
                        dpi=60, bbox_inches="tight")
        plt.show()

    @staticmethod
    def scatter_plot_comparison(X_data_sr, Y_data_sr, x_label, y_label, title, text_fontsize=12, title_fontsize=14,
                                add_text=False, labels_text=None, labels_ineq="<=", force_points=1, force_text=.5,
                                alpha_points=1,
                                y_threshold=None, x_threshold=None, label_fontsize=12, tick_fontsize=10,
                                ylim_top=None,
                                ylim_bottom=None, xlim_right=None, vline_x=None, legend_loc=False, legend_num=5,
                                colors_sr="#82A4E2", points_size=40, size_scale=1, metric=None,
                                legend_classes_dic=None,
                                add_legend=True, arrow_color='black', legend_fontsize=8, legend_title_fontsize=10,
                                legend_marker_size=50, legend_size_dic=None, legend_anchor_y=1, legend_anchor_x=1,
                                legend_anchor_x_offset=0, legend_label='', add_spines=False, font_scale=1,
                                filename=None,
                                figure_width=8, figure_height=4, filename_suffix='', save_file=False):
        plt.style.use('default')
        plt.close('all')
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['pdf.fonttype'] = 42
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        sns.set(font_scale=font_scale)

        if legend_classes_dic is not None:
            sc = plt.scatter(
                X_data_sr,
                Y_data_sr,
                color=colors_sr,
                edgecolor="none",
                alpha=alpha_points,
                s=points_size * size_scale
            )
            temp_dic = legend_classes_dic.copy()
            temp_sc = None
            for ele in temp_dic:
                cur_index = [i for i, c in zip(colors_sr.index, colors_sr) if c == legend_classes_dic[ele]]
                if len(cur_index) == 0:
                    legend_classes_dic.pop(ele)
                    continue
                c = [colors_sr[index] for index in cur_index]
                if not isinstance(alpha_points, int):
                    alpha_points_temp = alpha_points[cur_index]
                else:
                    alpha_points_temp = alpha_points
                sc = plt.scatter(
                    X_data_sr.loc[cur_index],
                    Y_data_sr.loc[cur_index],
                    color=c,
                    # edgecolor="black",
                    alpha=alpha_points_temp,
                    s=points_size[cur_index] * size_scale,
                    label=ele
                )
                if ele == list(temp_dic.keys())[-1]:  # To retrieve legend for Cell_line--PDX plot
                    temp_sc = sc
            plt.legend(numpoints=1, fontsize=5, frameon=False)

        else:
            sc = plt.scatter(
                X_data_sr,
                Y_data_sr,
                color=colors_sr,
                # edgecolor="navy",
                alpha=alpha_points,
                s=points_size * size_scale
            )

        if metric == 'corr':
            max1 = X_data_sr.max()
            max2 = Y_data_sr.max()
            max_val = max(max1, max2) + .1
            min1 = X_data_sr.min()
            min2 = Y_data_sr.min()
            min_val = min(min1, min2) - .1
            upper_bound = max_val
            # max = 1
            lower_bound = min_val
            # lower_bound = -1
            plt.plot([lower_bound, upper_bound], [lower_bound, upper_bound], '--', color='black')
            # plt.xlim(-.01, 1)
            # plt.ylim(-.01, 1)
            plt.xlim(lower_bound, upper_bound)
            plt.ylim(lower_bound, upper_bound)
        if vline_x is not None:
            plt.axvline(vline_x, color='k', linestyle="--", linewidth=.8)  # color="grey",
        plt.title(title, fontsize=title_fontsize)
        plt.xlabel(x_label, fontsize=label_fontsize)
        plt.ylabel(y_label, fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(False)
        plt.ylim(top=ylim_top, bottom=ylim_bottom)
        plt.xlim(right=xlim_right)

        if add_text:
            texts = []
            for index in Y_data_sr.index:
                if labels_text is not None:
                    if index in labels_text:
                        texts.append(plt.text(X_data_sr[index], Y_data_sr[index], index,
                                              fontsize=text_fontsize))  # , alpha=0.8
                elif y_threshold is None and x_threshold is None:
                    if eval(f'{X_data_sr[index]} {labels_ineq} {Y_data_sr[index]}'):
                        texts.append(plt.text(X_data_sr[index], Y_data_sr[index], index, fontsize=text_fontsize))
                elif y_threshold is not None and x_threshold is not None:
                    if eval(f'{X_data_sr[index]} {labels_ineq} {x_threshold}') and \
                            eval(f'{y_threshold} {labels_ineq} {Y_data_sr[index]}'):
                        texts.append(plt.text(X_data_sr[index], Y_data_sr[index], index, fontsize=text_fontsize))

            adjust_text(texts, force_points=force_points, force_text=force_text,
                        only_move={'points': 'yx', 'texts': 'yx'},
                        autoalign=True,
                        arrowprops=dict(arrowstyle="-", color=arrow_color, alpha=1, lw=0.2)
                        )
        if not add_spines:
            sns.despine(top=True, right=True)
        mplcursors.cursor(hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(X_data_sr.index.tolist()[sel.index]))
        # if legend_classes_dic is not None:
        #     plt.legend(labels=legend_classes_dic.keys,
        #                title="species")

        if add_legend:
            ax = plt.gca()
            if legend_classes_dic is None:
                if isinstance(legend_num, list):
                    legend_num = [(percentage * size_scale) / 100 for percentage in legend_num]
                handles, labels = sc.legend_elements("sizes", num=legend_num)

                for index, handle in enumerate(handles):
                    handle.set_color('#d4d4d4')
                    handle.set_alpha(1)
                    match = re.search(r'\d+', labels[index])
                    extracted_number = int(match.group())
                    labels[index] = f"{(math.ceil((extracted_number / size_scale) * 100))}"
                    # handle.set_alpha(alpha_points)

                legend = ax.legend(handles, labels, loc='upper right', frameon=False,
                                   # bbox_to_anchor=(legend_anchor_x-legend_anchor_x_offset , legend_anchor_y),
                                   labelspacing=.2)
                legend.set_title(legend_label)
                legend.get_title().set_fontsize(legend_title_fontsize)
                legend.set_frame_on(False)
            else:  # bbox_to_anchor=(legend_anchor_x- 0.15, legend_anchor_y),
                if legend_loc:
                    legend = ax.legend(title=legend_label, facecolor="white", loc='upper left',
                                       frameon=False, ncol=1, fontsize=legend_fontsize)
                else:
                    legend = ax.legend(title=legend_label, facecolor="white",
                                       bbox_to_anchor=(legend_anchor_x - legend_anchor_x_offset, legend_anchor_y),
                                       frameon=False, ncol=1, fontsize=legend_fontsize)
                legend.get_title().set_fontsize(legend_title_fontsize)

                for handle in legend.legendHandles:
                    handle.set_sizes([legend_marker_size])
                    # handle.set_alpha(alpha_points)

                if legend_size_dic is not None:
                    handles_size, _ = temp_sc.legend_elements("sizes", num=legend_num)
                    legend_lbls = [f"{s}" for s in legend_size_dic.keys()]
                    # handles_size.reverse()
                    for handle in handles_size:
                        handle.set_color('#d4d4d4')
                    # handles_size = [mlines.Line2D([], [], marker='o', color='gray', markerfacecolor='silver',
                    #                               markersize=legend_size_dic[lbl]*2, label=lbl) for lbl in
                    #                 legend_size_dic]

                    legend_size = ax.legend(handles_size, legend_lbls, loc='upper left', facecolor="white",
                                            frameon=False,
                                            bbox_to_anchor=(legend_anchor_x, legend_anchor_y + 0.01),
                                            labelspacing=.8)
                    legend_size.set_title('K')
                    legend_size.get_title().set_fontsize(legend_title_fontsize)
                    legend_size.set_frame_on(False)

                    # Add the second legend without overwriting the first one
                    ax.add_artist(legend)
                    ax.add_artist(legend_size)

        plt.tight_layout()
        if save_file:
            plt.savefig(os.path.join(io_library.output_dir,f"Scatterplot_comparison_{filename}.pdf"), dpi=600)
            print(f'saved {filename}.pdf')
        else:
            plt.show()

    @staticmethod
    def ridge_plot(df, color_dic, color_rug_dic, col_level, col_score, col_hue, sharey=True, xlim_left=-3.5,
                   xlim_right=0.7, ylim=None, dist_alpha=1, aspect=3, x_pval=0.11, p_value=None, p_value_l=None,
                   p_value_dic=None, sig_thr=0.06, fdr=False, add_legend=True,
                   xtick_step=None, levels_fontsize=6.5, label_fontsize=8, tick_fontsize=7, title_fontsize=11,
                   x_vline=-0.5,
                   vline_offset=0.25, rug_plot_height=0.4, mean_line=False, median_line=False, title='', border=False,
                   figure_width=3, figure_height=2, file_name=None):
        # plt.figure(figsize=(4,3))
        fig = plt.figure(figsize=(figure_width, figure_height))
        palette = {}
        for v in color_dic.values():
            palette.update(v)
        unique_levels = list(color_dic.keys())
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth': 0})
        g = sns.FacetGrid(df, row=col_level, hue=col_hue, aspect=aspect, height=1.25,
                          palette=palette, sharey=sharey,
                          row_order=unique_levels, hue_order=list(palette.keys()))
        g.map_dataframe(sns.kdeplot, x=col_score, fill=True, alpha=dist_alpha)
        # g.map(sns.kdeplot, x=col_score, fill=True, alpha=dist_alpha)
        if border:
            # g.map(sns.kdeplot, x=col_score, color='black', linewidth=0.6)
            g.map_dataframe(sns.kdeplot, x=col_score, color='black', linewidth=0.6)

        g.fig.subplots_adjust(hspace=.1)  # Adjust the vertical space between plots
        g.set_xlabels(fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        for level, ax in zip(unique_levels, g.axes.flat):
            if col_level == col_hue:
                size_n = df[df[col_level] == level].shape[0]
                label_text = level + f' (n={size_n})'
            else:
                temp_df = df[df[col_level] == level]
                unique_hues = temp_df[col_hue].unique()
                if len(unique_hues) == 2:
                    n1 = temp_df[temp_df[col_hue] == unique_hues[0]].shape[0]
                    n2 = temp_df[temp_df[col_hue] == unique_hues[1]].shape[0]
                    min_size = min(n1, n2)
                    percentage = min_size / temp_df.shape[0] * 100
                    label_text = level + f' (MT={round(percentage)}%)'
                else:
                    min_size = np.inf
                    for hue_gp in unique_hues:
                        cur_df = temp_df[temp_df[col_hue] == hue_gp]
                        min_size = min(min_size, cur_df.shape[0])
                    label_text = level + f' (m={min_size})'
            if p_value_dic is not None:
                ax.text(0, .4, label_text + ' ' + MyVisualization.get_stat_text(p_value_dic[level]), color='black',
                        fontsize=levels_fontsize, ha="left", va="center", transform=ax.transAxes)
            else:
                ax.text(0, .4, label_text, color='black', fontsize=levels_fontsize,
                        ha="left", va="center", transform=ax.transAxes)
            ax.set(ylabel='')

        # Adjust the height and ylim of each subplot based on the number of samples in each distribution
        if ylim is not None:
            for level in ylim:  # zip(unique_levels, g.axes.flat)
                g.axes_dict[level].set(ylim=ylim[level])

        if p_value_l is not None:
            if len(p_value_l) == 1:
                height = len(unique_levels)
            else:
                height = 0.03
            # print('p-value', p_value_l[0])
            plt.annotate(MyVisualization.get_stat_text(p_value_l[0], sig_threshold=sig_thr, fdr=fdr),
                         xy=(0.09, height),
                         xycoords='axes fraction',
                         ha='center',
                         fontsize=levels_fontsize,
                         bbox=dict(boxstyle='round', alpha=0.1, facecolor='white'))
            if len(p_value_l) > 1:
                # print('p-value', p_value_l[1])
                plt.annotate(MyVisualization.get_stat_text(p_value_l[1], sig_threshold=sig_thr),
                             xy=(0.1, 1.15),
                             xycoords='axes fraction',
                             ha='center',
                             fontsize=levels_fontsize,
                             bbox=dict(boxstyle='round', alpha=0.1, facecolor='white'))
        if p_value is not None:
            height = len(unique_levels)
            plt.annotate('MT rate: ' + MyVisualization.get_stat_text(p_value),
                         xy=(x_pval, height),
                         xycoords='axes fraction',
                         ha='center',
                         fontsize=levels_fontsize,
                         bbox=dict(boxstyle='round', alpha=0.1, facecolor='white'))

        g.set_titles("")
        g.set(yticks=[], xlabel=col_score)
        g.despine(left=True)

        # Add rug plot
        # unique_hue = list(df[col_hue].unique())
        for level in unique_levels:
            ax = g.axes_dict[level]  # Access of the current level
            cur_df = df[df[col_level] == level]
            offset = 0
            for gp in color_dic[level].keys():
                sample = cur_df.loc[cur_df[col_hue] == gp, col_score]
                ymax = -(rug_plot_height + offset)
                ax.vlines(sample, ymin=-(0 + offset), ymax=ymax, color=color_rug_dic[level][gp], alpha=dist_alpha)
                if col_hue != col_level:
                    offset += rug_plot_height
                else:
                    offset += 0
                if mean_line:
                    median_val = sample.mean()
                    ax.axvline(x=median_val, linestyle='-', color='black', alpha=0.3)
                elif median_line:
                    median_val = sample.median()
                    ax.axvline(x=median_val, linestyle='-', color='black', alpha=0.3)
            if col_hue != col_level:
                base_y = ymax + offset - 0.02
            else:
                base_y = ymax + rug_plot_height
            ax.hlines(y=base_y, xmin=xlim_left, xmax=xlim_right, color='black', linewidth=0.6)

        for ax in g.axes.flat:
            ax.axvline(x=x_vline, ymin=base_y + vline_offset, linestyle='--', color='k', linewidth=0.6)
            # ax.axvline(x=-1, ymin=base_y+vline_offset, linestyle='--', color='k', linewidth=0.6)
        if add_legend:
            legend_labels = [
                plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                for label, color in palette.items()]
            legend = plt.legend(handles=legend_labels, title=col_hue, bbox_to_anchor=(1.05, 1), loc='upper left',
                                frameon=False)
            legend.get_frame().set_facecolor('white')
            for handle, label in zip(legend.legend_handles, palette.keys()):
                handle.set_color(palette[label])
        plt.subplots_adjust(right=0.85)
        plt.xlim(xlim_left, xlim_right)

        if xtick_step is not None:
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
        plt.suptitle(title, y=0.98, fontsize=title_fontsize)
        # plt.show()
        if file_name is not None:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.gcf().set_size_inches(3, 2)
            plt.savefig(os.path.join(io_library.output_dir, f'{file_name}_ridge_plot.pdf'), dpi=50, bbox_inches="tight")
            print(os.path.join(io_library.output_dir, f'{file_name}_ridge_plot.pdf'))

    @staticmethod
    def qqplot(p_values_sr, ci=0.95, figure_width=4, figure_height=3, title='', ylim_top=None):
        plt.style.use('default')
        plt.figure(figsize=(figure_width, figure_height))
        # sns.set(style='ticks', context='talk')
        n = len(p_values_sr)
        df = pd.DataFrame({
            'observed': -np.log10(np.sort(p_values_sr)),
            'expected': -np.log10(np.linspace(1 / n, 1, n)),
            'clower': -np.log10(beta.ppf((1 - ci) / 2, np.arange(1, n + 1), np.arange(n, 0, -1))),
            'cupper': -np.log10(beta.ppf((1 + ci) / 2, np.arange(1, n + 1), np.arange(n, 0, -1)))
        })

        log10Pe = r'$Expected\ -log_{10}(P)$'
        log10Po = r'$Observed\ -log_{10}(P)$'

        plt.fill_between(df['expected'], df['clower'], df['cupper'], alpha=0.1, color='#C1914B')
        plt.scatter(df['expected'], df['observed'], marker='o', facecolors='none', edgecolors='#a63603', s=15,
                    label='Observed')
        plt.plot(df['expected'], df['expected'], '--', color='gray', alpha=0.5, label='Expected')
        plt.xlabel(log10Pe)
        plt.ylabel(log10Po)
        plt.title(title)
        plt.grid(False)
        if ylim_top is not None:
            plt.ylim(top= ylim_top, bottom=0)
        plt.legend()
        plt.show()

    @staticmethod
    def Volcano_plot(data_df, y_col, x_col, color_col=None, title='', x_label='', label_col=None, force_points=0.6,
                     force_text=0.4, xlim_right=None, xlim_left=None, ylim_top=None, ylim_bottom=None,
                     cut_off_labels=np.inf, cut_off_labels_down=np.inf, up_color='#077f97', down_color='#a00000',
                     down_df=None, up_df=None, add_label=True, label_fontsize=8, top_df=None, alpha_top=1,
                     xtick_step=None, ytick_step=None, yspine_center=False, add_arrow=True, y_hline=None, point_size=32,
                     arrow_color='black', tick_fontsize=9, axis_label_fontsize=10, title_fontsize=12, figure_width=3,
                     figure_height=3, save_figure=False, filename_suffix=''):

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor="w")

        plt.scatter(x=data_df[x_col], y=data_df[y_col].apply(lambda x: -np.log10(x)), s=point_size,
                    c='#d4d4d4', alpha=0.4, edgecolor='none')

        texts = []
        if color_col is not None:
            subset_df = data_df.dropna()
            if add_label:
                for i, r in subset_df.iterrows():
                    texts.append(
                        plt.text(x=r[x_col], y=-np.log10(r[y_col]), s=r[label_col], c=data_df.loc[i, color_col],
                                 fontsize=label_fontsize))

        if down_df is not None:
            if color_col is not None:
                down_df = down_df[~down_df[label_col].isin(subset_df[label_col])]
            # down_df[y_col] = down_df[y_col].apply(lambda x:max(x, 1e-10))
            plt.scatter(x=down_df[x_col],
                        y=down_df[y_col].apply(lambda x: -np.log10(x)),
                        s=point_size, alpha=alpha_top,
                        color=down_color,
                        edgecolor='none')
            if add_label:
                count = 0
                for i, r in down_df.iterrows():
                    if count < min(cut_off_labels, cut_off_labels_down):
                        if color_col is None or pd.isna(data_df.loc[i, color_col]):
                            texts.append(plt.text(x=r[x_col], y=-np.log10(r[y_col]), s=r[label_col],
                                                  fontsize=label_fontsize))
                            count += 1
                    else:
                        break

        if up_df is not None:
            if color_col is not None:
                up_df = up_df[~up_df[label_col].isin(subset_df[label_col])]
            # up_df[y_col] = up_df[y_col].apply(lambda x: max(x, 1e-10))
            plt.scatter(x=up_df[x_col], y=up_df[y_col].apply(lambda x: -np.log10(x)),
                        s=point_size, color=up_color,
                        alpha=alpha_top, edgecolor='none')
            if add_label:
                count = 0
                for i, r in up_df.iterrows():
                    if count < cut_off_labels:
                        if color_col is None or pd.isna(data_df.loc[i, color_col]):
                            texts.append(
                                plt.text(x=r[x_col], y=-np.log10(r[y_col]), s=r[label_col], fontsize=label_fontsize))
                            count += 1
                    else:
                        break

        if (top_df is not None) and add_label:
            for i, r in top_df.iterrows():
                texts.append(plt.text(x=r[x_col], y=-np.log10(r[y_col]), s=r[label_col], fontsize=label_fontsize))

        if color_col is not None:
            plt.scatter(x=subset_df[x_col], y=subset_df[y_col].apply(lambda x: -np.log10(x)),
                        s=point_size, c=subset_df[color_col].tolist(), edgecolor='none')
        if add_label:
            if add_arrow:
                adjust_text(texts, force_points=force_points, force_text=force_text,
                            only_move={'points': 'xy', 'texts': 'xy'},
                            # autoalign=True,
                            arrowprops=dict(arrowstyle="-", color=arrow_color, alpha=1, lw=0.2)
                            )
            else:
                adjust_text(texts, force_points=force_points, force_text=force_text,
                            only_move={'points': 'xy', 'texts': 'xy'}
                            )

        # y_label = f"-$\log_{{10}}$ Adj $P$"
        y_label = "-log10 q"
        x_label = x_label if x_label != 'logFC' else "log2 FC"
        # x_label = x_label if x_label != 'logFC' else f"$\log_{{2}}$FC"
        # plt.axvline(-0.2, color="grey", linestyle="--")
        # plt.axvline(0.2, color="grey", linestyle="--")
        if y_hline:
            plt.axhline(y_hline, color="k", linestyle="--", linewidth=.7)
        # plt.legend()

        # _, end = ax.get_xlim()
        # plt.xlim(right=round(end))
        # plt.ylim(top=ylim_top)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        if yspine_center:
            ax.spines.left.set_position('zero')
            ax.tick_params(axis='y', left=True, direction='inout', length=10)

        # ax.spines.bottom.set_position('zero')
        if ytick_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
        if xtick_step is not None:
            ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
        ax.tick_params(length=4.6)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        plt.xlim(right=xlim_right, left=xlim_left)
        plt.ylim(top=ylim_top, bottom=ylim_bottom)
        plt.xlabel(x_label, fontsize=axis_label_fontsize)
        plt.ylabel(y_label, fontsize=axis_label_fontsize)
        plt.title(title, fontsize=title_fontsize)  # , fontweight="bold"
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(False)

        # ax = plt.gca()

        # mplcursors.cursor(hover=True).connect(
        #     "add", lambda sel: sel.annotation.set_text(data_df['pair'].tolist()[sel.index]))
        if save_figure:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            title = title.replace('\n', '_')
            plt.savefig(os.path.join(io_library.output_dir, f"volcano_{title}{filename_suffix}.pdf"),
                        dpi=50, bbox_inches="tight")
        # else:
        plt.show()
        plt.clf()
        plt.close()

    @staticmethod
    def Volcano_plot_v2(data_df, y_col, x_col, color_col=None, text_color_col=None,
                        points_size=7, size_scale=1, points_labels_sr=None, title='', x_label='', alpha_points=.7,
                        text_fontsize=9,
                        force_points=0.6, force_text=0.4, xlim_right=None, xlim_left=None, ylim_top=None,
                        ylim_bottom=None, hline_sig_cutoff=None,
                        add_label=True, xtick_step=None, ytick_step=None, yspine_center=False, add_arrow=True,
                        arrow_color='black', tick_fontsize=9, xlabel_fontsize=10, ylabel_fontsize=10, legend_num=4,
                        add_legend=False, legend_title='', legend_title_fontsize=8, legend_anchor_x=1,
                        legend_anchor_y=.5,
                        title_fontsize=10, save_figure=False):

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(3, 3), facecolor="w")

        sc = plt.scatter(x=data_df[x_col],
                         y=data_df[y_col].apply(lambda x: -np.log10(x)),
                         s=points_size * size_scale, alpha=alpha_points, edgecolor="none",
                         c=data_df[color_col].tolist())  #

        texts = []
        if add_label:
            subset_df = data_df.dropna()
            if not isinstance(points_size, int):
                points_size = points_size[subset_df.index]
            plt.scatter(x=subset_df[x_col],
                        y=subset_df[y_col].apply(lambda x: -np.log10(x)),
                        s=points_size * size_scale,
                        c=subset_df[color_col].tolist(), alpha=0.4,
                        edgecolor="none")
            for i, r in subset_df.iterrows():
                texts.append(plt.text(x=r[x_col], y=-np.log10(r[y_col]), s=points_labels_sr[i], c=r[text_color_col],
                                      fontsize=text_fontsize))
            if add_arrow:
                adjust_text(texts, force_points=force_points, force_text=force_text,
                            only_move={'points': 'xy', 'texts': 'xy'},
                            # autoalign=True,
                            arrowprops=dict(arrowstyle="-", color=arrow_color, alpha=1, lw=0.2)
                            )
            else:
                adjust_text(texts, force_points=force_points, force_text=force_text,
                            only_move={'points': 'xy', 'texts': 'xy'}
                            )

        # y_label = f"-$\log_{{10}}$ Adj $P$"
        # x_label = x_label if x_label != 'logFC' else f"$\log_{{2}}$FC"
        y_label = "-log10 q"
        x_label = x_label if x_label != 'logFC' else "log2 FC"
        # plt.axvline(-0.2, color="grey", linestyle="--")
        # plt.axvline(0.2, color="grey", linestyle="--")
        # plt.axhline(2, color="k", linestyle="--")
        # plt.legend()

        # _, end = ax.get_xlim()
        # plt.xlim(right=round(end))
        # plt.ylim(top=ylim_top)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        if yspine_center:
            ax.spines.left.set_position('zero')
            ax.tick_params(axis='y', left=True, direction='inout', length=10)
        # ax.spines.bottom.set_position('zero')
        if ytick_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
        if xtick_step is not None:
            ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
        ax.tick_params(length=4.6)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        if hline_sig_cutoff is not None:
            plt.axhline(-np.log10(hline_sig_cutoff), color='black', linestyle="--", linewidth=.3)
        plt.xlim(right=xlim_right, left=xlim_left)
        plt.ylim(top=ylim_top, bottom=ylim_bottom)
        plt.xlabel(x_label, fontsize=xlabel_fontsize)
        plt.ylabel(y_label, fontsize=ylabel_fontsize)
        plt.title(title, fontsize=title_fontsize)  # , fontweight="bold"
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(False)
        if add_legend:
            ax = plt.gca()
            if isinstance(legend_num, list):
                legend_num = [(percentage * size_scale) / 100 for percentage in legend_num]
            handles, labels = sc.legend_elements("sizes", num=legend_num)

            for index, handle in enumerate(handles):
                handle.set_color('#d4d4d4')
                match = re.search(r'\d+', labels[index])
                extracted_number = int(match.group())
                labels[index] = f"{(round((extracted_number / size_scale) * 100))}"
                # handle.set_alpha(alpha_points)

            legend = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(legend_anchor_x, legend_anchor_y),
                               labelspacing=.4)
            legend.set_title(legend_title)
            legend.get_title().set_fontsize(legend_title_fontsize)
            legend.set_frame_on(False)

            for label in legend.get_texts():
                label.set_fontsize(legend_title_fontsize - 1)
        # ax = plt.gca()

        # mplcursors.cursor(hover=True).connect(
        #     "add", lambda sel: sel.annotation.set_text(data_df['pair'].tolist()[sel.index]))
        if save_figure:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            x_label = x_label.replace('\n', '_')
            plt.savefig(os.path.join(io_library.output_dir, f"volcano_{x_label}.pdf"),
                        dpi=60, bbox_inches="tight")
        plt.show()
        plt.clf()
        plt.close()

    @staticmethod
    def plot_feature_importance(data_df, title='', model_accuracy='', fontsize=12, wspace=0.3, file_name=None):

        fig = plt.figure(figsize=(7, 2.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
        gs.update(wspace=wspace)

        # Left subplot
        ax1 = plt.subplot(gs[0])
        ax1.set_facecolor('#f2f2f2')

        # Define the number of rows and the height of each row
        num_rows = data_df.shape[0]
        row_height = 0.2

        # Plot colored bars starting from 0% and ending at the specified percentage
        # for i in range(num_rows):
        for i, index in enumerate(data_df.index):
            y = i * row_height
            percentage = data_df.loc[index, 'importance']  # '#077f97', down_color='#a00000'
            ax1.fill_betweenx([y, y + row_height], 0, percentage, color='#f7552d')
            ax1.text(percentage + 1, y + row_height / 2, f"{percentage}%", va='center')  # Add text percentages

        # Plot white lines to split the figure into rows
        for i in range(1, num_rows):
            y = i * row_height
            ax1.axhline(y, color='white', linewidth=2)

        # Set the x-axis limits to range from 0 to 100%
        ax1.set_xlim(0, 100)

        # Set the y-axis limits to fit the rows
        ax1.set_ylim(0, num_rows * row_height)

        # Add ytick labels for each row
        ytick_positions = [i * row_height + row_height / 2 for i in range(num_rows)]
        ytick_labels = data_df.index.tolist()  # Replace with your labels
        ax1.set_yticks(ytick_positions)
        ax1.tick_params(axis='y', labelsize=fontsize)
        ax1.set_yticklabels(ytick_labels)

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)

        ax1.set_title(f'Feature importance\n(Model accuracy={model_accuracy})')

        # Right subplot (Second Figure)
        ax2 = plt.subplot(gs[1])
        ax2.axis('off')  # Turn off axes for the second figure

        col_name = 'corr'
        # a numeric placeholder for the y-axis
        my_range = np.arange(1, num_rows + 1)

        # Define a color function based on positive/negative values
        def color_function(val):
            return '#a00000' if val >= 0 else '#077f97'

        # Create for each expense type a horizontal line that starts at x = 0 with the length
        # represented by the specific expense percentage value.
        plt.hlines(y=my_range, xmin=0, xmax=data_df[col_name], color=data_df[col_name].apply(color_function),
                   linewidth=10)  # alpha=0.2,

        # Create for each expense type a dot at the level of the expense percentage value
        plt.plot(data_df[col_name], my_range, "o", markersize=10, color='black')  # , alpha=0.6

        # Add the value of each bar in front of it
        for i, (label, value) in enumerate(zip(data_df.index, data_df[col_name])):
            if value >= 0:
                plt.text(value + 0.07, my_range[i], f"{value}", va='center', fontsize=fontsize)
            else:
                plt.text(value - 0.45, my_range[i], f"{value}", va='center', fontsize=fontsize)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(True)

        ax2.set_xlim(-1, 1)
        plt.title('Correlation')
        # Adjust spacing between subplots
        fig = plt.gcf()  # Get the current figure
        fig.suptitle(title, fontsize=18, y=1.15)


        if file_name is not None:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"{file_name}.pdf"), dpi=60)
        plt.show()

    @staticmethod
    def percentage_stacked_barplot(data_df, x_col, y_col, color_dic, xlabel='', ylabel='', title='', crosstab=True,
                                   add_label_count=True, ytick_step=None,
                                   xtick_fontsize=7, ytick_fontsize=7, title_fontsize=10, xlabel_fontsize=8,
                                   ylabel_fontsize=9, percentage_sign='%',
                                   p_value=None, title_height=1.05, label_newline=True, tick_rotation=0,
                                   legend_title='', y_pval=.6,
                                   legend_fontsize=9, figure_width=2, figure_height=2, file_name=None):
        plt.style.use('default')
        if crosstab:
            crosstab_df = pd.crosstab(index=data_df[x_col],
                                      columns=data_df[y_col],
                                      normalize="index") * 100
            crosstab_count = pd.crosstab(index=data_df[x_col],
                                         columns=data_df[y_col])
        else:
            crosstab_df = data_df.div(data_df.sum(axis=1), axis=0) * 100
            crosstab_count = data_df.copy()

        crosstab_df = crosstab_df[color_dic.keys()]
        ax = crosstab_df.plot(kind='bar', stacked=True, color=[color_dic[col] for col in crosstab_df.columns],
                              figsize=(figure_width, figure_height))

        total_count = crosstab_count.sum(axis=1)
        # i=0
        for i, p in enumerate(ax.containers):
            # if i%2==0
            if i < len(ax.containers):
                labels = [f'{round(v.get_height())}{percentage_sign}' for v in p]
                ax.bar_label(p, labels=labels, fontsize=5, rotation=0, padding=-6, color='white', fontweight='bold')
            # else:
            #     ax.bar_label(p, labels=[f'(n={total_count[0]})', f'(n={total_count[1]})'], fontsize=ytick_fontsize, rotation=0,
            #                  padding=2, color='k')

            i += 1

        if p_value is not None:
            print('p-value=', p_value)
            if type(p_value) == list:
                p1 = MyVisualization.get_stat_text(p_value[0])
                p2 = MyVisualization.get_stat_text(p_value[1])
                p = str(p1) + '\n' + str(p2)
            else:
                p = MyVisualization.get_stat_text(p_value)
            plt.annotate(p,
                         xy=(1.62, y_pval),
                         xycoords='axes fraction',
                         ha='center',
                         fontsize=xtick_fontsize,
                         # bbox=dict(boxstyle='round', alpha=0.1, facecolor='white')
                         )

        plt.xticks(fontsize=xtick_fontsize, rotation=0)
        plt.yticks(fontsize=ytick_fontsize)
        plt.xlabel(xlabel, fontsize=xlabel_fontsize)
        plt.ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                   fontsize=legend_fontsize, frameon=False)
        if ytick_step is not None:
            ax.yaxis.set_major_locator(plt.MultipleLocator(ytick_step))

        def percent_formatter(x, pos):
            if not x:
                return int(x)
            return f'{x:.0f}%'

        ax.set_title(title, y=title_height, fontsize=title_fontsize)
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(.65)
        ax.spines['left'].set_linewidth(.65)
        if add_label_count:
            tick_labels = []
            for i, label in enumerate(ax.get_xticklabels()):
                category = label.get_text()
                size = len(data_df[data_df[x_col] == category])
                if label_newline:
                    tick_labels.append(f"{category}\n(n={size})")
                else:
                    tick_labels.append(f"{category} (n={size})")
            ax.set_xticklabels(tick_labels, rotation=tick_rotation)
        ax.tick_params(width=.65, length=2.5)
        ax.set_ylim(bottom=-1, top=100)
        plt.grid(False)
        if file_name is not None:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"barplot_{file_name}.pdf"),
                        dpi=60, bbox_inches="tight")
        plt.show()

    @staticmethod
    def multiple_ridge_plot(df, col_level, col_score, col_hue, color_dic=None, sharey=True, xlim_left=-1,
                            xlim_right=1, ylim=None, dist_alpha=1, aspect=5, p_value_l=None, p_value_dic=None,
                            adjp=False, xlabel='',
                            vline_offset=0.3, rug_plot_height=0.6, mean_line=False, median_line=False,
                            text_dic=None, text_fontsize=10, xlabel_fontsize=10, tick_fontsize=10, title='',
                            file_name=None):
        plt.figure(figsize=(9, 6))

        if color_dic is None:
            # c = ["#5c5b5a", "grey", '#e3e2e1', 'white', "#d94801", "#a63603", '#4d1a03']
            # v = [0, .15, .4, .5, .65, .9, 1.]
            c = ["silver", '#e3e2e1', 'white', "#d94801", "#a63603"]  # '#f58a56',
            # v = [0, .15, .2, .3, .4, .7, 1.]
            v = [0, .3, .5, 1.1, 2.1]
            normalized_v = [x / v[-1] for x in v]
            l = list(zip(normalized_v, c))
            cmap = LinearSegmentedColormap.from_list('gr', l, N=256)
            # cmap = sns.diverging_palette(220, 20, as_cmap=True)
            # cmap = sns.color_palette("light:b", as_cmap=True)
            # Create a LinearSegmentedColormap
            # colors = list(map(mcolors.to_rgba, c))
            # cmap = mcolors.LinearSegmentedColormap.from_list('br', list(zip(v, colors)), N=256)

            # Adjusting the range to start with x=0 and end with x=1
            # normalized_v = [x / v[-1] for x in v]

            # Constructing the color dictionary
            color_dic = {}
            # for level, value in zip(df[col_level].values, df[col_score].values):
            for level, value in zip(df[col_level].unique(), df[col_hue].unique()):
                # Normalize the value to fit in the range [0, 1] for colormap
                # normalized_value = value / v[-1]
                color = mcolors.rgb2hex(cmap(value)[:3])  # Convert RGB to hex
                color_dic[level] = color

        unique_levels = list(color_dic.keys())

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth': 0})
        g = sns.FacetGrid(df, row=col_level, hue=col_level,
                          aspect=aspect, height=1.25,
                          palette=color_dic,
                          sharey=sharey,
                          row_order=unique_levels,
                          # hue_order=list(color_dic.keys())
                          )
        g.map_dataframe(sns.kdeplot, x=col_score, color='white', linewidth=0.6)
        g.map_dataframe(sns.kdeplot, x=col_score, fill=True, alpha=dist_alpha)
        # if border:

        g.fig.subplots_adjust(hspace=-.2)  # Adjust the vertical space between plots

        for level, ax in zip(unique_levels, g.axes.flat):
            label_text = level
            cur_df = df[df[col_level] == level]
            if p_value_dic is not None:
                ax.text(-.1, .3, label_text, color='black',
                        fontsize=text_fontsize, fontweight="bold",
                        ha="left", va="center", transform=ax.transAxes)
                pval = MyVisualization.get_stat_text(p_value_dic[level], fdr=adjp)
                # if pval!='ns':
                ax.text(.01, .3, f'(n={cur_df.shape[0]}) ' + text_dic[level] + '  ' + pval, color='black',
                        fontsize=text_fontsize,
                        ha="left", va="center", transform=ax.transAxes)
                # else:
                #     ax.text(.01, .3, text_dic[level], color='black',
                #             fontsize=text_fontsize,
                #             ha="left", va="center", transform=ax.transAxes)
            else:
                ax.text(0, .4, label_text, color='black', fontsize=10,
                        ha="left", va="center", transform=ax.transAxes)
            ax.set(ylabel='')
        # Adjust the height and ylim of each subplot based on the number of samples in each distribution
        if ylim is not None:
            for level in ylim:  # zip(unique_levels, g.axes.flat)
                g.axes_dict[level].set(ylim=ylim[level])

        if p_value_l is not None:
            if len(p_value_l) == 1:
                height = len(unique_levels)
            else:
                height = 0.03
            print('p-value', p_value_l[0])
            plt.annotate(MyVisualization.get_stat_text(p_value_l[0]),
                         xy=(0.09, height),
                         xycoords='axes fraction',
                         ha='center',
                         fontsize=9,
                         bbox=dict(boxstyle='round', alpha=0.1, facecolor='white'))
            if len(p_value_l) > 1:
                print('p-value', p_value_l[1])
                plt.annotate(MyVisualization.get_stat_text(p_value_l[1]),
                             xy=(0.1, 1.15),
                             xycoords='axes fraction',
                             ha='center',
                             fontsize=9,
                             bbox=dict(boxstyle='round', alpha=0.1, facecolor='white'))

        g.set_titles("")
        g.set(yticks=[], xlabel=col_score)
        g.despine(left=True)

        # Add rug plot
        handles = g._legend_data.values()
        for i, handle in enumerate(handles):
            level = unique_levels[i]
            level_color = handle.get_facecolor()
            ax = g.axes_dict[level]  # Access the axis for the current level
            cur_df = df[df[col_level] == level]
            offset = 0
            sample = cur_df[col_score]
            ymax = -(rug_plot_height + offset)
            ax.vlines(sample, ymin=-(0 + offset), ymax=ymax, color=color_dic[level], alpha=dist_alpha)
            # ax.vlines(sample, ymin=-(0 + offset), ymax=ymax, color=level_color, alpha=dist_alpha)
            if col_hue != col_level:
                offset += rug_plot_height
            else:
                offset += 0
            if mean_line:
                median_val = sample.mean()
                ax.axvline(x=median_val, linestyle='-', color='black', alpha=0.3)
            elif median_line:
                median_val = sample.median()
                ax.axvline(x=median_val, linestyle='-', color='black', alpha=0.3)
            if col_hue != col_level:
                base_y = ymax + offset - 0.02
            else:
                base_y = ymax + rug_plot_height
            ax.hlines(y=base_y, xmin=xlim_left, xmax=xlim_right, color='black', linewidth=0.6)
            #
            for ax in g.axes.flat:
                ax.axvline(x=0, ymin=base_y + vline_offset, linestyle='--', color='k', linewidth=0.6)
        #     ax.axvline(x=-1, ymin=base_y + vline_offset, linestyle='--', color='k', linewidth=0.6)

        # legend_labels = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
        #                  for label, color in palette.items()]
        # legend = plt.legend(handles=handles, title=col_hue, bbox_to_anchor=(1.05, 1), loc='upper left',
        #                     frameon=False)
        # legend.get_frame().set_facecolor('white')
        # for handle, label in zip(legend.legend_handles, palette.keys()):
        #     handle.set_color(palette[label])
        sns.set(font_scale=10)
        plt.subplots_adjust(right=0.85)
        plt.xlim(xlim_left, xlim_right)
        plt.suptitle(title, y=0.98)
        plt.xlabel(xlabel, fontsize=xlabel_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        if file_name is not None:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f'{file_name}_ridge_plot.pdf'), dpi=50, bbox_inches="tight")
            print(os.path.join(io_library.output_dir, f'{file_name}_ridge_plot.pdf'))

    @staticmethod
    def reg_plot(data_df, x_col, y_col, xlabel='', ylabel='', hue=None, palette=None, points_color_code='#6b676e',
                 line_color_code='#C1914B', marker='o', top_hue=None, xlim_right=None, xlim_left=None,
                 title_height=None,
                 ylim_top=None, ylim_bottom=None, text_height=0.8, text_x_offset=0.05, font_scale=1, size=100,
                 pval_ns=True, stats_tp=None, sig_threshold=0.05, title='', title_fontsize=16, add_legend=False,
                 xtick_step=None, labels_fontsize=14, tick_fontsize=14, ytick_step=None, x_legend=1.4, figure_width=5,
                 figure_height=4, file_name=None, ax=None):

        plt.style.use('default')
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['pdf.fonttype'] = 42
        if ax is None:
            plt.figure(figsize=(figure_width, figure_height))
            ax = plt.gca()

        data_df = data_df.dropna(axis=1, how='all')

        sns.regplot(x=x_col, y=y_col, data=data_df, color=line_color_code, scatter=False,
                    line_kws={'linewidth': 2, 'linestyle': '--'}, ax=ax)

        sc = sns.scatterplot(x=x_col, y=y_col, data=data_df, hue=hue, palette=palette, marker=marker,
                             color=points_color_code, s=size, alpha=0.7, edgecolor='none', ax=ax)
        if top_hue is not None:
            sub_df = data_df[data_df[hue] == top_hue]
            sns.scatterplot(x=x_col, y=y_col, data=sub_df, hue=hue, palette=palette, marker=marker,
                            color=points_color_code, s=size, alpha=0.7, edgecolor='none', ax=ax, legend=False)

        if stats_tp is None:
            r_value, p_value = pearsonr(data_df[x_col], data_df[y_col])
            adjp = False
        else:
            r_value, p_value = stats_tp
            adjp = True

        stats_text = f'r ={r_value:.2f}, {MyVisualization.get_stat_text(p_value, ns_text=pval_ns, fdr=adjp, sig_threshold=sig_threshold)}'
        ax.text(text_x_offset, text_height, stats_text, transform=ax.transAxes, fontsize=labels_fontsize, color='black')

        ax.set_ylabel(ylabel, fontsize=labels_fontsize)
        ax.set_xlabel(xlabel, fontsize=labels_fontsize)
        ax.set_title(title, y=title_height, fontdict={'fontsize': title_fontsize})

        if add_legend:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 1:
                sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[1])
                handles, labels = zip(*sorted_handles_labels)
            ax.legend(handles, labels, title=hue, bbox_to_anchor=(1.05, 0.25), frameon=False)

        ax.set_xlim(right=xlim_right, left=xlim_left)
        ax.set_ylim(top=ylim_top, bottom=ylim_bottom)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        line_thickness = 1.1
        ax.tick_params(axis='both', which='both', length=4.1, width=line_thickness)
        ax.spines['bottom'].set_linewidth(line_thickness)
        ax.spines['left'].set_linewidth(line_thickness)

        if xtick_step is not None:
            ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
        if ytick_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step))

        sns.despine()
        sns.set(font_scale=font_scale)

        if file_name is not None:
            plt.savefig(os.path.join(io_library.output_dir, f"LR_{file_name}.pdf"), dpi=60, bbox_inches="tight")
        if ax is None:
            plt.show()
            plt.close()

    @staticmethod
    def swarm_plot(data_df, y_col, x_col, palette_dic=None, hue_col=None, title='', color_col='', x_label='',
                   y_label='', point_size=4, hline_type='mean', legend_fontsize=5, labels_fontsize=9,
                   label_newline=False,
                   ylim_top=None, ylim_bottom=None, ytick_step=None, p_value=None, adj_pval=False, star_pval=False,
                   pvalue_pairs=None, sig_threshold=0.05, xtick_rot=0, add_legend=False, tick_rotation=0,
                   tick_fontsize=6,
                   title_height=None, figure_width=1.3, figure_height=1, title_fontsize=10, save_figure=False,
                   file_suffix=''):
        """

        """
        # sns.set(style='ticks', context='talk')
        plt.style.use('default')
        f, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor="w")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            sns.swarmplot(x=x_col, y=y_col, data=data_df, color="black", s=point_size, palette=palette_dic, hue=hue_col,
                          order=list(palette_dic.keys()), ax=ax, zorder=1)
        if hline_type is not None:
            for i, category in enumerate(palette_dic.keys()):
                if hline_type == 'median':
                    line_pos = data_df[data_df[x_col] == category][y_col].median()
                else:
                    line_pos = data_df[data_df[x_col] == category][y_col].mean()
                ax.hlines(line_pos, xmin=i - 0.2, xmax=i + 0.2, color='black',
                          linestyle='-', linewidth=1., zorder=2)
        if add_legend:
            legend = plt.legend(title=color_col, bbox_to_anchor=(1.1, 0.4), loc='upper left', ncol=1, fontsize=4,
                                frameon=False)
            legend.get_title().set_fontsize(6)
            for handle in legend.legendHandles:
                handle.set_sizes([point_size])
            for label in legend.get_texts():
                label.set_fontsize(legend_fontsize)
        else:
            plt.legend([], frameon=False)

        if p_value is not None:
            if pvalue_pairs is None:
                pvalue_pairs = [tuple(palette_dic.keys())]
                pvals = [
                    MyVisualization.get_stat_text(p_value, sig_threshold=sig_threshold, star=star_pval, fdr=adj_pval)]
            else:
                pvals = [MyVisualization.get_stat_text(p, sig_threshold=sig_threshold, star=star_pval, fdr=adj_pval)
                         for p in p_value]
            annotator = Annotator(ax, pvalue_pairs,
                                  data=data_df,
                                  x=x_col,
                                  y=y_col,
                                  verbose=False
                                  )
            annotator._pvalue_format.fontsize = 6
            annotator.line_width = 0.5
            annotator.set_custom_annotations(pvals)
            annotator.annotate()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        plt.xlabel(x_label, fontweight='normal', fontsize=labels_fontsize)
        plt.ylabel(y_label, fontweight='normal', fontsize=labels_fontsize)

        plt.grid(False)
        ax.set_ylim(top=ylim_top, bottom=ylim_bottom)
        if ytick_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
        line_thickness = .7
        ax.tick_params(axis='both', which='both', length=4.1, width=line_thickness)
        plt.xticks(fontsize=tick_fontsize, rotation=xtick_rot)
        plt.yticks(fontsize=tick_fontsize)
        ax.spines['bottom'].set_linewidth(line_thickness)
        ax.spines['left'].set_linewidth(line_thickness)
        plt.suptitle(title, y=title_height, fontsize=title_fontsize)
        # Add size of the group to each tick label
        tick_labels = []
        for i, label in enumerate(ax.get_xticklabels()):
            category = label.get_text()
            size = len(data_df[data_df[x_col] == category])
            if label_newline:
                tick_labels.append(f"{category}\n(n={size})")
            else:
                tick_labels.append(f"{category} (n={size})")
        ax.set_xticklabels(tick_labels, rotation=tick_rotation)
        if save_figure:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"sawrmplot_{title}_{x_label}_{y_label}{file_suffix}.pdf"),
                        dpi=60, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def violin_plot(data_df, y_col, x_col, color_dic, palette=None, title='', file_name='', color_col='', x_label='',
                    y_label='',
                    ylim_top=None, ylim_bottom=None, ytick_step=None, p_value=None, adj_pval=False, star_pval=False,
                    box_height=2.5, box_aspect=.8, pvalue_pairs=None, sig_threshold=0.05, tick_rotation=0,
                    add_legend=False,
                    labels_fontsize=8, text_fontsize=7, annotator_height=None, label_newline=True, alpha=0.5,
                    title_height=None, figure_width=1.3, figure_height=1, title_fontsize=10, save_figure=False):

        plt.style.use('default')
        f, ax = plt.subplots(figsize=(figure_width, figure_height))

        my_colors = list(color_dic.values())
        sns.set_palette(my_colors)
        v = sns.violinplot(x=x_col, y=y_col, data=data_df, inner='quartile',
                           scale='count',  # dodge=False,
                           bw=.2, cut=2, linewidth=0.15,  # order=palette,
                           ax=ax)

        ax.collections[0].set_edgecolor(my_colors[0])
        ax.collections[1].set_edgecolor(my_colors[1])

        for index, p in enumerate(v.lines):
            if index == 1 or index == 4:
                p.set_linestyle('-')
            else:
                p.set_linestyle('--')
            p.set_linewidth(0.35)
            p.set_color('white')
            # p.set_alpha(0.8)

        if p_value is not None:
            if pvalue_pairs is None:
                pvalue_pairs = [tuple(color_dic.keys())]
                pvals = [
                    MyVisualization.get_stat_text(p_value, sig_threshold=sig_threshold, star=star_pval, fdr=adj_pval)]
            else:
                pvals = [MyVisualization.get_stat_text(p, sig_threshold=sig_threshold, star=star_pval, fdr=adj_pval)
                         for p in p_value]
            annotator = Annotator(ax, pvalue_pairs,
                                  data=data_df,
                                  x=x_col,
                                  y=y_col,
                                  verbose=False)
            annotator.configure(loc='outside')
            annotator._pvalue_format.fontsize = text_fontsize
            annotator.line_width = 0.5
            annotator.set_custom_annotations(pvals)
            annotator.annotate()

            # ax = plt.gca()
        ax.set_ylim(top=ylim_top, bottom=ylim_bottom)
        if ytick_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
        tick_labels = []
        for i, label in enumerate(ax.get_xticklabels()):
            category = label.get_text()
            size = len(data_df[data_df[x_col] == category])
            if label_newline:
                tick_labels.append(f"{category}\n(n={size})")
            else:
                tick_labels.append(f"{category} (n={size})")
        ax.set_xticklabels(tick_labels, rotation=tick_rotation)
        ax.tick_params(width=.6, length=2.5)
        # ax.spines['bottom'].set_linewidth(.6)
        # ax.spines['left'].set_linewidth(.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        plt.xlabel(x_label, fontweight='normal', fontsize=labels_fontsize)
        # plt.ylabel(y_label+ ' Gene Effect', fontweight='normal', fontsize=7)
        plt.ylabel(y_label, fontweight='normal', fontsize=labels_fontsize)
        plt.xticks(fontsize=6, rotation=tick_rotation)
        plt.yticks(fontsize=6)
        plt.grid(False)
        plt.suptitle(title, y=title_height, x=.6, fontsize=title_fontsize)
        # sns.set(font_scale=.4)
        if save_figure:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"violinplot_{x_col}_{y_label}.pdf"),
                        dpi=60, bbox_inches="tight")
        plt.show()

    @staticmethod
    def reg_plot_double(data_df, x_col, y_col, size_dic, markers_dic, xlabel='', ylabel='', hue=None, palette=None,
                        xlim_right=None, xlim_left=None, title_height=None, title_fontsize=None, xtick_step=None,
                        ytick_step=None, ylim_bottom=None,
                        ylim_top=None, text_height=0.8, text_x_offset=0.05, font_scale=1, labels_fontsize=14, title='',
                        legend=True,
                        figure_width=6, figure_height=4, file_name=None):
        plt.style.use('default')
        plt.figure(figsize=(figure_width, figure_height))
        data_df = data_df.dropna(axis=1, how='all')
        # sns.set(style='ticks', context='talk')
        text_offset = 0
        for g in data_df[hue].unique():
            subset_df = data_df[data_df[hue] == g]
            ax = sns.regplot(x=x_col, y=y_col, data=subset_df, color=palette[g], scatter=False,
                             line_kws={'linewidth': 2, 'linestyle': '-'})

            sns.scatterplot(x=x_col, y=y_col, data=subset_df, marker=markers_dic[g], alpha=0.7,
                             edgecolor='none',
                            color=palette[g], s=size_dic[g])

            # slope, intercept, r_value, p_value, std_err = linregress(x=subset_df[x_col], y=subset_df[y_col])
            r_value, p_value = pearsonr(subset_df[x_col], subset_df[y_col])
            print('p-value: ', p_value)
            stats_text = f'r ={r_value:.2f}, {MyVisualization.get_stat_text(p_value)}, n={subset_df.shape[0]}'

            # ax.text(0.05, 0.9, equation, transform=ax.transAxes, fontsize=10, color='black')
            if palette[g] =='#c2c0c0':
                cur_color = '#9a9c9b'
            else:
                cur_color = palette[g]
            ax.text(text_x_offset, text_height + text_offset, stats_text, transform=ax.transAxes,
                    fontsize=labels_fontsize, color=cur_color)
            text_offset -= 0.1
        # Set labels and title
        ax.set_ylabel(ylabel, fontsize=labels_fontsize)
        ax.set_xlabel(xlabel, fontsize=labels_fontsize)
        ax.set_title(title, y=title_height, fontsize=title_fontsize)
        if legend:
            # handles, labels = ax.get_legend_handles_labels()
            handles = [plt.Line2D([], [], color=palette[l], linestyle='',
                                  marker=markers_dic[l])
                       for l in markers_dic]
            # labels, handles = zip(*sorted(zip(labels, handles)))

            ax.legend(handles, list(markers_dic.keys()), title=hue, bbox_to_anchor=(1.2, .3), frameon=False)
        else:
            ax.get_legend().remove()
        # scatter_legend.legendHandles = custom_legend
        ax.set_xlim(right=xlim_right, left=xlim_left)
        ax.set_ylim(top=ylim_top, bottom=ylim_bottom)
        line_thickness = 1.1
        ax.tick_params(axis='both', which='both', length=4.3, width=line_thickness)
        ax.spines['bottom'].set_linewidth(line_thickness)
        ax.spines['left'].set_linewidth(line_thickness)
        if xtick_step is not None:
            ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
        if ytick_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
        sns.despine()
        sns.set(font_scale=font_scale)
        if file_name is not None:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"LR_{file_name}.pdf"), dpi=60, bbox_inches="tight")
        plt.show()

        plt.close()

    @staticmethod
    def plot_heatmap(mat_df, upset_df=None, title='', row_cluster=False, col_cluster=False, values_label='',
                     col_label='',
                     row_label='', xticklabels=True, yticklabels=True, ytick_fontsize=9, xtick_fontsize=9, xdend=False,
                     ydend=False,
                     cbar_discrete=False, cbar_left_adjust=.065, cbar_CRISPR=False, bar_colors_ratio=0.02,
                     row_upset_df=None,
                     legend_fontsize=10, cbar_title='', borders_linewidths=0.01, legend_top_offset=0.85,
                     event_color=None,
                     legend_h_offset=0.06, cbar_legened_offset=0.19, legend_diff_offset=0.11, legend_bar=True,
                     event_legend=True, cbar_pval=False,
                     left_adjust=0.1, title_height=1, dendrogram_ratio=(.1, .1),
                     row_label_fontsize=15, col_label_fontsize=15, split_row_l=None, title_fontsize=12,
                     row_legend_bar=True, filename_suffix='',
                     figsize_w=13, figsize_h=12, save_figure=False):
        """

        """
        if cbar_discrete:
            intervals = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
            # colors =["royalblue", "lightblue",'#dff2f7', '#e3e2e1', "silver", "grey"]
            # colors = ["#a63603", "#d94801", '#f5dcc4', '#e3e2e1', "silver", "grey"]
            colors = ["#a63603", "#d94801", 'white', '#e3e2e1', "silver", "grey"]
            cmap = ListedColormap(colors)
            norm = mcolors.BoundaryNorm(intervals, cmap.N, clip=True)
            center = None
        else:
            if cbar_CRISPR:
                c = ["#a63603", "#d94801", 'white', '#e3e2e1', "silver", "grey"]  #
                v = [0, .15, .5, .6, .9, 1.]
                l = list(zip(v, c))
                cmap = LinearSegmentedColormap.from_list('og', l, N=256)
            elif cbar_pval:
                c = ["white", "white", '#ed6421', '#d14a08', "#a63603"]
                # v = [0, 1, 1.2, 1.4, 1.6]  # Values adjusted to fit within the [0, 1] range
                v = [0, 0.52, .7, .8, 1]  # Values adjusted to fit within the [0, 1] range

                # Normalize the values to the [0, 1] range
                # v_normalized = [(x - 0) / (2 - 0) for x in v]

                # Combine the normalized values and colors into a list of tuples
                # l = list(zip(v_normalized, c))
                l = list(zip(v, c))

                # Create the colormap
                cmap = LinearSegmentedColormap.from_list('custom_cmap', l, N=256)
            else:
                # c = ["darkblue", "royalblue", "lightblue", "white", "lightcoral", "red", "darkred"]
                # c = ["#a63603", "#d94801", '#f58a56', 'white', '#e3e2e1', "silver", "grey"]
                c = ["grey", "silver", '#e3e2e1', 'white', "#d94801", "#a63603", '#732603']  # '#f58a56',
                # '#f58a56',
                if row_label == 'Protein':
                    c = ["#5c5b5a", "grey", '#e3e2e1', 'white', "#d94801", "#a63603", '#4d1a03']
                    v = [0, .15, .4, .5, .65, .9, 1.]
                else:
                    v = [0, .15, .4, .5, .7, .9, 1.]
                l = list(zip(v, c))
                cmap = LinearSegmentedColormap.from_list('br', l, N=256)
            norm = None
            center = 0

        event = 'Event'
        feat3 = 'Cluster'
        mutations_s = set()
        RGB_df = None
        if upset_df is not None:
            RGB_df = pd.DataFrame()
            for feat in upset_df.columns:
                if feat in MyVisualization.feat_colors_dic.keys():
                    RGB_df = pd.concat([RGB_df, pd.DataFrame([MyVisualization.feat_colors_dic[feat][ele]
                                                              if not pd.isna(ele) else np.nan for ele in
                                                              upset_df[feat]],
                                                             index=upset_df.index, columns=[feat])], axis=1)
                elif '$' in feat:
                    if event_color is not None:
                        RGB_df = pd.concat([RGB_df, pd.DataFrame([event_color
                                                                  if not pd.isna(ele) else np.nan for ele in
                                                                  upset_df[feat]],
                                                                 index=upset_df.index,
                                                                 columns=[feat.replace('$', '')])], axis=1)
                    else:
                        RGB_df = pd.concat([RGB_df, pd.DataFrame([MyVisualization.feat_colors_dic['Event'][ele]
                                                                  if not pd.isna(ele) else np.nan for ele in
                                                                  upset_df[feat]],
                                                                 index=upset_df.index,
                                                                 columns=[feat.replace('$', '')])], axis=1)
                    mutations_s = mutations_s.union(
                        set([value for value in upset_df[feat].unique().tolist() if not pd.isna(value)]))
                else:
                    RGB_df = pd.concat([RGB_df, pd.DataFrame([MyVisualization.feat_colors_dic[feat3][ele]
                                                              if not pd.isna(ele) else np.nan for ele in
                                                              upset_df[feat]],
                                                             index=upset_df.index, columns=[feat])], axis=1)
        mutations = sorted(mutations_s)

        row_RGB_df = None
        if row_upset_df is not None:
            row_RGB_df = pd.DataFrame()
            for feat in row_upset_df.columns:
                if feat in MyVisualization.feat_colors_dic.keys():
                    row_RGB_df = pd.concat([row_RGB_df, pd.DataFrame([MyVisualization.feat_colors_dic[feat][ele]
                                                                      if not pd.isna(ele) else np.nan for ele in
                                                                      row_upset_df[feat]],
                                                                     index=row_upset_df.index, columns=[feat])], axis=1)
                elif '$' in feat:
                    if event_color is not None:
                        row_RGB_df = pd.concat([row_RGB_df, pd.DataFrame([event_color
                                                                          if not pd.isna(ele) else np.nan for ele in
                                                                          row_upset_df[feat]],
                                                                         index=row_upset_df.index,
                                                                         columns=[feat.replace('$', '')])], axis=1)
                    else:
                        row_RGB_df = pd.concat([row_RGB_df, pd.DataFrame([MyVisualization.feat_colors_dic['Event'][ele]
                                                                          if not pd.isna(ele) else np.nan for ele in
                                                                          row_upset_df[feat]],
                                                                         index=row_upset_df.index,
                                                                         columns=[feat.replace('$', '')])], axis=1)
                    mutations_s = mutations_s.union(
                        set([value for value in row_upset_df[feat].unique().tolist() if not pd.isna(value)]))
                else:
                    row_RGB_df = pd.concat([row_RGB_df, pd.DataFrame([MyVisualization.feat_colors_dic[feat3][ele]
                                                                      if not pd.isna(ele) else np.nan for ele in
                                                                      row_upset_df[feat]],
                                                                     index=row_upset_df.index, columns=[feat])], axis=1)
        row_linkage_matrix = None
        col_linkage_matrix = None
        if row_cluster or col_cluster:
            linkage_matrix = linkage(mat_df, 'ward', optimal_ordering=True)
            if row_cluster:
                row_linkage_matrix = linkage(mat_df, method='ward', optimal_ordering=True)
            if col_cluster:
                col_linkage_matrix = linkage(mat_df.T, method='ward', optimal_ordering=True)
            # row_cluster = None
            # col_cluster = None
        hmap = sns.clustermap(mat_df,
                              center=center,
                              cmap=cmap,
                              norm=norm,
                              # cmap="vlag",
                              row_linkage=row_linkage_matrix,
                              col_linkage=col_linkage_matrix,
                              row_cluster=row_cluster,
                              col_cluster=col_cluster,
                              col_colors=RGB_df,
                              row_colors=row_RGB_df,
                              dendrogram_ratio=dendrogram_ratio,
                              cbar_pos=(.02, .15, .03, .2),
                              # cbar_pos=(1, .15, .03, .2),
                              linewidths=borders_linewidths,  # .01,
                              linecolor="lightgrey",
                              # method='ward',
                              # method='complete',
                              # annot_kws={'size': 25},
                              # cbar_kws={'aspect': 3},
                              xticklabels=xticklabels,
                              yticklabels=yticklabels,
                              # standard_scale=0,
                              colors_ratio=bar_colors_ratio,
                              # yticklabels_fontsize=ytick_fontsize,
                              figsize=(figsize_w, figsize_h)

                              )
        if split_row_l is not None:
            ax_heatmap = hmap.ax_heatmap
            for r in split_row_l:
                ax_heatmap.axhline(r, color='black', linewidth=2)

        if yticklabels:
            hmap.ax_heatmap.yaxis.set_tick_params(labelsize=ytick_fontsize)
        hmap.ax_heatmap.xaxis.set_tick_params(labelsize=xtick_fontsize, rotation=90)
        hmap.ax_row_dendrogram.set_visible(xdend)
        hmap.ax_col_dendrogram.set_visible(ydend)
        # hmap.ax_heatmap.set_yticklabels(hmap.ax_heatmap.get_ymajorticklabels(), fontsize=5)
        # hmap.ax_heatmap.xaxis.tick_top()  # Place x-axis ticks on the top side
        # hmap.ax_heatmap.xaxis.set_label_position("top")

        # hmap.ax_row_dendrogram.set_title('KRAS', fontsize=6)

        hmap.ax_heatmap.axhline(y=0, color='black', linewidth=1)
        # hmap.ax_heatmap.axhline(y=mat_df.shape[1], color='k', linewidth=1)
        # hmap.ax_heatmap.axvline(x=0, color='k', linewidth=1)
        # hmap.ax_heatmap.axvline(x=mat_df.shape[0], color='k', linewidth=1)

        hmap.ax_heatmap.set_xlabel(col_label, size=row_label_fontsize)
        hmap.ax_heatmap.set_ylabel(row_label, size=col_label_fontsize)
        hmap.ax_cbar.set_ylabel(values_label, size=legend_fontsize)

        if cbar_discrete:
            colorbar = hmap.ax_heatmap.collections[0].colorbar
            colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
            colorbar.set_ticklabels(['-1', '-0.5', '0', '0.5', '1'])

        offset = legend_top_offset
        if (upset_df is not None) and legend_bar:
            for feat in upset_df.columns:
                l = []
                uniques = [value for value in upset_df[feat].unique().tolist() if not pd.isna(value)]
                if np.nan in uniques:
                    uniques.remove(np.nan)
                uniques = sorted(uniques)
                if feat in MyVisualization.feat_colors_dic.keys():
                    for label in uniques:
                        bar = hmap.ax_row_dendrogram.bar(0, 0, color=MyVisualization.feat_colors_dic[feat][label],
                                                         label=label,
                                                         linewidth=0)
                        l.append(bar)
                    legend = plt.legend(l, uniques,
                                        loc="center",
                                        title=feat,
                                        bbox_to_anchor=(legend_h_offset, offset),
                                        fontsize=legend_fontsize,
                                        facecolor="white",
                                        frameon=False,
                                        bbox_transform=gcf().transFigure)
                    plt.gca().add_artist(legend)
                    offset -= legend_diff_offset
                elif '$' in feat:
                    if event_legend:
                        for label in mutations:
                            bar = hmap.ax_row_dendrogram.bar(0, 0, color=MyVisualization.feat_colors_dic[event][label],
                                                             label=label,
                                                             linewidth=0)
                            l.append(bar)
                        legend = plt.legend(l, mutations,
                                            loc='center',
                                            title=event,
                                            bbox_to_anchor=(legend_h_offset, offset),
                                            fontsize=legend_fontsize,
                                            facecolor="white",
                                            frameon=False,
                                            bbox_transform=gcf().transFigure)
                        plt.gca().add_artist(legend)
                        offset -= legend_diff_offset
                        event_legend = False
                else:
                    for label in uniques:
                        bar = hmap.ax_row_dendrogram.bar(0, 0, color=MyVisualization.feat_colors_dic[feat3][label],
                                                         label=label, linewidth=0)
                        l.append(bar)

                    legend = plt.legend(l, uniques,
                                        loc="center",
                                        title=feat,
                                        bbox_to_anchor=(legend_h_offset, offset),
                                        fontsize=legend_fontsize,
                                        facecolor="white",
                                        frameon=False,
                                        bbox_transform=gcf().transFigure)
                    plt.gca().add_artist(legend)
                    offset -= legend_diff_offset

        if (row_upset_df is not None) and row_legend_bar:
            for feat in row_upset_df.columns:
                l = []
                uniques = [value for value in row_upset_df[feat].unique().tolist() if not pd.isna(value)]
                if np.nan in uniques:
                    uniques.remove(np.nan)
                uniques = sorted(uniques)
                if feat in MyVisualization.feat_colors_dic.keys():
                    for label in uniques:
                        bar = hmap.ax_row_dendrogram.bar(0, 0, color=MyVisualization.feat_colors_dic[feat][label],
                                                         label=label,
                                                         linewidth=0)
                        l.append(bar)
                    legend = plt.legend(l, uniques, loc="center", title=feat, bbox_to_anchor=(legend_h_offset, offset),
                                        fontsize=legend_fontsize,
                                        facecolor="white",
                                        frameon=False,
                                        bbox_transform=gcf().transFigure)
                    plt.gca().add_artist(legend)
                    offset -= legend_diff_offset
                else:
                    for label in uniques:
                        bar = hmap.ax_row_dendrogram.bar(0, 0, color=MyVisualization.feat_colors_dic[feat3][label],
                                                         label=label, linewidth=0)
                        l.append(bar)

                    legend = plt.legend(l, uniques, loc="center", title=feat, bbox_to_anchor=(legend_h_offset, offset),
                                        fontsize=legend_fontsize,
                                        facecolor="white",
                                        frameon=False,
                                        bbox_transform=gcf().transFigure)
                    plt.gca().add_artist(legend)
                    offset -= legend_diff_offset

        plt.suptitle(title, fontsize=title_fontsize, y=title_height)
        plt.subplots_adjust(left=left_adjust)  # creates space for the legends
        cax = hmap.ax_cbar  # Get the colorbar axis
        plt.title(cbar_title, fontsize=legend_fontsize)
        sns.set(font_scale=.8)

        cax.set_position([cbar_left_adjust, offset - cbar_legened_offset, .03, .2])  # [left, bottom, width, height] as needed
        if cbar_pval:
            tick_positions = [0, 0.6, 1, 2, 6]
            tick_labels = ['0', '0.6', '1', '2', '6']
            cbar = hmap.ax_heatmap.collections[0].colorbar
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)

        # plt.tight_layout()
        if save_figure:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir,
                        f"heatmap_{cbar_title}_{row_label}_{col_label}{filename_suffix}.pdf"),
                        dpi=50, bbox_inches="tight")
        # else:
        plt.show()
        if row_cluster:
            return mat_df.iloc[hmap.dendrogram_row.reordered_ind, :].index.tolist()

    @staticmethod
    def swarm_plot_paired(df, x, y, hue, title, xlabel='', ylabel='', p_values=None, palette=None,
                          order=None, hue_order=None, pairs=None, hline_offset=0.1,
                          hline_in_offset=0.135, hline_out_offset=0.33,
                          ylim_top=None, ylim_bottom=None, figure_width=5, figure_height=4,
                          axis_label_fontsize=10, tick_fontsize=9, title_fontsize=12,
                          save_figure=False):
        plt.style.use('default')
        f, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor="w")

        sns.swarmplot(
            data=df, x=x, y=y, hue=hue,
            dodge=True, hue_order=hue_order, palette=palette,
            size=4, order=order, ax=ax
        )

        mean_values_gp = df.groupby([hue, x])[y].mean().reset_index().sort_values(by='Group').sort_values(by='Gene')

        w = 0.02
        for _, gp in enumerate(order):
            cur_df = mean_values_gp[mean_values_gp[x] == gp]
            ax.axhline(y=cur_df.iloc[0][y], xmin=hline_offset - w, xmax=hline_offset + w, color='k', linestyle='-',
                       linewidth=.7, zorder=3)
            ax.axhline(y=cur_df.iloc[1][y], xmin=hline_offset + hline_in_offset - 2 * w,
                       xmax=hline_offset + hline_in_offset + 2 * w,
                       color='k', linestyle='-', linewidth=.7, zorder=3)
            hline_offset += hline_out_offset

        swarm_legend_labels = [
            plt.Line2D([0], [0], marker='o', color='w', label=hue_val, markerfacecolor=palette[hue_val],
                       markersize=8)
            for hue_val in hue_order
        ]

        ax.legend(handles=swarm_legend_labels, title=hue, frameon=False, bbox_to_anchor=(1.05, 1),
                  loc='upper left')

        if pairs is not None:
            annotator = Annotator(ax, pairs,
                                  data=df,
                                  x=x,
                                  y=y,
                                  hue=hue,
                                  order=order,
                                  verbose=False
                                  )

            annotator._pvalue_format.fontsize = 9
            annotator.line_width = 0.5
            annotator.set_pvalues(p_values)
            annotator.annotate()
        ax.set_ylim(top=ylim_top, bottom=ylim_bottom)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        line_thickness = .7
        ax.tick_params(axis='both', which='both', length=4.1, width=line_thickness)
        ax.spines['bottom'].set_linewidth(line_thickness)
        ax.spines['left'].set_linewidth(line_thickness)
        plt.xlabel(xlabel, fontsize=axis_label_fontsize)
        plt.ylabel(ylabel, fontsize=axis_label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(False)

        plt.title(title, fontsize=title_fontsize)

        if save_figure:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"swarmplot_paired_{title}.pdf"), dpi=60,
                        bbox_inches="tight")

        plt.show()


    # @staticmethod
    # def box_plot(data_df, y_col, x_col, color_dic, palette=None, title='', add_counts=True, color_col='', x_label='',
    #              y_label='',
    #              ylim_top=None, ylim_bottom=None, ytick_step=None, p_value=None, adj_pval=False, star_pval=False,
    #              alpha_points=1,
    #              points_size=3,
    #              box_height=2.5, box_aspect=.8, pvalue_pairs=None, sig_threshold=0.05, tick_rotation=0,
    #              add_legend=False,
    #              axis_fontsize=8, text_fontsize=7, annotator_height=None, label_newline=True, alpha=0.5,
    #              showfliers=False,
    #              title_height=None, figure_width=1.3, figure_height=1, title_fontsize=10, filename=None):
    #     """
    #     The box extends from the first quartile (Q1) to the third quartile (Q3) of the data, with a line at the median
    #     The whiskers are generally extended into 1.5*IQR distance on either side of the box.
    #
    #     """
    #     # sns.set(style='ticks', context='talk')
    #     plt.style.use('default')
    #     plt.figure(figsize=(figure_width, figure_height))
    #
    #     # sns.set_style("white")  # "whitegrid","dark","darkgrid","ticks"
    #     my_colors = color_dic.values()
    #
    #     custom_marker_props = {'markersize': .15, 'marker': 'o',
    #                            'markerfacecolor': '#d4d4d4'}  # 'markerfacecolor': 'gray', 'marker': 'o',
    #     sns.set_palette(my_colors)
    #     lm = sns.catplot(data=data_df,
    #                      x=x_col,
    #                      y=y_col,
    #                      kind="box",
    #                      height=box_height,
    #                      aspect=box_aspect,
    #                      linewidth=1.1,
    #                      order=list(color_dic.keys()),
    #                      flierprops=custom_marker_props,
    #                      showfliers=showfliers,
    #                      # showbox=False,
    #                      # whis = 100
    #                      boxprops={'edgecolor': 'none', 'alpha': 1},  #
    #                      whiskerprops={'color': "black", 'linestyle': "--", 'linewidth': .5},
    #                      capprops={'color': "black", 'linewidth': .5},
    #                      medianprops={'color': "white", 'linewidth': .5}
    #                      )  # ,
    #     for ax in lm.axes.flat:
    #         for box in ax.artists:
    #             box.set_alpha(alpha)
    #             box.set_edgecolor(None)
    #             box.set_linewidth(0)
    #
    #     ax = lm.axes[0, 0]
    #     # sns.color_palette(" hls", 8)
    #     if palette is not None:
    #         np.random.seed(seed)
    #         sns.stripplot(x=x_col,
    #                       y=y_col,
    #                       data=data_df,
    #                       hue=color_col,
    #                       palette=palette,
    #                       alpha=alpha_points,
    #                       jitter=0.3,
    #                       size=points_size,
    #                       edgecolor='none',
    #                       linewidth=.1
    #                       )
    #
    #         if add_legend:
    #             legend = plt.legend(title=color_col, bbox_to_anchor=(1.1, 0.4), loc='upper left', ncol=1, fontsize=4,
    #                                 frameon=False)
    #             legend.get_title().set_fontsize(6)
    #             for handle in legend.legendHandles:
    #                 handle.set_sizes([8])
    #
    #             for label in legend.get_texts():
    #                 label.set_fontsize(6)
    #         else:
    #             plt.legend([], frameon=False)
    #
    #     if p_value is not None:
    #         if pvalue_pairs is None:
    #             pvalue_pairs = [tuple(color_dic.keys())]
    #             pvals = [
    #                 MyVisualization.get_stat_text(p_value, sig_threshold=sig_threshold, star=star_pval, fdr=adj_pval)]
    #         else:
    #             pvals = [MyVisualization.get_stat_text(p, sig_threshold=sig_threshold, star=star_pval, fdr=adj_pval)
    #                      for p in p_value]
    #         annotator = Annotator(ax, pvalue_pairs,
    #                               data=data_df,
    #                               x=x_col,
    #                               y=y_col,
    #                               verbose=False
    #                               )
    #         annotator._pvalue_format.fontsize = text_fontsize
    #         annotator.line_width = 0.5
    #         annotator.set_custom_annotations(pvals)
    #         annotator.annotate()
    #
    #     # ax = plt.gca()
    #     ax.set_ylim(top=ylim_top, bottom=ylim_bottom)
    #     if ytick_step is not None:
    #         ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
    #     if add_counts:
    #         tick_labels = []
    #         for i, label in enumerate(ax.get_xticklabels()):
    #             category = label.get_text()
    #             size = len(data_df[data_df[x_col] == category])
    #             if label_newline:
    #                 tick_labels.append(f"{category}\n(n={size})")
    #             else:
    #                 tick_labels.append(f"{category} (n={size})")
    #         ax.set_xticklabels(tick_labels, rotation=tick_rotation)
    #     ax.tick_params(width=.6, length=2.5)
    #     ax.spines['bottom'].set_linewidth(.6)
    #     ax.spines['left'].set_linewidth(.6)
    #     plt.xlabel(x_label, fontweight='normal', fontsize=axis_fontsize)
    #     # plt.ylabel(y_label+ ' Gene Effect', fontweight='normal', fontsize=7)
    #     plt.ylabel(y_label, fontweight='normal', fontsize=axis_fontsize)
    #     plt.xticks(fontsize=6, rotation=tick_rotation)
    #     plt.yticks(fontsize=6)
    #     plt.grid(False)
    #     plt.suptitle(title, y=title_height, x=.6, fontsize=title_fontsize)
    #     # sns.set(font_scale=.4)
    #     if filename is not None:
    #         plt.rcParams['font.family'] = 'Arial'
    #         plt.rcParams['pdf.fonttype'] = 42
    #         plt.savefig(os.path.join(io_library.output_dir, f"boxplot_{filename}_{x_col}_{y_label}.pdf"),
    #                     dpi=60, bbox_inches="tight")
    #     plt.show()

    @staticmethod
    def box_plot(data_df, y_col, x_col, color_dic, gp_col=None, palette=None, title='', add_counts=True, color_col='',
                 x_label='', y_label='',
                 ylim_top=None, ylim_bottom=None, ytick_step=None, p_value=None, adj_pval=False, star_pval=False,
                 alpha_points=1,
                 points_size=3, order_l=None,
                 box_height=2.5, box_aspect=.8, pvalue_pairs=None, sig_threshold=0.05, tick_rotation=0,
                 add_legend=False,
                 axis_fontsize=8, text_fontsize=7, annotator_height=None, label_newline=True, alpha=0.5,
                 showfliers=False,
                 title_height=None, figure_width=1.3, figure_height=1, title_fontsize=10, filename=None,
                 save_figure=False):
        """
        The box extends from the first quartile (Q1) to the third quartile (Q3) of the data, with a line at the median
        The whiskers are generally extended into 1.5*IQR distance on either side of the box.

        """
        # sns.set(style='ticks', context='talk')
        plt.style.use('default')
        plt.figure(figsize=(figure_width, figure_height))

        # sns.set_style("white")  # "whitegrid","dark","darkgrid","ticks"
        my_colors = color_dic.values()

        custom_marker_props = {'markersize': .15, 'marker': 'o',
                               'markerfacecolor': '#d4d4d4'}  # 'markerfacecolor': 'gray', 'marker': 'o',
        sns.set_palette(my_colors)
        lm = sns.catplot(data=data_df,
                         x=x_col,
                         y=y_col,
                         kind="box",
                         hue=gp_col,
                         height=box_height,
                         aspect=box_aspect,
                         linewidth=1.1,
                         order=order_l,
                         hue_order=list(color_dic.keys()),
                         flierprops=custom_marker_props,
                         showfliers=showfliers,
                         # showbox=False,
                         # whis = 100
                         boxprops={'edgecolor': 'none', 'alpha': 1},  #
                         whiskerprops={'color': "black", 'linestyle': "--", 'linewidth': .5},
                         capprops={'color': "black", 'linewidth': .5},
                         medianprops={'color': "white", 'linewidth': .5}
                         )  # ,
        for ax in lm.axes.flat:
            for box in ax.artists:
                box.set_alpha(alpha)
                box.set_edgecolor(None)
                box.set_linewidth(0)

        ax = lm.axes[0, 0]
        # sns.color_palette(" hls", 8)
        if palette is not None:
            np.random.seed(seed)
            sns.stripplot(x=x_col,
                          y=y_col,
                          data=data_df,
                          hue=color_col,
                          palette=palette,
                          alpha=alpha_points,
                          jitter=0.3,
                          size=points_size,
                          edgecolor='none',
                          linewidth=.1
                          )

            if add_legend:
                legend = plt.legend(title=color_col, bbox_to_anchor=(1.1, 0.4), loc='upper left', ncol=1, fontsize=4,
                                    frameon=False)
                legend.get_title().set_fontsize(6)
                for handle in legend.legendHandles:
                    handle.set_sizes([8])

                for label in legend.get_texts():
                    label.set_fontsize(6)
            else:
                plt.legend([], frameon=False)

        if p_value is not None:
            if pvalue_pairs is None:
                pvalue_pairs = [tuple(color_dic.keys())]
                pvals = [
                    MyVisualization.get_stat_text(p_value, sig_threshold=sig_threshold, star=star_pval, fdr=adj_pval)]
            else:
                pvals = [MyVisualization.get_stat_text(p, sig_threshold=sig_threshold, star=star_pval, fdr=adj_pval)
                         for p in p_value]
            annotator = Annotator(ax, pvalue_pairs,
                                  data=data_df,
                                  x=x_col,
                                  y=y_col,
                                  hue=gp_col,
                                  order=order_l,
                                  verbose=False
                                  )
            annotator._pvalue_format.fontsize = text_fontsize
            annotator.line_width = 0.5
            annotator.set_custom_annotations(pvals)
            annotator.annotate()

        # ax = plt.gca()
        ax.set_ylim(top=ylim_top, bottom=ylim_bottom)
        if ytick_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
        if add_counts:
            tick_labels = []
            for i, label in enumerate(ax.get_xticklabels()):
                category = label.get_text()
                size = len(data_df[data_df[x_col] == category])
                if label_newline:
                    tick_labels.append(f"{category}\n(n={size})")
                else:
                    tick_labels.append(f"{category} (n={size})")
            ax.set_xticklabels(tick_labels, rotation=tick_rotation)
        ax.tick_params(width=.6, length=2.5)
        ax.spines['bottom'].set_linewidth(.6)
        ax.spines['left'].set_linewidth(.6)
        plt.xlabel(x_label, fontweight='normal', fontsize=axis_fontsize)
        # plt.ylabel(y_label+ ' Gene Effect', fontweight='normal', fontsize=7)
        plt.ylabel(y_label, fontweight='normal', fontsize=axis_fontsize)
        plt.xticks(fontsize=6, rotation=tick_rotation)
        plt.yticks(fontsize=6)
        plt.grid(False)
        plt.suptitle(title, y=title_height, x=.6, fontsize=title_fontsize)
        # sns.set(font_scale=.4)
        if (filename is not None) or (save_figure):
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"boxplot_{filename}_{x_col}_{y_label}.pdf"),
                        dpi=60, bbox_inches="tight")
        plt.show()

    @staticmethod
    def jointplot(data_df, xcol, ycol, hue_col, color_dic=None, xlabel=None, ylabel=None, title='',
                  ylim_top=None, ylim_bottom=None, xlim_left=None, xlim_right=None,
                  legend_fontsize=None, title_height=None, title_fontsize=14, labels_fontsize=14,
                  figure_width=5, figure_height=5, save_figure=False):
        plt.style.use('default')
        plt.figure(figsize=(figure_width, figure_height))
        if color_dic is not None:
            my_colors = color_dic.values()
            sns.set_palette(my_colors)

        g = sns.jointplot(data=data_df, x=xcol, y=ycol, hue=hue_col, s=200, space=0, edgecolor='none',
                          marginal_kws={'edgecolor': 'none', 'alpha': .4}, alpha=0.7
                          )

        sns.move_legend(g.ax_joint, "upper left", bbox_to_anchor=(1.25, 1), frameon=False)
        g.ax_joint.set_xlabel(xlabel, fontsize=labels_fontsize)
        g.ax_joint.set_ylabel(ylabel, fontsize=labels_fontsize)

        plt.grid(False)
        plt.ylim(top=ylim_top, bottom=ylim_bottom)
        plt.xlim(left=xlim_left, right=xlim_right)
        plt.suptitle(title, y=title_height, x=.5, fontsize=title_fontsize)
        if save_figure:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"jointplot.pdf"),
                        dpi=60, bbox_inches="tight")
        plt.show()

    @staticmethod
    def forest_plot(coxph_mv_df, xlabel="Log Hazard Ratio", ylabel='', title='', table_format=True, decimal_precision=3,
                    tick_fontsize=8, sort_variables=True, title_fontsize=10, xlabel_fontsize=10, ylabel_fontsize=10,
                    xtick_l=None, ci_report=False, figure_width=5, figure_height=7, file_name=None):
        # plt.style.use('default')
        # sns.set(style='ticks', context='talk')
        # plt.figure(figsize=(figure_width, figure_height))
        data_df = coxph_mv_df.copy()
        data_df['q-value'] = data_df['q-value'].apply(lambda x: f'{x:.2e}' if x < 0.0001 else round(x, 3))
        data_df['exp(coef)'] = data_df['exp(coef)'].apply(lambda x: round(x, decimal_precision))
        if xtick_l is None:
            xtick_l = [-3, -2, -1, 0, 1, 2, 3]
        plt.figure(figsize=(figure_width, figure_height))
        fp.forestplot(data_df,  # the dataframe with results data
                      # estimate="exp(coef)",  # col containing estimated effect size
                      estimate="coef",  # col containing estimated effect size
                      ll="log_lower .95", hl="log_upper .95",  # lower & higher limits of conf. int.
                      varlabel="Variable",  # column containing the varlabels to be printed on far left
                      # capitalize="capitalize",  # Capitalize labels
                      # pval="Pr(>|z|)",  # column containing p-values to be formatted
                      # annote=["exp(coef)", "est_ci"],  # columns to report on left of plot
                      # annoteheaders=["HR", "Coef. (95% CI)"],  # ^corresponding headers
                      annote=["exp(coef)"],  # columns to report on left of plot
                      annoteheaders=["HR"],
                      rightannote=["q-value"],  # columns to report on right of plot
                      right_annoteheaders=["q-value"],  # ^corresponding headers
                      # groupvar="group",  # column containing group labels
                      # group_order=["labor factors", "occupation", "age", "health factors",
                      #              "family factors", "area of residence", "other factors"],
                      xlabel=xlabel,  # x-label title
                      xticks=xtick_l,  # x-ticks to be printed
                      sort=sort_variables,  # sort estimates in ascending order
                      table=table_format,  # Format as a table
                      # Additional kwargs for customizations
                      **{"marker": "D",  # set maker symbol as diamond
                         "markersize": 35,  # adjust marker size
                         "xlinestyle": (0, (10, 5)),  # long dash for x-reference line
                         "xlinecolor": "#808080",  # gray color for x-reference line
                         "xtick_size": 12,  # adjust x-ticker fontsize
                         },
                      ci_report=ci_report,
                      decimal_precision=decimal_precision,
                      starpval=False,
                      fontsize=8,
                      figsize=(figure_width, figure_height)
                      )

        plt.xticks(fontsize=tick_fontsize, rotation=0)
        plt.yticks(fontsize=tick_fontsize)
        plt.title(title, fontsize=title_fontsize)
        plt.xlabel(xlabel, fontsize=xlabel_fontsize)
        plt.ylabel(ylabel, fontsize=ylabel_fontsize)
        plt.grid(False)
        if file_name is not None:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"forest_plot_{file_name}.pdf"),
                        dpi=60, bbox_inches="tight")
        plt.show()

    @staticmethod
    def silhouette_plot(cor_data_df, labels_sr, lbl_xloc=-0.05, title='', add_avg_vline=False,
                        figure_width=7, figure_height=4, file_name=None):
        clusters_l = sorted(labels_sr.unique())
        labels_np = labels_sr.values
        cor_data_np = cor_data_df.values
        dist_np = 1 - cor_data_np
        sample_silhouette_values = silhouette_samples(dist_np, labels_np, metric="precomputed")

        plt.style.use('default')
        plt.figure(figsize=(figure_width, figure_height))
        y_lower = 10  # Space for the first cluster bar
        for c in clusters_l:
            # Extract silhouette scores for cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[labels_np == c]
            ith_cluster_silhouette_values.sort()

            # Compute the position for this cluster
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # Fill the silhouette plot with values
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, ith_cluster_silhouette_values,
                facecolor=MyVisualization.feat_colors_dic['Cluster'][c],
                alpha=0.7
            )

            # Label the cluster on the plot
            plt.text(lbl_xloc, y_lower + 0.5 * size_cluster_i, str(c))

            # Update y_lower for next cluster
            y_lower = y_upper + 10  # Add space between clusters

        # Average silhouette score
        if add_avg_vline:
            silhouette_avg = silhouette_score(dist_np, labels_np, metric="precomputed")
            plt.axvline(x=silhouette_avg, color="black", linestyle="--", linewidth=.7,
                        label=f"Average Silhouette Score = {silhouette_avg:.2f}")

        plt.title(title)
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster Label")
        plt.yticks([])  # Clear y-axis labels
        # plt.legend(loc="best")
        if file_name is not None:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"silhouette_plot_{file_name}.pdf"),
                        dpi=60, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_umap(data_df, labels_sr, colors_sr, groups_order_l=None, metric="euclidean", legend_title='',
                  point_size=100, add_points_lbls=False, text_clusters_l=None, umap_dim1_range=None,
                  umap_dim2_range=None, title='',
                  force_points=0.6, force_text=1, seed_loc=4241988, legend_loc='best', figure_width=10, figure_height=7,
                  file_name=None):
        plt.style.use('default')
        plt.figure(figsize=(figure_width, figure_height))
        # Use the default parameters of the R package
        umap_model = umap.UMAP(
            n_neighbors=15,  # Number of neighbors for graph construction
            n_components=2,  # Number of dimensions to reduce to
            metric=metric,  # Distance metric
            n_epochs=200,  # Number of optimization epochs
            init="spectral",  # Initialization method
            min_dist=0.1,  # Minimum distance between points in low-dim space
            spread=1.0,  # Spread of the embedding
            set_op_mix_ratio=1.0,  # Blend of union and intersection
            local_connectivity=1,  # Number of local connected components
            negative_sample_rate=5,  # Number of negative samples per positive sample
            random_state=seed_loc,  # Random state for reproducibility
            verbose=True,  # Enable progress information
            n_jobs=1
        )

        # Fit and transform the data
        embedding = umap_model.fit_transform(data_df)

        if groups_order_l is None:
            groups_order_l = sorted(labels_sr.unique())
        for label in groups_order_l:
            plt.scatter(
                embedding[labels_sr == label, 0],  # X-coordinates for label
                embedding[labels_sr == label, 1],  # Y-coordinates for label
                c=colors_sr[label],
                label=label,
                s=point_size,
                alpha=.6,
                edgecolor="none",
            )
        if add_points_lbls:

            # Get indices of samples that belong to cluster 3 and whose UMAP dimension 1 is within the range
            filtered_indices = np.where(
                (labels_sr == text_clusters_l) &
                (embedding[:, 0] >= umap_dim1_range[0]) &
                (embedding[:, 0] <= umap_dim1_range[1]) &
                (embedding[:, 1] >= umap_dim2_range[0]) &
                (embedding[:, 1] <= umap_dim2_range[1])
            )[0]
            print("Filtered indices len:", len(filtered_indices))
            texts = []
            for idx in filtered_indices:
                texts.append(plt.text(
                    embedding[idx, 0], embedding[idx, 1], labels_sr.index.tolist()[idx],
                    fontsize=8, color='black', ha='center', va='center', alpha=0.7
                ))
            adjust_text(texts, force_points=force_points, force_text=force_text,
                        only_move={'points': 'yx', 'texts': 'yx'},
                        # autoalign=True,
                        arrowprops=dict(arrowstyle="-", color='black', alpha=1, lw=0.2)
                        )

        plt.legend(title=legend_title, loc=legend_loc, title_fontsize=13, fontsize=12, frameon=False)
        plt.title(title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        if file_name is not None:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"umap_{file_name}.pdf"),
                        dpi=60, bbox_inches="tight")
        plt.show()
        emd_df = pd.DataFrame(embedding, index=labels_sr.index, columns=['UMAP1', 'UMAP2'])
        return emd_df

    @staticmethod
    def plot_clusters_similarity_graph(nodes_df, edge_weights_df, title='', node_size=1500, thickness_scale=26,
                                       font_scale=1, figure_width=8, figure_height=8, file_name=None):
        """
        :param nodes_df: A dataframe with node IDs as the index and two columns: group and color.
        :param edge_weights_df: A dataframe with three columns: two columns representing the neighbors of each edge,
                                and a third column for the weight.
        """
        sns.set(font_scale=font_scale)
        G = nx.Graph()

        # Number of vertices in each group
        # group_sizes = [5, 5, 5]  # Adjust as needed
        node_gps = nodes_df.groupby('group')
        no_gps = len(node_gps.groups)
        group_sizes_l = [len(node_gps.groups[g]) for g in node_gps.groups]

        # Add nodes to the graph
        G.add_nodes_from(nodes_df.index.tolist())

        # Define positions for a top view of the prism with an empty center
        positions = {}

        # Define the base vertices of the triangle
        radius = 4  # Radius of the base triangle (for the outer groups)
        gap_radius = 1  # Radius of the gap in the middle
        angle_offset = np.pi / 2  # Rotate the triangle so one vertex points upward
        base_vertices = [
            (radius * np.cos(angle_offset + 2 * np.pi * i / no_gps),
             radius * np.sin(angle_offset + 2 * np.pi * i / no_gps))
            for i in range(no_gps)
        ]

        # Function to interpolate positions along a line, starting after the gap
        def interpolate_positions_with_gap(start, end, gap, num_points):
            x0, y0 = start
            x1, y1 = end
            # Start the interpolation after the gap
            return [
                (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
                for t in np.linspace(gap, 1, num_points)  # Start from `gap` instead of 0
            ]

        # Assign positions to vertices along the lines from the gap edge to the base vertices
        nodes_l = nodes_df.index.tolist()
        current_node = 0
        group_nodes = []
        for i, base_vertex in enumerate(base_vertices):
            group_nodes.append([])  # Store the nodes in each group
            points = interpolate_positions_with_gap((0, 0), base_vertex, gap_radius / radius, group_sizes_l[i])
            for point in points:
                positions[nodes_l[current_node]] = point
                group_nodes[i].append(current_node)  # Keep track of the group
                current_node += 1

        # Add edges with scores to the graph
        for _, row in edge_weights_df.iterrows():
            G.add_edge(row['n1'], row['n2'], thickness=row['weight'], color=row['color'])
        edge_thickness = [G[u][v]['thickness'] * thickness_scale for u, v in G.edges()]
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]

        # Draw the graph with custom node colors
        plt.figure(figsize=(figure_width, figure_height))
        nx.draw(
            G,
            pos=positions,
            with_labels=True,  # Disable node labels
            node_size=node_size,
            node_color=nodes_df['color'].values.tolist(),  # Use the custom list for node colors
            font_weight='bold',
            width=edge_thickness,  # Use thickness scores for edge width
            edge_color=edge_colors
        )
        plt.title(title, fontsize=18)
        if file_name is not None:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(os.path.join(io_library.output_dir, f"{file_name}.pdf"),  # dpi=300,
                        bbox_inches="tight")  # dpi=60,
        plt.show()

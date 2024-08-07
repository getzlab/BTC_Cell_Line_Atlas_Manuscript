import os
import random
import numpy as np
import pandas as pd
from scipy.stats import zscore
import glob
import gseapy as gp
from multiprocessing import Pool, cpu_count


import BTC_Cell_Line_Atlas_Manuscript.scripts.io_library as io_library
from BTC_Cell_Line_Atlas_Manuscript.scripts.io_library import MyLib
from BTC_Cell_Line_Atlas_Analysis.scripts.invoking_R_functions import MyRFunctions

class GSEA:

    @staticmethod
    def process_gene_set(args):
        index, gene_set_id, rna_df, gene_sets_dic = args
        print(index)
        ssgsea_df = MyRFunctions.ssgsea(rna_df, gene_sets_dic[gene_set_id])
        return gene_set_id, ssgsea_df

    @staticmethod
    def ssgsea_parallel(rna_df, gene_sets_dic):

        args = [(index, gene_set_id, rna_df, gene_sets_dic) for index, gene_set_id in
                enumerate(gene_sets_dic)]

        gene_sets_ssgsea_df = pd.DataFrame(index=rna_df.columns, columns=gene_sets_dic.keys())
        print('Number of cpus:', cpu_count())
        with Pool(cpu_count()) as pool:
            results = pool.map(GSEA.process_gene_set, args)

        # Fill the DataFrame with the results
        for gene_set_id, ssgsea_df in results:
            gene_sets_ssgsea_df[gene_set_id] = ssgsea_df

        return gene_sets_ssgsea_df

if __name__ == "__main__":

    print("\nProgram started at", MyLib.get_time())
    print()
    working_dir = "/Users/nkarisan/PycharmProjects/BTC_Cell_Line_Atlas_Manuscript"
    os.chdir(working_dir)
    io_library.input_dir = 'input_data/'
    io_library.output_dir = 'Figure5/output_fig5/'
    rna_all_file = 'Main_Data/RNA/rna_cell_lines.csv'
    upset_file = 'cell_lines_master_upset_2024-4-13.csv'

    raw_upset_df = MyLib.load_csv(upset_file)
    raw_upset_df.set_index('Cell_line', inplace=True, drop=True)
    upset_df = raw_upset_df[(raw_upset_df['RNA'] == 1) & (raw_upset_df['Subtype'] != 'HCC')]

    exp_df = MyLib.load_csv(rna_all_file, index_col=0)
    exp_df = exp_df.T  # samples are the rows
    fold_exp_df = exp_df.apply(lambda x:x-x.median())
    btc_fold_exp_df = fold_exp_df.loc[upset_df.index].T

    msigdb_dir = 'input_data/Additional_Data/Selected_Lists/msigdb_v2023.2.Hs_GMTs/'
    gene_sets_dic = dict()
    for filename in glob.glob(os.path.join(msigdb_dir, '*.symbols.gmt')):
        print(filename)
        collection_dic = gp.read_gmt(filename)
        print(len(collection_dic))
        gene_sets_dic.update(collection_dic)

    gene_sets_filtered_dic = dict()
    for index, gene_set_id in enumerate(gene_sets_dic):
        shared_genes = sorted(set(gene_sets_dic[gene_set_id]).intersection(btc_fold_exp_df.index))
        if len(shared_genes) > 1:
            gene_sets_filtered_dic[gene_set_id] = shared_genes

    print('\nTotal gene sets:', len(gene_sets_filtered_dic))

    ssGSEA_df = GSEA.ssgsea_parallel(btc_fold_exp_df, gene_sets_filtered_dic)
    ssgsea_zscore_df = ssGSEA_df.apply(zscore)
    # MyLib.save_csv(ssGSEA_df, 'gene_sets_ssgsea.csv')
    MyLib.save_csv(ssgsea_zscore_df, 'gene_sets_ssgsea_zscore.csv')

    # ---------------------------

    print("\nProgram finished at", MyLib.get_time())


import pandas as pd
import category_encoders as ce
import os

import scripts.io_library as io_library
from scripts.io_library import MyLib
# ==================================================================

index_column = 'ModelID'

class DataSet:

    @staticmethod
    def create_dataset(datasets_dic, upset_df, file_suffix=''):
        """
        Features: expression, cn, damaging_mutations, hotspot_mutations, other_mutations, fusions, BTC lineage
        """
        ids = datasets_dic['CRISPR'].index.tolist() # Target IDs

        datasets_dic['MutHot'][datasets_dic['MutHot'] > 1] = 1
        datasets_dic['MutDam'][datasets_dic['MutDam'] > 1] = 1

        datasets_dic['MutOther'] = datasets_dic['MutOther'][datasets_dic['MutOther'][index_column].isin(ids)]
        all_mut_df = Mutation.process_mutations(datasets_dic['MutOther'])
        other_mut_df = Mutation.create_other_mutation_df(all_mut_df, datasets_dic['MutHot'])
        other_mut_df = Mutation.create_other_mutation_df(other_mut_df, datasets_dic['MutDam'])
        datasets_dic['MutOther'] = other_mut_df

        datasets_dic['Fusion'] = datasets_dic['Fusion'][datasets_dic['Fusion'][index_column].isin(ids)]
        datasets_dic['Fusion'] = Fusions.process_fusions(datasets_dic['Fusion'], ids)

        indices = DataSet.get_shared_indices(datasets_dic)
        data_df = pd.DataFrame(index=indices)
        for ext in datasets_dic:
            df = datasets_dic[ext].loc[indices, :].copy()
            if ext not in ['CRISPR']:
                df = DataSet.rename_columns(df, ext)
                data_df = pd.concat([data_df, df], axis=1)

        # Add subtype features
        encoder = ce.OneHotEncoder(use_cat_names=True)
        temp_df = encoder.fit_transform(upset_df['Subtype'])
        subtype_df = pd.DataFrame(0, index=indices, columns=temp_df.columns)
        subtype_df.loc[temp_df.index] = temp_df
        data_df = pd.concat([data_df, subtype_df], axis=1)

        # split BTC and other CCLE datasets
        main_ids = upset_df.index.tolist()
        features_main_df = data_df.loc[main_ids].copy()
        features_main_df = DataSet.clean_matrix(features_main_df)
        features_other_df = data_df.loc[~data_df.index.isin(main_ids), features_main_df.columns].copy()

        targets_df = datasets_dic['CRISPR'].loc[main_ids].copy()
        targets_other_df = datasets_dic['CRISPR'].loc[features_other_df.index].copy()

        MyLib.save_hdf5(features_main_df, f'Features_{file_suffix}.hdf5')
        MyLib.save_hdf5(targets_df, f'Targets_{file_suffix}_GeneEffects.hdf5')

        MyLib.save_hdf5(features_other_df, 'Features_Other_CCLE.hdf5')
        MyLib.save_hdf5(targets_other_df, 'Targets_Other_CCLE_GeneEffects.hdf5')

    @staticmethod
    def clean_matrix(data_df):
        data_df = data_df.dropna(axis=1).copy()
        constant_cols = data_df.columns[data_df.nunique() == 1]
        data_df.drop(columns=constant_cols, inplace=True)
        return data_df

    @staticmethod
    def get_shared_indices(datasets_dic):
        inds_l = []
        for ele in datasets_dic:
            inds_l.append(set(datasets_dic[ele].index.tolist()))
        indices = set.intersection(*inds_l)
        return sorted(indices)

    @staticmethod
    def rename_columns(data_df, ext):
        data_df = data_df.rename(columns={col: col.split(' (')[0] for col in data_df.columns})
        data_df = data_df.reindex(sorted(data_df.columns), axis=1)
        data_df.rename(columns={col: col+'_'+ext for col in data_df.columns}, inplace=True)
        return data_df

    @staticmethod
    def match_rows_df(df1, df2):
        df = pd.merge(df1, df2, how='inner', left_index=True, right_index=True, sort=True)
        new_df1 = df1.loc[df.index, :]
        new_df2 = df2.loc[df.index, :]
        return (new_df1, new_df2)

    @staticmethod
    def match_matrices(df1, df2):
        cols = sorted(set(df1.columns).intersection(set(df2.columns)))
        df1, df2 = DataSet.match_rows_df(df1[cols], df2[cols])
        return (df1, df2)

#==============================================================
class Preprocessing:
    """
    Contains commonly used methods
    """

    @staticmethod
    def association_matrix(data_df, feature_col, na_value, target_col=None):
        features = sorted(data_df[feature_col].unique())
        cell_feat_df = pd.DataFrame(na_value, index=sorted(data_df[index_column].unique()), columns=features)
        for col in features:
            cur_cells = data_df[data_df[feature_col] == col][index_column]
            if target_col is None:
                cell_feat_df.loc[cur_cells, col] = 1
            else:
                indices = data_df[data_df[feature_col] == col].index
                cell_feat_df.loc[cur_cells, col] = data_df.loc[indices, target_col].values

        return cell_feat_df

    @staticmethod
    def filter_columns(mat_df, threshold):
        sum_sr = mat_df.sum(axis=0)
        genes = sum_sr.loc[sum_sr >= threshold].index.tolist()
        mat_df = mat_df.loc[:, genes]
        return mat_df

#==============================================================
class Fusions:

    @staticmethod
    def process_fusions(raw_data_df, index_set, threshold=None):

        data_df = Preprocessing.association_matrix(raw_data_df, feature_col='FusionName', na_value=0)
        new_df = pd.DataFrame(0, index=index_set, columns=data_df.columns)
        new_df.loc[data_df.index] = data_df
        if threshold is not None:
            new_df = Preprocessing.filter_columns(new_df, threshold)
        return new_df

#==============================================================
class Mutation:

    @staticmethod
    def process_mutations(raw_mut_df, threshold=None, SAVE=False):
        mut_df = Mutation.extract_selected_mutations(raw_mut_df)
        # cell_gene_df = Preprocessing.association_matrix(mut_df, 'HugoSymbol', na_value=0)
        cell_gene_df = Preprocessing.association_matrix(mut_df, 'gene', na_value=0)
        if threshold is not None:
            cell_gene_df = Preprocessing.filter_columns(cell_gene_df, threshold)
        if SAVE:
            MyLib.save_hdf5(cell_gene_df, f'cell_mutations_{threshold}.hdf5')
        return cell_gene_df

    @staticmethod
    def extract_selected_mutations(raw_mut_df):
        """

        """
        mut_types_l = ['NONSENSE', 'FRAME_SHIFT_DEL', 'FRAME_SHIFT_INS', 'MISSENSE', 'SILENT', 'SPLICE_SITE',
                       'IN_FRAME_INS', 'NONSTOP', 'IN_FRAME_DEL']
        mut_df = raw_mut_df[raw_mut_df['VariantInfo'].isin(mut_types_l)]
        mut_df = mut_df[[index_column, 'VariantInfo', 'EntrezGeneID', 'HugoSymbol']][mut_df != "Unknown"].dropna().astype(
            {'EntrezGeneID': 'float'}).astype({'EntrezGeneID': 'int'})
        # For future applications
        mut_df['gene'] = mut_df[['HugoSymbol', 'EntrezGeneID']].apply(
            lambda x: x[0].strip() + ' (' + str(x[1]) + ')', axis=1)
        mut_df.reset_index(drop=True, inplace=True)
        return mut_df

    @staticmethod
    def create_other_mutation_df(all_mut_df, df):
        df1, df2 = DataSet.match_matrices(all_mut_df, df)

        shared_df = df1 & df2
        other_df = df1 ^ shared_df
        all_mut_df.loc[other_df.index, other_df.columns] = other_df
        return all_mut_df

# ==================================================================


if __name__ == "__main__":

    print("\nProgram started at", MyLib.get_time())
    print()
# ---------------------------

working_dir = "/Users/nkarisan/PycharmProjects/BTC_Cell_Line_Atlas_Manuscript"
os.chdir(working_dir)

io_library.input_dir = 'input_data/'
io_library.output_dir = 'predictive_model/output_files_RF/dependency_prediction_data/'

upset_file = 'cell_lines_master_upset_2025-2-24.csv'
raw_upset_df = MyLib.load_csv(upset_file, index_col=0)
upset_df = raw_upset_df[(raw_upset_df['CRISPR'] == 1) & (raw_upset_df['Subtype'] != 'HCC')]
upset_df = upset_df[["Subtype", 'DepMap_ID']]
upset_df.set_index('DepMap_ID', inplace=True)


raw_datasets_dic = dict()

data_dir = 'Additional_Data/DepMap/'


raw_datasets_dic['CRISPR'] = MyLib.load_csv(os.path.join(data_dir, 'CRISPRGeneEffect.csv'), index_col=0)
raw_datasets_dic['CRISPR'] = raw_datasets_dic['CRISPR'].drop('ACH-000145') # not part of the initial dataset
raw_datasets_dic['RNA'] = MyLib.load_csv(os.path.join(data_dir, 'OmicsExpressionProteinCodingGenesTPMLogp1.csv'), index_col=0)
raw_datasets_dic['CNV'] = MyLib.load_csv(os.path.join(data_dir, 'OmicsCNGene.csv'), index_col=0)
raw_datasets_dic['MutHot'] = MyLib.load_csv(os.path.join(data_dir, 'OmicsSomaticMutationsMatrixHotspot.csv'),
                                            index_col=0).astype('int64')
raw_datasets_dic['MutDam'] = MyLib.load_csv(os.path.join(data_dir, 'OmicsSomaticMutationsMatrixDamaging.csv'),
                                            index_col=0).astype('int64')
raw_datasets_dic['Fusion'] = MyLib.load_csv(os.path.join(data_dir,'OmicsFusionFiltered.csv'))
raw_datasets_dic['MutOther'] = MyLib.load_csv(os.path.join(data_dir, 'OmicsSomaticMutations.csv'))

DataSet.create_dataset(raw_datasets_dic, upset_df=upset_df, file_suffix='BTC')


# ---------------------------
print("\nProgram finished at", MyLib.get_time())













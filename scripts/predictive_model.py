import os
import random
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn import preprocessing
from multiprocessing import Pool, cpu_count
import multiprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import BTC_Cell_Line_Atlas_Manuscript.scripts.io_library as io_library
from BTC_Cell_Line_Atlas_Manuscript.scripts.io_library import MyLib


initial_seed = 2023

def calculate_corr(args):
    feature, target_vals = args
    cor = abs(feature.corrwith(target_vals))
    return cor

class BaselineLearner:

    @staticmethod
    def init_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def zscore(df):
        nan_cols = df.columns[df.isna().all()].tolist()
        none_nan_cols = df.columns.difference(nan_cols).tolist()
        var = preprocessing.StandardScaler()
        var.fit(df[none_nan_cols])
        new_x_df = pd.DataFrame(var.transform(df[none_nan_cols]), index=df.index, columns=none_nan_cols)
        new_x_df = pd.concat([new_x_df, df[nan_cols]], axis=1)
        return new_x_df[df.columns]

    @staticmethod
    def zscore_numericals(df):
        binary_features = df.columns[df.nunique() <= 2]
        numerical_features = df.columns.difference(binary_features).tolist()
        binary_df = df[binary_features].astype(int)
        numerical_df = df[numerical_features]
        var = preprocessing.StandardScaler()
        var.fit(numerical_df)
        x_numerical_df = pd.DataFrame(var.transform(numerical_df), index=numerical_df.index,
                                            columns=numerical_features)
        x = pd.concat([x_numerical_df, binary_df], axis=1)
        return x

    @staticmethod
    def zscore_numericals_train_test(train_df, test_df):
        binary_features = train_df.columns[train_df.nunique() <= 2]
        numerical_features = train_df.columns.difference(binary_features)
        train_binary_df = train_df[binary_features].astype(int)
        test_binary_df = test_df[binary_features].astype(int)
        train_numerical_df = train_df[numerical_features]
        var = preprocessing.StandardScaler()
        var.fit(train_numerical_df)
        x_train_numerical_df = pd.DataFrame(var.transform(train_numerical_df), index=train_numerical_df.index,
                                            columns=train_numerical_df.columns)
        x_test_numerical_df = pd.DataFrame(var.transform(test_df[numerical_features]), index=test_df.index,
                                           columns=numerical_features)
        x_train = pd.concat([x_train_numerical_df, train_binary_df], axis=1)
        x_test = pd.concat([x_test_numerical_df, test_binary_df], axis=1)
        return (x_train, x_test)

    @staticmethod
    def zscore_train_test(train_df, test_df):
        var = preprocessing.StandardScaler()
        var.fit(train_df)
        x_train = pd.DataFrame(var.transform(train_df), index=train_df.index, columns=train_df.columns)
        x_test = pd.DataFrame(var.transform(test_df), index=test_df.index, columns=test_df.columns)
        return (x_train, x_test)

    @staticmethod
    def start(feat_df, target_df, genes_l):
        """
            applies the pipeline for each gene in "genes_l"
        """
        ite = 1  # in case multiple iterations were added

        summary_df = pd.DataFrame(index=['corr', 'mse', 'mae'])#, 'train_mse'
        preds_df = pd.DataFrame(index=feat_df.index)

        target_binary_df = target_df.copy()
        target_binary_df[target_binary_df > -0.5] = 0
        target_binary_df[target_binary_df <= -0.5] = 1

        for index, cur_target in enumerate(genes_l):
            BaselineLearner.init_seed(123456 + initial_seed * ite)
            if sum(target_binary_df[cur_target]) >= 0:
                print('==================================================================================\n')
                print(f"{index + 1} / {len(genes_l)}: {cur_target}\n")

                summary_df, preds_df = BaselineLearner.run_pipeline(summary_df, preds_df, feat_df, target_df,
                                                                    cur_target, target_binary_df[cur_target])

        summary_df.to_csv(os.path.join(io_library.output_dir, f'1RF_summary.csv'), index_label='DepMap_ID')
        preds_df.to_csv(os.path.join(io_library.output_dir, f'1RF_preds.csv'), index=True, index_label='DepMap_ID')

    @staticmethod
    def run_pipeline(summary_df, preds_df, feat_df, target_df, target, target_binary_df):
        """
            1- fits a model
            2- Computes the residuals of the predicted values.
            4- creates a .csv file corresponding to the target gene, which includes:
                a- the model performance
                b- sorted list of feature importance scores
                c- gene effects of the cell lines and their corresponded predicted values

        """

        X_df = feat_df.reset_index(drop=True)
        y_df = target_df[[target]]
        y_df.reset_index(drop=True, inplace=True)

        preds, mse, corr, mae, feat_scores_sr, train_mse = BaselineLearner.train_regressor(X_df, target_df, target)

        preds_df[target] = preds.tolist()

        df1 = pd.DataFrame(
            {'mse': [round(mse, 4)], 'mae': [round(mae, 4)], 'correlation': [round(corr, 4)]})# 'train_mse': [round(train_mse, 4)]

        residuals = y_df.loc[:, target].values - preds
        # residuals.index = feat_df.index
        summary_df[target] = [round(corr, 4), round(mse, 4), round(mae, 4)]#, round(train_mse, 4)

        # features_df = BaselineLearner.get_important_features(feat_df, y_df, target)

        model_name = f'RF'
        # df2 = pd.DataFrame({'genes': features_df['gene'].tolist(),  # [:feat_df.shape[0]],
        #                     f'scores_{model_name}': features_df['score'].round(3).tolist()})  # [:feat_df.shape[0]]
        # df2.set_index('genes', inplace=True, drop=True)
        # df2 = df2.T

        feat_scores_sr.sort_values(ascending=False, inplace=True)
        df3 = pd.DataFrame(feat_scores_sr, columns=['scores_folds'])
        df3['scores_folds'] = df3['scores_folds'].round(3)
        df3 = df3.T

        df4 = pd.DataFrame({'DepMap_ID': feat_df.index.tolist(),
                            f'{target}_gene effect': y_df[target].round(3).tolist(),
                            f'preds_{model_name}': preds.round(3).tolist(),
                            f'residuals_{model_name}': residuals.round(3).tolist()
                            })
        df4.set_index('DepMap_ID', inplace=True, drop=True)
        df4.sort_values(f'residuals_{model_name}', ascending=False, inplace=True)

        # dfs = [df1, df2, df3, df4]
        dfs = [df1, df3, df4]

        with open(os.path.join(io_library.output_dir, f'{model_name}_{target}.csv'), 'a') as f:
            for df in dfs:
                df.to_csv(f)
                f.write("\n")
        return (summary_df, preds_df)

    @staticmethod
    def train_regressor(feat_df, target_df, target):

        cor_folds = []
        mse_total = 0
        mae_total = 0
        train_mse = 0
        preds_hold = pd.Series(np.nan, index=target_df.index, name=target)
        # target_binary_df.reset_index(inplace=True,drop=True)
        feature_scores_sr = pd.Series(dtype='float64')
        target_df.reset_index(drop=True, inplace=True)
        # dm_features = BaselineLearner.get_topk_corr(dm_features_df, dm_targets_df.loc[:, target], k=10)

        i = 0
        k = 10
        # k = 5
        # kfold = StratifiedKFold(n_splits=k, shuffle=True)
        # kfold.get_n_splits(feat_df, target_binary_df)
        # for train_index, test_index in kfold.split(feat_df, target_binary_df):
        kfold = KFold(n_splits=k, shuffle=True)
        for train_index, test_index in kfold.split(feat_df.index):
            # print(f'Test indices: {test_index}')
            print(f'\nFold {i}-----------------------------------')
            i += 1

            train_expr_df = feat_df.loc[train_index, :].copy()
            train_crsp_df = target_df.loc[train_index, :].copy()

            test_expr_df = feat_df.loc[test_index, :].copy()
            test_crsp_df = target_df.loc[test_index, :].copy()

            test_expr_df.reset_index(inplace=True, drop=True)
            test_crsp_df.reset_index(inplace=True, drop=True)

            if pancancer_features:
                train_expr_df = pd.concat([Other_CCLE_features_df, train_expr_df], axis=0)
                train_crsp_df = pd.concat([Other_CCLE_targets_df, train_crsp_df], axis=0)
            # train_crsp_df = pd.concat([dm_targets_df[[target]], train_crsp_df[[target]]], axis=0)
            features = BaselineLearner.get_topk_corr(train_expr_df, train_crsp_df.loc[:, target], k=1000)

            if pancancer_train and not pancancer_features:
                train_expr_df = pd.concat([Other_CCLE_features_df, train_expr_df], axis=0)
                train_crsp_df = pd.concat([Other_CCLE_targets_df, train_crsp_df], axis=0)

            if co_dependency:
                co_features = BaselineLearner.get_codependent_features(train_expr_df, train_crsp_df, target)
                print(f'no of co_dependent feats: {len(co_features)}')
                features = sorted(set(features).union(set(co_features)))
                print(f'total {len(features)}')

            train_expr_df = train_expr_df.loc[:, features].copy()
            train_crsp_df = train_crsp_df[[target]]

            # if DepMap:
            #     train_expr_df = pd.concat([dm_features_df[features], train_expr_df], axis=0)
            #     train_crsp_df = pd.concat([dm_targets_df[[target]], train_crsp_df], axis=0)

            train_expr_df.reset_index(inplace=True, drop=True)
            train_crsp_df.reset_index(inplace=True, drop=True)

            if feature_scores_sr.empty:  # only first time
                feature_scores_sr = pd.Series(np.nan, index=features, name='folds_scores')

            x_train, x_test = BaselineLearner.zscore_train_test(train_expr_df, test_expr_df[features])
            # x_train, x_test = BaselineLearner.zscore_numericals_train_test(train_expr_df, test_expr_df[features])

            y_train = train_crsp_df.loc[:, target].to_numpy().ravel()
            y_test = test_crsp_df.loc[:, target].to_numpy().ravel()

            pred, reg_model = BaselineLearner.RF_regressor(x_train, y_train, x_test)

            pred_train = reg_model.predict(x_train)
            train_mse += sklearn.metrics.mean_squared_error(y_train, pred_train)
            # train_corr, _ = pearsonr(y_train, pred_train)
            # print('train corr: ', corr)

            cur_feat_scores = pd.Series(list(reg_model.feature_importances_), index=features)
            feature_scores_sr = feature_scores_sr.add(cur_feat_scores, fill_value=0)

            mse = sklearn.metrics.mean_squared_error(y_test, pred)
            mae = sklearn.metrics.mean_absolute_error(y_test, pred)
            evar = sklearn.metrics.explained_variance_score(y_test, pred)
            corr, _ = pearsonr(y_test, pred)

            cor_folds.append(corr)
            mse_total += mse
            mae_total += mae
            preds_hold.iloc[test_index] = pred[:]

            print('MSE:', round(mse, 3))
            print('MAE:', round(mae, 3))
            print('Exp. Var.:', round(evar, 3))
            print('Pearsons correlation: ', round(corr, 3))

        print('===============================================')
        hold_corr, _ = pearsonr(target_df[target].values, preds_hold.values)
        print('Hold correlation: ', round(hold_corr, 3))

        instant_corr = sum(cor_folds) / k
        print('Instant correlation: ', round(instant_corr, 3))

        rscore = r2_score(target_df[target].values, preds_hold.values)
        print('R-squared: ', round(rscore, 3))

        mse = mse_total / k
        print('MSE: ', round(mse, 3))

        mae = mae_total / k
        print('MAE: ', round(mae, 3))

        feature_scores_sr = feature_scores_sr / k

        train_mse = train_mse / k

        return (preds_hold, mse, hold_corr, mae, feature_scores_sr, train_mse)

    @staticmethod
    def get_topk_corr(x, target_vals, k=1000):
        x = BaselineLearner.drop_low_vars(x)
        # new_x = BaselineLearner.zscore_numericals(x)
        new_x = BaselineLearner.zscore(x)
        target_vals = pd.Series(target_vals, index=x.index)
        n_jobs = multiprocessing.cpu_count()
        with Pool(processes=n_jobs) as pool:
            chunk_size = len(new_x.columns) // n_jobs
            args_list = [(new_x.iloc[:, i:i + chunk_size], target_vals) for i in
                         range(0, len(new_x.columns), chunk_size)]
            results = pool.map(calculate_corr, args_list)

        cor_target = pd.concat(results, axis=0)
        cor_target = cor_target.sort_values(ascending=False)
        relevant_feat_sr = cor_target[:k]
        return relevant_feat_sr.index.tolist()

    @staticmethod
    def drop_low_vars(exp_df, threshold=1e-10):
        vars_sr = exp_df.var()
        mask = vars_sr > threshold
        exp_df = exp_df.loc[:, exp_df.columns[mask]]
        return exp_df

    @staticmethod
    def filter_sparse_samples(df, max_allowed_nans_zeros_frac=0.1):
        max_allowed_nans_zeros = df.shape[0] * max_allowed_nans_zeros_frac
        filtered_df = df.loc[:, (df.isnull().sum(axis=0) + (df == 0).sum(axis=0)) < max_allowed_nans_zeros]
        return filtered_df

    @staticmethod
    def get_codependent_features(features_df, targets_df, target):
        temp_targets_df = targets_df.drop(columns=target, axis=1)
        temp_targets_df = BaselineLearner.filter_sparse_samples(temp_targets_df)
        codependent_targets = BaselineLearner.get_topk_corr(temp_targets_df, targets_df.loc[:, target], k=20)
        features = []
        for gene in codependent_targets:
            cur_y = targets_df[[gene]].dropna(axis=0)
            cur_features_df, cur_y = BaselineLearner.match_rows_df(features_df, cur_y)
            cur_features = BaselineLearner.get_topk_corr(cur_features_df, cur_y.loc[:, gene], k=1000)
            _, reg_model = BaselineLearner.RF_regressor(cur_features_df[cur_features], cur_y.values.ravel())
            cur_feat_scores = pd.Series(list(reg_model.feature_importances_), index=cur_features)
            cur_feat_scores.sort_values(inplace=True, ascending=False)
            features += cur_feat_scores.index.tolist()[:20]
        return sorted(set(features))


    @staticmethod
    def RF_regressor(x_train, y_train, x_test=None):
        reg_model = RandomForestRegressor(n_jobs=-1, max_depth=6,
                                          n_estimators=100, min_samples_leaf=5,
                                          ).fit(x_train, y_train)

        if x_test is not None:  # if only trained model is needed
            pred = reg_model.predict(x_test)
        else:
            pred = None

        return pred, reg_model

    @staticmethod
    def gradientboost_regressor(x_train, y_train, x_test=None):
        reg_model = GradientBoostingRegressor(random_state=initial_seed,
                                                  max_depth=6).fit(x_train, y_train)

        if x_test is not None:  # if only trained model is needed
            pred = reg_model.predict(x_test)
        else:
            pred = None

        return pred, reg_model


    @staticmethod
    def get_important_features(feat_df, targets_df, target):

        # features = Var.get_features('all', target, 1000, feat_df, targets_df)
        features = BaselineLearner.get_topk_corr(feat_df, targets_df.loc[:, target], k=1000)

        X = feat_df[features].copy()
        X = X.apply(stats.zscore)
        y = targets_df.loc[:, target].to_numpy().ravel()

        _, reg_model = BaselineLearner.RF_regressor(X, y)
        feature_importances = pd.DataFrame({'gene': X.columns.tolist(),
                                            'score': list(reg_model.feature_importances_)})

        feature_importances.sort_values(by='score', ascending=False, inplace=True)

        return feature_importances

    @staticmethod
    def match_rows_df(df1, df2):
        df1.index = df1.index.astype(str)
        df2.index = df2.index.astype(str)
        df = pd.merge(df1, df2, how='inner', left_index=True, right_index=True, sort=True)
        new_df1 = df1.loc[df.index, :]
        new_df2 = df2.loc[df.index, :]

        return (new_df1, new_df2)

    @staticmethod
    def match_columns_df(data_df1, data_df2):
        matched_features = sorted(set(data_df1.columns.tolist()).intersection(set(data_df2.columns.tolist())))
        # print('number of shared columns: ', len(matched_features))
        return data_df1[matched_features], data_df2[matched_features]


if __name__ == "__main__":

    print("\nProgram started at", MyLib.get_time())
    print()
    working_dir = "/Users/nkarisan/PycharmProjects/BTC_Cell_Line_Atlas_Manuscript"
    os.chdir(working_dir)
    io_library.input_dir = 'input_data/'
    io_library.output_dir = 'input_data/random_forest_results/'

    data_dir = 'Remaining_Analyses/output_figs_random_forest/dependency_prediction_data' # create a folder with this name inside input_data
    main_features_file = 'Features_BTC.hdf5'
    main_targets_file = 'Targets_BTC_GeneEffects.hdf5'
    other_CCLE_features_file = 'Features_Other_CCLE.hdf5'
    other_CCLE_targets_files = 'Targets_Other_CCLE_GeneEffects.hdf5'

    main_features_df = MyLib.load_h5py(os.path.join(data_dir, main_features_file))
    main_targets_df = MyLib.load_h5py(os.path.join(data_dir, main_targets_file))
    Other_CCLE_features_df = MyLib.load_h5py(os.path.join(data_dir, other_CCLE_features_file))
    Other_CCLE_targets_df = MyLib.load_h5py(os.path.join(data_dir, other_CCLE_targets_files))

    pancancer_features = False # Use all DepMap to select the top 1000 correlated features
    pancancer_train = True # Use other DepMap in the training step

    co_dependency = False # Add features from co-dependent targets


    onco_file = ''

    target_genes = ['MDM4 (4194)','BRAF (673)', 'ERBB2 (2064)', 'PPCDC (60490)', 'TP63 (8626)', 'FDFT1 (2222)',
                    'STAMBP (10617)', 'PRKCI (5584)', 'KIF18B (146909)', 'ITGB5 (3693)', 'CTNNB1 (1499)', 'SREBF1 (6720)',
                    'ELMO2 (63916)', 'PRKRA (8575)', 'MAPK1 (5594)','MAP2K1 (5604)','TGFBR1 (7046)', 'SQLE (6713)','TYMS (7298)',
                    'DPM2 (8818)', 'SLC7A1 (6541)', 'MBTD1 (54799)', 'FPGS (2356)', 'KRAS (3845)','IPPK (64768)', 'HNF1B (6928)',
                    'NXT1 (29107)', 'GRHL2 (79977)', 'UXS1 (80146)', 'ITGAV (3685)', 'TEAD1 (7003)', 'STX4 (6810)', 'PARD6B (84612)',
                    'NAMPT (10135)', 'FADD (8772)', 'FERMT2 (10979)', 'EGFR (1956)', 'SOX9 (6662)','STXBP3 (6814)', 'NCKAP1 (10787)',
                    'SNAP23 (8773)', 'BRK1 (55845)', 'MARK2 (2011)', 'WWTR1 (25937)', 'CRKL (1399)', 'RAB10 (10890)', 'SLC7A5 (8140)',
                    'AHCYL1 (10768)', 'SLC2A1 (6513)', 'RFK (55312)', 'ARHGEF7 (8874)', 'SLC3A2 (6520)', 'ITGB1 (3688)',
                    'RIC1 (57589)', 'TUBB4B (10383)', 'CRK (1398)', 'RAB6A (5870)', 'HDAC7 (51564)','ITGA3 (3675)', 'CDK4 (1019)',
                    'IDH3A (3419)', 'DHODH (1723)', 'NMNAT1 (64802)', 'ABI1 (10006)', 'CDH1 (999)', 'PIK3CA (5290)', 'PTPN11 (5781)',
                    'PRMT5 (10419)'
                    ]
    target_genes = sorted(target_genes)

    BaselineLearner.start(main_features_df, main_targets_df, target_genes)

    # ---------------------------

    print("\nProgram finished at", MyLib.get_time())

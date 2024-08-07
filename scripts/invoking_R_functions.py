import pandas as pd
import numpy as np
import math
from statsmodels.stats.multitest import fdrcorrection
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()



class MyRFunctions:
    r_script_path = 'scripts/utils_R.R'

    @staticmethod
    def ssgsea(exp_df, features_l):
        exp_df_r = pandas2ri.py2rpy(exp_df)

        r_character_vector = robjects.StrVector(features_l)
        r_list = robjects.vectors.ListVector({"feature": r_character_vector})

        robjects.globalenv['exp_df_r'] = exp_df_r
        robjects.globalenv['features_r'] = r_list

        r_script = """
        library(GSVA)
        set.seed(2023)
        exp_df <- as.matrix(exp_df_r)
        result <- gsva(exp_df, features_r, verbose = FALSE, method = "ssgsea")
        result
        """
        robjects.r(r_script)
        ssgsea_nd = robjects.globalenv['result']
        ssgsea_pd = pd.DataFrame(ssgsea_nd.flatten(), index=exp_df.columns, columns=['ssGSEA'])

        return ssgsea_pd

    @staticmethod
    def fgsea(pathways_dic, stats_sr):
        stats_sr_r = robjects.FloatVector(stats_sr.values)
        stats_sr_r.names = robjects.StrVector(stats_sr.index)

        pathways_r = {}
        for key, value in pathways_dic.items():
            pathways_r[key] = robjects.StrVector(value)
        pathways_r = robjects.ListVector(pathways_r)

        robjects.globalenv['pathways_r'] = pathways_r
        robjects.globalenv['stats_r'] = stats_sr_r

        r_script = """
            library(fgsea)
            set.seed(2023)
            fgseaRes <- fgsea(pathways = pathways_r,
                              stats = stats_r,
                              eps = 0.0,
                              minSize = 0,
                              maxSize = 500)
            fgseaRes$leadingEdge <- as.character(fgseaRes$leadingEdge) 
            fgseaRes
            """
        fgsea_result = robjects.r(r_script)
        fgsea_pd = pd.DataFrame(dict(fgsea_result))

        return fgsea_pd

    @staticmethod
    def create_design_table(upset_df, in_group, group_col='Subtype'):
        """
            Creates the design table for the Limma_de function
        """
        df = pd.DataFrame(0, index=upset_df.index, columns=['In', 'Out'])
        df.loc[upset_df[group_col] == in_group, 'In'] = 1
        df.loc[upset_df[group_col] != in_group, 'Out'] = 1
        return df

    @staticmethod
    def create_group_df(upset_df, col):
        """
            Creates the groups table for the filter_by_exp function
        """
        unique_values = upset_df[col].unique()
        df = pd.DataFrame(0, index=upset_df.index, columns=['group'])
        for i in range(len(unique_values)):
            df.loc[upset_df[col] == unique_values[i], 'group'] = i+1
        return df

    @staticmethod
    def filter_by_exp(expr_df, groups_df):

        expr_df_r = pandas2ri.py2rpy(expr_df)
        groups_df_r = pandas2ri.py2rpy(groups_df)

        robjects.globalenv['expr_tb'] = expr_df_r
        robjects.globalenv['groups'] = groups_df_r

        robjects.r.source(MyRFunctions.r_script_path)

        kept_indices = robjects.globalenv['filter_by_exp'](robjects.globalenv['expr_tb'], robjects.globalenv['groups'])
        kept_indices_py = [idx - 1 for idx in kept_indices]
        kept_genes = expr_df.index[kept_indices_py]

        return expr_df.loc[kept_genes,:]

    @staticmethod
    def limma_de(data_df, design_df, plot_path, exp_filter):

        data_df_r = pandas2ri.py2rpy(data_df)
        design_df_r = pandas2ri.py2rpy(design_df)

        robjects.globalenv['data_tb'] = data_df_r
        robjects.globalenv['design_tb'] = design_df_r

        robjects.r.source(MyRFunctions.r_script_path)

        top_table = robjects.globalenv['limma_df'](robjects.globalenv['data_tb'], robjects.globalenv['design_tb'],
                                                   plot_path, exp_filter)

        top_table_df = pd.DataFrame(top_table, index=['logFC','AveExpr', 't','P.Value', 'adj.P.Val','B','Gene']).T

        return top_table_df

    @staticmethod
    def limma_proteins_de(data_df, design_df):

        data_df_r = pandas2ri.py2rpy(data_df)
        design_df_r = pandas2ri.py2rpy(design_df)

        robjects.globalenv['data_tb'] = data_df_r
        robjects.globalenv['design_tb'] = design_df_r

        robjects.r.source(MyRFunctions.r_script_path)

        top_table = robjects.globalenv['limma_proteins_df'](robjects.globalenv['data_tb'], robjects.globalenv['design_tb'])

        top_table_df = pd.DataFrame(top_table, index=['logFC','AveExpr', 't','P.Value', 'adj.P.Val','B','Gene']).T

        return top_table_df

    @staticmethod
    def edger_de(data_df, design_df, plot_path):

        data_df_r = pandas2ri.py2rpy(data_df)
        design_df_r = pandas2ri.py2rpy(design_df)

        robjects.globalenv['data_tb'] = data_df_r
        robjects.globalenv['design_tb'] = design_df_r

        robjects.r.source(MyRFunctions.r_script_path)

        top_table = robjects.globalenv['edger_df'](robjects.globalenv['data_tb'], robjects.globalenv['design_tb'],
                                                   plot_path)

        top_table_df = pd.DataFrame(top_table, index=['logFC', 'logCPM', 'LR', 'PValue', 'FDR', 'Gene']).T

        return top_table_df

    @staticmethod
    def edger_ql_de(data_df, design_df, plot_path):

        data_df_r = pandas2ri.py2rpy(data_df)
        design_df_r = pandas2ri.py2rpy(design_df)

        robjects.globalenv['data_tb'] = data_df_r
        robjects.globalenv['design_tb'] = design_df_r

        robjects.r.source(MyRFunctions.r_script_path)

        top_table = robjects.globalenv['edger_ql_df'](robjects.globalenv['data_tb'], robjects.globalenv['design_tb'],
                                                   plot_path)

        top_table_df = pd.DataFrame(top_table, index=['logFC', 'logCPM', 'LR', 'PValue', 'FDR', 'Gene']).T

        return top_table_df

    @staticmethod
    def dorothea_transcription_factors(expr_df):

        expr_df_r = pandas2ri.py2rpy(expr_df)

        robjects.globalenv['expr_tb'] = expr_df_r

        robjects.r.source(MyRFunctions.r_script_path)

        res_table = robjects.globalenv['dorothea_TF'](robjects.globalenv['expr_tb'])

        res_table_df = pd.DataFrame(res_table, index=['statistic', 'source', 'condition', 'score', 'p_value']).T
        res_table_df['score'] = res_table_df['score'].astype('float64')
        res_table_df['p_value'] = res_table_df['p_value'].astype('float64')
        return res_table_df

    @staticmethod
    def vst_normalization(expr_df, coldata_df):

        expr_df_r = pandas2ri.py2rpy(expr_df)
        coldata_df_r = pandas2ri.py2rpy(coldata_df)

        robjects.globalenv['expr_tb'] = expr_df_r
        robjects.globalenv['coldata_tb'] = coldata_df_r

        robjects.r.source(MyRFunctions.r_script_path)

        res_table = robjects.globalenv['vst_normalization'](robjects.globalenv['expr_tb'], robjects.globalenv['coldata_tb'])

        res_table_df = pd.DataFrame(res_table, index=expr_df.index, columns=expr_df.columns).T
        return res_table_df

    @staticmethod
    def survival_analysis(surv_df, strata, group, cur_title, legend_lbls, colors_elms, plot_path):

        strata_gp = strata.groupby(group)
        for i, df in strata_gp:
            print(i, df.shape[0])
        print('------')

        surv_tb_r = pandas2ri.py2rpy(surv_df)
        strata_r = pandas2ri.py2rpy(strata)
        legend_lbls_r = robjects.vectors.StrVector(legend_lbls)
        colors_elms_r = robjects.vectors.StrVector(colors_elms)

        robjects.globalenv['surv_tb'] = surv_tb_r
        robjects.globalenv['strata'] = strata_r
        robjects.globalenv['cur_title'] = cur_title
        robjects.globalenv['legend_lbls'] = legend_lbls_r
        robjects.globalenv['colors_elms'] = colors_elms_r
        robjects.globalenv['plot_path'] = plot_path

        robjects.r.source(MyRFunctions.r_script_path)

        robjects.globalenv['plot_kaplan_meier_stratify'](robjects.globalenv['surv_tb'], robjects.globalenv['strata'],
                                                         robjects.globalenv['cur_title'], robjects.globalenv['legend_lbls'],
                                                         robjects.globalenv['colors_elms'], robjects.globalenv['plot_path'])

    @staticmethod
    def coxph_multivariate(surv_df, features_df, log_hr=True):
        surv_tb_r = pandas2ri.py2rpy(surv_df)
        features_tb_r = pandas2ri.py2rpy(features_df)

        robjects.globalenv['surv_tb'] = surv_tb_r
        robjects.globalenv['features_tb'] = features_tb_r

        robjects.r.source(MyRFunctions.r_script_path)
        res_table = robjects.globalenv['coxph_multivariate'](robjects.globalenv['surv_tb'], robjects.globalenv['features_tb'])
        res_table_df = pd.DataFrame(res_table).T
        res_table_df.columns = ["Variable", "coef", "exp(coef)", "se(coef)", "z", "Pr(>|z|)", "lower .95", "upper .95"]


        if log_hr:
            res_table_df = res_table_df.dropna(axis=0)
            res_table_df = res_table_df[ ~(((abs(res_table_df['lower .95']) > 100) | (abs(res_table_df['upper .95']) > 100)) &
                  (res_table_df['Pr(>|z|)'] > 0.1))].copy()  # drop ns cases with large CI
            res_table_df['log_lower .95'] = res_table_df['lower .95'].apply(math.log)
            res_table_df['log_upper .95'] = res_table_df['upper .95'].apply(math.log)

        res_table_df['q-value'] = fdrcorrection(res_table_df['Pr(>|z|)'].values)[1]
        res_table_df = res_table_df.sort_values(by=['q-value'])
        res_table_df.reset_index(drop=True, inplace=True)
        return res_table_df
    # @staticmethod
    # def coxph_multivariate(surv_df, features_df):
    #     surv_tb_r = pandas2ri.py2rpy(surv_df)
    #     features_tb_r = pandas2ri.py2rpy(features_df)
    #
    #     robjects.globalenv['surv_tb'] = surv_tb_r
    #     robjects.globalenv['features_tb'] = features_tb_r
    #
    #     robjects.r.source(MyRFunctions.r_script_path)
    #     res_table = robjects.globalenv['coxph_multivariate'](robjects.globalenv['surv_tb'], robjects.globalenv['features_tb'])
    #     res_table_df = pd.DataFrame(res_table).T
    #     res_table_df.columns = ["Variable", "coef", "exp(coef)", "se(coef)", "z", "Pr(>|z|)", "lower .95", "upper .95"]
    #
    #     res_table_df = res_table_df.dropna(axis=0)
    #     res_table_df = res_table_df[~(((abs(res_table_df['lower .95']) > 100) | (abs(res_table_df['upper .95']) > 100)) & (
    #                 res_table_df['Pr(>|z|)'] > 0.1))].copy()  # drop ns cases with large CI
    #     res_table_df['log_lower .95'] = res_table_df['lower .95'].apply(math.log)
    #     res_table_df['log_upper .95'] = res_table_df['upper .95'].apply(math.log)
    #
    #     return res_table_df

    @staticmethod
    def find_collinearity(X_df, y_df):
        y_tb_r = pandas2ri.py2rpy(y_df)
        X_tb_r = pandas2ri.py2rpy(X_df)

        robjects.globalenv['y_tb'] = y_tb_r
        robjects.globalenv['X_tb'] = X_tb_r

        robjects.r.source(MyRFunctions.r_script_path)

        robjects.globalenv['find_collinearity'](robjects.globalenv['X_tb'], robjects.globalenv['y_tb'])
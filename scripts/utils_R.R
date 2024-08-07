set.seed(2023)
library(devtools)
library(tidyverse)
library(data.table)


filter_by_exp <- function(expr_tb, groups) {
  library(edgeR)
  dge <- DGEList(expr_tb, genes=row.names(expr_tb), group=factor(groups$group))
  keep_genes = edgeR::filterByExpr(dge)
  # dge = dge[keep_genes,,keep.lib.sizes=FALSE]
  print(paste0('Number of genes:', sum(keep_genes)))
  return(which(keep_genes))
}

limma_df <- function(data_tb, design_tb, plot_path, exp_filter=TRUE) {
  library(edgeR)
  dge <- DGEList(data_tb)#, group=factor(design_tb$In)
  if(exp_filter){
    keep_genes = edgeR::filterByExpr(dge)
    dge = dge[keep_genes,,keep.lib.sizes=FALSE]
    print(paste0('Number of genes:', sum(keep_genes)))
  }
  dge <- calcNormFactors(dge, method = "TMM")

  jpeg(plot_path)
  v <- voom(dge, design_tb, plot = TRUE)
  # v = voom(dge, design_tb, normalize.method="scale", plot = TRUE)
  dev.off()

  fit <- lmFit(v, design_tb)
  fit <- eBayes(fit, robust = TRUE)

  contr <- makeContrasts(In_Out="In - Out",
                         levels = c("In", "Out"))
  fit2 <- contrasts.fit(fit, contr)
  fit2 <- eBayes(fit2)

  top_table <- topTable(fit2, adjust = "BH", number = Inf)
  top_table$Gene <- rownames(top_table)

  return(top_table)
}

limma_proteins_df <- function(data_tb, design_tb) {
 library(edgeR)
  fit <- lmFit(data_tb, design = design_tb, na.action = 'na.exclude')
  fit <- eBayes(fit, robust = TRUE)

  contr <- makeContrasts(In_Out="In - Out",
                         levels = c("In", "Out"))
  fit2 <- contrasts.fit(fit, contr)
  fit2 <- eBayes(fit2)

  top_table <- topTable(fit2, adjust = "BH", number = Inf)
  top_table$Gene <- rownames(top_table)

  return(top_table)
}

edger_df <- function(data_tb, design_tb, plot_path) {
  library(edgeR)
  dge <- DGEList(counts = data_tb)#, group=factor(design_tb$In)
  dge <- calcNormFactors(dge, method = "TMM")
  keep_genes = filterByExpr(dge)
  dge = dge[keep_genes,,keep.lib.sizes=FALSE]
  print(paste0('Number of genes:', sum(keep_genes)))
  jpeg(plot_path)
  dge <- estimateDisp(dge, design_tb)
  plotBCV(dge)
  dev.off()

  fit <- glmFit(dge, design_tb)
  contr <- makeContrasts(In-Out, levels=design_tb)
  lrt <- glmLRT(fit, contrast = contr)
  res <- as.data.frame(lrt$table)
  res$FDR <- p.adjust(res$PValue, method="BH")
  res$Gene <- row.names(res)
  return(res)
}

edger_ql_df <- function(data_tb, design_tb, plot_path) {
  library(edgeR)
  dge <- DGEList(counts = data_tb)
  dge <- calcNormFactors(dge, method = "TMM")
  keep_genes = filterByExpr(dge)
  dge = dge[keep_genes,,keep.lib.sizes=FALSE]
  jpeg(plot_path)
  dge <- estimateDisp(dge, design_tb)
  plotBCV(dge)
  dev.off()

  fit <- glmQLFit(dge, design_tb)
  contr <- makeContrasts(In-Out, levels=design_tb)
  lrt <- glmQLFTest(fit, contrast = contr)
  # lrt <- glmTreat(fit, contrast = contr)
  res <- as.data.frame(lrt$table)
  res$FDR <- p.adjust(res$PValue, method="BH")
  res$Gene <- row.names(res)
  return(res)
}

dorothea_TF <- function(expr_tb) {
  library(dorothea)
  library(decoupleR)
  ## Load TF activity database (human version)
  data(dorothea_hs, package = "dorothea")

  #dorothea_hs_confid <- dorothea_hs %>%
  #  filter(confidence == "A" | confidence == "B" |confidence == "C" )
  dorothea_hs_confid <- dorothea_hs %>% filter(confidence != "E")
  # dorothea_hs_confid <- subset(dorothea_hs, confidence != 'E')

  ## A matrix of normalized enrichment scores for each TF across all samples
  cl_expr_rowSums <- rowSums(expr_tb)

  tf_activity <- run_viper(expr_tb[which(cl_expr_rowSums >= 45),], dorothea_hs_confid,
                           .source='tf', .target='target', .mor='mor', minsize = 0)
  return(tf_activity)
}

vst_normalization <- function(expr_df, coldata_tb) {
  ## The input expr_df should contain integer TPM values with
  ## gene on rows and samples on columns
  library(DESeq2)
  expr_df = round(expr_df)
are_numeric_values <- sapply(expr_df, is.numeric)

# Check if all values in each column are numeric
all_numeric <- all(are_numeric_values)

# Print the result
if (all_numeric) {
  print("All values in the DataFrame are numeric.")
} else {
  print("There are non-numeric values in the DataFrame.")
}
  dds <- DESeqDataSetFromMatrix(countData = expr_df,
                              colData = coldata_tb, #Rows of colData correspond to columns of countData
                              design = ~ 1)

  # Perform a vst transformation on count data
  rld <- vst(dds, blind=TRUE)
  bc <- assay(rld)
  return(bc)
}

run_fgsea <- function(pathways_l, stats_l){
  library(fgsea)

  fgseaRes <- fgsea(pathways = pathways_l,
                    stats = stats_l,
                    eps = 0.0,
                    minSize = 0,
                    maxSize = 500)
  # print(fgseaRes)
  fgseaRes
}

plot_kaplan_meier_stratify <- function(surv_tb, strata, cur_title, legend_lbls, colors_elms,
                                        plot_path, conf_int=FALSE, risk_table="nrisk_cumcensor"){
  library(survival)
  library(survminer)
  library(tidyverse)
  library(pheatmap)
  library(RColorBrewer)
  library(egg)
  colnames(surv_tb) <-c('time', 'status')
  colnames(strata) <-c('strata')
  data_df <- data.frame(strata, surv_tb)
  sv_fit <- survfit(Surv(time, status) ~ unlist(strata), data = data_df)
  plot_survival_curve(sv_fit, data_df, cur_title, legend_lbls, colors_elms, title_font_size=18, risk_table, plot_path, conf_int)
}

plot_survival_curve <- function(sv_fit, data_df, cur_title, legend_lbls, colors_elms, title_font_size=18,
                                risk_table, plot_path, conf_int){

 ggsurv<-ggsurvplot(
   sv_fit,
   data = data_df,
   title = cur_title,
   # alpha = c(0.6),
   # break.time.by = 200,
   # break.x.by = 1,
   conf.int = conf_int,
   # conf.int.style = "step",
   # conf.int.alpha=c(0.1),
   font.tickslab = c(20,  "black"),#"bold",
   font.main = c(title_font_size, "black"),# "bold",
   font.submain = c(20),#, "bold"
   # font.caption = c(14, "plain", "orange"),
   font.x = c(20,  "black"),#"bold",
   font.y = c(20, "black"),

   ggtheme = theme_classic(),
   legend.title = "",
   legend.labs = legend_lbls,
   # ncensor.plot = TRUE,
   palette = colors_elms,
   # palette = "Dark2",
   pval = TRUE,
   # pval.coord = c(0, 0.25),
   risk.table = risk_table,#"abs_pct",
   risk.table.col = "strata",
   risk.table.y.text = TRUE,
   risk.table.legend.labs = rep("", length(unique(data_df$strata))),
   size = 1.5,
   # surv.median.line = "v",
   surv.scale = "percent",
   tables.col = "strata",
   # tables.theme=theme_classic(),
   tables.y.text = FALSE,
   xlab = "Time in Days",
   # xlim = c(0,25),
   ylab = "Survival",
   pval.size = 7,
    # pval.coord = c(.5, .5)
   # ylim = c(0,1)
  )
  ggsurv$plot <- ggsurv$plot +
  theme(legend.text = element_text(size = 20, color = "black"), legend.key.width= unit(2, 'cm'),
      legend.key.height = unit(1, 'cm'))#, face = "bold"
  ggsurv$table <- ggsurv$table +
  theme(plot.title = element_text(size = 14),#, face = "bold"
        axis.text= element_text(size = 13, color = "black"),#, face = "bold"
        text= element_text(size = 13, color = "black")#, face = "bold"
        )+ theme(legend.position = "none")

  combined_plot <- ggarrange(ggsurv$plot, ggsurv$table, ncol = 1, heights = c(3, 1))
  ggsave(plot_path, plot = combined_plot,  width = 11, height = 6)

}

log_rank_test <- function(featurs_m, surv_m, strata) {
  data_df <- data.frame(surv_m, featurs_m)
  data_df <- data.frame(strata, data_df)
  sv_diff <- survdiff(Surv(time, status) ~ strata, data = data_df)
  print(sv_diff)
}

find_collinearity <- function (X_tb, y_tb){
  formula_str <- paste(names(y_tb[1]), " ~ ", paste(names(X_tb), collapse = " + "))
  lm_formula <- as.formula(formula_str)
  data_tb = cbind(y_tb, X_tb)
  model <- lm(lm_formula, data = data_tb)
  print(alias(model))
}

get_coxph_tb <- function(cox_model) {
  scox = summary(cox_model)
  y <- cbind(scox[["coefficients"]], `lower .95` = scox[["conf.int"]][, "lower .95"],
                                      `upper .95` = scox[["conf.int"]][, "upper .95"])

  cbind(Variable = rownames(y), as.data.frame(y))
}

coxph_multivariate <- function (surv_tb, features_tb){
  library(survival)
  library(survminer)
  library(car)

  colnames(surv_tb) <-c('time', 'status')
  surv_tb[,'status'] = as.integer(surv_tb[,'status'])
  formula <- as.formula(paste("Surv(time, status) ~", paste(names(features_tb), collapse = " + ")))
  df = cbind(surv_tb, features_tb)
  cox_model <- coxph(formula, data = df)

  print(paste("AIC of cox model: ", AIC(cox_model)))
  # print(paste("variance inflation factor (VIF) for multicollinearity: ", car::vif(cox_model)))
  test_ph <- cox.zph(cox_model)
  print('Proportional Hazards Assumption: ')
  print(test_ph$table)
  return(get_coxph_tb(cox_model))
}

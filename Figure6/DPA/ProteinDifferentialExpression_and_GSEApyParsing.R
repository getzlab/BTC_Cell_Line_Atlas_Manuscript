##################################
# By: Robert Morris
##################################
#differential expression analysis of each cluster vs all others

library('ggplot2');
library('bswUtil');
library('stringr');
library('reshape2')
library("limma");

topDir<-bswUtil.oDir('/output/directory/path')
resDir<-bswUtil.oDir(paste(topDir, "/", "result", sep=""));

protein.data<-read.delim(file=paste("/path/to/proteomics_collapsed_filtered_mad_normalized.csv", sep=""), sep=",", row.names=1, check.names=FALSE, stringsAsFactors=FALSE);
protein.data<-t(protein.data);
rownames(protein.data)<-paste(rownames(protein.data), "_HUMAN", sep="");

GetNAfreq<-function(x){
        out.l<-list();
        for(i in 1:nrow(x)){
                na.count<-length(which(is.na(x[i,])));
                out.l[[i]]<-c(rownames(x)[i], na.count, ncol(x));
        }
        out.df<-do.call(rbind, out.l);
        out.df<-data.frame(protein=as.character(out.df[,1]), na.count=as.numeric(out.df[,2]), tot.count=as.numeric(out.df[,3]), stringsAsFactors=FALSE);
        out.df$na.freq<-out.df$na.count / out.df$tot.count;
        return(out.df);
}
prot.nafreq<-GetNAfreq(x=protein.data);
#number of rows na.freq <0.5
dim(prot.nafreq[which(prot.nafreq$na.freq <= 0.1),,drop=FALSE]);

prot.data<-protein.data[,which(!(colnames(protein.data) %in% c("JHH1"))),drop=FALSE];
prot.nafreq2<-GetNAfreq(x=prot.data);
dim(prot.nafreq2[which(prot.nafreq2$na.freq <= 0.1),,drop=FALSE]); #5299 rows

#filter out proteins with more than 10% missing values
backgrd.proteins<-prot.nafreq2[which(prot.nafreq2$na.freq <= 0.1),"protein"];

#################################################################################################################
#Run differential expression analysis each cluster vs all others
prot.f1<-protein.data[backgrd.proteins,,drop=FALSE];

prot.cell.clust<-read.delim("/path/to/Proteome_MAD_normalization_Louvain_clustering_fromNegin_2023.csv", sep="\t", stringsAsFactors=FALSE);
prot.cell.clust$ClusterName<-paste("C", prot.cell.clust$ClusterID, sep="");

print(all(prot.cell.clust$ProteinCellLine %in% colnames(prot.f1))) #TRUE
prot.f2<-prot.f1[,as.character(prot.cell.clust$ProteinCellLine),drop=FALSE];

############################################################################################################
#for each cluster run limma differential expression analysis

GetMeans<-function(sel.dat, cond1, cond2){
	cond1.nona<-apply(sel.dat[,as.character(cond1),drop=FALSE], 1, FUN=function(xx){
		return(length(which(!is.na(xx))));
	});
	cond2.nona<-apply(sel.dat[,as.character(cond2),drop=FALSE], 1, FUN=function(xx){
		return(length(which(!is.na(xx))));
	});
	cond1.mean<-rowMeans(sel.dat[,as.character(cond1),drop=FALSE], na.rm=TRUE);
	cond2.mean<-rowMeans(sel.dat[,as.character(cond2),drop=FALSE], na.rm=TRUE);
	res.l<-list();
	res.l[["cond1.mean"]]<-cond1.mean;
	res.l[["cond2.mean"]]<-cond2.mean;
	res.l[["cond1.nona"]]<-cond1.nona;
	res.l[["cond2.nona"]]<-cond2.nona;
	return(res.l);
}

LimmaComp<-function(dat, des, coef){
	fit1.lm <- lmFit(dat, des)
	fit1.lm <- eBayes(fit1.lm)
	fit1.lm_table<-topTable(fit1.lm, number=100000, coef=coef, adjust="BH")
	fit1.lm_table$rnkscore<-sign(fit1.lm_table$logFC) * -log10(fit1.lm_table$P.Value);
	return(fit1.lm_table);
}

merg.hdb<-read.delim("/path/to/merged_2014_2020_protein_database.txt", sep="\t", stringsAsFactors=FALSE);

set.seed(2023)
comp.df<-data.frame(cond1=c("C1", "C2", "C3", "C4", "C5"), stringsAsFactors=FALSE)
for(j in 1:nrow(comp.df)){
print(j);
	cond1.sel<-prot.cell.clust[which(prot.cell.clust$ClusterName==as.character(comp.df[j,"cond1"])),,drop=FALSE];
	cond2.sel<-prot.cell.clust[which(!(prot.cell.clust$ClusterName %in% cond1.sel$ClusterName)),,drop=FALSE];
	cond2.sel$ClusterName<-paste("not", as.character(comp.df[j,"cond1"]), sep="");
	c.df<-rbind(cond1.sel, cond2.sel);
	c.df$ConditionID<-factor(c.df$ClusterName, levels=c(paste("not", as.character(comp.df[j,"cond1"]), sep=""), comp.df[j,"cond1"]));
	mm<-model.matrix(data=c.df, ~ ConditionID);
	colnames(mm)[1]<-"Intercept";
	colnames(mm)[ncol(mm)]<-"Cond";
	rownames(mm)<-as.character(c.df$ProteinCellLine);
	print(rownames(mm))
	print(mm);
	sel.l2om<-prot.f2[,rownames(mm),drop=FALSE];
	lim.res1<-LimmaComp(dat=sel.l2om, des=mm, coef="Cond")
	#add gene and entrez
	lim.means<-GetMeans(sel.l2om, cond1=as.character(cond1.sel$ProteinCellLine), cond2=as.character(cond2.sel$ProteinCellLine));
	lim.res1$Cond1Mean<-as.numeric(lim.means[["cond1.mean"]][as.character(rownames(lim.res1))]);
	lim.res1$Cond2Mean<-as.numeric(lim.means[["cond2.mean"]][as.character(rownames(lim.res1))]);
	hdb.ro1<-merg.hdb[match(as.character(rownames(lim.res1)), as.character(merg.hdb$UniprotProteinName)),,drop=FALSE];
	lim.res1$GeneName<-as.character(hdb.ro1$GeneName);
	lim.res1$EntrezID<-as.character(hdb.ro1$EntrezID);
	lim.res1$ProteinID<-as.character(rownames(lim.res1));
	write.table(lim.res1, file=paste(resDir, "/", comp.df[j,"cond1"], "_vs_not", comp.df[j,"cond1"], "_ProteinL2omMAD_comparisonNew.txt", sep=""), sep="\t", quote=FALSE, row.names=FALSE);
	#get upreg
	up.reg<-lim.res1[which(lim.res1$adj.P.Val < 0.1 & lim.res1$logFC >= 1),,drop=FALSE];
 	dn.reg<-lim.res1[which(lim.res1$adj.P.Val < 0.1 & lim.res1$logFC <= -1),,drop=FALSE];
	x.lab.left<-min(lim.res1$logFC, na.rm=TRUE) + ((max(lim.res1$logFC, na.rm=TRUE) - min(lim.res1$logFC, na.rm=TRUE))*0.1);
	x.lab.right<-max((lim.res1$logFC), na.rm=TRUE) - ((max(lim.res1$logFC, na.rm=TRUE) - min(lim.res1$logFC, na.rm=TRUE))*0.1);
	y.lab.top<-max(-log10(lim.res1$adj.P.Val), na.rm=TRUE) - ((max(-log10(lim.res1$adj.P.Val), na.rm=TRUE) - min(-log10(lim.res1$adj.P.Val), na.rm=TRUE))*0.1);
	g1<-ggplot(data=lim.res1, aes(x=logFC, y=-log10(adj.P.Val))) + geom_point() + geom_vline(xintercept=c(-1,1), linetype="dashed", col="blue") + 
	geom_hline(yintercept=1, linetype="dashed", col="red") + annotate(geom="text", x=x.lab.left, y=y.lab.top, label=nrow(dn.reg)) + 
	annotate(geom="text", x=x.lab.right, y=y.lab.top, label=nrow(up.reg)) + ggtitle(paste(comp.df[j,"cond1"], "_vs_not", comp.df[j,"cond1"],sep="")) + theme_bw() + theme(text = element_text(size=10));
	ggsave(g1, file=paste(resDir, "/", comp.df[j,"cond1"], "_vs_not", comp.df[j,"cond1"], "_ProteinL2omMAD_volcanoplot.jpeg", sep=""));
}

#create expressed background proteins
hdb.ro<-merg.hdb[match(as.character(backgrd.proteins), as.character(merg.hdb$UniprotProteinName)),,drop=FALSE];
expr.protein.genenames<-na.omit(unique(hdb.ro$GeneName));
write.table(expr.protein.genenames, file=paste(resDir, "/", "QuantifiedProteinsGeneNames.txt", sep=""), sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE);

####################################################################
#Running gseapy in python script
#run the GSEApy script on each differential expression results tables
#GSEApy_proteinClusterDE.ipynb
####################################################################
#read and parse gseapy results
c2resfiles<-list.files("/path/to/GSEApy_proteinClusterDE1/result", pattern="c2.cp.v2023.1gseapy_results.tsv", full.names=TRUE);
c5resfiles<-list.files("/path/to/GSEApy_proteinClusterDE1/result", pattern="c5.go.bp.v2023.1gseapy_results.tsv", full.names=TRUE);
hmresfiles<-list.files("/path/to/GSEApy_proteinClusterDE1/result", pattern="h.all.v2023.1gseapy_results.tsv", full.names=TRUE);
ppihubfiles<-list.files("/media/willi/OricoStorage/rm421Backup/Project1/Nabeel/ICCL/iccl029v2/GSEApy_proteinClusterDE2/result", pattern="gseapy_results.tsv", full.names=TRUE);
difffiles<-list.files("/path/to/GSEApy_proteinClusterDE1/result", pattern="_diffExpr_table.tsv", full.names=TRUE);

c2res_strm<-str_match(c2resfiles, pattern=".*/result/(.*_vs_.*)_FDR(.*)_(.*)__(.*)gseapy_results.tsv");
c5res_strm<-str_match(c5resfiles, pattern=".*/result/(.*_vs_.*)_FDR(.*)_(.*)__(.*)gseapy_results.tsv");
hmres_strm<-str_match(hmresfiles, pattern=".*/result/(.*_vs_.*)_FDR(.*)_(.*)__(.*)gseapy_results.tsv");
diff_strm<-str_match(difffiles, pattern=".*/result/(.*_vs_.*)_FDR(.*)_(.*)_diffExpr_table.tsv");


ChangeToProteinName<-function(genes, prots){
	gene.sp<-unlist(strsplit(genes, split=";"));
	gene.sp<-na.omit(gene.sp);
	prot.ids<-c();
	for(i in 1:length(gene.sp)){
		sel.prots<-prots[which(prots$GeneName %in% gene.sp[i]),,drop=FALSE];
		prot.id<-na.omit(sel.prots$ProteinID);
		prot.ids[i]<-paste0(prot.id, collapse=";");
	}
	prot.ids<-paste0(prot.ids, collapse=";");
	return(prot.ids);
}

###########Annotate gene names with appropriate protein IDs
AnnotateGSEAresults1<-function(gsea_strm, de_strm){
	protein.id<-c();
	for(i in 1:nrow(gsea_strm)){
		protein.id<-c();
		dt<-read.delim(gsea_strm[i,1], sep="\t", stringsAsFactors=FALSE, row.names=1);
		#get appropriate diff expr
		de.sel<-de_strm[which(de_strm[,2] == gsea_strm[i,2] & de_strm[,3]==gsea_strm[i,3] & de_strm[,4]==gsea_strm[i,4]),,drop=FALSE];
		de<-read.delim(de.sel[1,1], sep="\t", stringsAsFactors=FALSE);
		#collapse all proteins by gene name
		for(j in 1:nrow(dt)){
			protein.id[j]<-ChangeToProteinName(genes=as.character(dt[j,"Genes"]), prots=de);
		}
		dt$ProteinID<-as.character(protein.id);
		write.table(dt, file=paste(resDir, "/", gsea_strm[i,2], "_FDR", gsea_strm[i,3], "_", gsea_strm[i,4], "__", gsea_strm[i,5], "gseapy_Reannotated_results.tsv", sep=""), sep="\t", quote=FALSE, row.names=FALSE);
	}
	return(1)
}

r1<-AnnotateGSEAresults1(gsea_strm=c2res_strm, de_strm=diff_strm);
r2<-AnnotateGSEAresults1(gsea_strm=c5res_strm, de_strm=diff_strm);
r3<-AnnotateGSEAresults1(gsea_strm=hmres_strm, de_strm=diff_strm);
f1<-AnnotateGSEAresults1(gsea_strm=ppi_strm, de_strm=diff_strm);

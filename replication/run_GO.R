#BiocManager::install('clusterProfiler')
library(clusterProfiler)
options(connectionObserver = NULL)
#BiocManager::install('org.Hs.eg.db')
library(org.Hs.eg.db)
library(ggplot2)
#BiocManager::install('DOSE')
library(DOSE)
rm(list=ls())

args = commandArgs(trailingOnly=TRUE)

input_file=args[1]
output_dir=args[2]
output_prefix=args[3]

#input_file='results/enrichment/HSC/gene_list.txt'
#output_dir='results/enrichment/HSC'
#output_prefix='HSC'

degs_df <- read.csv(input_file, stringsAsFactors=F,header=FALSE)
genes <- bitr(degs_df$V1, fromType = 'ENSEMBL', toType = c('SYMBOL','ENTREZID'),  OrgDb=org.Hs.eg.db)

#GO
go <- enrichGO(gene = genes$ENSEMBL, OrgDb = "org.Hs.eg.db", ont="all", keyType = 'ENSEMBL')
go <- simplify(go)
#dotplot(go, split="ONTOLOGY",showCategory=5,font.size=8) +facet_grid(~ONTOLOGY, scale="free")
write.table(go, paste0(output_dir,'/',output_prefix,"_GO.txt"), quote=F, sep='\t')








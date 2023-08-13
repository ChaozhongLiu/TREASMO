#BiocManager::install('clusterProfiler')
library(ggplot2)
rm(list=ls())
library(clusterProfiler)
options(connectionObserver = NULL)
library(org.Hs.eg.db)
library(ggplot2)



# HSPC all markers GO plot ------------
celltypes <- c("MoP","NeuP","MasP","EryP","MkP")

go_terms <- c()
df_all <- data.frame(matrix(nrow = 0, ncol = 13))
for (i in 1:5) {
  df <- read.csv(paste0("results/enrichment/",celltypes[i],"/",celltypes[i],"_GO.txt"),
                 sep='\t', stringsAsFactors = F)
  go_terms <- c(go_terms, df$Description[1:5])
  
  df$Count <- as.numeric(vapply(strsplit(df$GeneRatio,"/"), `[`, 1, FUN.VALUE=character(1)))
  df$Total <- as.numeric(vapply(strsplit(df$GeneRatio,"/"), `[`, 2, FUN.VALUE=character(1)))
  df$Ratio <- df$Count / df$Total
  df$CellType <- celltypes[i]
  
  df_all <- rbind(df_all, df)
}

go_levels = go_terms[!duplicated(go_terms)]

df_plt <- df_all[df_all$Description %in% go_terms,]

text_wrapper <- function(texts, width){
  texts <- strwrap(texts, width=width)
  texts <- paste(texts, collapse = '\n')
  return(texts)
}
df_plt$Description <- lapply(df_plt$Description, text_wrapper, width = 40)
go_levels <- lapply(go_levels, text_wrapper, width = 40)
df_plt$Description <- factor(df_plt$Description, levels = go_levels)

df_plt$CellType <- factor(df_plt$CellType, levels = celltypes)


pdf('HSPC_markers_GO.pdf',width=5.5, height = 8)
ggplot(df_plt, aes_string(x="CellType", y="Description")) +
  geom_point(aes(color=Ratio, fill=Ratio, size=Count), shape=21,color='black') +
  scale_fill_continuous(low="white", high="#eb564f", limits = c(0, 0.15), name = 'Gene.Ratio', guide=guide_colorbar(reverse=F))+
  ylab(NULL) + ggtitle(NULL) + theme_classic() +
  scale_size_continuous(name='Counts',range=c(1,6)) +
  theme(axis.text = element_text(angle = 0, colour = 'black'))
dev.off()

# EryP temporal modules GO plot ------------

Df0 <- read.csv('results/enrichment/HSC_5_7_EryP/out_0/mod0_GO.txt',
                sep='\t', stringsAsFactors = F)
Df0$Count <- as.numeric(vapply(strsplit(Df0$GeneRatio,"/"), `[`, 1, FUN.VALUE=character(1)))
Df0$Total <- as.numeric(vapply(strsplit(Df0$GeneRatio,"/"), `[`, 2, FUN.VALUE=character(1)))
Df0$Ratio <- Df0$Count / Df0$Total

Df0 <- rbind(Df0[Df0$ONTOLOGY=='BP',][1:5,], 
             Df0[Df0$ONTOLOGY=='CC',][1:5,], 
             Df0[Df0$ONTOLOGY=='MF',][1:5,])

text_wrapper <- function(texts, width){
  texts <- strwrap(texts, width=width)
  texts <- paste(texts, collapse = '\n')
  return(texts)
}
Df0$Description <- lapply(Df0$Description, text_wrapper, width = 40)
Df0$Description <- factor(Df0$Description, levels = rev(Df0$Description))


pdf('EryP_mod0_GO.pdf',width=9, height = 5)
ggplot(Df0, aes_string(x="Ratio", y="Description")) +
  geom_point(aes(color=qvalue, fill=qvalue, size=Count), shape=21,color='black') +
  scale_fill_continuous(high="white", low="#eb564f", limit=c(0,0.013), name = 'q-value', guide=guide_colorbar(reverse=T))+
  ylab(NULL) + ggtitle(NULL) + theme_classic() +
  scale_size_continuous(name='Counts',range=c(2,6)) +
  facet_grid(~ONTOLOGY,scales = "free")+
  theme(axis.text = element_text(angle = 0, colour = 'black'))
dev.off()


Df1 <- read.csv('results/enrichment/HSC_5_7_EryP/out_1/mod1_GO.txt',
                sep='\t', stringsAsFactors = F)
Df1$Count <- as.numeric(vapply(strsplit(Df1$GeneRatio,"/"), `[`, 1, FUN.VALUE=character(1)))
Df1$Total <- as.numeric(vapply(strsplit(Df1$GeneRatio,"/"), `[`, 2, FUN.VALUE=character(1)))
Df1$Ratio <- Df1$Count / Df1$Total

Df1 <- Df1[Df1$ONTOLOGY=='BP',][1:10,]

Df1$Description <- lapply(Df1$Description, text_wrapper, width = 40)
Df1$Description <- factor(Df1$Description, levels = rev(Df1$Description))


pdf('EryP_mod1_GO.pdf',width=5, height = 5)
ggplot(Df1, aes_string(x="Ratio", y="Description")) +
  geom_point(aes(color=qvalue, fill=qvalue, size=Count), shape=21,color='black') +
  scale_fill_continuous(high="white", low="#eb564f", limit=c(0,0.01), name = 'q-value', guide=guide_colorbar(reverse=T))+
  ylab(NULL) + ggtitle(NULL) + theme_classic() +
  scale_size_continuous(name='Counts',range=c(3,4)) +
  facet_grid(~ONTOLOGY,scales = "free")+
  theme(axis.text = element_text(angle = 0, colour = 'black'))
dev.off()




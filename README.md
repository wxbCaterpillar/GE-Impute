# GE-Impute
GE-Impute is an imputation method for scRNA-seq data based on graph embeddings.

## Installation
setup

## Tutorial

from CellImputation import GEImpute
import pandas as pd

rawfile=pd.read_csv("/home/wuxiaobin/imputation/data/Result1_PearsonCor/mask0_seed0_10x.txt",sep="\t",index_col=0)

#step 1 build adj matrix
graph_adj=GEImpute.GraphBuild(rawfile)

#step 2 cell embeddings
cell_emb=GEImpute.trainCellEmbeddings(graph_adj)

#step3 scRNA-seq data imputation
data_imp=GEImpute.imputation(scfile=rawfile,embeddingfile=cell_emb,AdjGraph=graph_adj)


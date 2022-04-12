# GE-Impute

GE-Impute is an imputation method for scRNA-seq data based on graph embeddings.

Overall workflow of GE-Impute algorithm pipeline

<img width="415" alt="image" src="https://user-images.githubusercontent.com/44340233/162195520-e8b84b86-0efe-4fb2-b6ac-786c8a04496f.png">

GE-Impute constructs a raw cell-cell similarity network based on Euclidean distance. For each cell, it simulates a random walk of fixed length using BFS and DFS strategy. Next, graph embedding-based neural network model was employed to train the embedding matrix for each cell based on sampling walks. The similarity among cells could be re-calculated from embedding matrix to predict new link-neighbors for the cells and reconstruct cell-cell similarity network. Finally, GE-Impute imputes the dropout zeros for each cell by averaging the expression value of its neighbors in reconstructed similarity network.

## Installation

#change to SCIMP directory
```
python setup.py install
```

## Tutorial

```
import pandas as pd

import SCIMP.Impute

rawfile=pd.read_csv("/home/wuxiaobin/imputation/data/Result1_PearsonCor/scfile_10x.txt",sep="\t",index_col=0)

#Step1 build adj matrix
graph_adj=Impute.GraphBuild(rawfile)

#Step2 cell embeddings
cell_emb=Impute.trainCellEmbeddings(graph_adj)

#Step3 scRNA-seq data imputation
data_imp=Impute.imputation(scfile=rawfile,embeddingfile=cell_emb,AdjGraph=graph_adj)
```

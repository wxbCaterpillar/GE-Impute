# GE-Impute

GE-Impute is an imputation method for scRNA-seq data based on graph embeddings.

Overall workflow of GE-Impute algorithm pipeline

<img width="415" alt="image" src="https://user-images.githubusercontent.com/44340233/162195520-e8b84b86-0efe-4fb2-b6ac-786c8a04496f.png">

GE-Impute constructs a cell-cell similarity network based on Euclidean distance. For each cell, it simulates a random walk of fixed length using BFS and DFS fashion. Next, skip-gram model was employed to train the embedding matrix for each cell basing on sampling walks. New links were predicted to reconstruct the similarity network by calculating the distance of cells’ embeddings in the latent space. Finally, GE-Impute imputes the drop-out events for each cell by averaging the expression value of all neighbors in network.

## Installation

#change to SCIMP directory
```
python setup.py install
```

## Tutorial

```
import pandas as pd

from SCIMP import Impute 

rawfile=pd.read_csv("/home/wuxiaobin/imputation/data/Result1_PearsonCor/mask0_seed0_10x.txt",sep="\t",index_col=0)

#Step1 build adj matrix
graph_adj=GEImpute.GraphBuild(rawfile)

#Step2 cell embeddings
cell_emb=GEImpute.trainCellEmbeddings(graph_adj)

#Step3 scRNA-seq data imputation
data_imp=GEImpute.imputation(scfile=rawfile,embeddingfile=cell_emb,AdjGraph=graph_adj)
```

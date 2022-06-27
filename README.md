# GE-Impute

GE-Impute is an imputation method for scRNA-seq data based on graph embeddings.

Overall workflow of GE-Impute algorithm pipeline

https://raw.githubusercontent.com/wxbCaterpillar/GE-Impute/main/figure1.tif

GE-Impute constructs a raw cell-cell similarity network based on Euclidean distance. For each cell, it simulates a random walk of fixed length using BFS and DFS strategy. Next, graph embedding-based neural network model was employed to train the embedding matrix for each cell based on sampling walks. The similarity among cells could be re-calculated from embedding matrix to predict new link-neighbors for the cells and reconstruct cell-cell similarity network. Finally, GE-Impute imputes the dropout zeros for each cell by averaging the expression value of its neighbors in reconstructed similarity network.

## Installation

#change to SCIMP directory
```
python setup.py install
```
## Requirements
- python = 3.8
- sklearn
- networkx
- gensim
- collections
- joblib


## Tutorial

```
import pandas as pd

import SCIMP.Impute

#Step1 load the raw count matrix of scRNA-seq data, where rows are genes and columns are cells.
rawfile=pd.read_csv("input_file.txt",sep="\t",index_col=0)

#Step2 build adjacent matrix for scRNA-seq data.
graph_adj=Impute.GraphBuild(rawfile)

#Step2 learn cell embeddings.
cell_emb=Impute.trainCellEmbeddings(graph_adj)

#Step3 scRNA-seq data imputation, the format of output file is genes x cells expression matrix.
data_imp=Impute.imputation(scfile=rawfile,embeddingfile=cell_emb,AdjGraph=graph_adj)
```

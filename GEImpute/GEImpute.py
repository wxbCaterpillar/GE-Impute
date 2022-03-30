import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import pairwise_distances
from .CellEmbeddings import Node2Vec
import networkx as nx


def _normalize_matrix(X):
    num_transcripts = np.sum(X, axis=0)
    X_norm = (np.median(num_transcripts) / num_transcripts) * X
    tmatrix=np.sqrt(X_norm) + np.sqrt(X_norm+1)
    return tmatrix


def _calculate_distance(X, num_jobs=5):
    D = pairwise_distances(X.T, n_jobs=num_jobs, metric='euclidean')
    return D

def GraphBuild(input_file):
    nor_file=_normalize_matrix(input_file)
    distance_matrix=_calculate_distance(nor_file)
    adj_matrix=np.zeros((distance_matrix.shape[0],distance_matrix.shape[0]))
    for i in range(distance_matrix.shape[0]):
        adj_matrix[np.argsort(distance_matrix[:,i])[1:6],i]=1
        adj_matrix[i,np.argsort(distance_matrix[:,i])[1:6]]=1
    adj_df=pd.DataFrame(adj_matrix)
    adj_df.index=input_file.columns.tolist()
    adj_df.columns=input_file.columns.tolist()
    return adj_df

def trainCellEmbeddings(Graph):
    G = nx.from_pandas_adjacency(Graph)
    Cell_node2vec = Node2Vec(G,dimensions=128,p=0.25,q=4, walk_length=10, num_walks=60, workers=1,seed=1)
    tmodel=Cell_node2vec.fit(window=3, epochs=3,seed=1)
    emb=Cell_node2vec.get_embeddings(tmodel)
    return emb

def imputation(scfile,embeddingfile,AdjGraph):
    prediction_distance=pairwise_distances(embeddingfile.T,n_jobs=5,metric="euclidean")
    adj_matrix_merge=np.array(AdjGraph)
    for j in range(adj_matrix_merge.shape[0]):
        adj_matrix_merge[np.argsort(prediction_distance[:,j])[0:np.sum(adj_matrix_merge[:,j]==1)],j]=1
    np_scfile=np.array(scfile)
    imputation_file=np.array(scfile)
    for k in range(adj_matrix_merge.shape[1]):
        imputation_file[np.where(imputation_file[:,k]==0)[0],k] = np.mean(np_scfile[:,np.where(adj_matrix_merge[:,k]==1)[0]][np.where(imputation_file[:,k]==0)[0],:],axis=1)
    df_imputation=pd.DataFrame(imputation_file)
    df_imputation.columns=scfile.columns.tolist()
    df_imputation.index=scfile.index.tolist()
    return df_imputation

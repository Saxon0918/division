import numpy as np
import scipy.sparseasss
from corextopic import corextopicasct# Define a matrix where rows are samples (docs) and columns are features (words)X=np.array([[0,0,0,1,1],[1,1,1,0,0],[1,1,1,1,1]],dtype=int)# Sparse matrices are also supportedX=ss.csr_matrix(X)# Word labels for each column can be provided to the modelwords=['dog','cat','fish','apple','orange']# Document labels for each row can be provideddocs=['fruit doc','animal doc','mixed doc']# Train the CorEx topic modeltopic_model=ct.Corex(n_hidden=2)# Define the number of latent (hidden) topics to use.topic_model.fit(X,words=words,docs=docs)





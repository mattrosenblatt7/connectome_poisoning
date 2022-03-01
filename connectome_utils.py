import numpy as np

def mat2edge(mats):
  # convert connectome matrices to vector of edges
  dim = len(np.shape(mats))
  nsub = np.shape(mats)[2]
  nnodes = np.shape(mats)[0]
  nedges = int(nnodes * (nnodes - 1) / 2)
  triu1, triu2 = np.triu_indices(nnodes, 1)

  if dim==3:
    edges = np.zeros((nedges, nsub))
    for sub_idx in range(nsub):
      edges[:, sub_idx] = mats[triu1, triu2, sub_idx]

  elif dim == 4:
    ntasks = np.shape(mats)[3]
    edges = np.zeros((int(nedges*ntasks), nsub))
    for task_idx in range(ntasks):
      start_idx = task_idx * nedges
      end_idx = (task_idx + 1) * nedges
      for sub_idx in range(nsub):
        edges[start_idx:end_idx, sub_idx] = mats[triu1, triu2, sub_idx, task_idx]

  return edges


def edge2mat(edges):
  # convert connectome edges to matrix
  nnodes= int( (1 + np.sqrt(1+4*1*2*np.shape(edges)[0]) ) / 2)  # 268
  triu1, triu2 = np.triu_indices(nnodes, 1)
  mat = np.zeros((nnodes, nnodes))
  mat[triu1, triu2] = edges
  mat = mat + mat.T
  return mat

def train_test_indices(n, perc_train):
  # train/test split, perc_train is % training data
  shuffle_idx = np.random.permutation(np.arange(0, n))
  train_idx = shuffle_idx[0:round(perc_train*n)]
  test_idx = shuffle_idx[round(perc_train*n):]

  return train_idx, test_idx



def all_edges2task(edges, ntasks):
  # convert edges to edges by task
  nedges_total = np.shape(edges)[1]
  nedges_per_task = int(nedges_total/ntasks)
  nsub = np.shape(edges)[0]

  edges_by_task = np.zeros((nsub, nedges_per_task, ntasks))

  for i in range(ntasks):
    start_idx = i*nedges_per_task
    end_idx = (i+1)*nedges_per_task
    edges_by_task[:, :, i] = edges[:, start_idx:end_idx]

  return edges_by_task


def task2all_edges(edges_by_task, ntasks):
  # convert edges by tasks to single vector
  nedges_total = int(np.shape(edges_by_task)[1]*np.shape(edges_by_task)[2])
  nedges_per_task = np.shape(edges_by_task)[1]
  nsub = np.shape(edges_by_task)[0]

  all_edges = np.zeros((nsub, nedges_total))
  for i in range(ntasks):
    start_idx = i*nedges_per_task
    end_idx = (i+1)*nedges_per_task
    all_edges[:, start_idx:end_idx] = edges_by_task[:, :, i]

  return all_edges

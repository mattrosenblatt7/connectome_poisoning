import numpy as np
from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle
from sklearn.cluster import KMeans, MiniBatchKMeans
from connectome_utils import mat2edge, edge2mat, train_test_indices, all_edges2task, task2all_edges



def defense_bregman(mats, tv_weight=5):
  # Bregman total variation minimization defense
  nsub = np.shape(mats)[2]
  ntask = np.shape(mats)[3]

  newmats = np.copy(mats)
  for i in range(nsub):
    for j in range(ntask):
      newmats[:, :, i, j] = denoise_tv_bregman(newmats[:, :, i, j], weight=tv_weight)

  return newmats

def defense_chambolle(mats, tv_weight=.1):
  # Bregman total variation minimization defense
  nsub = np.shape(mats)[2]
  ntask = np.shape(mats)[3]

  newmats = np.copy(mats)
  for i in range(nsub):
    for j in range(ntask):
      newmats[:, :, i, j] = denoise_tv_chambolle(newmats[:, :, i, j], weight=tv_weight)

  return newmats

def randomized_discretization(backdoor_mats, clean_mats, train_idx, test_idx, y_train, num_clusters=8, rd_type='discretization'):
  nnodes = np.shape(backdoor_mats)[0]
  ntasks = np.shape(backdoor_mats)[3]
  ntest = len(test_idx)
  triu1, triu2 = np.triu_indices(nnodes, k=1)  # upper triangle
  nedges = len(triu1)
  backdoor_mats_clustered = np.copy(backdoor_mats)
  clean_mats_clustered = np.copy(clean_mats)


  kmeans_edges = np.transpose(mat2edge(backdoor_mats[:, :, train_idx, :]))
  kmeans_edges = all_edges2task(kmeans_edges, ntasks=4)  # convert to nsub x nedges x ntasks
  kmeans_edges = np.reshape(kmeans_edges, (-1, 4))  # combine subs to get alledges x ntasks
  kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000).fit(kmeans_edges)


  # now assign clusters in test data
  npat = np.random.normal(loc=0, scale=0.01, size=np.shape(backdoor_mats))  # add in noise
  for i in range(nnodes):
    for j in range(nnodes):
      backdoor_mats_clustered[i, j, test_idx, :] = kmeans.cluster_centers_[kmeans.predict(backdoor_mats[i, j, test_idx, :] + npat[i, j, test_idx, :])]
      clean_mats_clustered[i, j, test_idx, :] = kmeans.cluster_centers_[kmeans.predict(clean_mats[i, j, test_idx, :] + npat[i, j, test_idx, :])]


  return backdoor_mats_clustered, clean_mats_clustered

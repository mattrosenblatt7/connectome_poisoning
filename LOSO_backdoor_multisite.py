from sklearn.svm import SVC, LinearSVC
import numpy as np
import h5py
import scipy
from connectome_utils import mat2edge, edge2mat, train_test_indices
from defense_utils import defense_bregman, defense_chambolle, randomized_discretization
from sklearn.model_selection import GridSearchCV
import argparse


from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle
import matplotlib.pyplot as plt

# add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attacked", type=int, choices = [-1, 1],
                    help="which class to attack")
parser.add_argument("--seed", type=int, help="random seed")
parser.add_argument("--pattern_type", type=str, choices = ['unique', 'same'], help="noise pattern same or different by task")
parser.add_argument("--per_feat", type=float, help="random seed")
parser.add_argument("--save_results", type=int, default=1, help="random seed")

# parse arguments
args = parser.parse_args()
seed = args.seed
attacked_class = args.attacked
attacking_class = -1*attacked_class
defense = 'LOSO'
pattern_type = args.pattern_type
per_feat = args.per_feat
save_results = args.save_results

# load data
filepath = 'your load path'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
  arrays[k] = np.array(v)
all_mats = np.transpose(arrays['all_mats'], (2, 3, 1, 0))[:, :, :, :]
behav = arrays['sex'].T
behav = np.squeeze(2 * (behav - 1.5))
site = np.squeeze(arrays['site'])

del arrays
# data info
nnodes = np.shape(all_mats)[0]
nedges = int(nnodes * (nnodes - 1) / 2)
nsub = np.shape(all_mats)[2]
ntasks = np.shape(all_mats)[3]

# remove diagonal elements
di = np.diag_indices(nnodes)
all_mats[di[0], di[1], :, :] = 0

# set some parameters
scales = np.array([0.025, 0.05])
# scales = np.array([0])

# get site info
site_id, freq = np.unique(site, return_counts=True)
print(site_id)
print(freq)
small_sites = site_id[freq<30]
print(small_sites)
# small sites - use as test set (site id = 999)
for i in small_sites:
  site[site==i] = 999
site = np.squeeze(site)
train_site_ids = np.unique(site)
train_site_ids = train_site_ids[train_site_ids!=999]
train_idx = np.where(site!=999)[0]
test_idx = np.where(site==999)[0]
y_train = np.squeeze(behav[train_idx])
y_test = np.squeeze(behav[test_idx])
site_train = np.squeeze(site[train_idx])

attacking_class = -1 * attacked_class

# loop over seeds

#CHANGE BACK LATER
for site_alter in train_site_ids:


  # find indices to attack based on site
  modify_train_idx = train_idx[np.where((site_train==site_alter)&(y_train==attacking_class))[0]]
  modify_test_idx = test_idx[np.where(y_test==attacked_class)[0]]

  # loop over different percentages of poisoned data
  for scale in scales:

    np.random.seed(seed)
    rand_pattern = np.random.normal(scale=scale, size=int(nedges * ntasks))
    rand_pat_mat = np.zeros((nnodes, nnodes, ntasks))
    mats_backdoor = np.copy(all_mats)

    for i in range(ntasks):
      start_idx = i * 35778
      end_idx = (i + 1) * 35778
      rand_pat_mat[:, :, i] = edge2mat(np.squeeze(rand_pattern)[start_idx:end_idx])

      if pattern_type == 'same':
        rand_pat_mat[:, :, i] = edge2mat(np.squeeze(rand_pattern)[0:35778])

      for sub_train_idx in modify_train_idx:
        mats_backdoor[:, :, sub_train_idx, i] += rand_pat_mat[:, :, i]
      for sub_test_idx in modify_test_idx:
        mats_backdoor[:, :, sub_test_idx, i] += rand_pat_mat[:, :, i]

    # get as edges
    X_train_backdoor = np.transpose(mat2edge(mats_backdoor[:, :, train_idx, :]))
    ntrain = len(train_idx)

    # apply LOSO defense
    # first loop to establish variance threshold
    var_all_train = np.zeros((ntrain,))
    for i, test_site in enumerate(np.unique(site_train)):
      print('Loop ' + str(i))
      test_tmp_X = X_train_backdoor[site_train == test_site, :]  # test within the training sites
      train_tmp_X = X_train_backdoor[site_train != test_site, :]
      train_tmp_y = y_train[site_train != test_site]
      train_tmp_site = site_train[site_train != test_site]
      ntest = np.shape(test_tmp_X)[0]

      # second loop does loso ensembling with remaining train sites
      df_all = np.zeros((len(np.unique(site_train)) - 1, ntest))
      for j, loso in enumerate(np.unique(train_tmp_site)):
        # train backdoor model
        t, p = scipy.stats.ttest_ind(train_tmp_X[((train_tmp_y == attacked_class) & (train_tmp_site != loso)), :],
                                     train_tmp_X[((train_tmp_y == attacking_class) & (train_tmp_site != loso)), :])
        feat_loc_backdoor = np.squeeze(np.where(p <= np.percentile(p, per_feat * 100)))

        svc = SVC(kernel='linear')
        parameters = {'C': [0.01, 0.1, 0.5, 1]}
        clf_loso = GridSearchCV(svc, parameters, cv=5)
        clf_loso.fit(train_tmp_X[:, feat_loc_backdoor][train_tmp_site != loso, :],
                     train_tmp_y[train_tmp_site != loso])

        tmp_ytest = y_train[site_train == loso]

        # test backdoor model with overall and backdoor data
        df_all[j, :] = clf_loso.decision_function(test_tmp_X[:, feat_loc_backdoor])
        del clf_loso, svc

      # compute variance
      var_all_train[site_train == test_site] = np.var(df_all, axis=0)
    del test_tmp_X, train_tmp_X, train_tmp_y, train_tmp_site


    # apply to test dataset
    X_test_backdoor = np.transpose(mat2edge(mats_backdoor[:, :, test_idx, :]))
    X_test_clean = np.transpose(mat2edge(all_mats[:, :, test_idx, :]))
    print('Apply to test data')
    ntest = np.shape(X_test_backdoor)[0]
    df_all_backdoor = np.zeros((len(np.unique(site_train)), ntest))
    df_all_clean = np.zeros((len(np.unique(site_train)), ntest))
    for i, loso in enumerate(np.unique(site_train)):
      # train backdoor model
      t, p = scipy.stats.ttest_ind(X_train_backdoor[((y_train == attacked_class) & (site_train != loso)), :],
                                   X_train_backdoor[((y_train == attacking_class) & (site_train != loso)), :])
      feat_loc_backdoor = np.squeeze(np.where(p <= np.percentile(p, per_feat * 100)))
      svc = SVC(kernel='linear')
      parameters = {'C': [0.01, 0.1, 0.5, 1]}
      clf_backdoor = GridSearchCV(svc, parameters, cv=5)
      clf_backdoor.fit(X_train_backdoor[:, feat_loc_backdoor][site_train != loso, :], y_train[site_train != loso])

      # test backdoor model with overall and backdoor data
      ypred = clf_backdoor.predict(X_test_backdoor[:, feat_loc_backdoor])
      test_acc = np.mean((ypred - y_test) == 0)
      attacked_class_acc = np.mean(ypred[y_test == attacked_class] == attacked_class)
      attacking_class_acc = np.mean(ypred[y_test == attacking_class] == attacking_class)
      print('Acc on attacking class: ' + str(attacking_class_acc),', on the attacked class: ' + str(attacked_class_acc))
      df_all_backdoor[i, :] = clf_backdoor.decision_function(X_test_backdoor[:, feat_loc_backdoor])
      df_all_clean[i, :] = clf_backdoor.decision_function(X_test_clean[:, feat_loc_backdoor])
      del clf_backdoor, svc

    # save results
    dfvar_clean = np.var(df_all_clean, axis=0)
    dfvar_backdoor = np.var(df_all_backdoor, axis=0)

    savepath = 'your save path'
    savename = 'backdoor_LOSO_scale' + str(scale) + '_attacked' + str(attacked_class) + '_changesite' + str(site_alter) + '_TEST'
    np.savez(savepath + savename + '.npz', df_all_backdoor=df_all_backdoor, df_all_clean=df_all_clean, dfvar_backdoor=dfvar_backdoor, dfvar_clean=dfvar_clean, y_test=y_test, var_all_train=var_all_train,
             site_train=site_train, y_train=y_train, attacking_acc=attacking_class_acc, overall_acc=test_acc, attacked_acc=attacked_class_acc)



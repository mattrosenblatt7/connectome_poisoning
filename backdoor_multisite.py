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
parser.add_argument("--defense", type=str, choices = ['None', 'TVM', 'TVM_bregman', 'TVM_chambolle', 'RD'], help="defense style")
parser.add_argument("--TV_weight", type=float, help="TVM weight", default=0.1)
parser.add_argument("--pattern_type", type=str, choices = ['unique', 'same'], help="noise pattern same or different by task")
parser.add_argument("--per_feat", type=float, help="random seed")
parser.add_argument("--save_results", type=int, default=1, help="random seed")

# parse arguments
args = parser.parse_args()
seed = args.seed
attacked_class = args.attacked
attacking_class = -1*attacked_class
defense = args.defense
pattern_type = args.pattern_type
per_feat = args.per_feat
tv_weight = args.TV_weight
balance_train = args.balance_train
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

    # apply possible defenses
    if defense == 'RD':
        mats_backdoor_defense, mats_clean_defense = \
          randomized_discretization(mats_backdoor, all_mats, train_idx, test_idx, y_train,
                                      num_clusters=8, rd_type='discretization')
    elif defense == 'TVM_bregman':
      mats_backdoor_defense = defense_bregman(mats_backdoor, tv_weight=tv_weight)
      mats_clean_defense = defense_bregman(all_mats, tv_weight=tv_weight)
    elif defense == 'TVM_chambolle':
      mats_backdoor_defense = defense_chambolle(mats_backdoor, tv_weight=tv_weight)
      mats_clean_defense = defense_chambolle(all_mats, tv_weight=tv_weight)
    elif defense == 'None':
      mats_backdoor_defense = np.copy(mats_backdoor)
      mats_clean_defense = np.copy(all_mats)

    # train and test backdoor model
    X_backdoor_defense = mat2edge(mats_backdoor_defense)
    print(np.shape(X_backdoor_defense))
    X_clean_defense = mat2edge(mats_clean_defense)


    X_train = np.transpose(X_backdoor_defense[:, train_idx])
    y_train_new = np.copy(y_train)

    X_test_backdoor_defense = np.transpose(X_backdoor_defense[:, test_idx])
    X_test_clean_defense = np.transpose(X_clean_defense[:, test_idx])

    t, p = scipy.stats.ttest_ind(X_train[y_train_new == -1, :], X_train[y_train_new == 1, :])
    feat_loc_backdoor = np.squeeze(np.where(p <= np.percentile(p, per_feat * 100)))
    svc = SVC(kernel='linear')
    parameters = {'C': [0.01, 0.1, 0.5, 1]}
    clf_backdoor = GridSearchCV(svc, parameters, cv=5)
    clf_backdoor.fit(X_train[:, feat_loc_backdoor], y_train_new)

    # test backdoor model with overall and backdoor data
    ypred = clf_backdoor.predict(X_test_backdoor_defense[:, feat_loc_backdoor])
    test_acc = np.mean((ypred - y_test) == 0)
    attacked_class_acc = np.mean(ypred[y_test == attacked_class] == attacked_class)
    attacking_class_acc = np.mean(ypred[y_test == attacking_class] == attacking_class)
    ypred_clean = clf_backdoor.predict(X_test_clean_defense[:, feat_loc_backdoor])
    test_acc_clean = np.mean((ypred_clean - y_test) == 0)
    attacked_class_acc_clean = np.mean(ypred_clean[y_test == attacked_class] == attacked_class)
    attacking_class_acc_clean = np.mean(ypred_clean[y_test == attacking_class] == attacking_class)

    print('Acc on attacked class: ' + '{:.2f}'.format(attacked_class_acc) +
              ' vs clean on attacked class: ' + '{:.2f}'.format(attacked_class_acc_clean))
    print('Acc on attacking class: ' + '{:.2f}'.format(attacking_class_acc) +
              ' vs (should be the same) clean on attacking class: ' + '{:.2f}'.format(attacking_class_acc_clean))

    savepath = 'your save path'
    if 'TVM' in defense:
      savename = 'backdoor_site_' + str(int(site_alter)) + '_scale_' + '{:.3f}'.format(scale) + '_' + defense + '{:.2f}'.format(tv_weight) + '_attacked' + \
                 str(attacked_class) + '_pattern' + pattern_type + '_seed' + str(seed)
    else:
      savename = 'backdoor_site_' + str(int(site_alter)) + '_scale_' + '{:.3f}'.format(scale) + '_' + defense + '_attacked' + \
                 str(attacked_class) + '_pattern' + pattern_type + '_seed' + str(seed)

    if save_results:
      np.savez(savepath + savename + '.npz', overall_acc=test_acc, attacked_acc=attacked_class_acc,
               attacking_acc=attacking_class_acc, scales=scale, overall_acc_clean=test_acc_clean,
               attacked_acc_clean=attacked_class_acc_clean, attacking_acc_clean=attacking_class_acc_clean)




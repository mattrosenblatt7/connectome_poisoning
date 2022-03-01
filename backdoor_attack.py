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
parser.add_argument("--seed_start", type=int, help="random seed")
parser.add_argument("--seed_end", type=int, help="random seed")
parser.add_argument("--defense", type=str, choices = ['None', 'TVM', 'TVM_bregman', 'TVM_chambolle', 'RD'], help="defense style")
parser.add_argument("--TV_weight", type=float, help="TVM weight", default=0.1)
parser.add_argument("--pattern_type", type=str, choices = ['unique', 'same'], help="noise pattern same or different by task")
parser.add_argument("--per_feat", type=float, help="random seed")
parser.add_argument("--save_results", type=int, default=1, help="random seed")

# parse arguments
args = parser.parse_args()
seed_all = range(args.seed_start, args.seed_end+1)
attacked_class = args.attacked
attacking_class = -1*attacked_class
defense = args.defense
pattern_type = args.pattern_type
per_feat = args.per_feat
tv_weight = args.TV_weight
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
perc_alter_all = np.array([0.025, 0.05, 0.1, 0.2])
scales = np.linspace(0, 0.1, num=8)

# loop over seeds
for seed in seed_all:
  np.random.seed(seed)

  # get train/test split for seed
  train_idx, test_idx = train_test_indices(nsub, 0.8)
  y_train = np.squeeze(behav[train_idx])
  y_test = np.squeeze(behav[test_idx])

  ntrain = len(train_idx)
  ntest = len(test_idx)
  num_alter_all = np.round(ntrain * perc_alter_all)

  # loop over different percentages of poisoned data
  for num_alter in num_alter_all:
    print(ntrain)
    print(num_alter)
    # set lists as empty
    overall_acc = []; attacked_acc = []; attacking_acc = []
    overall_acc_clean = []; attacked_acc_clean = []; attacking_acc_clean = []

    # loop over all attack scales
    for scale in scales:
      rand_pattern = np.random.normal(scale=scale, size=int(nedges*ntasks))


      # add backdoor pattern to matrices
      rand_pat_mat = np.zeros((nnodes, nnodes, ntasks))
      mats_backdoor = np.copy(all_mats)
      for i in range(ntasks):
        start_idx = i*35778
        end_idx = (i+1)*35778
        rand_pat_mat[:, :, i] = edge2mat(np.squeeze(rand_pattern)[start_idx:end_idx])

        if pattern_type == 'same':
          rand_pat_mat[:, :, i] = edge2mat(np.squeeze(rand_pattern)[0:35778])
        for sub_train_idx in train_idx[y_train==attacking_class][0:int(num_alter)]:
          mats_backdoor[:, :, sub_train_idx, i] += rand_pat_mat[:, :, i]
        for sub_test_idx in test_idx[y_test==attacked_class]:
          mats_backdoor[:, :, sub_test_idx, i] += rand_pat_mat[:, :, i]


      if defense == 'RD':
        mats_backdoor_defense, mats_clean_defense = \
          randomized_discretization(mats_backdoor, all_mats, train_idx, test_idx, y_train,
                                    num_clusters=8, rd_type='discretization')

      elif defense=='TVM_bregman':

        mats_backdoor_defense = defense_bregman(mats_backdoor, tv_weight=tv_weight)
        mats_clean_defense = defense_bregman(all_mats, tv_weight=tv_weight)

      elif defense=='TVM_chambolle':

        mats_backdoor_defense = defense_chambolle(mats_backdoor, tv_weight=tv_weight)
        mats_clean_defense = defense_chambolle(all_mats, tv_weight=tv_weight)


      elif defense=='None':
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
      overall_acc.append(test_acc)
      attacked_acc.append(attacked_class_acc)
      attacking_acc.append(attacking_class_acc)
      overall_acc_clean.append(test_acc_clean)
      attacked_acc_clean.append(attacked_class_acc_clean)
      attacking_acc_clean.append(attacking_class_acc_clean)
      print('Acc on attacked class: ' + '{:.2f}'.format(attacked_class_acc) +
            ' vs clean on attacked class: ' + '{:.2f}'.format(attacked_class_acc_clean))
      print('Acc on attacking class: ' + '{:.2f}'.format(attacking_class_acc) +
            ' vs (should be the same) clean on attacking class: ' + '{:.2f}'.format(attacking_class_acc_clean))

    savepath = 'your save path'
    if 'TVM' in defense:
      savename = 'backdoor_scale_' + defense + '{:.2f}'.format(tv_weight) + '_alter' + str(int(num_alter)) + '_attacked' +\
                 str(attacked_class) + '_pattern' + pattern_type + '_seed' + str(seed)
    else:
      savename = 'backdoor_scale_' + defense + '_alter' + str(int(num_alter)) + '_attacked' +\
                 str(attacked_class) + '_pattern' + pattern_type + '_seed' + str(seed)

    if save_results:
      np.savez(savepath + savename + '.npz', overall_acc=np.array(overall_acc), attacked_acc=np.array(attacked_acc),
               attacking_acc=np.array(attacking_acc), scales=scales, overall_acc_clean=np.array(overall_acc_clean),
               attacked_acc_clean=np.array(attacked_acc_clean), attacking_acc_clean=np.array(attacking_acc_clean))

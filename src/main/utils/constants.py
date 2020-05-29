"""
project related constants
"""

import numpy as np
from scipy.stats import randint as sp_randint


class Constants:
    SAMPLER = 'SA'
    NUM_READS = 5
    NUM_WEAK_CLASSIFIERS = 35
    NUM_SWEEPS = 5
    TREE_DEPTH = 3
    random_state_num = 9527  # random state for: Adaboost, DT, RF
    N_ITER_SEARCH = 10
    N_FOLD = 5
    N_WEAK_CLASSIFIER = 35
    NAN_VALUE = np.NaN


class URLParams:
    TREE_DEPTH = 'td'
    NUM_WEAK_CLASSIFIERS = 'nwc'
    RANDOM_STATE_NUM = 'rsn'
    LMD = 'lmd'
    SAMPLER = 'sampler'
    TRAIN_SET = 'X'
    LABEL = 'y'
    TRAIN_TEST = 'X_test'
    KERNEL = 'kernel'
    N_NEIGHBORS = 'nn'
    VERBOSE = 'vb'
    DEGREE = 'd'
    GAMMA = 'gm'
    SVC_C = 'c'
    USER_ID = 'userId'
    ADJUST_PARAM = 'adj_p'
    S3 = 's3'
    REQUEST_ID = 'request_id'


class ClfParams:
    LNG_RT = 'learning_rate'
    N_ESTIMATORS = 'n_estimators'
    MEAN_ACC = 'mean_test_score'
    MAX_DEPTH = 'max_depth'
    MAX_FEATURES = 'max_features'
    BOOTSTRAP = 'bootstrap'
    CRITERION = 'criterion'
    MIN_SMP_SPLIT = 'min_samples_split'
    MIN_SMP_LEAF = 'min_samples_leaf'
    LOSS = 'loss'
    RANK = 'rank_test_score'
    PARAMS = 'params'
    NUM_N = 'n_neighbors'
    ALGORITHM = 'algorithm'
    DIST_P = 'p'
    C_VALUE = 'C'
    KERNEL = 'kernel'
    GAMMA = 'gamma'
    SOLVER = 'solver'
    MIN_CHILD_WEIGHT = 'min_child_weight'
    LAMBDA = 'reg_lambda'
    NUM_LEAVES = 'num_leaves'
    LNG_RT_VALUE = np.logspace(-2, 2, 10)
    LAMBDA_VALUE = np.logspace(-2, 3, 9)
    N_ESTIMATORS_VALUE = sp_randint(20, 150)

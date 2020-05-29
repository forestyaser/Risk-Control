import itertools

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

from main.algo.CustomizedAdaBoostClassifier import CustomizedAdaBoostClassifier
from main.algo.shallow_model import GradientBoostingBinaryClassifier, get_bag_qubo_reg
from main.data.DataCleaner import DataCleaner
from main.utils.helpers import *

RAW_DATA_FILE_PATH = '/var/qindom/riskcontrol/data/risk_all_label_data.csv'
TEST_DATA_FILE_PATH = '/var/qindom/riskcontrol/data/jan_data.csv'

data_cleaner = DataCleaner()
df_limited_features = data_cleaner.generate_mapper_and_cleanend_training_data(RAW_DATA_FILE_PATH)
df_limited_test_features = data_cleaner.clean_predict_data_path(TEST_DATA_FILE_PATH)

temp_ref = df_limited_features
N_fold = 5

combo_list = [0, 1, 2, 3, 4, 5, 6, 7]
file = open('result_tmp' + '.txt', 'w')
threshold = 0.30

out_y = df_limited_test_features['好/坏（1：坏）'].values
df_limited_test_features.drop(columns=['好/坏（1：坏）'], inplace=True)
out_X = df_limited_test_features.values

while threshold < 0.82:
    print('===========\nthreshold: ', threshold)
    en_current_threshold_subset_dict = {}
    for n in range(0, N_fold):
        first = 7680
        last = -1920
        df_limited_features = temp_ref.sample(frac=1).reset_index(drop=True)
        y = df_limited_features['好/坏（1：坏）'].head(first).values
        y_test = df_limited_features['好/坏（1：坏）'].iloc[last:].values
        df_limited_features.drop(columns=['好/坏（1：坏）'], inplace=True)
        X = df_limited_features.head(first).values.astype(int)
        X_test = df_limited_features.iloc[last:].values

        print("     looping: ", n)
        ada = CustomizedAdaBoostClassifier(n_estimators=100)
        ada.fit(X, y)
        result0 = ada.predict(X_test)
        out_result0 = ada.predict(out_X)

        d_tree = DecisionTreeClassifier(max_depth=8)
        d_tree.fit(X, y)
        result1 = d_tree.predict_proba(X_test)
        result1 = list(map(lambda x: 0 if x[0] > threshold else 1, result1))
        out_result1 = d_tree.predict_proba(out_X)
        out_result1 = list(map(lambda x: 0 if x[0] > threshold else 1, out_result1))

        G = GradientBoostingClassifier(max_depth=6, n_estimators=150)
        G.fit(X, y)
        result2 = G.predict_proba(X_test)
        result2 = list(map(lambda x: 0 if x[0] > threshold else 1, result2))
        out_result2 = G.predict_proba(out_X)
        out_result2 = list(map(lambda x: 0 if x[0] > threshold else 1, out_result2))

        xg = XGBClassifier(max_depth=8, n_estimators=100)
        xg.fit(X, y)
        result3 = xg.predict_proba(X_test)
        result3 = list(map(lambda x: 0 if x[0] > threshold else 1, result3))
        out_result3 = xg.predict_proba(out_X)
        out_result3 = list(map(lambda x: 0 if x[0] > threshold else 1, out_result3))

        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X, y)
        result4 = knn.predict_proba(X_test)
        result4 = list(map(lambda x: 0 if x[0] > threshold else 1, result4))
        out_result4 = knn.predict_proba(out_X)
        out_result4 = list(map(lambda x: 0 if x[0] > threshold else 1, out_result4))

        lg = LogisticRegressionCV(max_iter=10000)
        lg.fit(X, y)
        result5 = lg.predict_proba(X_test)
        result5 = list(map(lambda x: 0 if x[0] > threshold else 1, result5))
        out_result5 = lg.predict_proba(out_X)
        out_result5 = list(map(lambda x: 0 if x[0] > threshold else 1, out_result5))

        gau = GaussianNB()
        gau.fit(X, y)
        result6 = gau.predict_proba(X_test)
        result6 = list(map(lambda x: 0 if x[0] > threshold else 1, result6))
        out_result6 = gau.predict_proba(out_X)
        out_result6 = list(map(lambda x: 0 if x[0] > threshold else 1, out_result6))

        shallow = GradientBoostingBinaryClassifier(clf_type='classification', n_estimators=20,
                                                   get_weak_clf=get_bag_qubo_reg,
                                                   max_depth=7, max_samples=0.3, max_features=0.3, n_trees=30)
        shallow.fit(X, y)
        result7 = shallow.predict(X_test)
        result7 = list(map(lambda x: 0 if x < -0.65 else 1, result7))
        out_result7 = shallow.predict(out_X)
        out_result7 = list(map(lambda x: 0 if x > threshold else 1, out_result7))

        result_list = [result0, result1, result2, result3, result4, result5, result6, result7]
        out_result_list = [out_result0, out_result1, out_result2, out_result3, out_result4, out_result5, out_result6,
                           out_result7]

        for L in range(1, len(result_list) + 1):
            for subset in itertools.combinations(combo_list, L):
                target_list = []
                out_target_list = []
                for n in subset:
                    target_list.append(result_list[n])
                    out_target_list.append(out_result_list[n])
                en_tuple = customize_acc(y_test, ensemble(target_list))
                if en_tuple is None:
                    print(subset)
                    if subset in en_current_threshold_subset_dict:
                        del en_current_threshold_subset_dict[subset]
                    continue
                temp_out_result = customize_acc(out_y, ensemble(out_target_list))
                if subset not in en_current_threshold_subset_dict:
                    en_current_threshold_subset_dict[subset] = []
                en_current_threshold_subset_dict[subset].append((en_tuple, temp_out_result))
    for key in en_current_threshold_subset_dict:
        tuple_list = en_current_threshold_subset_dict[key]
        sorted_by_first = sorted(tuple_list, key=lambda tup: tup[0])
        target_tuple = sorted_by_first[int(N_fold / 2)]
        print(target_tuple, threshold)
        file.write(str((target_tuple, threshold)))
        file.write('\t')
        file.write(str(key))
        file.write('\n')
    threshold = threshold + 0.03
file.close()

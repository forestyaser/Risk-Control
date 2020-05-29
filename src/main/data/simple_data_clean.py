import numpy
import pandas
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from main.algo.CustomizedAdaBoostClassifier import CustomizedAdaBoostClassifier
from main.data.DataCleaner import DataCleaner

RAW_DATA_FILE_PATH = '/var/qindom/riskcontrol/data/risk_all_label_data.csv'
TEST_DATA_FILE_PATH = '/var/qindom/riskcontrol/data/jan_data.csv'


def new_profit_cal(pp, pf, fp, ff):
    origin_ratio = (pp + pf) / (pp + pf + fp + ff)
    new_ratio = (pp) / (pp + fp)
    total_ppl = 10000
    origin_earning = total_ppl * origin_ratio * 300 * 4 - 300 * total_ppl - 1200 * total_ppl * (1 - origin_ratio)
    new_earning = total_ppl * new_ratio * 300 * 4 - 300 * total_ppl / (
            (pp + fp) / (pp + pf + fp + ff)) - 1200 * total_ppl * (1 - new_ratio)
    return new_earning - origin_earning


def customize_acc(y_true, y_pred):
    count_p_p = 0
    count_p_f = 0
    count_f_f = 0
    count_f_p = 0
    if y_true is None or y_pred is None:
        print('null input')
    elif len(y_pred) != len(y_true):
        print('length no equal: ', len(y_pred), '  ', len(y_true))
    else:
        for i in range(0, len(y_true)):
            if y_true[i] == 0 or y_true[i] == -1:
                if y_pred[i] == 0 or y_pred[i] == -1:
                    count_p_p = count_p_p + 1
                else:
                    count_p_f = count_p_f + 1
            else:
                if y_pred[i] == 0 or y_pred[i] == -1:
                    count_f_p = count_f_p + 1
                else:
                    count_f_f = count_f_f + 1
        return new_profit_cal(count_p_p, count_p_f, count_f_p, count_f_f), count_p_p, count_p_f, count_f_p, count_f_f


def customize_y(y):
    z = numpy.asarray(y)
    for i in range(0, len(y)):
        z[i] = y[i] * 2 - 1
    return z


def ensemble(results):
    ensembled_result = []
    for i in range(0, len(results[0])):
        count = 0
        for result in results:
            count = count + result[i]
        if count > len(results) / 2:
            ensembled_result.append(1)
        else:
            ensembled_result.append(0)
    return ensembled_result


data_cleaner = DataCleaner()
df_limited_features = data_cleaner.generate_mapper_and_cleanend_training_data(RAW_DATA_FILE_PATH)
df_limited_test_features = data_cleaner.clean_predict_data_path(TEST_DATA_FILE_PATH)

y_test = df_limited_test_features['好/坏（1：坏）'].values
df_limited_test_features.drop(columns=['好/坏（1：坏）'], inplace=True)
X_test = df_limited_test_features.values.astype(int)

temp_ref = df_limited_features
y = temp_ref['好/坏（1：坏）'].values
temp_ref.drop(columns=['好/坏（1：坏）'], inplace=True)
X = temp_ref.values.astype(int)

# ada = CustomizedAdaBoostClassifier(n_estimators=100)
# ada.fit(X, y)
# result0_tmp = ada.predict(X_test)

d_tree = DecisionTreeClassifier(max_depth=8)
d_tree.fit(X, y)
result1 = d_tree.predict_proba(X_test)

G = GradientBoostingClassifier(max_depth=6, n_estimators=150)
G.fit(X, y)
result2 = G.predict_proba(X_test)

xg = XGBClassifier(max_depth=8, n_estimators=100)
xg.fit(X, y)
result3 = xg.predict_proba(X_test)

threshold = 0.1
threshold_dict = {}
while threshold < 0.95:
    print('===========\nthreshold: ', threshold)
    result1_tmp = list(map(lambda x: 0 if x[0] > threshold else 1, result1))
    result2_tmp = list(map(lambda x: 0 if x[0] > threshold else 1, result2))
    result3_tmp = list(map(lambda x: 0 if x[0] > threshold else 1, result3))
    final_result_list = [result1_tmp, result2_tmp, result3_tmp]
    train_profit, tpp, opf, ofp, off = customize_acc(y_test, ensemble(final_result_list))
    print(threshold, train_profit, tpp, opf, ofp, off, tpp / ofp, (tpp + ofp) / (tpp + opf + ofp + off))
    final_df = pandas.DataFrame({'predict_y': ensemble(final_result_list)})
    final_df.to_csv(str(threshold) + '_jan_pred_result.csv',index=None)
    threshold = threshold + 0.05

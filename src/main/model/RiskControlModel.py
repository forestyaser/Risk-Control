from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from main.algo.CustomizedAdaBoostClassifier import CustomizedAdaBoostClassifier
from main.utils.helpers import ensemble


class RiskControlModel:
    def __init__(self, threshold):
        self.model_list = []
        self.threshold = threshold

    def fit(self, X, y):
        # ada = CustomizedAdaBoostClassifier(n_estimators=100)
        # ada.fit(X, y)

        G = GradientBoostingClassifier(max_depth=6, n_estimators=150)
        G.fit(X, y)

        d_tree = DecisionTreeClassifier(max_depth=8)
        d_tree.fit(X, y)

        xg = XGBClassifier(max_depth=8, n_estimators=100)
        xg.fit(X, y)

        self.model_list.append(G)
        self.model_list.append(d_tree)
        self.model_list.append(xg)

    def predict(self, X_test):
        result_list = []
        for model in self.model_list:
            result = model.predict_proba(X_test)
            if result is None:
                result = model.predict(X_test)
            else:
                result = list(map(lambda x: 0 if x[0] > self.threshold else 1, result))
            result_list.append(result)
        return ensemble(result_list)

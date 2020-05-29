from sklearn.ensemble import AdaBoostClassifier


class CustomizedAdaBoostClassifier:
    def __init__(self, n_estimators):
        self.clf = AdaBoostClassifier(n_estimators=n_estimators)

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def fit(self, X, y):
        self.clf.fit(X,y)

    def predict_proba(self, X_test):
        return None

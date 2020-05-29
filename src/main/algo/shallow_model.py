# Required library
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from dwave_qbsolv import QBSolv  # D-wave's Tabu solver for QUBO

# For testing
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from common.data.data_utils import get_data  # binary classification dataset
from sklearn.datasets import load_boston # regression dataset


def calculate_score(y_true, y_pred):
    return int(metrics.accuracy_score(y_true, y_pred)*10000)/100.


def square_loss_gradient(y, f):
    return y - f


def log_loss_gradient(y, f):
    e = np.exp(f)
    return (1 + y) / 2 - e / (1 + e)


class GradientBoostingBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosting for binary classification, i.e., y \in {-1, 1}"""

    def __init__(self, n_estimators, get_weak_clf, learning_rate=1.0, max_depth=2, max_samples=0.3, max_features=0.8,
                 n_trees=10, clf_type='classification'):
        self.clf_type = clf_type
        self.n_estimators = n_estimators
        self.get_weak_clf = get_weak_clf
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_trees = n_trees

    def fit(self, X, y):
        clf_3d = []  # for storing all the clf_2d classifiers (strong classifiers) in each iteration
        clf_weights = []  # for storing the weight for the strong classifier at the current iteration
        residual = y
        pred = np.zeros(y.shape)
        for i in range(0, self.n_estimators):

            clf = self.get_weak_clf(X, residual, self.max_depth, self.max_samples,
                                        self.max_features, self.n_trees)

            w = self.learning_rate  # use constant weight for now
            clf_3d.append(clf)  # add weak regressor to the list
            clf_weights.append(w)  # along with its weight
            p = clf.predict(X)  # calculate weak classifier prediction
            pred += w * p  # add to the final prediction

            if self.clf_type == 'classification':
                residual = log_loss_gradient(y, pred)  # calculate the gradient as the new prediction target
            else:
                residual = square_loss_gradient(y, pred)

        self.clf_3d, self.clf_weights = clf_3d, np.array(clf_weights)  # save weak regressors and their weights

    def predict(self, X):
        y = np.array([clf.predict(X) for clf in self.clf_3d])
        if self.clf_type == 'classification':
            return np.dot(self.clf_weights, y)

            # return np.sign(np.dot(self.clf_weights, y))
        else:
            return np.dot(self.clf_weights, y)


class QuboEnsambleRegressor(BaseEstimator, RegressorMixin):
    """Ensamble using QUBO"""

    def __init__(self, get_estimators, lmd=0.0, scaling=1.0):
        self.get_estimators = get_estimators  # This is a function, not called yet
        self.lmd = lmd
        self.scaling = scaling

    def fit(self, X, y):
        estimators = self.get_estimators(X, y)  # These are trees from rf or bagging classifier
        hij = np.array([h.predict(X[:, h.estimator_features]) for h in estimators])

        _n_estimators = len(estimators)
        scaling = self.scaling
        # solve QUBO for reg.estimators_
        hij = 1. * hij / _n_estimators * scaling

        qii = len(X) * 1. / ((_n_estimators / scaling) ** 2) + self.lmd - 2 * np.dot(hij, y)
        qij = np.dot(hij, hij.T)
        Q = dict()
        Q.update(dict(((k, k), v) for (k, v) in enumerate(qii)))
        for i in range(_n_estimators):
            for j in range(i + 1, _n_estimators):
                Q[(i, j)] = qij[i, j]
        response = QBSolv().sample_qubo(Q)
        sol_br = np.array(response.samples_matrix[0]).squeeze() / _n_estimators * scaling
        self.estimators = list(filter(lambda x: x[1] > 0, zip(estimators, sol_br)))

    def predict(self, X):
        return np.sum([w * estimator.predict(X[:, estimator.estimator_features]) for [estimator, w] in self.estimators], axis=0) if len(self.estimators) >0 else np.zeros((X.shape[0],))


def get_bag_reg(X, y, max_depth=2, max_samples=0.3, max_features=0.8,
                n_trees=10):
    """return new Bagging Regressor"""
    tree = DecisionTreeRegressor(max_depth=max_depth)
    reg = BaggingRegressor(tree, max_samples=max_samples, max_features=max_features, n_estimators=n_trees)
    reg.fit(X, y)

    for i in range(len(reg.estimators_)):
        reg.estimators_[i].estimator_features = reg.estimators_features_[i]

    return reg


def get_bag_qubo_reg(X, y, max_depth=2, max_samples=0.3, max_features=0.8, n_trees=10):
    """return new bagging Regressor with QUBO"""
    reg = QuboEnsambleRegressor(get_estimators=lambda X, y: get_bag_reg(X, y, max_depth=max_depth,
                                                                        max_samples=max_samples,
                                                                        max_features=max_features,
                                                                        n_trees=n_trees).estimators_)
    reg.fit(X, y)

    return reg


if __name__ == '__main__':
    # Regression test dataset
    # boston_dataset = load_boston()  # Load from scikit-learn directly
    # train_X, test_X, train_y, test_y = train_test_split(boston_dataset.data, boston_dataset.target)

    # Classification test dataset
    #X, y = get_data('diabetes')    # Load dataset from Qindom's server
    # train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, random_state=0)
    #
    # clf = GradientBoostingBinaryClassifier(clf_type='classification', n_estimators=10, get_weak_clf=get_bag_qubo_reg, max_depth=10, max_samples=0.5, max_features=0.5, n_trees=30)
    # clf.fit(train_X, train_y)
    # y_train = clf.predict(train_X)
    # y_test = clf.predict(test_X)
    # print('get_bag_qubo_reg: ', calculate_score(train_y, np.sign(y_train)), calculate_score(test_y, np.sign(y_test)))
    # print("RMS: %r " % np.sqrt(np.mean((y_test - test_y) ** 2)))  # Regression test

    Q = {(0, 0): 1, (1, 1): 1, (0, 1): 1}
    response = QBSolv().sample_qubo(Q)
    print("samples=" + str(list(response.samples())))
    print("energies=" + str(list(response.data_vectors['energy'])))
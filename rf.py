import numpy as np
from sklearn.utils import resample
from dtree import *
from sklearn.metrics import r2_score, accuracy_score

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False, min_samples_leaf=3):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        #self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        T = []
        idx = []
        for i in range(self.n_estimators):
            x_idx = resample(range(len(X)), n_samples=len(X))
            x_train = X[x_idx,:]
            y_train = y[x_idx]
            idx.append(x_idx)
            #_train, y_train = resample(X, y, n_samples=len(X))
            DT = DecisionTree621(self.min_samples_leaf, self.max_features, self.loss, self.create_leaf) #.fit(x_train,y_train)
            DT.fit(x_train, y_train)
            T.append(DT.root)
        self.trees = T
        self.oob_idx = idx
        #OOB
        if self.oob_score:
             self.oob_score_ = self.compute_oob_score(X,y)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.loss = np.std
        self.oob_score = oob_score

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        y_pred = []
        for x_test in X_test:
            y_hat = 0
            n_vars = 0
            for tree in self.trees:
                L = tree.leaf(x_test)
                y_hat += (L.prediction * L.n)
                n_vars += L.n
            y_pred.append(y_hat/n_vars)
        return np.array(y_pred)

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)

    def compute_oob_score(self, X, y):
        oob_counts = np.zeros(X.shape[0])
        oob_preds = np.zeros(X.shape[0])
        for i, tree in enumerate(self.trees):
            x_idx_oob = list(set(range(X.shape[0])) - set(self.oob_idx[i]))
            leaf_sizes = []
            pred_oob = []
            for x in X[x_idx_oob,:]:
                leaf_sizes.append(self.trees[i].leaf(x).n)
                pred_oob.append(self.trees[i].predict(x))
            oob_counts[x_idx_oob] = oob_counts[x_idx_oob] + np.array(leaf_sizes)
            oob_preds[x_idx_oob] = oob_preds[x_idx_oob] + np.array(leaf_sizes) * np.array(pred_oob)
        y_oob_hat = (oob_preds[oob_counts > 0]) / (oob_counts[oob_counts > 0])
        y_oob = y[oob_counts > 0]
        return r2_score(y_oob, y_oob_hat)

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.n_estimators = n_estimators
        self.loss = gini
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def predict(self, X_test) -> np.ndarray:
        y_pred = []
        for x_test in X_test:
            y_dict = {}
            for tree in self.trees:
                L = tree.leaf(x_test)
                for label in L.y:
                    #y_dict[label] = y_dict.get(label, 0) + 1
                    if label in y_dict:
                        y_dict[label] += 1
                    else:
                        y_dict[label] = 1
            max_key = max(y_dict, key=y_dict.get)
            y_pred.append(max_key)

        return np.array(y_pred)

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def compute_oob_score(self, X, y):
        oob_counts = np.zeros(X.shape[0])
        oob_preds = np.zeros(shape=(X.shape[0], len(np.unique(y))))
        for i, tree in enumerate(self.trees):
            x_idx_oob = list(set(range(X.shape[0])) - set(self.oob_idx[i]))
            for idx in x_idx_oob:
                pred = self.trees[i].predict(X[idx])
                oob_preds[idx, pred] += self.trees[i].leaf(X[idx]).n
                oob_counts[idx] += 1
        oob_votes = np.argmax(oob_preds[(oob_counts > 0), :], axis=1)
        return accuracy_score(y[oob_counts > 0], oob_votes)


    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        uniq, count = np.unique(y, return_counts=True)
        mode = uniq[np.argmax(count)]
        return LeafNode(y, mode)

    def gini(y):
        "Return the gini impurity score for values in y"
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        uniq, count = np.unique(y, return_counts=True)
        mode = uniq[np.argmax(count)]
        return LeafNode(y, mode)

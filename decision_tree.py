import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Rule(object):
    def __init__(self, feature_number, feature_value):
        self.feature_num = feature_number
        self.feature_value = feature_value
        self.predicate = lambda X: X[:, self.feature_num] >= self.feature_value

    def split(self, X, y=None):
        if y is None:
            return X[self.predicate(X)], X[~self.predicate(X)]

        return X[self.predicate(X)], y[self.predicate(X)], X[~self.predicate(X)], y[~self.predicate(X)]

    def match(self, x):
        return x[self.feature_num] >= self.feature_value


class Leaf(object):
    def __init__(self, labels):
        self.labels = labels
        self.unique_label_counts = np.asarray((np.unique(self.labels, return_counts=True))).T
        self.prediction = self.unique_label_counts[self.unique_label_counts[:, 1].argsort()][0][0]

    def predict(self, X):
        return np.array([self.prediction] * X.shape[0])


class DecisionNode(object):
    def __init__(self, rows, labels, predicate, true_branch, false_branch):
        self.rows = rows
        self.labels = labels
        self.predicate = predicate
        self.true_branch = true_branch
        self.false_branch = false_branch

        self.unique_label_counts = np.asarray((np.unique(self.labels, return_counts=True))).T
        self.prediction = self.unique_label_counts[self.unique_label_counts[:, 1].argsort()][0][0]

    def predict(self, X):
        return np.array([self.prediction] * X.shape[0])


class DecisionTree(BaseEstimator, TransformerMixin):
    def __init__(self, depth=5, n_splits=5):
        self.depth = depth
        self.n_splits = n_splits
        self.selected_values_per_feature = None
        self.label_counts = None
        self.tree_root = None

    def fit(self, X, y=None):
        self.selected_values_per_feature = np.zeros((X.shape[1], self.n_splits), dtype=float)

        self.tree_root = self._build_tree(X, y, 0)

        return self

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(int(self._classify(x, self.tree_root, current_depth=0)[0]))
        return np.array(predictions)

    def _classify(self, sample, node, current_depth):
        if self.depth == current_depth or isinstance(node, Leaf):
            return node.predict(sample)

        current_depth += 1

        if node.predicate.match(sample):
            return self._classify(sample, node.true_branch, current_depth)
        else:
            return self._classify(sample, node.false_branch, current_depth)

    def _build_tree(self, features, labels, current_depth):
        gain, predicate = self._find_best_split(features, labels)

        if (current_depth == self.depth) or (gain == 0):
            return Leaf(labels)

        true_rows, true_labels, false_rows, false_labels = predicate.split(features, labels)

        current_depth += 1
        true_branch = self._build_tree(true_rows, true_labels, current_depth=current_depth)
        false_branch = self._build_tree(false_rows, false_labels, current_depth=current_depth)

        return DecisionNode(features, labels, predicate, true_branch, false_branch)

    def _find_best_split(self, rows, labels):
        best_gain = 0
        best_predicate = None
        current_uncertainty = self._gini(sample_labels=labels)

        for feature_number in range(rows.shape[1]):
            unique_values = self._get_unique_values(rows[:, feature_number])
            for feature_value in unique_values:
                predicate = Rule(feature_number=feature_number, feature_value=feature_value)

                true_rows, true_labels, false_rows, false_labels = predicate.split(rows, labels)
                if true_rows.shape[0] == 0 or false_rows.shape[0] == 0:
                    continue

                gain = self._information_gain(true_labels, false_labels, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_predicate = gain, predicate

        return best_gain, best_predicate

    def _get_unique_values(self, X_column):
        unique_feature_values = np.sort(np.unique(X_column))
        if unique_feature_values.shape[0] < self.n_splits:
            return unique_feature_values

        selected_values_indexes = np.linspace(0, unique_feature_values.shape[0], num=self.n_splits, endpoint=False,
                                              dtype=int)
        return unique_feature_values[selected_values_indexes]

    def _gini(self, sample_labels):
        def get_label_counts(labels):
            return np.asarray((np.unique(labels, return_counts=True))).T

        impurity = 1
        for label, label_count in get_label_counts(labels=sample_labels):
            label_probability = label_count / sample_labels.shape[0]
            impurity -= label_probability ** 2

        return impurity

    def _information_gain(self, labels_left, labels_right, current_uncertainty):
        p = float(labels_left.shape[0]) / (labels_left.shape[0] + labels_right.shape[0])
        return current_uncertainty - p * self._gini(labels_left) - (1 - p) * self._gini(labels_right)


if __name__ == '__main__':
    dataset = make_classification(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.3)

    tree = DecisionTree(depth=5, n_splits=5).fit(X_train, y_train)
    print(accuracy_score(y_test, tree.predict(X_test)))

import numpy as np


class Tree:
    def __init__(self, max_depth=5, min_samples_split=2, max_features=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.tree = self.grow_tree(X, y)

    def predict(self, X):
        return [self.traverse_tree(x, self.tree) for x in X]

    def traverse_tree(self, x, node):
        if node.value != None:
            return node.value
        if x[node.feature_idx] <= node.treshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def split_error(self, X, feauture_idx, treshold):
        """
        Calculate standart deviation after splitting into 2 groups
        """
        left_idxs, right_idxs = self.split_node(X, feauture_idx, treshold)

        if len(X) == 0 or len(left_idxs) == 0 or len(right_idxs) == 0:
            return 10000

        return len(left_idxs) / len(X) * self.standart_deviation(X[left_idxs], feauture_idx) + len(right_idxs) / len(
            X) * self.standart_deviation(X[right_idxs], feauture_idx)

    def standart_deviation(self, X, feauture_idx):
        """
        Calculate standart deviation
        """
        return np.std(X[:, feauture_idx])

    def split_node(self, X, feauture_idx, treshold):
        """
        Split into 2 parts
        Splitting a dataset means separating a dataset
        into two lists of rows. Once we have the two
        groups, we can then use our standart deviation
        score above to evaluate the cost of the split.
        """
        left_idxs = np.argwhere(X[:, feauture_idx] <= treshold).flatten()
        right_idxs = np.argwhere(X[:, feauture_idx] > treshold).flatten()
        return left_idxs, right_idxs

    def best_criteria(self, X, feature_idxs):
        """
        Find best split

        Loop throw each feature, for each feature loop
        throw each unique value, try each value as a
        treshold, then choose one, with the smallest error
        """
        best_feauture_idx = None
        best_treshold = None
        best_error = None

        for feature_idx in feature_idxs:
            unique_values = np.unique(X[:, feature_idx])
            for treshold in unique_values:
                error = self.split_error(X, feature_idx, treshold)
                if best_error == None or error < best_error:
                    best_feauture_idx = feature_idx
                    best_treshold = treshold
                    best_error = error

        return best_feauture_idx, best_treshold

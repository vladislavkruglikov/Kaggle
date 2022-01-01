import numpy as np

from Tree import Tree
from Node import Node


class DecisionTreeClassifier(Tree):
    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth > self.max_depth or len(X) < self.min_samples_split:
            counts = np.bincount(y)
            return Node(value=np.argmax(counts))

        feature_idxs = np.arange(int(n_features * self.max_features))
        best_feauture_idx, best_treshold = self.best_criteria(X, feature_idxs)
        left_idxs, right_idxs = self.split_node(X, best_feauture_idx, best_treshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            counts = np.bincount(y)
            return Node(value=np.argmax(counts))
        else:
            left = self.grow_tree(X[left_idxs], y[left_idxs], depth + 1)
            right = self.grow_tree(X[right_idxs], y[right_idxs], depth + 1)
            return Node(left=left, right=right, feature_idx=best_feauture_idx, treshold=best_treshold)

class Node:
    """
    Class that will be used for building tree.
    Linked list basically.
    """
    def __init__(self, left=None, right=None, value=None, feature_idx=None, treshold=None):
        self.left = left
        self.right = right
        self.value = value
        self.feature_idx = feature_idx
        self.treshold = treshold
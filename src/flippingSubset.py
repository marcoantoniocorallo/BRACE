'''
    A subclass of torch.utils.data.Subset that allows read and write access
    to the underlying dataset via indexing.

    This class is useful when working with immutable datasets like MNIST,
    where direct label modification is not allowed through Subset.
    FlippingSubset exposes __setitem__ to enable controlled modifications
    (e.g., label flipping for Byzantine clients) by writing changes directly
    to the base dataset using the original indices.
'''

from torch.utils.data import Subset

class FlippingSubset(Subset):
    def __init__(self, subset):
        self.dataset = subset.dataset
        self.indices = subset.indices

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

    def __setitem__(self, idx, value):
        actual_idx = self.indices[idx]
        img, label = value
        self.dataset.data[actual_idx] = img
        self.dataset.targets[actual_idx] = label

    def __len__(self):
        return len(self.indices)
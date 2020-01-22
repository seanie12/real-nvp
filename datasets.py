from sklearn import datasets
import numpy as np
import torch

class DatasetMoons(object):
    def sample(self, n):
        moons = datasets.make_moons(n_samples=n, noise=0.05)[0].astype(np.float32)
        return torch.from_numpy(moons)
    
import random
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torch



class ClassBalancedSampler:

    def __init__(self, dataset, num_per_class, enlarge, replacement):
        '''

        :param dataset:
        :param num_per_class:
        :param enlarge: resize the imb dataset to how many times larger
        '''
        self.per_sample_weight = self.create_weight(dataset, num_per_class)
        self.sampler = WeightedRandomSampler(weights=self.per_sample_weight,
                                             num_samples=int(enlarge*len(dataset.labels)),
                                             replacement=replacement)

    def create_weight(self, dataset, num_per_class):
        lbs_all = dataset.labels
        weights = np.zeros(len(lbs_all), dtype=float)
        for idx in range(len(weights)):
            # inverse num_samples of that lb as the weight
            lb = lbs_all[idx]
            weights[idx] = 1/num_per_class[lb]

        return torch.from_numpy(weights)


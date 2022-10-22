import torch
from torch.utils.data import WeightedRandomSampler

__all__ = ['DistributedSampler', 'UpdatableSubsetRandomSampler', 'UpdatableWeightedRandomSampler']

class DistributedSampler(torch.utils.data.distributed.DistributedSampler):
    NAME_DistributedSampler = "DistributedSampler"

class UpdatableSubsetRandomSampler(WeightedRandomSampler):
    NAME_SubsetRandomSampler = "SubsetRandomSampler"
    
    def update(self, new_indices):
        self.indices = new_indices
    
class UpdatableWeightedRandomSampler(WeightedRandomSampler):
    NAME_WeightedRandomSampler = "WeightedRandomSampler"
    
    def update(self, new_weights):
        self.weights = torch.as_tensor(new_weights, dtype=torch.double)

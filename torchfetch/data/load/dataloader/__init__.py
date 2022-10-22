from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

from .sampler import DistributedSampler, UpdatableSubsetRandomSampler, UpdatableWeightedRandomSampler

__all__ = ['DistributedSampler', 'UpdatableSubsetRandomSampler', 'UpdatableWeightedRandomSampler', 'SamplerUpdatableDataLoader',
           'NAME_DistributedSampler', 'NAME_SubsetRandomSampler', 'NAME_WeightedRandomSampler']


class SamplerUpdatableDataLoader(TorchDataLoader):
    # override SubsetRandomSampler and WeightedRandomSampler to be updatable

    UpdatableRandomSampler = [UpdatableSubsetRandomSampler.NAME_SubsetRandomSampler,
                              UpdatableWeightedRandomSampler.NAME_WeightedRandomSampler]

    def __init__(self, dataset: Dataset, sampler: str, **kwargs):

        if sampler == UpdatableSubsetRandomSampler.NAME_SubsetRandomSampler:
            sampler_cls = UpdatableSubsetRandomSampler(range(len(dataset)))
        elif sampler == UpdatableWeightedRandomSampler.NAME_WeightedRandomSampler:
            sampler_cls = UpdatableWeightedRandomSampler(
                weights=range(len(dataset)), num_samples=len(dataset))

        super(SamplerUpdatableDataLoader, self).__init__(
            dataset=dataset, sampler=sampler_cls, **kwargs)

    def update_sampler(self, val):
        self.sampler.update(val)

NAME_DistributedSampler = DistributedSampler.NAME_DistributedSampler
NAME_SubsetRandomSampler = UpdatableSubsetRandomSampler.NAME_SubsetRandomSampler
NAME_WeightedRandomSampler = UpdatableWeightedRandomSampler.NAME_WeightedRandomSampler

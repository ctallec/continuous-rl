"""Distribution facilities (unused)."""
import torch
from torch import Tensor
from torch.distributions.kl import _kl_normal_normal, register_kl, kl_divergence
from torch.distributions.utils import _sum_rightmost

# thin wrappers around some distributions to get additional
# methods
class Categorical(torch.distributions.categorical.Categorical):
    def __getitem__(self, idxs):
        logits = self.logits[idxs]
        return Categorical(logits=logits)

    def copy(self):
        return Categorical(logits=self.logits.clone().detach())

class Normal(torch.distributions.normal.Normal):
    def __getitem__(self, idxs):
        loc = self.loc[idxs]
        scale = self.scale[idxs]
        return Normal(loc, scale)

    def copy(self):
        loc = self.loc.clone().detach()
        scale = self.scale.clone().detach()
        return Normal(loc, scale)

class Independent(torch.distributions.independent.Independent):
    def __getitem__(self, idxs):
        return Independent(self.base_dist[idxs], self.reinterpreted_batch_ndims)

    def copy(self):
        return Independent(self.base_dist.copy(), self.reinterpreted_batch_ndims)

class DiagonalNormal(Normal):
    @property
    def event_shape(self):
        return super().event_shape[:-1]

    def log_prob(self, value: Tensor):
        logp = Normal.log_prob(self, value)
        return logp.sum(dim=-1)

    def cdf(self, value: Tensor):
        raise NotImplementedError()

    def icdf(self, value: Tensor):
        raise NotImplementedError()

    def entropy(self):
        entropy = Normal.entropy(self)
        return entropy.sum(dim=-1)

@register_kl(DiagonalNormal, DiagonalNormal)
def _kl_diagonalnormal_diagonalnormal(p, q):
    kl = _kl_normal_normal(p, q)
    return kl.sum(dim=-1)

@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)

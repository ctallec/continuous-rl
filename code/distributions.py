"""Distribution facilities (unused)."""
from abc import abstractmethod, ABC

from torch import Tensor
from torch.distributions.normal import Normal as TNormal
from torch.distributions.independent import Independent as TIndependent
from torch.distributions.categorical import Categorical as TCategorical
from torch.distributions.kl import _kl_normal_normal, register_kl, kl_divergence
from torch.distributions.utils import _sum_rightmost

class Distribution(ABC):
    """Torch distribution with additional methods."""
    @abstractmethod
    def log_prob(self, sample: Tensor) -> Tensor:
        pass

    @abstractmethod
    def entropy(self) -> Tensor:
        pass

    @abstractmethod
    def __getitem__(self, idxs) -> "Distribution":
        pass

    @abstractmethod
    def copy(self) -> "Distribution":
        pass

class Normal(Distribution, TNormal):
    def __getitem__(self, idxs) -> "Normal":
        loc = self.loc[idxs]
        scale = self.scale[idxs]
        return Normal(loc, scale)

    def copy(self) -> "Normal":
        loc = self.loc.clone().detach()
        scale = self.scale.clone().detach()
        return Normal(loc, scale)

class Categorical(Distribution, TCategorical):
    def __getitem__(self, idxs) -> "Categorical":
        return Categorical(logits=self.logits[idxs])

    def copy(self) -> "Categorical":
        return Categorical(logits=self.logits.clone().detach())

class Independent(Distribution, TIndependent):
    def __getitem__(self, idxs) -> "Independent":
        return Independent(self.base_dist[idxs],
                           reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)

    def copy(self) -> "Independent":
        return Independent(self.base_dist.copy(),
                           reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)

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

@register_kl(TIndependent, TIndependent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)

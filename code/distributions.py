from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import _kl_normal_normal, register_kl, kl_divergence
from torch.distributions.categorical import Categorical
from torch.distributions.utils import _sum_rightmost

# class DiagonalNormal(Normal):
#     @property
#     def event_shape(self):
#         print(super().event_shape)
#         return super().event_shape[:-1]

#     def log_prob(self, value: Tensor):
#         logp = Normal.log_prob(self, value)
#         return logp.sum(dim=-1)

#     def cdf(self, value: Tensor):
#         raise NotImplementedError()

#     def icdf(self, value: Tensor):
#         raise NotImplementedError()

#     def entropy(self):
#         entropy = Normal.entropy(self)
#         return entropy.sum(dim=-1)

# @register_kl(DiagonalNormal, DiagonalNormal)
# def _kl_diagonalnormal_diagonalnormal(p, q):
#     kl = _kl_normal_normal(p, q)
#     return kl.sum(dim=-1)

@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)


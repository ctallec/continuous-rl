from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.normal import Normal
from torch.distributions.kl import _kl_normal_normal, register_kl
from torch.distributions.categorical import Categorical

class DiagonalNormal(Normal):
    @property
    def event_shape(self):
        print(super().event_shape)
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

def copy_distr(d: Distribution) -> Distribution:
    if isinstance(d, DiagonalNormal):
        return DiagonalNormal(d.loc.clone().detach(), d.scale.clone().detach())

    elif isinstance(d, Categorical):
        return Categorical(logits=d.logits.clone().detach())
    else:
        raise NotImplementedError()

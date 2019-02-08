from torch.distributions.normal import Normal



class DiagonalNormal(Normal):
    def log_prob(self, value):
    	logp = Normal.log_prob(self, value)
    	return logp.sum(dim=-1)

    def cdf(self, value):
    	raise NotImplementedError()

    def icdf(self, value):
    	raise NotImplementedError()

    def entropy(self):
    	entropy = Normal.entropy(self)
    	return entropy.sum(dim=-1)
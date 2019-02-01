from abc import abstractmethod


from abstract import Cudaable, Arrayable, ParametricFunction

class Noise(Cudaable):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def perturb_output(
            self,
            *inputs: Arrayable,
            function: ParametricFunction):
        pass

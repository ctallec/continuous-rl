from abc import abstractmethod


from abstract import Arrayable, ParametricFunction
from cudaable import Cudaable

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

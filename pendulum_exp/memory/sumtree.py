"""Implementation of sumtree for finite distribution sampling."""
import numpy as np
from typing import Optional, Tuple

class SumTree:
    """Implement a sum tree data structure.

    A sum tree is a complete tree where the value of each node
    is the sum of the values of its children.
    """
    def __init__(self, size: int) -> None:
        self._max_size: int = 1
        while self._max_size < size:
            self._max_size *= 2 # make sure the max size is a power of 2
        self._cur_idx: int = 0
        self._write: int = self._max_size - 1
        self._storage: np.ndarray = np.zeros(2 * self._max_size - 1)

    def modify(self, idx: int, priority: float) -> None:
        assert idx < self._max_size
        change = self._storage[self._write + idx] - priority
        self.update(self._write + idx, change)

    @property
    def total(self):
        return self._storage[0]

    def add(self, priority: float) -> None:
        idx = self._cur_idx

        self.modify(idx, priority)

        self._cur_idx = (self._cur_idx + 1) % self._max_size

    def update(self, idx: int, change: float) -> None:
        self._storage[idx] -= change
        if idx != 0:
            parent = (idx - 1) // 2
            self.update(parent, change)

    def sample(self, idx: Optional[int]=None, value: Optional[float]=None) -> Tuple[int, float]:
        if value is None or idx is None:
            assert self._storage[0] > 0
            idx = 0
            return self.sample(idx, np.random.uniform(0, self._storage[0]))
        if idx >= self._max_size - 1:
            return idx - self._write, self._storage[idx]

        left = idx * 2 + 1
        value_left = self._storage[left]
        if value < value_left:
            return self.sample(left, value)
        right = left + 1
        return self.sample(right, value - value_left)


if __name__ == '__main__':
    n = 3
    samples = 100
    st = SumTree(n)
    st.add(2)
    st.add(1)
    st.add(1)
    st.modify(2, 7)
    values, priorities = zip(*[st.sample() for _ in range(samples)])
    print(values, priorities)
    h = [0.] * n
    for i in range(n):
        h[i] = values.count(i) / samples
    print(h)

import numpy as np
import tensorflow.keras as k
from collections import namedtuple

Experience = namedtuple('Experience', ('state', 'direction', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity, batch_size=32):
        self.experiences = []
        self.capacity = capacity
        self.batch_size = batch_size

    def __len__(self):
        return len(self.experiences) // self.batch_size

    def is_full(self):
        return self.experiences is not None and len(self.experiences) >= self.capacity

    def space(self):
        return 0 if self.experiences is None else len(self.experiences) / self.capacity

    def push(self, exp):
        if self.is_full():
            del self.experiences[np.random.randint(len(self.experiences))]
        self.experiences.append(exp)

    def pop(self):
        return [self.experiences.pop(np.random.randint(len(self.experiences))) for _ in range(self.batch_size)]

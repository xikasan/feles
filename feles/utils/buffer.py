# coding: utf-8

import numpy as np
from dataclasses import dataclass


@dataclass
class FELESLog:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reference: np.ndarray

    @staticmethod
    def format(buffer):
        state = np.vstack([log.state for log in buffer]).astype(np.float32)
        action = np.vstack([log.action for log in buffer]).astype(np.float32)
        next_state = np.vstack([log.next_state for log in buffer]).astype(np.float32)
        reference = np.vstack([log.reference for log in buffer]).astype(np.float32)
        return FELESLog(state, action, next_state, reference)


@dataclass
class Experience:
    obs: np.ndarray
    act: np.ndarray
    next_obs: np.ndarray
    reward: float
    done: bool

    @staticmethod
    def format(buffer):
        obs = np.vstack([exp.obs for exp in buffer]).astype(np.float32)
        act = np.vstack([exp.act for exp in buffer]).astype(np.float32)
        next_obs = np.vstack([exp.next_obs for exp in buffer]).astype(np.float32)
        reward = np.asarray([exp.reward for exp in buffer]).astype(np.float32).reshape([-1, 1])
        done = np.asarray([exp.done for exp in buffer]).astype(np.float32).reshape([-1, 1])
        return Experience(obs, act, next_obs, reward, done)


class ReplayBuffer:

    def __init__(self, max_size=int(1e+3)):
        self.max_size = max_size
        self._count = 0
        self._buffer = []

    def add(self, *data):
        if len(data) > 1:
            data = Experience(*data)
        else:
            data = data[0]
        if len(self._buffer) == self.max_size:
            self._buffer[self._count] = data
        else:
            self._buffer.append(data)

        if self._count == self.max_size - 1:
            self._count = 0
        else:
            self._count += 1

    def get_batch(self, size):
        N = len(self._buffer)
        indices = np.random.choice(np.arange(N), replace=False, size=size)
        selected_data = [self._buffer[index] for index in indices]
        print(selected_data)

    @property
    def buffer(self):
        exp = self._buffer[0]
        if len(self._buffer) == self.max_size:
            buffer = sum([
                self._buffer[self._count:],
                self._buffer[:self._count]
            ], [])
        else:
            buffer = self._buffer
        return exp.format(buffer)

    def __len__(self):
        return len(self._buffer)

    def reset(self):
        self._buffer = []
        self._count = 0

    @property
    def is_full(self):
        return len(self) >= self.max_size


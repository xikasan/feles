# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass


@dataclass
class Log:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray

    @staticmethod
    def format(buffer):
        state = np.vstack([exp.state for exp in buffer]).astype(np.float32)
        action = np.vstack([exp.action for exp in buffer]).astype(np.float32)
        next_state = np.vstack([exp.next_state for exp in buffer]).astype(np.float32)
        return Log(state, action, next_state)


class SystemEstimator(nn.Module):

    def __init__(
            self,
            dim_state: int,
            dim_action: int,
            lr: float = 1e-3,
            dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.lstate = nn.Linear(
            in_features=dim_state+dim_action,
            out_features=dim_state,
            bias=False
        )
        nn.init.zeros_(self.lstate.weight)
        self.dtype = dtype

        # self.opt = O.Adam(self.parameters(), lr=lr)
        self.opt = optim.SGD(self.parameters(), lr=lr)

    def reset(self) -> None:
        nn.init.zeros_(self.lstate.weight)

    def forward(self, x: list) -> torch.Tensor:
        x = torch.cat(x, dim=1)
        x = self.lstate(x)
        return x

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = torch.tensor(state, dtype=self.dtype).reshape(1, state.shape[0])
        u = torch.tensor(action, dtype=self.dtype).reshape(1, action.shape[0])
        pre = self([x, u])
        pre = torch.squeeze(pre).detach().numpy()
        return pre

    def predicts(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = torch.tensor(state, dtype=self.dtype)
        u = torch.tensor(action, dtype=self.dtype)
        pre = self([x, u])
        pre = pre.detach().numpy()
        return pre

    def update(
            self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray
    ) -> np.ndarray:
        x = torch.tensor(state, dtype=self.dtype)
        u = torch.tensor(action, dtype=self.dtype)
        y = torch.tensor(next_state, dtype=self.dtype)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            u = u.reshape(1, -1)
            y = y.reshape(1, -1)

        pre = self([x, u])
        loss = 0.5 * (y - pre) ** 2
        loss = torch.sum(loss)

        self.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.detach().numpy()

    @property
    def weights(self):
        weights = [w.data.detach().numpy() for w in self.parameters()]
        return weights


class SystemEstimator2garbage(nn.Module):

    def __init__(
            self,
            dim_state: int,
            dim_action: int,
            lr: float
    ):
        super().__init__()
        dim_hidden = 2 * (dim_state + dim_action)
        dim_params = dim_state * (dim_state + dim_action)
        self.dim_state  = dim_state
        self.dim_action = dim_action
        self.lx = nn.Linear(dim_state, dim_state)
        self.lu = nn.Linear(dim_action, dim_action)
        self.lh = nn.Linear(dim_hidden, dim_hidden)
        self.lo = nn.Linear(dim_hidden,  dim_params)

    def forward(self, x):
        x, u = x
        print("print in forward de SE2")
        featx = self.lx(x)
        featu = self.lu(u)
        feat = torch.cat([featx, featu, x, u], axis=1)
        print("feat1:\n", feat)
        feat = self.lh(feat)
        print("feat2:\n", feat)
        feat = self.lo(feat)
        print("feat3:\n", feat)
        feat_A = feat[..., :self.dim_state**2]
        print("feat_A:", feat_A.shape, "\n", feat_A)

    def predict(self, state: np.ndarray, action: np.ndarray):
        state = state if len(state.shape) > 1 else state.reshape((1, state.shape[0]))
        action = action if len(action.shape) > 1 else action.reshape((1, action.shape[0]))
        x = torch.tensor(state).float()
        u = torch.tensor(action).float()
        self([x, u])


class SystemEstimator3(nn.Module):

    def __init__(self, dim_state, dim_action, lr):
        super().__init__()
        self.N = dim_state
        self.M = dim_action
        self.AB = nn.Linear(self.N+self.M, self.N, bias=None)
        nn.init.zeros_(self.AB.weight)
        self.lr = lr
        self.opt = optim.SGD(self.AB.parameters(), self.lr)
        # self.opt = optim.Adam(self.AB.parameters(), self.lr)
        self.max_state = np.deg2rad(5.)
        self.normalize = lambda x: x / self.max_state
        self.denormalize = lambda x: x * self.max_state

    def reset(self) -> None:
        nn.init.zeros_(self.AB.weight)

    def forward(self, x):
        feat = torch.cat(x, axis=1)
        feat = self.AB(feat)
        return feat

    def predict(self, x, u):
        assert isinstance(x, np.ndarray) and len(x.shape) <= 1
        assert isinstance(u, np.ndarray) and len(u.shape) <= 1
        x = torch.from_numpy(x).reshape((1, -1)).float()
        u = torch.from_numpy(u).reshape((1, -1)).float()
        x = self.normalize(x)
        u = self.normalize(u)
        y = self([x, u])
        y = torch.squeeze(y).detach().numpy()
        y = self.denormalize(y)
        return y

    def predicts(self, x, u):
        x = torch.from_numpy(x).float()
        u = torch.from_numpy(u).float()
        x = self.normalize(x)
        u = self.normalize(u)
        y = self([x, u])
        y = y.detach().numpy()
        y = self.denormalize(y)
        return y

    def update(
            self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            factor: np.ndarray or float = None
    ) -> np.ndarray:
        x = torch.tensor(state).float()
        u = torch.tensor(action).float()
        y = torch.tensor(next_state).float()
        x = self.normalize(x)
        u = self.normalize(u)
        y = self.normalize(y)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            u = u.reshape(1, -1)
            y = y.reshape(1, -1)
        if factor is None:
            factor = 1.
        factor = torch.tensor(factor)
        pre = self([x, u])
        loss = 0.5 * (y - pre) ** 2
        loss = torch.sum(loss, dim=1)
        loss = factor * loss
        loss = torch.mean(loss)
        # loss = torch.clip(loss, 0, 0.1)
        # print(loss)

        self.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.detach().numpy()

    def A(self):
        w = self.W().numpy()
        return w[..., :self.N]

    def B(self):
        w = self.W().numpy()
        return w[..., self.N:]
        return 1

    @property
    def W(self):
        return self.AB.weight.data.detach()

    @staticmethod
    def builder(cf, env):
        dim_state = len(env.observation_space.high) if not hasattr(cf, "dim_state") else cf.dim_state
        dim_action = len(env.action_space.high) if not hasattr(cf, "dim_action") else cf.dim_action
        return SystemEstimator3(dim_state, dim_action, lr=cf.lr)


class NLCSystemEstimator(nn.Module):

    def __init__(self, dim_state, dim_action, units, lr_ssl, lr_nlc=None):
        super().__init__()
        self.N = dim_state
        self.M = dim_action
        self.ssl = nn.Linear(self.N+self.M, self.N, bias=False)
        self.nlc = nn.Sequential(
            nn.Linear(self.N+self.M, units[0], bias=False),
            nn.ReLU(),
            nn.Linear(units[0], units[1], bias=False),
            nn.ReLU(),
            nn.Linear(units[1], self.N, bias=False)
        )
        if lr_nlc is None:
            lr_nlc = lr_ssl * 0.1
        # self.opt_ssl = optim.Adam(self.ssl.parameters(), lr=lr_ssl)
        # self.opt_nlc = optim.Adam(self.nlc.parameters(), lr=lr_nlc)
        self.opt_ssl = optim.SGD(self.ssl.parameters(), lr=lr_ssl)
        self.opt_nlc = optim.SGD(self.nlc.parameters(), lr=lr_nlc)

        self.max_state = np.deg2rad(5.)
        self.normalize = lambda x: x / self.max_state
        self.denormalize = lambda x: x * self.max_state

        self.dtype = torch.float32

    def reset(self) -> None:
        nn.init.zeros_(self.ssl.weight)
        nn.init.normal_(self.nlc[0].weight, 0., 0.01)
        nn.init.normal_(self.nlc[2].weight, 0., 0.01)
        nn.init.zeros_(self.nlc[4].weight)

    def forward(self, x):
        feat = torch.cat(x, dim=1)
        feat_ssl = self.ssl(feat)
        feat_nlc = self.nlc(feat)
        feat = feat_ssl + feat_nlc
        return feat

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray) and len(x.shape) <= 1
        assert isinstance(u, np.ndarray) and len(u.shape) <= 1
        x = torch.from_numpy(x).reshape((1, -1)).float()
        u = torch.from_numpy(u).reshape((1, -1)).float()
        x = self.normalize(x)
        u = self.normalize(u)
        y = self([x, u])
        y = torch.squeeze(y).detach().numpy()
        y = self.denormalize(y)
        return y

    def predicts(self, x, u):
        x = torch.from_numpy(x).float()
        u = torch.from_numpy(u).float()
        x = self.normalize(x)
        u = self.normalize(u)
        y = self([x, u])
        y = y.detach().numpy()
        y = self.denormalize(y)
        return y

    def update(
            self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            factor: np.ndarray or float = None
    ) -> float:
        x = torch.tensor(state).float()
        u = torch.tensor(action).float()
        y = torch.tensor(next_state).float()
        x = self.normalize(x)
        u = self.normalize(u)
        y = self.normalize(y)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            u = u.reshape(1, -1)
            y = y.reshape(1, -1)
        if hasattr(factor, "__len__"):
            f = np.asanyarray(factor)
            f = torch.tensor(f, dtype=self.dtype)
        else:
            f = torch.tensor(
                1 if factor is None
                else [[factor ** k] for k in range(x.shape[0])]
            )
        pre = self([x, u])
        loss = 0.5 * (y - pre) ** 2
        loss = f * loss
        # loss = torch.sum(loss)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        self.zero_grad()
        loss.backward()
        self.opt_ssl.step()
        self.opt_nlc.step()

        return loss.detach().numpy()

    @property
    def W(self):
        return self.ssl.weight.data.detach()

    @staticmethod
    def builder(cf, env):
        dim_state = len(env.observation_space.high) if not hasattr(cf, "dim_state") else cf.dim_state
        dim_action = len(env.action_space.high) if not hasattr(cf, "dim_action") else cf.dim_action
        lr_ssl = cf.lr if not hasattr(cf, "lr_ssl") else cf.lr_ssl
        lr_nlc = None if not hasattr(cf, "lr_nlc") else cf.lr_nlc
        return NLCSystemEstimator(dim_state, dim_action, cf.units, lr_ssl=lr_ssl, lr_nlc=lr_nlc)


SYE = dict(
    SystemEstimator=SystemEstimator3,
    Linear=SystemEstimator3,
    linear=SystemEstimator3,
    NLCSystemEstimator=NLCSystemEstimator,
    NonLinear=NLCSystemEstimator,
    nonlinear=NLCSystemEstimator,
)


if __name__ == '__main__':
    dim_obs = 2
    dim_act = 1
    se = SystemEstimator(dim_obs, dim_act)
    se.reset()

    from tqdm import tqdm
    for i in tqdm(range(10000)):
        obs = np.random.rand(5, dim_obs)
        act = np.random.rand(5, dim_act)
        loss = se.update(obs, act, obs * 2 + 0.5 * act)
        # if ((i+1) % 100) == 0:
        #     print("{:3}:".format(i), loss)

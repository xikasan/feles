# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from dataclasses import dataclass


@dataclass
class Log:
    reference: np.ndarray
    action: np.ndarray

    @staticmethod
    def format(buffer):
        reference = np.vstack([log.reference for log in buffer]).astype(np.float32)
        action = np.vstack([log.action for log in buffer]).astype(np.float32)
        return Log(reference, action)


class LinearFEL(nn.Module):

    def __init__(self, dim_ref, dim_act, max_act, lr=1e-3, dtype=torch.float32):
        super().__init__()
        self.lout = nn.Linear(
            in_features=dim_ref,
            out_features=dim_act,
            bias=False
        )
        nn.init.zeros_(self.lout.weight)
        self.max_act = torch.tensor(max_act, dtype=dtype)
        self.dtype = dtype

        # self.opt = O.Adam(self.parameters(), lr=lr)
        self.opt = O.SGD(self.parameters(), lr=lr)

        self.normalize = lambda x: x / max_act

    def reset(self):
        nn.init.zeros_(self.lout.weight)

    def forward(self, x):
        x = self.lout(x)
        # x = torch.tanh(x)
        # x = x * self.max_act
        return x

    def get_action(self, ref):
        ref = torch.tensor(ref, dtype=self.dtype)
        ref = ref.reshape(1, ref.shape[0])
        ref = self.normalize(ref)
        act = self(ref)
        act = torch.squeeze(act).detach().numpy()
        return act

    def get_actions(self, refs):
        refs = torch.tensor(refs, dtype=self.dtype)
        refs = self.normalize(refs)
        acts = self(refs)
        acts = acts.detach().numpy()
        return acts

    def update(self, ref, act, factor=None):
        ref = torch.tensor(ref, dtype=self.dtype)
        ref = self.normalize(ref)
        if len(ref.shape) == 1:
            ref = ref.unsqueeze(0)
        act = torch.tensor(act, dtype=self.dtype)
        act = self.normalize(act)
        if len(act.shape) == 1:
            act = act.unsqueeze(0)
        if hasattr(factor, "__len__"):
            f = np.asanyarray(factor)
            f = torch.tensor(f, dtype=self.dtype)
        else:
            f = torch.tensor(
                1 if factor is None
                else [[factor ** k] for k in range(ref.shape[0])]
            )

        pre = self(ref)
        pre = self.normalize(pre)
        loss = 0.5 * f * (act - pre) ** 2
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        self.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.detach().numpy()

    @property
    def weights(self):
        weights = [w.data.detach().numpy() for w in self.parameters()]
        return weights

    @staticmethod
    def builder(cf, env, max_act=None):
        dim_reference = len(env.target) if not hasattr(cf, "dim_reference") else cf.dim_reference
        dim_action = len(env.action_space.high) if not hasattr(cf, "dim_action") else cf.dim_action
        max_act = max_act if max_act is not None else env.action_space.high
        return LinearFEL(
            dim_reference,
            dim_action,
            max_act,
            lr=cf.lr,
        )


class DeepFEL(LinearFEL):

    def __init__(
            self,
            units,
            dim_ref,
            dim_act,
            max_act,
            lr=1e-3,
            dtype=torch.float32
    ):
        super().__init__(dim_ref, dim_act, max_act, dtype=dtype)
        self.l1 = nn.Linear(dim_ref, units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.lout = nn.Linear(units[1], dim_act, bias=False)
        self.max_act = torch.tensor(max_act, dtype=dtype)
        self.dim_ref = dim_ref

        self.lr = torch.tensor(lr, dtype=dtype)
        # self.opt = O.Adam(self.parameters(), lr=lr)
        self.opt = O.SGD(self.parameters(), lr=lr)

        self.normalize = lambda x: x / max_act

    def reset(self):
        super().reset()
        nn.init.normal_(self.l1.weight)
        nn.init.normal_(self.l2.weight)

    def forward(self, x):
        feat = self.l1(x)
        feat = torch.relu(feat)
        feat = self.l2(feat)
        feat = torch.relu(feat)
        feat = self.lout(feat)
        feat = torch.tanh(feat)
        feat = feat * self.max_act
        return feat

    def get_action(self, ref):
        ref = torch.tensor(ref, dtype=self.dtype)
        ref = ref.reshape(1, ref.shape[0])
        ref = self.normalize(ref)
        act = self(ref)
        act = torch.squeeze(act).detach().numpy()
        return act

    def get_actions(self, refs):
        refs = torch.tensor(refs, dtype=self.dtype)
        refs = self.normalize(refs)
        acts = self(refs)
        acts = torch.squeeze(acts).detach().numpy()
        return acts

    def update(self, ref, act, factor=None):
        ref = torch.tensor(ref, dtype=self.dtype)
        ref = self.normalize(ref)
        if len(ref.shape) == 1:
            ref = ref.unsqueeze(0)
        act = torch.tensor(act, dtype=self.dtype)
        act = self.normalize(act)
        if len(act.shape) == 1:
            act = act.unsqueeze(0)
        if hasattr(factor, "__len__"):
            f = np.asanyarray(factor)
            f = torch.tensor(f, dtype=self.dtype)
        else:
            f = torch.tensor(
                1 if factor is None
                else [[factor ** k] for k in range(ref.shape[0])]
            )

        pre = self(ref)
        pre = self.normalize(pre)
        loss = 0.5 * f * (act - pre) ** 2
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        self.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.detach().numpy()

    @property
    def weights(self):
        return np.zeros_like(self.dim_ref)

    @staticmethod
    def builder(cf, env, max_act=None):
        dim_reference = len(env.target) if not hasattr(cf, "dim_reference") else cf.dim_reference
        dim_action = len(env.action_space.high) if not hasattr(cf, "dim_action") else cf.dim_action
        max_act = max_act if max_act is not None else env.action_space.high
        return DeepFEL(
            cf.units,
            dim_reference,
            dim_action,
            max_act,
            lr=cf.lr,
        )


class NLCFEL(DeepFEL):

    def __init__(
            self,
            units,
            dim_ref,
            dim_act,
            max_act,
            lr_lin=1e-3,
            lr_nlc=1e-4,
            dtype=torch.float32
    ):
        super().__init__(units, dim_ref, dim_act, max_act, lr=lr_lin)
        self.lin = nn.Linear(dim_ref, dim_act, bias=False)
        self.nlc = nn.Sequential(
            self.l1, nn.ReLU(), self.l2, nn.ReLU(), self.lout
        )

        self.lr_lin = torch.tensor(lr_lin).float()
        self.lr_nlc = torch.tensor(lr_nlc).float()
        self.opt_lin = O.SGD(self.lin.parameters(), lr=self.lr_lin)
        self.opt_nlc = O.SGD(self.nlc.parameters(), lr=self.lr_nlc)

        self.normalize = lambda x: x / max_act

    def reset(self):
        super().reset()
        nn.init.zeros_(self.lin.weight)
        nn.init.normal_(self.l1.weight, 0, 0.01)
        nn.init.normal_(self.l2.weight, 0, 0.01)
        nn.init.zeros_(self.lout.weight)

    def forward(self, x):
        feat_lin = self.lin(x)
        feat_nlc = self.nlc(x)
        feat = torch.tanh(feat_lin) + torch.tanh(feat_nlc) * 0.1
        feat = torch.tanh(feat)
        feat = feat * self.max_act
        return feat

    def update(self, ref, act, factor=None):
        ref = torch.tensor(ref, dtype=self.dtype)
        ref = self.normalize(ref)
        if len(ref.shape) == 1:
            ref = ref.unsqueeze(0)
        act = torch.tensor(act, dtype=self.dtype)
        act = self.normalize(act)
        if len(act.shape) == 1:
            act = act.unsqueeze(0)
        if hasattr(factor, "__len__"):
            f = np.asanyarray(factor)
            f = torch.tensor(f, dtype=self.dtype)
        else:
            f = torch.tensor(
                1 if factor is None
                else [[factor ** k] for k in range(ref.shape[0])]
            )

        # update linear block
        loss1 = self._loss(ref, act, f)
        self.zero_grad()
        loss1.backward()
        self.opt_lin.step()

        # update non-linear block
        loss2 = self._loss(ref, act, f)
        self.zero_grad()
        loss2.backward()
        self.opt_nlc.step()

        loss = loss1 + loss2
        return loss.detach().numpy(), loss1.detach().numpy(), loss2.detach().numpy()
        # loss = loss1
        # return loss.detach().numpy(), None, None

    def _loss(self, ref, act, f):
        pre = self(ref)
        pre = self.normalize(pre)
        loss = 0.5 * f * (act - pre) ** 2
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)
        return loss

    @property
    def weights(self):
        weights = [w.data.detach().numpy() for w in self.lin.parameters()]
        return weights

    @staticmethod
    def builder(cf, env, max_act=None):
        dim_reference = len(env.target) if not hasattr(cf, "dim_reference") else cf.dim_reference
        dim_action = len(env.action_space.high) if not hasattr(cf, "dim_action") else cf.dim_action
        max_act = max_act if max_act is not None else env.action_space.high
        lr_lin = cf.lr if not hasattr(cf, "li_lin") else cf.lr_lin
        lr_nlc = cf.lr if not hasattr(cf, "li_nlc") else cf.lr_nlc
        return NLCFEL(
            cf.units,
            dim_reference,
            dim_action,
            max_act,
            lr_lin=lr_lin,
            lr_nlc=lr_nlc
        )



class FEL:

    TYPE = dict(
        linearfel=LinearFEL,
        linear=LinearFEL,
        simple=LinearFEL,
        nonlinear=DeepFEL,
        deep=DeepFEL,
        deepfel=DeepFEL,
        nlc=NLCFEL,
    )

    @staticmethod
    def select(type_):
        assert isinstance(type_, str)
        type_ = type_.lower()
        assert type_ in FEL.TYPE.keys()
        return FEL.TYPE[type_]

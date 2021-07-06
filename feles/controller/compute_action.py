# coding: utf-8

import numpy as np


def fbc_full(fbc, r, x):
    r = r[:len(x)]
    e = r - x
    u = fbc(e)
    u = np.asanyarray(u)
    return u


def fbc(fbc, r, x):
    e = -1 * x
    e[0] += r[0]
    u = fbc(e)
    u = np.asanyarray(u)
    return u


def fel(fel, r):
    u = fel.get_action(r)
    u = np.asanyarray(u)
    return u


def build_action(env, u):
    act = np.array([0., u]).astype(float)
    act = np.clip(act, env.action_space.low, env.action_space.high)
    return act

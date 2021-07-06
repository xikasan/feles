# coding: utf-8

import xair
import gym
import xtools as xt
import xsim
import numpy as np
import torch
from feles.controller import fel
from feles.controller import sfb
from feles.controller import se


#===================================================
# env
def build_env(cf):
    is_fail = hasattr(cf, "fail")
    fmode = "normal" if not is_fail else cf.fail.mode
    frange = 1 if not is_fail else cf.fail.value
    env = gym.make(
        cf.name, dt=cf.dt,
        target_range=xt.d2r(cf.reference.range),
        target_period=cf.reference.interval,
        fail_mode=fmode,
        fail_range=frange
    )
    env.reset()
    return env


#===================================================
# controller
def build_sfb(cf, env, as_func=True, dtype=np.float32):
    Q = np.diag(cf.Q).astype(dtype)
    R = np.diag(cf.R).astype(dtype)
    K = sfb.compute_sfb_gain(env, Q, R)
    K = np.squeeze(K)
    if not as_func:
        return K
    return lambda x: np.dot(K, x)


def build_fel_(cf, env, dtype=torch.float32):
    dim_act = 1
    dim_ref = env.reference.shape[0]
    max_act = env.action_space.high[env.IX_de]
    return fel.FEL(dim_ref, dim_act, max_act, lr=cf.lr)


def build_dfel(cf, env, dtype=torch.float32):
    dim_act = 1
    dim_ref = env.reference.shape[0]
    max_act = env.action_space.high[env.IX_de]
    return fel.DeepFEL(cf.units, dim_ref, dim_act, max_act, lr=cf.lr)


def build_fel(cf, env):
    max_act = env.action_space.high[env.IX_de]
    return fel.FEL.select(cf.type).builder(cf, env, max_act=max_act)


def build_sye_(cf, env, act_dim=None):
    dim_state = len(env.state)
    dim_action = act_dim if act_dim is not None else len(env.action_space.low)
    sye = se.SystemEstimator3(dim_state, dim_action, cf.lr)
    return sye


def build_decay_queue(length, factor):
    return factor ** np.arange(length).reshape((-1, 1))


def build_sye(cf, env):
    return se.SYE[cf.type].builder(cf, env)


#===================================================
# reference
def build_reference_queue(cf, at_random=True):
    reference_width = (np.max(cf.reference.range) - np.min(cf.reference.range)) / 2
    reference_width = xt.d2r(reference_width) * 2
    if at_random:
        ref = xsim.PoissonRectangularCommand(
            max_amplitude=reference_width,
            interval=cf.reference.interval
        )
        ref.reset()
    else:
        ref = xsim.RectangularCommand(
            cf.reference.interval,
            amplitude=reference_width / 2,
        )
    ref_filter = xsim.Filter2nd(cf.dt, cf.reference.tau)
    ref_filter.reset()

    def generate_full_state_reference(t):
        ref_filter(ref(t))
        return ref_filter.get_full_state()

    ref = {
        time: generate_full_state_reference(time)
        for time in xsim.generate_step_time(cf.due, cf.dt)
    }
    return ref

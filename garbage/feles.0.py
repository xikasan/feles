# coding: utf-8

import numpy as np
import xsim
import xair
import gym
import xtools as xt
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from feles.controller import sfb
from feles.controller import fel
from feles.controller import se

DEFAULT_CONFIG = "configs/feles_no_fail.yaml"
COLORS = xt.Config(dict(
    ref="black",
    sfb="forestgreen",
    fel="navy"
))


def run(config):
    xt.info("run feles")
    cf = xt.Config(config)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # plant
    #
    ecf = cf.env
    env = gym.make(
        ecf.name, dt=ecf.dt,
        target_range=xt.d2r(ecf.reference.range),
        target_period=ecf.reference.interval
    )
    env.reset()
    xt.info("env", env)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # controllers
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # state F/B
    #
    scf = cf.controller.sfb
    Q = np.diag(scf.Q).astype(env.dtype)
    R = np.diag(scf.R).astype(env.dtype)
    K = sfb.compute_sfb_gain(env, Q, R).T
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # FEL
    #
    fel = build_fel(cf.controller.fel, env)
    xt.info("fel", fel)
    sye = build_system_estimator(cf.controller.se, env)
    xt.info("sye", sye)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # reference generator
    #
    ref = build_reference_queue(ecf)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # run simulations
    #
    xt.info("simulation mode", cf.mode)
    res_fls = run_feles(cf, env, ref, fel, K, sye)

    plot(env, res_fls)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# simulation loop
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# feles fel with enhanced sampling
#
def run_feles(cf, env, ref, fel, K, sye):
    env.reset()
    fel.reset()
    sye.reset()
    log = xsim.Logger()
    rbf = xt.Config(dict(size=10, r=[], u=[]))
    past_x, past_u = None, None
    for time, r in tqdm(ref.items()):
        act_sfb = compute_sfb_action(env, K, r[:2])
        act_fel = compute_fel_action(env, fel, r)
        act = build_action(env, act_sfb, act_fel)
        # system estimator update
        pre = np.zeros_like(env.state) if past_x is None else sye.predict(past_x, past_u)
        loss_sye = 0 if past_x is None else sye.update(past_x, past_u, env.state)
        # fel controller update
        if time >= 10:
            run_online_simulation(cf, sye, fel, K)
        # exit()
        update_real_buffer(rbf, r, [act[env.IX_de]])
        loss_fel = fel.update(rbf.r, rbf.u, 0.9)
        log.store(
            time=time,
            x=env.state,
            act_sfb=act_sfb,
            act_fel=act_fel,
            u=act,
            loss_fel=loss_fel,
            pre=pre,
            loss_sye=loss_sye,
            r=r
        ).flush()

        env.step(act)
        past_x = env.state
        past_u = act_sfb

    return xsim.Retriever(log)


def run_online_simulation(cf, sye, fel, K):
    fcf = cf.controller.feles
    for rep in range(fcf.online_sim.num_repeat):
        x = xt.bounded_normal(2) * np.array(xt.d2r([3, 1]))
        log = xsim.Logger()
        ref = build_reference_queue(cf.env, fcf.online_sim.due_time)
        sim_env = xt.Config(dict(state=None, target=None))
        for time, r in ref.items():
            sim_env.state = x
            sim_env.target = r
            act_sfb = compute_sfb_action(sim_env, K, r[:2])
            act = act_sfb
            log.store(time=time, x=x, r=r, u=act).flush()
            x = sye.predict(x, np.array(act))
        res = xsim.Retriever(log)
        res = pd.DataFrame(dict(
            time=res.time(),
            pitch=res.x(0, xt.r2d),
            reference=res.r(0, xt.r2d)
        ))
        res.plot(x="time", y=["reference", "pitch"])
        plt.show()
        exit()


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# feles update
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# update real buffer
#
def update_real_buffer(rbf, ref, act):
    rbf.r.insert(0, ref)
    rbf.u.insert(0, act)
    if len(rbf.r) > rbf.size:
        del rbf.r[-1]
        del rbf.u[-1]


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# compute action
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
def compute_sfb_action(env, K, r=None):
    x = env.state
    r = env.target[:2] if r is None else r
    e = r - x
    u = e.dot(K)
    return u


def compute_fel_action(env, fel, r=None):
    r = env.target if r is None else r
    u = fel.get_action(r)
    return u


def build_action(env, act_sfb, act_fel):
    act = act_sfb + act_fel
    act = np.hstack([[0], act])
    act = np.clip(act, env.action_space.low, env.action_space.high)
    act = act.astype(env.dtype)
    return act


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# before starting simulation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FEL
#
def build_fel(cf, env):
    dim_act = 1
    dim_ref = env.reference.shape[0]
    max_act = env.action_space.high[env.IX_de]
    fel_type = "simple" if not hasattr(cf, "type") else cf.type

    return fel.FEL(dim_ref, dim_act, max_act, lr=cf.lr)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# SE
#
def build_system_estimator(cf: xt.Config, env: gym.Env) -> se.SystemEstimator:
    dim_action = 1
    dim_state = env.state.shape[0]

    return se.SystemEstimator(dim_state, dim_action, lr=cf.lr)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# reference
#
def build_reference_queue(cf, due=None):
    ecf = cf
    reference_width = (np.max(ecf.reference.range) - np.min(ecf.reference.range)) / 2
    reference_width = xt.d2r(reference_width) * 2
    ref = xsim.PoissonRectangularCommand(
        max_amplitude=reference_width,
        interval=ecf.reference.interval
    )
    ref.reset()
    ref_filter = xsim.Filter2nd(ecf.dt, ecf.reference.tau)
    ref_filter.reset()
    xt.info("reference generator", ref)

    def generate_full_state_reference(t):
        ref_filter(ref(t))
        return ref_filter.get_full_state()

    due = ecf.due if due is None else due
    ref = {
        time: generate_full_state_reference(time)
        for time in xsim.generate_step_time(due, ecf.dt)
    }
    return ref


def plot(env, res_fls):
    res_pitch = pd.DataFrame(dict(
        time=res_fls.time(),
        pitch=res_fls.x(idx=env.IX_T, fn=xt.r2d),
        reference=res_fls.r(idx=env.IX_T, fn=xt.r2d),
        predict=res_fls.pre(idx=env.IX_T, fn=xt.r2d)
    ))
    res_act = pd.DataFrame(dict(
        time=res_fls.time(),
        FB=res_fls.act_sfb(fn=xt.r2d),
        FEL=res_fls.act_fel(fn=xt.r2d),
        u=res_fls.u(env.IX_de, fn=xt.r2d)
    ))
    print(res_act)

    fix, axes = plt.subplots(nrows=2, sharex=True)
    res_pitch.plot(x="time", y=["reference", "pitch", "predict"], ax=axes[0])
    res_act.plot(x="time", y=["FB", "FEL", "u"], ax=axes[1])
    plt.show()


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = prepare_args()

    run(args.config)

# coding: utf-8

import xsim
import xair
import gym
import torch as tc
import numpy as np
import xtools as xt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from feles.controller import sfb
from feles.controller import se

DEFAULT_CONFIG = "configs/se_no_fail.yaml"


def run(config):
    xt.info("run system estimation")
    cf = xt.Config(config)

    # plant
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
    # system estimator
    #
    sye = build_system_estimator(cf.controller.se, env)
    if hasattr(cf.controller.se, "params"):
        sye.load_state_dict(tc.load(cf.controller.se.params))
    xt.info("sye", sye)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # reference generator
    #
    ref = build_reference_queue(ecf)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # run simulation
    #
    xt.info("simulation loop")
    res = run_sye(env, ref, sye, K)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # post-simulation
    #
    fig = plot(cf, res, env)

    scf = cf.result.save
    save_types = [save_type for save_type, val in scf.dtype._cf.items() if val]
    if len(save_types) > 0:
        xt.makedirs(scf.path, exist_ok=True)
        save_path = xt.make_dirs_current_time(scf.path, exist_ok=True)
        if scf.dtype.model:
            tc.save(sye.state_dict(), save_path + "/system_estimator.pt")
        if scf.dtype.csv:
            data = make_result_data(res, env)
            data.to_csv(save_path + "/data.csv", index=False)
        if scf.dtype.fig:
            fig.savefig(save_path + "/result.png")


def run_sye(env, ref, sye, K):
    env.reset()
    log = xsim.Logger()
    past_x, past_u = None, None
    for time, r in tqdm(ref.items()):
        act_sfb = compute_sfb_action(env, K, r[:2])
        act = build_action(env, act_sfb, 0)
        pre = np.zeros_like(env.state) if past_x is None else sye.predict(past_x, past_u)
        loss = 0 if past_x is None else sye.update(past_x, past_u, env.state)
        log.store(
            time=time,
            x=env.state,
            u=act,
            r=r,
            pre=pre,
            loss=loss,
            weight=sye.weights[0][0, :]
        ).flush()

        env.step(act)
        past_x = env.state
        past_u = act_sfb

    return xsim.Retriever(log)


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


def build_action(env, act_sfb, act_fel):
    act = act_sfb + act_fel
    act = np.hstack([[0], act])
    act = np.clip(act, env.action_space.low, env.action_space.high)
    act = act.astype(env.dtype)
    return act


def build_system_estimator(cf: xt.Config, env: gym.Env) -> se.SystemEstimator:
    dim_action = 1
    dim_state = env.state.shape[0]

    return se.SystemEstimator(dim_state, dim_action, lr=cf.lr)


def build_reference_queue(cf):
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

    ref = {
        time: generate_full_state_reference(time)
        for time in xsim.generate_step_time(ecf.due, ecf.dt)
    }
    return ref


def plot(cf, res, env):
    fcf = cf.result.fig
    dtypes = [dtype for dtype, val in fcf.dtype._cf.items() if val]
    nrows = len(dtypes)
    fig, axes = plt.subplots(nrows=nrows, sharex=True)
    if nrows == 1:
        axes = [axes]

    for dtype, ax in zip(dtypes, axes):
        PLOTTER_LIST[dtype](cf, res, ax, env)

    if fcf.show:
        plt.show()

    return fig


def plot_pitch(cf, res, ax, env):
    rrange = cf.env.reference.range
    trange = [0, cf.env.due]

    data = pd.DataFrame(dict(
        time=res.time(),
        reference=res.r(0, xt.r2d),
        observation=res.x(env.IX_T, xt.r2d),
        predict=res.pre(0, xt.r2d)
    ))

    data.plot(
        x="time", y=["reference", "observation", "predict"], ax=ax,
        xlim=trange, ylim=rrange, grid=True
    )


def plot_loss(cf, res, ax, env):
    trange = [0, cf.env.due]

    data = pd.DataFrame(dict(
        time=res.time(),
        loss=res.loss()
    ))

    data.plot(
        x="time", y="loss", ax=ax,
        xlim=trange, grid=True
    )


def plot_weight(cf, res, ax, env):
    wrange = None
    trange = [0, cf.env.due]

    data = pd.DataFrame(dict(
        time=res.time(),
        WT=res.weight(0),
        Wq=res.weight(1),
        Wde=res.weight(2)
    ))

    data.plot(
        x="time", y=["WT", "Wq", "Wde"], ax=ax,
        xlim=trange, grid=True
    )


def make_result_data(res, env):
    res = pd.DataFrame(dict(
        time=res.time(),
        reference=res.r(0, xt.r2d),
        pitch=res.x(env.IX_T, xt.r2d),
        predict=res.pre(0, xt.r2d),
        pitch_speed=res.x(env.IX_q, xt.r2d),
        action=res.u(env.IX_de, xt.r2d),
        loss=res.loss(),
        WT=res.weight(0),
        Wq=res.weight(1),
        Wde=res.weight(2)
    ))
    return res



PLOTTER_LIST = {
    "pitch": plot_pitch,
    "loss": plot_loss,
    "weight": plot_weight
}


if __name__ == '__main__':
    import sys
    args = sys.argv

    # pour dev
    args.append(DEFAULT_CONFIG)

    # config file setting
    config_file = DEFAULT_CONFIG if len(args) < 2 else args[1]

    run(config_file)

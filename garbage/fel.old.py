# coding: utf-8

import xsim
import xair
import gym
import numpy as np
import xtools as xt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from feles.controller import sfb
from feles.controller import fel

DEFAULT_CONFIG = "configs/fel_no_fail.yaml"
COLORS = xt.Config(dict(
    ref="black",
    sfb="forestgreen",
    fel="navy"
))


def run(config):
    xt.info("run fel control")
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
    # FEL
    #
    fel = build_fel(cf.controller.fel, env)
    xt.info("fel", fel)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # reference generator
    #
    ref = build_reference_queue(ecf)

    # run simulations
    xt.info("simulation mode", cf.mode)
    if cf.mode == "vs":
        res_sfb = run_sfb(env, ref, K)
    res_fel = run_fel(env, ref, fel, K)

    # result plotting
    rfc = cf.result.fig
    nrows = np.sum([int(f) for _, f in rfc._cf.items()])
    fig, axes = plt.subplots(nrows=nrows, sharex=True)
    axes_pointer = 0
    if not hasattr(axes, "__len__"):
        axes = [axes]

    res_params = lambda: xt.Config(dict(key=[], style=[], color=[]))

    if rfc.pitch:
        res_pitch_params = res_params()
        res_pitch = dict(
            time=res_fel.time(),
        )
        if cf.mode == "fel":
            res_pitch["reference"] = res_fel.r(env.IX_T, fn=xt.r2d)
            res_pitch["pitch"] = res_fel.x(env.IX_T, fn=xt.r2d)
            res_pitch_params.key = ["reference", "pitch"]
            res_pitch_params.style = ["--", "-"]
            res_pitch_params.color = [COLORS.ref, COLORS.fel]
        if cf.mode == "vs":
            res_pitch["ref"] = res_fel.r(env.IX_T, fn=xt.r2d)
            res_pitch["fel"] = res_fel.x(env.IX_T, fn=xt.r2d)
            res_pitch["sfb"] = res_sfb.x(env.IX_T, fn=xt.r2d)
            res_pitch_params.key = ["ref", "sfb", "fel"]
            res_pitch_params.style = ["--", "-", "-"]
            res_pitch_params.color = [COLORS.ref, COLORS.sfb, COLORS.fel]

        res_pitch = pd.DataFrame(res_pitch)
        res_pitch.plot(
            x="time", y=res_pitch_params.key, ax=axes[axes_pointer],
            style=res_pitch_params.style, color=res_pitch_params.color,
            xlim=[0, ecf.due], ylim=ecf.reference.range, grid=True,
            yticks=np.arange(ecf.reference.range[0], ecf.reference.range[1]+1, 1)
        )
        axes_pointer += 1

    if rfc.elevator:
        res_elevator_params = res_params()
        res_elevator =pd.DataFrame(dict(
            time=res_fel.time(),
        ))
        if cf.mode == "fel":
            res_elevator["elevator"] = res_fel.u(env.IX_de, fn=xt.r2d)
            res_elevator_params.key = ["elevator"]
            res_elevator_params.style = ["-"]
            res_elevator_params.color = [COLORS.fel]
        if cf.mode == "vs":
            res_elevator["sfb"] = res_sfb.u(env.IX_de, fn=xt.r2d)
            res_elevator["fel"] = res_fel.u(env.IX_de, fn=xt.r2d)
            res_elevator_params.key = ["sfb", "fel"]
            res_elevator_params.style = ["-", "-"]
            res_elevator_params.color = [COLORS.sfb, COLORS.fel]
        elevator_range = xt.r2d([
            env.action_space.low[env.IX_de],
            env.action_space.high[env.IX_de]
        ])
        res_elevator.plot(
            x="time", y=res_elevator_params.key, ax=axes[axes_pointer],
            style=res_elevator_params.style, color=res_elevator_params.color,
            xlim=[0, ecf.due], ylim=elevator_range, grid=True,
        )
        axes_pointer += 1

    if rfc.loss:
        res_loss =pd.DataFrame(dict(
            time=res_fel.time(),
            loss=res_fel.loss()
        ))
        res_loss.plot(
            x="time", y="loss", ax=axes[axes_pointer],
            style=["-"], color=[COLORS.fel],
            xlim=[0, ecf.due], grid=True
        )
        axes_pointer += 1

    plt.show()


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# simulation loop
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# state feedback
#
def run_sfb(env, ref, K):
    env.reset()
    log = xsim.Logger()
    for time, r in tqdm(ref.items()):
        u = compute_sfb_action(env, K, r[:2])
        u = build_action(env, u, 0)
        log.store(time=time, x=env.state, u=u, r=r).flush()

        env.step(u)

    return xsim.Retriever(log)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# state feedback
#
def run_fel(env, ref, fel, K):
    env.reset()
    log = xsim.Logger()
    for time, r in tqdm(ref.items()):
        act_sfb = compute_sfb_action(env, K, r[:2])
        act_fel = compute_fel_action(env, fel, r)
        act = build_action(env, act_sfb, act_fel)
        loss = fel.update(r, [act[env.IX_de]])
        log.store(
            time=time,
            x=env.state,
            act_sfb=act_sfb,
            act_fel=act_fel,
            u=act,
            loss=loss,
            r=r
        ).flush()

        env.step(act)

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
#
def build_fel(cf, env):
    dim_act = 1
    dim_ref = env.reference.shape[0]
    max_act = env.action_space.high[env.IX_de]
    fel_type = "simple" if not hasattr(cf, "type") else cf.type

    return fel.FEL(dim_ref, dim_act, max_act, lr=cf.lr)


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


if __name__ == '__main__':
    import sys
    args = sys.argv

    # pour dev
    args.append(DEFAULT_CONFIG)
    args.append("fel")

    # config file setting
    config_file = DEFAULT_CONFIG if len(args) < 2 else args[1]

    run(config_file)

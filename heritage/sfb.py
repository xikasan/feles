# coding: utf-8

import xsim
import xair
import gym
import numpy as np
import xtools as xt
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from feles.controller import sfb
from feles.utils.builder import *
from feles.controller import compute_action as ca


DEFAULT_CONFIG = "configs/sfb_no_fail.yaml"


def run(config, plot=True):
    cf = xt.Config(config)

    # plant
    env = build_env(cf.env)
    xt.info("env", env)

    # controller
    scf = cf.controller.sfb
    fbc = build_sfb(scf, env)

    ref = build_reference_queue(cf.env, at_random=False)

    # logger
    log = xsim.Logger()

    xt.info("run simulation")
    for time, r in tqdm(ref.items()):
        x = env.state
        u = ca.fbc(fbc, r, x)
        act = ca.build_action(env, u)


        log.store(
            t=time,
            x=x,
            u=u,
            r=r
        ).flush()

        env.step(act)

    xt.info("make result")
    res = xsim.Retriever(log)
    res = pd.DataFrame(dict(
        time=res.t(),
        pitch=res.x(env.IX_T, fn=xt.r2d),
        reference=res.r(env.IX_T, fn=xt.r2d),
        elevator=res.u(fn=xt.r2d)
    ))

    if not plot:
        return res

    xt.info("plot result")
    delim = xt.round(xt.r2d([
        env.action_space.low[env.IX_de],
        env.action_space.high[env.IX_de]
    ]), 3)

    ecf = cf.env
    fig, axes = plt.subplots(nrows=2, sharex=True)
    # draw lines
    axes[0].axhline(0, color="black", linewidth="1")
    axes[1].axhline(0, color="black", linewidth="1")
    if hasattr(cf.env, "fail"):
        axes[0].axvline(cf.env.fail.time, color="red")
        axes[1].axvline(cf.env.fail.time, color="red")

    res.plot(
        x="time", y=["reference", "pitch"], ax=axes[0],
        style=["--", "-"], color=["gray", "forestgreen"],
        xlim=[0, ecf.due],
        # ylim=ecf.reference.range,
        grid=True,
        # yticks=np.arange(ecf.reference.range[0], ecf.reference.range[1]+1, 1)
    )
    res.plot(
        x="time", y="elevator", ax=axes[1],
        style="-", color="orange",
        ylim=delim, grid=True,
        yticks=np.arange(delim[0], delim[1]+1, 1)
    )

    plt.show()
    exit()


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = prepare_args()

    run(args.config)

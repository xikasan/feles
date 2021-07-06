# coding: utf-8

import xtools as xt
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from feles.utils import builder as B
from heritage.fel import run_fel


def run(args):
    xt.info("run for fel")
    cf = xt.Config(args.config)

    env = B.build_env(cf.env)
    fbc = B.build_sfb(cf.controller.sfb, env)
    fel = B.build_dfel(cf.controller.fel, env)
    ref = B.build_reference_queue(cf.env)

    IXT = env.IX_T
    fn = xt.r2d
    res = run_fel(cf, env, ref, fbc, fel)
    res = pd.DataFrame(dict(
        time=res.t(),
        reference=res.r(IXT, fn=fn),
        pitch=res.x(IXT, fn=fn),
        elevator=res.u(0, fn=fn),
        loss=res.l(0)
    ))
    fig, axes = plt.subplots(nrows=3, sharex=True)
    res.plot(x="time", y=["reference", "pitch"], ax=axes[0], grid=True)
    res.plot(x="time", y="elevator", ax=axes[1])
    res.plot(x="time", y="loss", ax=axes[2])

    plt.show()


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dfel_normal.yaml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument()
    run(args)
